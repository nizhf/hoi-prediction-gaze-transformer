#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

sys.path.insert(0, "modules/object_tracking/yolov5")

import argparse
import logging
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import shelve
import json

from common.config_parser import get_config
from common.vidhoi_dataset import VidHOIDataset, dataset_collate_fn
from common.transforms import STTranTransform
from common.inference_utils import inference_one_epoch
from modules.sthoip_transformer.sttran_gaze import STTranGaze
from modules.sthoip_transformer.sttran_gaze import STTranGazeCrossAttention
from modules.object_tracking import FeatureExtractionResNet101


def eval_inference(opt, device):
    """
    Run inferences for all key frames in the validation set, output inference results to a JSON file
    """
    ## Init hyperparameters
    # evaluation mode
    if opt.detection == "":
        detection_mode = False
    else:
        detection_mode = True
        detection_name = opt.detection
    # path
    vidhoi_dataset_path = Path(opt.data)
    backbone_model_path = Path(opt.backbone)
    sttran_word_vector_dir = Path(opt.semantic)
    output_path = Path(opt.project)
    run_id = opt.name
    if detection_mode:
        log_eval_path = output_path / run_id / f"eval_det_{detection_name}"
    else:
        log_eval_path = output_path / run_id / "eval"
    log_eval_path.mkdir(parents=True, exist_ok=True)
    log_file_path = log_eval_path / "eval.log"
    result_path = log_eval_path / "all_results.json"
    # load hyperparameters from cfg file
    cfg = get_config(opt.cfg)
    # net and train hyperparameters
    img_size = opt.imgsz  # for feature backbone perprocessing
    sampling_mode = cfg["sampling_mode"]
    min_clip_length = cfg["min_clip_length"]
    max_clip_length = cfg["max_clip_length"]
    max_human_num = cfg["max_human_num"]
    batch_size = cfg["batch_size"]
    subset_val_len = opt.subset_val  # use a small subset to test the demo
    subset_val_shuffle = True if subset_val_len > 0 else False  # shuffle subset
    dim_transformer_ffn = cfg["dim_transformer_ffn"]
    sttran_enc_layer_num = cfg["sttran_enc_layer_num"]
    sttran_dec_layer_num = cfg["sttran_dec_layer_num"]
    sttran_sliding_window = cfg["sttran_sliding_window"]
    if sampling_mode == "window" or sampling_mode == "anticipation":
        max_clip_length = sttran_sliding_window  # max length for dataset
    # inference not need future type and ratio
    if sampling_mode == "anticipation":
        future_num = cfg["future_num"]
    else:
        future_num = 0
    future_type = "all"
    future_ratio = 0
    gaze_usage = opt.gaze  # how to use gaze
    global_token = opt.global_token  # use global token
    mlp_projection = cfg["mlp_projection"]  # MLP in input embedding
    sinusoidal_encoding = cfg["sinusoidal_encoding"]  # sinusoidal positional encoding
    separate_head = cfg["separate_head"]  # separate spatial relation prediction head
    loss_type = cfg["loss_type"]
    # set RNG seed for reproducibility
    random_seed = cfg["random_seed"]
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    ## Init logging
    # remove root logger (initialized by YOLOv5)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    # eval logger output to file and console
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(str(log_file_path), mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(name)s: %(message)s")
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    logger.info("Initializing dataset and model...")
    logger.info({**vars(opt), **cfg})

    ## Dataset
    # load validation dataset, save triplet histogram
    vidhoi_val_dataset = VidHOIDataset(
        annotations_file=vidhoi_dataset_path / "VidHOI_annotation/val_frame_annots.json",
        frames_dir=vidhoi_dataset_path / "images",
        min_length=min_clip_length,
        max_length=max_clip_length,
        max_human_num=max_human_num,
        train_ratio=0,
        subset_len=subset_val_len,
        subset_shuffle=subset_val_shuffle,
        transform=STTranTransform(img_size=img_size),
        annotation_mode=sampling_mode,
        logger=logger,
        future_num=future_num,
        future_type=future_type,
        future_ratio=future_ratio,
    )
    # class names
    object_classes = vidhoi_val_dataset.object_classes
    interaction_classes = vidhoi_val_dataset.interaction_classes
    spatial_class_idxes = vidhoi_val_dataset.spatial_class_idxes
    action_class_idxes = vidhoi_val_dataset.action_class_idxes
    num_object_classes = len(object_classes)
    num_interaction_classes = len(interaction_classes)
    num_spatial_classes = len(spatial_class_idxes)
    num_action_classes = len(action_class_idxes)
    logger.info(f"{num_object_classes} Possible Objects: {object_classes}")
    logger.info(f"{num_interaction_classes} Possible Interactions")
    logger.info(f"{num_spatial_classes} spatial relations: {[interaction_classes[idx] for idx in spatial_class_idxes]}")
    logger.info(f"{num_action_classes} actions: {[interaction_classes[idx] for idx in action_class_idxes]}")
    triplet_info, weight_info = vidhoi_val_dataset.analyse_split_weight(
        save_dir=log_eval_path, separate_head=separate_head
    )
    logger.info(f"{len(triplet_info['triplet_val_hist'])} triplet classes in evaluation dataset")
    logger.info(
        f"{weight_info['num_val_pairs']} human-object pairs in evaluation annotation, "
        f"and {np.sum(weight_info['interaction_val_hist'])} HOI triplets"
    )
    vidhoi_val_dataset.eval()
    vidhoi_val_dataloader = DataLoader(
        vidhoi_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset_collate_fn
    )
    # load detections and corresponding gaze
    if detection_mode:
        logger.info("Loading prepared validation detection results")
        val_detection_filename = vidhoi_dataset_path / f"VidHOI_detection/val_trace_{detection_name}.json"
        with val_detection_filename.open() as val_detection_file:
            val_detection_dict = json.loads(val_detection_file.read())
        val_gaze_filename = vidhoi_dataset_path / f"VidHOI_detection/val_frame_gazes_{detection_name}"
        val_gaze_dict = shelve.open(str(val_gaze_filename))
    else:
        # load gaze based on ground-truth bbox
        logger.info("Loading prepared validation ground-truth gaze features")
        val_gaze_filename = vidhoi_dataset_path / "VidHOI_gaze/val_frame_gazes_gt_bbox"
        val_gaze_dict = shelve.open(str(val_gaze_filename))
    logger.info("-" * 15 + " Validation dataset loading finished! " + "-" * 15)

    ## Models
    # backbone path is a directory, using a downloaded weight
    if backbone_model_path.stem == backbone_model_path.name:
        feature_backbone = FeatureExtractionResNet101(backbone_model_path, download=True, finetune=False).to(device)
    # backbone path is a file, directly load the full model
    else:
        feature_backbone = FeatureExtractionResNet101(backbone_model_path, download=False, finetune=False).to(device)
    # freeze backbone
    feature_backbone.requires_grad_(False)
    feature_backbone.eval()
    logger.info(f"ResNet101 feature backbone loaded from {backbone_model_path}.")
    # separate to spatial relation head and action head
    num_interaction_classes_loss = num_interaction_classes
    if separate_head:
        separate_head_num = num_spatial_classes
        # NOTE mlm loss needs no_interaction since there could be no positives in one head
        if loss_type == "mlm":
            num_interaction_classes_loss += 2
            separate_head_num = [num_spatial_classes + 1, -1]
            loss_type_dict = {"spatial_head": "mlm", "action_head": "mlm"}
        else:
            loss_type_dict = {"spatial_head": "bce", "action_head": "bce"}
            separate_head_num = [num_spatial_classes, -1]
        separate_head_name = ["spatial_head", "action_head"]
        class_idxes_dict = {"spatial_head": spatial_class_idxes, "action_head": action_class_idxes}
        loss_gt_dict = {"spatial_head": "spatial_gt", "action_head": "action_gt"}
    else:
        # NOTE mlm loss needs no_interaction
        if loss_type == "mlm":
            num_interaction_classes_loss += 1
            loss_type_dict = {"interaction_head": "mlm"}
        else:
            loss_type_dict = {"interaction_head": "bce"}
        separate_head_name = ["interaction_head"]
        separate_head_num = [-1]
        class_idxes_dict = {
            "interaction_head": [i for i in range(num_interaction_classes)],
        }
        loss_gt_dict = {
            "interaction_head": "interactions_gt",
        }
    # transformer model
    if gaze_usage == "cross":
        sttran_gaze_model = STTranGazeCrossAttention(
            num_interaction_classes=num_interaction_classes_loss,
            obj_class_names=object_classes,
            spatial_layer_num=sttran_enc_layer_num,
            cross_layer_num=1,
            temporal_layer_num=sttran_dec_layer_num - 1,
            dim_transformer_ffn=dim_transformer_ffn,
            d_gaze=512,
            cross_sa=True,
            cross_ffn=False,
            global_token=global_token,
            mlp_projection=mlp_projection,
            sinusoidal_encoding=sinusoidal_encoding,
            dropout=0,
            word_vector_dir=sttran_word_vector_dir,
            sliding_window=sttran_sliding_window,
            separate_head=separate_head_num,
            separate_head_name=separate_head_name,
        )
        logger.info(
            f"Spatial-temporal Transformer loaded. d_model={sttran_gaze_model.d_model}, "
            f"gaze cross first layer={sttran_gaze_model.d_gaze}, separate_head={sttran_gaze_model.separate_head}"
        )
    elif gaze_usage == "cross_all":
        sttran_gaze_model = STTranGazeCrossAttention(
            num_interaction_classes=num_interaction_classes_loss,
            obj_class_names=object_classes,
            spatial_layer_num=sttran_enc_layer_num,
            cross_layer_num=sttran_dec_layer_num,
            temporal_layer_num=0,
            dim_transformer_ffn=dim_transformer_ffn,
            d_gaze=512,
            cross_sa=True,
            cross_ffn=False,
            global_token=global_token,
            mlp_projection=mlp_projection,
            sinusoidal_encoding=sinusoidal_encoding,
            dropout=0,
            word_vector_dir=sttran_word_vector_dir,
            sliding_window=sttran_sliding_window,
            separate_head=separate_head_num,
            separate_head_name=separate_head_name,
        )
        logger.info(
            f"Spatial-temporal Transformer loaded. d_model={sttran_gaze_model.d_model}, "
            f"gaze cross all layers={sttran_gaze_model.d_gaze}, separate_head={sttran_gaze_model.separate_head}"
        )
    else:
        sttran_gaze_model = STTranGaze(
            num_interaction_classes=num_interaction_classes_loss,
            obj_class_names=object_classes,
            enc_layer_num=sttran_enc_layer_num,
            dec_layer_num=sttran_dec_layer_num,
            dim_transformer_ffn=dim_transformer_ffn,
            sinusoidal_encoding=sinusoidal_encoding,
            word_vector_dir=sttran_word_vector_dir,
            sliding_window=sttran_sliding_window,
            no_gaze=gaze_usage == "no",
            separate_head=separate_head_num,
            separate_head_name=separate_head_name,
        )
        logger.info(
            f"Spatial-temporal Transformer loaded. d_model={sttran_gaze_model.d_model}, "
            f"gaze concat={sttran_gaze_model.no_gaze}, separate_head={sttran_gaze_model.separate_head}"
        )
    sttran_gaze_model = sttran_gaze_model.to(device)
    incompatibles = sttran_gaze_model.load_state_dict(torch.load(opt.model))
    logger.info(f"Incompatible keys {incompatibles}")

    ## Evaluation
    # set to evaluation mode
    sttran_gaze_model.eval()
    vidhoi_val_dataset.eval()
    # inference script
    with torch.no_grad():
        if detection_mode:
            all_results = inference_one_epoch(
                vidhoi_val_dataloader,
                val_gaze_dict,
                val_detection_dict,
                sttran_gaze_model,
                feature_backbone,
                loss_type_dict,
                class_idxes_dict,
                loss_gt_dict,
                cfg,
                device,
                logger,
            )
        else:
            all_results = inference_one_epoch(
                vidhoi_val_dataloader,
                val_gaze_dict,
                None,
                sttran_gaze_model,
                feature_backbone,
                loss_type_dict,
                class_idxes_dict,
                loss_gt_dict,
                cfg,
                device,
                logger,
            )
    with result_path.open("w") as out:
        json.dump(all_results, out)

    logger.info(f"Finish! Results dumped to {result_path}")

    return all_results


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="path to STTran model")
    parser.add_argument("--cfg", type=str, default="configs/eval_hyp.yaml", help="path to hyperparameter configs")
    parser.add_argument("--backbone", type=str, default="weights/backbone", help="root folder for backbone weights")
    parser.add_argument(
        "--semantic", type=str, default="weights/semantic", help="root folder for word embedding weights"
    )
    parser.add_argument("--data", type=str, default="G:/datasets/VidOR/", help="dataset root path")
    parser.add_argument(
        "--detection",
        type=str,
        default="",
        help="empty: Oracle mode. Otherwise method name. Will load the detection and gaze from dataset_root/VidHOI_detection/val_trace_(method_name).json",
    )
    parser.add_argument("--subset-val", type=int, default=-1, help="sub val dataset length")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--project", default="../runs/sttran_gaze_vidhoi", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--gaze", type=str, default="concat", help="how to use gaze features: no, concat, cross")
    parser.add_argument("--global-token", action="store_true", help="Use global token, only for cross-attention mode")

    opt = parser.parse_args()
    return opt


def main(opt):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    eval_inference(opt, device)


if __name__ == "__main__":
    # signal.signal(signal.SIGINT, on_terminate_handler)
    opt = parse_opt()
    main(opt)
