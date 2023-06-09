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
from common.action_genome_dataset import AGDataset, dataset_collate_fn
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
    weights_path = Path(opt.weights)
    ag_dataset_path = Path(opt.data)
    backbone_model_path = weights_path / "backbone" / "resnet101-ag.pth"
    sttran_word_vector_dir = weights_path / "semantic"
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
    dim_gaze_heatmap = cfg["dim_gaze_heatmap"]
    dim_transformer_ffn = cfg["dim_transformer_ffn"]
    sttran_enc_layer_num = cfg["sttran_enc_layer_num"]
    sttran_dec_layer_num = cfg["sttran_dec_layer_num"]
    sttran_sliding_window = cfg["sttran_sliding_window"]
    if sampling_mode == "window" or sampling_mode == "anticipation":
        max_clip_length = sttran_sliding_window  # max length for dataset
    if sampling_mode == "anticipation":
        future_num = cfg["future_num"]
        future_type = cfg["future_type"]
        future_ratio = cfg["future_ratio"]
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
    ag_val_dataset = AGDataset(
        dataset_dir=ag_dataset_path,
        mode="test",
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
    object_classes = ag_val_dataset.object_classes
    interaction_classes = ag_val_dataset.relationship_classes
    attention_class_idxes = ag_val_dataset.attention_idxes
    spatial_class_idxes = ag_val_dataset.spatial_idxes
    contacting_class_idxes = ag_val_dataset.contacting_idxes
    num_object_classes = len(object_classes)
    num_interaction_classes = len(interaction_classes)
    num_attention_classes = len(attention_class_idxes)
    num_spatial_classes = len(spatial_class_idxes)
    num_contacting_classes = len(contacting_class_idxes)
    logger.info(f"{num_object_classes} Possible Objects: {object_classes}")
    logger.info(f"{num_interaction_classes} Possible Interactions")
    logger.info(
        f"{num_attention_classes} attention classes: {[interaction_classes[idx] for idx in attention_class_idxes]}"
    )
    logger.info(f"{num_spatial_classes} spatial relations: {[interaction_classes[idx] for idx in spatial_class_idxes]}")
    logger.info(
        f"{num_contacting_classes} contacting relations: {[interaction_classes[idx] for idx in contacting_class_idxes]}"
    )
    triplet_info, weight_info = ag_val_dataset.analyse_split_weight(save_dir=log_eval_path, separate_head=separate_head)
    logger.info(f"{len(triplet_info['triplet_val_hist'])} triplet classes in evaluation dataset")
    logger.info(
        f"{weight_info['num_val_pairs']} human-object pairs in evaluation annotation, "
        f"and {np.sum(weight_info['interaction_val_hist'])} HOI triplets"
    )
    ag_val_dataset.eval()
    ag_val_dataloader = DataLoader(
        ag_val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=dataset_collate_fn
    )
    # load detections and corresponding gaze
    if detection_mode:
        logger.info("Loading prepared validation detection results")
        ag_detection_filename = ag_dataset_path / f"action_genome_detection/val_trace_{detection_name}.json"
        with ag_detection_filename.open() as val_detection_file:
            val_detection_dict = json.loads(val_detection_file.read())
        val_gaze_filename = ag_dataset_path / f"action_genome_detection/val_frame_gazes_{detection_name}"
        val_gaze_dict = shelve.open(str(val_gaze_filename))
    else:
        # load gaze based on ground-truth bbox
        logger.info("Loading prepared validation ground-truth gaze features")
        val_gaze_filename = ag_dataset_path / "action_genome_gaze/val_frame_gazes_gt_bbox"
        val_gaze_dict = shelve.open(str(val_gaze_filename))
    logger.info("-" * 15 + " Validation dataset loading finished! " + "-" * 15)

    ## Models
    # load pretrained faseterrcnn backbone on AG
    feature_backbone = FeatureExtractionResNet101(backbone_model_path, download=False, finetune=True).to(device)
    # freeze backbone
    feature_backbone.requires_grad_(False)
    feature_backbone.eval()
    logger.info(f"ResNet101 feature backbone loaded from {backbone_model_path}.")
    # separate to three heads
    num_interaction_classes_loss = num_interaction_classes
    # NOTE action genome annotation already contains no_interaction
    separate_head_name = ["attention_head", "spatial_head", "contacting_head"]
    separate_head_num = [num_attention_classes, num_spatial_classes, -1]
    if loss_type == "mlm":
        loss_type_dict = {
            "attention_head": "ce",
            "spatial_head": "mlm",
            "contacting_head": "mlm",
        }
    else:
        loss_type_dict = {
            "attention_head": "ce",
            "spatial_head": "bce",
            "contacting_head": "bce",
        }
    class_idxes_dict = {
        "attention_head": attention_class_idxes,
        "spatial_head": spatial_class_idxes,
        "contacting_head": contacting_class_idxes,
    }
    loss_gt_dict = {
        "attention_head": "attention_gt",
        "spatial_head": "spatial_gt",
        "contacting_head": "contacting_gt",
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
            f"Spatial-temporal Transformer loaded. d_model={sttran_gaze_model.d_model}, gaze cross first layer={sttran_gaze_model.d_gaze}, separate_head={sttran_gaze_model.separate_head}"
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
            mlp_projection=mlp_projection,
            sinusoidal_encoding=sinusoidal_encoding,
            dropout=0,
            sliding_window=sttran_sliding_window,
            separate_head=separate_head_num,
            separate_head_name=separate_head_name,
        )
        logger.info(
            f"Spatial-temporal Transformer loaded. d_model={sttran_gaze_model.d_model}, gaze cross all layers={sttran_gaze_model.d_gaze}, separate_head={sttran_gaze_model.separate_head}"
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
            f"Spatial-temporal Transformer loaded. d_model={sttran_gaze_model.d_model}, gaze concat={sttran_gaze_model.no_gaze}, separate_head={sttran_gaze_model.separate_head}"
        )
    sttran_gaze_model = sttran_gaze_model.to(device)
    incompatibles = sttran_gaze_model.load_state_dict(torch.load(opt.model))
    logger.info(f"Incompatible keys {incompatibles}")

    ## Evaluation
    # set to evaluation mode
    sttran_gaze_model.eval()
    ag_val_dataset.eval()
    # inference script
    with torch.no_grad():
        if detection_mode:
            all_results = inference_one_epoch(
                ag_val_dataloader,
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
                mlm_add_no_interaction=False,
                human_label=1,
            )
        else:
            all_results = inference_one_epoch(
                ag_val_dataloader,
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
                mlm_add_no_interaction=False,
                human_label=1,
            )
    with result_path.open("w") as out:
        json.dump(all_results, out)

    logger.info(f"Finish! Results dumped to {result_path}")

    return all_results


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="path to STTran model")
    parser.add_argument("--cfg", type=str, default="configs/eval_hyp.yaml", help="path to hyperparameter configs")
    parser.add_argument("--weights", type=str, default="weights", help="root folder for all pretrained weights")
    parser.add_argument("--data", type=str, default="G:/datasets/action_genome/", help="dataset root path")
    parser.add_argument(
        "--detection",
        type=str,
        default="",
        help="empty: Oracle mode. Otherwise method name. Will load the detection and gaze from dataset_root/VidHOI_detection/val_trace_(method_name).json",
    )
    parser.add_argument("--subset-val", type=int, default=-1, help="sub val dataset length")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--project", default="../runs/sttran_gaze_ag", help="save to project/name")
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
