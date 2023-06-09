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
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter, GaussianBlur, Compose
from pathlib import Path
import shelve
import signal

from common.config_parser import get_config
from common.vidhoi_dataset import VidHOIDataset, dataset_collate_fn
from common.transforms import STTranTransform, ClipRandomHorizontalFlipping
from common.train_utils import train_one_epoch_gt_bbox
from common.eval_utils import evaluate_gt_bbox
from common.logging_utils import WandbLogger, PandasMetricLogger, PandasClassAPLogger, ClipLossesLogger
from common.lr_scheduler.transformer_lr_scheduler import TransformerLRScheduler
from common.early_stopping import EarlyStopping
from common.focal_loss import FocalBCEWithLogitLoss
from modules.sthoip_transformer.sttran_gaze import STTranGaze
from modules.sthoip_transformer.sttran_gaze import STTranGazeCrossAttention
from modules.object_tracking import FeatureExtractionResNet101


global wandb_logger
global wandb_finished


def on_terminate_handler(signal, frame):
    # finish wandb before
    print("Ctrl+C received, stop training...")
    global wandb_finished
    if not wandb_finished:
        global wandb_logger
        wandb_logger.finish_run()
        wandb_finished = True
    sys.exit(0)


def train_gt_bbox(opt, device):
    global wandb_finished
    wandb_finished = False
    ## Init hyperparameters
    # path
    weights_path = Path(opt.weights)
    vidhoi_dataset_path = Path(opt.data)
    backbone_model_path = weights_path / "backbone"
    sttran_word_vector_dir = weights_path / "semantic"
    output_path = Path(opt.project)
    run_id = opt.name
    log_weight_suffix = "weights"
    log_weight_path = output_path / run_id / log_weight_suffix
    log_file_path = output_path / run_id / "train.log"
    log_csv_path = output_path / run_id / "metrics.csv"
    log_triplet_ap_path = output_path / run_id / "class_ap.csv"
    log_clip_losses_path = output_path / run_id / "train_losses.json"
    log_weight_path.mkdir(parents=True, exist_ok=True)
    save_period = opt.save_period
    # load hyperparameters from opt, and cfg file
    cfg = get_config(opt.cfg)
    # net and train hyperparameters
    img_size = opt.imgsz  # for yolov5 and feature backbone perprocessing
    # yolov5_stride = 32  # should come from yolov5 model
    sampling_mode = cfg["sampling_mode"]
    min_clip_length = cfg["min_clip_length"]
    max_clip_length = cfg["max_clip_length"]
    max_human_num = cfg["max_human_num"]
    train_ratio = cfg["train_ratio"]
    batch_size = cfg["batch_size"]
    batch_size_val = cfg["batch_size_val"]
    subset_train_len = opt.subset_train  # use a small subset to test the demo
    subset_train_shuffle = True if subset_train_len > 0 else False  # shuffle subset
    # if exceed, skip due to GPU Memory
    # max_interaction_pairs = cfg["max_interaction_pairs"]
    # max_interaction_pairs_per_frame = cfg["max_interaction_pairs_per_frame"]
    # dim_gaze_heatmap = cfg["dim_gaze_heatmap"]
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
    hflip_p = cfg["hflip_p"]
    color_jitter = cfg["color_jitter"]
    gaze_usage = opt.gaze  # how to use gaze
    global_token = opt.global_token  # use global token
    mlp_projection = cfg["mlp_projection"]
    sinusoidal_encoding = cfg["sinusoidal_encoding"]
    separate_head = cfg["separate_head"]  # separate spatial relation prediction head
    finetune_backbone = opt.finetune_backbone  # whether finetune backbone
    max_epochs = opt.epochs
    warmup_steps = opt.warmup
    init_lr = cfg["init_lr"]
    peak_lr = cfg["peak_lr"]
    final_lr = cfg["final_lr"]
    final_lr_scale = cfg["final_lr_scale"]
    dropout = cfg["dropout"]
    decay_steps = cfg["decay_steps"]
    weight_decay = cfg["weight_decay"]
    # clip_norm_max = cfg["clip_norm_max"]
    # clip_norm_type = cfg["clip_norm_type"]
    loss_type = cfg["loss_type"]
    loss_balance_type = cfg["loss_balance_type"]
    loss_balance_power = cfg["loss_balance_power"]
    loss_balance_beta = cfg["loss_balance_beta"]
    loss_focal_gamma = cfg["loss_focal_gamma"]
    early_patience = cfg["early_patience"]
    early_min_epoch = cfg["early_min_epoch"]
    early_min_improvement = cfg["early_min_improvement"]
    early_metric = cfg["early_metric"]
    interaction_conf_threshold = cfg["interaction_conf_threshold"]
    eval_k = cfg["eval_k"]
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
    # train logger output to file and console
    logger = logging.getLogger("train")
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
    # wandb
    global wandb_logger
    wandb_logger = WandbLogger(
        config={**vars(opt), **cfg},
        project=output_path.name,
        wandb_dir=output_path / run_id,
        disabled=opt.disable_wandb,
    )
    if wandb_logger.disabled:
        logger.info("Wandb disabled.")
    # metrics logger
    pandas_metric_header = [
        "train/loss",
        "train/loss_spatial",
        "train/loss_action",
        "train/lr",
        "val/loss",
        "val/loss_spatial",
        "val/loss_action",
        "metrics/mAP",
    ]
    for thres in interaction_conf_threshold:
        pandas_metric_header.append(f"metrics/precision_{thres}")
        pandas_metric_header.append(f"metrics/recall_{thres}")
        pandas_metric_header.append(f"metrics/accuracy_{thres}")
        pandas_metric_header.append(f"metrics/f1_{thres}")
        pandas_metric_header.append(f"metrics/hamming_loss_{thres}")
    for thres in interaction_conf_threshold:
        for k in eval_k:
            pandas_metric_header.append(f"metrics/recall@k/recall@{k}_{thres}")
    pandas_metric_logger = PandasMetricLogger(log_csv_path, pandas_metric_header)
    # clip losses logger
    json_clip_losses_logger = ClipLossesLogger(log_clip_losses_path)
    # class AP logger
    pandas_ap_logger = PandasClassAPLogger(log_triplet_ap_path)
    logger.info(f"Output metrics to {log_csv_path} and {log_triplet_ap_path}")

    ## Dataset
    if color_jitter:
        post_transform = ClipRandomHorizontalFlipping(
            hflip_p,
            Compose([ColorJitter(0.3, 0.3, 0.3), GaussianBlur((5, 9), (0.1, 5))]),
        )
    else:
        post_transform = ClipRandomHorizontalFlipping(hflip_p)
    # load training dataset
    vidhoi_train_dataset = VidHOIDataset(
        annotations_file=vidhoi_dataset_path / "VidHOI_annotation/train_frame_annots.json",
        frames_dir=vidhoi_dataset_path / "images",
        min_length=min_clip_length,
        max_length=max_clip_length,
        max_human_num=max_human_num,
        train_ratio=train_ratio,
        subset_len=subset_train_len,
        subset_shuffle=subset_train_shuffle,
        transform=STTranTransform(img_size=img_size),
        post_transform=post_transform,
        annotation_mode=sampling_mode,
        logger=logger,
        future_num=future_num,
        future_type=future_type,
        future_ratio=future_ratio,
    )
    # class names
    object_classes = vidhoi_train_dataset.object_classes
    interaction_classes = vidhoi_train_dataset.interaction_classes
    spatial_class_idxes = vidhoi_train_dataset.spatial_class_idxes
    action_class_idxes = vidhoi_train_dataset.action_class_idxes
    num_object_classes = len(object_classes)
    num_interaction_classes = len(interaction_classes)
    num_spatial_classes = len(spatial_class_idxes)
    num_action_classes = len(action_class_idxes)
    logger.info(f"{num_object_classes} Possible Objects: {object_classes}")
    logger.info(f"{num_interaction_classes} Possible Interactions")
    logger.info(f"{num_spatial_classes} spatial relations: {[interaction_classes[idx] for idx in spatial_class_idxes]}")
    logger.info(f"{num_action_classes} actions: {[interaction_classes[idx] for idx in action_class_idxes]}")
    # analyse training dataset, get histogram of triplets and interactions
    if loss_balance_type == "inverse":
        triplet_info, weight_info = vidhoi_train_dataset.analyse_split_weight(
            save_dir=output_path / run_id,
            method=loss_balance_type,
            separate_head=separate_head,
            power=loss_balance_power,
        )
    elif loss_balance_type == "effective":
        triplet_info, weight_info = vidhoi_train_dataset.analyse_split_weight(
            save_dir=output_path / run_id,
            method=loss_balance_type,
            separate_head=separate_head,
            beta=loss_balance_beta,
        )
    else:
        triplet_info, weight_info = vidhoi_train_dataset.analyse_split_weight(
            save_dir=output_path / run_id,
            method=loss_balance_type,
            separate_head=separate_head,
        )
    logger.info(
        f"{len(triplet_info['triplet_train_hist'])} triplet classes in training subset, "
        f"{len(triplet_info['triplet_val_unique'])}/{len(triplet_info['triplet_val_hist'])} unique triplet classes in validation subset"
    ),
    logger.info(
        f"{weight_info['num_train_pairs']} human-object pairs in training subset annotation, "
        f"and {np.sum(weight_info['interaction_train_hist'])} HOI triplets"
    )
    logger.info(
        f"{weight_info['num_val_pairs']} human-object pairs in validation subset annotation, "
        f"and {np.sum(weight_info['interaction_val_hist'])} HOI triplets"
    )
    # training dataloader
    vidhoi_train_dataset.train()
    vidhoi_train_dataloader = DataLoader(
        vidhoi_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=dataset_collate_fn
    )
    if train_ratio < 1.0:
        # validation dataloader
        vidhoi_train_dataset.eval()
        vidhoi_val_dataloader = DataLoader(
            vidhoi_train_dataset, batch_size=batch_size_val, shuffle=False, num_workers=4, collate_fn=dataset_collate_fn
        )
    # load prepared head and gaze features
    logger.info("Loading prepared training head and gaze features")
    train_gaze_filename = vidhoi_dataset_path / "VidHOI_gaze/train_frame_gazes_gt_bbox"
    train_gaze_dict = shelve.open(str(train_gaze_filename))

    logger.info("-" * 10 + " Dataset loading finished! " + "-" * 10)

    ## Models
    if finetune_backbone:
        feature_backbone = FeatureExtractionResNet101(
            backbone_model_path, download=True, finetune=True, finetune_layers=["backbone_head"]
        ).to(device)
    else:
        feature_backbone = FeatureExtractionResNet101(
            backbone_model_path, download=True, finetune=False, finetune_layers=[]
        ).to(device)
    trainable_backbone_names = []
    trainable_backbone_params = []
    for name, param in feature_backbone.named_parameters():
        if param.requires_grad:
            trainable_backbone_names.append(name)
            trainable_backbone_params.append(param)
    logger.info(
        f"ResNet101 feature backbone loaded from {backbone_model_path}. Finetuning weights: {trainable_backbone_names}"
    )
    # separate to spatial relation head and action head
    num_interaction_classes_loss = num_interaction_classes
    if separate_head:
        # NOTE mlm loss needs no_interaction since there could be no positives in one head
        if loss_type == "mlm":
            num_interaction_classes_loss += 2
            separate_head_num = [num_spatial_classes + 1, -1]
        else:
            separate_head_num = [num_spatial_classes, -1]
        separate_head_name = ["spatial_head", "action_head"]
    else:
        # NOTE mlm loss needs no_interaction
        if loss_type == "mlm":
            num_interaction_classes_loss += 1
        separate_head_name = ["interaction_head"]
        separate_head_num = [-1]
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
            dropout=dropout,
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
            dropout=dropout,
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

    ## Loss function & optimizer
    # Multi-label margin loss
    if loss_type == "mlm":
        # weight not works for mlm # NOTE add sigmoid before loss!
        if separate_head:
            loss_fn_dict = {
                "spatial_head": nn.MultiLabelMarginLoss(reduction="mean"),
                "action_head": nn.MultiLabelMarginLoss(reduction="mean"),
            }
            # no weight for validation
            loss_val_dict = {
                "spatial_head": nn.MultiLabelMarginLoss(reduction="mean"),
                "action_head": nn.MultiLabelMarginLoss(reduction="mean"),
            }
            loss_type_dict = {"spatial_head": "mlm", "action_head": "mlm"}
        else:
            loss_fn_dict = {"interaction_head": nn.MultiLabelMarginLoss(reduction="mean")}
            # no weight for validation
            loss_val_dict = {"interaction_head": nn.MultiLabelMarginLoss(reduction="mean")}
            loss_type_dict = {"interaction_head": "mlm"}
    # Focal loss
    elif loss_type == "focal":
        if separate_head:
            loss_fn_dict = {
                "spatial_head": FocalBCEWithLogitLoss(
                    gamma=loss_focal_gamma,
                    alpha=None,
                    pos_weight=torch.Tensor(weight_info["weight_train_spatial"]).to(device),
                    reduction="mean",
                ),
                "action_head": FocalBCEWithLogitLoss(
                    gamma=loss_focal_gamma,
                    alpha=None,
                    pos_weight=torch.Tensor(weight_info["weight_train_action"]).to(device),
                    reduction="mean",
                ),
            }
            # no weight for validation, use bce
            loss_val_dict = {
                "spatial_head": nn.BCEWithLogitsLoss(reduction="mean"),
                "action_head": nn.BCEWithLogitsLoss(reduction="mean"),
            }
            loss_type_dict = {"spatial_head": "focal", "action_head": "focal"}
        else:
            loss_fn_dict = {
                "interaction_head": FocalBCEWithLogitLoss(
                    gamma=loss_focal_gamma,
                    alpha=None,
                    pos_weight=torch.Tensor(weight_info["weight_train"]).to(device),
                    reduction="mean",
                )
            }
            # no weight for validation, use bce
            loss_val_dict = {"interaction_head": nn.BCEWithLogitsLoss(reduction="mean")}
            loss_type_dict = {"interaction_head": "focal"}
    # Binary cross-entropy loss with logit loss, not using weight here
    else:
        if separate_head:
            loss_fn_dict = {
                "spatial_head": nn.BCEWithLogitsLoss(
                    # pos_weight=torch.Tensor(weight_info["weight_train_spatial"]).to(
                    #     device
                    # ),
                    reduction="mean",
                ),
                "action_head": nn.BCEWithLogitsLoss(
                    # pos_weight=torch.Tensor(weight_info["weight_train_action"]).to(
                    #     device
                    # ),
                    reduction="mean",
                ),
            }
            # no weight for validation, use bce
            loss_val_dict = {
                "spatial_head": nn.BCEWithLogitsLoss(reduction="mean"),
                "action_head": nn.BCEWithLogitsLoss(reduction="mean"),
            }
            loss_type_dict = {"spatial_head": "bce", "action_head": "bce"}
        else:
            loss_fn_dict = {
                "interaction_head": nn.BCEWithLogitsLoss(
                    # pos_weight=torch.Tensor(weight_info["weight_train"]).to(device),
                    reduction="mean",
                )
            }
            # no weight for validation, use bce
            loss_val_dict = {"interaction_head": nn.BCEWithLogitsLoss(reduction="mean")}
            loss_type_dict = {"interaction_head": "bce"}
    if separate_head:
        class_idxes_dict = {"spatial_head": spatial_class_idxes, "action_head": action_class_idxes}
        loss_gt_dict = {"spatial_head": "spatial_gt", "action_head": "action_gt"}
    else:
        class_idxes_dict = {
            "interaction_head": [i for i in range(num_interaction_classes)],
        }
        loss_gt_dict = {
            "interaction_head": "interactions_gt",
        }

    # AdamW optimizer
    if len(trainable_backbone_params) == 0:
        trainable_params = sttran_gaze_model.parameters()
    else:
        trainable_params = trainable_backbone_params + list(sttran_gaze_model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=init_lr, weight_decay=weight_decay)
    # Scheduler with warmup
    scheduler = TransformerLRScheduler(
        optimizer=optimizer,
        init_lr=init_lr,
        peak_lr=peak_lr,
        final_lr=final_lr,
        final_lr_scale=final_lr_scale,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
    )
    # Early stopping based on validation metrics
    early_stopping = EarlyStopping(
        patience=early_patience, minimum_improvement=early_min_improvement, start_epoch=early_min_epoch
    )
    logger.info(
        f"Using {loss_type} loss with AdamW optimizer with init configs {optimizer.defaults}, TransformerLRScheduler and EarlyStopping"
    )
    logger.info("-" * 10 + " Training Start " + "-" * 10)

    ## Training
    # init some epoch logging
    epoch_lr = init_lr
    for epoch in range(max_epochs):
        # prepare metrics for wandb and pandas
        metrics = {}
        # set to train mode
        vidhoi_train_dataset.train()
        sttran_gaze_model.train()
        if finetune_backbone:
            feature_backbone.train()
        else:
            feature_backbone.eval()
        # run train procedure for one epoch
        train_mloss, train_clip_losses, used_time = train_one_epoch_gt_bbox(
            vidhoi_train_dataloader,
            train_gaze_dict,
            sttran_gaze_model,
            loss_fn_dict,
            optimizer,
            feature_backbone,
            loss_type_dict,
            class_idxes_dict,
            loss_gt_dict,
            cfg,
            epoch,
            device,
            logger,
        )
        # log train loss, train lr
        logger.info(f"Train Epoch {epoch}: loss {train_mloss['mloss']:.4f}, time {used_time}, lr {epoch_lr:.4e}")
        if train_mloss["mloss"] != train_mloss["mloss"]:
            logger.fatal("Loss NaN, stop")
            return 1
        metrics["train/loss"] = train_mloss["mloss"]
        metrics["train/loss_spatial"] = train_mloss["mloss_spatial_head"]
        metrics["train/loss_action"] = train_mloss["mloss_action_head"]
        metrics["train/lr"] = epoch_lr
        json_clip_losses_logger.add_entry(train_clip_losses, epoch)
        # save model and backbone
        if save_period > 0 and epoch % save_period == 0:
            weight_save_path = log_weight_path / f"epoch_{epoch}.pt"
            torch.save(sttran_gaze_model.state_dict(), str(weight_save_path))
            logger.info(f"model saved to {weight_save_path}")
            if finetune_backbone:
                backbone_save_path = log_weight_path / f"backbone_{epoch}.pt"
                torch.save(feature_backbone.state_dict(), str(backbone_save_path))
                logger.info(f"backbone saved to {backbone_save_path}")
        weight_last_save_path = log_weight_path / "last.pt"
        torch.save(sttran_gaze_model.state_dict(), str(weight_last_save_path))
        if finetune_backbone:
            backbone_last_save_path = log_weight_path / f"backbone_last.pt"
            torch.save(feature_backbone.state_dict(), str(backbone_last_save_path))
        if train_ratio < 1.0:
            # set to evaluation mode
            vidhoi_train_dataset.eval()
            sttran_gaze_model.eval()
            feature_backbone.eval()
            # evaluation script
            with torch.no_grad():
                val_mloss, val_used_time, metrics_dict = evaluate_gt_bbox(
                    vidhoi_val_dataloader,
                    train_gaze_dict,
                    sttran_gaze_model,
                    loss_val_dict,
                    feature_backbone,
                    loss_type_dict,
                    class_idxes_dict,
                    loss_gt_dict,
                    cfg,
                    epoch,
                    device,
                    logger,
                )
            # save ap per triplet class
            pandas_ap_logger.add_entry(
                metrics_dict["ap"],
                metrics_dict["triplets_type"],
                metrics_dict["triplets_count"],
                object_classes,
                interaction_classes,
                epoch,
                hio=True,
            )
            # calculate validation mean metrics
            for idx_thres, thres in enumerate(interaction_conf_threshold):
                metrics[f"metrics/recall_{thres}"] = np.nanmean(metrics_dict["recall"][idx_thres])
                metrics[f"metrics/precision_{thres}"] = np.nanmean(metrics_dict["precision"][idx_thres])
                metrics[f"metrics/accuracy_{thres}"] = np.nanmean(metrics_dict["accuracy"][idx_thres])
                metrics[f"metrics/f1_{thres}"] = np.nanmean(metrics_dict["f1"][idx_thres])
                metrics[f"metrics/hamming_loss_{thres}"] = np.nanmean(metrics_dict["hamming_loss"][idx_thres])
                for idx_k, k in enumerate(eval_k):
                    metrics[f"metrics/recall@k/recall@{k}_{thres}"] = np.mean(
                        metrics_dict["recall_k"][idx_thres][idx_k]
                    )
            # log val loss
            metrics["val/loss"] = val_mloss["mloss"]
            metrics["val/loss_spatial"] = val_mloss["mloss_spatial_head"]
            metrics["val/loss_action"] = val_mloss["mloss_action_head"]
            val_log_info = f"Val Epoch {epoch}: loss {val_mloss['mloss']:.4f}, time {val_used_time} "
            # mAP independent to threshold, but has different methods
            metrics["metrics/mAP"] = np.nanmean(metrics_dict["ap"])
            # only print the last stats entry
            val_log_info += f"mAP {metrics['metrics/mAP']:.4f} "
            val_log_info += f"precision_{thres} {metrics[f'metrics/precision_{thres}']:.4f} "
            val_log_info += f"recall_{thres} {metrics[f'metrics/recall_{thres}']:.4f} "
            val_log_info += f"accuracy_{thres} {metrics[f'metrics/accuracy_{thres}']:.4f} "
            val_log_info += f"f1_{thres} {metrics[f'metrics/f1_{thres}']:.4f} "
            val_log_info += f"hamming_loss_{thres} {metrics[f'metrics/hamming_loss_{thres}']:.4f} "
            val_log_info += f"recall@{k}_{thres} {metrics[f'metrics/recall@k/recall@{k}_{thres}']:.4f} "
            logger.info(val_log_info)
        # upload wandb log
        wandb_logger.log_metric(metrics)
        wandb_logger.log_epoch()
        # output to csv
        pandas_metric_logger.add_entry(metrics, epoch)
        # scheduler step
        epoch_lr = scheduler.step()
        if train_ratio < 1.0:
            # early stopping step,
            if early_metric.lower() == "map":
                # monitor mAP
                stop, not_improved_epochs = early_stopping.step(epoch, metrics["metrics/mAP"])
            elif early_metric.lower() == "recall":
                # monitor recall@50_0.8
                stop, not_improved_epochs = early_stopping.step(epoch, metrics["metrics/recall@k/recall@50_0.8"])
            elif early_metric.lower() == "loss":
                # monitor negative loss
                stop, not_improved_epochs = early_stopping.step(epoch, -val_mloss["mloss"])
            else:
                # no early stop
                logger.info("Early stopping not set, continue")
                continue
            if stop:
                logger.info(
                    f"Val {early_metric} not improved for {not_improved_epochs} epochs. Early Stopping. "
                    f"Best val {early_metric} at epoch {early_stopping.best_epoch}"
                )
                break
            else:
                if not_improved_epochs == 0:
                    logger.info(f"New best {early_metric}. Store best model")
                    weight_best_save_path = log_weight_path / "best.pt"
                    torch.save(sttran_gaze_model.state_dict(), str(weight_best_save_path))
                else:
                    logger.info(
                        f"Val {early_metric} not improved for {not_improved_epochs} epochs, be patient. "
                        f"Best val {early_metric} at epoch {early_stopping.best_epoch}"
                    )
    # close wandb session
    wandb_logger.finish_run()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/train_hyp.yaml", help="path to hyperparameter configs")
    parser.add_argument("--weights", type=str, default="weights", help="root folder for all pretrained weights")
    parser.add_argument(
        "--finetune-backbone", action="store_true", help="also finetune the ResNet backbone during training"
    )
    parser.add_argument("--data", type=str, default="G:/datasets/VidOR/", help="dataset root path")
    parser.add_argument("--subset-train", type=int, default=-1, help="sub train dataset length")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--warmup", type=int, default=3, help="number of warmup epochs")
    parser.add_argument("--project", default="../runs/sttran_gaze_vidhoi", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--save-period", type=int, default=1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--disable-wandb", action="store_true", help="disable wandb logging")
    parser.add_argument(
        "--gaze", type=str, default="concat", help="how to use gaze features: no, concat, cross, cross_all"
    )
    parser.add_argument("--global-token", action="store_true", help="Use global token, only for cross-attention mode")

    opt = parser.parse_args()
    return opt


def main(opt):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_gt_bbox(opt, device)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, on_terminate_handler)
    opt = parse_opt()
    main(opt)
