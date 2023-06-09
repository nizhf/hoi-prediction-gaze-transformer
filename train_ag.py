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
from pathlib import Path
import shelve
import signal

from common.config_parser import get_config
from common.action_genome_dataset import AGDataset, dataset_collate_fn
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
    ## Init hyperparameters
    # path
    weights_path = Path(opt.weights)
    ag_dataset_path = Path(opt.data)
    backbone_model_path = weights_path / "backbone" / "resnet101-ag.pth"
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
    # NOTE must separate head for action genome
    # separate_head = cfg["separate_head"]  # separate spatial relation prediction head
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
        "train/loss_attention",
        "train/loss_spatial",
        "train/loss_contacting",
        "train/lr",
        "val/loss",
        "val/loss_attention",
        "val/loss_spatial",
        "val/loss_contacting",
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
    # load training dataset
    ag_train_dataset = AGDataset(
        dataset_dir=ag_dataset_path,
        mode="train",
        min_length=min_clip_length,
        max_length=max_clip_length,
        train_ratio=train_ratio,
        subset_len=subset_train_len,
        subset_shuffle=subset_train_shuffle,
        transform=STTranTransform(img_size=img_size),
        post_transform=ClipRandomHorizontalFlipping(hflip_p),
        annotation_mode=sampling_mode,
        logger=logger,
        future_num=future_num,
        future_type=future_type,
        future_ratio=future_ratio,
    )
    # class names
    object_classes = ag_train_dataset.object_classes
    interaction_classes = ag_train_dataset.relationship_classes
    attention_class_idxes = ag_train_dataset.attention_idxes
    spatial_class_idxes = ag_train_dataset.spatial_idxes
    contacting_class_idxes = ag_train_dataset.contacting_idxes
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
    # analyse training dataset, get histogram of triplets and interactions
    if loss_balance_type == "inverse":
        triplet_info, weight_info = ag_train_dataset.analyse_split_weight(
            save_dir=output_path / run_id, method=loss_balance_type, separate_head=True, power=loss_balance_power
        )
    elif loss_balance_type == "effective":
        triplet_info, weight_info = ag_train_dataset.analyse_split_weight(
            save_dir=output_path / run_id, method=loss_balance_type, separate_head=True, beta=loss_balance_beta
        )
    else:
        triplet_info, weight_info = ag_train_dataset.analyse_split_weight(
            save_dir=output_path / run_id, method=loss_balance_type, separate_head=True
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
    ag_train_dataset.train()
    ag_train_dataloader = DataLoader(
        ag_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=dataset_collate_fn
    )
    if train_ratio < 1.0:
        # validation dataloader
        ag_train_dataset.eval()
        ag_val_dataloader = DataLoader(
            ag_train_dataset, batch_size=batch_size_val, shuffle=False, num_workers=2, collate_fn=dataset_collate_fn
        )
    # load prepared head and gaze features
    logger.info("Loading prepared training head and gaze features")
    train_gaze_filename = ag_dataset_path / "action_genome_gaze/train_frame_gazes_gt_bbox"
    train_gaze_dict = shelve.open(str(train_gaze_filename))

    logger.info("-" * 10 + " Dataset loading finished! " + "-" * 10)

    ## Models
    # load pretrained faseterrcnn backbone on AG
    feature_backbone = FeatureExtractionResNet101(backbone_model_path, download=False, finetune=True).to(device)
    # freeze backbone
    feature_backbone.requires_grad_(False)
    feature_backbone.eval()
    trainable_backbone_names = []
    trainable_backbone_params = []
    for name, param in feature_backbone.named_parameters():
        if param.requires_grad:
            trainable_backbone_names.append(name)
            trainable_backbone_params.append(param)
    logger.info(
        f"ResNet101 feature backbone loaded from {backbone_model_path}. Finetuning weights: {trainable_backbone_names}"
    )
    # separate to three heads
    num_interaction_classes_loss = num_interaction_classes
    # NOTE action genome annotation already contains no_interaction
    separate_head_name = ["attention_head", "spatial_head", "contacting_head"]
    separate_head_num = [num_attention_classes, num_spatial_classes, -1]
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
        # weight not works for mlm
        # NOTE add sigmoid before mlm loss!
        loss_fn_dict = {
            "attention_head": nn.CrossEntropyLoss(reduction="mean"),
            "spatial_head": nn.MultiLabelMarginLoss(reduction="mean"),
            "contacting_head": nn.MultiLabelMarginLoss(reduction="mean"),
        }
        # no weight for validation
        loss_val_dict = {
            "attention_head": nn.CrossEntropyLoss(reduction="mean"),
            "spatial_head": nn.MultiLabelMarginLoss(reduction="mean"),
            "contacting_head": nn.MultiLabelMarginLoss(reduction="mean"),
        }
        loss_type_dict = {"attention_head": "ce", "spatial_head": "mlm", "contacting_head": "mlm"}
    # Focal loss
    elif loss_type == "focal":
        loss_fn_dict = {
            "attention_head": nn.CrossEntropyLoss(
                weight=torch.Tensor(weight_info["weight_train_attention"]).to(device), reduction="mean"
            ),
            "spatial_head": FocalBCEWithLogitLoss(
                gamma=loss_focal_gamma,
                alpha=None,
                pos_weight=torch.Tensor(weight_info["weight_train_spatial"]).to(device),
                reduction="mean",
            ),
            "contacting_head": FocalBCEWithLogitLoss(
                gamma=loss_focal_gamma,
                alpha=None,
                pos_weight=torch.Tensor(weight_info["weight_train_contacting"]).to(device),
                reduction="mean",
            ),
        }
        # no weight for validation
        loss_val_dict = {
            "attention_head": nn.CrossEntropyLoss(reduction="mean"),
            "spatial_head": nn.BCEWithLogitsLoss(reduction="mean"),
            "contacting_head": nn.BCEWithLogitsLoss(reduction="mean"),
        }
        loss_type_dict = {"attention_head": "ce", "spatial_head": "focal", "contacting_head": "focal"}
    # Binary cross-entropy loss with logit loss
    else:
        loss_fn_dict = {
            "attention_head": nn.CrossEntropyLoss(
                weight=torch.Tensor(weight_info["weight_train_attention"]).to(device), reduction="mean"
            ),
            "spatial_head": nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor(weight_info["weight_train_spatial"]).to(device), reduction="mean"
            ),
            "contacting_head": nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor(weight_info["weight_train_contacting"]).to(device), reduction="mean"
            ),
        }
        # no weight for validation
        loss_val_dict = {
            "attention_head": nn.CrossEntropyLoss(reduction="mean"),
            "spatial_head": nn.BCEWithLogitsLoss(reduction="mean"),
            "contacting_head": nn.BCEWithLogitsLoss(reduction="mean"),
        }
        loss_type_dict = {"attention_head": "ce", "spatial_head": "bce", "contacting_head": "bce"}
    class_idxes_dict = {
        "attention_head": attention_class_idxes,
        "spatial_head": spatial_class_idxes,
        "contacting_head": contacting_class_idxes,
    }
    loss_gt_dict = {"attention_head": "attention_gt", "spatial_head": "spatial_gt", "contacting_head": "contacting_gt"}

    # AdamW optimizer
    optimizer = optim.AdamW(sttran_gaze_model.parameters(), lr=init_lr, weight_decay=weight_decay)
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
    logger.info(f"Using {loss_type} loss with AdamW optimizer, TransformerLRScheduler and EarlyStopping")
    logger.info("-" * 10 + " Training Start " + "-" * 10)

    ## Training
    # init some epoch logging
    epoch_lr = init_lr
    for epoch in range(max_epochs):
        # prepare metrics for wandb and pandas
        metrics = {}
        # set to train mode
        ag_train_dataset.train()
        sttran_gaze_model.train()
        # run train procedure for one epoch
        train_mloss, train_clip_losses, used_time = train_one_epoch_gt_bbox(
            ag_train_dataloader,
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
            mlm_add_no_interaction=False,
            human_label=1,
        )
        # log train loss, train lr
        logger.info(f"Train Epoch {epoch}: loss {train_mloss['mloss']:.4f}, time {used_time}, lr {epoch_lr:.4e}")
        metrics["train/loss"] = train_mloss["mloss"]
        metrics["train/loss_attention"] = train_mloss["mloss_attention_head"]
        metrics["train/loss_spatial"] = train_mloss["mloss_spatial_head"]
        metrics["train/loss_contacting"] = train_mloss["mloss_contacting_head"]
        metrics["train/lr"] = epoch_lr
        json_clip_losses_logger.add_entry(train_clip_losses, epoch)
        # save model
        if save_period > 0 and epoch % save_period == 0:
            weight_save_path = log_weight_path / f"epoch_{epoch}.pt"
            torch.save(sttran_gaze_model.state_dict(), str(weight_save_path))
            logger.info(f"model saved to {weight_save_path}")
        weight_last_save_path = log_weight_path / "last.pt"
        torch.save(sttran_gaze_model.state_dict(), str(weight_last_save_path))
        if train_ratio < 1.0:
            # set to evaluation mode
            ag_train_dataset.eval()
            sttran_gaze_model.eval()
            # evaluation script
            with torch.no_grad():
                val_mloss, val_used_time, metrics_dict = evaluate_gt_bbox(
                    ag_val_dataloader,
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
                    mlm_add_no_interaction=False,
                    human_label=1,
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
            metrics["val/loss_attention"] = val_mloss["mloss_attention_head"]
            metrics["val/loss_spatial"] = val_mloss["mloss_spatial_head"]
            metrics["val/loss_contacting"] = val_mloss["mloss_contacting_head"]
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
    parser.add_argument("--data", type=str, default="G:/datasets/action_genome/", help="dataset root path")
    parser.add_argument("--subset-train", type=int, default=-1, help="sub train dataset length")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--warmup", type=int, default=3, help="number of warmup epochs")
    parser.add_argument("--project", default="../runs/sttran_gaze_ag", help="save to project/name")
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
