#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

sys.path.insert(0, "modules/object_tracking/yolov5")

import argparse
import logging
import numpy as np
from pathlib import Path
import json

from common.metrics import (
    mean_average_precision_qpic,
    sample_based_metrics_all,
    human_centric_top_k_all,
)
from common.logging_utils import PandasClassAPLogger, PandasMetricLogger
from common.config_parser import get_config


def eval_metrics(opt):
    """
    load the result JSON file, compute different metrics, output to some csv files.
    """
    ## path
    result_path = Path(opt.result)
    output_path = opt.output
    if len(output_path) == 0:
        output_path = result_path.parent
    log_file_path = output_path / "eval_metrics.log"
    log_ap_qpic_path = output_path / "class_ap_all_qpic.csv"
    log_metrics_path = output_path / "all_metrics.csv"
    log_human_centric_metrics_path = output_path / "human_centric_metrics.csv"
    dataset_path = Path(opt.data)
    annotation_path = dataset_path / "annotations"
    # collect the object classes
    object_classes = ["__background__"]
    with (annotation_path / "object_classes.txt").open("r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            object_classes.append(line)
    object_classes[9] = "closet/cabinet"
    object_classes[11] = "cup/glass/bottle"
    object_classes[23] = "paper/notebook"
    object_classes[24] = "phone/camera"
    object_classes[31] = "sofa/couch"

    # collect relationship classes
    relationship_classes = []
    with (annotation_path / "relationship_classes.txt").open("r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            relationship_classes.append(line)
    relationship_classes[0] = "looking_at"
    relationship_classes[1] = "not_looking_at"
    relationship_classes[5] = "in_front_of"
    relationship_classes[7] = "on_the_side_of"
    relationship_classes[10] = "covered_by"
    relationship_classes[11] = "drinking_from"
    relationship_classes[13] = "have_it_on_the_back"
    relationship_classes[15] = "leaning_on"
    relationship_classes[16] = "lying_on"
    relationship_classes[17] = "not_contacting"
    relationship_classes[18] = "other_relationship"
    relationship_classes[19] = "sitting_on"
    relationship_classes[20] = "standing_on"
    relationship_classes[25] = "writing_on"

    # load hyperparameters from cfg file
    cfg = get_config(opt.cfg)
    # some eval settings
    rare_limit = cfg["rare_limit"]
    iou_threshold = cfg["iou_threshold"]
    interaction_conf_threshold = cfg["interaction_conf_threshold"]
    recall_k_value = cfg["recall_k"]
    top_k_value = cfg["top_k"]
    frames_to_remove = cfg["frames_to_remove"]
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
    logger.info("Start computing metrics...")
    logger.info({**vars(opt), **cfg})
    # class AP logger
    pandas_ap_qpic_logger = PandasClassAPLogger(log_ap_qpic_path)
    # other metric logger
    metrics_header = [
        "mAP_QPIC",
        "mAP_rare_QPIC",
        "mAP_non_rare_QPIC",
    ]
    metric_names = ["recall", "precision", "accuracy", "f1", "hamming_loss"]
    for name in metric_names:
        for thres in interaction_conf_threshold:
            metrics_header.append(f"{name}_{thres}")
    for thres in interaction_conf_threshold:
        for k in recall_k_value:
            metrics_header.append(f"recall@{k}_{thres}")
    pandas_metric_logger = PandasMetricLogger(log_metrics_path, metrics_header)
    # human-centric metrics
    human_metrics_header = []
    metric_names = ["recall", "precision", "accuracy", "f1"]
    for name in metric_names:
        for thres in interaction_conf_threshold:
            for k in top_k_value:
                human_metrics_header.append(f"{name}_h@{k}_{thres}")
    pandas_human_metric_logger = PandasMetricLogger(log_human_centric_metrics_path, human_metrics_header)

    # load results
    logger.info(f"Load results from {result_path}")
    with result_path.open() as result_file:
        all_results = json.load(result_file)

    # remove 168 results as VidHOI baseline
    all_results_less_168 = []
    for result in all_results:
        image_id = f"{result['video_name']}_{result['frame_id']}"
        if image_id not in frames_to_remove:
            all_results_less_168.append(result)
    logger.info(f"Totally {len(all_results)} results, after remove {len(all_results_less_168)} results")

    ## Metrics
    metrics = {}
    # human-centric metrics
    human_metrics = {}
    logger.info("Computing human-centric sample-based metrics...")
    recall_top_all, precision_top_all, accuracy_top_all, f1_top_all = human_centric_top_k_all(
        all_results, interaction_conf_threshold, top_k_value, logger, iou_threshold
    )
    for idx_thres, thres in enumerate(interaction_conf_threshold):
        for idx_k, k in enumerate(top_k_value):
            human_metrics[f"recall_h@{k}_{thres}"] = np.nanmean(recall_top_all[idx_thres][idx_k])
            human_metrics[f"precision_h@{k}_{thres}"] = np.nanmean(precision_top_all[idx_thres][idx_k])
            human_metrics[f"accuracy_h@{k}_{thres}"] = np.nanmean(accuracy_top_all[idx_thres][idx_k])
            human_metrics[f"f1_h@{k}_{thres}"] = np.nanmean(f1_top_all[idx_thres][idx_k])
    pandas_human_metric_logger.add_entry(human_metrics, 0)

    # all sample based metrics
    logger.info("Computing sample-based metrics...")
    recall_all, precision_all, accuracy_all, f1_all, hamming_loss_all, recall_k_all = sample_based_metrics_all(
        all_results, interaction_conf_threshold, recall_k_value, logger, iou_threshold
    )
    for idx_thres, thres in enumerate(interaction_conf_threshold):
        metrics[f"recall_{thres}"] = np.nanmean(recall_all[idx_thres])
        metrics[f"precision_{thres}"] = np.nanmean(precision_all[idx_thres])
        metrics[f"accuracy_{thres}"] = np.nanmean(accuracy_all[idx_thres])
        metrics[f"f1_{thres}"] = np.nanmean(f1_all[idx_thres])
        metrics[f"hamming_loss_{thres}"] = np.nanmean(hamming_loss_all[idx_thres])
        for idx_k, k in enumerate(recall_k_value):
            metrics[f"recall@{k}_{thres}"] = np.mean(recall_k_all[idx_thres][idx_k])

    # mAP QPIC
    logger.info("Computing mAP QPIC...")
    ap, rare_ap, non_rare_ap, max_recall, count_triplets = mean_average_precision_qpic(
        all_results, logger, rare_limit=rare_limit, iou_threshold=iou_threshold
    )
    triplets_ap = []
    triplets_type = []
    triplets_type_num = []
    for triplet_type, count in count_triplets.items():
        triplets_type.append(triplet_type)
        triplets_ap.append(ap[triplet_type])
        triplets_type_num.append(count)
    pandas_ap_qpic_logger.add_entry(
        triplets_ap, triplets_type, triplets_type_num, object_classes, relationship_classes, 0, hio=True
    )
    metrics["mAP_QPIC"] = np.mean(list(ap.values()))
    metrics["mAP_rare_QPIC"] = np.mean(list(rare_ap.values()))
    metrics["mAP_non_rare_QPIC"] = np.mean(list(non_rare_ap.values()))

    # save results
    pandas_metric_logger.add_entry(metrics, 0)
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")

    logger.info("Finish")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=str, help="path to result JSON file")
    parser.add_argument("--cfg", type=str, default="configs/metrics.yaml", help="Path to config file")
    parser.add_argument(
        "--data", type=str, default="G:/datasets/action_genome/", help="dataset root path, for category names"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="output metrics to the specified path, if not given, to the same directory as the result JSON",
    )

    opt = parser.parse_args()
    return opt


def main(opt):
    eval_metrics(opt)


if __name__ == "__main__":
    # signal.signal(signal.SIGINT, on_terminate_handler)
    opt = parse_opt()
    main(opt)
