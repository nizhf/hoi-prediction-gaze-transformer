#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from tqdm import tqdm
import datetime

from common.train_utils import cat_batch, fill_sttran_entry_train, compute_loss
from common.model_utils import gt_mlm_to_bce, concat_separated_head
from common.metrics import mean_average_precision_qpic, sample_based_metrics_all


@torch.no_grad()
def evaluate_gt_bbox(
    val_dataloader,
    val_gaze_dict,
    model,
    loss_fn_dict,
    feature_backbone,
    loss_type_dict,
    class_idxes_dict,
    loss_gt_dict,
    cfg,
    epoch,
    device,
    logger,
    mlm_add_no_interaction=True,
    human_label=0,
):
    """
    Evaluation with ground-truth bbox, for training.
    """
    interaction_conf_threshold = cfg["interaction_conf_threshold"]
    loss_type = cfg["loss_type"]
    eval_k = cfg["eval_k"]
    sampling_mode = cfg["sampling_mode"]
    rare_limit = cfg["rare_limit"]
    iou_threshold = cfg["iou_threshold"]

    mloss_dict = {f"mloss_{head_name}": 0 for head_name in loss_fn_dict.keys()}
    mloss_dict["mloss"] = 0

    all_results = []

    # iterate through videos
    tbar = tqdm(val_dataloader, unit="batch")
    # process each window, then concatenate
    for idx, (frames_list, annotations_list, meta_info_list) in enumerate(tbar):
        entry_list = []
        video_name_list = []
        frame_idx_offset_list = []
        for frames, annotations, meta_info in zip(frames_list, annotations_list, meta_info_list):
            video_name = meta_info["video_name"]
            frame_ids = meta_info["frame_ids"]
            frame_idx_offset = meta_info["frame_idx_offset"]
            if sampling_mode == "clip":
                video_name_list += [video_name] * len(frame_ids)
                frame_idx_offset_list += [frame_idx_offset + i for i in range(len(frame_ids))]
            else:
                video_name_list.append(video_name)
                frame_idx_offset_list.append(frame_idx_offset)
            # preprocessing
            sttran_frames = frames.to(device)
            # fill the entry, no need to care Oracle or Detection
            entry = fill_sttran_entry_train(
                sttran_frames,
                annotations,
                val_gaze_dict[video_name][frame_idx_offset:],
                feature_backbone,
                meta_info,
                loss_type_dict,
                class_idxes_dict,
                loss_gt_dict,
                device,
                mlm_add_no_interaction,
                human_label,
            )
            tbar.set_description(
                f"Val Epoch {epoch}: {video_name}+{frame_idx_offset}: {len(frames)} frames, "
                f"{len(annotations['pair_idxes'])} gt interactions, {len(entry['pair_idxes'])} pairs, "
                f"mean loss {mloss_dict['mloss']:.4f}"
            )
            # evaluation should not skip any videos
            entry_list.append(entry)

        # concatenate the batch to a single sequence
        interaction_gt_names = ["interactions_gt", *list(loss_gt_dict.values())]
        entry_batch, out_im_idxes = cat_batch(
            entry_list, sampling_mode, cfg["sttran_sliding_window"], interaction_gt_names, cfg["split_window"]
        )
        entry_batch["pair_idxes"] = entry_batch["pair_idxes"].to(device)
        if cfg["split_window"] != "no":
            for i in range(len(entry_batch["full_heatmaps"])):
                entry_batch["full_heatmaps"][i] = entry_batch["full_heatmaps"][i].to(device)
        entry_batch["obj_heatmaps"] = entry_batch["obj_heatmaps"].to(device)
        entry_batch["pred_labels"] = entry_batch["pred_labels"].to(device)
        for gt_name in loss_gt_dict.values():
            entry_batch[gt_name] = entry_batch[gt_name].to(device)
        entry_batch["windows"] = entry_batch["windows"].to(device)
        entry_batch["windows_out"] = entry_batch["windows_out"].to(device)
        entry_batch["exist_mask"] = entry_batch["exist_mask"].to(device)
        entry_batch["change_mask"] = entry_batch["change_mask"].to(device)

        # forward
        pred = model(entry_batch)

        # extreme case: no current human-object pair exists in the future
        if not torch.any(entry_batch["exist_mask"]):
            logger.debug(f"No pairs exist in the future, skip batch {idx}")
            continue

        # compute loss
        loss_dict = compute_loss(pred, entry_batch["exist_mask"], loss_fn_dict, loss_type_dict, loss_gt_dict)
        # compute mean losses
        mloss_dict["mloss"] = ((mloss_dict["mloss"] * idx) + loss_dict["loss"].item()) / (idx + 1)
        for head_name in loss_fn_dict.keys():
            mloss_dict[f"mloss_{head_name}"] = (
                (mloss_dict[f"mloss_{head_name}"] * idx) + loss_dict[head_name].item()
            ) / (idx + 1)

        # concatenate spatial and action output distribution, mlm remove no_interaction in each head
        if cfg["separate_head"]:
            # sigmoid or softmax
            for head_name in loss_type_dict.keys():
                if loss_type_dict[head_name] == "ce":
                    pred[head_name] = torch.softmax(pred[head_name], dim=-1)
                else:
                    pred[head_name] = torch.sigmoid(pred[head_name])
            # in training, length gt = length prediction
            len_preds = len(pred["interactions_gt"])
            interaction_distribution = concat_separated_head(
                pred,
                len_preds,
                loss_type_dict,
                class_idxes_dict,
                device,
                mlm_add_no_interaction,
            )
        else:
            # mlm remove no_interaction
            if loss_type == "mlm" and mlm_add_no_interaction:
                interaction_distribution = pred["interaction_head"][:, :-1]
            else:
                interaction_distribution = pred["interaction_head"]
            # sigmoid to output probability
            interaction_distribution = torch.sigmoid(interaction_distribution)

        # mlm gt to bce format
        interactions_gt = pred["interactions_gt"].cpu().numpy()
        if loss_type == "mlm":
            interactions_gt = gt_mlm_to_bce(interactions_gt, mlm_add_no_interaction)

        idx_left = idx_right = 0
        # process output
        for i, out_im_idx in enumerate(out_im_idxes):
            detected_out_idxes = pred["bboxes"][:, 0] == out_im_idx
            # offset = the first bbox index in this window
            detected_idx_offset = detected_out_idxes.nonzero(as_tuple=True)[0][0]
            pair_out_idxes = pred["im_idxes"] == out_im_idx
            bboxes = pred["bboxes"][detected_out_idxes, 1:]
            pred_labels = pred["pred_labels"][detected_out_idxes]
            confidences = torch.ones_like(pred_labels)
            pair_idxes = pred["pair_idxes"][pair_out_idxes] - detected_idx_offset
            # only need gt bboxes in the last frame, NOTE here pred same as gt
            bboxes_gt = pred["bboxes"][detected_out_idxes, 1:]
            labels_gt = pred["pred_labels"][detected_out_idxes]  # pred same as gt
            # ids_gt = pred["ids_gt"][gt_out_idxes]
            pair_idxes_gt = pred["pair_idxes"][pair_out_idxes] - detected_idx_offset
            # handle interaction distributions
            idx_left = idx_right
            idx_right += len(pair_idxes)
            inter = interaction_distribution[idx_left:idx_right]
            inter_gt = interactions_gt[idx_left:idx_right]
            exist_mask = pred["exist_mask"][idx_left:idx_right]
            change_mask = pred["change_mask"][idx_left:idx_right]
            # window-wise result entry
            window_result = {
                "bboxes": bboxes.numpy().tolist(),
                "pred_labels": pred_labels.cpu().numpy().tolist(),
                "confidences": confidences.cpu().numpy().tolist(),
                "pair_idxes": pair_idxes.cpu().numpy().tolist(),
                "interaction_distribution": inter.cpu().numpy().tolist(),
                "bboxes_gt": bboxes_gt.numpy().tolist(),
                "labels_gt": labels_gt.cpu().numpy().tolist(),
                # "ids_gt": ids_gt.numpy().tolist(),  # ground-truth ids, important for anticipation
                "pair_idxes_gt": pair_idxes_gt.cpu().numpy().tolist(),
                "interactions_gt": inter_gt.tolist(),
            }
            if sampling_mode == "anticipation":
                window_result["exist_mask"] = exist_mask.cpu().numpy().tolist()
                window_result["change_mask"] = change_mask.cpu().numpy().tolist()

            all_results.append(window_result)

            # extreme case: no current human-object pair exists in the future
            if not torch.any(exist_mask):
                logger.debug(f"No pairs exist in the future, in {video_name_list[i]}+{frame_idx_offset_list[i]}")
        # ---------------------- end batch ------------------------

    used_time = datetime.timedelta(seconds=tbar.last_print_t - tbar.start_t)

    logger.info("Computing sample-based metrics...")
    recall_all, precision_all, accuracy_all, f1_all, hamming_loss_all, recall_k_all = sample_based_metrics_all(
        all_results,
        interaction_conf_threshold,
        eval_k,
        logger,
        iou_threshold,
        quiet=True,
    )
    logger.info("Computing mAP with our method...")
    ap, _, _, _, count_triplets = mean_average_precision_qpic(
        all_results, logger, rare_limit, iou_threshold, quiet=True
    )
    triplets_ap = []
    triplets_type = []
    triplets_type_num = []
    for triplet_type, count in count_triplets.items():
        triplets_type.append(triplet_type)
        triplets_ap.append(ap[triplet_type])
        triplets_type_num.append(count)

    metrics_dict = {
        "ap": triplets_ap,
        "precision": precision_all,
        "recall": recall_all,
        "accuracy": accuracy_all,
        "f1": f1_all,
        "hamming_loss": hamming_loss_all,
        "recall_k": recall_k_all,
        "triplets_type": triplets_type,
        "triplets_count": triplets_type_num,
    }

    return mloss_dict, used_time, metrics_dict
