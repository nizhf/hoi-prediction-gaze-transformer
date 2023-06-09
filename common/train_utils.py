#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from tqdm import tqdm
import datetime
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_

from common.image_processing import mask_heatmap_roi
from common.model_utils import (
    generate_gt_interactions,
    bbox_features_roi_align,
    construct_sliding_window,
    generate_sliding_window_mask,
    fill_full_heatmaps,
)


def train_one_epoch_gt_bbox(
    train_dataloader,
    train_gaze_dict,
    model,
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
    mlm_add_no_interaction=True,
    human_label=0,
):
    """
    For training, only consider the ground-truth objects and pairs.
    """
    # init some loss logging
    clip_losses = []
    mloss_dict = {f"mloss_{head_name}": 0 for head_name in loss_fn_dict.keys()}
    mloss_dict["mloss"] = 0
    # iterate through batches
    tbar = tqdm(train_dataloader, unit="batch")
    sampling_mode = cfg["sampling_mode"]
    # process each batch
    for idx, (frames_list, annotations_list, meta_info_list) in enumerate(tbar):
        entry_list = []
        entry_length = 0
        # process each window/clip
        for frames, annotations, meta_info in zip(frames_list, annotations_list, meta_info_list):
            video_name = meta_info["video_name"]
            frame_idx_offset = meta_info["frame_idx_offset"]
            # preprocessing
            sttran_frames = frames.to(device)
            # fill the entry
            entry = fill_sttran_entry_train(
                sttran_frames,
                annotations,
                train_gaze_dict[video_name][frame_idx_offset:],
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
                f"Train Epoch {epoch}: {video_name}+{frame_idx_offset}, {len(frames)} frames, "
                f"{len(annotations['pair_idxes'])} gt interactions, {len(entry['pair_idxes'])} pairs, "
                f"mean loss {mloss_dict['mloss']:.4f}"
            )
            # skip still too large clips
            if entry_length + len(entry["pair_idxes"]) > cfg["max_interaction_pairs"]:
                logger.debug(
                    f"{idx}: Too long {entry_length}+{len(entry['pair_idxes'])}, "
                    f"skip window {video_name}+{frame_idx_offset}, actual batch size {len(entry_list)}"
                )
                continue
            entry_list.append(entry)
            entry_length += len(entry["pair_idxes"])
            # ---------------------- end window/clip ----------------------------

        # all clips too long, skip current step
        if len(entry_list) == 0:
            continue

        # concatenate the batch to a single sequence
        interaction_gt_names = ["interactions_gt", *list(loss_gt_dict.values())]
        entry_batch, _ = cat_batch(
            entry_list,
            sampling_mode,
            cfg["sttran_sliding_window"],
            interaction_gt_names,
            cfg["split_window"],
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

        # extreme case: no current human-object pair exists in the future
        if not torch.any(entry_batch["exist_mask"]):
            logger.debug(f"No pairs exist in the future, skip batch {idx}")
            continue

        # forward
        pred = model(entry_batch)

        # compute loss
        loss_dict = compute_loss(pred, entry_batch["exist_mask"], loss_fn_dict, loss_type_dict, loss_gt_dict)
        if torch.isnan(loss_dict["loss"]):
            logger.fatal(f"NaN in loss at {idx}, stop training")
            logger.debug(model.parameters())
            mloss_dict["mloss"] = torch.nan
            break

        # backward propagation
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        clip_grad_norm_(
            model.parameters(),
            max_norm=cfg["clip_norm_max"],
            norm_type=cfg["clip_norm_type"],
        )
        clip_grad_norm_(
            feature_backbone.parameters(),
            max_norm=cfg["clip_norm_max"],
            norm_type=cfg["clip_norm_type"],
        )
        optimizer.step()
        clip_losses.append(loss_dict["loss"].item())
        mloss_dict["mloss"] = ((mloss_dict["mloss"] * idx) + loss_dict["loss"].item()) / (idx + 1)
        for head_name in loss_fn_dict.keys():
            mloss_dict[f"mloss_{head_name}"] = (
                (mloss_dict[f"mloss_{head_name}"] * idx) + loss_dict[head_name].item()
            ) / (idx + 1)
        # try to clear gpu memory cache
        # torch.cuda.empty_cache()
        # ---------------------- end batch ------------------------

    used_time = datetime.timedelta(seconds=tbar.last_print_t - tbar.start_t)
    return mloss_dict, clip_losses, used_time


def compute_loss(pred, exist_mask, loss_fn_dict, loss_type_dict, loss_gt_dict):
    loss_dict = {}
    loss_sum = 0
    # For each prediction head
    for loss_name in loss_fn_dict.keys():
        # mlm need sigmoid, others not
        if loss_type_dict[loss_name] == "mlm":
            loss = loss_fn_dict[loss_name](
                torch.sigmoid(pred[loss_name][exist_mask]), pred[loss_gt_dict[loss_name]][exist_mask]
            )
        else:
            loss = loss_fn_dict[loss_name](pred[loss_name][exist_mask], pred[loss_gt_dict[loss_name]][exist_mask])
        loss_sum += loss
        loss_dict[loss_name] = loss
    loss_dict["loss"] = loss_sum
    return loss_dict


def fill_sttran_entry_train(
    frames,
    annotations,
    heatmap_list,
    feature_backbone,
    meta_info,
    loss_type_dict,
    class_idxes_dict,
    loss_gt_dict,
    device,
    mlm_add_no_interaction=True,
    human_label=0,
):
    """
    Prepare entries for model training, use ground-truth bbox, generate all possible pairs

    Args:
        frames (List(Tensor)): list of frame Tensors
        annotations (dict): everything from annotation
        heatmap_list (dict): preprocessed gaze heatmap list
        feature_backbone: feature backbone class
        meta_info (dict): meta_info from the dataset
        loss_type_dict (dict): bce or mlm, affect ground-truth label vector
        class_idxes_dict (dict): interaction types
        loss_gt_dict (dict): names for separated heads
        device: cuda or cpu
        mlm_add_no_interaction (bool): mlm add no interaction
        human_label: label of human

    Returns:
        entry dict
    """
    # generate ground-truth labels from the annotations
    gt_entry = generate_gt_interactions(
        annotations,
        loss_type_dict,
        class_idxes_dict,
        loss_gt_dict,
        mlm_add_no_interaction,
        human_label,
    )
    pred_labels = gt_entry["labels_gt"]  # use gt labels as detected labels
    bboxes = gt_entry["bboxes_gt"]  # use gt bboxes as detected bboxes
    ids = gt_entry["ids_gt"]  # use gt ids as detected ids
    pair_idxes = gt_entry["pair_idxes_gt"]  # generated pair_idxes from gt
    im_idxes = gt_entry["im_idxes_gt"]  # im_idxes for the generated pair_idxes
    pair_human_ids = torch.zeros_like(im_idxes)
    pair_object_ids = torch.zeros_like(im_idxes)
    im_idx_min = im_idxes.min()
    im_idx_max = im_idxes.max()
    frame_len = im_idx_max - im_idx_min + 1
    # Make bboxes valid (all >= 0)
    bboxes[:, 1:] = torch.maximum(bboxes[:, 1:], torch.Tensor([0]))
    original_shape = meta_info["original_shape"]
    # full heatmap for cross-attention in 64x64
    full_heatmaps = defaultdict(lambda: [None] * frame_len)
    obj_heatmaps = []  # croped heatmap in 27x27
    # im_idx + frame_idx_offset is the frame index in the original video
    for pair_idx, (im_idx, pair) in enumerate(zip(im_idxes, pair_idxes)):
        human_id = ids[pair[0]].item()
        object_id = ids[pair[1]].item()
        object_bbox = bboxes[pair[1]][1:]
        pair_human_ids[pair_idx] = human_id
        pair_object_ids[pair_idx] = object_id
        heatmap = heatmap_list[im_idx][human_id]
        # no head detected, set zero
        if not isinstance(heatmap, np.ndarray):
            heatmap = np.zeros((64, 64))
        # normalize to [0, 1]
        else:
            # TODO maybe no need to normalize
            heatmap_min, heatmap_max = np.min(heatmap), np.max(heatmap)
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min) * 1
        # horizontal flipping
        if "hflip" in meta_info and meta_info["hflip"]:
            heatmap = np.flip(heatmap, -1)
        # fill full_heatmaps list with current heatmap
        full_heatmaps[human_id][im_idx - im_idx_min] = heatmap
        # get the heatmap part in the object bbox
        masked_heatmap = mask_heatmap_roi(heatmap, object_bbox, original_shape)
        obj_heatmaps.append(masked_heatmap)
    # process human not visible in between
    full_heatmaps = fill_full_heatmaps(full_heatmaps, frame_len)
    for human_id, human_hms in full_heatmaps.items():
        full_heatmaps[human_id] = torch.Tensor(np.array(human_hms)).unsqueeze(1)
    obj_heatmaps = torch.Tensor(np.array(obj_heatmaps)).unsqueeze(1)

    # Extract features
    # scale up/down bounding boxes, from original frame to transformed frame
    img_size = frames.shape[2:]
    scale = np.min(img_size) / np.min(original_shape[:2])
    bboxes[:, 1:] = bboxes[:, 1:] * scale
    # bbox feature, union bbox feature, union bbox mask
    bboxes_features, union_features, masked_bboxes = bbox_features_roi_align(
        frames,
        bboxes,
        pair_idxes,
        im_idxes,
        feature_backbone,
        small_batch_size=5,
        device=device,
    )

    # scale back, important for inference
    bboxes[:, 1:] = bboxes[:, 1:] / scale

    # fill the entry
    entry = {
        "pred_labels": pred_labels,  # labels from ground-truth
        "bboxes": bboxes,  # bboxes from ground-truth
        "ids": ids,  # object id from ground-truth
        # "confidences": confidences,  # score from object detector, no need for training
        "pair_idxes": pair_idxes,  # subject-object pairs, from ground-truth
        "pair_human_ids": pair_human_ids,  # human id in each pair, from ground-truth
        "pair_object_ids": pair_object_ids,  # object id in each pair, from ground-truth
        "im_idxes": im_idxes,  # each pair belongs to which frame index
        "features": bboxes_features,  # extracted features of each bbox
        "union_feats": union_features,  # extracted features of each pair union bbox
        # "union_box": union_bboxes,  # union bbox of each pair
        "spatial_masks": masked_bboxes,  # mask of each pair
        "full_heatmaps": full_heatmaps,  # full heatmaps of each person
        "obj_heatmaps": obj_heatmaps,  # partial heatmaps of each person-object pair
        "exist_mask": gt_entry["exist_mask"],  # mask of existing future pair
        "change_mask": gt_entry["change_mask"],  # mask of changed human-object pair
        "labels_gt": gt_entry["labels_gt"],  # ground-truth labels, no need for training
        "bboxes_gt": gt_entry["bboxes_gt"],  # ground-truth bboxes, no need for training
        "im_idxes_gt": gt_entry["im_idxes_gt"],  # ground-truth im_idxes
    }
    # ground-truth separated head
    interaction_gt_names = ["interactions_gt", *list(loss_gt_dict.values())]
    for gt_name in interaction_gt_names:
        entry[gt_name] = gt_entry[gt_name]
    return entry


def cat_batch(entry_list, sampling_mode, sliding_window, interaction_gt_names, split_window):
    """
    Concatenate a list of entries to a batch entry. Also generate sliding windows.
    For training, not for inference with detections
    """
    entry_batch = defaultdict(list)
    entry_batch["full_heatmaps"] = {}
    window_list = []
    im_idx_offset = 0  # frame index offset
    bbox_idx_offset = 0  # bbox index offset
    pair_idx_offset = 0  # pair representation index offset
    id_offset = 0  # object id offset
    # for each entry
    for entry in entry_list:
        # append to batch entry
        entry_batch["pred_labels"].append(entry["pred_labels"])
        bboxes = entry["bboxes"]
        bboxes[:, 0] += im_idx_offset
        entry_batch["bboxes"].append(bboxes)
        entry_batch["ids"].append(entry["ids"] + id_offset)
        entry_batch["pair_idxes"].append(entry["pair_idxes"] + bbox_idx_offset)
        entry_batch["pair_human_ids"].append(entry["pair_human_ids"] + id_offset)
        entry_batch["pair_object_ids"].append(entry["pair_object_ids"] + id_offset)
        entry_batch["im_idxes"].append(entry["im_idxes"] + im_idx_offset)
        entry_batch["features"].append(entry["features"])
        entry_batch["union_feats"].append(entry["union_feats"])
        entry_batch["spatial_masks"].append(entry["spatial_masks"])
        for human_id, human_gaze_heatmaps in entry["full_heatmaps"].items():
            entry_batch["full_heatmaps"][human_id + id_offset] = human_gaze_heatmaps
        entry_batch["obj_heatmaps"].append(entry["obj_heatmaps"])

        im_idx_max = torch.max(entry["im_idxes"])
        im_idx_offset += im_idx_max + 1
        bbox_idx_offset += len(entry["bboxes"])
        id_offset += torch.max(entry["ids"]).item() + 1

        # generate sliding windows
        window_len = len(entry["im_idxes"])
        window_list_entry, exist_mask, change_mask, interaction_gt_dict, _ = construct_sliding_window(
            entry,
            sampling_mode,
            sliding_window,
            pair_idx_offset,
            interaction_gt_names,
            gt=True,
        )
        pair_idx_offset += window_len
        window_list += window_list_entry
        for gt_name in interaction_gt_names:
            entry_batch[gt_name].append(interaction_gt_dict[gt_name])
        entry_batch["exist_mask"].append(exist_mask)
        entry_batch["change_mask"].append(change_mask)

    # concatenate everything
    entry_batch["pred_labels"] = torch.cat(entry_batch["pred_labels"])
    entry_batch["bboxes"] = torch.cat(entry_batch["bboxes"])
    entry_batch["ids"] = torch.cat(entry_batch["ids"])
    entry_batch["pair_idxes"] = torch.cat(entry_batch["pair_idxes"])
    entry_batch["pair_human_ids"] = torch.cat(entry_batch["pair_human_ids"])
    entry_batch["pair_object_ids"] = torch.cat(entry_batch["pair_object_ids"])
    entry_batch["im_idxes"] = torch.cat(entry_batch["im_idxes"])
    entry_batch["features"] = torch.cat(entry_batch["features"])
    entry_batch["union_feats"] = torch.cat(entry_batch["union_feats"])
    entry_batch["spatial_masks"] = torch.cat(entry_batch["spatial_masks"])
    entry_batch["obj_heatmaps"] = torch.cat(entry_batch["obj_heatmaps"])
    for gt_name in interaction_gt_names:
        entry_batch[gt_name] = torch.cat(entry_batch[gt_name])
    entry_batch["exist_mask"] = torch.cat(entry_batch["exist_mask"])
    entry_batch["change_mask"] = torch.cat(entry_batch["change_mask"])

    # convert (window_start, window_end) to masks
    entry_batch, windows, windows_out, out_im_idxes, _ = generate_sliding_window_mask(
        entry_batch, window_list, split_window=split_window
    )
    entry_batch["windows"] = windows
    entry_batch["windows_out"] = windows_out

    return entry_batch, out_im_idxes
