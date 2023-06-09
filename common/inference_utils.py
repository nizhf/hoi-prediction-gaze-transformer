#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import datetime
from tqdm import tqdm
from collections import defaultdict

from common.model_utils import (
    fill_detected_detection,
    fill_detected_gt_bbox,
    fill_full_heatmaps,
    bbox_features_roi_align,
    construct_sliding_window,
    generate_gt_interactions,
    generate_sliding_window_mask,
    concat_separated_head,
    gt_mlm_to_bce,
)
from common.image_processing import mask_heatmap_roi


@torch.no_grad()
def inference_one_epoch(
    val_dataloader,
    val_gaze_dict,
    val_detection_dict,
    model,
    feature_backbone,
    loss_type_dict,
    class_idxes_dict,
    loss_gt_dict,
    cfg,
    device,
    logger,
    mlm_add_no_interaction=True,
    human_label=0,
):
    """
    Run inference for the full validation dataset, output a dict with all results
    window_result = {
        "video_name": video name
        "frame_id": this frame id
        "bboxes": detected bboxes, [x1, y1, x2, y2]
        "pred_labels": detected labels
        "confidences": detection confidences
        "pair_idxes": all detected human-object pairs
        "interaction_distribution": predicted interaction distribution
        "bboxes_gt": ground-truth object bboxes, [x1, y1, x2, y2]
        "labels_gt": ground-truth object labels
        "ids_gt": ground-truth object ids, maybe important for anticipation
        "pair_idxes_gt": gt human-object pairs
        "interactions_gt": gt interactions
        # only for anticipation:
        "exist_mask": gt pair exists in the future
        "change_mask": gt pair interactions changes in the future
        "bboxes_future_gt": gt bboxes in the future
        "labels_future_gt": gt labels in the future
        "ids_future_gt": gt object id in the future
        "future_frame_id": future frame id
    }
    """
    all_results = []
    loss_type = cfg["loss_type"]
    sampling_mode = cfg["sampling_mode"]
    # iterate through videos
    tbar = tqdm(val_dataloader, unit="batch")
    # process each window, NOTE only consider the first entry in the batch (batch size should be 1)
    for idx, (frames_list, annotations_list, meta_info_list) in enumerate(tbar):
        frames = frames_list[0]
        annotations = annotations_list[0]
        meta_info = meta_info_list[0]
        video_name = meta_info["video_name"]
        frame_ids = meta_info["frame_ids"]
        frame_idx_offset = meta_info["frame_idx_offset"]
        # preprocessing
        sttran_frames = frames.to(device)
        # Detection mode
        if val_detection_dict is not None:
            detected = fill_detected_detection(val_detection_dict[video_name], frame_idx_offset, frame_ids, human_label)
        # Oracle mode
        else:
            detected = fill_detected_gt_bbox(annotations, human_label)
        # fill the entry with detections
        entry = fill_sttran_entry_inference(
            sttran_frames,
            detected,
            val_gaze_dict[video_name][frame_idx_offset:],
            feature_backbone,
            meta_info,
            loss_type_dict,
            class_idxes_dict,
            loss_gt_dict,
            device,
            annotations,
            mlm_add_no_interaction,
            human_label,
        )
        tbar.set_description(
            f"Inference: {video_name}+{frame_idx_offset}: {len(frames)} frames, "
            f"{len(entry['pair_idxes_gt'])} gt pairs, {len(entry['pair_idxes'])} detected pairs"
        )
        tbar.refresh()
        # generate sliding window masks and coresponding ground-truth interaction labels
        interaction_gt_names = ["interactions_gt", *list(loss_gt_dict.values())]
        window_list, exist_mask, change_mask, interaction_gt_dict, window_list_gt = construct_sliding_window(
            entry,
            sampling_mode,
            cfg["sttran_sliding_window"],
            0,
            interaction_gt_names,
            gt=True,
        )
        entry, windows, windows_out, out_im_idxes, out_im_idxes_gt = generate_sliding_window_mask(
            entry, window_list, window_list_gt, split_window=cfg["split_window"]
        )

        interactions_gt = interaction_gt_dict["interactions_gt"]
        # only do model forward if any valid window exists
        if len(windows) > 0:
            # everything to GPU
            entry["pair_idxes"] = entry["pair_idxes"].to(device)
            if cfg["split_window"] != "no":
                for i in range(len(entry["full_heatmaps"])):
                    entry["full_heatmaps"][i] = entry["full_heatmaps"][i].to(device)
            entry["obj_heatmaps"] = entry["obj_heatmaps"].to(device)
            entry["pred_labels"] = entry["pred_labels"].to(device)
            entry["windows"] = windows.to(device)
            entry["windows_out"] = windows_out.to(device)

            # forward, NOTE not using pred here, because pred can be falsely passed to next loop with no valid window
            entry = model(entry)

            # concatenate spatial and action output distribution, mlm remove no_interaction in each head
            if cfg["separate_head"]:
                # sigmoid or softmax
                for head_name in loss_type_dict.keys():
                    if loss_type_dict[head_name] == "ce":
                        entry[head_name] = torch.softmax(entry[head_name], dim=-1)
                    else:
                        entry[head_name] = torch.sigmoid(entry[head_name])
                # in inference, length prediction may != length gt
                # len_preds = len(interactions_gt)
                len_preds = len(entry[list(loss_type_dict.keys())[0]])
                interaction_distribution = concat_separated_head(
                    entry,
                    len_preds,
                    loss_type_dict,
                    class_idxes_dict,
                    device,
                    mlm_add_no_interaction,
                )
            else:
                # mlm remove no_interaction
                if loss_type == "mlm" and mlm_add_no_interaction:
                    interaction_distribution = entry["interaction_head"][:, :-1]
                else:
                    interaction_distribution = entry["interaction_head"]
                # sigmoid to output probability
                interaction_distribution = torch.sigmoid(interaction_distribution)

        # mlm gt to bce format
        if loss_type == "mlm":
            interactions_gt = gt_mlm_to_bce(interactions_gt.numpy(), mlm_add_no_interaction)
        else:
            interactions_gt = interactions_gt.numpy()

        # process output
        idx_left = idx_right = 0
        idx_left_gt = idx_right_gt = 0
        for out_im_idx in out_im_idxes_gt:
            # gt bboxes in the last frame
            gt_out_idxes = entry["bboxes_gt"][:, 0] == out_im_idx
            # offset = the first bbox index in this window
            gt_idx_offset = gt_out_idxes.nonzero(as_tuple=True)[0][0]
            gt_pair_out_idxes = entry["im_idxes_gt"] == out_im_idx
            bboxes_gt = entry["bboxes_gt"][gt_out_idxes, 1:]
            labels_gt = entry["labels_gt"][gt_out_idxes]
            ids_gt = entry["ids_gt"][gt_out_idxes]
            pair_idxes_gt = entry["pair_idxes_gt"][gt_pair_out_idxes] - gt_idx_offset
            # handle gt interaction distributions
            idx_left_gt = idx_right_gt
            idx_right_gt += len(pair_idxes_gt)
            inter_gt = interactions_gt[idx_left_gt:idx_right_gt]
            # window-wise result entry
            window_anno = {
                "video_name": video_name,  # video name
                "frame_id": frame_ids[out_im_idx],  # this frame id
                "bboxes_gt": bboxes_gt.numpy().tolist(),  # ground-truth object bboxes
                "labels_gt": labels_gt.numpy().tolist(),  # ground-truth object labels
                "ids_gt": ids_gt.numpy().tolist(),  # ground-truth ids, important for anticipation
                "pair_idxes_gt": pair_idxes_gt.numpy().tolist(),  # gt pair idxes
                "interactions_gt": inter_gt.tolist(),  # gt interactions
            }
            if sampling_mode == "anticipation":
                anticipation_gt = {
                    "exist_mask": exist_mask[idx_left_gt:idx_right_gt].numpy().tolist(),
                    "change_mask": change_mask[idx_left_gt:idx_right_gt].numpy().tolist(),
                    "labels_future_gt": entry["labels_future_gt"].numpy().tolist(),
                    "bboxes_future_gt": entry["bboxes_future_gt"][:, 1:].numpy().tolist(),
                    "ids_future_gt": entry["ids_future_gt"].numpy().tolist(),
                    "future_frame_id": entry["anticipation_frame_id"],
                }
                window_anno = {**window_anno, **anticipation_gt}

            window_prediction = {
                "bboxes": [],
                "pred_labels": [],
                "confidences": [],
                "pair_idxes": [],
                "interaction_distribution": [],
            }
            # case 1, nothing detected in the full clip, result all []
            if len(entry["bboxes"]) > 0:
                det_out_idxes = entry["bboxes"][:, 0] == out_im_idx
                # case 2, nothing detected in this window, result all []
                if det_out_idxes.any():
                    # offset = the first bbox index in this window
                    det_idx_offset = det_out_idxes.nonzero(as_tuple=True)[0][0]
                    bboxes = entry["bboxes"][det_out_idxes, 1:]
                    pred_labels = entry["pred_labels"][det_out_idxes]
                    confidences = entry["confidences"][det_out_idxes]
                    window_prediction["bboxes"] = bboxes.numpy().tolist()
                    window_prediction["pred_labels"] = pred_labels.cpu().numpy().tolist()
                    window_prediction["confidences"] = confidences.numpy().tolist()

                    pair_out_idxes = entry["im_idxes"] == out_im_idx
                    # case 3, no human-object pair detected, pair_idxes and distribution []
                    if pair_out_idxes.any():
                        # case 4, have everything
                        pair_idxes = entry["pair_idxes"][pair_out_idxes] - det_idx_offset
                        # handle interaction distributions
                        idx_left = idx_right
                        idx_right += len(pair_idxes)
                        inter = interaction_distribution[idx_left:idx_right]
                        window_prediction["pair_idxes"] = pair_idxes.cpu().numpy().tolist()
                        window_prediction["interaction_distribution"] = inter.cpu().numpy().tolist()

            window_result = {**window_anno, **window_prediction}
            all_results.append(window_result)

            # extreme case: no current human-object pair exists in the future
            if not torch.any(entry["exist_mask"][gt_pair_out_idxes]):
                logger.debug(f"No pairs exist in the future, in {video_name}+{frame_idx_offset}")
        # ---------------------- end batch ------------------------

    used_time = datetime.timedelta(seconds=tbar.last_print_t - tbar.start_t)
    logger.info(f"Evaluation Epoch finished, used time {used_time}")

    # output list of result dict
    return all_results


def fill_sttran_entry_inference(
    frames,
    detected,
    heatmap_list,
    feature_backbone,
    meta_info,
    loss_type_dict,
    class_idxes_dict,
    loss_gt_dict,
    device,
    annotations=None,
    mlm_add_no_interaction=True,
    human_label=0,
):
    """
    Prepare entries for model inference

    Args:
        frames (List(Tensor)): list of frame Tensors
        detected (dict): pred_labels, bboxes, ids, pair_idxes, im_idxes
        heatmap_list (dict): preprocessed gaze heatmap list
        feature_backbone: feature backbone class
        video_name (str): video name
        original_shape: original frame shape, for roi align scaling
        frame_idx_offset (int): offset of clip in the video, for match gaze to interaction
        loss_type (str): bce or mlm, affect ground-truth label vector
        num_interaction_classes (int): bce = number of actual interaction classes, mlm must +1
        device: cuda or cpu
        gt_bbox (bool): detected bboxes and labels are ground truth?
        annotations (dict): if inference is running on a video with ground truth, also provide ground truth for comparison

    Returns:
        entry dict
    """
    # from meta_info
    original_shape = meta_info["original_shape"]
    # Results from object detection/ground-truth and pair generation
    pred_labels = torch.LongTensor(detected["pred_labels"])
    bboxes = torch.Tensor(detected["bboxes"])
    pair_idxes = torch.LongTensor(detected["pair_idxes"])
    ids = detected["ids"]
    confidences = torch.Tensor(detected["confidences"])
    im_idxes = torch.LongTensor(detected["im_idxes"])
    pair_human_ids = torch.zeros_like(im_idxes)
    pair_object_ids = torch.zeros_like(im_idxes)
    # make bboxes valid (all >= 0)
    if len(bboxes) > 0:
        bboxes[:, 1:] = torch.maximum(bboxes[:, 1:], torch.Tensor([0]))
    # Results from gaze following, here use prepared heatmap
    if len(im_idxes) > 0:
        im_idx_min = im_idxes.min()
        im_idx_max = im_idxes.max()
        frame_len = im_idx_max - im_idx_min + 1
        # full heatmap for cross-attention in 64x64
        full_heatmaps = defaultdict(lambda: [None] * frame_len)
        obj_heatmaps = []
        # im_idx + frame_idx_offset is the frame index in the original video
        for pair_idx, (im_idx, pair) in enumerate(zip(im_idxes, pair_idxes)):
            human_id = ids[pair[0]]
            object_id = ids[pair[1]]
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
    else:
        full_heatmaps = {}
        obj_heatmaps = torch.Tensor([])

    # Extract features
    # scale up/down bounding boxes, from original frame to transformed frame
    img_size = frames.shape[2:]
    scale = np.min(img_size) / np.min(original_shape[:2])
    if len(bboxes) > 0:
        bboxes[:, 1:] = bboxes[:, 1:] * scale
    # no human-object pair detected
    if len(im_idxes) == 0:
        bboxes_features = []
        union_features = []
        masked_bboxes = []
    # bbox feature, union bbox feature, union bbox mask
    else:
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
    if len(bboxes) > 0:
        bboxes[:, 1:] = bboxes[:, 1:] / scale

    entry = {
        "pred_labels": pred_labels,  # labels from object detector
        "bboxes": bboxes,  # bboxes from object detector
        "ids": ids,  # object id from object detector
        "confidences": confidences,  # score from object detector
        "pair_idxes": pair_idxes,  # subject-object pairs, generated after object detector
        "pair_human_ids": pair_human_ids,  # human id in each pair, from ground-truth
        "pair_object_ids": pair_object_ids,  # object id in each pair, from ground-truth
        "im_idxes": im_idxes,  # each pair belongs to which frame index
        "features": bboxes_features,  # extracted features of each bbox
        "union_feats": union_features,  # extracted features of each pair union bbox
        # "union_box": union_bboxes,  # union bbox of each pair
        "spatial_masks": masked_bboxes,  # mask of each pair
        "full_heatmaps": full_heatmaps,  # full heatmaps of each person
        "obj_heatmaps": obj_heatmaps,  # partial heatmaps of each person in pair
    }
    # only generate ground-truth labels if annotations are given
    if annotations is not None:
        gt_entry = generate_gt_interactions(
            annotations,
            loss_type_dict,
            class_idxes_dict,
            loss_gt_dict,
            mlm_add_no_interaction,
            human_label,
        )
        # fill the entry
        entry = {
            **entry,  # detections and processed features
            **gt_entry,  # ground-truth object labels, interaction labels, and future labels
        }
        if "anticipation" in annotations:
            entry["anticipation_frame_id"] = meta_info["anticipation_frame_id"]

    return entry


# real application
def inference_once_real():
    pass
