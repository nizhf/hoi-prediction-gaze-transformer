#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from common.image_processing import union_roi, mask_union_roi, convert_annotation_frame_to_video


def fill_detected_detection(detections, frame_idx_offset, frame_ids, human_label=0):
    """
    Fill the detected dict with real detections

    Args:
        detections (dict): labels, bboxes, ids

    Returns:
        detected (dict): bboxes, pred_labels, ids, pair_idxes, im_idxes
    """
    len_clip = len(frame_ids)
    det_labels = detections["labels"][frame_idx_offset : frame_idx_offset + len_clip]
    det_bboxes = detections["bboxes"][frame_idx_offset : frame_idx_offset + len_clip]
    det_ids = detections["ids"][frame_idx_offset : frame_idx_offset + len_clip]
    det_confidences = detections["confidences"][frame_idx_offset : frame_idx_offset + len_clip]
    bboxes, ids, pred_labels, confidences = convert_annotation_frame_to_video(
        det_bboxes, det_ids, det_labels, det_confidences
    )
    if len(bboxes) == 0:
        # no detection
        pair_idxes = []
        im_idxes = []
    else:
        # Generate human-object pairs
        pair_idxes, im_idxes = bbox_pair_generation(bboxes, pred_labels, human_label)
    detected = {
        "bboxes": bboxes,
        "pred_labels": pred_labels,
        "ids": ids,
        "confidences": confidences,
        "pair_idxes": pair_idxes,
        "im_idxes": im_idxes,
    }
    return detected


def fill_detected_gt_bbox(annotations, human_label=0):
    """
    Fill the detected dict with ground-truth bounding boxes

    Args:
        annotations (dict): labels, bboxes, ids

    Returns:
        detected (dict): bboxes, pred_labels, ids, pair_idxes, im_idxes
    """
    pred_labels = annotations["labels"]
    bboxes = annotations["bboxes"]
    ids = annotations["ids"]
    # Generate human-object pairs
    pair_idxes, im_idxes = bbox_pair_generation(bboxes, pred_labels, human_label)
    detected = {
        "bboxes": bboxes,
        "pred_labels": pred_labels,
        "ids": ids,
        "confidences": [1.0] * len(bboxes),
        "pair_idxes": pair_idxes,
        "im_idxes": im_idxes,
    }
    return detected


def bbox_pair_generation(bboxes, labels, human_label=0):
    """
    From list of bboxes and labels, generate pairs

    Args:
        bboxes (List[nx5]): object bboxes
        labels (List[nx1]): object labels

    Returns:
        pair_idxes, im_idxes
    """
    pair_idxes = []
    im_idxes = []
    idx_left = 0
    idx_right = 0
    last_frame_idx = 0
    for idx, bbox in enumerate(bboxes):
        # new frame, process last frame
        if bbox[0] != last_frame_idx:
            idx_right = idx
            for i_subj in range(idx_left, idx_right):
                # human, pair with all other objects
                if labels[i_subj] == human_label:
                    for i_obj in range(idx_left, idx_right):
                        if i_subj != i_obj:
                            pair_idxes.append([i_subj, i_obj])
                            im_idxes.append(last_frame_idx)
            # set for new frame
            idx_left = idx
        last_frame_idx = bbox[0]
    # add the last frame
    idx_right = idx + 1
    for i_subj in range(idx_left, idx_right):
        # human, pair with all other objects
        if labels[i_subj] == human_label:
            for i_obj in range(idx_left, idx_right):
                if i_subj != i_obj:
                    pair_idxes.append([i_subj, i_obj])
                    im_idxes.append(last_frame_idx)

    return pair_idxes, im_idxes


def fill_full_heatmaps(full_heatmaps, frame_len):
    """
    For each human, fill empty heatmap frame with previous or latter heatmap

    Args:
        full_heatmaps (dict): full heatmaps
        frame_len (int): length of sequence
    """
    for human_id, human_heatmaps in full_heatmaps.items():
        for i in range(frame_len):
            if human_heatmaps[i] is None:
                # first try fill with previous gaze
                for j in range(i - 1, -1, -1):
                    if human_heatmaps[j] is not None:
                        human_heatmaps[i] = human_heatmaps[j]
                        break
                # then try fill with latter gaze
                if human_heatmaps[i] is None:
                    for j in range(i + 1, frame_len):
                        if human_heatmaps[j] is not None:
                            human_heatmaps[i] = human_heatmaps[j]
                            break
                # should be filled, otherwise this human does not exist at all
                assert human_heatmaps[i] is not None, "full gazemaps empty"
    return full_heatmaps


def match_pairs_generated_gt(
    pair_idxes_generated: list,
    pair_idxes_gt: list,
    interactions_gt: list,
):
    """
    Match the generated pair_idxes to gt pair_idxes

    Args:
        pair_idxes_generated (list): generated pair_idxes
        pair_idxes_gt (list): ground-truth pair_idxes
        interactions_gt (list): ground-truth interactions

    Returns:
        list: list of gt interactions for the generated pair_idxes
    """
    matched_interactions = []
    for pair_idx_generated in pair_idxes_generated:
        try:
            idx = pair_idxes_gt.index(pair_idx_generated)
            matched_interactions.append(interactions_gt[idx])
        except ValueError:
            # this gt pair has no gt interaction label
            matched_interactions.append([])
    return matched_interactions


def match_pairs_anticipation_gt(
    pair_idxes_generated: list,
    ids_generated: list,
    anticipation: dict,
    human_label=0,
):
    """
    Match pairs in the latest frame of a window to the future frame annotation

    Args:
        pair_idxes_generated (list): generated pair_idxes
        ids_generated (list): generated ids
        anticipation (dict): anticipation entries
        human_label (int): label for human, 0 in VidHOI, 1 in Action Genome

    Returns:
        Future gt interactions for generated pair_idxes, whether the pair exists, whether the interaction changes
    """
    # generate all possible human-object pairs in the future
    pairs_future = []
    for label, sub_id in zip(anticipation["labels"], anticipation["ids"]):
        # human, pair with all other objects
        if label == human_label:
            for obj_id in anticipation["ids"]:
                if sub_id != obj_id:
                    pairs_future.append([sub_id, obj_id])

    # match current pairs to future pairs, exclude those pairs not existing in the future (mask False)
    matched_interactions = []
    exist_mask = []
    change_mask = []
    for pair_idx_generated in pair_idxes_generated:
        sub_idx = pair_idx_generated[0]
        obj_idx = pair_idx_generated[1]
        pair_id = [ids_generated[sub_idx], ids_generated[obj_idx]]
        # this pair does not exist in the future, set False
        if pair_id not in pairs_future:
            exist_mask.append(False)
            change_mask.append(False)
            matched_interactions.append([])
        # pair exists in the future, check ground-truth interaction anticipation
        else:
            exist_mask.append(True)
            try:
                interaction_idx = anticipation["pair_ids"].index(pair_id)
                matched_interactions.append(anticipation["interactions"][interaction_idx])
            except ValueError:
                matched_interactions.append([])
            # check pair interactions change in the future
            if pair_id in anticipation["changes"]["pair_ids"]:
                change_mask.append(True)
            else:
                change_mask.append(False)
    return matched_interactions, exist_mask, change_mask


def generate_mlm_gt(
    matched_interactions,
    class_idxes,
    mlm_add_no_interaction=True,
):
    # Multi-label Margin loss: [labels, -1, -1, ...]
    # add no_interaction at the end if needed
    num_interaction_classes = len(class_idxes)
    if mlm_add_no_interaction:
        num_interaction_classes += 1
    interactions_gt = -torch.ones(
        (len(matched_interactions), num_interaction_classes),
        dtype=torch.long,
    )
    for pair_idx, interaction_frame in enumerate(matched_interactions):
        current_idx = 0
        for interaction in interaction_frame:
            if interaction in class_idxes:
                interactions_gt[pair_idx, current_idx] = class_idxes.index(interaction)
                current_idx += 1
        # no gt
        if current_idx == 0 and mlm_add_no_interaction:
            interactions_gt[pair_idx, 0] = num_interaction_classes - 1
    return interactions_gt


def generate_bce_gt(
    matched_interactions,
    class_idxes,
):
    # Binary Cross-Entropy style loss: one-hot encoding
    num_interaction_classes = len(class_idxes)
    interactions_gt = torch.zeros((len(matched_interactions), num_interaction_classes))
    for pair_idx, interaction_frame in enumerate(matched_interactions):
        for interaction in interaction_frame:
            if interaction in class_idxes:
                interaction_idx = class_idxes.index(interaction)
                interactions_gt[pair_idx, interaction_idx] = 1
    return interactions_gt


def generate_gt_interactions(
    annotations,
    loss_type_dict,
    class_idxes_dict,
    loss_gt_dict,
    mlm_add_no_interaction=True,
    human_label=0,
):
    """
    Generate ground-truth interaction labels, i.e. 50-d vector {0, 1} for each human-object pair.
    """
    labels_gt = torch.LongTensor(annotations["labels"])
    bboxes_gt = torch.Tensor(annotations["bboxes"])
    ids_gt = torch.LongTensor(annotations["ids"])
    # generate all possible gt human-object pairs, including those with no interaction
    pair_idxes_gt_generated, im_idxes_gt_generated = bbox_pair_generation(
        annotations["bboxes"], annotations["labels"], human_label
    )
    pair_idxes_gt = torch.LongTensor(pair_idxes_gt_generated)
    im_idxes_gt = torch.LongTensor(im_idxes_gt_generated)
    # anticipation, match generated gt pairs to the future ground-truth HOI annotations
    if "anticipation" in annotations:
        matched_interactions, exist_mask, change_mask = match_pairs_anticipation_gt(
            pair_idxes_gt_generated,
            annotations["ids"],
            annotations["anticipation"],
        )
        labels_future_gt = annotations["anticipation"]["labels"]
        bboxes_future_gt = annotations["anticipation"]["bboxes"]
        ids_future_gt = annotations["anticipation"]["ids"]
    # not anticipation, match generated gt pairs to current ground-truth HOI annotation
    else:
        matched_interactions = match_pairs_generated_gt(
            pair_idxes_gt_generated,
            annotations["pair_idxes"],
            annotations["interactions"],
        )
        exist_mask = [True] * len(pair_idxes_gt_generated)
        change_mask = [False] * len(pair_idxes_gt_generated)
        labels_future_gt = []
        bboxes_future_gt = []
        ids_future_gt = []
    exist_mask = torch.BoolTensor(exist_mask)
    change_mask = torch.BoolTensor(change_mask)
    labels_future_gt = torch.LongTensor(labels_future_gt)
    bboxes_future_gt = torch.Tensor(bboxes_future_gt)
    ids_future_gt = torch.LongTensor(ids_future_gt)

    # generate ground truth label vector
    num_interaction_classes = 0
    for class_idxes in class_idxes_dict.values():
        num_interaction_classes += len(class_idxes)
    # all interactions in one
    class_idxes = [i for i in range(num_interaction_classes)]
    if "mlm" in loss_type_dict.values():
        interactions_gt = generate_mlm_gt(matched_interactions, class_idxes, mlm_add_no_interaction)
    else:
        interactions_gt = generate_bce_gt(matched_interactions, class_idxes)
    interaction_gt_dict = {"interactions_gt": interactions_gt}
    # separated heads
    for loss_name in loss_type_dict.keys():
        loss_type = loss_type_dict[loss_name]
        class_idxes = class_idxes_dict[loss_name]
        gt_name = loss_gt_dict[loss_name]
        if loss_type == "mlm":
            interaction_gt_dict[gt_name] = generate_mlm_gt(matched_interactions, class_idxes, mlm_add_no_interaction)
        else:
            interaction_gt_dict[gt_name] = generate_bce_gt(matched_interactions, class_idxes)

    gt = {
        "labels_gt": labels_gt,  # ground-truth labels
        "bboxes_gt": bboxes_gt,  # ground-truth bboxes
        "ids_gt": ids_gt,  # ground-truth ids
        "pair_idxes_gt": pair_idxes_gt,  # ground-truth pair_idxes
        "im_idxes_gt": im_idxes_gt,  # ground-truth im_idxes
        "exist_mask": exist_mask,  # future human-object exist mask
        "change_mask": change_mask,  # future interaction change mask
        "labels_future_gt": labels_future_gt,  # gt future labels
        "bboxes_future_gt": bboxes_future_gt,  # gt future bboxes
        "ids_future_gt": ids_future_gt,  # gt future ids
    }
    # ground-truth interactions, separated heads
    for gt_name in interaction_gt_dict.keys():
        gt[gt_name] = interaction_gt_dict[gt_name]
    return gt


def bbox_features_roi_align(
    frames,
    bboxes,
    pair_idx,
    im_idx,
    feature_backbone,
    small_batch_size=5,
    device="cuda:0",
):
    """
    Generate bbox features, union bbox features, and union bbox masks

    Args:
        frames (Tensor): all frames
        bboxes (Tensor): [description]
        pair_idx (Tensor): [description]
        im_idx (Tensor): [description]
        feature_backbone (backbone object): [description]
        small_batch_size (int, optional): [description]. Defaults to 5.
        device (str, optional): [description]. Defaults to "cuda:0".

    Returns:
        [type]: [description]
    """
    # generate base feature map for each frame
    frame_idx = 0
    base_feature_maps = torch.tensor([]).to(device)
    while frame_idx < frames.shape[0]:
        # compute images in batch and collect all frames data in the video
        # because of GPU RAM usage
        if frame_idx + small_batch_size < frames.shape[0]:
            im_data_small_batch = frames[frame_idx : frame_idx + small_batch_size]
        else:
            im_data_small_batch = frames[frame_idx:]
        base_feature_maps_batch = feature_backbone.backbone_base(im_data_small_batch)
        base_feature_maps = torch.cat((base_feature_maps, base_feature_maps_batch), 0)
        frame_idx += small_batch_size
    # roi features
    bboxes_features, _ = feature_backbone(frames, bboxes.to(device), base_feature_maps)
    # union bboxes and union features
    union_bboxes = union_roi(bboxes, pair_idx, im_idx)
    # union_features = feature_backbone.roi_align(base_feature_maps, union_bboxes)
    union_features, _ = feature_backbone(frames, union_bboxes.to(device), base_feature_maps)

    # masked union bboxes
    pair_rois = torch.cat((bboxes[pair_idx[:, 0], 1:], bboxes[pair_idx[:, 1], 1:]), 1).data.cpu().numpy()
    masked_bboxes = torch.Tensor(mask_union_roi(pair_rois, 27) - 0.5).to(device)
    return bboxes_features, union_features, masked_bboxes


def construct_sliding_window(entry, sampling_mode, sliding_window, pair_idx_offset, interaction_gt_names, gt=True):
    """
    Construct sliding windows from the single entry, with given pair_idx offset

    Args:
        entry (dict): entries
        sampling_mode (str): data sampling mode (clip, window, anticipation)
        sliding_window (int): length of sliding window
        pair_idx_offset (int): offset in the batch
        interaction_gt_names (list): interaciton names
        gt (bool, optional): ground-truth labels available? Defaults to True.

    Returns:
        _type_: _description_
    """
    # given ground-truth, we know how many frames we have
    if gt:
        im_idx_max = torch.max(entry["im_idxes_gt"])
        windows_gt = []
        window_len_gt = len(entry["im_idxes_gt"])
    else:
        if len(entry["im_idxes"]) > 0:
            im_idx_max = torch.max(entry["im_idxes"])
        else:
            im_idx_max = -1  # no valid human-object pairs in this window, skip
    windows = []
    interaction_gt_dict = {}
    # clip mode, sliding window for all frames
    if sampling_mode == "clip":
        # windows for detections
        for im_idx in range(im_idx_max + 1):
            im_idx_start = max(0, im_idx - sliding_window + 1)
            # im_idx_end needs +1, because [start:end] does not include end
            im_idx_end = im_idx + 1
            window_start = -1
            end_exist = False
            pair_idx = -1
            for pair_idx, idx in enumerate(entry["im_idxes"]):
                # nothing detected
                if len(entry["im_idxes"]) == 0:
                    continue
                # the first pair_idx meets the im_idx interval
                if idx >= im_idx_start and idx < im_idx_end and window_start < 0:
                    window_start = pair_idx
                # the last im_idx in this window exists
                if idx == im_idx_end - 1:
                    end_exist = True
                # reach the next window, -1
                if idx >= im_idx_end:
                    pair_idx -= 1
                    break
            # reached the next window, or the end of clip, +1
            window_end = pair_idx + 1
            # window valid
            if window_start >= 0 and end_exist:
                window = [pair_idx_offset + window_start, pair_idx_offset + window_end]
            # window invalid: no starting point found, or last frame not available
            else:
                window = [pair_idx_offset, pair_idx_offset]
            windows.append(window)
        if gt:
            # ground-truth windows
            for im_idx in range(im_idx_max + 1):
                im_idx_start = max(0, im_idx - sliding_window + 1)
                # im_idx_end needs +1, because [start:end] does not include end
                im_idx_end = im_idx + 1
                window_start = -1
                for pair_idx, idx in enumerate(entry["im_idxes_gt"]):
                    # the first pair_idx meets the im_idx_start
                    if idx == im_idx_start and window_start < 0:
                        window_start = pair_idx
                    # reach the next window, -1
                    if idx >= im_idx_end:
                        pair_idx -= 1
                        break
                # reached the next window, or the end of clip, +1
                window_end = pair_idx + 1
                window = [pair_idx_offset + window_start, pair_idx_offset + window_end]
                windows_gt.append(window)
            for gt_name in interaction_gt_names:
                interaction_gt_dict[gt_name] = entry[gt_name]
            exist_mask = entry["exist_mask"]
            change_mask = entry["change_mask"]
    # window mode and anticipation mode, only window for the last frame
    else:
        window_len = len(entry["im_idxes"])
        end_exist = False
        window = [pair_idx_offset, pair_idx_offset + window_len]
        for idx in entry["im_idxes"]:
            # the last im_idx in this window exists
            if idx == im_idx_max:
                end_exist = True
                break
        if end_exist:
            windows.append(window)
        else:
            windows.append([pair_idx_offset, pair_idx_offset])

        if gt:
            window = [pair_idx_offset, pair_idx_offset + window_len_gt]
            windows_gt.append(window)
            last_frame_count = torch.sum(entry["im_idxes_gt"] == im_idx_max)
            for gt_name in interaction_gt_names:
                interaction_gt_dict[gt_name] = entry[gt_name][-last_frame_count:]
            exist_mask = entry["exist_mask"][-last_frame_count:]
            change_mask = entry["change_mask"][-last_frame_count:]
    if gt:
        return windows, exist_mask, change_mask, interaction_gt_dict, windows_gt
    else:
        return windows


def generate_sliding_window_mask(entry, window_list, windows_gt=None, split_window="person"):
    """
    Convert (window_start, window_end) to masks

    Args:
        entry (dict): entry, either single entry or concatenated entry_batch
        window_list (list): list of (window_start, window_end) idxes

    Returns:
        entry: the same entry dict as input, with window and output mask
    """
    # convert (window_start, window_end) to masks
    out_im_idxes = []
    out_im_idxes_gt = []
    all_windows = []
    all_windows_out = []
    for idx, window in enumerate(window_list):
        window_mask = torch.zeros_like(entry["im_idxes"]).bool()
        window_mask[window[0] : window[1]] = True
        # only append if window not empty
        if torch.any(window_mask):
            all_windows.append(window_mask)
            out_im_idx = entry["im_idxes"][window[1] - 1]
            out_im_idxes.append(out_im_idx)
            out_mask = entry["im_idxes"] == out_im_idx
            all_windows_out.append(out_mask)
        # additionally append gt output im_idx, for evaluation in Detection mode
        if windows_gt is not None:
            window_gt = windows_gt[idx]
            out_im_idx_gt = entry["im_idxes_gt"][window_gt[1] - 1]
            out_im_idxes_gt.append(out_im_idx_gt)

    if len(all_windows) > 0:
        if split_window == "person":
            # human_id window mask
            unique_human_ids = torch.unique(entry["pair_human_ids"])
            human_windows = []
            human_windows_out = []
            full_heatmaps = []
            for human_id in unique_human_ids:
                for window_mask, out_mask in zip(all_windows, all_windows_out):
                    human_window_mask = (entry["pair_human_ids"] == human_id) & window_mask
                    human_window_out_mask = human_window_mask & out_mask
                    if torch.any(human_window_out_mask):
                        human_windows.append(human_window_mask)
                        human_windows_out.append(human_window_out_mask)
                        full_heatmaps.append(entry["full_heatmaps"][human_id.item()])
            windows = torch.vstack(human_windows)
            windows_out = torch.vstack(human_windows_out)
            entry["full_heatmaps"] = full_heatmaps
        elif split_window == "pair":
            # pair id window mask
            pair_ids = entry["pair_human_ids"] * 100000 + entry["pair_object_ids"]
            unique_pair_ids = torch.unique(pair_ids)
            pair_windows = []
            pair_windows_out = []
            full_heatmaps = []
            for pair_id in unique_pair_ids:
                for window_mask, out_mask in zip(all_windows, all_windows_out):
                    pair_window_mask = (pair_ids == pair_id) & window_mask
                    pair_window_out_mask = pair_window_mask & out_mask
                    # TODO how to handle pair window with invisible object
                    if torch.any(pair_window_out_mask):
                        pair_windows.append(pair_window_mask)
                        pair_windows_out.append(pair_window_out_mask)
                        human_id = torch.div(pair_id, 100000, rounding_mode="floor")
                        full_heatmaps.append(entry["full_heatmaps"][human_id.item()])
            windows = torch.vstack(pair_windows)
            # should be only the last entry
            windows_out = torch.vstack(pair_windows_out)
            entry["full_heatmaps"] = full_heatmaps
        else:
            # not split sliding window
            windows = torch.vstack(all_windows)
            windows_out = torch.vstack(all_windows_out)
    else:
        windows = torch.Tensor(all_windows)
        windows_out = torch.Tensor(all_windows)

    return entry, windows, windows_out, out_im_idxes, out_im_idxes_gt


def concat_separated_head(pred, len_preds, loss_type_dict, class_idxes_dict, device, mlm_add_no_interaction=True):
    """
    Concatenate results from separated prediction heads to a single vector 
    """
    len_classes = 0
    for head_name in loss_type_dict.keys():
        if pred[head_name].shape[0] != len_preds:
            raise ValueError(f"{pred[head_name].shape}, {len_preds}")
        # assert pred[head_name].shape[0] == len_preds
        len_classes += len(class_idxes_dict[head_name])
    # ignore mlm no_interaction
    concatenated = torch.zeros((len_preds, len_classes), device=device)
    for head_name in loss_type_dict.keys():
        class_idxes = class_idxes_dict[head_name]
        if loss_type_dict[head_name] == "mlm" and mlm_add_no_interaction:
            concatenated[:, class_idxes] = pred[head_name][:, :-1]
        else:
            concatenated[:, class_idxes] = pred[head_name]
    return concatenated


def gt_mlm_to_bce(mlm_gt, mlm_add_no_interaction=True):
    """
    Convert gt interaction vector from MLM loss to BCE loss
    """
    if mlm_add_no_interaction:
        bce_gt = np.zeros((mlm_gt.shape[0], mlm_gt.shape[1] - 1))
        no_interaction = mlm_gt.shape[1] - 1
    else:
        bce_gt = np.zeros_like(mlm_gt)
        no_interaction = -9999
    for idx, anno in enumerate(mlm_gt):
        for label in anno:
            if label == -1:
                break
            if label != no_interaction:
                bce_gt[idx, label] = 1
    return bce_gt
