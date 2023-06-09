#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math


def bbox_iou(bbox1, bbox2):
    """
    Bounding box intersection over union, boxes should be [x1, y1, x2, y2] format
    """
    # compute intersection area
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    intersection_area = max(x2 - x1, 0) * max(y2 - y1, 0)
    if intersection_area == 0:
        return 0

    # compute bbox1 area
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    # compute bbox2 area
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    # compute union area
    union_area = bbox1_area + bbox2_area - intersection_area
    # intersection over union
    iou = intersection_area / union_area

    return iou


def bbox_iou_mat(bboxes_gt, labels_gt, bboxes, pred_labels, iou_threshold=0.5, check_label=True):
    """
    Compute IoU for a list of predictions
    Match predictions to gt
    """
    # no available bboxes
    if len(bboxes_gt) == 0 or len(bboxes) == 0:
        return {}, {}
    iou_mat = np.zeros((len(bboxes_gt), len(bboxes)))
    for i1, (bbox1, label1) in enumerate(zip(bboxes_gt, labels_gt)):
        for i2, (bbox2, label2) in enumerate(zip(bboxes, pred_labels)):
            iou_mat[i1, i2] = bbox_iou(bbox1, bbox2)
            # additionally check labels are the same
            if check_label and label1 != label2:
                iou_mat[i1, i2] = 0
    # possible matches
    iou_mat_match = np.zeros_like(iou_mat)
    iou_mat_match[iou_mat >= iou_threshold] = 1

    # ([idx_gt], [idx_pred])
    match_pairs = np.nonzero(iou_mat_match)
    # {obj_idx_pred: [list of matched obj_idx_gt]}
    match_bboxes_dict = {}
    match_bboxes_overlaps = {}
    # there are matched pairs
    if len(match_pairs[1]) > 0:
        for idx_gt, idx_pred in zip(match_pairs[0], match_pairs[1]):
            if idx_pred not in match_bboxes_dict.keys():
                match_bboxes_dict[idx_pred] = []
                match_bboxes_overlaps[idx_pred] = []
            match_bboxes_dict[idx_pred].append(idx_gt)
            match_bboxes_overlaps[idx_pred].append(iou_mat[idx_gt, idx_pred])

    return match_bboxes_dict, match_bboxes_overlaps


def generate_gt_triplets_dict(all_results, only_change=False):
    """
    Count positive triplets in all ground-truth annotations
    """
    tp, fp, scores, count_gt = {}, {}, {}, {}
    triplet_classes_gt = []
    # for each frame: [list of positive gt (subj_idx, interaction, obj_idx)]
    triplets_idx_gt_all = []
    for result in all_results:
        triplets_idx_gt_frame = []
        labels_gt = result["labels_gt"]
        pair_idxes_gt = result["pair_idxes_gt"]
        interaction_distribution_gt = result["interactions_gt"]
        # anticipation, check pair exists in the future
        if "exist_mask" in result:
            exist_mask = result["exist_mask"]
        else:
            exist_mask = None

        if "change_mask" in result and only_change:
            change_mask = result["change_mask"]
        else:
            change_mask = None

        for idx_pair_gt, (pair_idx_gt, interactions_gt) in enumerate(zip(pair_idxes_gt, interaction_distribution_gt)):
            # gt_pair not exist in future, skip
            if exist_mask is not None and not exist_mask[idx_pair_gt]:
                continue
            # only need changed action, and not changed
            if change_mask is not None and not change_mask[idx_pair_gt]:
                continue
            for interaction_gt, interaction_score_gt in enumerate(interactions_gt):
                # negative gt
                if interaction_score_gt < 1:
                    continue
                # positive gt
                subj_idx_gt = pair_idx_gt[0]
                subj_label_gt = labels_gt[subj_idx_gt]  # person
                obj_idx_gt = pair_idx_gt[1]
                obj_label_gt = labels_gt[obj_idx_gt]  # object
                triplet_class_gt = (subj_label_gt, interaction_gt, obj_label_gt)
                # append to frame gt
                triplets_idx_gt_frame.append((subj_idx_gt, interaction_gt, obj_idx_gt))
                # append to triplet classes
                if triplet_class_gt not in triplet_classes_gt:
                    triplet_classes_gt.append(triplet_class_gt)
                    tp[triplet_class_gt] = []
                    fp[triplet_class_gt] = []
                    scores[triplet_class_gt] = []
                    count_gt[triplet_class_gt] = 0
                # count + 1
                count_gt[triplet_class_gt] += 1

        triplets_idx_gt_all.append(triplets_idx_gt_frame)
    return tp, fp, scores, count_gt, triplet_classes_gt, triplets_idx_gt_all


def generate_triplets_scores(pair_idxes, confidences, interaction_distribution, multiply=False, top_k=100, thres=0.0):
    """
    Get top k triplet predictions
    """
    # compute scores of human-object pairs in current frame
    triplets_scores = []
    # [score, idx_pair, idx_interaction]
    for idx_pair, (pair_idx_pred, interactions_pred) in enumerate(zip(pair_idxes, interaction_distribution)):
        for interaction_pred, interaction_score in enumerate(interactions_pred):
            subj_idx_pred = pair_idx_pred[0]
            obj_idx_pred = pair_idx_pred[1]
            subj_confidence = confidences[subj_idx_pred]
            obj_confidence = confidences[obj_idx_pred]
            if multiply:
                # interaction_score * bbox_score
                # NOTE in QPIC, here only obj_confidence
                score = interaction_score * subj_confidence * obj_confidence
            else:
                score = math.log(interaction_score if interaction_score > 0 else 1e-300)
                score += math.log(subj_confidence if subj_confidence > 0 else 1e-300)
                score += math.log(obj_confidence if obj_confidence > 0 else 1e-300)
            # only keep triplet with score >= threshold
            if score >= thres:
                triplets_scores.append((score, idx_pair, interaction_pred))
    # keep the top k scores
    triplets_scores.sort(reverse=True)
    triplets_scores = triplets_scores[:top_k]

    return triplets_scores
