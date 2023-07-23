#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import linear_sum_assignment


def assign_human_head_video(
    frames,
    original_frames,
    bboxes,
    ids,
    labels,
    head_detector,
    iou_thres,
    device,
    method="hungarian",
    human_label=0,
    verbose=False,
):
    head_bboxes_video = []
    for im_idx, (frame, frame0) in enumerate(zip(frames, original_frames)):
        frame = frame.to(device)
        head_bboxes, _, _, head_scores, _ = head_detector.detect_one(
            frame, frame0, draw=False, save_path=None, cap=None, fourcc=None, include_person=False
        )
        if verbose:
            print(f"im_idx: {im_idx}")
        # no object detected, empty dict
        if len(bboxes) == 0:
            head_bboxes_video.append({})
            if verbose:
                print("No object detected")
            continue
        bboxes = np.array(bboxes)
        ids = np.array(ids)
        labels = np.array(labels)
        human_idxes = (bboxes[:, 0] == im_idx) & (labels == human_label)
        # no human bbox in this frame (detection mode), append empty dict
        if not np.any(human_idxes):
            head_bboxes_video.append({})
            if verbose:
                print("No human detected")
            continue
        human_bboxes = bboxes[human_idxes][:, 1:]
        human_ids = ids[human_idxes]
        association_frame = assign_human_head_frame(
            head_bboxes, head_scores, human_bboxes, human_ids, iou_thres, method, verbose
        )
        head_bboxes_video.append(association_frame)

    return head_bboxes_video


def assign_human_head_frame(
    head_bboxes,
    head_scores,
    human_bboxes,
    human_ids,
    iou_thres,
    method="hungarian",
    verbose=False,
):
    # {person id: head bbox} of this frame
    head_bbox_dict = {}
    # no human detected, return empty dict
    if len(human_bboxes) == 0:
        return head_bbox_dict
    # no head detected, return empty assignment
    if len(head_bboxes) == 0:
        head_bbox_dict = {human_id: [] for human_id in human_ids}
        return head_bbox_dict
    # calculate head bbox area (head_len)
    head_areas = (head_bboxes[:, 2] - head_bboxes[:, 0]) * (head_bboxes[:, 3] - head_bboxes[:, 1])
    # prepare bbox matrix
    bboxes1 = np.repeat(head_bboxes[:, None, :], len(human_bboxes), axis=1)
    bboxes2 = np.repeat(human_bboxes[None, :, :], len(head_bboxes), axis=0)
    # calculate intersection area (head_len, human_len)
    x1 = np.maximum(bboxes1[:, :, 0], bboxes2[:, :, 0])
    y1 = np.maximum(bboxes1[:, :, 1], bboxes2[:, :, 1])
    x2 = np.minimum(bboxes1[:, :, 2], bboxes2[:, :, 2])
    y2 = np.minimum(bboxes1[:, :, 3], bboxes2[:, :, 3])
    intersection_areas = (x2 - x1) * (y2 - y1)
    intersection_areas[x2 - x1 <= 0] = 0
    intersection_areas[y2 - y1 <= 0] = 0
    # ratio of intersection area to the head (head_len, human_len)
    intersection_ratios = intersection_areas / head_areas[:, None]
    # get valid heads for each person (intersection ratio >= threshold)
    valid_heads = intersection_ratios >= iou_thres
    # calculate border distance (head_len, human_len)
    border_distances = np.min(np.abs(bboxes1 - bboxes2), axis=2)
    width = bboxes2[:, :, 2] - bboxes2[:, :, 0]
    height = bboxes2[:, :, 3] - bboxes2[:, :, 1]
    border_distance_ratios = border_distances / np.minimum(width, height)
    # Hungarian algorithm
    head_scores = np.repeat(head_scores[:, None], len(human_bboxes), axis=1)
    if verbose:
        print("=" * 40)
        print(f"head \n{head_bboxes}")
        print(f"human \n{human_bboxes}")
        print(f"score \n{head_scores}")
        print(f"dis \n{border_distance_ratios}")
        print(f"IoH \n{intersection_areas}")
        print(f"IoH ratio \n{intersection_ratios}")
        print(valid_heads)
    if method == "hungarian":
        head_bbox_dict = hungarian_assignment(
            human_ids, head_bboxes, head_scores, border_distance_ratios, valid_heads, verbose
        )
    # greedy algorithm
    else:
        head_bbox_dict = greedy_assignment(human_ids, head_bboxes, border_distances, valid_heads)
    return head_bbox_dict


def hungarian_assignment(
    human_ids,
    head_bboxes,
    head_scores,
    border_distance_ratios,
    valid_heads,
    verbose=False,
):
    head_bbox_dict = {}
    head_included = np.arange(len(head_bboxes))
    human_included = np.arange(len(human_ids))
    # exclude invalid heads (not covered by any human bbox)
    for head_idx in head_included:
        if not np.any(valid_heads[head_idx, :]):
            head_included[head_idx] = -1
    head_included = np.delete(head_included, head_included == -1)
    # exclude human with no head bbox
    for human_idx in human_included:
        if not np.any(valid_heads[:, human_idx]):
            human_included[human_idx] = -1
            head_bbox_dict[human_ids[human_idx]] = []
    human_included = np.delete(human_included, human_included == -1)
    # all heads or persons excluded, return
    if len(head_included) == 0 or len(human_included) == 0:
        return head_bbox_dict

    # the remaining mask and scores
    valid_heads = valid_heads[head_included, :][:, human_included]
    head_scores_inverse = 1 - head_scores[head_included, :][:, human_included]
    border_distance_ratios = border_distance_ratios[head_included, :][:, human_included]
    cost_matrix = head_scores_inverse + border_distance_ratios * 4

    # scipy.linear_sum_assignment, first try use np.inf
    cost_matrix[~valid_heads] = np.inf
    if verbose:
        print(head_included)
        print(human_included)
        print(cost_matrix)
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError:
        # infeasible with np.inf, instead using very large number, and post-processing
        # print("infeasible matrix, current cost matrix:")
        # print(cost_matrix)
        cost_matrix[~valid_heads] = 99999999
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for i, j in zip(row_ind, col_ind):
        head_idx = head_included[i]
        human_idx = human_included[j]
        if valid_heads[i, j]:
            head_bbox_dict[human_ids[human_idx]] = head_bboxes[head_idx]

    # check all human_id assigned
    for human_id in human_ids:
        if human_id not in head_bbox_dict:
            head_bbox_dict[human_id] = []

    return head_bbox_dict


def greedy_assignment(human_ids, head_bboxes, border_distances, valid_heads):
    head_bbox_dict = {}
    # head bboxes that are already matched to a person id
    matched_heads = []
    # for each person bbox, find a head bbox
    for human_idx, human_id in enumerate(human_ids):
        best_head_idx = -1
        best_border_distance = np.inf
        for head_idx in range(len(head_bboxes)):
            # head already matched, skip
            if head_idx in matched_heads:
                continue
            # possible head person pair, calculate distance of head border to the nearest person border
            if valid_heads[head_idx, human_idx]:
                # nearest person bbox
                if border_distances[head_idx, human_idx] < best_border_distance:
                    best_head_idx = head_idx
                    best_border_distance = border_distances[head_idx, human_idx]
        # matching head found, assign head bbox to person id
        if best_head_idx >= 0:
            head_bbox_dict[human_id] = head_bboxes[best_head_idx]
            matched_heads.append(best_head_idx)
        # no matching head found, empty bbox
        else:
            head_bbox_dict[human_id] = []
    return head_bbox_dict


def calculate_intersection_over_head(head_bbox, human_bbox):
    x1 = max(head_bbox[0], human_bbox[0])
    y1 = max(head_bbox[1], human_bbox[1])
    x2 = min(head_bbox[2], human_bbox[2])
    y2 = min(head_bbox[3], human_bbox[3])
    # calculate intersection area
    if x2 > x1 and y2 > y1:
        union_area = (x2 - x1) * (y2 - y1)
    else:
        union_area = 0
    return union_area
