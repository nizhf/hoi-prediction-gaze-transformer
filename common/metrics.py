#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from sklearn.metrics import hamming_loss, confusion_matrix

from common.metrics_utils import bbox_iou, bbox_iou_mat, generate_gt_triplets_dict, generate_triplets_scores


# Based on detectron2 pascal_voc_evaluation.py
def voc_ap(recall, precision, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_ap_all_class(
    tp,
    fp,
    scores,
    count_gt,
    triplet_classes_gt,
    rare_triplets,
    non_rare_triplets,
    logger,
    use_07_metric=False,
):
    # Compute average precision for each triplet class
    ap = {}
    rare_ap = {}
    non_rare_ap = {}
    max_recall = {}

    for triplet in triplet_classes_gt:
        count_triplet = count_gt[triplet]
        # no gt positive, skip
        if count_triplet == 0:
            continue
        tp_triplet = np.array(tp[triplet])
        fp_triplet = np.array(fp[triplet])
        # no true positives, skip
        if len(tp_triplet) == 0:
            ap[triplet] = 0
            max_recall[triplet] = 0
            if triplet in rare_triplets:
                rare_ap[triplet] = 0
            elif triplet in non_rare_triplets:
                non_rare_ap[triplet] = 0
            else:
                logger.warning(f"Triplet {triplet} not exist in rare and non-rare set.")
            continue
        # sort scores decreasingly
        scores_triplet = np.array(scores[triplet])
        sort_inds = np.argsort(-scores_triplet)
        tp_triplet = tp_triplet[sort_inds]
        fp_triplet = fp_triplet[sort_inds]
        tp_triplet_cum = np.cumsum(tp_triplet)
        fp_triplet_cum = np.cumsum(fp_triplet)
        # recall
        rec = tp_triplet_cum / count_triplet
        # precision
        prec = tp_triplet_cum / (tp_triplet_cum + fp_triplet_cum)
        # average precision, max recall
        ap[triplet] = voc_ap(rec, prec, use_07_metric)
        max_recall[triplet] = np.max(rec)
        if triplet in rare_triplets:
            rare_ap[triplet] = ap[triplet]
        elif triplet in non_rare_triplets:
            non_rare_ap[triplet] = ap[triplet]
        else:
            logger.warning(f"Triplet {triplet} not exist in rare and non-rare set.")

    return ap, rare_ap, non_rare_ap, max_recall


# adapted from QPIC hico_eval.py
def mean_average_precision_qpic(
    all_results,
    logger,
    rare_limit=25,
    iou_threshold=0.5,
    quiet=False,
    only_change=False,
):
    """
    mAP method from QPIC, support multi-label.
    Find matched gt triplet (all subj, interaction, obj match), disregard whether one human-object pair is consistently
    matched to a gt pair.
    """
    # Construct dicts of triplet class from gruond-truth
    logger.info("Process ground-truth triplets...")
    tp, fp, scores, count_gt, triplet_classes_gt, triplets_gt_all = generate_gt_triplets_dict(all_results, only_change)
    # get rare and non-rare triplets
    rare_triplets = []
    non_rare_triplets = []
    for triplet, count in count_gt.items():
        if count < rare_limit:
            rare_triplets.append(triplet)
        else:
            non_rare_triplets.append(triplet)
    # Construct tp fp
    logger.info("Process predictions...")
    for result, triplets_gt_frame in tqdm(zip(all_results, triplets_gt_all), total=len(all_results), disable=quiet):
        # nothing exists in the future, skip
        if len(triplets_gt_frame) == 0:
            continue
        pred_labels = result["pred_labels"]
        bboxes = result["bboxes"]
        confidences = result["confidences"]
        pair_idxes = result["pair_idxes"]
        interaction_distribution = result["interaction_distribution"]
        # ground-truth
        bboxes_gt = result["bboxes_gt"]
        labels_gt = result["labels_gt"]
        pair_idxes_gt = result["pair_idxes_gt"]
        if "exist_mask" in result:
            exist_mask = result["exist_mask"]
        else:
            exist_mask = None

        if "change_mask" in result and only_change:
            change_mask = result["change_mask"]
        else:
            change_mask = None

        # compute scores of human-object pairs in current frame
        triplets_scores = generate_triplets_scores(pair_idxes, confidences, interaction_distribution, multiply=True)

        # if len(bboxes_gt) > 0:
        # compute IoU, get matched pairs
        bbox_pred_gt_map, bbox_pred_gt_overlap = bbox_iou_mat(
            bboxes_gt, labels_gt, bboxes, pred_labels, iou_threshold=iou_threshold
        )
        for score, idx_pair, interaction_pred in triplets_scores:
            # predicted triplet label
            pair_idx_pred = pair_idxes[idx_pair]
            subj_idx_pred = pair_idx_pred[0]
            obj_idx_pred = pair_idx_pred[1]
            subj_label_pred = pred_labels[subj_idx_pred]  # person
            obj_label_pred = pred_labels[obj_idx_pred]

            triplet_class_pred = (subj_label_pred, interaction_pred, obj_label_pred)
            # not exist triplet class, ignore
            if triplet_class_pred not in triplet_classes_gt:
                continue

            is_matched = False
            max_overlap = 0
            max_triplet_gt_idx = -1
            matched_triplets_gt = set()
            # predicted subject and object are matched to any ground-truth objects
            if len(bbox_pred_gt_map) > 0 and subj_idx_pred in bbox_pred_gt_map and obj_idx_pred in bbox_pred_gt_map:
                # all possible subject matches
                match_subj_gt_idxes = bbox_pred_gt_map[subj_idx_pred]
                match_subj_overlaps = bbox_pred_gt_overlap[subj_idx_pred]
                # all possible object matches
                match_obj_gt_idxes = bbox_pred_gt_map[obj_idx_pred]
                match_obj_overlaps = bbox_pred_gt_overlap[obj_idx_pred]
                for idx_triplet_gt, (
                    subj_idx_gt,
                    interaction_gt,
                    obj_idx_gt,
                ) in enumerate(triplets_gt_frame):
                    if (
                        subj_idx_gt in match_subj_gt_idxes
                        and obj_idx_gt in match_obj_gt_idxes
                        and interaction_pred == interaction_gt
                    ):
                        is_matched = True
                        min_overlap_pair = min(
                            match_subj_overlaps[match_subj_gt_idxes.index(subj_idx_gt)],
                            match_obj_overlaps[match_obj_gt_idxes.index(obj_idx_gt)],
                        )
                        if min_overlap_pair > max_overlap:
                            max_overlap = min_overlap_pair
                            max_triplet_gt_idx = idx_triplet_gt
            # matched to a gt triplet, and this gt triplet is not matched before, tp
            if is_matched:
                if max_triplet_gt_idx not in matched_triplets_gt:
                    matched_triplets_gt.add(max_triplet_gt_idx)
                    # only need changed action, and not changed
                    if change_mask is not None and not change_mask[max_idx_pair_gt]:
                        continue
                # gt triplet already matched to another pred triplet, fp
                else:
                    is_matched = False
            else:
                # not matched, if the corresponding gt pair does not exist in the future, skip
                if max_triplet_gt_idx >= 0:
                    subj_idx_gt = triplets_gt_frame[max_triplet_gt_idx][0]
                    obj_idx_gt = triplets_gt_frame[max_triplet_gt_idx][2]
                    max_idx_pair_gt = pair_idxes_gt.index([subj_idx_gt, obj_idx_gt])
                    if exist_mask is not None and not exist_mask[max_idx_pair_gt]:
                        continue
                    # only need changed action, and not changed
                    if change_mask is not None and not change_mask[max_idx_pair_gt]:
                        continue

            if is_matched:
                tp[triplet_class_pred].append(1)
                fp[triplet_class_pred].append(0)
            else:
                tp[triplet_class_pred].append(0)
                fp[triplet_class_pred].append(1)
            scores[triplet_class_pred].append(score)

        # else:
        #     # no ground-truth bbox, all predictions are false positives
        #     # NOTE should not happen in VidHOI dataset
        #     for score, idx_pair, interaction_pred in triplets_scores:
        #         # predicted triplet label
        #         pair_idx_pred = pair_idxes[idx_pair]
        #         subj_idx_pred = pair_idx_pred[0]
        #         obj_idx_pred = pair_idx_pred[1]
        #         subj_label_pred = pred_labels[subj_idx_pred]  # person
        #         obj_label_pred = pred_labels[obj_idx_pred]
        #         triplet_class_pred = (subj_label_pred, interaction_pred, obj_label_pred)
        #         # not exist triplet class, ignore
        #         if triplet_class_pred not in triplet_classes_gt:
        #             continue
        #         tp[triplet_class_pred].append(0)
        #         fp[triplet_class_pred].append(1)
        #         scores[triplet_class_pred].append(score)

    # compute mAP
    ap, rare_ap, non_rare_ap, max_recall = compute_ap_all_class(
        tp,
        fp,
        scores,
        count_gt,
        triplet_classes_gt,
        rare_triplets,
        non_rare_triplets,
        logger,
        use_07_metric=True,
    )
    mean_ap = np.mean(list(ap.values()))
    mean_ap_rare = np.mean(list(rare_ap.values()))
    mean_ap_non_rare = np.mean(list(non_rare_ap.values()))
    mean_max_recall = np.mean(list(max_recall.values()))

    logger.info(
        f"QPIC mAP: {mean_ap}, rare: {mean_ap_rare}, non-rare: {mean_ap_non_rare}, mean max recall: {mean_max_recall}"
    )

    return ap, rare_ap, non_rare_ap, max_recall, count_gt


def human_centric_top_k_all(all_results, threshold, k, logger, iou_threshold=0.5, quiet=False, only_change=False):
    """
    Compute human-centric top-k metrics. For each human as subject, evaluate with his top-k triplets
    """
    if not isinstance(threshold, list):
        threshold = [threshold]
    if not isinstance(k, list):
        k = [k]

    recall_top_all = []
    precision_top_all = []
    accuracy_top_all = []
    f1_top_all = []

    for i in range(len(threshold)):
        recall_top_all.append([])
        precision_top_all.append([])
        accuracy_top_all.append([])
        f1_top_all.append([])

        for j in range(len(k)):
            recall_top_all[i].append([])
            precision_top_all[i].append([])
            accuracy_top_all[i].append([])
            f1_top_all[i].append([])

    interaction_len = len(all_results[0]["interactions_gt"][0])

    num_changed = 0
    num_exist = 0
    num_human = 0
    num_gt_human = 0

    if not quiet:
        logger.info("Process predictions...")

    for result in tqdm(all_results, disable=quiet):
        pred_labels = result["pred_labels"]
        bboxes = result["bboxes"]
        confidences = result["confidences"]
        pair_idxes = result["pair_idxes"]
        interaction_distribution = np.array(result["interaction_distribution"])
        # ground-truth
        bboxes_gt = result["bboxes_gt"]
        labels_gt = result["labels_gt"]
        pair_idxes_gt = result["pair_idxes_gt"]
        interaction_distribution_gt = result["interactions_gt"]
        if "exist_mask" in result:
            exist_mask = result["exist_mask"]
        else:
            exist_mask = None

        if "change_mask" in result and only_change:
            change_mask = result["change_mask"]
        else:
            change_mask = None

        human_idxes_gt_set = set()
        # get total number of human in gt
        for pair_idx_gt in pair_idxes_gt:
            human_idxes_gt_set.add(pair_idx_gt[0])
        human_count_gt = len(human_idxes_gt_set)
        num_gt_human += human_count_gt

        # no human-object pair detected
        if len(pair_idxes) == 0:
            for idx_thres in range(len(threshold)):
                for idx_k in range(len(k)):
                    # append value for each human
                    for _ in range(human_count_gt):
                        # recall 0
                        recall_top_all[idx_thres][idx_k].append(0)
                        # precision N/A
                        precision_top_all[idx_thres][idx_k].append(np.nan)
                        # accuracy 0
                        accuracy_top_all[idx_thres][idx_k].append(0)
                        # f1 0
                        f1_top_all[idx_thres][idx_k].append(0)
            continue

        # match detected pairs to ground truth pairs, greedy assignment
        triplet_scores = []
        exist_mask_pred = []
        change_mask_pred = []
        interaction_gts = []
        idx_pair_pred_gt_map = {}
        idx_pair_gt_matched = set()
        for idx_pair, pair_idx_pred in enumerate(pair_idxes):
            subj_idx_pred = pair_idx_pred[0]
            obj_idx_pred = pair_idx_pred[1]
            subj_label_pred = pred_labels[subj_idx_pred]  # person
            obj_label_pred = pred_labels[obj_idx_pred]
            subj_bbox_pred = bboxes[subj_idx_pred]
            obj_bbox_pred = bboxes[obj_idx_pred]
            # triplet score, multiply
            subj_confidence = confidences[subj_idx_pred]
            obj_confidence = confidences[obj_idx_pred]
            triplet_scores.append(interaction_distribution[idx_pair] * subj_confidence * obj_confidence)
            # match to gt pair
            max_idx_pair_gt = -1
            max_iou = 0
            # iterate over all gt pairs
            for idx_pair_gt, pair_idx_gt in enumerate(pair_idxes_gt):
                # gt triplet label
                subj_idx_gt = pair_idx_gt[0]
                subj_label_gt = labels_gt[subj_idx_gt]  # person
                obj_idx_gt = pair_idx_gt[1]
                obj_label_gt = labels_gt[obj_idx_gt]  # object
                # gt bbox
                subj_bbox_gt = bboxes_gt[subj_idx_gt]
                obj_bbox_gt = bboxes_gt[obj_idx_gt]
                subj_iou = bbox_iou(subj_bbox_gt, subj_bbox_pred)
                obj_iou = bbox_iou(obj_bbox_gt, obj_bbox_pred)
                # match: labels are same, both iou>=0.5
                if (
                    subj_label_gt == subj_label_pred
                    and obj_label_gt == obj_label_pred
                    and subj_iou >= iou_threshold
                    and obj_iou >= iou_threshold
                ):
                    # compare iou to previous candidates
                    min_iou_pair = min(subj_iou, obj_iou)
                    if min_iou_pair > max_iou:
                        max_iou = min_iou_pair
                        max_idx_pair_gt = idx_pair_gt
            # matched to a gt pair, first check exist in future and changed in future
            if max_idx_pair_gt >= 0:
                # append to change list
                if change_mask is not None:
                    change_mask_pred.append(change_mask[max_idx_pair_gt])
                else:
                    change_mask_pred.append(True)
                # not exist in future, append empty
                if exist_mask is not None and not exist_mask[max_idx_pair_gt]:
                    interaction_gts.append([])
                    exist_mask_pred.append(False)
                    # assign to gt pair
                    if max_idx_pair_gt not in idx_pair_gt_matched:
                        idx_pair_pred_gt_map[idx_pair] = max_idx_pair_gt
                        idx_pair_gt_matched.add(max_idx_pair_gt)
                else:
                    # gt pair not assigned
                    if max_idx_pair_gt not in idx_pair_gt_matched:
                        idx_pair_pred_gt_map[idx_pair] = max_idx_pair_gt
                        idx_pair_gt_matched.add(max_idx_pair_gt)
                        interaction_gts.append(interaction_distribution_gt[max_idx_pair_gt])
                        exist_mask_pred.append(True)
                    # gt pair already assigned
                    else:
                        interaction_gts.append([0.0] * interaction_len)
                        exist_mask_pred.append(True)
            # no match, all fp
            else:
                interaction_gts.append([0.0] * interaction_len)
                exist_mask_pred.append(True)
                change_mask_pred.append(True)

        # categorize detected pairs to different human ids {human_idx: [idx_pair belongs to this human]}
        human_idx_pair_dict = {}
        for idx_pair, pair_idx_pred in enumerate(pair_idxes):
            human_idx = pair_idx_pred[0]
            if human_idx not in human_idx_pair_dict:
                human_idx_pair_dict[human_idx] = []
            human_idx_pair_dict[human_idx].append(idx_pair)

        # concatenate triplet scores for each human for metrics computation
        gts_human_all = []
        distributions_human_all = []
        gt_counts = []
        for human_idx, idx_pair_pred_list in human_idx_pair_dict.items():
            num_human += 1
            triplet_scores_human = []
            gts_human = []
            distributions_human = []
            human_changed = False
            for idx_pair_pred in idx_pair_pred_list:
                # only append if exist in the future
                if exist_mask_pred[idx_pair_pred]:
                    triplet_scores_human.append(triplet_scores[idx_pair_pred])
                    gts_human.append(interaction_gts[idx_pair_pred])
                    distributions_human.append(interaction_distribution[idx_pair_pred])
                    if change_mask_pred[idx_pair_pred]:
                        human_changed = True
            # only append if any human-object pair for this human exist, otherwise skip
            if len(triplet_scores_human) > 0:
                num_exist += 1
                # check changed if needed
                if not human_changed and only_change:
                    continue
                num_changed += 1
                # concatenate all distributions for this human to one
                triplet_scores_human = np.hstack(triplet_scores_human)
                gts_human = np.hstack(gts_human)
                distributions_human = np.hstack(distributions_human)
                gt_counts.append(np.sum(gts_human))
                # sort triplet scores
                sorted_idx = np.argsort(-triplet_scores_human)
                gts_human = gts_human[sorted_idx]
                distributions_human = distributions_human[sorted_idx]
                gts_human_all.append(gts_human)
                distributions_human_all.append(distributions_human)

        for idx_thres, thres in enumerate(threshold):
            for idx_k, kk in enumerate(k):
                gts = []
                preds = []
                # keep top-k predictions for each human
                for gts_human, distributions_human in zip(gts_human_all, distributions_human_all):
                    gts.append(gts_human[:kk])
                    preds_human = distributions_human >= thres
                    preds.append(preds_human[:kk])
                recalls = multi_label_recall(gts, preds, gt_counts)
                precisions = multi_label_precision(gts, preds)
                accuracies = multi_label_accuracy(gts, preds, gt_counts)
                f1s = multi_label_f1(gts, preds, gt_counts)

                recall_top_all[idx_thres][idx_k].append(recalls)
                precision_top_all[idx_thres][idx_k].append(precisions)
                accuracy_top_all[idx_thres][idx_k].append(accuracies)
                f1_top_all[idx_thres][idx_k].append(f1s)
    if not quiet:
        logger.info(
            f"Totally {num_human} detected persons, {num_exist} exist, {num_changed} changed. {num_gt_human} gt persons"
        )
    return recall_top_all, precision_top_all, accuracy_top_all, f1_top_all


def sample_based_metrics_all(all_results, threshold, k, logger, iou_threshold=0.5, quiet=False, only_change=False):
    """
    Compute all sample based metrics, support evaluate with multiple thresholds
    """
    if not isinstance(threshold, list):
        threshold = [threshold]
    if not isinstance(k, list):
        k = [k]

    recall_k_all = []

    recall_all = []
    precision_all = []
    accuracy_all = []
    f1_all = []
    hamming_loss_all = []

    for i in range(len(threshold)):
        recall_all.append([])
        precision_all.append([])
        accuracy_all.append([])
        f1_all.append([])
        hamming_loss_all.append([])

        recall_k_all.append([])
        for j in range(len(k)):
            recall_k_all[i].append([])

    interaction_len = len(all_results[0]["interactions_gt"][0])

    if not quiet:
        logger.info("Process predictions...")
    for result in tqdm(all_results, disable=quiet):
        pred_labels = result["pred_labels"]
        bboxes = result["bboxes"]
        confidences = result["confidences"]
        pair_idxes = result["pair_idxes"]
        interaction_distribution = np.array(result["interaction_distribution"])
        # ground-truth
        bboxes_gt = result["bboxes_gt"]
        labels_gt = result["labels_gt"]
        pair_idxes_gt = result["pair_idxes_gt"]
        interaction_distribution_gt = result["interactions_gt"]
        if "exist_mask" in result:
            exist_mask = result["exist_mask"]
        else:
            exist_mask = None

        if "change_mask" in result and only_change:
            change_mask = result["change_mask"]
        else:
            change_mask = None

        # no human-object pair detected
        if len(pair_idxes) == 0:
            for idx_thres in range(len(threshold)):
                # recall 0
                recall_all[idx_thres].append(0)
                # precision N/A
                precision_all[idx_thres].append(np.nan)
                # accuracy 0
                accuracy_all[idx_thres].append(0)
                # f1 0
                f1_all[idx_thres].append(0)
                # hamming loss N/A
                hamming_loss_all[idx_thres].append(np.nan)
                # recall@k all 0
                for idx_k in range(len(k)):
                    recall_k_all[idx_thres][idx_k].append(0)
            continue

        # using threshold, get positive predictions
        interaction_preds = [interaction_distribution >= thres for thres in threshold]
        # triplet scores, for recall@k
        triplet_scores = []

        # match detected pairs to ground truth pairs, greedy assignment
        available_mask_pred = []
        interaction_gts = []
        idx_pair_pred_gt_map = {}
        idx_pair_gt_matched = set()
        for idx_pair, pair_idx_pred in enumerate(pair_idxes):
            subj_idx_pred = pair_idx_pred[0]
            obj_idx_pred = pair_idx_pred[1]
            subj_label_pred = pred_labels[subj_idx_pred]  # person
            obj_label_pred = pred_labels[obj_idx_pred]
            subj_bbox_pred = bboxes[subj_idx_pred]
            obj_bbox_pred = bboxes[obj_idx_pred]
            # triplet score, multiply
            subj_confidence = confidences[subj_idx_pred]
            obj_confidence = confidences[obj_idx_pred]
            triplet_scores.append(interaction_distribution[idx_pair] * subj_confidence * obj_confidence)
            # match to gt pair
            max_idx_pair_gt = -1
            max_iou = 0
            # iterate over all gt pairs
            for idx_pair_gt, pair_idx_gt in enumerate(pair_idxes_gt):
                # gt triplet label
                subj_idx_gt = pair_idx_gt[0]
                subj_label_gt = labels_gt[subj_idx_gt]  # person
                obj_idx_gt = pair_idx_gt[1]
                obj_label_gt = labels_gt[obj_idx_gt]  # object
                # gt bbox
                subj_bbox_gt = bboxes_gt[subj_idx_gt]
                obj_bbox_gt = bboxes_gt[obj_idx_gt]
                subj_iou = bbox_iou(subj_bbox_gt, subj_bbox_pred)
                obj_iou = bbox_iou(obj_bbox_gt, obj_bbox_pred)
                # match: labels are same, both iou>=0.5
                if (
                    subj_label_gt == subj_label_pred
                    and obj_label_gt == obj_label_pred
                    and subj_iou >= iou_threshold
                    and obj_iou >= iou_threshold
                ):
                    # compare iou to previous candidates
                    min_iou_pair = min(subj_iou, obj_iou)
                    if min_iou_pair > max_iou:
                        max_iou = min_iou_pair
                        max_idx_pair_gt = idx_pair_gt
            # matched to a gt pair, first check exist in future and changed in future
            if max_idx_pair_gt >= 0:
                # not exist or not changed in future, append empty
                if (exist_mask is not None and not exist_mask[max_idx_pair_gt]) or (
                    change_mask is not None and not change_mask[max_idx_pair_gt]
                ):
                    interaction_gts.append([])
                    available_mask_pred.append(False)
                    # assign to gt pair
                    if max_idx_pair_gt not in idx_pair_gt_matched:
                        idx_pair_pred_gt_map[idx_pair] = max_idx_pair_gt
                        idx_pair_gt_matched.add(max_idx_pair_gt)
                else:
                    # gt pair not assigned
                    if max_idx_pair_gt not in idx_pair_gt_matched:
                        idx_pair_pred_gt_map[idx_pair] = max_idx_pair_gt
                        idx_pair_gt_matched.add(max_idx_pair_gt)
                        interaction_gts.append(interaction_distribution_gt[max_idx_pair_gt])
                        available_mask_pred.append(True)
                    # gt pair already assigned
                    else:
                        interaction_gts.append([0.0] * interaction_len)
                        available_mask_pred.append(True)
            # no match, all fp
            else:
                interaction_gts.append([0.0] * interaction_len)
                available_mask_pred.append(True)

        # process prediction exist mask
        interaction_preds = [inter[available_mask_pred, :] for inter in interaction_preds]
        interaction_gts = [inter for (inter, exist) in zip(interaction_gts, available_mask_pred) if exist]
        triplet_scores = [sc for (sc, exist) in zip(triplet_scores, available_mask_pred) if exist]
        # to numpy array
        interaction_gts = np.array(interaction_gts)
        triplet_scores = np.array(triplet_scores)
        # compute everything
        for idx_thres in range(len(threshold)):
            recalls = multi_label_recall(interaction_gts, interaction_preds[idx_thres])
            precisions = multi_label_precision(interaction_gts, interaction_preds[idx_thres])
            accuracies = multi_label_accuracy(interaction_gts, interaction_preds[idx_thres])
            f1s = multi_label_f1(interaction_gts, interaction_preds[idx_thres])
            ham_losses = hamming_loss_batch(interaction_gts, interaction_preds[idx_thres])

            recall_all[idx_thres].append(recalls)
            precision_all[idx_thres].append(precisions)
            accuracy_all[idx_thres].append(accuracies)
            f1_all[idx_thres].append(f1s)
            hamming_loss_all[idx_thres].append(ham_losses)

        # recall@k
        for idx_thres in range(len(threshold)):
            sorted_idx = np.argsort(-triplet_scores.flatten())
            rec_k = recall_k(
                interaction_gts.flatten(),
                interaction_preds[idx_thres].flatten(),
                sorted_idx,
                k,
            )
            if len(rec_k) > 0:
                for idx_k in range(len(k)):
                    recall_k_all[idx_thres][idx_k].append(rec_k[idx_k])

    return (
        recall_all,
        precision_all,
        accuracy_all,
        f1_all,
        hamming_loss_all,
        recall_k_all,
    )


def recall_k(gts, preds, sorted_idx, k):
    if not isinstance(k, list):
        k = [k]
    rec_k = []

    sorted_gts = gts[sorted_idx]
    sorted_preds = preds[sorted_idx]
    count_gt = np.sum(sorted_gts)

    # if no gt, cannot compute recall
    if count_gt > 0:
        for kk in k:
            top_preds = sorted_preds[:kk]
            top_gts = sorted_gts[:kk]
            tp = confusion_matrix(top_gts, top_preds, labels=[0, 1])[1, 1]
            rec_k.append(tp / count_gt)

    return rec_k


def multi_label_recall(gts, preds, gt_counts=[]):
    recalls = []
    for idx, (gt, pred) in enumerate(zip(gts, preds)):
        if len(gt_counts) > 0:
            gt_count = gt_counts[idx]
        else:
            gt_count = np.sum(gt)
        # only handle the valid case (# positive gt > 0)
        gt = gt.astype(bool)
        pred = pred.astype(bool)
        intersection = gt * pred
        # |TP| / (|TP| + |FN|)
        if gt_count > 0:
            rec = np.sum(intersection) / gt_count
            recalls.append(rec)
    # no gt, cannot judge
    if len(recalls) == 0:
        return np.nan
    return np.mean(recalls)


def multi_label_precision(gts, preds):
    precisions = []
    for gt, pred in zip(gts, preds):
        # only handle the valid case (# positive prediction > 0)
        gt = gt.astype(bool)
        pred = pred.astype(bool)
        intersection = gt * pred
        # |TP| / (|TP| + |FP|)
        if np.sum(pred) > 0:
            prec = np.sum(intersection) / np.sum(pred)
            precisions.append(prec)
    # no prediction
    if len(precisions) == 0:
        # but gt, precision 0
        if np.sum(gts[:]) > 0:
            return 0
        # also no gt, cannot judge
        return np.nan
    return np.mean(precisions)


def multi_label_accuracy(gts, preds, gt_counts=[]):
    accuracies = []
    for idx, (gt, pred) in enumerate(zip(gts, preds)):
        # only handle the valid case (# positive gt > 0)
        gt = gt.astype(bool)
        pred = pred.astype(bool)
        intersection = gt * pred
        if len(gt_counts) > 0:
            gt_count = gt_counts[idx]
        else:
            gt_count = np.sum(gt)
        count_union = np.sum(pred) + gt_count - np.sum(intersection)
        # |TP| / |union|
        if gt_count or np.sum(pred) > 0:
            acc = np.sum(intersection) / count_union
            accuracies.append(acc)
    # no detection, no prediction, cannot judge
    if len(accuracies) == 0:
        return np.nan
    return np.mean(accuracies)


def multi_label_f1(gts, preds, gt_counts=[]):
    f1s = []
    for idx, (gt, pred) in enumerate(zip(gts, preds)):
        if len(gt_counts) > 0:
            gt_count = gt_counts[idx]
        else:
            gt_count = np.sum(gt)
        # only handle the valid case (# positive gt > 0)
        gt = gt.astype(bool)
        pred = pred.astype(bool)
        intersection = gt * pred
        if gt_count > 0 or np.sum(pred) > 0:
            f1 = 2 * np.sum(intersection) / (gt_count + np.sum(pred))
            f1s.append(f1)
    # no detection, no prediction, cannot judge
    if len(f1s) == 0:
        return np.nan
    return np.mean(f1s)


def hamming_loss_batch(gts, preds):
    ham_losses = []
    for gt, pred in zip(gts, preds):
        ham_loss = hamming_loss(gt, pred)
        ham_losses.append(ham_loss)
    if len(ham_losses) == 0:
        return np.nan
    return np.mean(ham_losses)
