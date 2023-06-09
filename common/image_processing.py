#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import numpy as np
import torch
import cv2


def crop_roi_list(img: np.ndarray, bboxes: List[np.ndarray]):
    """
    Crop the image with given bounding boxes

    Args:
        img (np.ndarray): image, (C, H, W) or (H, W, C), RGB or BGR.
        bboxes (List[np.ndarray]): list of bounding boxes, [x1, y1, x2, y2].

    Returns:
        List[np.ndarray]: List of croped images, `None` means that bbox is not valid.
    """
    roi_list = []
    for bbox in bboxes:
        roi_list.append(crop_roi(img, bbox))
    return roi_list


def crop_roi(img: np.ndarray, bbox: np.ndarray):
    """
    Crop the image with a given bounding box

    Args:
        img (np.ndarray): image, (C, H, W) or (H, W, C), RGB or BGR
        bbox (np.ndarray): bounding box, [x1, y1, x2, y2]

    Returns:
        np.ndarray: croped image. `None` if bounding box not valid
    """
    img_shape = img.shape
    bbox = bbox.astype(np.int32)
    # (C, H, W)
    if img_shape[0] == 3:
        if np.all(bbox >= 0) and bbox[3] <= img_shape[1] and bbox[2] <= img_shape[2]:
            roi = img[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]
        else:
            roi = None
    # (H, W, C)
    else:
        if np.all(bbox >= 0) and bbox[3] <= img_shape[0] and bbox[2] <= img_shape[1]:
            roi = img[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
        else:
            roi = None
    return roi


def mask_roi(img: np.ndarray, bbox: np.ndarray):
    """
    Create a 0/1 mask for the given bbox

    Args:
        img (np.ndarray): image.
        bbox (np.ndarray): bounding box, [x1, y1, x2, y2]

    Returns:
        np.ndarray: mask. `None` if bounding box not valid
    """
    img_shape = img.shape
    bbox = bbox.astype(np.int32)
    # (C, H, W)
    if img_shape[0] == 3:
        if np.all(bbox >= 0) and bbox[3] <= img_shape[1] and bbox[2] <= img_shape[2]:
            mask = np.zeros(img_shape[1:], dtype=np.float32)
            mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 1
        else:
            mask = None
    # (H, W, C)
    else:
        if np.all(bbox >= 0) and bbox[3] <= img_shape[0] and bbox[2] <= img_shape[1]:
            mask = np.zeros(img_shape[:2], dtype=np.float32)
            mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 1
        else:
            mask = None
    return mask


def union_roi(bboxes: torch.Tensor, pair: torch.Tensor, im_idx: torch.Tensor):
    """
    Combine two bounding boxes to a union box

    Args:
        bboxes (torch.Tensor): bounding boxes tensor, shape (N, 5), each entry is in [img_idx, x1, y1, x2, y2].
        pair (torch.Tensor): bounding box pairs tensor, shape (M, 2), each entry is in [bbox1_idx, bbox2_idx].
        im_idx (torch.Tensor): image index tensor, shape (M, 1), image index of each bbox pair.

    Returns:
        [type]: [description]
    """
    union_boxes = torch.cat(
        (
            im_idx[:, None],
            torch.min(bboxes[:, 1:3][pair[:, 0]], bboxes[:, 1:3][pair[:, 1]]),
            torch.max(bboxes[:, 3:5][pair[:, 0]], bboxes[:, 3:5][pair[:, 1]]),
        ),
        1,
    )
    return union_boxes


def mask_union_roi(box_pairs, pooling_size=27):
    """
    Create a 0/1 mask for the given bounding box pairs
    Adapted from STTran/lib/draw_rectangles/draw_rectangles.pyx

    Args:
        boxes: (N, 8) ndarray of float. everything has arbitrary ratios
        pooling_size:

    Returns:
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    def minmax(x):
        return min(max(x, 0), 1)

    N = box_pairs.shape[0]

    uboxes = np.zeros((N, 2, pooling_size, pooling_size))

    for n in range(N):
        x1_union = min(box_pairs[n, 0], box_pairs[n, 4])
        y1_union = min(box_pairs[n, 1], box_pairs[n, 5])
        x2_union = max(box_pairs[n, 2], box_pairs[n, 6])
        y2_union = max(box_pairs[n, 3], box_pairs[n, 7])

        w = x2_union - x1_union
        h = y2_union - y1_union

        for i in range(2):
            # Now everything is in the range [0, pooling_size].
            x1_box = (box_pairs[n, 0 + 4 * i] - x1_union) * pooling_size / w
            y1_box = (box_pairs[n, 1 + 4 * i] - y1_union) * pooling_size / h
            x2_box = (box_pairs[n, 2 + 4 * i] - x1_union) * pooling_size / w
            y2_box = (box_pairs[n, 3 + 4 * i] - y1_union) * pooling_size / h
            # print("{:.3f}, {:.3f}, {:.3f}, {:.3f}".format(x1_box, y1_box, x2_box, y2_box))
            for j in range(pooling_size):
                y_contrib = minmax(j + 1 - y1_box) * minmax(y2_box - j)
                for k in range(pooling_size):
                    x_contrib = minmax(k + 1 - x1_box) * minmax(x2_box - k)
                    # print("j {} yc {} k {} xc {}".format(j, y_contrib, k, x_contrib))
                    uboxes[n, i, j, k] = x_contrib * y_contrib
    return uboxes


def mask_heatmap_roi(heatmap, bbox, original_shape, pooling_size=27):
    # upscale heatmap to original_shape
    width, height = original_shape[1], original_shape[0]
    heatmap_up = cv2.resize(heatmap, dsize=(width, height))
    # crop roi
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.clone().long()
    # handle ugly shape
    if bbox[3] <= bbox[1]:
        xmax = bbox[1] + 1
    else:
        xmax = bbox[3]
    if bbox[2] <= bbox[0]:
        ymax = bbox[0] + 1
    else:
        ymax = bbox[2]
    heatmap_roi = heatmap_up[bbox[1] : xmax, bbox[0] : ymax]
    # handle more ugly shape
    try:
        # down/up-scale heatmap_roi to 27x27
        heatmap_roi = cv2.resize(heatmap_roi, dsize=(pooling_size, pooling_size))
    except:
        print("bbox shape ugly", heatmap_up.shape, bbox)
        heatmap_roi = np.zeros((pooling_size, pooling_size))
    return heatmap_roi


def convert_annotation_frame_to_video(bboxes, ids, labels, confidences):
    bboxes_video = []
    ids_video = []
    labels_video = []
    confidences_video = []
    for im_idx, (bboxes_frame, ids_frame, labels_frame, confidences_frame) in enumerate(
        zip(bboxes, ids, labels, confidences)
    ):
        for bbox in bboxes_frame:
            bboxes_video.append([im_idx, *bbox])
        ids_video += ids_frame
        labels_video += labels_frame
        confidences_video += confidences_frame
    return bboxes_video, ids_video, labels_video, confidences_video
