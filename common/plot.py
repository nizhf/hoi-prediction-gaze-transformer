#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import enum
from typing import List
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2


# from yolov5.utils.plots
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb("#" + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def draw_bboxes(
    img: np.ndarray,
    bboxes: List[np.ndarray],
    ids: List[int],
    labels: List[int],
    names: List[str],
    confidences: List[float],
):
    """
    Draw bounding boxes in the given image. Will change the original image, so make `copy()` if necessary.

    Args:
        img (np.ndarray): Image in shape (H, W, C), BGR.
        bboxes (List[np.ndarray]): position of bounding boxes (x1, y1, x2, y2).
        ids (List[int]): object id.
        labels (List[int]): object label, e.g. (1, 19).
        names (List[str]): object name, e.g. (person, horse).
        confidences (List[float]): confidence scores.

    Returns:
        np.ndarray: image with bounding boxes and annotations, (H, W, C), BGR
    """
    for bbox, obj_id, label, name, confidence in zip(bboxes, ids, labels, names, confidences):
        img = draw_bbox(img, bbox, obj_id, label, name, confidence)
    return img


def draw_bbox(
    img: np.ndarray,
    bbox: np.ndarray,
    obj_id: int = None,
    label: int = None,
    name: str = None,
    confidence: float = None,
    line_width: int = 2,
    color=None,
    txt_color=(255, 255, 255),
):
    """
    Draw one bounding box. Will change the original image, so make `copy()` if necessary.
    Based on yolov5.utils.plots.Annotator.

    Args:
        img (np.ndarray): Image in shape (H, W, C), BGR.
        bbox (np.ndarray): position of bounding box (x1, y1, x2, y2).
        obj_id (int, optional): object id. Defaults to None.
        label (int, optional): object label, e.g. `1`. Defaults to None.
        name (str, optional): object name, e.g. `person`. Defaults to None.
        confidence (float, optional): confidence score. Defaults to None.
        line_width (int, optional): line width of bounding box. Defaults to 2.
        color ([type], optional): color of bounding box. Defaults to None.
        txt_color (tuple, optional): color of label text. Defaults to (255, 255, 255).

    Returns:
        np.ndarray: image with bounding box and annotation, (H, W, C), BGR
    """
    # not a valid bbox, return original image
    if len(bbox) < 4:
        warnings.warn(f"bbox {bbox} not valid")
        return img
    if color is None:
        if label is not None:
            color = colors(label, True)
        else:
            color = colors(np.random.randint(20), True)
    # draw object box
    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
    # generate label text
    label_text = ""
    if obj_id is not None:
        label_text += f"{obj_id} "
    if name is not None:
        label_text += f"{name} "
    if confidence is not None:
        label_text += f"{confidence:.2f}"
    # draw text box
    if len(label_text) > 0:
        # font thickness
        tf = max(line_width - 1, 1)
        # text width, height
        w, h = cv2.getTextSize(label_text, 0, fontScale=line_width / 3, thickness=tf)[0]
        # label fits outside box
        outside = p1[1] - h - 3 >= 0
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # filled label text area
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
        # label text
        cv2.putText(
            img,
            label_text,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            line_width / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return img


def draw_gaze_arrow(
    img,
    head_bbox,
    heatmap,
    output_resolution=64,
    color=(0, 200, 0),
    linewidth=2,
):
    """
    Draw human gaze direction as arrow

    Args:
        img (np.ndarray): original image
        head_bbox (np.ndarray): bbox location of head
        heatmap (np.ndarray): gaze attention heatmap
        output_resolution (int, optional): heatmap resolution. Defaults to 64.
        color (tuple, optional): line color. Defaults to (0, 200, 0).
        linewidth (int, optional): line width. Defaults to 2.

    Returns:
        np.ndarray: image with gaze arrows
    """
    width, height = img.shape[1], img.shape[0]
    # find maximum in heatmap
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = idx[0], idx[1]
    # scale maximum point to original image size
    norm_x, norm_y = pred_x / output_resolution, pred_y / output_resolution
    center = (int(norm_x * width), int(norm_y * height))
    radius = int(height / 50.0)
    # draw circle on gaze target
    cv2.circle(img, center=center, radius=radius, color=color, thickness=linewidth)
    # draw line from bbox to gaze target
    pt1 = (int((head_bbox[0] + head_bbox[2]) / 2), int((head_bbox[1] + head_bbox[3]) / 2))
    cv2.line(img, pt1, center, color=color, thickness=linewidth)
    return img


def draw_heatmap_overlay(
    img,
    heatmap,
    heatmap_resized=None,
):
    """
    Draw heatmap overlay

    Args:
        img (np.ndarray): original image
        heatmap (np.ndarray): original heatmap of any shape, not normalized to [0, 255]
        heatmap_resized (np.ndarray, optional): modulated and resized heatmap. Defaults to None.

    Returns:
        np.ndarray: image with heatmap overlay
    """
    if heatmap_resized is None:
        width, height = img.shape[1], img.shape[0]
        heatmap_modulated = heatmap * 255
        heatmap_resized = cv2.resize(heatmap_modulated, (width, height))
        heatmap_resized = np.clip(heatmap_resized, 0, 255).astype(np.uint8)
        heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    img = img * 0.6 + heatmap_resized * 0.4
    img = img.astype(np.uint8)
    return img


def plot_image_grids(imgs, labels, BGR=True):
    """
    TODO 

    Args:
        imgs ([type]): [description]
        labels ([type]): [description]
        BGR (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    num_imgs = len(imgs)
    num_grids = int(np.ceil(np.sqrt(num_imgs)).item())
    fig = plt.figure()
    for idx, (img, label) in enumerate(zip(imgs, labels), start=1):
        if BGR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(num_grids, num_grids, idx)
        ax.imshow(img)
        ax.set_title(label)
    return fig
