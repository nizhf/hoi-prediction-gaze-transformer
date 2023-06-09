#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Tuple
from pathlib import Path
import numpy as np
import cv2
import torch

from modules.gaze_following.detecting_attended_visual_targets.model import (
    ModelSpatioTemporal,
)
from common.config_parser import get_config
from common.image_processing import crop_roi, mask_roi
from common.plot import draw_bbox, draw_gaze_arrow, draw_heatmap_overlay, colors
from common.download import download_file_from_dropbox
from common.transforms import DetectingAttendedTransform
from .detecting_attended import DetectingAttendedVisualTargets


class GazeFollowing:
    def __init__(
        self,
        weight_path: str = "weights/detecting_attended/model_videoatttarget.pt",
        config_path: str = "configs/gaze_following.yaml",
        device: str = "cuda:0",
    ):
        print('Initializing Gaze Following "Detecting Attended Visual Targets in Video" Module...')
        self.gaze_following_cfg = get_config(config_path)
        self.device = device
        self.input_resolution = self.gaze_following_cfg["DETECTING_ATTENDED"]["INPUT_RESOLUTION"]
        self.output_resolution = self.gaze_following_cfg["DETECTING_ATTENDED"]["OUTPUT_RESOLUTION"]
        self.out_threshold = self.gaze_following_cfg["DETECTING_ATTENDED"]["OUT_THRESHOLD"]
        self.gaze_follower = DetectingAttendedVisualTargets(
            weight_path=weight_path,
            device=self.device,
            input_resolution=self.input_resolution,
            output_resolution=self.output_resolution,
            num_lstm_layers=self.gaze_following_cfg["DETECTING_ATTENDED"]["NUM_LSTM_LAYERS"],
        )
        self.input_resolution = self.gaze_following_cfg["DETECTING_ATTENDED"]["INPUT_RESOLUTION"]
        self.image_transform = DetectingAttendedTransform(self.input_resolution)
        print("Gaze Following Module Initialization Finished.")

    def detect_one(
        self,
        frame: np.ndarray,
        head_bbox: np.ndarray,
        hidden_state: Tuple[torch.Tensor] = None,
        draw: bool = True,
        frame_to_draw: np.ndarray = None,
        id: int = None,
        confidence: float = None,
        arrow: bool = False,
    ):
        """[summary]

        Args:
            frame (np.ndarray): scene image, (H, W, C), !!BGR!!
            head_bbox (np.ndarray): (x1, y1, x2, y2)
            hidden_state (Tuple[torch.Tensor]): hidden state of LSTM, if `None`, use zero state. Defaults to `None`.
            draw (bool): If True, draw the bounding boxes. Defaults to `True`.
            frame_to_draw (np.ndarray): If set, draw on this image instead of the original frame. Defaults to `None`.
            id (int): head id for plot. Defaults to `None`.
            confidence (float): head confidence for plot. Defaults to `None`.
            arrow (bool, optional): if True, draw the arrow, not the heatmap on top of the image. Defaults to `False`.
        """
        if draw:
            if frame_to_draw is not None:
                frame_original = frame_to_draw
            else:
                frame_original = frame.copy()
        original_shape = frame.shape

        # crop head images and head masks
        head_image = crop_roi(frame, head_bbox)
        head_mask = mask_roi(frame, head_bbox)

        # preprocessing
        # (B, 3, H, W)
        # frame = self.image_transform(frame).unsqueeze(0).to(self.device)
        frame = self.image_transform(frame).to(self.device)
        # (B, 3, H, W)
        # head_image = self.image_transform(head_image).unsqueeze(0).to(self.device)
        head_image = self.image_transform(head_image).to(self.device)
        # (B, 1, H, W)
        head_mask = (
            torch.from_numpy(cv2.resize(head_mask, (self.input_resolution, self.input_resolution)))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        # feed to network
        heatmap, inout, hx = self.gaze_follower.detect(frame, head_image, head_mask, hidden_state, torch.IntTensor([1]))

        # heatmap modulation
        heatmap = heatmap.squeeze()
        heatmap_modulated = heatmap * 255
        inout = inout.cpu().numpy().squeeze()
        inout_modulated = 1 / (1 + np.exp(-inout))
        inout_modulated = (1 - inout_modulated) * 255
        heatmap_resized = cv2.resize(heatmap_modulated, (original_shape[1], original_shape[0])) - inout_modulated
        heatmap_resized = np.clip(heatmap_resized, 0, 255).astype(np.uint8)
        heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        if draw:
            # select a color
            color = colors(id, bgr=True) if id is not None else None
            # draw head bounding box
            frame_annotated = frame_original
            # frame_annotated = draw_bbox(
            #     frame_original,
            #     head_bbox,
            #     obj_id=id,
            #     label=1,
            #     confidence=confidence,
            #     color=color,
            # )
            # draw arrows
            if arrow:
                # only draw arrows if gaze target in frame
                frame_annotated = draw_gaze_arrow(
                    frame_annotated,
                    head_bbox,
                    heatmap_modulated,
                    output_resolution=self.output_resolution,
                    color=color,
                )
                # if inout_modulated < self.out_threshold:
                #     frame_annotated = draw_gaze_arrow(
                #         frame_annotated,
                #         head_bbox,
                #         heatmap_modulated,
                #         output_resolution=self.output_resolution,
                #         color=color,
                #     )
            # or draw the heatmap overlay
            else:
                frame_annotated = draw_heatmap_overlay(frame_annotated, None, heatmap_resized)
            # additionally return the frame with annotation
            return (
                heatmap,
                inout,
                hx,
                heatmap_resized,
                inout_modulated,
                frame_annotated,
            )

        return heatmap, inout, hx, heatmap_resized, inout_modulated, None
