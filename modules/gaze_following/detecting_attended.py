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
from common.download import download_file_from_dropbox


class DetectingAttendedVisualTargets:
    """
    Gaze Following method introduced in Detecting Attended Visual Targets in Video

    Args:
        weight_path (str): path to the model weights. Defaults to "weights/attention_target/model_videoatttarget.pt".
        input_resolution (int): input image resolution, used in the transform. Defaults to 224.
        output_resolution (int): output heatmap resolution. Defaults to 64.
        num_lstm_layer (int): number of LSTM layers. Defaults to 2.
        device (str): device. Defaults to "cuda:0".
    """

    def __init__(
        self,
        weight_path: str = "weights/detecting_attended/model_videoatttarget.pt",
        input_resolution: int = 224,
        output_resolution: int = 64,
        num_lstm_layers: int = 2,
        device: str = "cuda:0",
    ):
        self.device = device
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.num_lstm_layers = num_lstm_layers
        # initialize model
        self.model = ModelSpatioTemporal(num_lstm_layers=self.num_lstm_layers)
        self.model.to(self.device)
        # check weight exists, if not, download from Dropbox
        if not Path(weight_path).exists():
            print("Download Detecting Attended Visual Targets weights...")
            download_file_from_dropbox(
                "https://www.dropbox.com/s/ywd16kcv06vn93x/model_videoatttarget.pt",
                weight_path,
            )
        # load weights
        model_dict = self.model.state_dict()
        snapshot = torch.load(weight_path)
        snapshot = snapshot["model"]
        model_dict.update(snapshot)
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def detect(
        self,
        image: torch.Tensor,
        head_image: torch.Tensor,
        head_mask: torch.Tensor,
        hidden_state: Tuple[torch.Tensor] = None,
        batch_sizes: torch.Tensor = torch.IntTensor([1]),
    ) -> Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor]]:
        """
        Detect gaze direction and generate attention heatmap. Also detect whether the gaze target is inside or outside the scene.
        Can accept a batch of frames (multiple videos at multiple time steps). Use pack_padded_sequence to generate such data sequence.

        Args:
            image (torch.Tensor): the original image Tensor (batch&time, 3, input_resolution, input_resolution),
                can contain multiple streams (batches) and multiple time steps. Use pack_padded_sequence to generate such Tensor.
            head_image (torch.Tensor): the head image Tensor (batch&time, 3, input_resolution, input_resolution),
                can contain multiple streams (batches) and multiple time steps. Use pack_padded_sequence to generate such Tensor.
            head_mask (torch.Tensor): the head 0/1 mask Tensor (batch&time, 3, input_resolution, input_resolution),
                can contain multiple streams (batches) and multiple time steps. Use pack_padded_sequence to generate such Tensor.
            hidden_state (Tuple[torch.Tensor], optional): the last LSTM hidden state, if None, generate a zero state.
                Defaults to None.
            batch_sizes (torch.Tensor, optional): batch_sizes in PackSequence, [3, 2] meaning 3 videos at time step t=0,
                2 videos at t=1. Defaults to torch.IntTensor([1]).

        Returns:
            Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor]]: heatmap list, in_out, hx (hidden_state)
        """
        # init hidden state
        if hidden_state is None:
            hidden_state = (
                torch.zeros((self.num_lstm_layers, batch_sizes[0].int().item(), 512, 7, 7)).to(self.device),
                torch.zeros((self.num_lstm_layers, batch_sizes[0].int().item(), 512, 7, 7)).to(self.device),
            )  # (num_layers, batch_size, feature dims)

        # forward pass
        deconv, inout_val, hx = self.model(
            image, head_mask, head_image, hidden_scene=hidden_state, batch_sizes=batch_sizes
        )
        # generate heatmap
        heatmap_list = []
        for idx in range(len(batch_sizes)):
            scaled_heatmap = cv2.resize(
                deconv[idx].squeeze().cpu().numpy(),
                dsize=(self.output_resolution, self.output_resolution),
                interpolation=cv2.INTER_LINEAR,
            )
            heatmap_list.append(scaled_heatmap)

        return np.stack(heatmap_list, axis=0), inout_val, hx
