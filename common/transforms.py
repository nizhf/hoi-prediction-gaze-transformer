#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import cv2
import torch
from torchvision.transforms.functional import to_tensor, normalize, resize, hflip
from modules.object_tracking.yolov5.utils.augmentations import letterbox


class YOLOv5Transform:
    def __init__(self, img_size, stride):
        self.img_size = img_size
        self.stride = stride

    def __call__(self, frame):
        # padded resize
        frame, _, _ = letterbox(frame, new_shape=self.img_size, stride=self.stride, auto=True)
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # transform to Tensor
        frame = to_tensor(frame)
        # (B, C, H, W)
        if frame.ndimension() == 3:
            frame = frame.unsqueeze(0)
        return frame


class STTranTransform:
    def __init__(self, img_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.img_size = img_size
        self.mean = mean
        self.std = std

    def __call__(self, frame):
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # transform to Tensor (HWC to CHW, normalize to [0, 1])
        frame = to_tensor(frame)
        # normalize, use the common mean and std
        frame = normalize(frame, mean=self.mean, std=self.std, inplace=True)
        # resize
        frame = resize(frame, self.img_size)
        # (B, C, H, W)
        if frame.ndimension() == 3:
            frame = frame.unsqueeze(0)
        return frame


class DetectingAttendedTransform:
    def __init__(self, input_resolution, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.input_resolution = input_resolution
        self.mean = mean
        self.std = std

    def __call__(self, frame):
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # transform to Tensor (HWC to CHW, normalize to [0, 1])
        frame = to_tensor(frame)
        # normalize, use the common mean and std
        frame = normalize(frame, mean=self.mean, std=self.std, inplace=True)
        # resize
        frame = resize(frame, (self.input_resolution, self.input_resolution), antialias=True)
        # (B, C, H, W)
        if frame.ndimension() == 3:
            frame = frame.unsqueeze(0)
        return frame


class ClipRandomHorizontalFlipping:
    """
    Randomly flip a sequence of frames horizontally. Also flip the bbox annotations.
    The input frame should be a list of torch.Tensor, in (B, C, H, W) or (C, H, W) shape

    Args:
        p (float, optional): Probability of flipping. Defaults to 0.5.
    """

    def __init__(self, p=0.5, post_transform=None):
        self.p = p
        self.post_transform = post_transform

    def __call__(self, frames, annotations, meta_info):
        new_frames = []
        # roll dice to determine whether flipping
        if random.random() >= self.p:
            meta_info["hflip"] = False
            if self.post_transform is not None:
                for frame in frames:
                    new_frames.append(self.post_transform(frame))
            else:
                for frame in frames:
                    new_frames.append(frame)
            return new_frames, annotations, meta_info

        # horizontal flip the frames
        if self.post_transform is not None:
            for frame in frames:
                new_frames.append(self.post_transform(frame.flip(-1)))
        else:
            for frame in frames:
                new_frames.append(frame.flip(-1))
        # get original frame shape from meta_info
        original_shape = meta_info["original_shape"]
        width = original_shape[1]
        # horizontal flip the bboxes
        # frame mode: [{object_id: [xmin, ymin, xmax, ymax]}]
        if "traces" in annotations:
            for trace in annotations["traces"]:
                for bbox in trace.values():
                    x1 = width - bbox[2]
                    x2 = width - bbox[0]
                    bbox[0] = x1
                    bbox[2] = x2
        # clip, window mode: [[im_idx, xmin, ymin, xmax, ymax]]
        else:
            for bbox in annotations["bboxes"]:
                x1 = width - bbox[3]
                x2 = width - bbox[1]
                bbox[1] = x1
                bbox[3] = x2
        # anticipation mode, additionally process the bboxes in future
        if "anticipation" in annotations:
            for bbox in annotations["anticipation"]["bboxes"]:
                x1 = width - bbox[3]
                x2 = width - bbox[1]
                bbox[1] = x1
                bbox[3] = x2
        meta_info["hflip"] = True
        return new_frames, annotations, meta_info
