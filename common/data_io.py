#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, IterableDataset, DataLoader
import cv2

# acceptable image suffixes
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
# acceptable video suffixes
VID_FORMATS = ["mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv"]


class VideoDatasetLoader:
    """
    Load a video dataset for HOI prediction. The dataset is fed to the object tracking module.

    Args:
        path (str | Path): path to the dataset. Can be a directory or a video file.
        img_size (int): frame size after resize. Default to `640`.
        stride (int): stride for padding preprocessing. Default to `32`.
    """

    def __init__(self, path, transform=None, additional_transform=None):
        # load files from the directory
        self.path = Path(path)
        if "*" in str(self.path):
            file_paths_temp = sorted(glob.glob(str(self.path), recursive=True))
            file_paths = [Path(x) for x in file_paths_temp]
        elif self.path.is_dir():
            file_paths = sorted(self.path.iterdir())
        elif self.path.is_file():
            file_paths = [self.path]
        else:
            raise Exception(f"ERROR: {path} does not exist")
        # load supported videos
        self.video_paths = sorted([x for x in file_paths if x.suffix.lower()[1:] in VID_FORMATS])
        self.length = len(self.video_paths)
        self.video_cap = None
        self.new_video(self.video_paths[0])
        self.transform = transform
        self.additional_transform = additional_transform

    def __iter__(self):
        self.video_count = 0
        if self.video_cap is not None:
            self.video_cap.release()
        self.new_video(self.video_paths[0])
        return self

    def __next__(self):
        if self.video_count == self.length:
            raise StopIteration
        video_path = self.video_paths[self.video_count]
        success, frame0 = self.video_cap.read()
        # no return, end of video, read the next video
        if not success:
            self.video_count += 1
            self.video_cap.release()
            if self.video_count == self.length:
                raise StopIteration
            else:
                video_path = self.video_paths[self.video_count]
                self.new_video(video_path)
                success, frame0 = self.cap.read()
        self.frame_count += 1
        s = f"video {self.video_count + 1}/{self.length} ({self.frame_count}/{self.frame_num}) {video_path}: "
        # transform
        if self.transform:
            frame = self.transform(frame0)
        else:
            frame = frame0
        # meta_info
        meta_info = {
            "video_count": self.video_count,
            "video_path": video_path,
            "frame_count": self.frame_count,
            "frame_num": self.frame_num,
        }
        if self.additional_transform:
            frame_add = self.additional_transform(frame0)
            meta_info["additional"] = frame_add
        return frame, frame0, self.video_cap, s, meta_info

    def __len__(self):
        return self.length

    def new_video(self, path):
        self.frame_count = 0
        self.video_cap = cv2.VideoCapture(str(path))
        self.frame_num = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)


class FrameDatasetLoader(Dataset):
    """
    Load a video dataset (frames as images) for HOI prediction. The dataset is fed to the object tracking module.

    Args:
        path (str | Path | List): path to the dataset. If given a path with subdirectories, read all subdirectories.
            If given a list, add all subdirectories of all elements in the list.
        img_size (int): frame size after resize. Default to `640`.
        stride (int): stride for padding preprocessing. Default to `32`.
    """

    def __init__(self, path, transform=None, interval=1, start_idx=0):
        if isinstance(path, list):
            # TODO support multiple folders
            pass
        else:
            self.path = Path(path)
            sub_paths = list(self.path.iterdir())
            # contains only files, a single video
            if all([x.is_file() for x in sub_paths]):
                self.video_paths = [self.path]
            # read frames in all subdirectories
            else:
                self.video_paths = sorted([x for x in sub_paths if x.is_dir()])

        self.length = len(self.video_paths)

        self.interval = interval
        self.start_idx = start_idx
        self.new_video(self.video_paths[0])

        self.transform = transform

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __iter__(self):
        self.video_count = 0
        self.frame_count = 0
        return self

    def __next__(self):
        # current video finished
        if self.frame_count == self.frame_num:
            self.video_count += 1
            # dataset reaches the end
            if self.video_count == self.length:
                raise StopIteration
            # read a new video
            else:
                self.new_video(self.video_paths[self.video_count])
        # load current frame
        frame_path = self.frame_paths[self.frame_count]
        frame0 = cv2.imread(str(frame_path))
        self.frame_count += 1
        s = f"video {self.video_count + 1}/{self.length} ({self.frame_count}/{self.frame_num}) {frame_path}: "
        # transform
        if self.transform is not None:
            frame = self.transform(frame0)
        else:
            frame = frame0
        # meta_info
        meta_info = {
            "video_path": self.video_paths[self.video_count],
            "video_count": self.video_count,
            "frame_count": self.frame_count,
            "frame_num": self.frame_num,
            "frame_path": frame_path,
        }
        # None as a placeholder for cap in VideoDatasetLoader
        return frame, frame0, None, s, meta_info

    def __len__(self):
        return self.length

    def new_video(self, path):
        self.frame_count = 0
        self.current_path = path
        frame_paths = self.current_path.iterdir()
        self.frame_paths = sorted([x for x in frame_paths if x.suffix.lower()[1:] in IMG_FORMATS])
        self.frame_paths = self.frame_paths[self.start_idx :: self.interval]
        # read all frames in this video
        # self.frames = [cv2.imread(str(frame_path)) for frame_path in self.frame_paths]
        self.frame_num = len(self.frame_paths)
