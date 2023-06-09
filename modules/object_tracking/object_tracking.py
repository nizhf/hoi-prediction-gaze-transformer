#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import cv2
import torch
import gdown

from .yolov5_deepsort import ObjectDetectionYolov5, ObjectTrackingDeepSortYolov5
from common.plot import draw_bboxes
from common.config_parser import get_config

crowd_human_url = "https://drive.google.com/uc?id=1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb"


class ObjectTracking:
    """
    The Object Tracking Module

    Args:
        yolo_weights_path (str): path to the YOLOv5 weights. Default to `"weights/yolov5/yolov5l.pt"`).
        deep_sort_model_dir (str): folder path of DeepSORT model. Default to `"weights/deep_sort/"`.
        config_path (str): path to the Object Tracking config file. Default to `"configs/object_tracking.yaml"`.
        device (str): device name. Default to `"cuda:0"`.
    """

    def __init__(
        self,
        yolo_weights_path: str = "weights/yolov5/yolov5l.pt",
        deep_sort_model_dir: str = "weights/deep_sort/",
        config_path: str = "configs/object_tracking.yaml",
        device="cuda:0",
    ):
        print("Initializing Object Tracking YOLOv5+DeepSORT Module...")
        # load configs
        self.configs = get_config(config_path)
        print(self.configs)
        # initialize YOLOv5 and Deep SORT with given configs
        self.object_tracker = ObjectTrackingDeepSortYolov5(
            yolo_weights_path=yolo_weights_path,
            deep_sort_model_dir=deep_sort_model_dir,
            deep_sort_model_type=self.configs["DEEPSORT"]["MODEL_TYPE"],
            yolov5_conf_thres=self.configs["YOLOv5"]["CONF_THRESHOLD"],
            yolov5_iou_thres=self.configs["YOLOv5"]["IOU_THRESHOLD"],
            deepsort_max_dist=self.configs["DEEPSORT"]["MAX_DIST"],
            deepsort_min_confidence=self.configs["DEEPSORT"]["MIN_CONFIDENCE"],
            # deepsort_nms_max_overlap=self.configs["DEEPSORT"]["NMS_MAX_OVERLAP"],
            deepsort_max_iou_distance=self.configs["DEEPSORT"]["MAX_IOU_DISTANCE"],
            deepsort_max_age=self.configs["DEEPSORT"]["MAX_AGE"],
            deepsort_max_time_since_update=self.configs["DEEPSORT"]["MAX_TIME_SINCE_UPDATE"],
            deepsort_n_init=self.configs["DEEPSORT"]["N_INIT"],
            deepsort_nn_budget=self.configs["DEEPSORT"]["NN_BUDGET"],
            classes=None,
            device=device,
        )
        self.yolov5_stride = self.object_tracker.yolov5_stride
        self.video_writer = None
        self.last_out_path = None
        # self.im_show_available = check_imshow()
        print(f"{len(self.object_tracker.class_names)} available objects: {self.object_tracker.class_names}")
        print("Object Tracking Module Initialization Finished.")

    def clear(self):
        """
        Clear the deepsort memory
        """
        self.object_tracker.clear()

    def warmup(self, frame, original_frame):
        """
        After initializing the model, the first two results are empty. This function run the first tracking twice.

        Args:
            frame (_type_): _description_
            original_frame (_type_): _description_
        """
        # original frame to numpy
        if isinstance(original_frame, torch.Tensor):
            original_frame = original_frame.numpy()
        for _ in range(self.configs["DEEPSORT"]["N_INIT"] - 1):
            self.object_tracker.track(frame, original_frame)

    def track_one(
        self,
        frame: torch.Tensor,
        original_frame: np.ndarray,
        draw: bool = True,
        save_path: str = None,
        cap: cv2.VideoCapture = None,
        fourcc: str = "mp4v",
        include_person: bool = True,
    ):
        """
        Call the object_tracking_module.track(frame, original_frame)
        Do some post processing, such as draw bounding boxes, display results, or save the video

        Args:
            frame (torch.Tensor): the down-scaled and padded frame, float [0, 1] Tensor shape (B, C, H, W).
            original_frame (np.ndarray): the original frame, shape (H, W, C), cv2 BGR.
            draw (bool): If True, draw the bounding boxes. Defaults to `True`.
            save_path (str): Path to save the annotated video, if `None`, not save. Defaults to `None`.
            cap (cv2.VideoCapture): Original video format information. Defaults to `None`.
            fourcc (str): output video codec. Default to `"mp4v"`.
            include_person (bool): include person. Default to `True`.

        Returns:
            Tuple[List]: lists of bounding boxes [x1, y1, x2, y2], ids, labels (1, 19), names (person, horse),
                confidence scores, and annotated frame if needed (else None)
        """
        # original frame to numpy
        if isinstance(original_frame, torch.Tensor):
            original_frame = original_frame.numpy()
        # run object tracking once
        bboxes, ids, labels, names, confidences = self.object_tracker.track(frame, original_frame)
        # exclude person if needed
        if not include_person:
            not_person_idx = np.array(labels) != 1
            bboxes = np.array(bboxes)[not_person_idx]
            ids = np.array(ids)[not_person_idx]
            labels = np.array(labels)[not_person_idx]
            names = np.array(names)[not_person_idx]
            confidences = np.array(confidences)[not_person_idx]
        # draw bounding boxes if needed
        if draw or save_path is not None:
            frame_annotated = draw_bboxes(original_frame.copy(), bboxes, ids, labels, names, confidences)
            # save the annotated video
            if save_path is not None:
                # new video, create a new video writer
                if self.last_out_path != save_path:
                    self.last_out_path = save_path
                    if isinstance(self.video_writer, cv2.VideoWriter):
                        # release previous video writer
                        self.video_writer.release()
                    if cap:
                        # video
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        # stream
                        fps, w, h = 30, frame_annotated.shape[1], frame_annotated.shape[0]
                    self.video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                self.video_writer.write(frame_annotated)

            # additionally return the annotated frame
            return bboxes, ids, labels, names, confidences, frame_annotated

        return bboxes, ids, labels, names, confidences, None


class HeadDetection:
    def __init__(
        self,
        crowd_human_weight_path: str = "weights/yolov5/crowdhuman_yolov5m.pt",
        config_path: str = "configs/object_tracking.yaml",
        device="cuda:0",
    ):
        print("Initializing Head Detection YOLOv5 Module...")
        # download crowd human yolov5 weights if not exists
        crowd_human_weight_path = Path(str(crowd_human_weight_path).strip().replace("'", ""))
        if not crowd_human_weight_path.exists():
            print("Download Crowd human weights...")
            gdown.download(crowd_human_url, str(crowd_human_weight_path))
        # load configs
        self.configs = get_config(config_path)
        print(self.configs)
        # initialize YOLOv5 with given configs
        self.object_detector = ObjectDetectionYolov5(
            yolo_weights_path=crowd_human_weight_path,
            yolov5_conf_thres=self.configs["YOLOv5"]["CONF_THRESHOLD"],
            yolov5_iou_thres=self.configs["YOLOv5"]["IOU_THRESHOLD"],
            device=device,
        )

        self.yolov5_stride = self.object_detector.yolov5_stride
        self.video_writer = None
        self.last_out_path = None
        print(f"Stride: {self.yolov5_stride}")
        print(f"{len(self.object_detector.class_names)} available objects: {self.object_detector.class_names}")
        print("Head Detection Module Initialization Finished.")

    def detect_one(
        self,
        frame: torch.Tensor,
        original_frame: np.ndarray,
        draw: bool = True,
        save_path: str = None,
        cap: cv2.VideoCapture = None,
        fourcc: str = "mp4v",
        include_person: bool = False,
    ):
        """
        Call the object_detector.detect(frame, original_frame)
        Do some post processing, such as draw bounding boxes, display results, or save the video

        Args:
            frame (torch.Tensor): the down-scaled and padded frame, float [0, 1] Tensor shape (B, C, H, W).
            original_frame (np.ndarray): the original frame, shape (H, W, C), cv2 BGR.
            draw (bool): If True, draw the bounding boxes. Defaults to `True`.
            save_path (str): Path to save the annotated video, if `None`, not save. Defaults to `None`.
            cap (cv2.VideoCapture): Original video format information. Defaults to `None`.
            fourcc (str): output video codec. Default to `"mp4v"`.
            include_person (bool): include person or only faces. Default to `False`.

        Returns:
            Tuple[List]: lists of bounding boxes [x1, y1, x2, y2] (numpy array), labels (1, 19), names (person, horse), confidence scores, and annotated frame if needed (else None)
        """
        # run object tracking once
        bboxes, labels, confidences = self.object_detector.detect(frame, original_frame)
        # exclude person if needed
        if len(bboxes) > 0:
            if not include_person:
                # head_idx = np.array(labels) == 1
                # bboxes = np.array(bboxes)[head_idx]
                # labels = np.array(labels)[head_idx]
                # confidences = np.array(confidences)[head_idx]
                head_idx = labels[0].cpu().numpy() == 1
                # round to closest integer pixel
                bboxes = bboxes[0].cpu().numpy()[head_idx].round()
                labels = labels[0].cpu().numpy()[head_idx]
                confidences = confidences[0].cpu().numpy()[head_idx]
            else:
                # round to closest integer pixel
                bboxes = bboxes[0].cpu().numpy().round()
                labels = labels[0].cpu().numpy()
                confidences = confidences[0].cpu().numpy()
        names = [self.object_detector.class_names[int(label)] for label in labels]
        # draw bounding boxes if needed
        if draw or save_path is not None:
            frame_annotated = draw_bboxes(
                original_frame.copy(), bboxes, [None] * len(bboxes), labels, names, confidences
            )
            # save the annotated video
            if save_path is not None:
                # new video, create a new video writer
                if self.last_out_path != save_path:
                    self.last_out_path = save_path
                    if isinstance(self.video_writer, cv2.VideoWriter):
                        # release previous video writer
                        self.video_writer.release()
                    if cap:
                        # video
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        # stream
                        fps = 30
                        w = frame_annotated.shape[1]
                        h = frame_annotated.shape[0]
                    fourcc = cv2.VideoWriter_fourcc(*fourcc)
                    self.video_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                # write the frame
                self.video_writer.write(frame_annotated)

            # additionally return the annotated frame
            return bboxes, labels, names, confidences, frame_annotated

        return bboxes, labels, names, confidences, None


class HeadTracking:
    """
    The Head Tracking Module

    Args:
        yolo_weights_path (str): path to the YOLOv5 weights. Default to `"weights/yolov5/crowdhuman_yolov5m.pt"`).
        deep_sort_model_type (str): type of DeepSORT model, see reid repo. Default to `"osnet_x1_0"`.
        config_path (str): path to the Object Tracking config file. Default to `"configs/object_tracking.yaml"`.
        device (str): device name. Default to `"cuda:0"`.
    """

    def __init__(
        self,
        crowd_human_weight_path: str = "weights/yolov5/crowdhuman_yolov5m.pt",
        deep_sort_model_dir: str = "weights/deep_sort/",
        config_path: str = "configs/object_tracking.yaml",
        device="cuda:0",
    ):
        print("Initializing Head Tracking YOLOv5+DeepSORT Module...")
        # download crowd human yolov5 weights if not exists
        crowd_human_weight_path = Path(str(crowd_human_weight_path).strip().replace("'", ""))
        if not crowd_human_weight_path.exists():
            print("Download Crowd human weights...")
            gdown.download(crowd_human_url, str(crowd_human_weight_path))
        # load configs
        self.configs = get_config(config_path)
        print(self.configs)
        # initialize YOLOv5 and Deep SORT with given configs
        self.object_tracker = ObjectTrackingDeepSortYolov5(
            yolo_weights_path=crowd_human_weight_path,
            deep_sort_model_dir=deep_sort_model_dir,
            deep_sort_model_type=self.configs["DEEPSORT"]["MODEL_TYPE"],
            yolov5_conf_thres=self.configs["YOLOv5"]["CONF_THRESHOLD"],
            yolov5_iou_thres=self.configs["YOLOv5"]["IOU_THRESHOLD"],
            deepsort_max_dist=self.configs["DEEPSORT"]["MAX_DIST"],
            deepsort_min_confidence=self.configs["DEEPSORT"]["MIN_CONFIDENCE"],
            # deepsort_nms_max_overlap=self.configs["DEEPSORT"]["NMS_MAX_OVERLAP"],
            deepsort_max_iou_distance=self.configs["DEEPSORT"]["MAX_IOU_DISTANCE"],
            deepsort_max_age=self.configs["DEEPSORT"]["MAX_AGE"],
            deepsort_max_time_since_update=self.configs["DEEPSORT"]["MAX_TIME_SINCE_UPDATE"],
            deepsort_n_init=self.configs["DEEPSORT"]["N_INIT"],
            deepsort_nn_budget=self.configs["DEEPSORT"]["NN_BUDGET"],
            classes=None,
            device=device,
        )
        self.yolov5_stride = self.object_tracker.yolov5_stride
        self.video_writer = None
        self.last_out_path = None
        print(f"{len(self.object_tracker.class_names)} available objects: {self.object_tracker.class_names}")
        print("Head Tracking Module Initialization Finished.")

    def track_one(
        self,
        frame: torch.Tensor,
        original_frame: np.ndarray,
        draw: bool = True,
        save_path: str = None,
        cap: cv2.VideoCapture = None,
        fourcc: str = "mp4v",
        include_person: bool = False,
    ):
        """
        Call the object_tracker.track(frame, original_frame)
        Do some post processing, such as draw bounding boxes, display results, or save the video

        Args:
            frame (torch.Tensor): the down-scaled and padded frame, float [0, 1] Tensor shape (B, C, H, W).
            original_frame (np.ndarray): the original frame, shape (H, W, C), cv2 BGR.
            draw (bool): If True, draw the bounding boxes. Defaults to `True`.
            save_path (str): Path to save the annotated video, if `None`, not save. Defaults to `None`.
            cap (cv2.VideoCapture): Original video format information. Defaults to `None`.
            fourcc (str): output video codec. Default to `"mp4v"`.
            include_person (bool): include person or only faces. Default to `False`.

        Returns:
            Tuple[List]: lists of bounding boxes [x1, y1, x2, y2], ids, labels (1, 19), names (person, horse),
                confidence scores, and annotated frame if needed (else None)
        """
        # run object tracking once
        bboxes, ids, labels, names, confidences = self.object_tracker.track(frame, original_frame)
        # exclude person if needed
        if not include_person:
            head_idx = np.array(labels) == 1
            bboxes = np.array(bboxes)[head_idx]
            ids = np.array(ids)[head_idx]
            labels = np.array(labels)[head_idx]
            names = np.array(names)[head_idx]
            confidences = np.array(confidences)[head_idx]
        # draw bounding boxes if needed
        if draw or save_path is not None:
            frame_annotated = draw_bboxes(original_frame.copy(), bboxes, ids, labels, names, confidences)
            # save the annotated video
            if save_path is not None:
                # new video, create a new video writer
                if self.last_out_path != save_path:
                    self.last_out_path = save_path
                    if isinstance(self.video_writer, cv2.VideoWriter):
                        # release previous video writer
                        self.video_writer.release()
                    if cap:
                        # video
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        # stream
                        fps = 30
                        w = frame_annotated.shape[1]
                        h = frame_annotated.shape[0]
                    fourcc = cv2.VideoWriter_fourcc(*fourcc)
                    self.video_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                # write the frame
                self.video_writer.write(frame_annotated)

            # additionally return the annotated frame
            return bboxes, ids, labels, names, confidences, frame_annotated

        return bboxes, ids, labels, names, confidences, None
