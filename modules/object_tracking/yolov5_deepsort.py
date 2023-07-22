#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from typing import List
import torch
import numpy as np
import time
import torchvision

from .yolov5.models.common import DetectMultiBackend
from .yolov5.utils.general import scale_boxes, xyxy2xywh, xywh2xyxy
from .yolov5.utils.metrics import box_iou
from .deep_sort.deep_sort import DeepSort


class ObjectDetectionYolov5:
    r"""
    The Object Detection class using YOLOv5, without tracking.

    Args:
        yolo_weights_path (str): path to the YOLOv5 weights. Default to `"weights/yolov5/yolov5l.pt"`).
        deep_sort_weights_path (str): path to the Deep SORT weights. Default to `"weights/deep_sort/ckpt.t7"`.
        yolov5_conf_thres (float): YOLOv5 parameter, object confidence threshold for NMS (non-maximum suppression). Default to `0.4`.
        yolov5_iou_thres (float): YOLOv5 parameter, IOU threshold for NMS (non-maximum suppression). Default to `0.5`.
        classes (List[int]): filter of tracking classes. Default to `None`.
        device (str): device name. Default to `"cuda:0"`.
    """

    def __init__(
        self,
        yolo_weights_path: str = "weights/yolov5/yolov5l.pt",
        yolov5_conf_thres: float = 0.4,
        yolov5_iou_thres: float = 0.5,
        classes: List[int] = None,
        device="cuda:0",
    ):
        self.conf_thres = yolov5_conf_thres
        self.iou_thres = yolov5_iou_thres
        self.device = device
        self.classes = classes

        # load YOLOv5 model
        self.yolov5_model = DetectMultiBackend(yolo_weights_path, device=torch.device(self.device))
        # model stride
        self.yolov5_stride = self.yolov5_model.stride
        # get class names
        self.class_names = (
            self.yolov5_model.module.names if hasattr(self.yolov5_model, "module") else self.yolov5_model.names
        )

        # set eval mode
        self.yolov5_model.eval()

        # run inference once to warmup
        self.yolov5_model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(self.yolov5_model.model.parameters())))

    def detect(self, frame: torch.Tensor, original_frame: np.ndarray):
        """
        Detect the objects in the given frame

        Args:
            frame (torch.Tensor): the down-scaled and padded frame, float [0, 1] Tensor shape (B, C, H, W).
            original_frame (np.ndarray): the original frame, shape (H, W, C), cv2 BGR.

        Returns:
            Tuple[List]: lists of bounding boxes [x1, y1, x2, y2], labels (1, 19), names (person, horse), and confidence scores
        """
        # detect objects
        pred = self.yolov5_model(frame)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]  # latest YOLOv5 model outputs [inference_out, loss_out] in validation mode
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, time_limit=1.0)
        # process detection results
        bbox_list = []
        label_list = []
        confidence_list = []
        for detected in pred:
            if detected is not None and len(detected) > 0:
                # rescale bboxes from down-scaled size to original size
                detected[:, :4] = scale_boxes(frame.shape[2:], detected[:, :4], original_frame.shape[:2])
                bbox_list.append(detected[:, :4])
                label_list.append(detected[:, 5])
                confidence_list.append(detected[:, 4])

        return bbox_list, label_list, confidence_list


class ObjectTrackingDeepSortYolov5:
    r"""
    The Object Tracker class using YOLOv5 + DeepSort

    Args:
        yolo_weights_path (str): path to the YOLOv5 weights. Default to `"weights/yolov5/yolov5l.pt"`).
        deep_sort_weights_path (str): path to the Deep SORT weights. Default to `"weights/deep_sort/ckpt.t7"`.
        yolov5_conf_thres (float): YOLOv5 parameter, object confidence threshold for NMS (non-maximum suppression).
            Default to `0.4`.
        yolov5_iou_thres (float): YOLOv5 parameter, IOU threshold for NMS (non-maximum suppression). Default to `0.5`.
        deepsort_max_dist (float): Deep SORT parameter, max cosine distance. Defaults to `0.2`.
        deepsort_min_confidence (float): Deep SORT parameter, min confidence. Defaults to `0.3`.
        deepsort_max_iou_distance (float): Deep SORT parameter, max IOU distance. Defaults to `0.7`.
        deepsort_max_age (int): Deep SORT parameter, maximum number of missed misses before a track is deleted.
            Defaults to `70`.
        deepsort_n_init (int): Deep SORT parameter, number of consecutive detections before the track is confirmed.
            Defaults to `3`.
        deepsort_nn_budget (int): Deep SORT parameter, fix samples per class to at most this number,
            removes the oldest samples when the budget is reached. Defaults to `100`.
        classes (List[int]): filter of tracking classes. Default to `None`.
        device (str): device name. Default to `"cuda:0"`.
    """

    def __init__(
        self,
        yolo_weights_path: str = "weights/yolov5/yolov5l.pt",
        deep_sort_model_dir: str = "weights/deep_sort/",
        deep_sort_model_type: str = "osnet_x1_0",
        yolov5_conf_thres: float = 0.4,
        yolov5_iou_thres: float = 0.5,
        deepsort_max_dist: float = 0.2,
        deepsort_min_confidence: float = 0.3,
        # deepsort_nms_max_overlap: float=0.5,
        deepsort_max_iou_distance: float = 0.7,
        deepsort_max_age: int = 70,
        deepsort_max_time_since_update: int = 1,
        deepsort_n_init: int = 3,
        deepsort_nn_budget: int = 100,
        classes: List[int] = None,
        device="cuda:0",
    ):
        self.conf_thres = yolov5_conf_thres
        self.iou_thres = yolov5_iou_thres
        self.device = device
        self.classes = classes

        # initialize Deep SORT
        self.deepsort = DeepSort(
            model_type=deep_sort_model_type,
            model_dir=deep_sort_model_dir,
            max_dist=deepsort_max_dist,
            min_confidence=deepsort_min_confidence,
            max_iou_distance=deepsort_max_iou_distance,
            max_age=deepsort_max_age,
            max_time_since_update=deepsort_max_time_since_update,
            n_init=deepsort_n_init,
            nn_budget=deepsort_nn_budget,
            use_cuda=device != "cpu",
        )

        # initialize YOLOv5
        self.yolov5_object_detector = ObjectDetectionYolov5(
            yolo_weights_path, yolov5_conf_thres, yolov5_iou_thres, classes, device
        )
        self.yolov5_stride = self.yolov5_object_detector.yolov5_stride
        # get class names
        self.class_names = self.yolov5_object_detector.class_names

    def track(self, frame: torch.Tensor, original_frame: np.ndarray):
        """
        Track the objects in the given frame

        Args:
            frame (torch.Tensor): the down-scaled and padded frame, float [0, 1] Tensor shape (B, C, H, W).
            original_frame (np.ndarray): the original frame, shape (H, W, C), cv2 BGR.

        Returns:
            Tuple[List]: lists of bounding boxes [x1, y1, x2, y2] (numpy array), ids, labels (1, 19), names (person, horse), and confidence scores
        """
        # detect objects
        bbox_detected, label_detected, confidence_detected = self.yolov5_object_detector.detect(frame, original_frame)
        # process detection results
        bbox_list = []
        id_list = []
        label_list = []
        name_list = []
        confidence_list = []
        for bboxes, labels, confs in zip(bbox_detected, label_detected, confidence_detected):
            if bboxes is not None and len(bboxes) > 0:
                # prepare data for Deep SORT
                xywhs = xyxy2xywh(bboxes)
                # pass detections to Deep SORT
                deepsort_outs = self.deepsort.update(xywhs.cpu(), confs.cpu(), labels.cpu(), original_frame)
                # process tracking results
                if len(deepsort_outs) > 0:
                    for out, conf in zip(deepsort_outs, confs):
                        bbox = out[:4]
                        id = int(out[4])
                        label = int(out[5])
                        name = self.class_names[label]
                        bbox_list.append(bbox)
                        id_list.append(id)
                        label_list.append(label)
                        name_list.append(name)
                        confidence_list.append(conf.item())
            else:
                self.deepsort.increment_ages()

        return bbox_list, id_list, label_list, name_list, confidence_list

    def clear(self):
        """
        start a new video
        """
        self.deepsort.tracker.tracks = []
        self.deepsort.tracker._next_id = 1


# adapted from yolov5.utils.general.non_max_suppression, changed time limit
def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    time_limit=0.030,
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    LOGGER = logging.getLogger("yolov5")

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = time_limit * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING: NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output
