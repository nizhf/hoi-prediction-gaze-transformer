import warnings
import os
from typing import Tuple, List, Dict, Optional, Union
from collections import OrderedDict
import torch
from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torchvision
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.resnet import Bottleneck, ResNet
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


class FasterRCNNFeatureExtractor(FasterRCNN):
    def __init__(self, backbone, num_classes=91) -> None:
        super().__init__(backbone=backbone, num_classes=num_classes)
        self.roi_align = self.roi_heads.box_roi_pool

    # TODO modify this function to also return the features
    def forward(self, images, targets=None):
        # type: (List[torch.Tensor], Optional[List[Dict[str, torch.Tensor]]]) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            "Expected target boxes to be a tensor"
                            "of shape [N, 4], got {:}.".format(boxes.shape)
                        )
                else:
                    raise ValueError(
                        "Expected target boxes to be of type "
                        "Tensor, got {:}.".format(type(boxes))
                    )

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        " Found invalid box {} for target at index {}.".format(
                            degen_bb, target_idx
                        )
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn(
                    "RCNN always returns a (Losses, Detections) tuple in scripting"
                )
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

def fasterrcnn_resnet101(
    num_classes=91,
    pretrained_model_path="weights/fasterrcnn/fasterrcnn_resnet101.pt",
    pretrained_backbone_dir="weights/backbone/",
    trainable_layers=3,
):
    """
    Construct a FasterRCNN object detector + feature extractor with ResNet101 backbone

    Args:
        num_classes (int, optional): [description]. Defaults to 91.
        pretrained_model_path (str, optional): [description]. Defaults to "weights/fasterrcnn/fasterrcnn_resnet101_fpn.pt".
        pretrained_backbone_dir (str, optional): [description]. Defaults to "weights/backbone/".
        trainable_layers (int, optional): [description]. Defaults to 3.

    Returns:
        FasterRCNNFeatureExtractor: The detector
    """
    if pretrained_model_path is not None:
        # no need to download the backbone if pretrained is set
        pretrained_backbone_dir = None
    # init ResNet
    backbone = ResNet(
        block=Bottleneck, layers=[3, 4, 23, 3], norm_layer=misc_nn_ops.FrozenBatchNorm2d
    )
    # load pretrained backbone
    if pretrained_backbone_dir is not None:
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet101-63fe2227.pth",
            model_dir=pretrained_backbone_dir,
            progress=True,
        )
        backbone.load_state_dict(state_dict)
    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
        :trainable_layers
    ]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    # load pretrained model
    model = FasterRCNNFeatureExtractor(backbone, num_classes)
    if pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path)
        model.load_state_dict(state_dict["model"])

    return model


def fasterrcnn_resnet101_fpn(
    num_classes=91,
    pretrained_model_path="weights/fasterrcnn/",
    pretrained_backbone_path="weights/backbone/",
    pretrained_download=True,
    trainable_layers=3,
):
    """
    Construct a FasterRCNN object detector + feature extractor with ResNet101+FPN backbone
    Partially based on torchvision resnet_fpn_backbone()

    Args:
        num_classes (int, optional): [description]. Defaults to `91`.
        pretrained_model_path (str, optional): [description]. Defaults to `"weights/fasterrcnn/fasterrcnn_resnet101_fpn.pt"`.
        pretrained_backbone_dir (str, optional): [description]. Defaults to `"weights/backbone/"`.
        pretrained_download (bool, optional): download the weights. Defaults to `True`.
        trainable_layers (int, optional): [description]. Defaults to `3`.

    Returns:
        FasterRCNNFeatureExtractor: The detector
    """
    if pretrained_model_path is not None:
        # no need to download the backbone if pretrained is set
        pretrained_backbone_path = None
    # init ResNet
    backbone_resnet = ResNet(
        block=Bottleneck, layers=[3, 4, 23, 3], norm_layer=misc_nn_ops.FrozenBatchNorm2d
    )
    # load pretrained backbone
    if pretrained_backbone_path is not None:
        if pretrained_download:
            state_dict = load_state_dict_from_url(
                "https://download.pytorch.org/models/resnet101-63fe2227.pth",
                model_dir=pretrained_backbone_path,
                progress=True,
            )
        else:
            state_dict = torch.load(pretrained_backbone_path)
        backbone_resnet.load_state_dict(state_dict)
    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
        :trainable_layers
    ]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone_resnet.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    extra_blocks = LastLevelMaxPool()
    returned_layers = [1, 2, 3, 4]
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone_resnet.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    # construct backbone with fpn
    backbone = BackboneWithFPN(
        backbone_resnet,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=extra_blocks,
    )

    # load pretrained model
    model = FasterRCNNFeatureExtractor(backbone, num_classes)
    if pretrained_model_path is not None:
        if pretrained_download:
            state_dict = load_state_dict_from_url(
                "https://ababino-models.s3.amazonaws.com/resnet101_7a82fa4a.pth",
                model_dir=pretrained_model_path,
            )
        else:
            state_dict = torch.load(pretrained_model_path)
        model.load_state_dict(state_dict["model"])

    return model
