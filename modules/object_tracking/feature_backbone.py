from typing import List, Tuple
import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import RoIAlign, MultiScaleRoIAlign
from torchvision.models.resnet import ResNet, Bottleneck, model_urls
from torch.hub import load_state_dict_from_url


class FeatureExtractionResNet101(nn.Module):
    def __init__(
        self,
        backbone_weights_dir="weights/backbone/",
        download=True,
        finetune=False,
        finetune_layers=[],
    ):
        super().__init__()
        backbone = ResNet(Bottleneck, [3, 4, 23, 3])
        # download pretrained backbone
        if download:
            state_dict = load_state_dict_from_url(
                model_urls["resnet101"], model_dir=str(backbone_weights_dir), progress=True
            )
            backbone.load_state_dict(state_dict)
        # or use given ResNet weight for finetune
        elif finetune:
            state_dict = torch.load(backbone_weights_dir)
            backbone.load_state_dict(state_dict)

        self.backbone_base = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.backbone_head = nn.Sequential(backbone.layer4, backbone.avgpool)
        # self.roi_align = MultiScaleRoIAlign(["0"], (7, 7), 0)
        self.roi_align = RoIAlign((7, 7), 1.0 / 16.0, 0, True)
        self.out_features = backbone.fc.in_features

        # not download, not finetune, then load full model
        if not download and not finetune:
            self.load_state_dict(torch.load(backbone_weights_dir))

        # set trainable layers
        for name, param in self.named_parameters():
            param.requires_grad_(False)
            for finetune_name in finetune_layers:
                if finetune_name in name:
                    param.requires_grad_(True)

    def forward(
        self,
        x: torch.Tensor,
        bboxes: List[torch.Tensor],
        feature_map: torch.Tensor = None,
        image_shape: List[Tuple[int, int]] = None,
    ):
        if feature_map is None:
            feature_map = self.backbone_base(x)
        feature_bboxes = self.roi_align(feature_map, bboxes)
        feature_bboxes = self.backbone_head(feature_bboxes)
        feature_bboxes = torch.flatten(feature_bboxes, 1)
        return feature_bboxes, feature_map


class FeatureExtractionResNet50(nn.Module):
    def __init__(
        self,
        backbone_weights_dir="weights/backbone/",
        download=True,
        finetune=False,
        finetune_layers=[],
    ):
        super().__init__()
        backbone = ResNet(Bottleneck, [3, 4, 6, 3])
        # download pretrained backbone
        if download:
            state_dict = load_state_dict_from_url(
                model_urls["resnet101"], model_dir=str(backbone_weights_dir), progress=True
            )
            backbone.load_state_dict(state_dict)
        # or use given ResNet weight for finetune
        elif finetune:
            state_dict = torch.load(backbone_weights_dir)
            backbone.load_state_dict(state_dict)

        self.backbone_base = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.backbone_head = nn.Sequential(backbone.layer4, backbone.avgpool)
        # self.roi_align = MultiScaleRoIAlign(["0"], (7, 7), 0)
        self.roi_align = RoIAlign((7, 7), 1.0 / 16.0, 0, True)
        self.out_features = backbone.fc.in_features
        # not download, not finetune, then load full model
        if not download and not finetune:
            self.load_state_dict(torch.load(backbone_weights_dir))
        # set trainable layers
        for name, param in self.named_parameters():
            param.requires_grad_(False)
            for finetune_name in finetune_layers:
                if finetune_name in name:
                    param.requires_grad_(True)

    def forward(
        self,
        x: torch.Tensor,
        bboxes: List[torch.Tensor],
        feature_map: torch.Tensor = None,
        image_shape: List[Tuple[int, int]] = None,
    ):
        if feature_map is None:
            feature_map = self.backbone_base(x)
        feature_bboxes = self.roi_align(feature_map, bboxes)
        feature_bboxes = self.backbone_head(feature_bboxes)
        feature_bboxes = torch.flatten(feature_bboxes, 1)
        return feature_bboxes, feature_map


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        arch="resnet101",
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        model_dir="weights/backbone/",
        pretrained=True,
        progress=True,
        device="cuda:0",
    ):
        super().__init__()
        self.model = ResNet(block, layers).to(device)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch], model_dir=model_dir, progress=progress)
            self.model.load_state_dict(state_dict)
        self.model = create_feature_extractor(self.model, return_nodes={"avgpool": "0"})

    def forward(self, x):
        x = self.model(x)["0"]
        x = torch.flatten(x, 1)
        return x
