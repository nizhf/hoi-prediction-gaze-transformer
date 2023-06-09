#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalBCEWithLogitLoss(nn.modules.loss._Loss):
    """Focal Loss with binary cross-entropy
    Implement the focal loss with class-balanced loss, using binary cross-entropy as criterion
    Following paper "Class-Balanced Loss Based on Effective Number of Samples" (CVPR2019)

    Args:
        gamma (int, optional): modulation factor gamma in focal loss. Defaults to 2.
        alpha (int, optional): modulation factor alpha in focal loss. If a integer, apply to all;
            if a list or array or tensor, regard as alpha for each class; if none, no alpha. Defaults to None.
        weight (Optional[torch.Tensor], optional): weight to each class, !not the same as alpha. Defaults to None.
        size_average (_type_, optional): _description_. Defaults to None.
        reduce (_type_, optional): _description_. Defaults to None.
        reduction (str, optional): _description_. Defaults to "mean".
    """

    def __init__(
        self,
        gamma=2,
        alpha=None,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super(FocalBCEWithLogitLoss, self).__init__(size_average, reduce, reduction)
        self.gamma = gamma
        # a number for all, or a Tensor with the same num_classes as input
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = alpha
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)
        self.weight: Optional[torch.Tensor]
        self.pos_weight: Optional[torch.Tensor]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.repeat(input.shape[0], 1)
            else:
                alpha_t = torch.ones_like(input) * self.alpha
        else:
            alpha_t = None

        ce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        # pt = torch.exp(-ce)
        # modulator = ((1 - pt) ** self.gamma)
        # following author's repo https://github.com/richardaecn/class-balanced-loss/blob/master/src/cifar_main.py#L226-L266
        # explaination https://github.com/richardaecn/class-balanced-loss/issues/1
        # A numerically stable implementation of modulator.
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * target * input - self.gamma * torch.log1p(torch.exp(-input)))
        # focal loss
        fl_loss = modulator * ce
        # alpha
        if alpha_t is not None:
            alpha_t = alpha_t * target + (1 - alpha_t) * (1 - target)
            fl_loss = alpha_t * fl_loss
        # pos weight
        if self.pos_weight is not None:
            fl_loss = self.pos_weight * fl_loss
        # reduction
        if self.reduction == "mean":
            return fl_loss.mean()
        elif self.reduction == "sum":
            return fl_loss.sum()
        else:
            return fl_loss
