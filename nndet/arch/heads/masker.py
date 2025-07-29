# nndet/arch/heads/masker.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import TypeVar, List, Dict, Optional

from nndet.arch.heads.comb import AbstractHead


class MaskBCELoss(nn.Module):

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred_logits: Tensor, targets: Tensor) -> Tensor:
        return self.loss_weight * self.loss_fn(pred_logits, targets)


class MaskHead(AbstractHead):

    def __init__(
            self,
            conv,
            in_channels: int,
            num_classes: int,  # 前景類別數量
            anchors_per_pos: int,
            num_convs: int = 4,
            internal_channels: int = 256,
            mask_size: int = 28,  # 預測的遮罩大小
    ):
        """
        A simple Mask Head for instance segmentation.
        It predicts a class-agnostic mask for each anchor.
        """
        super().__init__()
        self.dim = conv.dim
        self.mask_size = mask_size
        self.num_classes = num_classes
        self.anchors_per_pos = anchors_per_pos

        _conv_internal = nn.Sequential()
        _in = in_channels
        for i in range(num_convs):
            _conv_internal.add_module(name=f"mask_conv{i}",
                                      module=conv(
                                          _in,
                                          internal_channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                      ))
            _in = internal_channels
        self.conv_internal = _conv_internal

        out_channels = anchors_per_pos

        self.conv_out = conv(
            internal_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            add_norm=False,
            add_act=False,
            bias=True,
        )

        self.loss = MaskBCELoss(loss_weight=1.0)

    def forward(self, x: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        x: 來自 FPN 的多層級特徵圖
        """
        outputs = []
        for p in x:
            feat = self.conv_internal(p)
            feat = self.conv_out(
                feat)  # [N, A, H, W, D] where A = anchors_per_pos

            # 使用雙線性插值將特徵圖縮放至 mask_size
            if self.dim == 2:
                # [N, A, H, W] -> [N, A, mask_size, mask_size]
                feat = F.interpolate(feat,
                                     size=(self.mask_size, self.mask_size),
                                     mode='bilinear',
                                     align_corners=False)
                # [N*A, mask_size, mask_size] -> [N*A, mask_size^2]
                feat = feat.view(feat.size(0) * feat.size(1), -1)
            else:  # dim == 3
                # [N, A, H, W, D] -> [N, A, mask_size, mask_size, mask_size]
                feat = F.interpolate(feat,
                                     size=(self.mask_size, self.mask_size,
                                           self.mask_size),
                                     mode='trilinear',
                                     align_corners=False)
                # [N*A, mask_size, mask_size, mask_size] -> [N*A, mask_size^3]
                feat = feat.view(feat.size(0) * feat.size(1), -1)

            outputs.append(feat)

        mask_logits = torch.cat(outputs,
                                dim=0)  # [Num_Anchors_Total, mask_size^dim]

        return {"mask_logits": mask_logits}

    # compute_loss 和 postprocess_for_inference 會在主模型中實現
    def compute_loss(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            "This method should be implemented in the main model")

    def postprocess_for_inference(self, *args,
                                  **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            "This method should be implemented in the main model")


MaskerType = TypeVar('MaskerType', bound=MaskHead)
