"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Callable, TypeVar
from abc import abstractmethod

from loguru import logger

from nndet.arch.layers.scale import Scale
from torch import Tensor

CONV_TYPES = (nn.Conv2d, nn.Conv3d)


class InstanceSegmenter(nn.Module):

    @abstractmethod
    def compute_loss(self, pred_masks: Tensor, target_masks: Tensor,
                     **kwargs) -> Tensor:
        """
        Compute instance segmentation loss (BCE loss)

        Args:
            pred_masks (Tensor): predicted instance masks [N, 7*28*28]
            target_masks (Tensor): target instance masks [N, 7*28*28]

        Returns:
            Tensor: loss
        """
        raise NotImplementedError


class BaseInstanceSegmenter(InstanceSegmenter):

    def __init__(
            self,
            conv: Callable,
            in_channels: int,
            internal_channels: int,
            anchors_per_pos: int,
            num_levels: int,
            num_convs: int = 4,
            learn_scale: bool = True,
            output_size: Tuple[int, int, int] = (7, 28, 28),
            **kwargs,
    ):
        """
        Base Instance Segmenter Head
        
        Args:
            conv: conv generator
            in_channels: number of input channels
            internal_channels: number of internal channels
            anchors_per_pos: number of anchors per position
            num_levels: number of FPN levels
            num_convs: number of convolutions
            learn_scale: learn additional scale for each level
            output_size: output mask size (z, y, x)
        """
        super().__init__()
        self.in_channels = in_channels
        self.internal_channels = internal_channels
        self.anchors_per_pos = anchors_per_pos
        self.num_levels = num_levels
        self.num_convs = num_convs
        self.learn_scale = learn_scale
        self.output_size = output_size
        self.mask_size = output_size[0] * output_size[1] * output_size[
            2]  # 7 * 28 * 28

        self.conv_internal = self.build_conv_internal(conv, **kwargs)
        self.conv_out = self.build_conv_out(conv)

        if self.learn_scale:
            self.scales = self.build_scales()

        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.init_weights()

    def build_conv_internal(self, conv, **kwargs):
        """
        Build internal convolutions
        """
        _conv_internal = nn.Sequential()
        _conv_internal.add_module(name="c_in",
                                  module=conv(
                                      self.in_channels,
                                      self.internal_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      **kwargs,
                                  ))
        for i in range(self.num_convs):
            _conv_internal.add_module(name=f"c_internal{i}",
                                      module=conv(
                                          self.internal_channels,
                                          self.internal_channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          **kwargs,
                                      ))
        return _conv_internal

    def build_conv_out(self, conv):
        """
        Build final convolutions
        """
        out_channels = self.anchors_per_pos * self.mask_size  # anchors_per_pos * (7 * 28 * 28)
        return conv(
            self.internal_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            add_norm=False,
            add_act=False,
            bias=True,
        )

    def build_scales(self) -> nn.ModuleList:
        """
        Build additional scalar values per level
        """
        logger.info("Learning level specific scalar in instance segmenter")
        return nn.ModuleList([Scale() for _ in range(self.num_levels)])

    def forward(self, x: torch.Tensor, level: int, **kwargs) -> torch.Tensor:
        """
        Forward input

        Args:
            x: input feature map of size [N x C x Y x X x Z]
            level: FPN level

        Returns:
            torch.Tensor: instance mask logits for each anchor
                [N, n_anchors, 7*28*28]
        """
        mask_logits = self.conv_out(self.conv_internal(x))

        if self.learn_scale:
            mask_logits = self.scales[level](mask_logits)

        # Permute dimensions to get proper ordering
        axes = (0, 2, 3, 1) if x.ndim == 4 else (0, 2, 3, 4, 1)  # 2D or 3D
        mask_logits = mask_logits.permute(*axes)
        mask_logits = mask_logits.contiguous()
        mask_logits = mask_logits.view(x.size()[0], -1, self.mask_size)
        return mask_logits

    def compute_loss(
        self,
        pred_masks: Tensor,
        target_masks: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Compute instance segmentation loss (BCE loss)

        Args:
            pred_masks: predicted instance masks [N, 7*28*28]
            target_masks: target instance masks [N, 7*28*28]

        Returns:
            Tensor: loss
        """
        return self.loss(pred_masks, target_masks, **kwargs)

    def init_weights(self) -> None:
        """
        Init weights with normal distribution (mean=0, std=0.01)
        """
        logger.info("Overwriting instance segmenter conv weight init")
        for layer in self.modules():
            if isinstance(layer, CONV_TYPES):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)


# Type alias
InstanceSegmenterType = TypeVar("InstanceSegmenterType",
                                bound=InstanceSegmenter)
