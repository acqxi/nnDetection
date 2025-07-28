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
from torch import Tensor
from typing import TypeVar, List
import logging

logger = logging.getLogger(__name__)


class Controller(nn.Module):
    """
    Controller Head for CondInst: predicts parameters for dynamic convolution kernels.
    """
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 num_mask_params: int = 169,  # Total parameters for F_1, F_2, F_3
                 **kwargs,
                 ):
        """
        Initialize Controller Head
        
        Args:
            conv: Convolution modules which handles a single layer
            in_channels: number of input channels
            internal_channels: number of channels internally used
            anchors_per_pos: number of anchors per position
            num_levels: number of decoder levels
            num_convs: number of convolutions
            add_norm: en-/disable normalization layers in internal layers
            num_mask_params: number of parameters for dynamic mask generation
            **kwargs: additional keyword arguments
        """
        super().__init__()
        self.dim = conv.dim
        self.num_levels = num_levels
        self.num_convs = num_convs
        self.num_mask_params = num_mask_params
        self.anchors_per_pos = anchors_per_pos
        self.in_channels = in_channels
        self.internal_channels = internal_channels

        self.conv_internal = self.build_conv_internal(conv, add_norm=add_norm, **kwargs)
        self.conv_out = self.build_conv_out(conv)
        
        self.init_weights()
        logger.info(f"Building Controller Head with {self.num_mask_params} parameters per instance.")

    def build_conv_internal(self, conv, **kwargs):
        """
        Build internal convolutions
        """
        _conv_internal = nn.Sequential()
        _conv_internal.add_module(
            name="c_in",
            module=conv(
                self.in_channels,
                self.internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **kwargs,
            ))
        for i in range(self.num_convs):
            _conv_internal.add_module(
                name=f"c_internal{i}",
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
        Build final convolutions for mask parameters output
        """
        out_channels = self.anchors_per_pos * self.num_mask_params
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

    def forward(self, fmaps: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass to predict mask parameters for each anchor at each level
        
        Args:
            fmaps: list of feature maps from each pyramid level
            
        Returns:
            List[torch.Tensor]: predicted mask parameters for each level
                Each tensor has shape [batch_size, num_anchors_at_level, num_mask_params]
        """
        mask_params_all = []
        for level, feature_map in enumerate(fmaps):
            mask_params = self.conv_out(self.conv_internal(feature_map))
            
            # Permute dimensions to get [batch, height, width, depth, channels] format
            axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
            mask_params = mask_params.permute(*axes)
            mask_params = mask_params.contiguous()
            
            # Reshape to [batch_size, num_anchors_at_level, num_mask_params]
            mask_params = mask_params.view(feature_map.size()[0], -1, self.num_mask_params)
            mask_params_all.append(mask_params)
            
        return mask_params_all

    def init_weights(self) -> None:
        """
        Init weights with normal distribution (mean=0, std=0.01)
        """
        logger.info("Overwriting controller conv weight init")
        CONV_TYPES = (nn.Conv2d, nn.Conv3d)
        for layer in self.modules():
            if isinstance(layer, CONV_TYPES):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

    def compute_loss(self, *args, **kwargs):
        """
        Controller itself doesn't have independent loss - its gradients come from Mask Head
        """
        pass


ControllerType = TypeVar('ControllerType', bound=Controller)
