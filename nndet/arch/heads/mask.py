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
from torch import Tensor
from typing import List

from nndet.losses.segmentation import SoftDiceLoss


class DynamicMaskHead(nn.Module):
    """
    Dynamic Mask Head for CondInst: generates instance masks using dynamically generated convolution kernels
    """
    def __init__(self,
                 dim: int = 3,
                 num_dynamic_convs: int = 3,
                 in_channels: int = 32,  # Should match FPN output channels
                 internal_channels: int = 8,
                 ):
        """
        Initialize Dynamic Mask Head
        
        Args:
            dim: number of spatial dimensions (2 or 3)
            num_dynamic_convs: number of dynamic convolution layers
            in_channels: number of input channels from FPN
            internal_channels: number of internal channels for dynamic convs
        """
        super().__init__()
        self.dim = dim
        self.num_dynamic_convs = num_dynamic_convs
        self.in_channels = in_channels
        self.internal_channels = internal_channels
        self.conv_op = nn.Conv3d if dim == 3 else nn.Conv2d

        # Calculate total parameters needed for each dynamic convolution layer
        self.param_splits = []
        kernel_size = 3
        
        # Layer 1: in_channels -> internal_channels
        if dim == 3:
            w1 = self.in_channels * self.internal_channels * kernel_size * kernel_size * kernel_size
        else:
            w1 = self.in_channels * self.internal_channels * kernel_size * kernel_size
        b1 = self.internal_channels
        self.param_splits.append(w1 + b1)
        
        # Layer 2: internal_channels -> internal_channels  
        if dim == 3:
            w2 = self.internal_channels * self.internal_channels * kernel_size * kernel_size * kernel_size
        else:
            w2 = self.internal_channels * self.internal_channels * kernel_size * kernel_size
        b2 = self.internal_channels
        self.param_splits.append(w2 + b2)
        
        # Layer 3: internal_channels -> 1 (mask output)
        if dim == 3:
            w3 = self.internal_channels * 1 * 1 * 1 * 1
        else:
            w3 = self.internal_channels * 1 * 1 * 1
        b3 = 1
        self.param_splits.append(w3 + b3)

        self.num_mask_params = sum(self.param_splits)
        self.loss = SoftDiceLoss(nonlin=torch.nn.Sigmoid(), batch_dice=True, do_bg=True)

    def forward(self, features: Tensor, mask_params: Tensor) -> Tensor:
        """
        Generate instance masks using dynamic convolution parameters
        
        Args:
            features: feature map from FPN (e.g., P3), [N, C, H, W, (D)]
            mask_params: dynamic parameters from Controller, [num_instances, num_params]
            
        Returns:
            Tensor: predicted instance masks [num_instances, 1, H, W, (D)]
        """
        num_instances = mask_params.size(0)
        if num_instances == 0:
            if self.dim == 3:
                return torch.empty(0, 1, *features.shape[2:], device=features.device)
            else:
                return torch.empty(0, 1, *features.shape[2:], device=features.device)

        # Split parameters for each dynamic convolution layer
        params_split = mask_params.split(self.param_splits, dim=-1)

        # Use the first image in the batch for mask generation (this is a simplification)
        # In practice, you would need to associate each instance with its corresponding image
        features_single = features[0:1]  # Take first image [1, C, H, W, (D)]
        
        # Expand features for all instances: [num_instances, C, H, W, (D)]
        if self.dim == 3:
            x = features_single.expand(num_instances, -1, -1, -1, -1)
        else:
            x = features_single.expand(num_instances, -1, -1, -1)

        # Dynamic convolution layers
        # Layer 1: in_channels -> internal_channels
        w1, b1 = params_split[0][:, :-self.internal_channels], params_split[0][:, -self.internal_channels:]
        x = self.dynamic_conv(x, w1, b1, kernel_size=3, padding=1)
        x = F.relu(x)

        # Layer 2: internal_channels -> internal_channels
        w2, b2 = params_split[1][:, :-self.internal_channels], params_split[1][:, -self.internal_channels:]
        x = self.dynamic_conv(x, w2, b2, kernel_size=3, padding=1)
        x = F.relu(x)

        # Layer 3: internal_channels -> 1 (final mask output)
        w3, b3 = params_split[2][:, :-1], params_split[2][:, -1:]
        x = self.dynamic_conv(x, w3, b3, kernel_size=1, padding=0)

        return x.sigmoid()  # Output predicted masks with sigmoid activation

    def dynamic_conv(self, features, weights, biases, kernel_size=3, padding=1, **kwargs):
        """
        Perform dynamic convolution with instance-specific kernels
        
        Args:
            features: input features [N_inst, C_in, H, W, (D)]
            weights: convolution weights [N_inst, C_out * C_in * K*K*(K)]
            biases: convolution biases [N_inst, C_out]
            kernel_size: convolution kernel size
            padding: convolution padding
            **kwargs: additional convolution parameters
            
        Returns:
            Tensor: convolution output [N_inst, C_out, H, W, (D)]
        """
        n_inst, c_in = features.shape[:2]
        c_out = biases.shape[1]

        # Reshape weights for grouped convolution
        if self.dim == 3:
            weights = weights.reshape(n_inst * c_out, c_in, kernel_size, kernel_size, kernel_size)
            features = features.reshape(1, n_inst * c_in, *features.shape[2:])
            
            # Grouped 3D convolution
            conv_output = F.conv3d(
                features, 
                weights, 
                bias=None, 
                stride=1, 
                padding=padding, 
                dilation=1, 
                groups=n_inst
            )
            
            # Reshape and add bias
            conv_output = conv_output.reshape(n_inst, c_out, *conv_output.shape[2:])
            conv_output = conv_output + biases.reshape(n_inst, -1, 1, 1, 1)
        else:
            weights = weights.reshape(n_inst * c_out, c_in, kernel_size, kernel_size)
            features = features.reshape(1, n_inst * c_in, *features.shape[2:])
            
            # Grouped 2D convolution
            conv_output = F.conv2d(
                features, 
                weights, 
                bias=None, 
                stride=1, 
                padding=padding, 
                dilation=1, 
                groups=n_inst
            )
            
            # Reshape and add bias
            conv_output = conv_output.reshape(n_inst, c_out, *conv_output.shape[2:])
            conv_output = conv_output + biases.reshape(n_inst, -1, 1, 1)
            
        return conv_output

    def compute_loss(self, pred_masks, target_masks):
        """
        Compute mask prediction loss
        
        Args:
            pred_masks: predicted masks
            target_masks: ground truth masks
            
        Returns:
            Tensor: computed loss
        """
        return self.loss(pred_masks, target_masks)
