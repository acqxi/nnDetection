# nndet/arch/heads/controller.py (最終修正版)
import torch
import torch.nn as nn
from torch import Tensor
from typing import TypeVar
from loguru import logger

# 關鍵：我們不再從 BaseRegressor 繼承，而是直接使用 nn.Module
# 但我們會複製 regressor 的內部建構邏輯，使其行為一致


class Controller(nn.Module):
    """
    CondInst 的控制器頭：預測動態卷積核的參數。
    這個版本直接繼承自 nn.Module，以避免來自父類別的未知副作用。
    """

    def __init__(
        self,
        conv,
        in_channels: int,
        internal_channels: int,
        anchors_per_pos: int,
        num_levels: int,
        num_convs: int = 3,
        add_norm: bool = True,
        num_mask_params: int = 169,  # 會被動態計算
        **kwargs,
    ):
        super().__init__()
        self.dim = conv.dim
        self.in_channels = in_channels
        self.internal_channels = internal_channels
        self.num_convs = num_convs
        self.anchors_per_pos = anchors_per_pos
        self.num_mask_params = num_mask_params

        # --- 邏輯從 BaseRegressor 複製而來 ---
        self.conv_internal = self.build_conv_internal(conv,
                                                      add_norm=add_norm,
                                                      **kwargs)
        self.conv_out = self.build_conv_out(conv)
        self.init_weights()
        # ------------------------------------

        logger.info(
            f"Building Controller Head with {self.num_mask_params} parameters per instance."
        )

    def build_conv_internal(self, conv, **kwargs):
        """
        建立內部卷積層 (邏輯同 BaseRegressor)
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
        建立輸出卷積層，輸出通道數為 mask_params 的數量。
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

    # 關鍵：forward 方法接收單一的 feature_map 和 level，與框架設計一致
    def forward(self, x: Tensor, level: int, **kwargs) -> Tensor:
        """
        對單一 FPN 層級的特徵圖進行前向傳播。
        """
        mask_params = self.conv_out(self.conv_internal(x))

        # 調整維度以匹配輸出格式
        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        mask_params = mask_params.permute(*axes)
        mask_params = mask_params.contiguous()

        # Reshape to [batch_size, num_anchors_at_level, num_mask_params]
        mask_params = mask_params.view(x.size()[0], -1, self.num_mask_params)
        return mask_params

    def init_weights(self) -> None:
        """
        初始化權重 (邏輯同 BaseRegressor)
        """
        logger.info("Overwriting controller conv weight init")
        CONV_TYPES = (nn.Conv2d, nn.Conv3d)
        for layer in self.modules():
            if isinstance(layer, CONV_TYPES):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

    def compute_loss(self, *args, **kwargs):
        """ Controller 本身沒有獨立的 loss """
        pass


ControllerType = TypeVar('ControllerType', bound=Controller)
