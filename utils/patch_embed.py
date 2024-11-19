""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on code in:
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision/tree/main/big_vision

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from utils.format import Format, nchw_to
from utils.helpers import to_2tuple
from utils.trace_utils import _assert

_logger = logging.getLogger(__name__)

class PatchEmbed(nn.Module):
    def __init__(
            self,
            img_size: Optional[int] = (512, 512),
            patch_size: int = (16, 16),
            in_chans: int = 3,
            embed_dim: int = 128,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.img_size = img_size if img_size is not None else None
        self.patch_size = patch_size if patch_size is not None else None
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        if img_size is not None:
            self.img_size = img_size
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)]) # (512, 512) (16, 16)
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW


        # 初始化 `num_patches` 为 None
        self.num_patches = None

        # Patch embedding convolutional layers
        self.patch = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=15, stride=1, bias=bias),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1, bias=bias)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        print(x.shape)
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")

        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Apply convolution layers
        x = self.patch(x)  # B, embed_dim, H', W'

        # Update grid_size and num_patches based on output of patch embeddings
        self.grid_size = x.shape[2:]
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        x = self.layer_norm(x)

        return x




# Need to read about this.
class PatchEmbedWithSize(PatchEmbed):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=flatten,
            output_fmt=output_fmt,
            bias=bias,
        )

    def forward(self, x) -> Tuple[torch.Tensor, List[int]]:
        B, C, H, W = x.shape
        if self.img_size is not None:
            _assert(H % self.patch_size[0] == 0, f"Input image height ({H}) must be divisible by patch size ({self.patch_size[0]}).")
            _assert(W % self.patch_size[1] == 0, f"Input image width ({W}) must be divisible by patch size ({self.patch_size[1]}).")

        x = self.proj(x)
        grid_size = x.shape[-2:]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x, grid_size




    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """