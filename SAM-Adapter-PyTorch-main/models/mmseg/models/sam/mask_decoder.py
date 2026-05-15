# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d

# add:
from model.mlp.sMLP_block import sMLPBlock


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2), # 转置卷积层，用于将输入特征图的尺寸扩大;。通过指定 kernel_size=2 和 stride=2，实现了对输入特征图的每个像素进行两倍的上采样。
            LayerNorm2d(transformer_dim // 4), #  2D 归一化层，用于规范化上一层的输出。
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
    '''
    forward：可以根据输入的 prompt 学习掩码生成的高度非线性映射,为 prompt 驱动生成模型提供掩码预测的关键能力。
    提供根据 prompt 预测掩码的具体实现。它发挥了 MaskDecoder 类的强大功能,
    可以解码出复杂的定制化掩码,为实现高质量的 prompt 驱动生成模型提供强有力的支持。
    '''
    def forward(  # 作用：根据图像和prompt的embedding预测掩码
        self,
        image_embeddings: torch.Tensor,     # 图像编码器的输出
        image_pe: torch.Tensor,             # 形状相同的位置编码
        sparse_prompt_embeddings: torch.Tensor,    # 点和框的embedding
        dense_prompt_embeddings: torch.Tensor,     # 掩码输入的embedding
        multimask_output: bool,                    # 是否返回多个掩码或单个掩码
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """

        # 一组可学习的 output tokens
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for outptu
        if multimask_output:     # decoder的输入
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)    # 拼接iou_token和mask_tokens作为输出tokens
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)  # 拓展至batch大小
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)   # 注意 decoder会有两个地方的输入|与sparse embedding拼接作为tokens

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) # 将数组中的元素重复多次，并在重复的过程中保持它们的顺序
        src = src + dense_prompt_embeddings     # 可见，src包含了image_embeddings，tokens， dense embedding的信息
        # dense prompt由于包含密集的空间信息，与image embedding所在的特征空间一致性更高，所以直接与image embedding相加融合。?
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)    # positional embedding # 位置编码
        b, c, h, w = src.shape 

        # '''
        # src --> mlps => src_post
        # src += src_post 
        # etc


   
        # Run the transformer
        # 进入一个两层transformer结构的decoder做融合
        hs, src = self.transformer(src, pos_src, tokens) # +MLPS... # 将 src 和 pos_src 以及 tokens 输入 transformer, 获得 hs 和 src
        # iou_token_out & mask_tokens_out是凭空出现的可学习的
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src) # convolution
        hyper_in_list: List[torch.Tenmsor] = []
        for i in range(self.num_mask_tokens):           # 右上的MLP（某个图）
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])) # 又有做mlp
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # ↑mask token 被从tokens中分离出来

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out) # iou_prediction_head： mlp

        return masks, iou_pred




# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):  # 定义一个可以配置隐藏层层数和激活函数的多层感知机模型，并根据输入数据进行前向传播。
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
