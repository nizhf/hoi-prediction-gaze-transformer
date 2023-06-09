#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import Transformer
import copy
import math


# NOTE Most Modules are modified based on PyTorch implementation and STTran implementation
class TransformerSpatialLayer(nn.Module):
    def __init__(
        self,
        d_model=1936,
        nhead=4,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # local attention
        x = src
        x2, local_attention_weights = self._sa_block(x, src_mask, src_key_padding_mask)
        x = self.norm1(x + x2)
        x = self.norm2(x + self._ff_block(x))

        return x, local_attention_weights

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, attention_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        return self.dropout1(x), attention_weights

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# NOTE Transformer decoder layer, but without self-attention block
class TransformerTemporalLayer(nn.Module):
    def __init__(
        self,
        d_model=1936,
        nhead=4,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
    ):
        super().__init__()

        self.multihead2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # NOTE norm2 should be there, but not in original code
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(
        self,
        global_input: torch.Tensor,
        position_embed: Optional[torch.Tensor] = 0,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = global_input
        x2, global_attention_weights = self._mha_block(
            x, position_embed, memory_mask, tgt_key_padding_mask
        )
        x = self.norm2(x + x2)  # NOTE original code use norm3 here
        x = self.norm3(x + self._ff_block(x))  # NOTE original code has no norm here
        return x, global_attention_weights

    # no self-attention block

    # multihead attention block, different than PyTorch implementation (and vanilla transformer)
    def _mha_block(
        self,
        x: torch.Tensor,
        position_embed: torch.Tensor,
        memory_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, attention_weights = self.multihead2(
            query=x + position_embed,
            key=x + position_embed,
            value=x,
            attn_mask=memory_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        return self.dropout2(x), attention_weights

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


# NOTE almost the same as PyTorch implementation, only plus return attention
class TransformerSpatial(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, src_key_padding_mask):
        output = src
        weights = torch.zeros(
            [self.num_layers, output.shape[1], output.shape[0], output.shape[0]]
        ).to(output.device)

        for i, layer in enumerate(self.layers):
            output, local_attention_weights = layer(
                output, src_key_padding_mask=src_key_padding_mask
            )
            weights[i] = local_attention_weights
        if self.num_layers > 0:
            return output, weights
        else:
            return output, None


# NOTE almost the same as PyTorch implementation, only plus return attention
class TransformerTemporal(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, global_input, position_embed, memory_mask, tgt_key_padding_mask):
        output = global_input
        weights = torch.zeros(
            [self.num_layers, output.shape[1], output.shape[0], output.shape[0]]
        ).to(output.device)

        for i, layer in enumerate(self.layers):
            output, global_attention_weights = layer(
                output, position_embed, memory_mask, tgt_key_padding_mask
            )
            weights[i] = global_attention_weights

        if self.num_layers > 0:
            return output, weights
        else:
            return output, None


class STTranTransformer(nn.Module):
    """Spatial Temporal Transformer
    local_attention: spatial encoder
    global_attention: temporal decoder
    position_embedding: frame encoding (window_size*dim)
    mode: both--use the features from both frames in the window
          latter--use the features from the latter frame in the window
    """

    def __init__(
        self,
        enc_layer_num=1,
        dec_layer_num=3,
        d_model=1992,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        sliding_window=2,
        sinusoidal_encoding=False,
    ):
        super(STTranTransformer, self).__init__()
        self.sliding_window = sliding_window
        self.nhead = nhead
        self.sinusoidal_encoding = sinusoidal_encoding

        encoder_layer = TransformerSpatialLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.spatial_attention = TransformerSpatial(encoder_layer, enc_layer_num)

        decoder_layer = TransformerTemporalLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.temporal_attention = TransformerTemporal(decoder_layer, dec_layer_num)

        # positional encoding for sliding window
        if self.sinusoidal_encoding:
            # sinusoidal encoding
            position_embedding = torch.zeros(self.sliding_window, d_model)
            position = torch.arange(
                0, self.sliding_window, dtype=torch.float
            ).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            position_embedding[:, 0::2] = torch.sin(position * div_term)
            position_embedding[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("position_embedding", position_embedding)
        else:
            self.position_embedding = nn.Embedding(self.sliding_window, d_model)
            nn.init.uniform_(self.position_embedding.weight)

    def forward(self, features, im_idxes, windows, windows_out):
        # the highest pair number in a single frame
        max_pair = torch.sum(im_idxes == torch.mode(im_idxes)[0])
        # total frame number
        num_frame = int(im_idxes[-1] + 1)
        # spatial block input init (Length, Batch (#frame), Feature)
        spatial_input = torch.zeros(
            [max_pair, num_frame, features.shape[1]], device=features.device
        )
        # spatial block padding mask init (Batch (#frame), Length)
        spatial_padding_masks = torch.zeros(
            [num_frame, max_pair], dtype=torch.bool, device=features.device
        )
        # fill spatial block input and mask
        for i in range(num_frame):
            spatial_input[: torch.sum(im_idxes == i), i, :] = features[im_idxes == i]
            spatial_padding_masks[i, torch.sum(im_idxes == i) :] = 1

        # spatial attention
        spatial_output, spatial_attention_weights = self.spatial_attention(
            spatial_input, spatial_padding_masks
        )
        # batch second to batch first
        spatial_output = (
            (spatial_output.permute(1, 0, 2))
            .contiguous()
            .view(-1, features.shape[1])[spatial_padding_masks.view(-1) == 0]
        )

        # known sliding windows
        num_sliding_window = len(windows)
        max_temporal_input = torch.max(torch.sum(windows, dim=1))
        # temporal block input init (Length, Batch #window, Feature)
        temporal_input = torch.zeros(
            [
                max_temporal_input,
                num_sliding_window,
                features.shape[1],
            ],
            device=features.device,
        )
        # temporal block positional embedding init (Length, Batch #window, Feature)
        temporal_position_embed = torch.zeros(
            [
                max_temporal_input,
                num_sliding_window,
                features.shape[1],
            ]
        ).to(features.device)
        temporal_idx = -torch.ones(
            [max_temporal_input, num_sliding_window], dtype=torch.long
        ).to(features.device)
        # temporal block padding mask init (Batch #window, Length)
        temporal_padding_masks = torch.zeros(
            [num_sliding_window, max_temporal_input],
            dtype=torch.bool,
        ).to(features.device)

        # fill everything in each sliding window
        for idx_window, window in enumerate(windows):
            slice_len = torch.sum(window)
            # fill temporal input
            temporal_input[:slice_len, idx_window, :] = spatial_output[window]
            temporal_idx[:slice_len, idx_window] = im_idxes[window]
            # mask padding
            temporal_padding_masks[idx_window, slice_len:] = 1
            # positional encoding
            im_idx_start = temporal_idx[0, idx_window]
            for idx in range(slice_len):
                # copy from positional embedding
                if self.sinusoidal_encoding:
                    temporal_position_embed[
                        idx, idx_window, :
                    ] = self.position_embedding[
                        temporal_idx[idx, idx_window] - im_idx_start
                    ]
                else:
                    temporal_position_embed[
                        idx, idx_window, :
                    ] = self.position_embedding.weight[
                        temporal_idx[idx, idx_window] - im_idx_start
                    ]

        # temporal attention
        temporal_output, temporal_attention_weights = self.temporal_attention(
            temporal_input,
            temporal_position_embed,
            None,
            temporal_padding_masks,
        )

        # output matrix init, (#interactions, d_model)
        output = torch.zeros((len(im_idxes), features.shape[1])).to(features.device)
        output_mask = torch.zeros_like(im_idxes).bool().to(features.device)
        for idx_window, (window, window_out) in enumerate(zip(windows, windows_out)):
            slice_len = torch.sum(window)
            out_len = torch.sum(window_out)
            output[window_out, :] = temporal_output[
                slice_len - out_len : slice_len, idx_window
            ]
            output_mask = output_mask | window_out
        output = output[output_mask]

        return output, temporal_attention_weights, spatial_attention_weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
