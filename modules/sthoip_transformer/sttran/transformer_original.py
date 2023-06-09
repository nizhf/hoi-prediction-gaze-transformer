#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import Transformer
import copy

# NOTE The same implementation as in STTran "Spatial-Temporal Transformer for Dynamic Scene Graph Generation"
class TransformerEncoderLayer(nn.Module):
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
    ):
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
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
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
class TransformerDecoderLayer(nn.Module):
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
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, global_input, tgt_key_padding_mask, position_embed):
        x = global_input
        x2, global_attention_weights = self._mha_block(
            x, position_embed, tgt_key_padding_mask
        )
        x = self.norm3(x + x2)  # NOTE original code use norm3 here
        x = x + self._ff_block(x)  # NOTE original code has no norm here
        return x, global_attention_weights

    # no self-attention block

    # multihead attention block, different than PyTorch implementation (and vanilla transformer)
    def _mha_block(
        self, x: torch.Tensor, position_embed: torch.Tensor, key_padding_mask
    ) -> torch.Tensor:
        x, attention_weights = self.multihead2(
            query=x + position_embed,
            key=x + position_embed,
            value=x,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        return self.dropout2(x), attention_weights

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


# NOTE almost the same as PyTorch implementation, only plus return attention
class TransformerEncoder(nn.Module):
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
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, global_input, tgt_key_padding_mask, position_embed):
        output = global_input
        weights = torch.zeros(
            [self.num_layers, output.shape[1], output.shape[0], output.shape[0]]
        ).to(output.device)

        for i, layer in enumerate(self.layers):
            output, global_attention_weights = layer(
                output, tgt_key_padding_mask, position_embed
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
        d_model=1936,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        mode=None,
    ):
        super(STTranTransformer, self).__init__()
        self.mode = mode

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.local_attention = TransformerEncoder(encoder_layer, enc_layer_num)

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.global_attention = TransformerDecoder(decoder_layer, dec_layer_num)

        self.position_embedding = nn.Embedding(2, d_model)  # present and next frame
        nn.init.uniform_(self.position_embedding.weight)

    def forward(self, features, im_idx):
        rel_idx = torch.arange(im_idx.shape[0])
        # the highest pair number in a single frame
        max_pair = torch.sum(im_idx == torch.mode(im_idx)[0])
        # total frame number
        num_frame = int(im_idx[-1] + 1)
        # encoder input
        rel_input = torch.zeros([max_pair, num_frame, features.shape[1]]).to(
            features.device
        )
        # encoder mask
        masks = torch.zeros([num_frame, max_pair], dtype=torch.bool).to(features.device)
        # TODO Padding/Mask maybe don't need for-loop
        for i in range(num_frame):
            rel_input[: torch.sum(im_idx == i), i, :] = features[im_idx == i]
            masks[i, torch.sum(im_idx == i) :] = 1

        # spatial encoder
        local_output, local_attention_weights = self.local_attention(rel_input, masks)
        local_output = (
            (local_output.permute(1, 0, 2))
            .contiguous()
            .view(-1, features.shape[1])[masks.view(-1) == 0]
        )

        # decoder input
        global_input = torch.zeros([max_pair * 2, num_frame - 1, features.shape[1]]).to(
            features.device
        )
        # positional embedding
        position_embed = torch.zeros(
            [max_pair * 2, num_frame - 1, features.shape[1]]
        ).to(features.device)
        idx = -torch.ones([max_pair * 2, num_frame - 1]).to(features.device)
        idx_plus = -torch.ones([max_pair * 2, num_frame - 1], dtype=torch.long).to(
            features.device
        )  # TODO
        # decoder mask
        global_masks = torch.zeros([num_frame - 1, max_pair * 2], dtype=torch.bool).to(
            features.device
        )

        # original code
        # sliding window size = 2
        for j in range(num_frame - 1):
            global_input[
                : torch.sum((im_idx == j) + (im_idx == j + 1)), j, :
            ] = local_output[(im_idx == j) + (im_idx == j + 1)]
            idx[: torch.sum((im_idx == j) + (im_idx == j + 1)), j] = im_idx[
                (im_idx == j) + (im_idx == j + 1)
            ]
            idx_plus[: torch.sum((im_idx == j) + (im_idx == j + 1)), j] = rel_idx[
                (im_idx == j) + (im_idx == j + 1)
            ]  # TODO

            position_embed[
                : torch.sum(im_idx == j), j, :
            ] = self.position_embedding.weight[0]
            position_embed[
                torch.sum(im_idx == j) : torch.sum(im_idx == j)
                + torch.sum(im_idx == j + 1),
                j,
                :,
            ] = self.position_embedding.weight[1]
        global_masks = (
            (torch.sum(global_input.view(-1, features.shape[1]), dim=1) == 0)
            .view(max_pair * 2, num_frame - 1)
            .permute(1, 0)
        )

        # temporal decoder
        global_output, global_attention_weights = self.global_attention(
            global_input, global_masks, position_embed
        )

        output = torch.zeros_like(features)

        if self.mode == "both":
            # both
            for j in range(num_frame - 1):
                if j == 0:
                    output[im_idx == j] = global_output[:, j][idx[:, j] == j]

                if j == num_frame - 2:
                    output[im_idx == j + 1] = global_output[:, j][idx[:, j] == j + 1]
                else:
                    output[im_idx == j + 1] = (
                        global_output[:, j][idx[:, j] == j + 1]
                        + global_output[:, j + 1][idx[:, j + 1] == j + 1]
                    ) / 2

        elif self.mode == "latter":
            # later
            for j in range(num_frame - 1):
                if j == 0:
                    output[im_idx == j] = global_output[:, j][idx[:, j] == j]

                output[im_idx == j + 1] = global_output[:, j][idx[:, j] == j + 1]

        return output, global_attention_weights, local_attention_weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
