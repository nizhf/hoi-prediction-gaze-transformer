#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import torch
import torch.nn as nn
from .mha_utils import MultiheadAttentionStable
import copy
import math


# Transformer encoder layer with only self-attention
class SelfAttentionEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=1736,
        nhead=8,
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
        x2 = self._sa_block(x, src_mask, src_key_padding_mask)
        x = self.norm1(x + x2)
        x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# Transformer encoder layer with cross-attention and optional self-attention
class CrossAttentionEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=1736,
        d_kv=256,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
        q_sa=False,
        q_ffn=False,
        kv_sa=True,
        kv_ffn=False,
    ):
        super().__init__()
        self.nhead = nhead
        self.q_sa = q_sa
        self.q_ffn = q_ffn
        self.kv_sa = kv_sa
        self.kv_ffn = kv_ffn

        # self attention key & value
        if kv_sa:
            self.self_attn_kv = nn.MultiheadAttention(d_kv, nhead, dropout=dropout)
            self.dropout_kv = nn.Dropout(dropout)
            self.norm_kv = nn.LayerNorm(d_kv)
            # key & value FFN if required
            if self.kv_ffn:
                self.linear1_kv = nn.Linear(d_kv, dim_feedforward)
                self.dropout_kv_l1 = nn.Dropout(dropout)
                self.linear2_kv = nn.Linear(dim_feedforward, d_kv)
                self.dropout_kv_l2 = nn.Dropout(dropout)
                self.norm_kv_ffn = nn.LayerNorm(d_kv)

        # query may come from a SA layer, so no SA needed here
        if q_sa:
            # self attention query
            self.self_attn_q = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.dropout_q = nn.Dropout(dropout)
            self.norm_q = nn.LayerNorm(d_model)
            # query FFN if required
            if self.q_ffn:
                self.linear1_q = nn.Linear(d_model, dim_feedforward)
                self.dropout_q_l1 = nn.Dropout(dropout)
                self.linear2_q = nn.Linear(dim_feedforward, d_model)
                self.dropout_q_l2 = nn.Dropout(dropout)
                self.norm_q_ffn = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MultiheadAttentionStable(d_model, nhead, dropout=dropout, kdim=d_kv, vdim=d_kv)
        self.dropout_ca = nn.Dropout(dropout)
        self.norm_ca = nn.LayerNorm(d_model)
        # cross attention FFN
        self.linear1_ca = nn.Linear(d_model, dim_feedforward)
        self.dropout_ca_l1 = nn.Dropout(dropout)
        self.linear2_ca = nn.Linear(dim_feedforward, d_model)
        self.dropout_ca_l2 = nn.Dropout(dropout)
        self.norm_ca_ffn = nn.LayerNorm(d_model)

        self.activation = activation

    def forward(self, q, kv, q_padding_mask, kv_padding_mask):
        """

        Args:
            q (torch.Tensor): shape (L, N, qdim)
            kv (torch.Tensor): shape (S, N, kvdim)
            q_padding_mask (torch.Tensor): shape (N, L)
            kv_padding_mask (torch.Tensor): shape (N, S)

        Returns:
            _type_: _description_
        """
        if self.kv_sa:
            # key & value self-attention
            kv_sa, _ = self.self_attn_kv(
                kv, kv, kv, attn_mask=None, key_padding_mask=kv_padding_mask, need_weights=False
            )
            kv_sa = self.dropout_kv(kv_sa)
            kv = self.norm_kv(kv + kv_sa)
            if self.kv_ffn:
                kv_ffn = self.linear1_kv(kv)
                kv_ffn = self.activation(kv_ffn)
                kv_ffn = self.dropout_kv_l1(kv_ffn)
                kv_ffn = self.linear2_kv(kv_ffn)
                kv_ffn = self.dropout_kv_l2(kv_ffn)
                kv = self.norm_kv_ffn(kv + kv_ffn)

        if self.q_sa:
            # query self-attention
            q_sa, _ = self.self_attn_q(q, q, q, attn_mask=None, key_padding_mask=q_padding_mask, need_weights=False)
            q_sa = self.dropout_q(q_sa)
            q = self.norm_q(q + q_sa)
            if self.q_ffn:
                q_ffn = self.linear1_q(q)
                q_ffn = self.activation(q_ffn)
                q_ffn = self.dropout_q_l1(q_ffn)
                q_ffn = self.linear2_q(q_ffn)
                q_ffn = self.dropout_q_l2(q_ffn)
                q = self.norm_q_ffn(q + q_ffn)

        # cross-attention
        # expand q_padding_mask (N, L) to attn_mask (Nxnhead, L, S)
        attn_mask = (
            q_padding_mask.unsqueeze(2).repeat(1, 1, kv_padding_mask.shape[1]).repeat_interleave(self.nhead, dim=0)
        )
        # key_padding_mask (N, S) same as function input
        q_ca, cross_attn_weights = self.cross_attn(
            q, kv, kv, attn_mask=attn_mask, key_padding_mask=kv_padding_mask, need_weights=True
        )
        q_ca = self.dropout_ca(q_ca)
        q = self.norm_ca(q + q_ca)

        q_ca_ffn = self.linear1_ca(q)
        q_ca_ffn = self.activation(q_ca_ffn)
        q_ca_ffn = self.dropout_ca_l1(q_ca_ffn)
        q_ca_ffn = self.linear2_ca(q_ca_ffn)
        q_ca_ffn = self.dropout_ca_l2(q_ca_ffn)
        q = self.norm_ca_ffn(q + q_ca_ffn)

        return q, kv, cross_attn_weights


# Spatial encoder, for all pairs in one frame
class SpatialEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    # spatial encoder no positional encoding
    def forward(self, src, src_key_padding_mask):
        x = src
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


# Temporal encoder, only self-attention for sliding window of pairs
class TemporalEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, dropout=0.1):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
            self.use_dropout = True
        else:
            self.use_dropout = False

    def forward(self, src, pos_encoding, src_key_padding_mask, dropout=True):
        if self.num_layers == 0:
            return src
        if dropout and self.use_dropout:
            x = self.dropout(src + pos_encoding)
        else:
            x = src + pos_encoding
        # add positional encoding for temporal encoder
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


# Cross-attention encoder
class CrossEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, dropout=0.1):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        if dropout > 0:
            self.dropout_q = nn.Dropout(p=dropout)
            self.dropout_kv = nn.Dropout(p=dropout)
            self.use_dropout = True
        else:
            self.use_dropout = False

    def forward(self, q, kv, pos_encoding_q, pos_encoding_kv, q_padding_mask, kv_padding_mask, dropout=True):
        if dropout and self.use_dropout:
            q = self.dropout_q(q + pos_encoding_q)
            kv = self.dropout_kv(kv + pos_encoding_kv)
        else:
            q = q + pos_encoding_q
            kv = kv + pos_encoding_kv

        cross_attn_weights = torch.zeros((self.num_layers, q.shape[1], q.shape[0], kv.shape[0])).to(q.device)

        for i, layer in enumerate(self.layers):
            # kv self-attention every layer
            q, kv, ca = layer(q, kv, q_padding_mask, kv_padding_mask)
            cross_attn_weights[i] = ca

        if self.num_layers > 0:
            return q, kv, cross_attn_weights
        else:
            return q, kv, None


class STTranTransformerGaze(nn.Module):
    """Spatial Temporal Transformer
    local_attention: spatial encoder
    global_attention: temporal decoder
    position_embedding: frame encoding (window_size*dim)

    """

    def __init__(
        self,
        spatial_layer_num=1,
        cross_layer_num=1,
        temporal_layer_num=2,
        d_model=1736,
        d_cross=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        sliding_window=2,
        kv_sa=True,
        kv_ffn=False,
        global_token=False,
        sinusoidal_encoding=False,
    ):
        super(STTranTransformerGaze, self).__init__()
        self.sliding_window = sliding_window
        self.d_model = d_model
        self.d_cross = d_cross
        self.nhead = nhead
        self.global_token = global_token
        self.cross_layer_num = cross_layer_num
        self.sinusoidal_encoding = sinusoidal_encoding

        # cross attention key&value concatenation with global feature
        if self.global_token:
            self.d_cross += self.d_model

        self_attn_layer = SelfAttentionEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.spatial_self_attention = SpatialEncoder(self_attn_layer, spatial_layer_num)
        # no dropout for pos encoding
        self.temporal_self_attention = TemporalEncoder(self_attn_layer, temporal_layer_num, dropout=0)

        cross_attn_layer = CrossAttentionEncoderLayer(
            d_model=self.d_model,
            d_kv=self.d_cross,
            nhead=self.nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            q_sa=False,
            q_ffn=False,
            kv_sa=kv_sa,
            kv_ffn=kv_ffn,
        )
        if self.cross_layer_num > 1:
            # first CA layer, query no self-attention, dropout for pos encoding
            self.temporal_cross_attention_first = CrossEncoder(cross_attn_layer, 1, dropout=dropout)
            # further CA layers, query with self-attention
            cross_attn_layer_further = CrossAttentionEncoderLayer(
                d_model=self.d_model,
                d_kv=self.d_cross,
                nhead=self.nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                q_sa=True,
                q_ffn=False,
                kv_sa=kv_sa,
                kv_ffn=kv_ffn,
            )
            # not adding pos encoding, so no dropout
            self.temporal_cross_attention_further = CrossEncoder(
                cross_attn_layer_further, cross_layer_num - 1, dropout=0
            )
        else:
            # no dropout for pos encoding
            self.temporal_cross_attention = CrossEncoder(cross_attn_layer, cross_layer_num, dropout=0)

        # positional encoding for sliding window
        if self.sinusoidal_encoding:
            # sinusoidal encoding
            pos_encoding_query = torch.zeros(self.sliding_window, self.d_model)
            position = torch.arange(0, self.sliding_window, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
            pos_encoding_query[:, 0::2] = torch.sin(position * div_term)
            pos_encoding_query[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pos_encoding_query", pos_encoding_query)

            pos_encoding_cross = torch.zeros(self.sliding_window, self.d_cross)
            div_term = torch.exp(torch.arange(0, self.d_cross, 2).float() * (-math.log(10000.0) / self.d_cross))
            pos_encoding_cross[:, 0::2] = torch.sin(position * div_term)
            pos_encoding_cross[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pos_encoding_cross", pos_encoding_cross)
        else:
            # learned encoding
            self.pos_encoding_query = nn.Embedding(self.sliding_window, self.d_model)
            nn.init.uniform_(self.pos_encoding_query.weight)
            self.pos_encoding_cross = nn.Embedding(self.sliding_window, self.d_cross)
            nn.init.uniform_(self.pos_encoding_cross.weight)

        # learned global dummy token
        if self.global_token:
            self.global_token_embed = nn.Embedding(1, d_model)
            nn.init.uniform_(self.global_token_embed.weight)

    def forward(self, features, gaze_features, im_idxes, windows, windows_out):
        device = features.device
        d_gaze = gaze_features[0].shape[-1]
        ### Spatial self-attention ###
        # the highest pair number in a single frame
        max_pair = torch.sum(im_idxes == torch.mode(im_idxes)[0])
        # total frame number
        num_frame = int(im_idxes[-1] + 1)

        # add global dummy token
        if self.global_token:
            # spatial block input init (Length, Batch (#frame), d_model)
            spatial_input = torch.zeros((max_pair + 1, num_frame, self.d_model), device=device)
            # spatial block padding mask init (Batch (#frame), Length)
            spatial_padding_masks = torch.zeros((num_frame, max_pair + 1), dtype=torch.bool, device=device)
            # fill spatial block input and mask
            for i in range(num_frame):
                # global token at position 0
                spatial_input[0, i, :] = self.global_token_embed.weight
                spatial_input[1 : (torch.sum(im_idxes == i) + 1), i, :] = features[im_idxes == i]
                spatial_padding_masks[i, (torch.sum(im_idxes == i) + 1) :] = 1
            # spatial attention forward
            spatial_output = self.spatial_self_attention(spatial_input, spatial_padding_masks)
            # extract global features
            global_output = spatial_output[0, :, :]  # (Batch (#frame), d_model)
            # remove paddings
            spatial_output_mask = spatial_padding_masks[:, 1:].contiguous().view(-1) == 0
            # batch second to batch first
            spatial_output = (
                (spatial_output[1:].permute(1, 0, 2)).contiguous().view(-1, self.d_model)[spatial_output_mask]
            )

        else:
            # spatial block input init (Length, Batch (#frame), d_model)
            spatial_input = torch.zeros((max_pair, num_frame, self.d_model), device=device)
            # spatial block padding mask init (Batch (#frame), Length)
            spatial_padding_masks = torch.zeros((num_frame, max_pair), dtype=torch.bool, device=device)
            # fill spatial block input and mask
            for i in range(num_frame):
                spatial_input[: torch.sum(im_idxes == i), i, :] = features[im_idxes == i]
                spatial_padding_masks[i, torch.sum(im_idxes == i) :] = 1
            # spatial attention forward
            spatial_output = self.spatial_self_attention(spatial_input, spatial_padding_masks)
            # remove paddings
            spatial_output_mask = spatial_padding_masks.view(-1) == 0
            # batch second to batch first
            spatial_output = (spatial_output.permute(1, 0, 2)).contiguous().view(-1, self.d_model)[spatial_output_mask]
        ### Spatial self-attention End ###

        ### Cross-Attention ###
        # known sliding windows
        num_sliding_window = len(windows)
        max_len_window = torch.max(torch.sum(windows, dim=1))
        # query input init: (max_len_window, Batch #window, d_model)
        temporal_input = torch.zeros((max_len_window, num_sliding_window, self.d_model), device=device)
        # query positional encoding init (max_len_window, Batch #window, d_model)
        temporal_pos_encoding = torch.zeros((max_len_window, num_sliding_window, self.d_model), device=device)
        temporal_idx = -torch.ones((max_len_window, num_sliding_window), dtype=torch.long, device=device)
        # query padding mask init (Batch #window, max_len_window)
        temporal_padding_masks = torch.zeros((num_sliding_window, max_len_window), dtype=torch.bool, device=device)

        # key & value input init: (len_window, Batch #window, d_gaze)
        cross_input = torch.zeros((self.sliding_window, num_sliding_window, self.d_cross), device=device)
        # key & value positional encoding init (len_window, Batch #window, d_gaze)
        cross_pos_encoding = torch.zeros([self.sliding_window, num_sliding_window, self.d_cross], device=device)
        # key & value padding mask init (Batch #window, len_window)
        cross_padding_masks = torch.zeros((num_sliding_window, self.sliding_window), dtype=torch.bool, device=device)

        # fill everything in each sliding window
        for idx_window, window in enumerate(windows):
            temporal_slice_len = torch.sum(window)
            # fill temporal (query) input
            temporal_input[:temporal_slice_len, idx_window, :] = spatial_output[window]
            temporal_idx[:temporal_slice_len, idx_window] = im_idxes[window]
            # temporal mask padding
            temporal_padding_masks[idx_window, temporal_slice_len:] = 1
            # temporal positional encoding
            im_idx_start = temporal_idx[0, idx_window]
            for idx in range(temporal_slice_len):
                if self.sinusoidal_encoding:
                    temporal_pos_encoding[idx, idx_window, :] = self.pos_encoding_query[
                        temporal_idx[idx, idx_window] - im_idx_start
                    ]
                else:
                    # copy from positional encoding
                    temporal_pos_encoding[idx, idx_window, :] = self.pos_encoding_query.weight[
                        temporal_idx[idx, idx_window] - im_idx_start
                    ]

            # fill gaze (key & value) input
            cross_slice_len = len(gaze_features[idx_window])
            cross_input[:cross_slice_len, idx_window, :d_gaze] = gaze_features[idx_window]
            # fill global features
            if self.global_token:
                # get frame slice from windows_out
                im_idx_end = im_idxes[windows_out[idx_window]][0]
                cross_input[:cross_slice_len, idx_window, d_gaze:] = global_output[
                    im_idx_end - cross_slice_len + 1 : im_idx_end + 1
                ]
            # cross mask padding
            cross_padding_masks[idx_window, cross_slice_len:] = 1
            # cross positional encoding
            if self.sinusoidal_encoding:
                cross_pos_encoding[:cross_slice_len, idx_window, :] = self.pos_encoding_cross[
                    (self.sliding_window - cross_slice_len) :, :
                ]

            else:
                cross_pos_encoding[:cross_slice_len, idx_window, :] = self.pos_encoding_cross.weight[
                    (self.sliding_window - cross_slice_len) :, :
                ]

        # cross attention forward
        if self.cross_layer_num > 1:
            # first cross layer, add pos encoding
            temporal_output, cross_output, cross_attention_weights = self.temporal_cross_attention_first(
                temporal_input,
                cross_input,
                temporal_pos_encoding,
                cross_pos_encoding,
                temporal_padding_masks,
                cross_padding_masks,
                dropout=False,
            )
            # further cross layer, not adding pos encoding
            temporal_output, _, cross_attention_weights = self.temporal_cross_attention_further(
                temporal_output, cross_output, 0, 0, temporal_padding_masks, cross_padding_masks, dropout=False
            )
        else:
            temporal_output, _, cross_attention_weights = self.temporal_cross_attention(
                temporal_input,
                cross_input,
                temporal_pos_encoding,
                cross_pos_encoding,
                temporal_padding_masks,
                cross_padding_masks,
                dropout=False,
            )

        ### Cross-Attention End ###

        ### Temporal self-attention ###
        # temporal attention forward
        temporal_output = self.temporal_self_attention(temporal_output, temporal_pos_encoding, temporal_padding_masks)
        # output matrix init, (#interactions, d_model)
        output = torch.zeros((len(im_idxes), self.d_model), device=device)
        output_mask = torch.zeros_like(im_idxes, device=device).bool()
        for idx_window, (window, window_out) in enumerate(zip(windows, windows_out)):
            temporal_slice_len = torch.sum(window)
            out_len = torch.sum(window_out)
            output[window_out, :] = temporal_output[temporal_slice_len - out_len : temporal_slice_len, idx_window]
            output_mask = output_mask | window_out
        output = output[output_mask]
        ### Temporal self-attention End ###

        return output, cross_attention_weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
