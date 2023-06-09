#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sttran.transformer import STTranTransformer
from .sttran.transformer_gaze import STTranTransformerGaze
from .word_vectors import glove_embedding_vectors


# Modified STTran with gaze features, one or two prediction head
class STTranGaze(nn.Module):
    """
    Spatial-temporal transformer with gaze

    Args:
        num_interaction_classes ([type]): [description]
        obj_class_names ([type]): [description]
        dim_gaze_heatmap (int, optional): [description]. Defaults to 64.
        enc_layer_num (int, optional): [description]. Defaults to 1.
        dec_layer_num (int, optional): [description]. Defaults to 3.
        dim_transformer_ffn (int, optional): [description]. Defaults to 2048.
        word_vector_dir (str, optional): [description]. Defaults to "weights/semantic/".
        no_gaze (bool, optional): disable gaze features. Defaults to False.
        separate_head (int, optional): separate the prediction head, first head contains this number of output, the other one num_interaction_classes-separate_head. Default to 0.
    """

    def __init__(
        self,
        num_interaction_classes,
        obj_class_names,
        enc_layer_num=1,
        dec_layer_num=3,
        dim_transformer_ffn=2048,
        sinusoidal_encoding=False,
        word_vector_dir="weights/semantic/",
        sliding_window=2,
        no_gaze=False,
        separate_head=[-1],
        separate_head_name=["interaction_head"],
    ):
        super().__init__()
        self.obj_class_names = obj_class_names
        self.num_interaction_classes = num_interaction_classes
        self.sliding_window = sliding_window
        self.no_gaze = no_gaze
        if not isinstance(separate_head, list):
            separate_head = [separate_head]
        self.separate_head = separate_head
        if not isinstance(separate_head_name, list):
            separate_head_name = [separate_head_name]
        self.separate_head_name = separate_head_name
        self.d_model = 512 * 3 + 200
        # self.d_model = 512 * 2 + 256 + 200
        if not self.no_gaze:
            # self.d_model += 512
            self.d_model += 256

        ###################################
        # self.union_func = nn.Conv2d(1024, 256, 1, 1)
        # self.union_func = nn.Sequential(
        #     nn.Conv2d(1024, 256, 1, 1), nn.AdaptiveAvgPool2d(1)
        # )
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 256 // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256 // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.AdaptiveAvgPool2d(1),
        )  # in 27x27, out 7x7
        self.conv_gaze = nn.Sequential(
            nn.Conv2d(1, 256 // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256 // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.AdaptiveAvgPool2d(1),
        )  # in 27x27, out 7x7
        self.subj_fc = nn.Linear(2048, 512)
        self.obj_fc = nn.Linear(2048, 512)
        self.union_fc = nn.Linear(2048, 256)
        # self.vr_fc = nn.Linear(256 * 7 * 7, 512)
        # self.gaze_fc = nn.Linear(int(256 * 7 * 7), 512)

        embed_vecs = glove_embedding_vectors(
            self.obj_class_names, wv_type="6B", wv_dir=str(word_vector_dir), wv_dim=200
        )
        self.obj_embed = nn.Embedding(len(self.obj_class_names), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.glocal_transformer = STTranTransformer(
            enc_layer_num=enc_layer_num,
            dec_layer_num=dec_layer_num,
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=dim_transformer_ffn,
            dropout=0.1,
            sliding_window=self.sliding_window,
            sinusoidal_encoding=sinusoidal_encoding,
        )

        acc_num_interactions = 0
        # separate to multiple heads
        for idx, (head_num, head_name) in enumerate(zip(self.separate_head, self.separate_head_name)):
            if head_num <= 0:
                if idx != len(self.separate_head) - 1:
                    raise ValueError("-1 can only be at the end of head_num list")
                out_dim = self.num_interaction_classes - acc_num_interactions
                self.separate_head[idx] = out_dim
                new_head = nn.Linear(self.d_model, out_dim)
            else:
                new_head = nn.Linear(self.d_model, head_num)
                acc_num_interactions += head_num
            self.add_module(head_name, new_head)

        # if self.separate_head <= 0:
        #     self.interaction_head = nn.Linear(
        #         self.d_model, self.num_interaction_classes
        #     )
        # else:
        #     self.spatial_head = nn.Linear(self.d_model, self.separate_head)
        #     self.action_head = nn.Linear(
        #         self.d_model, self.num_interaction_classes - self.separate_head
        #     )

    def forward(self, entry):
        # visual part
        # subject (person) features 512-d
        subj_repr = entry["features"][entry["pair_idxes"][:, 0]]
        subj_repr = self.subj_fc(subj_repr)
        subj_repr = F.normalize(subj_repr)
        # object features 512-d
        obj_repr = entry["features"][entry["pair_idxes"][:, 1]]
        obj_repr = self.obj_fc(obj_repr)
        obj_repr = F.normalize(obj_repr)
        # visual relation features 512-d
        # union_feats = self.union_func(entry["union_feats"])
        # union_feats = F.normalize(union_feats.view(-1, 256))
        union_feats = self.union_fc(entry["union_feats"])
        union_feats = F.normalize(union_feats)
        mask_feats = self.conv_spatial(entry["spatial_masks"])
        mask_feats = F.normalize(mask_feats.view(-1, 256))
        # vr = union_feats + mask_feats
        # vr = self.vr_fc(vr.view(-1, 256 * 7 * 7))
        # vr = F.normalize(vr.view(-1, 256))
        # entry["features"].cpu()

        if self.no_gaze:
            x_visual = torch.cat((subj_repr, obj_repr, union_feats, mask_feats), 1)
            # x_visual = torch.cat((subj_repr, obj_repr, vr), 1)
        else:
            # gaze features 512-d
            gaze = self.conv_gaze(entry["obj_heatmaps"])
            # gaze = self.gaze_fc(gaze.view(-1, 256 * 7 * 7))
            gaze = F.normalize(gaze.view(-1, 256))
            # concatenate to 2048-d vector
            x_visual = torch.cat((subj_repr, obj_repr, union_feats, mask_feats, gaze), 1)
            # x_visual = torch.cat((subj_repr, obj_repr, vr, gaze), 1)

        # semantic part, subject (always human) embedding not needed
        # object semantic embedding 200-d
        obj_class = entry["pred_labels"][entry["pair_idxes"][:, 1]]
        obj_emb = self.obj_embed(obj_class)
        obj_emb = F.normalize(obj_emb)
        x_semantic = obj_emb

        # concatenate to 1992-d vector (no gaze 1736-d)
        relation_features = torch.cat((x_visual, x_semantic), dim=1)

        # Spatial-Temporal Transformer
        (
            global_output,
            global_attention_weights,
            local_attention_weights,
        ) = self.glocal_transformer(
            features=relation_features,
            im_idxes=entry["im_idxes"],
            windows=entry["windows"],
            windows_out=entry["windows_out"],
        )

        # classify interactions, use Sigmoid to output probability for each class
        for head_name in self.separate_head_name:
            entry[head_name] = getattr(self, head_name)(global_output)

        # if self.separate_head <= 0:
        #     entry["interaction_distribution"] = self.interaction_head(global_output)
        # else:
        #     entry["spatial_distribution"] = self.spatial_head(global_output)
        #     entry["action_distribution"] = self.action_head(global_output)

        return entry


# Modified STTran, gaze as cross attention
class STTranGazeCrossAttention(nn.Module):
    """
    Spatial-temporal transformer with gaze

    Args:
        num_interaction_classes ([type]): [description]
        obj_class_names ([type]): [description]
        dim_gaze_heatmap (int, optional): [description]. Defaults to 64.
        enc_layer_num (int, optional): [description]. Defaults to 1.
        dec_layer_num (int, optional): [description]. Defaults to 3.
        dim_transformer_ffn (int, optional): [description]. Defaults to 2048.
        word_vector_dir (str, optional): [description]. Defaults to "weights/semantic/".
        no_gaze (bool, optional): disable gaze features. Defaults to False.
        separate_head (int, optional): separate the prediction head, first head contains this number of output,
            the other one num_interaction_classes-separate_head. Default to 0.
    """

    def __init__(
        self,
        num_interaction_classes,
        obj_class_names,
        spatial_layer_num=1,
        cross_layer_num=1,
        temporal_layer_num=2,
        dim_transformer_ffn=2048,
        d_gaze=512,
        cross_sa=True,
        cross_ffn=False,
        global_token=False,  # add global token to spatial encoder
        mlp_projection=False,
        sinusoidal_encoding=False,
        dropout=0.1,
        word_vector_dir="weights/semantic/",
        sliding_window=2,
        separate_head=[-1],
        separate_head_name=["interaction_head"],
    ):
        super().__init__()
        self.obj_class_names = obj_class_names
        self.num_interaction_classes = num_interaction_classes
        self.sliding_window = sliding_window
        if not isinstance(separate_head, list):
            separate_head = [separate_head]
        self.separate_head = separate_head
        if not isinstance(separate_head_name, list):
            separate_head_name = [separate_head_name]
        self.separate_head_name = separate_head_name
        self.d_gaze = d_gaze
        self.d_model = 512 * 2 + 256 * 2 + 200

        ###################################
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 256 // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256 // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.AdaptiveAvgPool2d(1),
        )  # in 2x27x27, out 256x7x7, then average pool to 256-d
        self.conv_gaze = nn.Sequential(
            nn.Conv2d(1, self.d_gaze // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.d_gaze // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.d_gaze // 2, self.d_gaze, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.d_gaze, momentum=0.01),
            nn.AdaptiveAvgPool2d(1),
        )  # in 64x64, out d_gazex16x16, then average pool to d_gaze

        if mlp_projection:
            # MLP for down-projection
            self.subj_fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(1024, 512),
                nn.Dropout(p=dropout),
            )
            self.obj_fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(1024, 512),
                nn.Dropout(p=dropout),
            )
            self.union_fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(1024, 256),
                nn.Dropout(p=dropout),
            )
        else:
            # simple down-projection matrix
            self.subj_fc = nn.Linear(2048, 512)
            self.obj_fc = nn.Linear(2048, 512)
            self.union_fc = nn.Linear(2048, 256)

        embed_vecs = glove_embedding_vectors(
            self.obj_class_names, wv_type="6B", wv_dir=str(word_vector_dir), wv_dim=200
        )
        self.obj_embed = nn.Embedding(len(self.obj_class_names), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.glocal_transformer = STTranTransformerGaze(
            spatial_layer_num=spatial_layer_num,
            cross_layer_num=cross_layer_num,
            temporal_layer_num=temporal_layer_num,
            d_model=self.d_model,
            d_cross=self.d_gaze,
            nhead=8,
            dim_feedforward=dim_transformer_ffn,
            dropout=dropout,
            sliding_window=self.sliding_window,
            kv_sa=cross_sa,
            kv_ffn=cross_ffn,
            global_token=global_token,
            sinusoidal_encoding=sinusoidal_encoding,
        )

        acc_num_interactions = 0
        # separate to multiple heads
        for idx, (head_num, head_name) in enumerate(zip(self.separate_head, self.separate_head_name)):
            if head_num <= 0:
                if idx != len(self.separate_head) - 1:
                    raise ValueError("-1 can only be at the end of head_num list")
                out_dim = self.num_interaction_classes - acc_num_interactions
                self.separate_head[idx] = out_dim
                new_head = nn.Linear(self.d_model, out_dim)
            else:
                new_head = nn.Linear(self.d_model, head_num)
                acc_num_interactions += head_num
            self.add_module(head_name, new_head)

    def forward(self, entry):
        # visual part
        # subject (person) features 512-d
        subj_repr = entry["features"][entry["pair_idxes"][:, 0]]
        subj_repr = self.subj_fc(subj_repr)
        subj_repr = F.normalize(subj_repr)
        # object features 512-d
        obj_repr = entry["features"][entry["pair_idxes"][:, 1]]
        obj_repr = self.obj_fc(obj_repr)
        obj_repr = F.normalize(obj_repr)
        # visual relation features 512-d
        union_feats = self.union_fc(entry["union_feats"])
        union_feats = F.normalize(union_feats)
        mask_feats = self.conv_spatial(entry["spatial_masks"])
        mask_feats = F.normalize(mask_feats.view(-1, 256))

        # concatenate to 1536-d vector
        x_visual = torch.cat((subj_repr, obj_repr, union_feats, mask_feats), 1)

        # semantic part, subject (always human) embedding not needed
        # object semantic embedding 200-d
        obj_class = entry["pred_labels"][entry["pair_idxes"][:, 1]]
        obj_emb = self.obj_embed(obj_class)
        obj_emb = F.normalize(obj_emb)
        x_semantic = obj_emb

        # concatenate to 1736-d vector
        relation_features = torch.cat((x_visual, x_semantic), dim=1)

        # gaze features 512-d
        gaze_features = []
        for full_heatmaps in entry["full_heatmaps"]:
            gaze_feature = self.conv_gaze(full_heatmaps)
            gaze_feature = F.normalize(gaze_feature.view(-1, self.d_gaze))
            gaze_features.append(gaze_feature)

        # Spatial-Temporal Transformer
        global_output, cross_attention_weights = self.glocal_transformer(
            features=relation_features,
            gaze_features=gaze_features,
            im_idxes=entry["im_idxes"],
            windows=entry["windows"],
            windows_out=entry["windows_out"],
        )

        # classify interactions, use Sigmoid to output probability for each class
        for head_name in self.separate_head_name:
            entry[head_name] = getattr(self, head_name)(global_output)

        return entry
