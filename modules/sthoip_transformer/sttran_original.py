#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .sttran.transformer_original import STTranTransformer
from .word_vectors import glove_embedding_vectors


# The original STTran from "Spatial-Temporal Transformer for Dynamic Scene Graph Generation"
class STTran(nn.Module):
    def __init__(
        self,
        attention_class_num=None,
        spatial_class_num=None,
        contact_class_num=None,
        obj_classes=None,
        rel_classes=None,
        enc_layer_num=None,
        dec_layer_num=None,
        word_vector_dir="weights/semantic/",
    ):
        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        """
        super().__init__()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num

        ###################################
        self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256 // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        self.subj_fc = nn.Linear(2048, 512)
        self.obj_fc = nn.Linear(2048, 512)
        self.vr_fc = nn.Linear(256 * 7 * 7, 512)

        embed_vecs = glove_embedding_vectors(obj_classes, wv_type="6B", wv_dir=word_vector_dir, wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

        self.glocal_transformer = STTranTransformer(
            enc_layer_num=enc_layer_num,
            dec_layer_num=dec_layer_num,
            d_model=1936,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            mode="latter",
        )

        self.a_rel_compress = nn.Linear(1936, self.attention_class_num)
        self.s_rel_compress = nn.Linear(1936, self.spatial_class_num)
        self.c_rel_compress = nn.Linear(1936, self.contact_class_num)

    def forward(self, entry):
        # visual part
        subj_rep = entry["features"][entry["pair_idx"][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        obj_rep = entry["features"][entry["pair_idx"][:, 1]]
        obj_rep = self.obj_fc(obj_rep)
        vr = self.union_func1(entry["union_feat"]) + self.conv(entry["spatial_masks"])
        vr = self.vr_fc(vr.view(-1, 256 * 7 * 7))
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)

        # semantic part
        subj_class = entry["pred_labels"][entry["pair_idx"][:, 0]]
        obj_class = entry["pred_labels"][entry["pair_idx"][:, 1]]
        subj_emb = self.obj_embed(subj_class)
        obj_emb = self.obj_embed2(obj_class)
        x_semantic = torch.cat((subj_emb, obj_emb), 1)

        rel_features = torch.cat((x_visual, x_semantic), dim=1)
        # Spatial-Temporal Transformer
        (
            global_output,
            global_attention_weights,
            local_attention_weights,
        ) = self.glocal_transformer(features=rel_features, im_idx=entry["im_idx"])

        entry["attention_distribution"] = self.a_rel_compress(global_output)
        entry["spatial_distribution"] = self.s_rel_compress(global_output)
        entry["contacting_distribution"] = self.c_rel_compress(global_output)

        entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
        entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])

        return entry
