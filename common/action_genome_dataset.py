#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import torch
import json
from pathlib import Path
import logging
import random
import pickle

from torch.utils.data import Dataset


def dataset_collate_fn(data):
    frames_list, annotations_list, meta_info_list = zip(*data)
    frames = [torch.cat(f) for f in frames_list]
    return frames, annotations_list, meta_info_list


# partly based on STTran Action Genome dataset file
class AGDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        mode="train",
        transform=None,
        additional_transform=None,
        post_transform=None,
        min_length=1,
        max_length=99999999,
        annotation_mode="window",
        train_ratio=1.0,
        subset_len=-1,
        subset_shuffle=False,
        logger=None,
        **kwargs,
    ):
        self.mode = mode  # train or test
        self.annotation_mode = annotation_mode
        self.transform = transform
        self.additional_transform = additional_transform  # {name: transform_func}
        self.post_transform = post_transform
        self.min_length = 1 if min_length <= 1 else min_length
        self.max_length = 1 if max_length <= 1 else max_length
        self.train_ratio = train_ratio
        self.subset_len = subset_len
        self.subset_shuffle = subset_shuffle
        # train or validation
        self.is_train = True

        if logger is None:
            logger = logging.getLogger("dataset")
            logger.setLevel(logging.DEBUG)
        self.logger = logger

        logger.info(f"Loading Action Genome {self.mode} dataset {str(dataset_dir)} ...")

        dataset_dir = Path(dataset_dir)
        self.frames_dir = dataset_dir / "frames"
        self.annotations_dir = dataset_dir / "annotations"
        # collect the object classes
        self.object_classes = ["__background__"]
        with (self.annotations_dir / "object_classes.txt").open("r") as f:
            for line in f.readlines():
                line = line.strip("\n")
                self.object_classes.append(line)
        self.object_classes[9] = "closet/cabinet"
        self.object_classes[11] = "cup/glass/bottle"
        self.object_classes[23] = "paper/notebook"
        self.object_classes[24] = "phone/camera"
        self.object_classes[31] = "sofa/couch"

        # collect relationship classes
        self.relationship_classes = []
        with (self.annotations_dir / "relationship_classes.txt").open("r") as f:
            for line in f.readlines():
                line = line.strip("\n")
                self.relationship_classes.append(line)
        self.relationship_classes[0] = "looking_at"
        self.relationship_classes[1] = "not_looking_at"
        self.relationship_classes[5] = "in_front_of"
        self.relationship_classes[7] = "on_the_side_of"
        self.relationship_classes[10] = "covered_by"
        self.relationship_classes[11] = "drinking_from"
        self.relationship_classes[13] = "have_it_on_the_back"
        self.relationship_classes[15] = "leaning_on"
        self.relationship_classes[16] = "lying_on"
        self.relationship_classes[17] = "not_contacting"
        self.relationship_classes[18] = "other_relationship"
        self.relationship_classes[19] = "sitting_on"
        self.relationship_classes[20] = "standing_on"
        self.relationship_classes[25] = "writing_on"

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]
        self.attention_idxes = [0, 1, 2]
        self.spatial_idxes = [3, 4, 5, 6, 7, 8]
        self.contacting_idxes = [i for i in range(9, 26)]

        if self.annotation_mode == "clip":
            self._generate_clip_annotation()
        elif self.annotation_mode == "window":
            self._generate_window_annotation()
        elif self.annotation_mode == "anticipation":
            if "future_num" not in kwargs:
                logger.warning(f"Future frame number (param future_num) not provided, default to 1")
                self.future_num = 1
            else:
                self.future_num = kwargs["future_num"]
            if "future_type" not in kwargs:
                logger.warning(
                    f"Future interaction type (param future_type) not provided, could be spatial, action, or all, "
                    f"default to all"
                )
                self.future_type = "all"
            else:
                self.future_type = kwargs["future_type"]
            if "future_ratio" not in kwargs:
                logger.warning(
                    f"Ratio of changes in anticipation annotation (param future_ratio) not provided, "
                    f"default to 1 (only changes)"
                )
                self.future_ratio = 1.0
            else:
                self.future_ratio = kwargs["future_ratio"]

            self._generate_anticipation_annotation()
            logger.info(
                f"{self.videos_change_count}/{len(self.videos_change_list)} valid videos have at "
                f"least one interaction change."
            )
        elif self.annotation_mode == "frame":
            raise NotImplementedError("Frame mode not implemented yet")
        else:
            logger.warning(
                f'annotation_mode can only be "frame", "clip", "window", or "anticipation", '
                f"current is {self.annotation_mode}, use default frame mode"
            )
            self.annotation_mode = "frame"
            raise NotImplementedError("Frame mode not implemented yet")

        # create small subset if needed
        if self.subset_shuffle:
            random.shuffle(self.output_list)
        if self.subset_len > 0:
            self.output_list = self.output_list[: self.subset_len]
        # split to train and validation dataset
        split_index = int(len(self.output_list) * self.train_ratio)
        self.train_list = self.output_list[:split_index]
        self.val_list = self.output_list[split_index:]

        # special post processing for window mode, map validation index to video index
        if self.annotation_mode == "window":
            self.generate_val_window_map()
        elif self.annotation_mode == "anticipation":
            self.generate_val_window_map(self.future_num)

        self.logger.info(
            f"Action Genome dataset loaded, totally {len(self.video_name_list)} valid videos, "
            f"{self.count_valid_frames} valid frames"
        )
        self.logger.info(f"{len(self.droped_short_list)} videos too short, removed")
        self.logger.info(
            f"{len(self.droped_no_human_list)} videos and {self.count_no_human_frames} frames "
            f"have no human annotation, removed"
        )

    def __len__(self):
        if self.is_train:
            # window mode output only one window for one video in training
            return len(self.train_list)
        else:
            # window mode output all windows in validation
            if self.annotation_mode == "window" or self.annotation_mode == "anticipation":
                return len(self.val_window_output_map)
            else:
                return len(self.val_list)

    def __getitem__(self, index):
        if self.annotation_mode == "clip":
            return self._get_clip(index)
        elif self.annotation_mode == "window":
            return self._get_window(index)
        elif self.annotation_mode == "anticipation":
            return self._get_anticipation(index)
        else:
            raise NotImplementedError("Frame mode not implemented yet")

    def train(self, mode=True):
        self.is_train = mode

    def eval(self):
        self.is_train = False

    def generate_val_window_map(self, future_num=-1):
        """
        Call when you change the train_list and val_list manually
        """
        # get_item index belongs to which video
        self.val_window_output_map = []
        # get_item index belongs to which window
        self.val_window_acc_num = []
        last_acc_num = 0
        for output_idx in self.val_list:
            # window mode (no anticipation)
            if future_num == -1:
                video_idx = output_idx
                video_len = len(self.frame_ids_list[video_idx]) - self.min_length + 1
            # anticipation mode
            else:
                video_idx = self.videos_change_list[output_idx]["video_idx"]
                video_len = len(self.frame_ids_list[video_idx]) - self.min_length + 1 - future_num
            self.val_window_output_map += [output_idx] * video_len
            self.val_window_acc_num += [last_acc_num] * video_len
            last_acc_num += video_len

    def _load_frames(self, video_name, frame_ids):
        # read frames in BGR as original, and apply transforms
        output_frames = []
        output_frames_original = []
        output_additional_frames_dict = {}
        for frame_id in frame_ids:
            frame_filename = self.frames_dir / frame_id
            frame0 = cv2.imread(str(frame_filename))
            output_frames_original.append(frame0)
            # frame transform
            if self.transform:
                frame = self.transform(frame0)
            else:
                frame = frame0.copy()
            # more transforms are included in meta_info
            if self.additional_transform:
                for transform_name, transform_func in self.additional_transform.items():
                    if transform_name not in output_additional_frames_dict:
                        output_additional_frames_dict[transform_name] = []
                    additional_frame = transform_func(frame0)
                    output_additional_frames_dict[transform_name].append(additional_frame)
            output_frames.append(frame)
        return output_frames, output_frames_original, output_additional_frames_dict

    def _get_window_from_im_idx(self, video_idx, im_idx_start, im_idx_end):
        # return the frames and annotations of one video
        video_name = self.video_name_list[video_idx]
        frame_ids = self.frame_ids_list[video_idx][im_idx_start:im_idx_end]
        (
            output_frames,
            output_frames_original,
            output_additional_frames_dict,
        ) = self._load_frames(video_name, frame_ids)
        # assume all frames in one video has the same shape
        frame_shape = output_frames_original[-1].shape
        # get bboxes and pairs slice
        bbox_start, bbox_end, pair_start, pair_end = self._find_slice_bboxes_pairs(video_idx, im_idx_start, im_idx_end)
        # clip annotations, minus idx offset
        labels, bboxes, ids, pair_idxes, im_idxes, interactions = self._cut_annotations(
            video_idx, im_idx_start, bbox_start, bbox_end, pair_start, pair_end
        )

        meta_info = {
            "video_name": video_name,
            "video_index": video_idx,
            "frame_ids": frame_ids,
            "original_frames": output_frames_original,
            "original_shape": frame_shape,
            "frame_idx_offset": im_idx_start,
            **output_additional_frames_dict,
        }

        annotations = {
            "labels": labels,
            "bboxes": bboxes,
            "ids": ids,
            "pair_idxes": pair_idxes,
            "im_idxes": im_idxes,
            "interactions": interactions,
        }

        return output_frames, annotations, meta_info

    def _find_slice_bboxes_pairs(self, video_idx, im_idx_start, im_idx_end):
        bboxes = self.bboxes_list[video_idx]
        im_idxes = self.im_idxes_list[video_idx]
        # get the end index of bbox list
        bbox_start = -1
        for bbox_idx, bbox in enumerate(bboxes):
            if bbox[0] == im_idx_start and bbox_start < 0:
                bbox_start = bbox_idx
            if bbox[0] == im_idx_end:
                bbox_idx -= 1
                break
        bbox_end = bbox_idx + 1
        # get the end index of pair list
        pair_start = -1
        for pair_idx, im_idx in enumerate(im_idxes):
            if im_idx == im_idx_start and pair_start < 0:
                pair_start = pair_idx
            if im_idx == im_idx_end:
                pair_idx -= 1
                break
        pair_end = pair_idx + 1

        return bbox_start, bbox_end, pair_start, pair_end

    def _cut_annotations(self, video_idx, im_idx_start, bbox_start, bbox_end, pair_start, pair_end):
        labels = self.labels_list[video_idx][bbox_start:bbox_end]
        bboxes = self.bboxes_list[video_idx][bbox_start:bbox_end]
        bboxes = np.array(bboxes)
        bboxes[:, 0] -= im_idx_start
        bboxes = bboxes.tolist()
        ids = self.ids_list[video_idx][bbox_start:bbox_end]
        pair_idxes = self.pairs_list[video_idx][pair_start:pair_end]
        pair_idxes = (np.array(pair_idxes) - bbox_start).tolist()
        im_idxes = self.im_idxes_list[video_idx][pair_start:pair_end]
        im_idxes = (np.array(im_idxes) - im_idx_start).tolist()
        interactions = self.interactions_list[video_idx][pair_start:pair_end]

        return labels, bboxes, ids, pair_idxes, im_idxes, interactions

    def _get_clip(self, index):
        # train set or val set
        if self.is_train:
            output = self.train_list[index]
        else:
            output = self.val_list[index]
        video_idx = output["video_idx"]
        im_idx_start = output["im_idx_start"]
        im_idx_end = output["im_idx_end"]
        bbox_start = output["bbox_start"]
        bbox_end = output["bbox_end"]
        pair_start = output["pair_start"]
        pair_end = output["pair_end"]

        # return the frames and annotations of one clip
        video_name = self.video_name_list[video_idx]
        frame_ids = self.frame_ids_list[video_idx][im_idx_start:im_idx_end]
        (
            output_frames,
            output_frames_original,
            output_additional_frames_dict,
        ) = self._load_frames(video_name, frame_ids)
        # assume all frames in one video has the same shape
        frame_shape = output_frames_original[-1].shape

        # clip annotations, minus idx offset
        labels, bboxes, ids, pair_idxes, im_idxes, interactions = self._cut_annotations(
            video_idx, im_idx_start, bbox_start, bbox_end, pair_start, pair_end
        )

        meta_info = {
            "video_name": video_name,
            "video_index": video_idx,
            "frame_ids": frame_ids,
            "original_frames": output_frames_original,
            "original_shape": frame_shape,
            "frame_idx_offset": im_idx_start,
            **output_additional_frames_dict,
        }

        annotations = {
            "labels": labels,
            "bboxes": bboxes,
            "ids": ids,
            "pair_idxes": pair_idxes,
            "im_idxes": im_idxes,
            "interactions": interactions,
        }

        # post transformation, e.g. augmentation, only apply in training mode
        if self.post_transform and self.is_train:
            output_frames, annotations, meta_info = self.post_transform(output_frames, annotations, meta_info)

        return output_frames, annotations, meta_info

    def _get_window(self, index):
        # train set or val set
        if self.is_train:
            video_idx = self.train_list[index]
            window_idx = np.random.randint(self.min_length - 1, len(self.frame_ids_list[video_idx]))
        else:
            video_idx = self.val_window_output_map[index]
            # index - self.val_window_acc_num[index] is the no. of sliding window
            # + self.min_length - 1 to get the last frame index in this window
            window_idx = index - self.val_window_acc_num[index] + self.min_length - 1

        im_idx_start = max(0, window_idx - self.max_length + 1)
        im_idx_end = window_idx + 1

        output_frames, annotations, meta_info = self._get_window_from_im_idx(video_idx, im_idx_start, im_idx_end)

        # post transformation, e.g. augmentation
        if self.post_transform and self.is_train:
            output_frames, annotations, meta_info = self.post_transform(output_frames, annotations, meta_info)

        return output_frames, annotations, meta_info

    def _get_frame(self, index):
        # train set or val set
        if self.is_train:
            video_index = self.train_list[index]
        else:
            video_index = self.val_list[index]
        # return the frames and annotations of one video
        video_name = self.video_name_list[video_index]
        frame_ids = self.frame_ids_list[video_index]
        (
            output_frames,
            output_frames_original,
            output_additional_frames_dict,
        ) = self._load_frames(video_name, frame_ids)
        # assume all frames in one video has the same shape
        frame_shape = output_frames_original[-1].shape

        meta_info = {
            "video_name": video_name,
            "video_index": video_index,
            "frame_ids": frame_ids,
            "original_frames": output_frames_original,
            "original_shape": frame_shape,
            **output_additional_frames_dict,
        }

        annotations = {
            "objects": self.objects_list[video_index],
            "traces": self.traces_list[video_index],
            "interactions": self.interactions_list[video_index],
        }

        # post transformation, e.g. augmentation
        if self.post_transform and self.is_train:
            output_frames, annotations, meta_info = self.post_transform(output_frames, annotations, meta_info)

        return output_frames, annotations, meta_info

    def _get_anticipation(self, index):
        # train set, determine select change or no-change
        if self.is_train:
            output_idx = self.train_list[index]
            output_change_list = self.videos_change_list[output_idx]
            video_idx = output_change_list["video_idx"]
            # has change, and randomly select change
            if len(output_change_list["changes"]) > 0 and random.random() < self.future_ratio:
                # randomly choose an anticipation from the list
                anticipation_im_idx = random.choice(output_change_list["changes"])["im_idx"]
            # not has change, or ramdomly select no-change
            else:
                # randomly choose any valid anticipation window
                anticipation_im_idx = np.random.randint(
                    self.min_length - 1 + self.future_num,
                    len(self.frame_ids_list[video_idx]),
                )

        # val set, evaluate on all anticipation windows
        else:
            output_idx = self.val_window_output_map[index]
            output_change_list = self.videos_change_list[output_idx]
            video_idx = output_change_list["video_idx"]
            # index - self.val_window_acc_num[index] is the no. of sliding window
            # + self.min_length - 1 + self.future_num to get the anticipation index
            anticipation_im_idx = index - self.val_window_acc_num[index] + self.min_length - 1 + self.future_num
        # the chosen anticipation window has change
        try:
            temp_idx = output_change_list["im_idxes"].index(anticipation_im_idx)
            changes = output_change_list["changes"][temp_idx]
        # the chosen anticipation window has no change
        except ValueError:
            changes = {"im_idx": anticipation_im_idx, "pair_ids": []}

        window_idx = anticipation_im_idx - self.future_num
        im_idx_start = max(0, window_idx - self.max_length + 1)
        im_idx_end = window_idx + 1

        # get the sliding window frames and annotations
        output_frames, annotations, meta_info = self._get_window_from_im_idx(video_idx, im_idx_start, im_idx_end)

        # additionally output the anticipation annotation
        bbox_start, bbox_end, pair_start, pair_end = self._find_slice_bboxes_pairs(
            video_idx, anticipation_im_idx, anticipation_im_idx + 1
        )
        labels, bboxes, ids, pair_idxes, im_idxes, interactions = self._cut_annotations(
            video_idx, anticipation_im_idx, bbox_start, bbox_end, pair_start, pair_end
        )
        anticipation_frame_id = self.frame_ids_list[video_idx][anticipation_im_idx]
        # for anticipation, don't need pair_idxes and im_idxes, instead use pair_id
        pair_ids = []
        for pair in pair_idxes:
            sub_idx = pair[0]
            obj_idx = pair[1]
            pair_ids.append([ids[sub_idx], ids[obj_idx]])
        anticipation = {
            "labels": labels,
            "bboxes": bboxes,
            "ids": ids,
            "pair_ids": pair_ids,
            "interactions": interactions,
            "changes": changes,
        }
        annotations["anticipation"] = anticipation
        meta_info["anticipation_frame_id"] = anticipation_frame_id

        # post transformation, e.g. augmentation
        if self.post_transform and self.is_train:
            output_frames, annotations, meta_info = self.post_transform(output_frames, annotations, meta_info)

        return output_frames, annotations, meta_info

    def analyse_split_weight(self, save_dir=None, method="inverse", separate_head=True, **kwargs):
        """
        Generate triplet histogram in dataset, find unique triplet in validation subset.
        Generate interaction histogram in dataset, calculate weight with given method.
        If save_path set, store result to json files.
        """
        # interaction weight init
        num_classes = len(self.relationship_classes)
        num_train_pairs = 0
        interaction_train_hist = np.zeros(num_classes)
        num_val_pairs = 0
        interaction_val_hist = np.zeros(num_classes)
        # triplet histogram init
        triplet_train_hist = {}
        triplet_val_hist = {}
        triplet_val_unique = []
        # process training subset
        for output in self.train_list:
            if self.annotation_mode == "clip":
                video_idx = output["video_idx"]
                pair_start = output["pair_start"]
                pair_end = output["pair_end"]
            else:
                video_idx = output
                pair_start = 0
                pair_end = len(self.pairs_list[video_idx])
            labels = self.labels_list[video_idx]
            pair_idxes = self.pairs_list[video_idx][pair_start:pair_end]
            interactions_frame = self.interactions_list[video_idx][pair_start:pair_end]
            for pair_idx, interactions in zip(pair_idxes, interactions_frame):
                num_train_pairs += 1
                subj_name = self.object_classes[labels[pair_idx[0]]]
                obj_name = self.object_classes[labels[pair_idx[1]]]
                for interaction in interactions:
                    interaction_train_hist[interaction] += 1
                    interaction_name = self.relationship_classes[interaction]
                    triplet = f"{subj_name}-{interaction_name}-{obj_name}"
                    if triplet in triplet_train_hist:
                        triplet_train_hist[triplet] += 1
                    else:
                        triplet_train_hist[triplet] = 1
        # calculate train subset weight
        weight_train = self._calculate_weight(num_train_pairs, interaction_train_hist, method, separate_head, **kwargs)
        # process validation subset
        for output in self.val_list:
            if self.annotation_mode == "clip":
                video_idx = output["video_idx"]
                pair_start = output["pair_start"]
                pair_end = output["pair_end"]
            else:
                video_idx = output
                pair_start = 0
                pair_end = len(self.pairs_list[video_idx])
            labels = self.labels_list[video_idx]
            pair_idxes = self.pairs_list[video_idx][pair_start:pair_end]
            interactions_frame = self.interactions_list[video_idx][pair_start:pair_end]
            for pair_idx, interactions in zip(pair_idxes, interactions_frame):
                num_val_pairs += 1
                subj_name = self.object_classes[labels[pair_idx[0]]]
                obj_name = self.object_classes[labels[pair_idx[1]]]
                for interaction in interactions:
                    interaction_val_hist[interaction] += 1
                    interaction_name = self.relationship_classes[interaction]
                    triplet = f"{subj_name}-{interaction_name}-{obj_name}"
                    if triplet in triplet_val_hist:
                        triplet_val_hist[triplet] += 1
                    else:
                        triplet_val_hist[triplet] = 1
                        # unique in val subset
                        if triplet not in triplet_train_hist:
                            triplet_val_unique.append(triplet)
        # calculate validation subset weight
        weight_val = self._calculate_weight(num_val_pairs, interaction_val_hist, method, separate_head, **kwargs)
        json_dict_triplet = {
            "triplet_train_hist": triplet_train_hist,
            "triplet_val_hist": triplet_val_hist,
            "triplet_val_unique": triplet_val_unique,
        }
        json_dict_weight = {
            "interaction_classes": self.relationship_classes,
            "num_train_pairs": num_train_pairs,
            "num_val_pairs": num_val_pairs,
            "interaction_train_hist": interaction_train_hist.tolist(),
            "interaction_val_hist": interaction_val_hist.tolist(),
        }
        # save if required
        if save_dir is not None:
            save_dir = Path(save_dir)
            json_txt_triplet = json.dumps(json_dict_triplet)
            with (save_dir / "dataset_info.json").open("w") as f:
                f.write(json_txt_triplet)

            if separate_head:
                json_dict_weight["weight_train_attention"] = weight_train[0].tolist()
                json_dict_weight["weight_train_spatial"] = weight_train[1].tolist()
                json_dict_weight["weight_train_contacting"] = weight_train[2].tolist()
                json_dict_weight["weight_val_attention"] = weight_train[0].tolist()
                json_dict_weight["weight_val_spatial"] = weight_train[1].tolist()
                json_dict_weight["weight_val_contacting"] = weight_train[2].tolist()
            else:
                json_dict_weight["weight_train"] = weight_train
                json_dict_weight["weight_val"] = weight_val
            json_txt_weight = json.dumps(json_dict_weight)
            with (save_dir / "class_weight.json").open("w") as f:
                f.write(json_txt_weight)

        return json_dict_triplet, json_dict_weight

    def _calculate_weight(self, num_pairs, interaction_hist, method, separate_head, **kwargs):
        num_classes = len(self.relationship_classes)
        # #negative / #positive
        if method == "neg_pos":
            weight = num_pairs - interaction_hist
            denominator = np.maximum(1, interaction_hist)  # avoid division by 0
            weight = np.maximum(1, weight)  # avoid division by 0
            weight = weight / denominator
        # effective number of samples following Class-Balanced Loss Based on Effective Number of Samples
        elif method == "effective":
            if "beta" in kwargs:
                beta = kwargs["beta"]
            else:
                beta = 0.9
                self.logger.info("Inverse effective number of sample using default beta=0.9")
            # avoid division by 0 and strange behaviour
            weight = np.maximum(5, interaction_hist)
            weight = (1 - beta) / (1 - beta**weight)
        # inverse positive number
        elif method == "inverse":
            if "power" in kwargs:
                power = kwargs["power"]
            else:
                power = 1
                self.logger.info("Inverse number of sample using default power=1")
            weight = np.maximum(1, interaction_hist)  # avoid division by 0
            weight = 1 / (weight**power)
        # default no weighting
        else:
            weight = np.ones(num_classes)

        # normalize, split to spatial relation and action
        if separate_head:
            weight = weight / np.sum(weight) * num_classes
            weight_attention = weight[self.attention_idxes]
            weight_spatial = weight[self.spatial_idxes]
            # weight_spatial = (
            #     weight_spatial / np.sum(weight_spatial) * num_spatial_classes
            # )
            weight_contacting = weight[self.contacting_idxes]
            # weight_action = weight_action / np.sum(weight_action) * num_action_classes
        else:
            weight = weight / np.sum(weight) * num_classes

        if separate_head:
            return [weight_attention, weight_spatial, weight_contacting]
        else:
            return weight

    def _load_dataset(self):
        # load annotation files
        object_path = self.annotations_dir / "object_bbox_and_relationship.pkl"
        with object_path.open("rb") as f:
            object_bbox = pickle.load(f)
        person_path = self.annotations_dir / "person_bbox.pkl"
        with person_path.open("rb") as f:
            person_bbox = pickle.load(f)

        # collect valid frames
        video_dict = {}
        for frame_id in person_bbox.keys():
            if object_bbox[frame_id][0]["metadata"]["set"] == self.mode:
                frame_valid = False
                for obj_anno in object_bbox[frame_id]:
                    # the frame is valid if there is visible bbox
                    if obj_anno["visible"]:
                        frame_valid = True
                if frame_valid:
                    video_name = frame_id.split("/")[0]
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(frame_id)
                    else:
                        video_dict[video_name] = [frame_id]
        # sort each video frame_ids list
        for video_name in video_dict.keys():
            video_dict[video_name].sort()

        # init annotation lists
        self.output_list = []  # related to the annotation mode
        self.video_name_list = []  # [video_name]
        self.video_size_list = []
        self.frame_ids_list = []  # [[frame_id]]

        self.droped_short_list = []  # [video_name], too short videos
        self.droped_no_human_list = []  # [video_name]
        self.count_no_human_frames = 0
        self.count_valid_frames = 0

        self.labels_list = []  # [[object_label]] n_object x 1
        self.bboxes_list = []  # [[[frame_idx, bbox]]] for each object, n_object x 5
        self.ids_list = []  # [[object_id]] for each object, n_object x 1
        self.pairs_list = []  # [[[subj_bbox_idx, obj_bbox_idx]]] n_pair x 2
        self.im_idxes_list = []  # [[im_idx]] for each pair n_pair x 1
        self.interactions_list = []  # [[interaction_label]] for each pair n_pair x 1

        # collect annotations
        for video_name in video_dict.keys():
            frame_ids = []
            labels = []
            bboxes = []
            ids = []
            pairs = []
            im_idxes = []
            interactions = []

            im_idx = 0
            for frame_id in video_dict[video_name]:
                # no gt human in this frame, skip
                if person_bbox[frame_id]["bbox"].shape[0] == 0:
                    self.count_no_human_frames += 1
                    continue
                else:
                    frame_ids.append(frame_id)

                # add person bbox
                for hid, bbox in enumerate(person_bbox[frame_id]["bbox"]):
                    assert hid == 0, "More than one person annotated!"
                    # person bbox xyxy
                    bboxes.append([im_idx, bbox[0], bbox[1], bbox[2], bbox[3]])
                    labels.append(1)  # human has label 1
                    ids.append(0)  # assign human id 0
                # this human idx in bboxes list
                subj_idx = len(bboxes) - 1

                # add object bbox and interactions
                for obj_anno in object_bbox[frame_id]:
                    if obj_anno["visible"]:
                        assert obj_anno["bbox"] != None, "Warning! The object is visible without bbox"
                        # object name to index
                        label = self.object_classes.index(obj_anno["class"])
                        labels.append(label)
                        # object bbox xywh to xyxy
                        bbox = [
                            im_idx,
                            obj_anno["bbox"][0],
                            obj_anno["bbox"][1],
                            obj_anno["bbox"][0] + obj_anno["bbox"][2],
                            obj_anno["bbox"][1] + obj_anno["bbox"][3],
                        ]
                        bboxes.append(bbox)
                        # NOTE annotation file does not provide trace id, so here only a placeholder
                        ids.append(9)  # assign object id 9
                        obj_idx = len(bboxes) - 1
                        # human object pair
                        pair = [subj_idx, obj_idx]
                        pairs.append(pair)
                        im_idxes.append(im_idx)
                        # interaction, all in one
                        interaction = []
                        for attn_rel in obj_anno["attention_relationship"]:
                            inter_idx = self.relationship_classes.index(attn_rel)
                            interaction.append(inter_idx)
                        for spa_rel in obj_anno["spatial_relationship"]:
                            inter_idx = self.relationship_classes.index(spa_rel)
                            interaction.append(inter_idx)
                        for cont_rel in obj_anno["contacting_relationship"]:
                            inter_idx = self.relationship_classes.index(cont_rel)
                            interaction.append(inter_idx)
                        interactions.append(interaction)
                im_idx += 1

            if len(frame_ids) >= self.min_length:
                self.video_name_list.append(video_name)
                self.video_size_list.append(person_bbox[frame_id]["bbox_size"])
                self.frame_ids_list.append(frame_ids)
                self.labels_list.append(labels)
                self.bboxes_list.append(bboxes)
                self.ids_list.append(ids)
                self.pairs_list.append(pairs)
                self.im_idxes_list.append(im_idxes)
                self.interactions_list.append(interactions)
                self.count_valid_frames += im_idx
            elif len(frame_ids) > 0:
                self.droped_short_list.append(video_name)
            else:
                self.droped_no_human_list.append(video_name)

    # def _generate_frame_annotation(self):
    #     """
    #     Load annotations, for each frame in each video: {object_id: bbox, class}, [person_id, object_id, interaction]
    #     output_list is simply the index of video [video_idx]
    #     """
    #     with open(str(self.annotations_file), "r") as f:
    #         annotation_json = json.load(f)
    #     # init lists
    #     self.objects_list = []  # [{object_id: object_class}]
    #     self.traces_list = []  # [[{object_id: bbox}]]
    #     self.interactions_list = []  # [[[person_id, object_id, interaction]]]
    #     # init entry of each video
    #     frame_ids = []
    #     objects = {}
    #     traces = []
    #     interactions = []
    #     last_video_folder = annotation_json[0]["video_folder"]
    #     last_video_id = annotation_json[0]["video_id"]
    #     last_frame_id = ""
    #     for entry in annotation_json:
    #         # next annotation entry belongs to a new video
    #         if (
    #             entry["video_folder"] != last_video_folder
    #             or entry["video_id"] != last_video_id
    #         ):
    #             # only keep videos with sufficient annotation length
    #             if len(frame_ids) >= self.min_length:
    #                 self.video_name_list.append(last_video_folder + "/" + last_video_id)
    #                 self.frame_ids_list.append(frame_ids)
    #                 self.objects_list.append(objects)
    #                 self.traces_list.append(traces)
    #                 self.interactions_list.append(interactions)
    #             else:
    #                 self.droped_short_list.append(
    #                     last_video_folder + "/" + last_video_id
    #                 )
    #             # clear buffer
    #             frame_ids = []
    #             objects = {}
    #             traces = []
    #             interactions = []
    #             # save next video name
    #             last_video_folder = entry["video_folder"]
    #             last_video_id = entry["video_id"]
    #             last_frame_id = ""

    #         # next annotation entry belongs to a new frame
    #         if entry["frame_id"] != last_frame_id:
    #             last_frame_id = entry["frame_id"]
    #             frame_ids.append(last_frame_id)
    #             traces.append({})
    #             interactions.append([])

    #         # fill object classes list of one video
    #         objects[entry["person_id"]] = 0
    #         objects[entry["object_id"]] = entry["object_class"]
    #         # fill object bbox list of one frame in one video
    #         traces[-1][entry["person_id"]] = [
    #             entry["person_box"]["xmin"],
    #             entry["person_box"]["ymin"],
    #             entry["person_box"]["xmax"],
    #             entry["person_box"]["ymax"],
    #         ]
    #         traces[-1][entry["object_id"]] = [
    #             entry["object_box"]["xmin"],
    #             entry["object_box"]["ymin"],
    #             entry["object_box"]["xmax"],
    #             entry["object_box"]["ymax"],
    #         ]
    #         # fill interaction list of one frame in one video
    #         interactions[-1].append(
    #             [entry["person_id"], entry["object_id"], entry["action_class"]]
    #         )
    #     # add the last video
    #     # only keep videos with sufficient annotation length
    #     if len(frame_ids) >= self.min_length:
    #         self.video_name_list.append(last_video_folder + "/" + last_video_id)
    #         self.frame_ids_list.append(frame_ids)
    #         self.objects_list.append(objects)
    #         self.traces_list.append(traces)
    #         self.interactions_list.append(interactions)
    #     else:
    #         self.droped_short_list.append(last_video_folder + "/" + last_video_id)

    #     # arange output list
    #     self.output_list = [i for i in range(len(self.video_name_list))]

    def _generate_window_annotation(self):
        """
        Sliding window based, output_list is simply the index of video [video_idx]
        """
        # load annotation file, and filter out invalid frames
        self._load_dataset()
        # arange output list, check video length >= min_length
        self.output_list = [
            i for i in range(len(self.video_name_list)) if len(self.frame_ids_list[i]) >= self.min_length
        ]

    def _generate_clip_annotation(self):
        """
        Short clips with length limitation, output_list: {video_idx, [frame_start, frame_end], [bbox_start, bbox_end], [pair_start, pair_end], frame_offset}
        """
        # load annotation file, and filter out invalid frames
        self._load_dataset()
        # then cut to clips
        # for each video
        for video_idx, (video_name, frame_ids, labels, bboxes, im_idxes) in enumerate(
            zip(
                self.video_name_list,
                self.frame_ids_list,
                self.labels_list,
                self.bboxes_list,
                self.im_idxes_list,
            )
        ):
            video_len = len(frame_ids)
            bbox_end = 0
            pair_end = 0
            # for each clip
            for im_idx_start in range(0, video_len, self.max_length):
                frame_id_slice = frame_ids[im_idx_start : im_idx_start + self.max_length]
                clip_len = len(frame_id_slice)
                # following the last clip
                bbox_start = bbox_end
                pair_start = pair_end
                if clip_len < self.min_length:
                    # too short clip, skip clip
                    self.droped_short_list.append(f"{video_name}/{frame_id_slice[0]}-{frame_id_slice[-1]}")
                    continue
                im_idx_end = im_idx_start + clip_len

                # check too many human, and get the end index of bbox list
                bbox_start_temp = bbox_start
                for im_idx in range(im_idx_start, im_idx_end):
                    for bbox_idx, (label, bbox) in enumerate(
                        zip(labels[bbox_start_temp:], bboxes[bbox_start_temp:]),
                        bbox_start_temp,
                    ):
                        # next frame
                        if bbox[0] > im_idx:
                            break
                    bbox_start_temp = bbox_idx
                # get the end index of bbox list
                for bbox_idx, bbox in enumerate(bboxes[bbox_start:], bbox_start):
                    if bbox[0] == im_idx_end:
                        bbox_idx -= 1
                        break
                bbox_end = bbox_idx + 1
                # get the end index of pair list
                for pair_idx, im_idx in enumerate(im_idxes[pair_start:], pair_start):
                    if im_idx == im_idx_end:
                        pair_idx -= 1
                        break
                pair_end = pair_idx + 1

                # add current clip to the output_list
                output = {
                    "video_idx": video_idx,
                    "im_idx_start": im_idx_start,
                    "im_idx_end": im_idx_end,
                    "bbox_start": bbox_start,
                    "bbox_end": bbox_end,
                    "pair_start": pair_start,
                    "pair_end": pair_end,
                }
                self.output_list.append(output)

    def _generate_anticipation_annotation(self):
        # load annotation file, and filter out invalid frames
        self._load_dataset()
        # find all interaction change time point, output_list: [video_idx] of videos available for anticipation (not too short)
        # video_change_list: {video_idx, time_change_im_idx, related_pairs}
        self.videos_change_list = []
        self.videos_change_count = 0
        # for each video
        for video_idx in range(len(self.video_name_list)):
            ids = self.ids_list[video_idx]
            pairs = self.pairs_list[video_idx]
            im_idxes = self.im_idxes_list[video_idx]
            interactions_video = self.interactions_list[video_idx]
            video_change_list = {"video_idx": video_idx, "im_idxes": [], "changes": []}

            pair_last_map = {}
            last_im_idx = -1
            last_changed = False
            # for each human-object pair
            for pair, im_idx, interactions in zip(pairs, im_idxes, interactions_video):
                # new frame, store old entry, init new entry
                if im_idx != last_im_idx:
                    if last_changed and self._check_anticipation_window_valid(changed_interactions, video_idx):
                        video_change_list["changes"].append(changed_interactions)
                        video_change_list["im_idxes"].append(last_im_idx)
                    last_changed = False
                    changed_interactions = {
                        "im_idx": im_idx,
                        "pair_ids": [],
                    }
                sub_idx = pair[0]
                obj_idx = pair[1]
                pair_id = [ids[sub_idx], ids[obj_idx]]
                pair_str = str(pair_id)
                interactions = sorted(interactions)
                # only consider spatial relations
                if self.future_type == "spatial":
                    interactions_interest = [
                        interaction for interaction in interactions if interaction in self.spatial_class_idxes
                    ]
                # only consider actions
                elif self.future_type == "action":
                    interactions_interest = [
                        interaction for interaction in interactions if interaction in self.action_class_idxes
                    ]
                # consider all interactions
                else:
                    interactions_interest = interactions
                # interested interactions exist for this pair
                if len(interactions_interest) > 0:
                    # old pair, check interested interactions changed
                    if pair_str in pair_last_map:
                        if interactions_interest != pair_last_map[pair_str]:
                            changed_interactions["pair_ids"].append(pair_id)
                            pair_last_map[pair_str] = interactions_interest
                            last_changed = True
                    # new pair, store in the map
                    else:
                        pair_last_map[pair_str] = interactions_interest

                last_im_idx = im_idx
            # append the last frame
            if last_changed and self._check_anticipation_window_valid(changed_interactions, video_idx):
                video_change_list["changes"].append(changed_interactions)
                video_change_list["im_idxes"].append(last_im_idx)
            # only consider videos with sufficient length for anticipation
            if last_im_idx + 1 >= self.future_num + self.min_length:
                self.videos_change_list.append(video_change_list)
                if len(video_change_list["changes"]) > 0:
                    self.videos_change_count += 1

        # arange output list
        self.output_list = [i for i in range(len(self.videos_change_list))]

    def _check_anticipation_window_valid(self, changed_interactions, video_idx):
        # check whether the interaction change is happend in the existing human-object pair in the past window
        window_im_idx_last = changed_interactions["im_idx"] - self.future_num
        # window not available (too short, or idx < 0)
        if window_im_idx_last < self.min_length - 1:
            return False
        ids = self.ids_list[video_idx]
        bboxes = self.bboxes_list[video_idx]
        im_idx_mask = np.array(bboxes)[:, 0] == window_im_idx_last
        ids_frame = np.array(ids)[im_idx_mask]
        for pair_id in changed_interactions["pair_ids"]:
            # human-object pair with changed interactions not exists in the window
            if pair_id[0] not in ids_frame or pair_id[1] not in ids_frame:
                return False
        return True
