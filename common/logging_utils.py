#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import csv
import pandas as pd
import numpy as np
import json

try:
    import wandb
except ImportError:
    wandb = False


class WandbLogger:
    def __init__(self, config, project, wandb_dir, epoch=0, disabled=False):
        if not wandb or disabled:
            self.disabled = True
        else:
            self.disabled = False
        if not self.disabled:
            absolute_dir = Path(wandb_dir).resolve()
            self.wandb = wandb.init(config=config, project=project, dir=absolute_dir)
        self.epoch = epoch
        self.metric_dict = {}

    def log_train_loss(self, loss):
        self.metric_dict["train/loss"] = loss

    def log_train_lr(self, lr):
        self.metric_dict["train/lr"] = lr

    def log_val_loss(self, loss):
        self.metric_dict["val/loss"] = loss

    def log_val_metric(self, metrics):
        for key, value in metrics.items():
            self.metric_dict["metrics/" + key] = value

    def log_metric(self, metrics):
        self.metric_dict.update(metrics)

    def log_val_metric_k(self, metrics, k):
        for key, value in metrics.items():
            self.metric_dict[f"metrics@{k}/" + key] = value

    def log_epoch(self):
        if not self.disabled:
            self.wandb.log(self.metric_dict, step=self.epoch)
        self.metric_dict.clear()
        self.epoch += 1

    def finish_run(self):
        # finish not logged dict
        if not self.disabled:
            if self.metric_dict:
                self.wandb.log(self.metric_dict)
            self.wandb.finish()


class CSVLogger:
    def __init__(self, csv_dir) -> None:
        self.csv_dir = Path(csv_dir)

    def init_header(self, header):
        # init csv header, use "w"
        with self.csv_dir.open("w", newline="") as f:
            writer = csv.writer(f, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

    def add_entry(self, entry):
        # append to file
        with self.csv_dir.open("a", newline="") as f:
            writer = csv.writer(f, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(entry)


class PandasMetricLogger:
    def __init__(self, pandas_dir, header) -> None:
        self.pandas_dir = Path(pandas_dir)
        self.metric_table = pd.DataFrame(columns=header)
        self.metric_table.index.name = "epoch"
        self.header = header

    def add_entry(self, entry, epoch):
        entry = pd.DataFrame(entry, columns=self.header, index=[epoch])
        self.metric_table = pd.concat([self.metric_table, entry])
        self.metric_table.to_csv(self.pandas_dir)


class ClipLossesLogger:
    def __init__(self, json_dir):
        self.json_dir = Path(json_dir)
        self.clip_losses_dict = {}

    def add_entry(self, clip_losses, epoch):
        self.clip_losses_dict[epoch] = clip_losses
        output_txt = json.dumps(self.clip_losses_dict)
        with self.json_dir.open("w") as f:
            f.write(output_txt)


class PandasClassAPLogger:
    def __init__(self, pandas_dir) -> None:
        self.pandas_dir = Path(pandas_dir)
        self.ap_table = pd.DataFrame()

    def add_entry(
        self, triplets_ap, triplets_type, triplets_type_num, object_classes, interaction_classes, epoch, hio=False
    ):
        available_triplets = ~np.isnan(triplets_ap)
        available_triplets_type = np.array(triplets_type)[available_triplets]
        available_triplets_type_num = np.array(triplets_type_num)[available_triplets]
        available_triplets_ap = np.array(triplets_ap)[available_triplets]
        new_entry = {}
        new_entry_column = []
        new_entry_num = {}
        # if epoch 0, additionally append triplet number
        for i, triplet in enumerate(available_triplets_type):
            if hio:
                subj = object_classes[triplet[0]]
                interaction = interaction_classes[triplet[1]]
                obj = object_classes[triplet[2]]
            else:
                subj = object_classes[triplet[0]]
                obj = object_classes[triplet[1]]
                interaction = interaction_classes[triplet[2]]
            triplet_name = f"{subj}-{interaction}-{obj}"
            new_entry_column.append(triplet_name)
            new_entry[triplet_name] = available_triplets_ap[i]
            if epoch == 0:
                new_entry_num[triplet_name] = available_triplets_type_num[i]
        if epoch == 0:
            new_entry_num = pd.DataFrame(new_entry_num, columns=new_entry_column, index=["number"])
            self.ap_table = pd.concat([self.ap_table, new_entry_num])
            self.ap_table.index.name = "epoch"
        new_entry = pd.DataFrame(new_entry, columns=new_entry_column, index=[epoch])
        self.ap_table = pd.concat([self.ap_table, new_entry])
        self.ap_table.to_csv(self.pandas_dir)
