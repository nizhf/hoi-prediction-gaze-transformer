#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

class EarlyStopping:
    """
    Early stopping monitor

    Args:
        patience ([type]): [description]
        minimum_improvement ([type]): [description]
        start_epoch ([type]): [description]
    """
    def __init__(self, patience=3, minimum_improvement=0, start_epoch=10):
        self.best_metric = -np.inf  # -validation loss, or anything
        self.best_epoch = 0
        self.patience = patience
        self.minimum_improvement = minimum_improvement
        self.start_epoch = start_epoch

    def step(self, epoch, metric):
        """
        Monitor the metric to determine early stop

        Args:
            epoch (int): current epoch number
            metric (float): any performance metric, higher = better

        Returns:
            stop (bool), not_improved_epochs (int): [description]
        """
        # metric better, store best
        if metric > self.best_metric + self.minimum_improvement:
            self.best_metric = metric
            self.best_epoch = epoch
        # minimum training epochs not reached, continue
        if epoch < self.start_epoch:
            return False, epoch - self.best_epoch
        # beyond patience, early stop
        if epoch - self.best_epoch >= self.patience:
            return True, epoch - self.best_epoch
        # nothing happens, continue
        return False, epoch - self.best_epoch
        