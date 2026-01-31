import os
import sys
from functools import wraps
from pathlib import Path
import re
import random
import numpy as np
import torch
import argparse
from typing import Callable


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def print_formatted_dict(d: dict) -> None:
    for key, value in d.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def int_or_str_for_argparse(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        if value.lower() == "all":
            return value.lower()
        else:
            raise argparse.ArgumentTypeError("Value must be an integer or 'all'")


def select_best_metrics(
    metrics: dict, target_mode: str = "test", target_metric: str = "acc"
) -> dict:
    best_metrics = {}

    # Find the epoch with the best target metric
    target_metric_values = metrics[target_mode][target_metric]
    best_epoch_index = target_metric_values.index(max(target_metric_values))
    best_metrics["best_epoch"] = best_epoch_index + 1  # Epoch starts from 1

    # Now gather metrics from all modes for this epoch
    for mode, mode_metrics in metrics.items():
        for metric_name, metric_values in mode_metrics.items():
            best_metrics[f"{mode}_{metric_name}"] = metric_values[best_epoch_index]

    return best_metrics


class EarlyStopping:
    def __init__(
        self,
        patience: int = 7,
        verbose: bool = True,
        delta: float = 0,
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.val_loss_min = float("inf")
        self.delta = delta
        self.early_stop = False

    def __call__(
        self,
        val_loss: float,
        model: torch.nn.Module,
        checkpoint_path: str | None = None,
    ) -> None:
        if self.val_loss_min - val_loss < self.delta:
            # Not improved enough (need to improve at least delta)
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter} out of {self.patience} "
                    f"(best val_loss: {self.val_loss_min:.6f}, current val_loss: {val_loss:.6f}"
                    f", delta: {self.val_loss_min - val_loss:.6f} < {self.delta:.6f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improved enough
            if self.verbose:
                print(
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
                )
            self.save_checkpoint(model, checkpoint_path)
            self.val_loss_min = val_loss
            self.counter = 0

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_path: str | None = None,
    ) -> None:
        if checkpoint_path:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
