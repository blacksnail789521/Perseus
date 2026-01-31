from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
import numpy as np
from rich.table import Table
from rich.console import Console
import warnings
import argparse
import pandas as pd
import random
from typing import Iterator

from dataset_loader.dataset_tools import (
    show_dfs_info,
    show_dataset_stats,
    plot_x_y,
    get_label_mapping_with_encoder,
    print_mapping,
)


class Dataset_Segmentation(Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        mode: str = "train",
        data_name: str = "Pump_V35",
        granularity_level: int = 0,
    ) -> None:
        super().__init__()
        self.args = args
        self.granularity_level = granularity_level
        self.data_folder = Path(args.root_folder / "dataset" / data_name)

        # Make sure granularity_level is int larger than 0
        assert granularity_level >= 0 and isinstance(
            granularity_level, int
        ), "granularity_level must be an int larger than 0"

        # Get label encoder (from all data)
        self.get_label_encoder()

        # Read all dataframes
        csv_files = sorted(self.data_folder.glob("*.csv"))
        if getattr(args, "only_one_df", False):
            csv_files = [csv_files[0]]
        dfs = self.read_dfs(csv_files)

        # Separate x and y, and encode the labels with target granularity_level
        dfs_x, dfs_y = [], []
        for df in dfs:
            # x
            df_x = df.iloc[:, :-1].copy()
            dfs_x.append(df_x)

            # y
            df_y = df.iloc[:, -1:].copy()
            df_y["state"] = self.label_encoder.transform(df_y["state"])
            dfs_y.append(df_y)
        if mode == "train" or mode == "all":
            print(
                f"## granularity_level: {self.granularity_level} ## "
                f"group_size: {args.group_size} ##"
            )
            show_dfs_info(
                dfs_x,
                dfs_y,
                data_name,
                self.granularity_level,
            )  # just show for the first time

        # Normalize the features
        mean = np.mean(np.concatenate([df_x.values for df_x in dfs_x], axis=0), axis=0)
        std = np.std(np.concatenate([df_x.values for df_x in dfs_x], axis=0), axis=0)
        for df_x in dfs_x:
            df_x.loc[:, :] = (df_x.values - mean) / std

        # Create x (features) and y (labels) for each df_x, df_y
        # Also need to create timestamp and df_id
        self.x = []
        self.y = []
        self.timestamp, self.df_id = [], []
        for df_id, (df_x, df_y) in enumerate(zip(dfs_x, dfs_y)):
            # Downsample if needed
            if args.downsample_rate > 1:
                df_x = df_x.iloc[:: args.downsample_rate, :]  # select every nth row
                assert len(df_x) >= self.args.subseq_len, "Downsampled df is too short"
                df_y = df_y.iloc[:: args.downsample_rate, :]

            # Get features, labels, and timestamp
            features = df_x.values
            labels = df_y["state"].values
            timestamps = df_x.index.values

            # Use subsequence to create self.x, self.y, and self.timestamp
            for i in range(
                0, len(df_x) - self.args.subseq_len + 1, self.args.subseq_len
            ):
                self.x.append(features[i : i + self.args.subseq_len])
                self.y.append(labels[i : i + self.args.subseq_len])
                self.timestamp.append(timestamps[i : i + self.args.subseq_len])

                # Add df_id
                self.df_id.append(df_id)

        # Select samples based on mode (train: 70%, val: 15%, test: 15%)
        if mode == "train":
            self._select_split(0.0, 0.7)
        elif mode == "val":
            self._select_split(0.7, 0.85)
        elif mode == "test":
            self._select_split(0.85, 1.0)
        elif mode == "all":
            # **Do nothing; keep 100% of data**
            pass
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _select_split(self, start_frac, end_frac):
        start_idx = int(len(self.x) * start_frac)
        end_idx = int(len(self.x) * end_frac)
        self.x = self.x[start_idx:end_idx]
        self.y = self.y[start_idx:end_idx]
        self.timestamp = self.timestamp[start_idx:end_idx]
        self.df_id = self.df_id[start_idx:end_idx]

    def get_label_encoder(self) -> None:
        # Read all dataframes
        dfs = self.read_dfs(sorted(self.data_folder.glob("*.csv")))

        # Make sure all dfs have the same columns
        assert all(
            list(df.columns) == list(dfs[0].columns) for df in dfs
        ), "All dataframes must have the same columns"

        # Get all labels (we know they must be in the last column with the type int)
        y_all = sum([list(map(int, df.iloc[:, -1].values)) for df in dfs], [])

        # Encode the labels with decreasing granularity
        self.label_mapping, self.label_encoder = get_label_mapping_with_encoder(
            y_all, self.args.group_size, self.granularity_level
        )

    def read_dfs(self, selected_files: list[Path]) -> list[pd.DataFrame]:
        dfs = []
        for csv_file in selected_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        assert len(dfs) > 0, "No dataframes read"
        return dfs

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # 1) Fetch the stored subsequence (non-overlapping across the long sequence)
        subseq_x_np = self.x[idx]  # [subseq_len, C]
        subseq_y_np = self.y[idx]  # [subseq_len]
        subseq_len = int(len(subseq_y_np))

        # 2) Windowing params
        window_len = int(self.args.window_len)
        window_stride = int(self.args.window_stride)

        # Validate compatibility (your config should ensure this)
        assert subseq_len >= window_len, "subseq_len must be >= window_len"
        assert (subseq_len - window_len) % window_stride == 0, (
            "(subseq_len - window_len) must be divisible by window_stride; "
            f"got {subseq_len=}, {window_len=}, {window_stride=}"
        )

        num_windows = 1 + (subseq_len - window_len) // window_stride

        # 3) Build overlapped sliding windows within the subsequence
        win_x_list, win_y_list = [], []
        for win_idx in range(num_windows):
            window_start = win_idx * window_stride
            window_end = window_start + window_len
            win_x_list.append(subseq_x_np[window_start:window_end])  # [window_len, C]
            win_y_list.append(subseq_y_np[window_start:window_end])  # [window_len]

        # 4) Convert to tensors
        subseq_x = torch.as_tensor(subseq_x_np, dtype=torch.float32)  # [subseq_len, C]
        subseq_y = torch.as_tensor(subseq_y_np, dtype=torch.long)  # [subseq_len]
        win_x = torch.stack(
            [torch.as_tensor(w, dtype=torch.float32) for w in win_x_list], dim=0
        )  # [num_windows, window_len, C]
        win_y = torch.stack(
            [torch.as_tensor(w, dtype=torch.long) for w in win_y_list], dim=0
        )  # [num_windows, window_len]

        return subseq_x, subseq_y, win_x, win_y, self.granularity_level


class InterleavedDataLoader:
    def __init__(self, dataloaders: list[DataLoader]) -> None:
        self.dataloaders = dataloaders
        self.iterators = []
        self.current_index = 0

    def __iter__(self) -> "InterleavedDataLoader":
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
        self.current_index = 0
        return self

    def __next__(self) -> any:  # type: ignore
        if not self.iterators:
            raise StopIteration

        if self.current_index >= len(self.iterators):
            self.current_index = 0

        while self.iterators:
            try:
                batch = next(self.iterators[self.current_index])
                self.current_index += 1
                return batch
            except StopIteration:
                del self.iterators[self.current_index]
                if not self.iterators:
                    raise StopIteration
                if self.current_index >= len(self.iterators):
                    self.current_index = 0

    def __len__(self) -> int:
        return sum(len(dataloader) for dataloader in self.dataloaders)


def load_dataloader(
    args: argparse.Namespace, train_shuffle: bool = True
) -> tuple[InterleavedDataLoader, InterleavedDataLoader, InterleavedDataLoader, dict]:
    assert args.data_name in [
        "Pump_V35",
        "Pump_V36",
        "Pump_V38",
        "MoCap",
        "ActRecTut",
        "USC-HAD",
        "PAMAP2",
        "_confidential",
    ], "Invalid data_name"

    assert args.window_len <= args.subseq_len, "window_len must be <= subseq_len"
    assert args.window_stride <= args.window_len, "window_stride must be <= window_len"

    if args.enable_transfer_learning:
        print(
            "Transfer learning enabled: \n"
            f"{args.source_data_name} (granularity_level: {args.source_granularity_levels})\n"
            "--> \n"
            f"{args.target_data_name} (granularity_level: {args.target_granularity_levels})"
        )

        # Load source dataset (for training & validation)
        train_loaders, val_loaders = [], []
        for granularity_level in args.source_granularity_levels:
            train_dataset = Dataset_Segmentation(
                args,
                mode="all",
                data_name=args.source_data_name,
                granularity_level=granularity_level,
            )
            val_dataset = Dataset_Segmentation(
                args,
                mode="val",
                data_name=args.source_data_name,
                granularity_level=granularity_level,
            )

            train_loaders.append(
                DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=train_shuffle
                )
            )
            val_loaders.append(
                DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            )

        # Load target dataset (for testing)
        test_loaders = []
        for granularity_level in args.target_granularity_levels:
            test_dataset = Dataset_Segmentation(
                args,
                mode="all",
                data_name=args.target_data_name,
                granularity_level=granularity_level,
            )
            test_loaders.append(
                DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            )

    else:
        print(f"Standard learning mode: {args.data_name}")

        train_loaders, val_loaders, test_loaders = [], [], []
        for granularity_level in args.granularity_levels:
            train_dataset = Dataset_Segmentation(
                args,
                mode="train",
                data_name=args.data_name,
                granularity_level=granularity_level,
            )
            val_dataset = Dataset_Segmentation(
                args,
                mode="val",
                data_name=args.data_name,
                granularity_level=granularity_level,
            )
            test_dataset = Dataset_Segmentation(
                args,
                mode="test",
                data_name=args.data_name,
                granularity_level=granularity_level,
            )

            train_loaders.append(
                DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=train_shuffle
                )
            )
            val_loaders.append(
                DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            )
            test_loaders.append(
                DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            )

    combined_train_loader = InterleavedDataLoader(train_loaders)
    combined_val_loader = InterleavedDataLoader(val_loaders)
    combined_test_loader = InterleavedDataLoader(test_loaders)

    print("----------------------------------------")
    print(f"### Dataset Information ###")
    class_distributions = show_dataset_stats(
        ConcatDataset([loader.dataset for loader in train_loaders]),
        ConcatDataset([loader.dataset for loader in val_loaders]),
        ConcatDataset([loader.dataset for loader in test_loaders]),
        show_K=False,
    )

    return (
        combined_train_loader,
        combined_val_loader,
        combined_test_loader,
        class_distributions,
    )
