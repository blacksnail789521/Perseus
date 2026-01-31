from torch.utils.data import ConcatDataset, Dataset
from rich.table import Table
from rich.console import Console
import warnings
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from copy import deepcopy
from collections import defaultdict

def show_dfs_info(
    dfs_x: list[pd.DataFrame],
    dfs_y: list[pd.DataFrame],
    data_name: str,
    granularity_level: int,
) -> None:
    C_list, T_list, K_list = [], [], []
    N = 0
    state_length_list = []
    states_set = set()

    for df_x, df_y in zip(deepcopy(dfs_x), deepcopy(dfs_y)):
        T, C = df_x.shape
        C_list.append(C), T_list.append(T)  # type: ignore
        N += 1

        K = len(df_y["state"].unique())
        K_list.append(K)

        # accumulate all unique states
        states_set |= set(df_y["state"].unique())

        # Detect changes by shifting the 'states' column and comparing with the original
        df_y["shifted"] = df_y["state"].shift(1) != df_y["state"]
        df_y["group"] = df_y["shifted"].cumsum()

        # Count the lengths of each group
        state_lengths = df_y.groupby("group")["state"].count().tolist()
        state_length_list.extend(state_lengths)

    assert all(
        C == C_list[0] for C in C_list
    ), "All dataframes must have the same number of columns"
    print(f"C: {C_list[0]}")
    print(f"T: {min(T_list):,} ~ {max(T_list):,}")
    print(f"N: {N}")
    print(f"--- Granularity {granularity_level} [{data_name}] ---")
    if min(K_list) == max(K_list):
        print(f"K: {K_list[0]}")
    else:
        print(f"K: {min(K_list)} ~ {max(K_list)}")
    print(f"State lengths: {min(state_length_list):,} ~ {max(state_length_list):,}")
    print(f"States: {sorted(states_set)}")
    print(f"Avg state length: {int(np.mean(state_length_list)):,}")


def show_dataset_stats(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    check_mean_std: bool = False,
    show_K: bool = True,
) -> dict:
    # * Combine all datasets
    all_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    # * N
    print(
        f"N: {len(all_dataset)} (train: {len(train_dataset)}, "  # type: ignore
        f"val: {len(val_dataset)}, test: {len(test_dataset)})"  # type: ignore
    )

    # * C
    print(f"C: {all_dataset[0][0].shape[1]}")

    # * K
    if show_K:
        unique_labels = set()
        for item in all_dataset:
            unique_labels.add(item[1].item())
        print(f"K: {len(unique_labels)}")

    # * T
    print(f"T: {all_dataset[0][0].shape[0]}")

    # * Show the mean and std of all the samples in the training set
    if check_mean_std:
        x_list = []  # [(1, T, C), ...]
        for idx in range(len(train_dataset)):  # type: ignore
            x = train_dataset[idx][0]
            if isinstance(x, np.ndarray):
                x = torch.tensor(x)
            x_list.append(x.unsqueeze(0))
        all_x = torch.cat(x_list, dim=0)  # (N, T, C)
        means = all_x.mean(dim=[0, 1])  # (C,)
        stds = all_x.std(dim=[0, 1])  # (C,)
        print("Mean:", means.numpy())
        print("Std:", stds.numpy())
        # assert torch.allclose(
        #     means, torch.zeros_like(means), atol=1
        # ), f"Mean is not 0 but {means}"
        # assert torch.allclose(
        #     stds, torch.ones_like(stds), atol=1
        # ), f"Std is not 1 but {stds}"
        if not torch.allclose(means, torch.zeros_like(means), atol=1):
            warnings.warn(f"Mean is not 0 but {means}")
        if not torch.allclose(stds, torch.ones_like(stds), atol=1):
            warnings.warn(f"Std is not 1 but {stds}")

    # Class Distribution Analysis for each dataset
    if show_K:
        console = Console()
        datasets = {
            "Train": train_dataset,
            "Validation": val_dataset,
            "Test": test_dataset,
        }
        unique_labels = set()
        for dataset in datasets.values():
            for _, label in dataset:
                label = label.item() if isinstance(label, torch.Tensor) else label
                unique_labels.add(label)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Class", justify="right")
        for name in datasets:
            table.add_column(name, justify="right")

        class_distributions = {
            label: {name: 0 for name in datasets} for label in unique_labels
        }
        for name, dataset in datasets.items():
            for _, label in dataset:
                label = label.item() if isinstance(label, torch.Tensor) else label
                class_distributions[label][name] += 1

        for label in sorted(unique_labels):
            row = [str(label)]
            for name in datasets:
                total_samples = len(datasets[name])  # type: ignore
                count = class_distributions[label][name]
                percentage = count / total_samples * 100
                row.append(f"{count} ({percentage:.2f}%)")
            table.add_row(*row)

        console.print("\nClass Distribution Across Datasets:")
        console.print(table)

        return class_distributions
    else:
        return {}


def update_args_from_dataset(args: argparse.Namespace) -> argparse.Namespace:
    # Set args based on data_name
    if args.data_name in ["Pump_V35", "Pump_V36", "Pump_V38"]:
        args.C = 9
        args.K = 43
    elif args.data_name == "MoCap":
        args.C = 4
        args.K = 9
    elif args.data_name == "ActRecTut":
        args.C = 23
        args.K = 6
    elif args.data_name == "USC-HAD":
        args.C = 6
        args.K = 12
    elif args.data_name == "PAMAP2":
        args.C = 9
        args.K = 13
    elif args.data_name == "_confidential":
        args.C = 3
        args.K = 5
    else:
        raise NotImplementedError(
            f"Data name '{args.data_name}' is not implemented yet."
        )

    # Set group_size and downsample_rate into args if not set
    if not hasattr(args, "group_size"):
        args.group_size = 2
    if not hasattr(args, "downsample_rate"):
        args.downsample_rate = 1

    # Set enable_transfer_learning if not set
    if not hasattr(args, "enable_transfer_learning"):
        args.enable_transfer_learning = False

    # If transfer learning is enabled, set source_granularity_levels as the granularity_levels
    if args.enable_transfer_learning:
        args.granularity_levels = args.source_granularity_levels

    return args


def plot_x_y(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_idx: int = 0,
    args: argparse.Namespace | None = None,
    granularity_level: int = 0,
) -> None:
    # Create a figure with two subplots (vertically arranged)
    plt.figure(figsize=(10, 10))

    # Plot x on the first subplot
    plt.subplot(2, 1, 1)  # (rows, columns, index of this subplot)
    plt.plot(x.numpy(), label="x")  # Convert tensor to numpy array for plotting
    plt.xlabel("Index")
    plt.ylabel("Value of x")
    plt.title("Plot of x")
    # plt.legend()

    # Plot y on the second subplot
    plt.subplot(2, 1, 2)
    plt.plot(y.numpy(), label="y")  # Convert tensor to numpy array for plotting
    plt.xlabel("Index")
    plt.ylabel("Value of y")
    plt.title("Plot of y")
    # plt.legend()

    # Save the figure
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    if args is None:
        plot_path = Path("plot", f"plot_{batch_idx}.png")
    else:
        plot_path = Path(
            "plot",
            f"[{args.data_name}] (granularity_level={granularity_level}, "
            f"group_size={args.group_size}) plot_{batch_idx}.png",
        )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)


def merge_labels(
    labels: list[int], group_size: int = 2, num_increase_granularity: int = 1
) -> dict[int, int]:
    current_labels = labels[
        :
    ]  # Copy the initial labels to avoid modifying the original list
    final_mapping = {label: label for label in labels}  # Start with an identity mapping

    # If we don't need to decrease granularity, we need to at least make label start from 0
    if num_increase_granularity == 0:
        num_increase_granularity = 1
        group_size = 1

    for _ in range(num_increase_granularity):
        # Identify unique labels in the order of appearance
        unique_labels = []
        seen = set()
        for label in current_labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)

        # Create a new label mapping, merging 'group_size' consecutive unique labels
        label_mapping = {}
        new_label = 0
        for i in range(0, len(unique_labels), group_size):
            # Group 'group_size' labels together
            for j in range(group_size):
                if i + j < len(unique_labels):
                    label_mapping[unique_labels[i + j]] = new_label
            new_label += 1

        # Update current labels to the new mapped labels
        current_labels = [label_mapping[label] for label in current_labels]

        # Update final mapping to reflect this change
        final_mapping = {k: label_mapping.get(v, v) for k, v in final_mapping.items()}

    return final_mapping


def create_label_encoder(label_mapping: dict[int, int]) -> LabelEncoder:
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(list(label_mapping.keys()))
    label_encoder.transform = lambda x: np.array([label_mapping[label] for label in x])  # type: ignore

    return label_encoder


def get_label_mapping_with_encoder(
    labels: list[int], group_size: int = 2, granularity_level: int = 1
) -> tuple[dict[int, int], LabelEncoder]:
    # Get label_mapping
    label_mapping = merge_labels(labels, group_size, granularity_level)

    # Create label_encoder
    label_encoder = create_label_encoder(label_mapping)

    return label_mapping, label_encoder


def print_mapping(label_encoder: LabelEncoder) -> None:
    print(
        "Label mapping:",
        dict(
            zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))  # type: ignore
        ),
    )
