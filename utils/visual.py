from rich.table import Table
from rich.console import Console
import plotext as pltt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from matplotlib import pyplot as plt


STYLE_COLOR = {"train": "blue", "val": "green", "test": "red"}


def visualize_metric_by_table(history: dict) -> None:
    # Extract metrics and modes
    metrics = history["test"].keys()
    modes = history.keys()

    # Show table
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Epoch")
    for mode in modes:
        for metric in metrics:
            table.add_column(
                f"{mode.capitalize()} {metric.upper()}", style=STYLE_COLOR[mode]
            )
    for epoch in range(len(history["test"]["loss"])):
        row = [str(epoch + 1)]
        for mode in modes:
            for metric in metrics:
                value = history[mode][metric][epoch]
                row.append(f"{value:.3f}")
        table.add_row(*row)
    console.print(table)


def visualize_metric_by_plot(history: dict) -> None:
    # Extract metrics and modes
    metrics = history["test"].keys()
    modes = history.keys()

    # Show plot for each metric
    pltt.clf()  # Clear any previous plot data
    pltt.plot_size(200, 20)
    pltt.subplots(1, len(metrics))
    for i, metric in enumerate(metrics):
        epochs = [epoch + 1 for epoch in range(len(history["test"]["loss"]))]
        pltt.subplot(1, i + 1)

        # Plotting's settings
        pltt.title(f"{metric.upper()} vs Epoch")
        pltt.xticks(epochs)

        # Plot data
        for mode in modes:
            pltt.plot(
                epochs,
                history[mode][metric],
                color=STYLE_COLOR[mode],
                label=f"{mode.capitalize()} {metric.upper()}",
            )

    # Show all plots
    pltt.show()


def visualize_metric(history: dict, mode: str = "table") -> None:
    if mode == "table":
        visualize_metric_by_table(history)
    elif mode == "plot":
        visualize_metric_by_plot(history)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def plot_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    enable_show: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Show the first 5 predictions
    x, y = next(iter(test_loader))
    for i in range(5):
        plt.imshow(x[i, 0, :, :], cmap="gray")
        y_pred = model(x[i].unsqueeze(0).to(device))
        plt.title(f"Label: {y[i]}, Prediction: {torch.argmax(y_pred)}")

        # Save the figure
        plot_folder = Path("plots")
        plot_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_folder / f"prediction_{i}.png")
        if enable_show:
            plt.show()
        plt.close()
