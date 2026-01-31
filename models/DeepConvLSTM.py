import torch
import torch.nn as nn
import argparse

from layers.einops_modules import RearrangeModule


class DeepConvLSTM(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

        # Define the Convolutional layers
        self._set_conv_layers()

        # Define the LSTM layers
        self._set_lstm_layers()

        # Define the output layer
        self._set_output_layer()

    def _set_conv_layers(self) -> None:
        self.conv_layers = nn.Sequential(
            RearrangeModule("B T C -> B C T"),
            nn.Conv1d(
                self.args.C,
                self.args.num_filters,
                self.args.filter_size,
                padding=self.args.filter_size // 2,
            ),  # (B, NF, T), where NF is the number of filters
            nn.ReLU(),
            nn.Conv1d(
                self.args.num_filters,
                self.args.num_filters,
                self.args.filter_size,
                padding=self.args.filter_size // 2,
            ),  # (B, NF, T)
            nn.ReLU(),
            nn.Conv1d(
                self.args.num_filters,
                self.args.num_filters,
                self.args.filter_size,
                padding=self.args.filter_size // 2,
            ),  # (B, NF, T)
            nn.ReLU(),
            nn.Conv1d(
                self.args.num_filters,
                self.args.num_filters,
                self.args.filter_size,
                padding=self.args.filter_size // 2,
            ),  # (B, NF, T)
            nn.ReLU(),
            RearrangeModule("B NF T -> B T NF"),
        )  # (B, T, C) -> (B, T, NF)

    def _set_lstm_layers(self) -> None:
        self.lstm = nn.LSTM(
            input_size=self.args.num_filters,
            hidden_size=self.args.d_model,
            # num_layers=self.args.n_layers,
            num_layers=2,
            batch_first=True,
        )  # (B, T, NF) -> (B, T, D)

    def _set_output_layer(self) -> None:
        self.output_layer = nn.Sequential(
            nn.Dropout(self.args.dropout),
            nn.Linear(self.args.d_model, self.args.K),  # (B, T, D) -> (B, T, K)
        )  # (B, T, D) -> (B, T, K)

    def forward(
        self, x: torch.Tensor, prompt: dict | None = None
    ) -> torch.Tensor:  # (B, T, C)
        B, T, C = x.shape

        # Convolutional layers
        x = self.conv_layers(x)  # (B, T, NF)

        # LSTM layers
        x, _ = self.lstm(x)  # (B, T, D)

        # Output layer
        x = self.output_layer(x)  # (B, T, K)

        return x
