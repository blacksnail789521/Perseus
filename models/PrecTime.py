import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from layers.RevIN import RevIN
from tsai.all import TemporalConvNet


class PrecTime(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.window = getattr(self.args, "window", 50)

        num_layers = getattr(self.args, "num_layers", 1)
        dropout = self.args.dropout
        cnn_channel = getattr(self.args, "cnn_channel", 128)
        rnn1_channel = getattr(self.args, "rnn1_channel", 100)
        rnn2_channel = getattr(self.args, "rnn2_channel", 200)

        # Normalization layer (RevIN) applied per window.
        self.revin = RevIN(self.args.C, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=-1)

        # Layer normalization applied to the context detection output.
        self.layernorm = nn.LayerNorm(rnn2_channel * 2)

        # Context detection stage using two GRU layers.
        self.rnn1 = nn.GRU(
            self.args.C,
            rnn1_channel,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.rnn2 = nn.GRU(
            rnn1_channel * 2,
            rnn2_channel,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Prediction refinement branch using TemporalConvNet modules.
        self.refine = TemporalConvNet(
            rnn2_channel * 2, [cnn_channel] * num_layers, ks=5, dropout=dropout
        )
        self.refine2 = TemporalConvNet(
            cnn_channel, [cnn_channel] * num_layers, ks=3, dropout=dropout
        )

        # Final decoder: maps the refined features to segmentation classes.
        self.decoder2 = nn.Linear(cnn_channel, self.args.K)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, C)
        Returns:
            out_final: Output tensor of shape (B, T, K) containing log-probabilities per timestep.
        """
        B, T, C = x.size()
        window = self.window
        hr_outputs = []  # list to collect outputs from each window chunk

        # Initialize separate hidden states for each GRU layer.
        hidden1 = None  # for rnn1 (shape: (num_layers*directions, B, rnn1_channel))
        hidden2 = None  # for rnn2 (shape: (num_layers*directions, B, rnn2_channel))

        # Process the time series in windows/chunks.
        # If T is not an exact multiple of window, the last chunk may be shorter.
        for i in range(0, T, window):
            # Extract a window from the time series: shape (B, L_win, C)
            x_win = x[:, i : i + window, :]
            # Normalize the current window (RevIN expects (B, L, C))
            x_win = self.revin(x_win, "norm")

            # Process through the GRU-based context detection module.
            if hidden1 is None and hidden2 is None:
                # First window: no previous hidden state.
                out, hidden1 = self.rnn1(x_win)
                out = self.dropout(out)
                out, hidden2 = self.rnn2(out)
            else:
                # Subsequent windows: use previous hidden states.
                out, hidden1 = self.rnn1(x_win, hidden1)
                out = self.dropout(out)
                out, hidden2 = self.rnn2(out, hidden2)

            # out has shape (B, L_win, 2 * rnn2_channel)
            hr_outputs.append(out)

        # Concatenate outputs along the time dimension.
        hr = torch.cat(hr_outputs, dim=1)  # shape: (B, T, 2 * rnn2_channel)

        # Apply layer normalization.
        hr = self.layernorm(hr)

        # Refined prediction branch:
        # Transpose to (B, channels, T) for TemporalConvNet modules.
        d2 = hr.transpose(1, 2)
        d2 = self.refine(d2)  # expected shape: (B, cnn_channel, T)
        d2 = self.refine2(d2)  # still (B, cnn_channel, T)
        d2 = self.dropout(d2)
        # Transpose back to (B, T, cnn_channel)
        d2 = d2.transpose(1, 2)

        # Final decoder maps each timestep's refined features to class logits.
        out_final = self.decoder2(d2)  # shape: (B, T, K)
        out_final = self.softmax(
            out_final
        )  # log-probabilities along the class dimension

        return out_final


if __name__ == "__main__":
    # Define arguments
    args = argparse.Namespace(
        # window=50,
        dropout=0.1,
        K=4,
        C=3,
    )

    # Define dummy input parameters.
    B = 2  # Batch size
    T = 250  # Number of timesteps
    C = 3  # Number of input channels/features
    window = 50  # Window length for processing chunks

    # Create a random input tensor of shape (B, T, C)
    x = torch.randn(B, T, C)

    # Instantiate the model.
    model = PrecTime(args)

    # Forward pass.
    output = model(x)

    # Print the output shape. Expected shape: (B, T, K)
    print("Output shape:", output.shape)
