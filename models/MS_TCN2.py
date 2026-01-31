import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


class MS_TCN2(nn.Module):
    """
    Multi-Stage Temporal Convolutional Network v2 (MS-TCN2) for time series segmentation.

    This model consists of a Prediction Generation stage followed by several Refinement stages.
    The input is expected to be of shape (B, T, C) and the output will be of shape (B, T, K).
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        num_layers_PG = getattr(self.args, "num_layers_PG", 4)
        num_layers_R = getattr(self.args, "num_layers_R", 3)
        num_R = getattr(self.args, "num_R", 2)
        num_f_maps = getattr(self.args, "num_f_maps", 64)

        self.PG = Prediction_Generation(
            num_layers_PG, num_f_maps, self.args.C, self.args.K
        )
        self.Rs = nn.ModuleList(
            [
                copy.deepcopy(
                    Refinement(num_layers_R, num_f_maps, self.args.K, self.args.K)
                )
                for _ in range(num_R)
            ]
        )

    def forward(self, x):
        # x is expected to be (B, T, C); convert to (B, C, T) for Conv1d.
        x = x.permute(0, 2, 1)
        out = self.PG(x)
        for R in self.Rs:
            # Refinement stages work on a softmaxed version of the previous output.
            out = R(F.softmax(out, dim=1))
        # out is now of shape (B, num_classes, T); permute back to (B, T, num_classes).
        out = out.permute(0, 2, 1)
        return out


class Prediction_Generation(nn.Module):
    """
    The Prediction Generation stage applies an initial 1x1 convolution followed by a series of layers.
    Each layer performs two parallel dilated convolutions with different dilation rates,
    fuses their outputs via a 1x1 convolution, applies ReLU and dropout, and uses a residual connection.
    Finally, a 1x1 convolution produces the logits.
    """

    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()
        self.num_layers = num_layers
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, kernel_size=1)

        self.conv_dilated_1 = nn.ModuleList(
            [
                nn.Conv1d(
                    num_f_maps,
                    num_f_maps,
                    kernel_size=3,
                    padding=2 ** (num_layers - 1 - i),
                    dilation=2 ** (num_layers - 1 - i),
                )
                for i in range(num_layers)
            ]
        )

        self.conv_dilated_2 = nn.ModuleList(
            [
                nn.Conv1d(
                    num_f_maps, num_f_maps, kernel_size=3, padding=2**i, dilation=2**i
                )
                for i in range(num_layers)
            ]
        )

        self.conv_fusion = nn.ModuleList(
            [
                nn.Conv1d(2 * num_f_maps, num_f_maps, kernel_size=1)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C, T)
        f = self.conv_1x1_in(x)
        for i in range(self.num_layers):
            f_in = f
            dilated1 = self.conv_dilated_1[i](f)
            dilated2 = self.conv_dilated_2[i](f)
            fused = self.conv_fusion[i](torch.cat([dilated1, dilated2], dim=1))
            f = F.relu(fused)
            f = self.dropout(f)
            f = f + f_in
        out = self.conv_out(f)
        return out


class Refinement(nn.Module):
    """
    The Refinement stage refines the predictions produced by the Prediction Generation stage.
    It first applies a 1x1 convolution to adjust the number of channels, then a sequence
    of DilatedResidualLayers, and finally a 1x1 convolution produces the refined logits.
    """

    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps))
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: (B, num_classes, T)
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    """
    A dilated residual layer consists of a dilated convolution, a 1x1 convolution,
    dropout, and a residual connection.
    """

    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        # x shape: (B, in_channels, T)
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


if __name__ == "__main__":
    # # Define arguments
    args = argparse.Namespace(
        dropout=0.1,
        K=4,
        C=3,
    )

    B, T, C = 2, 100, 3
    x = torch.randn(B, T, C)

    # Instantiate MS_TCN2:
    model = MS_TCN2(args)
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)  # Expected: (B, T, K)
