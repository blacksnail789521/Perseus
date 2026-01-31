import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


class ConvBNReLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=1, activation="relu"
    ):
        """
        A block that applies padding, a Conv1d, a nonlinearity, and BatchNorm1d.
        """
        super().__init__()
        # Calculate padding so that the convolution output length matches the input (when stride=1)
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.layers = nn.Sequential(
            nn.ConstantPad1d(padding=(padding, padding), value=0),
            nn.Conv1d(
                in_channels, out_channels, kernel_size, dilation=dilation, bias=True
            ),
            nn.ReLU() if activation == "relu" else nn.Identity(),
            nn.BatchNorm1d(out_channels),
        )
        # Initialize convolution weights
        nn.init.xavier_uniform_(self.layers[1].weight)
        nn.init.zeros_(self.layers[1].bias)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(
        self,
        filters=[16, 32, 64, 128],
        in_channels=5,
        maxpool_kernels=[10, 8, 6, 4],
        kernel_size=5,
        dilation=2,
    ):
        """
        The encoder applies several ConvBNReLU blocks followed by max-pooling.
        """
        super().__init__()
        assert len(filters) == len(
            maxpool_kernels
        ), f"Number of filters ({len(filters)}) must equal the number of maxpool kernel sizes ({len(maxpool_kernels)})!"
        self.depth = len(filters)

        # Create encoder blocks. Each block has two convolutional layers.
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNReLU(
                        in_channels=in_channels if i == 0 else filters[i - 1],
                        out_channels=filters[i],
                        kernel_size=kernel_size,
                        dilation=dilation,
                    ),
                    ConvBNReLU(
                        in_channels=filters[i],
                        out_channels=filters[i],
                        kernel_size=kernel_size,
                        dilation=dilation,
                    ),
                )
                for i in range(self.depth)
            ]
        )

        # Use ceil_mode=True to ensure the output length is at least 1.
        self.maxpools = nn.ModuleList(
            [nn.MaxPool1d(kernel_size=k, ceil_mode=True) for k in maxpool_kernels]
        )

        # Bottom (bottleneck) part of the network
        self.bottom = nn.Sequential(
            ConvBNReLU(
                in_channels=filters[-1],
                out_channels=filters[-1] * 2,
                kernel_size=kernel_size,
            ),
            ConvBNReLU(
                in_channels=filters[-1] * 2,
                out_channels=filters[-1] * 2,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, x):
        shortcuts = []
        for block, maxpool in zip(self.blocks, self.maxpools):
            z = block(x)
            shortcuts.append(z)
            x = maxpool(z)
        encoded = self.bottom(x)
        return encoded, shortcuts


class Decoder(nn.Module):
    def __init__(
        self,
        filters=[128, 64, 32, 16],
        upsample_kernels=[4, 6, 8, 10],
        in_channels=256,
        kernel_size=5,
    ):
        """
        The decoder upsamples the bottleneck features and fuses them with
        the corresponding encoder outputs (skip connections).
        """
        super().__init__()
        self.depth = len(filters)

        # Upsample blocks: each upsamples and applies a convolution.
        self.upsamples = nn.ModuleList(
            [
                nn.Sequential(
                    # Upsample using nearest-neighbor interpolation with a fixed scale factor.
                    nn.Upsample(scale_factor=scale, mode="nearest"),
                    ConvBNReLU(
                        in_channels=in_channels if i == 0 else filters[i - 1],
                        out_channels=filters[i],
                        kernel_size=kernel_size,
                    ),
                )
                for i, scale in enumerate(upsample_kernels)
            ]
        )

        # After concatenation with the skip connection, apply a block of convolutions.
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNReLU(
                        in_channels=in_channels if i == 0 else filters[i - 1],
                        out_channels=filters[i],
                        kernel_size=kernel_size,
                    ),
                    ConvBNReLU(
                        in_channels=filters[i],
                        out_channels=filters[i],
                        kernel_size=kernel_size,
                    ),
                )
                for i in range(self.depth)
            ]
        )

    def forward(self, z, shortcuts):
        # Reverse the order of shortcuts so that the first decoder block
        # uses the deepest encoder feature map.
        for upsample, block, shortcut in zip(
            self.upsamples, self.blocks, reversed(shortcuts)
        ):
            z = upsample(z)
            # Adjust the time dimension if needed:
            if z.size(-1) != shortcut.size(-1):
                diff = shortcut.size(-1) - z.size(-1)
                if diff > 0:
                    # If upsampled z is too short, pad it.
                    z = F.pad(z, (diff // 2, diff - diff // 2))
                elif diff < 0:
                    # If upsampled z is too long, crop it.
                    crop = -diff
                    z = z[..., crop // 2 : z.size(-1) - (crop - crop // 2)]
            # Concatenate along the channel dimension
            z = torch.cat([shortcut, z], dim=1)
            z = block(z)
        return z


class U_Time(nn.Module):
    def __init__(
        self,
        args,
    ):
        """
        U-Time model for time series segmentation.
          - Input shape: (B, T, C)
          - Output shape: (B, T, K)
        """
        super().__init__()
        self.args = args
        filters = getattr(self.args, "filters", [16, 32, 64, 128])
        in_channels = self.args.C
        maxpool_kernels = getattr(self.args, "maxpool_kernels", [10, 8, 6, 4])
        kernel_size = getattr(self.args, "kernel_size", 5)
        dilation = getattr(self.args, "dilation", 2)
        K = self.args.K

        self.encoder = Encoder(
            filters=filters,
            in_channels=in_channels,
            maxpool_kernels=maxpool_kernels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        # The decoder uses reversed filters and maxpool kernels.
        self.decoder = Decoder(
            filters=filters[::-1],
            upsample_kernels=maxpool_kernels[::-1],
            in_channels=filters[-1] * 2,
            kernel_size=kernel_size,
        )
        # Final 1x1 convolution to map to K.
        self.final_conv = nn.Conv1d(
            in_channels=filters[0], out_channels=K, kernel_size=1
        )
        nn.init.xavier_uniform_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x):
        # Expect input shape (B, T, C); convert to (B, C, T) for Conv1d.
        x = x.permute(0, 2, 1)
        encoded, shortcuts = self.encoder(x)
        decoded = self.decoder(encoded, shortcuts)
        logits = self.final_conv(decoded)
        # Convert back to shape (B, T, K)
        logits = logits.permute(0, 2, 1)
        return logits


if __name__ == "__main__":
    # Define arguments
    args = argparse.Namespace(
        # filters=[16, 32, 64, 128],
        # maxpool_kernels=[10, 8, 6, 4],
        # kernel_size=5,
        # dilation=2,
        K=5,
        C=3,
    )

    B = 2
    T = 250
    C = 3

    # Create a random input tensor of shape (B, T, C)
    x = torch.randn(B, T, C)

    # Instantiate the model
    model = U_Time(args)

    # Forward pass: output shape will be (B, T, K)
    output = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
