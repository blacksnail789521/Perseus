import torch
import torch.nn as nn
import argparse
from pathlib import Path
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.einops_modules import RearrangeModule
from einops import rearrange
from tsai.all import ResBlock, TemporalConvNet


def load_transformer(args: argparse.Namespace, mask_flag: bool = False) -> nn.Module:
    return Encoder(
        [
            EncoderLayer(
                AttentionLayer(
                    FullAttention(
                        mask_flag=mask_flag,  # Encoder: False, Decoder: True
                        factor=5,  # Unused
                        attention_dropout=args.dropout,
                        output_attention=False,
                    ),
                    args.d_model,
                    args.n_heads,
                ),
                args.d_model,
                d_ff=None,  # 4 * args.d_model
                dropout=args.dropout,
                activation=getattr(args, 'activation', 'relu'),  # Default to 'relu'
            )
            for l in range(args.n_layers)
        ],
        norm_layer=torch.nn.LayerNorm(args.d_model),
    ).float()


def load_transformer_pytorch(
    args: argparse.Namespace, mask_flag: bool = False
) -> nn.Module:
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.n_heads,
            dim_feedforward=4 * args.d_model,
            dropout=args.dropout,
            activation=args.activation,
        ),
        num_layers=args.n_layers,
        norm=nn.LayerNorm(args.d_model),
    ).float()


def load_resnet(args: argparse.Namespace, kss: list[int] = [7, 5, 3]) -> nn.Module:
    class ResNet(nn.Module):
        def __init__(self, d_model, kss=[7, 5, 3]):
            super().__init__()

            self.resblock1 = ResBlock(d_model, d_model, kss=kss)
            self.resblock2 = ResBlock(d_model, d_model, kss=kss)
            self.resblock3 = ResBlock(d_model, d_model, kss=kss)

        def forward(self, x):  # (B, T, C)
            x = x.permute(0, 2, 1)  # (B, C, T)
            x = self.resblock1(x)  # (B, D/2, T)
            x = self.resblock2(x)  # (B, D, T)
            x = self.resblock3(x)  # (B, D, T)
            x = x.permute(0, 2, 1)  # (B, T, D)
            return x

    return ResNet(args.d_model, kss=kss).float()


def load_lstm(
    args: argparse.Namespace,
    hidden_size: int = 128,
    n_layers: int = 1,
    bidirectional: bool = False,
) -> nn.Module:
    class BiLSTMWithProjection(nn.Module):
        def __init__(self, C, hidden_size=6, n_layers=1, bidirectional=False):
            super().__init__()
            self.hidden_size = hidden_size
            self.n_layers = n_layers
            # Bidirectional LSTM
            self.lstm = nn.LSTM(
                C,
                hidden_size,
                n_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
            # Linear layer to project output back to input feature size
            lienar_input_size = hidden_size * 2 if bidirectional else hidden_size
            self.linear = nn.Linear(lienar_input_size, C)

        def forward(self, x):
            # Forward propagate LSTM
            output, _ = self.lstm(x)
            # Pass the output through the linear layer
            output = self.linear(output)
            return output

    return BiLSTMWithProjection(
        args.d_model,
        hidden_size=hidden_size,
        n_layers=n_layers,
        bidirectional=bidirectional,
    ).float()


def load_tcn(args: argparse.Namespace, ks: int = 7) -> nn.Module:
    return nn.Sequential(
        RearrangeModule("B T D -> B D T"),
        TemporalConvNet(
            c_in=args.d_model,
            layers=[args.d_model] * args.n_layers,
            ks=ks,
            dropout=args.dropout,
        ),
        RearrangeModule("B D T -> B T D"),
    ).float()


def load_encoder(args: argparse.Namespace, **kwargs) -> nn.Module:
    encoder_arch = getattr(args, "encoder_arch", "transformer_encoder")
    print(f"Loading encoder: {encoder_arch}")
    if encoder_arch == "transformer_encoder":
        return load_transformer(args, mask_flag=False)
    elif encoder_arch == "transformer_encoder_pytorch":
        return load_transformer_pytorch(args, mask_flag=True)
    elif encoder_arch == "transformer_decoder":
        return load_transformer(args, mask_flag=True)
    elif encoder_arch == "resnet":
        return load_resnet(args, **kwargs)
    elif encoder_arch == "tcn":
        return load_tcn(args, **kwargs)
    elif encoder_arch == "lstm":
        return load_lstm(args, bidirectional=False, **kwargs)
    elif encoder_arch == "bilstm":
        return load_lstm(args, bidirectional=True, **kwargs)
    else:
        raise ValueError(f"Unknown encoder architecture: {encoder_arch}")
