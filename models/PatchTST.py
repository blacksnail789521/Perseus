import torch
import torch.nn as nn
from pathlib import Path
from einops import rearrange
import argparse

from layers.Embed import DataEmbedding, Patching
from layers.RevIN import RevIN
from layers.einops_modules import RearrangeModule
from models._load_encoder import load_encoder


class PatchTST(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        # self.args.enable_channel_independence = True
        self.args.enable_channel_independence = False

        # RevIN (without affine transformation)
        self.revin = RevIN(self.args.C, affine=False)

        # Input layer
        self._set_input_layer()

        # Encoder
        self.encoder = load_encoder(self.args)

        # Output layer
        self._set_output_layer()
        self.segment_projection = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.args.C, self.args.K)
        )

    def _set_input_layer(self) -> None:
        self.patching = Patching(
            self.args.patch_len,
            self.args.patch_stride,
            self.args.enable_channel_independence,
        )  # (B, T_in, C) -> (B * C, T_p, P) (Enable CI) or (B, T_p, C * P) (Disable CI)
        if self.args.enable_channel_independence:
            self.input_layer = DataEmbedding(
                last_dim=self.args.patch_len,
                d_model=self.args.d_model,
                dropout=self.args.dropout,
                pos_embed_type=getattr(self.args, "pos_embed_type", "fixed"),
                token_embed_type=getattr(self.args, "token_embed_type", "linear"),
                token_embed_kernel_size=getattr(
                    self.args, "token_embed_kernel_size", 3
                ),
            )  # (B * C, T_p, P) -> (B * C, T_p, D)
        else:
            self.input_layer = DataEmbedding(
                last_dim=self.args.C * self.args.patch_len,
                d_model=self.args.d_model,
                dropout=self.args.dropout,
                pos_embed_type=getattr(self.args, "pos_embed_type", "fixed"),
                token_embed_type=getattr(self.args, "token_embed_type", "linear"),
                token_embed_kernel_size=getattr(
                    self.args, "token_embed_kernel_size", 3
                ),
            )  # (B, T_p, C * P) -> (B, T_p, D)

    def _set_output_layer(self) -> None:
        if self.args.enable_channel_independence:
            self.output_layer = nn.Sequential(
                RearrangeModule(
                    "(B C) T_p D -> B (C T_p D)",
                    C=self.args.C,
                    T_p=self.args.T_p,
                    D=self.args.d_model,
                ),  # (B * C, T_p, D) -> (B, C * T_p * D)
                nn.Dropout(self.args.dropout),
                nn.Linear(
                    self.args.C * self.args.T_p * self.args.d_model,
                    self.args.window_len * self.args.C,
                ),  # (B, C * T_p * D) -> (B, T_out * C)
                RearrangeModule(
                    "B (T_out C) -> B T_out C",
                    T_out=self.args.window_len,
                    C=self.args.C,
                ),  # (B, T_out * C) -> (B, T_out, C)
            )  # (B * C, T_p, D) -> (B, T_out, C)
        else:
            self.output_layer = nn.Sequential(
                RearrangeModule(
                    "B T_p D -> B (T_p D)",
                    T_p=self.args.T_p,
                    D=self.args.d_model,
                ),  # (B, T_p, D) -> (B, T_p * D)
                nn.Dropout(self.args.dropout),
                nn.Linear(
                    self.args.T_p * self.args.d_model, self.args.window_len * self.args.C
                ),  # (B, T_p * D) -> (B, T_out * C)
                RearrangeModule(
                    "B (T_out C) -> B T_out C",
                    T_out=self.args.window_len,
                    C=self.args.C,
                ),  # (B, T_out * C) -> (B, T_out, C)
            )  # (B, T_p, D) -> (B, T_out, C)

    def forward(
        self, x: torch.Tensor, prompt: dict | None = None
    ) -> torch.Tensor:  # (B, T_in, C)
        B, T_in, C = x.shape

        # Instance Normalization
        x = self.revin(x, "norm")  # (B, T_in, C)

        # Patching
        x = self.patching(
            x
        )  # (B * C, T_p, P) [Enable CI] or (B, T_p, C * P) [Disable CI]

        # Model (input_layer -> encoder -> output_layer -> segment_projection)
        x = self.input_layer(
            x
        )  # (B * C, T_p, D) [Enable CI] or (B, T_p, D) [Disable CI]
        z = self.encoder(x)  # (B * C, T_p, D) [Enable CI] or (B, T_p, D) [Disable CI]
        y = self.output_layer(z)  # (B, T_out, C)
        y = self.segment_projection(y)  # (B, T_out, K)

        return y
