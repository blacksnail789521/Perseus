import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.RevIN import RevIN


class iTransformer(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

        # RevIN (without affine transformation)
        self.revin = RevIN(self.args.C, affine=False)

        # Input layer
        self.input_layer = DataEmbedding_inverted(
            c_in=self.args.window_len,
            d_model=self.args.d_model,
            pos_embed_type=getattr(self.args, "pos_embed_type", "fixed"),
            freq=getattr(self.args, "freq", "h"),
            dropout=self.args.dropout,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            getattr(self.args, "factor", 5),  # Unused
                            attention_dropout=self.args.dropout,
                            output_attention=False,
                        ),
                        self.args.d_model,
                        self.args.n_heads,
                    ),
                    self.args.d_model,
                    getattr(self.args, "d_ff", None),
                    dropout=self.args.dropout,
                    activation=self.args.activation,
                )
                for l in range(self.args.n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.args.d_model),
        )

        # Output layer
        self.output_layer = nn.Linear(
            self.args.d_model, self.args.window_len, bias=True
        )
        self.segment_projection = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.args.C, self.args.K),
        )

    def forward(
        self, x: torch.Tensor, prompt: torch.Tensor | None = None
    ) -> torch.Tensor:  # (B, T_in, C)
        B, T_in, C = x.shape

        # Instance Normalization
        x = self.revin(x, "norm")  # (B, T_in, C)

        # Model (input_layer -> encoder -> output_layer -> segment_projection)
        x = self.input_layer(x)  # (B, C, D)
        z = self.encoder(x, attn_mask=None)  # (B, C, D)
        y = self.output_layer(z).permute(0, 2, 1)[:, :, :C]  # (B, T_out, C)
        y = self.segment_projection(y)  # (B, T_out, K)

        return y
