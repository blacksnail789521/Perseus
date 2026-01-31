import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from einops import rearrange
import argparse

from layers.Embed import DataEmbedding, Patching
from layers.RevIN import RevIN
from layers.einops_modules import RearrangeModule
from models._load_encoder import load_encoder


class PromptTSS(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

        self.prompt_encoder = PromptEncoder(args)  # Encodes prompt information
        self.time_series_encoder = TimeSeriesEncoder(args)  # Encodes time series data
        self.state_decoder = StateDecoder(args)  # Decodes embeddings to states

    def forward(
        self, x: torch.Tensor, prompts: dict | None = None
    ) -> torch.Tensor:  # x: (B, T, C), prompt: (B, T)
        B, T, C = x.shape

        # Encode time series into embeddings
        time_series_embedding = self.time_series_encoder(x)  # Shape: (B, T_p, D)

        # Encode prompts into embeddings
        prompt_embedding = self.prompt_encoder(prompts, B, T)  # Shape: (B, T, D)

        # Decode the states using the embeddings
        states = self.state_decoder(
            prompt_embedding, time_series_embedding
        )  # Shape: (B, T, K)

        return states


class PromptEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.D = args.d_model
        self.K = args.K

        # Boundary prompt embedding layer
        self.boundary_embedding = nn.Embedding(
            2, self.D
        )  # 2: No boundary, Boundary exists

        # Label prompt embedding layer
        self.label_embedding = nn.Linear(
            2 * self.K, self.D
        )  # First K: correct, Second K: incorrect

    def forward(self, prompts: dict, B: int, T: int) -> torch.Tensor:
        device = next(self.parameters()).device  # Get the device of the model

        # Initialize boundary and label prompt tensors
        boundary_prompt = torch.full(
            (B, T), -1, dtype=torch.long, device=device
        )  # -1 indicates NA
        label_prompt = torch.zeros(
            (B, T, 2 * self.K), dtype=torch.float, device=device
        )  # Shape [B, T, 2K]

        # print("Boundary Prompt:", boundary_prompt)
        # print("Label Prompt:", label_prompt)

        # Fill the boundary and label prompts
        for key, value in prompts.items():
            if len(key) == 2:  # Boundary prompts (type, time)
                prompt_type, time_index = key
                if prompt_type == "boundary":
                    # Set boundary prompt values
                    value = torch.tensor(
                        value, device=device
                    )  # Convert values to a tensor
                    boundary_prompt[:, time_index] = value  # Shape [B]

            elif len(key) == 3:  # Label prompts (type, aspect, time)
                prompt_type, prompt_aspect, time_index = key
                if prompt_type == "label":
                    if prompt_aspect == "correct":
                        # Set one-hot correct labels
                        value = torch.tensor(
                            value, device=device
                        )  # Convert values to a tensor
                        for b in range(B):
                            label_prompt[b, time_index, value[b]] = 1
                    elif prompt_aspect == "incorrect":
                        # Set multi-hot incorrect labels
                        for b in range(B):
                            for incorrect_label in value[b]:
                                label_prompt[
                                    b, time_index, self.K + incorrect_label
                                ] = 1

        # print("Boundary Prompt:", boundary_prompt)
        # print("Label Prompt:", label_prompt)

        # Project boundary prompt
        boundary_mask = boundary_prompt != -1  # Mask for given boundary prompts
        boundary_prompt = torch.clamp(
            boundary_prompt, min=0
        )  # Replace -1 with 0 for embedding lookup
        boundary_embedding = self.boundary_embedding(boundary_prompt)  # Shape [B, T, D]
        boundary_embedding = boundary_embedding * boundary_mask.unsqueeze(
            -1
        )  # Zero out NA positions

        # Project label prompt
        label_mask = label_prompt.sum(dim=-1) > 0  # Mask for given label prompts
        label_embedding = self.label_embedding(label_prompt)  # Shape [B, T, D]
        label_embedding = label_embedding * label_mask.unsqueeze(
            -1
        )  # Zero out NA positions

        # Combine embeddings by summing them
        final_embedding = boundary_embedding + label_embedding  # Shape [B, T, D]

        return final_embedding


class TimeSeriesEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

        # RevIN (without affine transformation)
        self.revin = RevIN(self.args.C, affine=False)

        # Input layer
        self._set_input_layer()

        # Encoder
        self.encoder = load_encoder(self.args)  # transformer_encoder

    def _set_input_layer(self) -> None:
        self.patching = Patching(
            self.args.patch_len,
            self.args.patch_stride,
            enable_channel_independence=False,
        )  # (B, T, C) -> (B, T_p, C * P)
        self.input_layer = DataEmbedding(
            last_dim=self.args.C * self.args.patch_len,
            d_model=self.args.d_model,
            dropout=self.args.dropout,
            pos_embed_type=getattr(self.args, "pos_embed_type", "fixed"),
            token_embed_type=getattr(self.args, "token_embed_type", "linear"),
            token_embed_kernel_size=getattr(self.args, "token_embed_kernel_size", 3),
        )  # (B, T_p, C * P) -> (B, T_p, D)

    def forward(
        self, x: torch.Tensor, prompt: torch.Tensor | None = None
    ) -> torch.Tensor:  # (B, T, C)
        B, T, C = x.shape

        # Instance Normalization
        x = self.revin(x, "norm")  # (B, T, C)

        # Patching
        x = self.patching(x)  #  (B, T_p, C * P)

        # Model (input_layer -> encoder)
        x = self.input_layer(x)  #  (B, T_p, D)
        z = self.encoder(x)  #  (B, T_p, D)

        return z


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, skip_first_layer_pe=False):
        super().__init__()
        self.skip_first_layer_pe = skip_first_layer_pe

        # Self-attention for prompts
        self.prompt_self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention: prompts to time series
        self.prompt_to_series_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Bottleneck MLP
        self.mlp_prompt = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

        # Reverse cross-attention: time series to prompts
        self.series_to_prompt_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, z_p, z_x, query_pe=None, key_pe=None):
        """
        Args:
            z_p (torch.Tensor): Prompt embeddings, shape (B, T, D).
            z_x (torch.Tensor): Time series embeddings, shape (B, T_p, D).
            query_pe (torch.Tensor): Positional encoding for queries, shape (B, T, D).
            key_pe (torch.Tensor): Positional encoding for keys, shape (B, T_p, D).

        Returns:
            torch.Tensor, torch.Tensor: Updated prompt and time series embeddings.
        """
        # Self-attention on prompts
        if self.skip_first_layer_pe:
            z_p = self.prompt_self_attention(z_p, z_p, z_p)[0]
        else:
            z_p = self.prompt_self_attention(z_p + query_pe, z_p + query_pe, z_p)[0]
        z_p = self.norm1(z_p)

        # Cross-attention: prompts to time series
        z_p_residual = z_p
        z_p = self.prompt_to_series_attention(z_p + query_pe, z_x + key_pe, z_x)[0]
        z_p = self.norm2(z_p_residual + z_p)

        # Bottleneck MLP on prompts
        z_p_residual = z_p
        z_p = self.mlp_prompt(z_p)
        z_p = self.norm3(z_p_residual + z_p)

        # Reverse cross-attention: time series to prompts
        z_x_residual = z_x
        z_x = self.series_to_prompt_attention(z_x + key_pe, z_p + query_pe, z_p)[0]
        z_x = self.norm4(z_x_residual + z_x)

        return z_p, z_x


class StateDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.d_ff = args.d_ff
        self.n_state_decoder_blocks = args.n_state_decoder_blocks
        self.dropout = args.dropout

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, args.max_len, self.d_model)
        )

        # Two-way Transformer blocks
        self.decoder_blocks = nn.ModuleList(
            [
                TwoWayAttentionBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    skip_first_layer_pe=(i == 0),
                )
                for i in range(self.n_state_decoder_blocks)
            ]
        )

        # Final attention layer
        self.final_attention = nn.MultiheadAttention(
            self.d_model, self.n_heads, dropout=self.dropout, batch_first=True
        )
        self.norm_final_attention = nn.LayerNorm(self.d_model)

        # Output projection layer
        self.state_projection = nn.Linear(self.d_model, args.K)

    def forward(self, z_p, z_x):
        """
        Args:
            z_p (torch.Tensor): Prompt embeddings, shape (B, T, D).
            z_x (torch.Tensor): Time series embeddings, shape (B, T_p, D).

        Returns:
            torch.Tensor: Output states, shape (B, T, K).
        """
        # Apply positional encoding
        query_pe = self.positional_encoding[:, : z_p.size(1), :]
        key_pe = self.positional_encoding[:, : z_x.size(1), :]

        for block in self.decoder_blocks:
            z_p, z_x = block(z_p, z_x, query_pe=query_pe, key_pe=key_pe)

        # Final attention layer
        z_p = z_p + query_pe
        z_x = z_x + key_pe
        attn_out = self.final_attention(z_p, z_x, z_x)[0]
        z_p = self.norm_final_attention(z_p + attn_out)

        # Project to output states
        s = self.state_projection(z_p)  # Shape: (B, T, K)
        return s


if __name__ == "__main__":
    # Define arguments
    args = argparse.Namespace(
        n_layers=3,  # for time series encoder
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_state_decoder_blocks=3,
        dropout=0.1,
        max_len=2048,
        K=5,  # Number of label classes
        C=3,  # Number of input channels/features
        patch_len=16,
        patch_stride=8,
    )

    # Initialize the model
    prompt_tss = PromptTSS(args)

    # Example inputs
    B, T, C = 2, 30, 3  # Batch size, Time length, Channels
    x = torch.randn(B, T, C)  # Time series data

    # Example prompts
    prompts = {
        ("boundary", 5): [1, 0],  # Boundary prompt
        ("label", "correct", 8): [2, 1],  # Correct label prompt
        ("label", "incorrect", 10): [[0, 4], [3]],  # Incorrect label prompt
    }

    # Forward pass
    output_states = prompt_tss(x, prompts)

    print("Output Shape:", output_states.shape)  # Expected shape: (B, T, num_states)
