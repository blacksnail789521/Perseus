import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from einops import rearrange
import argparse
import math

from layers.Embed import DataEmbedding, Patching
from layers.RevIN import RevIN
from layers.einops_modules import RearrangeModule
from models._load_encoder import load_encoder


class MemoryBank:
    """
    Minimal persistent memory:
      - Stores tokens in a single [N, D] tensor.
      - capacity: math.inf/None = infinite; int K = keep last K tokens (FIFO).
      - Writes are detached, no gradients across windows.
    """

    def __init__(
        self,
        feature_dim: int,
        device: str = "cuda",
        capacity: int | float | None = math.inf,
    ):
        self.D = int(feature_dim)
        self.device = device
        self.capacity = math.inf if capacity is None else capacity
        self._bank = torch.empty(0, self.D, device=self.device)

    def reset(self):
        """Clear all memories."""
        self._bank = torch.empty(0, self.D, device=self.device)

    @torch.no_grad()
    def write(self, tokens: torch.Tensor | None):
        """Append tokens [N_new, D] (detached, moved to device)."""
        if tokens is None or tokens.numel() == 0:
            return
        t = tokens.detach().to(self.device)
        assert t.size(-1) == self.D, f"Expected D={self.D}, got {t.size(-1)}"
        self._bank = torch.cat([self._bank, t], dim=0)

        # FIFO if capacity is finite
        if isinstance(self.capacity, (int, float)) and math.isfinite(self.capacity):
            K = max(0, int(self.capacity))
            if K == 0:
                self._bank = torch.empty(0, self.D, device=self.device)
            elif self._bank.size(0) > K:
                self._bank = self._bank[-K:]

    def read(self) -> torch.Tensor:
        """Return all memories [N, D] (may be empty)."""
        return self._bank


class MemoryBankBatch:
    """
    Batch wrapper over B independent MemoryBank's (one per subsequence).

    Minimal API:
      - reset_for_batch(B): initialize B empty banks
      - read()            : stack to [B, N, D] (N equal across banks; 0 allowed)
      - write(tokens)     : append per-subsequence tokens [B, N_new, D]
    """

    def __init__(
        self,
        feature_dim: int,
        device: str = "cuda",
        capacity: int | float | None = math.inf,
    ):
        self.D = int(feature_dim)
        self.device = device
        self.capacity = math.inf if capacity is None else capacity
        self.banks: list[MemoryBank] = []
        self._B = 0

    def reset_for_batch(self, B: int):
        """Create B empty banks for the upcoming subsequence batch."""
        self._B = int(B)
        self.banks = [
            MemoryBank(self.D, device=self.device, capacity=self.capacity)
            for _ in range(self._B)
        ]

    @torch.no_grad()
    def write(self, tokens: torch.Tensor | None):
        """
        Append CURRENT tokens per subsequence.
        tokens: [B, N_new, D] or None. Detach & FIFO handled by each MemoryBank.
        """
        if tokens is None or tokens.numel() == 0:
            return
        assert (
            tokens.dim() == 3
            and tokens.size(0) == self._B
            and tokens.size(-1) == self.D
        ), f"Expected [B, N_new, D]=[{self._B}, *, {self.D}], got {tuple(tokens.shape)}"
        t = tokens.detach().to(self.device)
        for i in range(self._B):
            self.banks[i].write(t[i])  # MemoryBank.write expects [N_new, D]

    def read(self) -> torch.Tensor:
        """
        Return stacked memories [B, N, D]. N may be 0 at k=0.
        All banks must have equal N at a given iteration.
        """
        if self._B == 0:
            return torch.empty((0, 0, self.D), device=self.device)
        Ns = [b.read().shape[0] for b in self.banks]
        assert (
            len(set(Ns)) <= 1
        ), "All subsequence banks must have the same N at this iteration."
        N = Ns[0]
        if N == 0:
            return torch.empty((self._B, 0, self.D), device=self.device)
        return torch.stack([b.read() for b in self.banks], dim=0)  # [B, N, D]

    @property
    def num_tokens(self) -> int:
        """
        Number of memory tokens stored per subsequence (all banks kept in sync).
        Returns 0 if there are no banks or they are empty.
        """
        if self._B == 0 or not self.banks:
            return 0
        Ns = [b.read().shape[0] for b in self.banks]
        if len(set(Ns)) > 1:
            raise RuntimeError("Banks have differing token counts.")
        return Ns[0]

    @property
    def total_tokens(self) -> int:
        """Total memory tokens across all subsequences (B * num_tokens)."""
        return self._B * self.num_tokens


class PromptEncoder(nn.Module):
    """
    Perseus prompt encoder: one prompt -> one token [1, D].

    Inputs (per batch of P prompts per subsequence):
      prompt_types  : Long [B, P] (0=label, 1=boundary)
      label_vec     : Float [B, P, 2K]    (valid iff type==label; else zeros)
      boundary_bits : Long  [B, P] {0,1}  (valid iff type==boundary; else zeros)
      aspect_bits   : Long  [B, P] {0=correct, 1=incorrect}  (explicit flag)

    Output:
      z_p : Float [B, P, 1, D]
    """

    def __init__(self, args, use_aspect_embed: bool = True):
        super().__init__()
        self.use_aspect_embed = use_aspect_embed

        # label: Linear(2K -> D)
        self.label_proj = nn.Linear(2 * args.K, args.d_model)
        # boundary: Embedding(2 -> D)
        self.boundary_embed = nn.Embedding(2, args.d_model)
        # type id: 0=label, 1=boundary
        self.type_embed = nn.Embedding(2, args.d_model)
        # aspect id: 0=correct, 1=incorrect (optional but recommended for clarity)
        if self.use_aspect_embed:
            self.aspect_embed = nn.Embedding(2, args.d_model)

    def forward(
        self,
        prompt_types: torch.Tensor,  # [B, P] long
        label_vec: torch.Tensor,  # [B, P, 2K] float
        boundary_bits: torch.Tensor,  # [B, P] long (0/1)
        aspect_bits: torch.Tensor,  # [B, P] long (0/1)
    ) -> torch.Tensor:  # [B, P, 1, D]
        B, P = prompt_types.shape

        # branch tokens
        label_tok = self.label_proj(label_vec)  # [B,P,D]
        bnd_tok = self.boundary_embed(boundary_bits)  # [B,P,D]
        type_tok = self.type_embed(prompt_types)  # [B,P,D]
        if self.use_aspect_embed:
            asp_tok = self.aspect_embed(aspect_bits)  # [B,P,D]

        is_label = (prompt_types == 0).unsqueeze(-1)  # [B,P,1]
        z = is_label * label_tok + (~is_label) * bnd_tok

        z = z + type_tok
        if self.use_aspect_embed:
            z = z + asp_tok

        return z.unsqueeze(2)  # [B,P,1,D]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MemoryEncoder(nn.Module):
    """
    Fuse a single prompt token [*, 1, D] with its local TS context [*, T_p_ctx, D]
    using (Pre-LN) Cross-Attn + FFN, repeated N times. Output one memory token [*, 1, D].
    """

    def __init__(self, args, num_layers: int = 2, ffn_expansion: int = 4):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "ln_q": nn.LayerNorm(args.d_model),
                        "ln_kv": nn.LayerNorm(args.d_model),
                        "attn": nn.MultiheadAttention(
                            embed_dim=args.d_model,
                            num_heads=args.n_heads,
                            dropout=args.dropout,
                            batch_first=True,
                        ),  # batch_first: (B, L, D)
                        "drop_res_attn": nn.Dropout(args.dropout),
                        "ln_ff": nn.LayerNorm(args.d_model),
                        "ffn": FeedForward(
                            args.d_model, expansion=ffn_expansion, dropout=args.dropout
                        ),
                        "drop_res_ff": nn.Dropout(args.dropout),
                    }
                )
            )

    def forward(
        self,
        prompt_tok: torch.Tensor,  # [B*, 1, D]
        ts_ctx_tok: torch.Tensor,  # [B*, T_p_ctx, D]
        attn_bias: torch.Tensor | None = None,  # (unused, reserved)
    ) -> torch.Tensor:  # [B*, 1, D]
        q = prompt_tok
        k = ts_ctx_tok
        for blk in self.layers:
            # Pre-LN + Cross-Attn (Q=prompt, K/V=ctx)
            q_norm = blk["ln_q"](q)
            k_norm = blk["ln_kv"](k)
            attn_out, _ = blk["attn"](q_norm, k_norm, k_norm, need_weights=False)
            q = q + blk["drop_res_attn"](attn_out)

            # Pre-LN + FFN
            ff_in = blk["ln_ff"](q)
            q = q + blk["drop_res_ff"](blk["ffn"](ff_in))
        return q  # [B*, 1, D]


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


class TwoWayBlock(nn.Module):
    """
    One Pre-LN two-way block:
      1) TS self-attn (X)
      2) TS->Mem cross-attn: Q=X, K/V=M   (MEMORY READ)
      3) Mem self-attn (M)
      4) Mem->TS cross-attn: Q=M, K/V=X   (write-back into M)
      5) FFNs for both streams
    Note: Head reads X after stacking L blocks (see StateDecoder).
    """

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.0, ffn_expansion: int = 4
    ):
        super().__init__()
        # X stream
        self.ln_x_sa = nn.LayerNorm(d_model)
        self.x_self = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_x_sa = nn.Dropout(dropout)

        self.ln_x_ca = nn.LayerNorm(d_model)
        self.ln_m_for_x = nn.LayerNorm(d_model)
        self.x_from_m = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_x_ca = nn.Dropout(dropout)

        self.ln_x_ff = nn.LayerNorm(d_model)
        self.ff_x = FeedForward(d_model, ffn_expansion, dropout)
        self.drop_x_ff = nn.Dropout(dropout)

        # M stream
        self.ln_m_sa = nn.LayerNorm(d_model)
        self.m_self = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_m_sa = nn.Dropout(dropout)

        self.ln_m_ca = nn.LayerNorm(d_model)
        self.ln_x_for_m = nn.LayerNorm(d_model)
        self.m_from_x = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_m_ca = nn.Dropout(dropout)

        self.ln_m_ff = nn.LayerNorm(d_model)
        self.ff_m = FeedForward(d_model, ffn_expansion, dropout)
        self.drop_m_ff = nn.Dropout(dropout)

    def forward(
        self, X: torch.Tensor, M: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) X self-attn
        x_sa_in = self.ln_x_sa(X)
        x_sa_out, _ = self.x_self(x_sa_in, x_sa_in, x_sa_in, need_weights=False)
        X = X + self.drop_x_sa(x_sa_out)

        # 2) X <- (K/V=M)  (TS->Mem cross-attn: queries are X)
        x_q = self.ln_x_ca(X)
        m_kv = self.ln_m_for_x(M)
        x_ca_out, _ = self.x_from_m(x_q, m_kv, m_kv, need_weights=False)
        X = X + self.drop_x_ca(x_ca_out)

        # 3) M self-attn
        m_sa_in = self.ln_m_sa(M)
        m_sa_out, _ = self.m_self(m_sa_in, m_sa_in, m_sa_in, need_weights=False)
        M = M + self.drop_m_sa(m_sa_out)

        # 4) M <- (K/V=X)  (Mem->TS cross-attn: queries are M)
        m_q = self.ln_m_ca(M)
        x_kv = self.ln_x_for_m(X)
        m_ca_out, _ = self.m_from_x(m_q, x_kv, x_kv, need_weights=False)
        M = M + self.drop_m_ca(m_ca_out)

        # 5) FFNs
        X = X + self.drop_x_ff(self.ff_x(self.ln_x_ff(X)))
        M = M + self.drop_m_ff(self.ff_m(self.ln_m_ff(M)))
        return X, M


class StateDecoder(nn.Module):
    """
    Two-way transformer over (TS tokens, Memory tokens), stacked L times.
    Head predicts per-patch logits [B, T_p, K] then overlap-average to [B, T, K].
    """

    def __init__(self, args, ffn_expansion: int = 4):
        super().__init__()
        self.patch_len = args.patch_len
        self.patch_stride = args.patch_stride

        self.blocks = nn.ModuleList(
            [
                TwoWayBlock(
                    args.d_model,
                    args.n_heads,
                    dropout=args.dropout,
                    ffn_expansion=ffn_expansion,
                )
                for _ in range(args.n_state_decoder_blocks)
            ]
        )

        self.ln_head = nn.LayerNorm(args.d_model)
        self.head = nn.Linear(args.d_model, args.K)  # patch logits: [B, T_p, K]

        # de-patchify via conv_transpose1d overlap-add
        # weights are created on-the-fly in forward to match device/dtype

    def _overlap_average(self, patch_logits: torch.Tensor) -> torch.Tensor:
        """
        patch_logits: [B, T_p, K]
        returns:      [B, T, K], where T = (T_p-1)*stride + patch_len
        """
        B, T_p, K = patch_logits.shape
        # Move to channel-first for conv_transpose1d: [B, K, T_p]
        x = patch_logits.transpose(1, 2).contiguous()  # [B, K, T_p]
        device, dtype = x.device, x.dtype
        # per-class de-patchify kernel of ones (groups=K)
        w = torch.ones(K, 1, self.patch_len, device=device, dtype=dtype)  # [K,1,Lk]
        out = F.conv_transpose1d(x, w, stride=self.patch_stride, groups=K)  # [B, K, T]

        # build counts by transposed conv on ones
        ones = torch.ones(B, 1, T_p, device=device, dtype=dtype)
        w1 = torch.ones(1, 1, self.patch_len, device=device, dtype=dtype)
        counts = F.conv_transpose1d(ones, w1, stride=self.patch_stride)  # [B, 1, T]
        counts = torch.clamp(counts, min=1e-6)

        out = out / counts  # broadcast over class channel
        return out.transpose(1, 2).contiguous()  # -> [B, T, K]

    def forward(
        self,
        z_x: torch.Tensor,  # [B, T_p, D]  (TS tokens)
        M: torch.Tensor,  # [B, N_all, D] (Memory tokens)
    ) -> torch.Tensor:  # [B, T, K]
        X, Mem = z_x, M
        for blk in self.blocks:
            X, Mem = blk(X, Mem)
        # patch head at TS stream
        patch_logits = self.head(self.ln_head(X))  # [B, T_p, K]
        return self._overlap_average(patch_logits)  # [B, T, K]


class Perseus(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

        self.prompt_encoder = PromptEncoder(args)  # Encodes prompt information
        self.ts_encoder = TimeSeriesEncoder(args)  # Encodes time series data
        self.memory_encoder = MemoryEncoder(
            args
        )  # Encodes prompts with context into memory tokens
        self.state_decoder = StateDecoder(args)  # Decodes embeddings to states


if __name__ == "__main__":
    # Define arguments
    args = argparse.Namespace(
        n_layers=3,  # for time series encoder
        d_model=128,
        n_heads=4,
        n_state_decoder_blocks=3,
        dropout=0.1,
        K=5,  # Number of label classes
        C=3,  # Number of input channels/features
        patch_len=16,
        patch_stride=8,
    )

    # Initialize the model
    model = Perseus(args)

    batch_size = 16
    window_len = 256
    window_stride = 64
    num_windows = 8
    subseq_len = (num_windows - 1) * window_stride + window_len

    # Example inputs
    subseq_x = torch.randn(batch_size, subseq_len, args.C)  # (B, Ls, C)
    win_x = torch.randn(batch_size, num_windows, window_len, args.C)  # (B, W, T, C)

    N_all = 3
    M_all = torch.empty((batch_size, N_all, args.d_model))  # Initial empty memory

    # Forward pass
    for win_idx in range(num_windows):
        xB = win_x[:, win_idx]  # [B, T, C]
        z_x = model.ts_encoder(xB)  # [B, T_p, D]
        # raise Exception(z_x.shape)
        y_pred = model.state_decoder(z_x, M_all)  # [B, T, K]
        assert y_pred.shape == (
            batch_size,
            window_len,
            args.K,
        ), f"Unexpected y_pred shape: {y_pred.shape}, expected {(batch_size, window_len, args.K)}"
        break

    print("Perseus model test passed.")