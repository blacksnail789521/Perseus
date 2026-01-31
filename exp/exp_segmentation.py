import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from einops import rearrange
from collections import defaultdict
from typing import DefaultDict

from dataset_loader.dataset_loader import load_dataloader, InterleavedDataLoader
from models.Perseus import Perseus, MemoryBankBatch
from utils.visual import visualize_metric
from utils.tools import EarlyStopping
import random
import time
import math

MODEL_MAP = {
    "Perseus": Perseus,
}


class ReshapedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(ReshapedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, y_pred, y):
        # Reshape y_pred from (B, T, K) to (B * T, K), y from (B, T) to (B * T)
        y_pred_reshaped = rearrange(y_pred, "B T K -> (B T) K")
        y_reshaped = rearrange(y, "B T -> (B T)")

        # Calculate and return the loss
        return self.criterion(y_pred_reshaped, y_reshaped)


class Exp_Segmentation(object):
    def __init__(self, args) -> None:
        self.args = args
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._get_data()
        self._get_model()
        self._set_criterion()
        self._set_optimizer()
        self._set_early_stopping()

    def _get_data(self):
        # Load data
        self.train_loader, self.val_loader, self.test_loader, class_distributions = (
            load_dataloader(self.args)
        )

    def _get_model(self):
        # Dynamically get the model class based on the model name
        ModelClass = MODEL_MAP.get(self.args.model_name, None)
        if ModelClass is None:
            raise ValueError(f"Unknown model name: {self.args.model_name}")

        if self.args.model_name == "Perseus":
            self.model = ModelClass(self.args).to(self.args.device)
        else:
            raise Exception("We do not support non-Perseus models in this code.")

    def _set_criterion(self):
        self.criterion = ReshapedCrossEntropyLoss()

    def _set_optimizer(self):
        # optimizer
        self.optimizer = getattr(optim, self.args.optim)(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        # scheduler
        if self.args.lr_scheduler != "none":
            lr_scheduler_params = self.args.lr_scheduler_params[self.args.lr_scheduler]
            if self.args.lr_scheduler == "CyclicLR":
                lr_scheduler_params["base_lr"] = self.args.learning_rate
                lr_scheduler_params["cycle_momentum"] = (
                    True if self.args.optim == "SGD" else False
                )
            elif self.args.lr_scheduler == "OneCycleLR":
                lr_scheduler_params["steps_per_epoch"] = len(self.train_loader)
                lr_scheduler_params["epochs"] = self.args.epochs
            self.scheduler = getattr(optim.lr_scheduler, self.args.lr_scheduler)(
                self.optimizer, **lr_scheduler_params
            )

    def _set_early_stopping(self):
        self.early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=self.args.delta
        )

    @torch.no_grad()
    def sample_prompts_and_ctx(
        self,
        subseq_x: torch.Tensor,  # [B, Ls, C]
        subseq_y: torch.Tensor,  # [B, Ls]
        P: int,  # prompts per subsequence
        T_ctx: int,  # context length (WRITE)
        K: int,  # #classes
        p_label: float = 0.5,  # P(type = label)
        p_correct_label: float = 0.5,  # P(correct | label)
        p_correct_bnd: float = 0.5,  # P(correct | boundary)
        # --- global noise control ---
        p_wrong: float | None = None,  # if set, with prob p_wrong the payload is flipped vs GT (independent of declared aspect)
        # --- NEW (optional) controls for prompt placement wrt sliding windows ---
        win_len: int | None = None,  # READ window length (T)
        win_stride: int | None = None,  # READ window stride (S)
        win_hit_pct: (
            float | None
        ) = None,  # target fraction of windows containing ≥1 prompt idx (per subseq)
    ) -> tuple[
        torch.Tensor,  # prompt_idx   [B, P]  (CENTER of ctx)
        torch.Tensor,  # ts_ctx       [B, P, T_ctx, C]
        torch.Tensor,  # prompt_types [B, P] {0=label, 1=boundary}
        torch.Tensor,  # label_vec    [B, P, 2K]
        torch.Tensor,  # boundary_bits[B, P] {0,1}
        torch.Tensor,  # aspect_bits  [B, P] {0=correct, 1=incorrect}
    ]:
        """
        Returns:
            prompt_idx    : Long  [B, P]  (CENTER index of the WRITE context; i.e., the prompt timestamp)
            ts_ctx        : Float [B, P, T_ctx, C] (context crop for WRITE; safe tail padding)
            prompt_types  : Long  [B, P] {0=label, 1=boundary}
            label_vec     : Float [B, P, 2K]
            boundary_bits : Long  [B, P] {0,1}
            aspect_bits   : Long  [B, P] {0=correct, 1=incorrect}

        Notes:
        - If win_hit_pct is None (default), behavior is identical to Step 1 (uniform random starts).
        - If win_hit_pct is provided, we attempt to achieve that fraction of "hit" windows by
            constructing prompt_idx to cover approximately H_target windows, with an upper bound given by
            P * ceil(win_len / win_stride). Extra prompts (if any) are placed inside already-hit windows
            so the achieved hit ratio does not overshoot.
        """
        device = subseq_x.device
        B, Ls, C = subseq_x.shape

        # ============================================================
        # 1) CREATE prompt_idx (center timestamps)  ← ONLY CHANGED PART
        # ============================================================
        if win_hit_pct is None or win_len is None or win_stride is None:
            # Original behavior: sample start positions uniformly, then center.
            hi = max(1, Ls - T_ctx + 1)
            start_idx = torch.randint(0, hi, (B, P), device=device)  # [B,P]
            prompt_idx = torch.clamp(start_idx + (T_ctx // 2), max=Ls - 1)  # [B,P]
        else:
            # Controlled coverage wrt sliding windows
            T = int(win_len)
            S = int(win_stride)

            # window geometry
            W = 1 if Ls < T else ((Ls - T) // S + 1)
            k = ((T - 1) // S) + 1  # ~= ceil(T/S)

            # centers that keep ctx fully inside without truncation
            c_lo = T_ctx // 2
            c_hi = Ls - 1 - (T_ctx // 2)

            # feasible window range: windows whose [start,end] intersects [c_lo,c_hi]
            if W > 0 and c_lo <= c_hi:
                w_first = max(0, math.ceil((c_lo - (T - 1)) / S))  # ensure end >= c_lo
                w_last = min(W - 1, math.floor(c_hi / S))  # ensure start <= c_hi
                F = max(0, w_last - w_first + 1)  # feasible windows count
            else:
                w_first, w_last, F = 0, -1, 0

            # target & capacity (respect feasibility)
            H_target = int(round(float(win_hit_pct) * W))
            H_target = max(0, min(W, H_target))
            H_max = min(W, F, P * k)
            H_eff = min(H_target, H_max)

            # how many prompts we actually need
            M_needed = 0 if H_eff == 0 else int((H_eff + k - 1) // k)  # ceil(H_eff/k)
            M_needed = min(M_needed, P)

            prompt_list = []
            if F > 0 and M_needed > 0:
                w_ptr = w_first
                remaining = H_eff
                for _ in range(M_needed):
                    # number of windows this prompt should cover (contiguous, ≤k, stay within feasible range)
                    c_i = min(k, remaining, w_last - w_ptr + 1)
                    w_min = w_ptr
                    w_max = w_ptr + c_i - 1

                    # intersection for a valid center:
                    # it must lie in all selected windows AND in [c_lo, c_hi]
                    lower = max(w_max * S, c_lo)
                    upper = min(w_min * S + (T - 1), (w_max + 1) * S - 1, c_hi)
                    # (by construction lower <= upper)
                    t_center = int(
                        upper
                    )  # choose rightmost to avoid leaking into w_max+1

                    prompt_list.append(t_center)
                    w_ptr += c_i
                    remaining -= c_i

            # fill extra prompts by duplicating (no coverage increase)
            while len(prompt_list) < P:
                if len(prompt_list) == 0:
                    # no feasible hits; pick a safe center
                    t_center = min(max(c_lo, 0), max(Ls - 1, 0))
                else:
                    t_center = prompt_list[len(prompt_list) % len(prompt_list)]
                prompt_list.append(int(t_center))

            prompt_list = prompt_list[:P]
            prompt_idx = torch.tensor(
                prompt_list, device=device, dtype=torch.long
            ).view(1, P)
            if B > 1:
                prompt_idx = prompt_idx.expand(B, P).contiguous()

            # derive start positions for ctx crops from prompt_idx (centered)
            start_idx = torch.clamp(
                prompt_idx - (T_ctx // 2), min=0, max=max(0, Ls - T_ctx)
            )

        # make sure the returned prompt_idx matches the true ctx center after clamping
        prompt_idx = start_idx + (T_ctx // 2)

        # ============================================================
        # 2) CONTEXT CROPS (unchanged)
        # ============================================================
        ts_ctx = torch.zeros((B, P, T_ctx, C), device=device, dtype=subseq_x.dtype)
        # safe tail padding (no wrap-around)
        for b in range(B):
            for p in range(P):
                s = int(start_idx[b, p])
                e = min(s + T_ctx, Ls)
                if e > s:
                    ts_ctx[b, p, : e - s] = subseq_x[b, s:e]

        # ============================================================
        # 3) BOUNDARY MAP (unchanged)
        # ============================================================
        boundary = (subseq_y[:, 1:] != subseq_y[:, :-1]).long()  # [B, Ls-1]
        boundary = torch.cat(
            [torch.zeros(B, 1, device=device, dtype=torch.long), boundary], dim=1
        )  # [B, Ls]

        # ============================================================
        # 4) TYPES & PAYLOADS (unchanged)
        # ============================================================
        prompt_types = (
            torch.rand(B, P, device=device) > p_label
        ).long()  # 0 w.p. p_label else 1
        label_vec = torch.zeros((B, P, 2 * K), device=device, dtype=torch.float)
        boundary_bits = torch.zeros((B, P), device=device, dtype=torch.long)
        aspect_bits = torch.zeros(
            (B, P), device=device, dtype=torch.long
        )  # 0=correct, 1=incorrect

        # ---------------- LABEL prompts (aspect independent from noise) ----------------
        mask_label = prompt_types == 0
        if mask_label.any():
            ib, ip = mask_label.nonzero(as_tuple=True)
            t_lab = prompt_idx[ib, ip]                 # [Nlab]
            y_sel = subseq_y[ib, t_lab]                # [Nlab] in [0..K-1]

            # annotator-declared aspect (independent of noise)
            decl_is_correct = torch.rand_like(y_sel.float()) < float(p_correct_label)
            decl_is_incorrect = ~decl_is_correct

            # we’ll sample a wrong flag per prompt if requested
            if p_wrong is not None and p_wrong > 0.0:
                wrong_flag = (torch.rand_like(y_sel.float()) < float(p_wrong))
            else:
                wrong_flag = torch.zeros_like(y_sel, dtype=torch.bool)

            # ---- declared CORRECT ----
            if decl_is_correct.any():
                idx = decl_is_correct.nonzero(as_tuple=True)[0]
                b, p, y = ib[idx], ip[idx], y_sel[idx]
                w = wrong_flag[idx]

                # not wrong: payload = GT in first half
                if (~w).any():
                    i = (~w).nonzero(as_tuple=True)[0]
                    b0, p0, y0 = b[i], p[i], y[i]
                    label_vec[b0, p0, y0] = 1.0
                    aspect_bits[b0, p0] = 0

                # wrong: payload = non-GT in first half
                if w.any():
                    i = w.nonzero(as_tuple=True)[0]
                    b1, p1, y1 = b[i], p[i], y[i]
                    r = torch.randint(0, K - 1, (y1.numel(),), device=device)
                    inc = r + (r >= y1).long()          # ensure inc != y1
                    label_vec[b1, p1, inc] = 1.0
                    aspect_bits[b1, p1] = 0

            # ---- declared INCORRECT ----
            if decl_is_incorrect.any():
                idx = decl_is_incorrect.nonzero(as_tuple=True)[0]
                b, p, y = ib[idx], ip[idx], y_sel[idx]
                w = wrong_flag[idx]

                # not wrong: payload = non-GT in second half
                if (~w).any():
                    i = (~w).nonzero(as_tuple=True)[0]
                    b0, p0, y0 = b[i], p[i], y[i]
                    r = torch.randint(0, K - 1, (y0.numel(),), device=device)
                    inc = r + (r >= y0).long()
                    label_vec[b0, p0, K + inc] = 1.0
                    aspect_bits[b0, p0] = 1

                # wrong: payload = GT in second half (claims GT is “incorrect”)
                if w.any():
                    i = w.nonzero(as_tuple=True)[0]
                    b1, p1, y1 = b[i], p[i], y[i]
                    label_vec[b1, p1, K + y1] = 1.0
                    aspect_bits[b1, p1] = 1

        # --------------- BOUNDARY prompts (aspect independent from noise) ---------------
        mask_bnd = prompt_types == 1
        if mask_bnd.any():
            ib, ip = mask_bnd.nonzero(as_tuple=True)
            gt = boundary[ib, prompt_idx[ib, ip]].float()   # [Nbnd] in {0,1}

            # annotator-declared aspect (independent of noise)
            decl_is_correct_b = torch.rand_like(gt) < float(p_correct_bnd)

            # nominal payload per declared aspect
            val_nom = torch.where(decl_is_correct_b.bool(), gt, 1.0 - gt)

            # wrongness flips the relation to *ground truth*
            if p_wrong is not None and p_wrong > 0.0:
                wrong_b = (torch.rand_like(gt) < float(p_wrong)).bool()
                # if correct-declared and wrong -> 1-gt; if incorrect-declared and wrong -> gt
                val_fin = torch.where(
                    wrong_b,
                    torch.where(decl_is_correct_b.bool(), 1.0 - gt, gt),
                    val_nom,
                )
            else:
                val_fin = val_nom

            boundary_bits[ib, ip] = val_fin.long()
            aspect_bits[ib, ip] = (~decl_is_correct_b.bool()).long()   # 0=declared-correct, 1=declared-incorrect


        # (optional) strict assertion for label exclusivity:
        left = label_vec[..., :K].sum(dim=-1)  # in {0,1}
        right = label_vec[..., K:].sum(dim=-1)  # in {0,1}
        assert (
            ((left == 0) ^ (right == 0)) | (~mask_label)
        ).all(), "Label prompt must be exclusively correct OR incorrect."

        return prompt_idx, ts_ctx, prompt_types, label_vec, boundary_bits, aspect_bits

    def iter_train(self) -> tuple[dict, dict]:
        # --- AMP scaler (match PromptTSS style) ---
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)  # type: ignore

        # --- Metrics containers (match PromptTSS style) ---
        metrics = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list),
        }
        iter_reports = {
            "train": [],
            "val": [],
            "test": [],
        }

        # --- Convenience knobs ---
        P_prompts = getattr(self.args, "P_prompts", 1)
        T_ctx = self.args.context_len

        # --- Training epochs ---
        for epoch in range(self.args.epochs):
            self.model.train()
            train_losses = []

            # progress bar
            iter_data = (
                tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}/{self.args.epochs}, Training Loss: {0}",
                )
                if self.args.use_tqdm
                else self.train_loader
            )

            for batch_idx, (
                subseq_x,
                subseq_y,
                win_x,
                win_y,
                granularity_level,
            ) in enumerate(iter_data):
                # Skip singleton batches (for stability with BatchNorm etc)
                if subseq_x.shape[0] == 1:
                    print(
                        f"Warning: Skipping singleton batch {batch_idx} in epoch {epoch + 1}"
                    )
                    continue

                # Move to device
                subseq_x = subseq_x.float().to(self.args.device)  # [B, Ls, C]
                subseq_y = subseq_y.long().to(self.args.device)  # [B, Ls]
                win_x = win_x.float().to(self.args.device)  # [B, W, T, C]
                win_y = win_y.long().to(self.args.device)  # [B, W, T]

                B, W, T, C = win_x.shape

                # Fresh per-subsequence banks for this subsequence-batch
                bank = MemoryBankBatch(
                    feature_dim=self.args.d_model,
                    device=self.args.device,
                    capacity=math.inf,
                )
                bank.reset_for_batch(B)

                # Iterative training within this subsequence-batch
                for iter_num in range(self.args.num_iter_train):

                    # ? 1. Zero grad
                    self.optimizer.zero_grad(set_to_none=True)

                    # -------------------------- MEMORY WRITE --------------------------
                    # - Read OLD memory tokens per subsequence (DETACHED) -> [B, N_old, D]
                    # - Sample P prompts per subsequence and extract TS contexts (length T_ctx)
                    # - Encode CURRENT tokens M_cur: prompt_encoder + ts_encoder -> memory_encoder
                    # - Concatenate M_all = [M_old, M_cur] (per subsequence)
                    with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                        M_old = (
                            bank.read()
                        )  # [B, N_old, D] (N_old = iter_num * P_prompts; 0 at iter_num=0)

                        (
                            prompt_idx,
                            ts_ctx,
                            p_types,
                            p_label_vec,
                            p_bnd_bits,
                            p_aspects,
                        ) = self.sample_prompts_and_ctx(
                            subseq_x=subseq_x,  # [B,Ls,C]
                            subseq_y=subseq_y,  # [B,Ls]
                            P=P_prompts,
                            T_ctx=T_ctx,
                            K=self.args.K,
                            p_label=getattr(self.args, "p_label", 0.5),
                            p_correct_label=getattr(self.args, "p_correct_label", 0.5),
                            p_correct_bnd=getattr(self.args, "p_correct_bnd", 0.5),
                            p_wrong=getattr(self.args, "p_wrong", None),
                            win_len=self.args.window_len,
                            win_stride=self.args.window_stride,
                            win_hit_pct=getattr(self.args, "win_hit_pct", None),
                        )

                        B, P = p_types.shape
                        ts_ctx_flat = ts_ctx.reshape(
                            B * P, T_ctx, ts_ctx.size(-1)
                        )  # [B*P, T_ctx, C]

                        # one prompt -> one token
                        prompt_tok = self.model.prompt_encoder(
                            p_types, p_label_vec, p_bnd_bits, p_aspects
                        )  # [B,P,1,D]
                        prompt_tok_f = prompt_tok.view(B * P, 1, -1)  # [B*P,1,D]
                        ts_ctx_tok = self.model.ts_encoder(
                            ts_ctx_flat
                        )  # [B*P,Tp_ctx,D]

                        M_cur_flat = self.model.memory_encoder(
                            prompt_tok_f, ts_ctx_tok
                        )  # [B*P,1,D]
                        M_cur = M_cur_flat.squeeze(1).view(B, P, -1)  # [B,P,D]

                        M_all = (
                            torch.cat([M_old.to(M_cur.dtype), M_cur], dim=1)
                            if M_old.size(1)
                            else M_cur
                        )  # [B,N_all,D]

                    # -------------------------- MEMORY READ ---------------------------
                    # Loop over all sliding windows win_idx=0..W-1:
                    #   - z_x = ts_encoder(win_x[:, win_idx])                -> [B, T_p, D]
                    #   - logits = state_decoder(z_x, M_all)           -> [B, T, K]
                    #   - loss = CE(logits, win_y[:, win_idx])               -> scalar
                    # Accumulate grads across all windows, then step ONCE.
                    loss_accum = 0.0  # for logging only
                    total_loss_tensor = None  # for backprop
                    for win_idx in range(W):
                        xB = win_x[:, win_idx]  # [B, T, C]
                        yB = win_y[:, win_idx]  # [B, T]
                        with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                            # ? 2. Call the model
                            z_x = self.model.ts_encoder(xB)  # [B, T_p, D]
                            y_pred = self.model.state_decoder(z_x, M_all)  # [B, T, K]

                            # ? 3. Calculate loss
                            loss = self.criterion(y_pred, yB)

                            # NaN guard (same spirit as PromptTSS)
                            if torch.isnan(loss).any():
                                print(
                                    f"Warning: NaN loss at batch {batch_idx}, iter {iter_num}, window {win_idx}. Skipping this window."
                                )
                                continue

                        # ? 4. Backward (accumulate across W windows)
                        # scaler.scale(loss).backward()  # type: ignore
                        # loss_accum += float(loss.detach())
                        # accumulate for one backward call
                        total_loss_tensor = (
                            loss
                            if total_loss_tensor is None
                            else (total_loss_tensor + loss)
                        )
                        loss_accum += float(loss.detach())

                    # average over windows (optional but stable)
                    total_loss_tensor = total_loss_tensor / max(W, 1)  # type: ignore
                    scaler.scale(total_loss_tensor).backward()  # type: ignore

                    # Clip + step once per iteration
                    scaler.unscale_(self.optimizer)  # type: ignore
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip
                    )
                    scaler.step(self.optimizer)  # type: ignore
                    scaler.update()  # type: ignore

                    train_losses.append(loss_accum)

                    # Commit CURRENT memory tokens (DETACHED) so they become M_old next iteration
                    with torch.no_grad():
                        bank.write(M_cur)  # [B, P, D]
                        # print(
                        #     f"After iteration {iter_num+1}, batch {batch_idx+1}: "
                        #     f"MemoryBank contains {bank.num_tokens} tokens per sequence."
                        # )

                    # tqdm description
                    if self.args.use_tqdm:
                        iter_data.set_description(  # type: ignore
                            f"Epoch {epoch + 1}/{self.args.epochs}, Training Loss: {np.mean(train_losses) if len(train_losses)>0 else 0:.4f}"
                        )

            # ==========================
            # End of epoch: evaluation
            # ==========================
            print(
                f"Epoch {epoch + 1}/{self.args.epochs}, Calculating iterative train_metrics ..."
            )
            train_iter_reports = self.get_metrics_ind_iterative(self.train_loader)
            for i, m in enumerate(train_iter_reports, start=1):
                printable = ", ".join([f"{k}: {v:.4f}" for k, v in m.items()])
                print(f"[train][Iter {i}] {printable}")
            for metric_name, metric_value in train_iter_reports[-1].items():
                metrics["train"][metric_name].append(metric_value)

            print(
                f"Epoch {epoch + 1}/{self.args.epochs}, Calculating iterative val_metrics ..."
            )
            val_iter_reports = self.get_metrics_ind_iterative(self.val_loader)
            for i, m in enumerate(val_iter_reports, start=1):
                printable = ", ".join([f"{k}: {v:.4f}" for k, v in m.items()])
                print(f"[val][Iter {i}] {printable}")
            for metric_name, metric_value in val_iter_reports[-1].items():
                metrics["val"][metric_name].append(metric_value)

            print(
                f"Epoch {epoch + 1}/{self.args.epochs}, Calculating iterative test_metrics ..."
            )
            test_iter_reports = self.get_metrics_ind_iterative(self.test_loader)
            for i, m in enumerate(test_iter_reports, start=1):
                printable = ", ".join([f"{k}: {v:.4f}" for k, v in m.items()])
                print(f"[test][Iter {i}] {printable}")
            for metric_name, metric_value in test_iter_reports[-1].items():
                metrics["test"][metric_name].append(metric_value)

            # Save per-epoch reports
            iter_reports["train"].append(train_iter_reports)
            iter_reports["val"].append(val_iter_reports)
            iter_reports["test"].append(test_iter_reports)

            # Show metrics
            visualize_metric(metrics, mode="table")
            visualize_metric(metrics, mode="plot")

            # Early stopping
            self.early_stopping(
                metrics["val"]["loss"][-1], self.model, self.args.checkpoint_saving_path
            )
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            # Learning rate scheduler
            if self.args.lr_scheduler != "none":
                previous_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1}/{self.args.epochs}, Learning Rate: {previous_lr} -> {current_lr}"
                    f" (lr_scheduler: {self.args.lr_scheduler})"
                )

        return metrics, iter_reports

    def calculate_segmentation_metrics(
        self,
        trues: list[np.ndarray],
        preds: list[np.ndarray],
        return_kappa: bool = False,
    ) -> tuple[float, float, float | None]:
        acc_scores = []
        f1_scores = []
        kappa_scores = []

        # Calculate metrics for each pair of true and predicted labels
        for i, (true, pred) in enumerate(zip(trues, preds)):
            acc_scores.append(accuracy_score(true, pred))
            f1_scores.append(f1_score(true, pred, average="macro"))
            if return_kappa:
                kappa = cohen_kappa_score(true, pred)
                # Log problematic cases
                if np.isnan(kappa):
                    print(f"NaN Kappa detected in segment {i}")
                    print(
                        f"Original True labels: {true}, Original Predicted labels: {pred}"
                    )
                    print(
                        f"True labels: {np.unique(true)}, Predicted labels: {np.unique(pred)}"
                    )
                    print(
                        f"Variance in True labels: {np.var(true)}, Variance in Predicted labels: {np.var(pred)}"
                    )
                kappa_scores.append(kappa)

        # Calculate the average accuracy, F1 score, and Cohen's kappa
        average_acc = float(np.mean(acc_scores))
        average_f1 = float(np.mean(f1_scores))
        if return_kappa:
            average_kappa = float(np.mean(kappa_scores))
            return average_acc, average_f1, average_kappa
        else:
            return average_acc, average_f1, None

    def calculate_clustering_metrics(
        self, trues: list[np.ndarray], preds: list[np.ndarray]
    ) -> tuple[float, float]:
        ari_scores = []
        nmi_scores = []

        # Calculate metrics for each pair of true and predicted labels
        for true, pred in zip(trues, preds):
            ari = adjusted_rand_score(true, pred)
            nmi = normalized_mutual_info_score(true, pred)
            ari_scores.append(ari)
            nmi_scores.append(nmi)

        # Calculate the average ARI and NMI
        average_ari = float(np.mean(ari_scores))
        average_nmi = float(np.mean(nmi_scores))

        return average_ari, average_nmi

    def get_metrics_ind_iterative(
        self,
        data_loader: InterleavedDataLoader,
        add_clustering_metrics: bool = True,
        add_inference_time: bool = False,  # set True if you want per-iter timing
    ) -> list[dict[str, float]]:
        """
        Perseus iterative inference:
        - Dataloader yields (subseq_x, subseq_y, win_x, win_y, granularity_level)
        - For each test iteration r:
            MEMORY WRITE: build per-subsequence M_cur from P prompts + T_ctx context (from subseq_x)
            MEMORY READ : run all windows in the subsequence with M_all = [M_old, M_cur]
            Commit M_cur to the per-subsequence banks (DETACHED), then continue to next r
        - Accumulate loss and predictions per iteration; compute metrics at the end.

        Returns:
        metrics_per_iter: list of dicts of length R=self.args.num_iter_test
                            each dict has keys: loss, acc, mf1[, ari, nmi][, avg_inference_time_per_batch]
        """
        assert getattr(self.args, "num_iter_test", 1) >= 1
        R = int(self.args.num_iter_test)

        P_prompts = getattr(self.args, "P_prompts_test", 1)
        T_ctx = self.args.context_len

        # -----------------------------
        # Per-iteration collectors
        # -----------------------------
        total_losses_per_iter: list[float] = [0.0 for _ in range(R)]
        total_preds_per_iter: list[list[np.ndarray]] = [
            [] for _ in range(R)
        ]  # each holds list of (T,) arrays
        total_trues_windows: list[np.ndarray] = []  # collected once (same for all r)
        per_iter_times: list[list[float]] = [[] for _ in range(R)]  # optional timing

        # We normalize "loss" by #batches (same scale across runs); keep it simple and robust.
        num_batches = 0

        self.model.eval()
        with torch.no_grad():
            iterator = tqdm(data_loader) if self.args.use_tqdm else data_loader
            for batch in iterator:
                # New eval dataloader: (subseq_x, subseq_y, win_x, win_y, granularity_level)
                subseq_x, subseq_y, win_x, win_y, granularity_level = batch

                subseq_x = subseq_x.float().to(self.args.device)  # [B, Ls, C]
                subseq_y = subseq_y.long().to(
                    self.args.device
                )  # [B, Ls] (not directly used here, but kept for completeness)
                win_x = win_x.float().to(self.args.device)  # [B, W, T, C]
                win_y = win_y.long().to(self.args.device)  # [B, W, T]

                B, W, T, C = win_x.shape
                num_batches += 1

                # Collect trues ONCE for this batch: accumulate all windows' labels as (T,) arrays
                for j in range(W):
                    true_np = win_y[:, j].detach().cpu().numpy()  # [B, T]
                    total_trues_windows.extend(
                        list(true_np)
                    )  # append B items of shape (T,)

                # Fresh per-subsequence memory banks (like training)
                bank = MemoryBankBatch(
                    feature_dim=self.args.d_model,
                    device=self.args.device,
                    capacity=math.inf,
                )
                bank.reset_for_batch(B)

                # Iterative inference
                for r in range(R):
                    # Optional timing
                    if add_inference_time:
                        t0 = time.perf_counter()

                    # =================================================================
                    # MEMORY WRITE (once per inference iteration)
                    # - Read OLD per-subsequence memory  M_old: [B, N_old, D]
                    # - Sample P prompts per subsequence from subseq_x; extract T_ctx contexts
                    # - Encode CURRENT tokens M_cur: [B, P, D]
                    # - Build M_all = concat(M_old, M_cur): [B, N_all, D]
                    # =================================================================
                    with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                        M_old = bank.read()  # [B, N_old, D] (0 at r=0)

                        (
                            prompt_idx,
                            ts_ctx,
                            p_types,
                            p_label_vec,
                            p_bnd_bits,
                            p_aspects,
                        ) = self.sample_prompts_and_ctx(
                            subseq_x=subseq_x,  # [B,Ls,C]
                            subseq_y=subseq_y,  # [B,Ls]
                            P=P_prompts,
                            T_ctx=T_ctx,
                            K=self.args.K,
                            p_label=getattr(self.args, "p_label", 0.5),
                            p_correct_label=getattr(self.args, "p_correct_label", 0.5),
                            p_correct_bnd=getattr(self.args, "p_correct_bnd", 0.5),
                            p_wrong=getattr(self.args, "p_wrong", None),
                            win_len=self.args.window_len,
                            win_stride=self.args.window_stride,
                            win_hit_pct=getattr(self.args, "win_hit_pct", None),
                        )

                        B, P = p_types.shape
                        ts_ctx_flat = ts_ctx.reshape(
                            B * P, T_ctx, ts_ctx.size(-1)
                        )  # [B*P, T_ctx, C]

                        # one prompt -> one token
                        prompt_tok = self.model.prompt_encoder(
                            p_types, p_label_vec, p_bnd_bits, p_aspects
                        )  # [B,P,1,D]
                        prompt_tok_f = prompt_tok.view(B * P, 1, -1)  # [B*P,1,D]
                        ts_ctx_tok = self.model.ts_encoder(
                            ts_ctx_flat
                        )  # [B*P,Tp_ctx,D]

                        M_cur_flat = self.model.memory_encoder(
                            prompt_tok_f, ts_ctx_tok
                        )  # [B*P,1,D]
                        M_cur = M_cur_flat.squeeze(1).view(B, P, -1)  # [B,P,D]

                        M_all = (
                            torch.cat([M_old.to(M_cur.dtype), M_cur], dim=1)
                            if M_old.size(1)
                            else M_cur
                        )  # [B,N_all,D]

                    # =================================================================
                    # MEMORY READ (run all windows using M_all; no optimization here)
                    # - For win_idx=0..W-1:
                    #     z_x    = ts_encoder(win_x[:, win_idx])         -> [B, T_p, D]
                    #     logits = state_decoder(z_x, M_all)       -> [B, T, K]
                    #     loss   = CE(logits, win_y[:, win_idx])
                    #   Accumulate per-window loss, collect preds for metrics.
                    # =================================================================
                    loss_sum = 0.0
                    for win_idx in range(W):
                        xB = win_x[:, win_idx]  # [B, T, C]
                        yB = win_y[:, win_idx]  # [B, T]
                        with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                            z_x = self.model.ts_encoder(xB)  # [B, T_p, D]
                            y_pred = self.model.state_decoder(z_x, M_all)  # [B, T, K]
                            loss = self.criterion(y_pred, yB)

                        loss_sum += float(loss.item())

                        # collect predictions (append B items of shape (T,))
                        pred_np = (
                            torch.argmax(y_pred, dim=-1).detach().cpu().numpy()
                        )  # [B, T]
                        total_preds_per_iter[r].extend(list(pred_np))

                    # Average the window losses within this batch for stability
                    total_losses_per_iter[r] += loss_sum / max(W, 1)

                    # Commit CURRENT tokens to banks (DETACHED) for the next iteration r+1
                    bank.write(M_cur)  # [B, P, D]

                    if add_inference_time:
                        t1 = time.perf_counter()
                        per_iter_times[r].append(t1 - t0)

        # -----------------------------
        # Build per-iteration metrics
        # -----------------------------
        # Flatten trues once: list of (T,) arrays -> (N_windows*T,)
        true_1d = np.concatenate(total_trues_windows, axis=0)

        metrics_per_iter: list[dict[str, float]] = []
        for r in range(R):
            preds_1d = np.concatenate(total_preds_per_iter[r], axis=0)
            assert (
                true_1d.shape == preds_1d.shape
            ), f"Shape mismatch: true {true_1d.shape} vs pred {preds_1d.shape}"

            acc, mf1, _ = self.calculate_segmentation_metrics(
                [true_1d], [preds_1d], return_kappa=False
            )

            iter_metrics = {
                # average loss per batch (already averaged over windows inside the loop)
                "loss": total_losses_per_iter[r] / max(num_batches, 1),
                "acc": float(acc),
                "mf1": float(mf1),
            }

            if add_clustering_metrics:
                ari, nmi = self.calculate_clustering_metrics([true_1d], [preds_1d])
                iter_metrics["ari"] = float(ari)
                iter_metrics["nmi"] = float(nmi)

            if add_inference_time and len(per_iter_times[r]) > 0:
                iter_metrics["avg_inference_time_per_batch"] = float(
                    np.mean(per_iter_times[r])
                )

            metrics_per_iter.append(iter_metrics)

        return metrics_per_iter
