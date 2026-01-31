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
from models.PatchTST import PatchTST
from models.DeepConvLSTM import DeepConvLSTM
from models.PromptTSS import PromptTSS
from models.iTransformer import iTransformer
from models.PrecTime import PrecTime
from models.U_Time import U_Time
from models.MS_TCN2 import MS_TCN2
from models.MultipleGranularityModel import MultiGranularityModel
from utils.visual import visualize_metric
from utils.tools import EarlyStopping
import random
import time
import math

MODEL_MAP = {
    "PatchTST": PatchTST,
    "PrecTime": PrecTime,
    "MS-TCN++": MS_TCN2,
    "U-Time": U_Time,
    "DeepConvLSTM": DeepConvLSTM,
    "iTransformer": iTransformer,
    "PromptTSS": PromptTSS,
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


class Exp_Segmentation_Baseline(object):
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

        # If using PromptTSS, we do NOT use MultiGranularityModel
        if self.args.model_name == "PromptTSS":
            self.model = ModelClass(self.args).to(self.args.device)
        else:
            # Use MultiGranularityModel for baselines that do NOT support multiple granularity
            self.model = MultiGranularityModel(self.args, model_class=ModelClass).to(
                self.args.device
            )
            # assert (
            #     self.args.num_iter_train == 1
            # ), "Only PromptTSS supports multiple iterations"

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

    def create_prompts(
        self,
        y,
        n_min,
        n_max,
        prompt_types=["label", "boundary"],
        p_wrong: float | None = None,   # independent reliability knob
    ):
        B, T = y.shape
        n_prompts = random.randint(n_min, n_max)

        boundary = (y[:, 1:] != y[:, :-1]).float()
        boundary = torch.cat([torch.zeros(B, 1).to(y.device), boundary], dim=1)

        if p_wrong is not None:
            if not (0.0 <= float(p_wrong) <= 1.0):
                raise ValueError("p_wrong must be in [0, 1]")
            p_wrong = float(p_wrong)

        prompts = {}

        for _ in range(n_prompts):
            time_index = random.randint(0, T - 1)
            prompt_type = random.choice(prompt_types)

            # draw a wrong flag per-prompt (shared across the batch for this key)
            is_wrong = (p_wrong is not None) and (random.random() < p_wrong)

            if prompt_type == "boundary":
                base = boundary[:, time_index].tolist()
                value = ([1.0 - v for v in base] if is_wrong else base)
                key = ("boundary", time_index)

            elif prompt_type == "label":
                # declared aspect stays as before (legacy distribution)
                prompt_aspect = random.choice(["correct", "incorrect"])

                if prompt_aspect == "correct":
                    if not is_wrong:
                        # payload should equal GT
                        value = [y[b, time_index].item() for b in range(B)]
                    else:
                        # WRONG: payload should NOT equal GT (still stored under 'correct')
                        value = [
                            random.choice([k for k in range(self.args.K) if k != y[b, time_index].item()])
                            for b in range(B)
                        ]
                else:  # "incorrect"
                    if not is_wrong:
                        # payload should be a non-GT class (list-of-lists)
                        value = [[
                            random.choice([k for k in range(self.args.K) if k != y[b, time_index].item()])
                        ] for b in range(B)]
                    else:
                        # WRONG: payload equals GT but stored under 'incorrect'
                        value = [[y[b, time_index].item()] for b in range(B)]

                key = ("label", prompt_aspect, time_index)

            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")

            # merge semantics unchanged (only ('label','incorrect',t) accumulates)
            if key in prompts:
                if len(key) == 3 and key[0] == "label" and key[1] == "incorrect":
                    existing_values = prompts[key]
                    prompts[key] = [
                        list(set(existing_values[b] + value[b])) for b in range(B)
                    ]
                continue
            else:
                prompts[key] = value

        return prompts


    def combine_prompts(self, existing_prompts, new_prompts):
        combined_prompts = existing_prompts.copy()

        for key, new_value in new_prompts.items():
            if key in combined_prompts:
                # Handle merging only for (label, incorrect)
                if len(key) == 3:  # Check for label prompt (keys with aspect)
                    prompt_type, prompt_aspect, _ = key
                    if prompt_type == "label" and prompt_aspect == "incorrect":
                        # Merge incorrect label prompts (multi-hot)
                        existing_values = combined_prompts[key]
                        combined_prompts[key] = [
                            list(set(existing_values[b] + new_value[b]))
                            for b in range(len(existing_values))
                        ]
                    else:
                        # For other label prompts, replace the existing prompt
                        combined_prompts[key] = new_value
                else:
                    # For boundary prompts, replace the existing prompt
                    combined_prompts[key] = new_value
            else:
                # Add new prompt to the dictionary
                combined_prompts[key] = new_value

        return combined_prompts

    def plan_prompts_for_windows(
        self, W: int, P_prompts: int, win_hit_pct: float | None
    ):
        """
        Decide how many NEW prompts to add to each of W windows this iteration.
        Returns:
            prompts_per_win : List[int] length W, sum == P_prompts
            hit_mask        : Bool tensor [W], True where prompts_per_win[w] > 0
            H_target, H_eff : ints for logging (requested vs achieved #hit windows)
        Notes:
            - In window-local baselines a single prompt lives in exactly one window, so
            max hit windows per iteration is min(W, P_prompts).
            - We avoid overshoot by placing extra prompts inside already-hit windows.
        """
        if win_hit_pct is None or W == 0 or P_prompts <= 0:
            return (
                [0] * W,
                torch.zeros(W, dtype=torch.bool, device=self.args.device),
                0,
                0,
            )

        H_target = int(round(W * float(win_hit_pct)))
        H_target = max(0, min(W, H_target))
        # Capacity for baselines: each prompt can hit only one window
        H_max = min(W, P_prompts)
        H_eff = min(H_target, H_max)

        # Choose which windows get hit (evenly spaced & deterministic-ish)
        if H_eff > 0:
            import numpy as np

            idx = np.linspace(0, W - 1, num=H_eff, dtype=int).tolist()
        else:
            idx = []

        prompts_per_win = [0] * W
        for w in idx:
            prompts_per_win[w] = 1  # one prompt to mark this window as a hit

        # Distribute remaining prompts inside the already-hit windows (no new hits)
        remaining = P_prompts - H_eff
        i = 0
        while remaining > 0 and len(idx) > 0:
            w = idx[i % len(idx)]
            prompts_per_win[w] += 1
            remaining -= 1
            i += 1

        # If H_eff==0 but P_prompts>0, we still must assign prompts; keep them in one window
        if H_eff == 0 and P_prompts > 0:
            w0 = 0
            prompts_per_win[w0] = P_prompts

        hit_mask = torch.tensor(
            [c > 0 for c in prompts_per_win], dtype=torch.bool, device=self.args.device
        )

        # ✅ budget sanity: this iteration must allocate exactly P_prompts
        assert (
            sum(prompts_per_win) == P_prompts
        ), "Prompt budget mismatch this iteration"

        return prompts_per_win, hit_mask, H_target, H_eff

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
        win_hit_pct = getattr(self.args, "win_hit_pct", None)

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

                # Keep a separate prompt dict for each window (because keys are window-local)
                prompt_pools = [dict() for _ in range(W)]  # length W

                # Iterative training within this subsequence-batch
                for iter_num in range(self.args.num_iter_train):

                    # ? 1. Zero grad
                    self.optimizer.zero_grad(set_to_none=True)

                    prompts_per_win, hit_mask, H_target, H_eff = (
                        self.plan_prompts_for_windows(W, P_prompts, win_hit_pct)
                    )
                    # optional: quick log
                    # print(
                    #     f"[baseline][train][iter {iter_num}] H_tgt={H_target}, H_eff={H_eff}, sumP={sum(prompts_per_win)}"
                    # )

                    loss_accum = 0.0  # for logging only
                    total_loss_tensor = None  # for backprop
                    for win_idx in range(W):
                        x = win_x[:, win_idx]  # [B, T, C]
                        y = win_y[:, win_idx]  # [B, T]

                        n_here = prompts_per_win[win_idx]
                        if n_here > 0:
                            new_prompts = self.create_prompts(
                                y,
                                n_here,
                                n_here,
                                p_wrong=getattr(self.args, "p_wrong", None),
                            )
                            prompt_pools[win_idx] = self.combine_prompts(
                                prompt_pools[win_idx], new_prompts
                            )
                        # else: this is a FREE window this iteration → add no prompts

                        with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                            # ? 2. Call the model
                            if self.args.model_name == "PromptTSS":
                                y_pred = self.model(x, prompt_pools[win_idx])
                            else:
                                y_pred = self.model(
                                    x,
                                    prompt_pools[win_idx],
                                    granularity_level[0].item(),
                                )  # Remember that all granularity levels are the same in the same batch

                            # ? 3. Calculate loss
                            loss = self.criterion(y_pred, y)

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
        assert getattr(self.args, "num_iter_test", 1) >= 1
        R = int(self.args.num_iter_test)

        P_prompts = getattr(self.args, "P_prompts_test", 1)
        win_hit_pct = getattr(self.args, "win_hit_pct", None)

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

                # Keep a separate prompt dict for each window (because keys are window-local)
                prompt_pools = [dict() for _ in range(W)]  # length W

                # Iterative inference
                for r in range(R):
                    # Optional timing
                    if add_inference_time:
                        t0 = time.perf_counter()

                    # plan once per iteration for this subsequence-batch
                    prompts_per_win, hit_mask, H_target, H_eff = (
                        self.plan_prompts_for_windows(W, P_prompts, win_hit_pct)
                    )
                    # print(
                    #     f"[baseline][eval][iter {r}] H_tgt={H_target}, H_eff={H_eff}, sumP={sum(prompts_per_win)}"
                    # )

                    loss_sum = 0.0
                    for win_idx in range(W):
                        x = win_x[:, win_idx]  # [B, T, C]
                        y = win_y[:, win_idx]  # [B, T]

                        n_here = prompts_per_win[win_idx]
                        if n_here > 0:
                            new_prompts = self.create_prompts(
                                y,
                                n_here,
                                n_here,
                                p_wrong=getattr(self.args, "p_wrong", None),
                            )
                            prompt_pools[win_idx] = self.combine_prompts(
                                prompt_pools[win_idx], new_prompts
                            )

                        with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                            y_pred = self.model(x, prompt_pools[win_idx])
                            loss = self.criterion(y_pred, y)

                        loss_sum += float(loss.item())

                        # collect predictions (append B items of shape (T,))
                        pred_np = (
                            torch.argmax(y_pred, dim=-1).detach().cpu().numpy()
                        )  # [B, T]
                        total_preds_per_iter[r].extend(list(pred_np))

                    # Average the window losses within this batch for stability
                    total_losses_per_iter[r] += loss_sum / max(W, 1)

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
