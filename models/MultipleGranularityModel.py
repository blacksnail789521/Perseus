import torch
import torch.nn as nn
import argparse
from models.PromptTSS import PromptTSS


class MultiGranularityModel(nn.Module):
    def __init__(self, args: argparse.Namespace, model_class: nn.Module) -> None:
        super().__init__()
        self.args = args

        if model_class is PromptTSS:
            raise ValueError(
                f"{model_class.__name__} already supports multi-granularity!"
            )

        self.models: nn.ModuleDict = nn.ModuleDict(
            {str(level): model_class(args) for level in args.granularity_levels}
        )

    def forward(
        self,
        x: torch.Tensor,
        prompt: dict | None = None,
        granularity_level: int | None = None,
    ) -> torch.Tensor:
        # * --- Training ---
        if self.training:
            assert (
                granularity_level is not None
            ), "Training requires a specified granularity level!"
            return self.models[str(granularity_level)](x)

        # * --- Inference ---
        y_pred_logits_list = [
            model(x) for model in self.models.values()
        ]  # since self.models is a dictionary
        return (
            self._select_best_granularity(y_pred_logits_list, prompt)
            if prompt is not None
            else y_pred_logits_list[0]
        )

    def _select_best_granularity(
        self, y_pred_logits_list: list[torch.Tensor], prompt: dict
    ) -> torch.Tensor:
        best_accuracy = -float("inf")
        best_y_pred_logits = None

        for y_pred_logits in y_pred_logits_list:
            y_pred = torch.argmax(y_pred_logits, dim=-1)  # Converts (B, T, K) -> (B, T)
            accuracy = self._compute_prompt_accuracy(y_pred, prompt)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_y_pred_logits = y_pred_logits

        assert best_y_pred_logits is not None, "best_y_pred_logits is None"

        return best_y_pred_logits

    def _compute_prompt_accuracy(self, y_pred: torch.Tensor, prompt: dict) -> float:
        B, T = y_pred.shape
        total_prompts = 0
        correct_hits = 0

        # Precompute boundary predictions once
        pred_boundary = (y_pred[:, 1:] != y_pred[:, :-1]).float()
        pred_boundary = torch.cat(
            [torch.zeros(B, 1).to(y_pred.device), pred_boundary], dim=1
        )

        for key, prompt_values in prompt.items():
            time_index = key[-1]  # Get the timestamp (last element of key)

            # Skip out-of-bounds timestamps before further processing
            if time_index >= T:
                continue

            if key[0] == "boundary":
                correct_hits += sum(
                    int(pred_boundary[b, time_index].item() == prompt_values[b])
                    for b in range(B)
                )
                total_prompts += B

            elif key[0] == "label":
                aspect = key[1]

                if aspect == "correct":
                    correct_hits += sum(
                        int(y_pred[b, time_index].item() == prompt_values[b])
                        for b in range(B)
                    )

                elif aspect == "incorrect":
                    for b in range(B):
                        if y_pred[b, time_index].item() not in prompt_values[b]:
                            correct_hits += 1

                total_prompts += B

        return correct_hits / total_prompts if total_prompts > 0 else 0.0
