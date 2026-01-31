import torch
import argparse
from pathlib import Path
import json

from exp.exp_segmentation import Exp_Segmentation
from exp.exp_segmentation_baseline import Exp_Segmentation_Baseline
from utils.tools import set_seed, print_formatted_dict, select_best_metrics
from dataset_loader.dataset_tools import update_args_from_dataset


def get_args_from_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perseus")

    # * basic config
    parser.add_argument(
        "--overwrite_args",
        action="store_true",
        help="overwrite args with fixed_params and tunable_params",
        default=False,
    )
    parser.add_argument(
        "--checkpoint_saving_path",
        type=str,
        default="./checkpoints/checkpoint.pth",
        help="checkpoint saving path",
    )
    parser.add_argument(
        "--use_tqdm",
        action="store_true",
        help="use tqdm for progress bar",
        # default=False,
        default=True,
    )

    # * data loader
    parser.add_argument(
        "--data_name",
        type=str,
        default="Pump_V35",
        choices=["Pump_V35", "Pump_V36", "Pump_V38"],
        help="data name",
    )
    parser.add_argument(
        "--subseq_len",
        type=int,
        default=704,
        help="subsequence length (having several sliding windows inside)",
    )
    parser.add_argument(
        "--window_len", type=int, default=256, help="sliding window length"
    )
    parser.add_argument(
        "--context_len", type=int, default=256, help="context sequence length"
    )
    parser.add_argument("--window_stride", type=int, default=1, help="window stride")
    parser.add_argument(
        "--win_hit_pct",
        type=float,
        default=None,
        help="the target percentage (range: 0 to 1) of sliding windows to be covered by prompts",
    )
    parser.add_argument(
        "--p_wrong", type=float, default=None, help="percentage of wrong labels"
    )
    parser.add_argument("--K", type=int, default=43, help="number of classes")
    parser.add_argument("--C", type=int, default=9, help="number of features")
    parser.add_argument(
        "--downsample_rate",
        type=int,
        default=1,
        help="downsample rate for the data",
    )
    parser.add_argument(
        "--padding",
        type=bool,
        default=False,
        help="padding for the data (to make sure the length for each df is the same)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=2,
        help="group size for the granularity level",
    )
    parser.add_argument(
        "--granularity_levels",
        type=int,
        nargs="+",
        default=[0],
        help="granularity levels",
    )
    parser.add_argument(
        "--only_one_df",
        type=bool,
        default=False,
        help="use only one df for training and testing",
    )

    # * model architecture
    parser.add_argument(
        "--model_name",
        type=str,
        default="Perseus",
        choices=["Perseus"],
        help="model name",
    )
    parser.add_argument(
        "--encoder_arch",
        type=str,
        default="transformer_encoder_pytorch",
        # default="transformer_decoder",
        # default="resnet",
        # default="tcn",
        # default="lstm",
        choices=[
            "transformer_encoder",
            "transformer_encoder_pytorch",
            "transformer_decoder",
            "resnet",
            "tcn",
            "lstm",
            "bilstm",
        ],
        help="encoder architecture",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=3,
        help="number of layers in time series encoder",
    )
    parser.add_argument(
        "--base_d_model",
        type=int,
        default=64,
        help="the base number of embedding dimension (d_model = base_d_model * n_heads)",
    )
    parser.add_argument("--n_heads", type=int, default=12, help="number of heads")
    parser.add_argument(
        "--d_ff",
        type=int,
        default=256,
        help="bottle neck dimension in state decoder",
    )
    parser.add_argument(
        "--n_state_decoder_blocks",
        type=int,
        default=6,
        help="number of state decoder blocks",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=2048,
        help="maximum length for the positional encoding",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--activation",
        type=str,
        default="gelu",
        help="activation function",
        choices=["relu", "gelu"],
    )
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument(
        "--patch_stride", type=int, default=8, help="stride length for patching"
    )
    parser.add_argument(
        "--num_filters",
        type=int,
        default=64,
        help="number of filters for the convolutional layers",
    )
    parser.add_argument(
        "--filter_size",
        type=int,
        default=5,
        help="filter size for the convolutional layers",
    )

    # * training_stage_params
    parser.add_argument(
        "--optim",
        type=str,
        default="AdamW",
        help="optimizer",
        choices=["Adam", "AdamW", "RMSprop"],
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="StepLR",
        help="learning rate scheduler",
        choices=[
            "none",
            "StepLR",  # step_size=30, gamma=0.1. Decays the LR by gamma every step_size epochs.
            "ExponentialLR",  # gamma=0.95
            "CosineAnnealingLR",  # T_max=50
            "CyclicLR",  # base_lr=lr, max_lr=0.1, step_size_up=20
            "OneCycleLR",  # max_lr=0.1, steps_per_epoch=len(train_loader), epochs=num_epochs
        ],
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="l2 weight decay",
    )
    parser.add_argument("--epochs", type=int, default=10, help="epochs for training")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="data loader num workers"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="early stopping patience"
    )
    parser.add_argument("--delta", type=float, default=0, help="early stopping delta")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        # default=False,
        default=True,  # faster
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="gradient clipping max norm"
    )
    parser.add_argument(
        "--num_iter_train",
        type=int,
        default=8,
        help="number of iterations for training",
    )
    parser.add_argument(
        "--num_iter_test",
        type=int,
        default=1,
        help="number of iterations for testing",
    )
    parser.add_argument(
        "--P_prompts",
        type=int,
        default=3,
        help="number of prompts to generate",
    )
    parser.add_argument(
        "--P_prompts_test",
        type=int,
        default=3,
        help="number of prompts to generate for testing",
    )

    # * GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")

    args, _ = parser.parse_known_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.root_folder = Path.cwd()  # Set this outside of the trainable function
    args.lr_scheduler_params = {}

    return args


def update_args_from_fixed_params(
    args: argparse.Namespace, fixed_params: dict
) -> argparse.Namespace:
    # Update args
    for key, value in fixed_params.items():
        if not hasattr(args, key):
            print(f"AttributeError: {key} not found in args")
        print("### [Fixed] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args_from_tunable_params(
    args: argparse.Namespace, tunable_params: dict
) -> argparse.Namespace:
    # Update args
    for key, value in tunable_params.items():
        if not hasattr(args, key):
            print(f"AttributeError: {key} not found in args")
        print("### [Tunable] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args(
    args: argparse.Namespace,
    fixed_params: dict,
    tunable_params: dict,
) -> argparse.Namespace:
    # Check if there are duplicated keys
    duplicated_keys = set(fixed_params.keys()) & set(tunable_params.keys())
    assert not duplicated_keys, f"Duplicated keys found: {duplicated_keys}"

    # Update args from fixed_params, tunable_params, and dataset
    if args.overwrite_args:
        args = update_args_from_fixed_params(args, fixed_params)
        args = update_args_from_tunable_params(args, tunable_params)
    args = update_args_from_dataset(args)

    # Set d_model
    args.d_model = args.base_d_model * args.n_heads
    print("### Set d_model to {}".format(args.d_model))

    # Set T_p (patching)
    args.T_p = int((args.window_len - args.patch_len) / args.patch_stride + 1)
    print("### Set T_p to {}".format(args.T_p))
    print(f"Args in experiment: {args}")

    return args


def trainable(
    tunable_params: dict,
    fixed_params: dict,
    args: argparse.Namespace,
) -> tuple[dict, dict]:
    # Update args
    args = update_args(args, fixed_params, tunable_params)

    # Train the model
    if args.model_name == "Perseus":
        exp = Exp_Segmentation(args)
    else:
        exp = Exp_Segmentation_Baseline(args)
    metrics, iter_reports = exp.iter_train()
    print_formatted_dict(metrics)

    return (
        select_best_metrics(metrics, target_mode="test", target_metric="acc"),
        iter_reports,
    )


if __name__ == "__main__":
    """------------------------------------"""
    # data_name = "Pump_V35"  # granularity_levels = [0, 1, 2]
    # data_name = "Pump_V36"  # granularity_levels = [0, 1, 2]
    # data_name = "Pump_V38"  # granularity_levels = [0, 1, 2]
    # data_name = "USC-HAD"  # granularity_levels = [0, 1]
    data_name = "PAMAP2"  # granularity_levels = [0, 1]
    # data_name = "_confidential"  # granularity_levels = [0, 1]

    model_name = "Perseus"
    # model_name = "PromptTSS"
    # model_name = "PrecTime"
    # model_name = "MS-TCN++"
    # model_name = "U-Time"
    # model_name = "DeepConvLSTM"
    # model_name = "iTransformer"
    # model_name = "PatchTST"

    tunable_params_path = None
    # tunable_params_path = Path(
    #     "exp_settings_and_results",
    #     "single_granularity",
    #     model_name,
    #     f"{data_name}.json",
    # )

    batch_size = 4
    # batch_size = 8  # use this for running time
    # batch_size = 16
    # batch_size = 64
    # batch_size = 256

    num_workers = 4

    # window_len = 64
    # window_len = 128
    window_len = 256  # use this for most experiments
    # window_len = 512
    # window_len = 1024
    # window_len = 2048
    window_len *= 2 if data_name in ["USC-HAD", "PAMAP2"] else 1

    # context_len = window_len // 2
    context_len = window_len  # default
    # context_len = window_len * 2
    # context_len = window_len * 4

    window_stride = 64
    # window_stride = 128
    window_stride *= 2 if data_name in ["USC-HAD", "PAMAP2"] else 1

    win_hit_pct = 0.25  # ? default
    # win_hit_pct = 0.5
    # win_hit_pct = 1


    p_wrong = None  # ? default, i.e., use p_correct_label and p_correct_bnd
    # p_wrong = 0.1 # 10% wrong labels

    # One subsequence contains 8 sliding windows by default
    # num_windows = 1
    # num_windows = 2
    # num_windows = 4
    num_windows = 8  # ? default
    # num_windows = 16
    # num_windows = 32
    subseq_len = (num_windows - 1) * window_stride + window_len

    # ! Ablation for "Effect of Subsequence Length"
    enable_ablation_sub_len = True
    # enable_ablation_sub_len = False
    if enable_ablation_sub_len:
        # when doing ablation for subseq_len, set win_hit_pct so that we have 2 windows hit
        win_hit_pct = 2 / num_windows

    group_size = 2
    # group_size = 3

    granularity_levels = [0]
    # granularity_levels = [0, 1]
    # granularity_levels = [0, 1, 2]
    """------------------------------------"""
    # Set all random seeds (Python, NumPy, PyTorch)
    set_seed(42)

    # Setup args
    args = get_args_from_parser()

    # Setup fixed params
    fixed_params = {
        "data_name": data_name,
        "model_name": model_name,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "subseq_len": subseq_len,
        "window_len": window_len,
        "context_len": context_len,
        "window_stride": window_stride,
        "win_hit_pct": win_hit_pct,
        "p_wrong": p_wrong,
        "group_size": group_size,
        "granularity_levels": granularity_levels,
    }

    # Setup tunable params
    if tunable_params_path is None:
        # tunable_params = {}
        tunable_params = {
            # ? model architecture
            "n_layers": 3,
            # "n_layers": 2,
            "base_d_model": 64,
            "n_heads": 2,
            "dropout": 0.1,
            # "activation": "gelu",
            "activation": "relu",
            "patch_len": 16,
            "patch_stride": 8,
            # ? training_stage_params
            # "num_iter_train": 1,
            # "P_prompts": 32,
            "num_iter_train": 8,
            "P_prompts": 4,
            # ? some test params
            # "num_iter_test": 8,
            # "P_prompts_test": 4,  # use this for iterative inference
            "num_iter_test": 1,
            # "P_prompts_test": int(fixed_params["subseq_len"] * 0.00),  # 0% prompts (testing)
            # "P_prompts_test": int(fixed_params["subseq_len"] * 0.01),  # 1% prompts (testing)
            # "P_prompts_test": int(
            #     fixed_params["subseq_len"] * 0.05
            # ),  # 5% prompts (testing)
            # "P_prompts_test": int(fixed_params["subseq_len"] * 0.10),  # 10% prompts (testing)
            # "P_prompts_test": int(fixed_params["subseq_len"] * 0.25),  # 25% prompts (testing)
            "P_prompts_test": 32,  # ! ablation for "Effect of Subsequence Length"
            "optim": "AdamW",
            # "learning_rate": 0.001,  # GRU
            "learning_rate": 0.0001,  # Transformer
            "weight_decay": 0.01,
            # "epochs": 10,
            "epochs": 20,
            "lr_scheduler": "none",
            # "lr_scheduler": "StepLR",
            "lr_scheduler_params": {
                "StepLR": {
                    "step_size": 1,
                    "gamma": 0.1,
                },
                "ExponentialLR": {
                    "gamma": 0.95,
                },
                "CosineAnnealingLR": {
                    "T_max": 2,
                },
                "CyclicLR": {
                    "max_lr": 0.1,
                    "step_size_up": 3,
                    "step_size_down": 3,
                },
                "OneCycleLR": {
                    "max_lr": 0.1,
                },
            },
        }
    else:
        with open(tunable_params_path, "r") as f:
            tunable_params = json.load(f)

    # Run
    best_metrics, iter_reports = trainable(tunable_params, fixed_params, args)
    print_formatted_dict(best_metrics)
    print("### Done ###")
