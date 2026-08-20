# Perseus

Official implementation of **Perseus: Interactive Time Series Segmentation with Sparse Supervision via Stateful Memory**, accepted at IEEE ICDM 2026.

Perseus is a fully supervised time-series segmenter with an inference-time prompting pathway. Sparse label and boundary cues are written to a persistent memory bank and can then guide sliding windows outside the locally prompted region.

## Repository status

This repository is the camera-ready release snapshot. It contains the model, baselines, data loader, and preprocessed public benchmark files used by the released experiments. The proprietary IndustryMG dataset is not included and cannot be redistributed because of third-party confidentiality restrictions.

## Environment

The experiments were developed with Python 3.10 and PyTorch 1.13.1. A CUDA-capable GPU is recommended.

```bash
conda create -n perseus python=3.10
conda activate perseus
pip install -r requirements.txt
```

PyTorch wheels are platform-specific. If the pinned PyTorch package does not match your CUDA installation, install the appropriate PyTorch 1.13.1 build first and then install the remaining requirements.

## Data preparation

The loader expects one directory per dataset:

```text
dataset/
└── <dataset_name>/
    ├── sequence_01.csv
    ├── sequence_02.csv
    └── ...
```

Each CSV must contain feature columns followed by one integer state-label column as the final column. Files are sorted by filename, normalized per dataset, divided into non-overlapping subsequences, and split chronologically into 70% training, 15% validation, and 15% testing subsequences.

The released loader supports `Pump_V35`, `Pump_V36`, `Pump_V38`, `USC-HAD`, `PAMAP2`, `MoCap`, and `ActRecTut`. Coarser labels for datasets without native hierarchies are constructed at load time by merging consecutive fine-grained states in first-occurrence order; `--group_size` controls the merge factor and `--granularity_levels` selects the requested levels.

IndustryMG uses the same CSV interface internally, but its files are intentionally absent from this release. The public datasets are sufficient to run the open-data experiments and inspect the complete training and inference pipeline.

## Training and evaluation

`main.py` is the experiment entry point. Training, validation, and test metrics are computed during the same run; checkpoint saving and early stopping follow validation loss.

For a Pump V35 single-step run with the paper's default eight-window geometry and grouped 5% prompt budget:

```bash
python main.py \
  --data_name Pump_V35 \
  --subseq_len 704 \
  --window_len 256 \
  --context_len 256 \
  --window_stride 64 \
  --win_hit_pct 0.25 \
  --granularity_levels 0 1 2 \
  --group_size 2 \
  --num_iter_train 8 \
  --P_prompts 4 \
  --num_iter_test 1 \
  --P_prompts_test 35 \
  --base_d_model 64 \
  --n_heads 2 \
  --patch_len 16 \
  --patch_stride 8 \
  --epochs 10 \
  --checkpoint_saving_path checkpoints/pump_v35.pth
```

The three paper protocols use different test budgets:

- **Single-step:** one inference iteration with `P_prompts_test = floor(0.05 * subseq_len)`.
- **Iterative:** eight inference iterations with four new prompts per iteration, for 32 cumulative prompts.
- **Subsequence-length ablation:** a fixed 32-prompt budget while varying the number of windows.

For the activity datasets and ablations, edit the clearly marked experiment block under `if __name__ == "__main__":` and run:

```bash
python main.py --overwrite_args
```

The configuration block records the window geometry, prompt budget, granularity levels, optimizer, and epoch count used by that run. Checkpoints are written to `--checkpoint_saving_path`; per-epoch ACC, macro F1, ARI, and NMI are printed to the terminal.

## Reproducibility

The release uses seed **42** for Python, NumPy, and PyTorch; see `set_seed(42)` in `main.py`. Prompt sampling uses the same seeded generators. CUDA kernels can still introduce platform-dependent numerical variation, so record the GPU, CUDA, and package versions when comparing exact values.

To reproduce a result:

1. Start from a clean environment and install `requirements.txt`.
2. Confirm the dataset CSV layout and the chronological split described above.
3. Select the dataset, granularity levels, window geometry, and one of the three prompt protocols.
4. Run `main.py` and retain the printed arguments, checkpoint, and terminal metrics.
5. Repeat with additional seeds if estimating variance; the camera-ready headline tables use the original seed-42 protocol.

## Code and data availability

Code and preprocessing instructions are maintained at <https://github.com/blacksnail789521/Perseus>. IndustryMG cannot be released because it contains proprietary industrial measurements supplied under confidentiality restrictions; no IndustryMG records are present in this repository.

## Contact

Questions about the release can be sent to `blacksnail789521.cs10@nycu.edu.tw`.

## License

This code is released under the MIT License. Dataset files remain subject to their original providers' terms.
