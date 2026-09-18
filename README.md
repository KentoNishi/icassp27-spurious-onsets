# Causal Analysis and Mitigation of Spurious Onsets in Full-Duplex Speech LLMs

Kento Nishi

[arXiv](https://arxiv.org/abs/2609.13445) / [PDF](https://arxiv.org/pdf/2609.13445)

## Usage

### Setup

Install on Linux with CUDA, two NVIDIA GPUs, and Hugging Face access to [PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1):

```bash
./build
./fetch-noise
```

### Experiments

Run the experiments with 40 trials per model:

```bash
./calibrate
./run
```

For the paper's 500-rollout mitigation experiment:

```bash
./run --extend 500
```

The extended run can also reuse released results, so you can run it directly after setup.

### Figures

Generate Figure 2 from your runs:

```bash
./.venv/moshi/bin/python plot_results.py runs/moshi/results.json runs/personaplex/results.json --mechanism-only
```

Generate Figure 3 and the Table 1 values from the paper's results (requires `uv`):

```bash
curl -fLO https://github.com/KentoNishi/icassp27-spurious-onsets/releases/download/results-2026-09-16/mitigation_results.json
uv run plot_mitigation.py
```

See `uv run plot_mitigation.py --help` for plotting options.

## Citation

```bibtex
@misc{nishi2026causal,
  title={Causal Analysis and Mitigation of Spurious Onsets in Full-Duplex Speech LLMs},
  author={Nishi, Kento},
  year={2026},
  eprint={2609.13445},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
