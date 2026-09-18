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

GPU IDs, paths, and the PyTorch wheel index are configured in `config`.

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

This resumes saved runs, or downloads the [initial results](https://github.com/KentoNishi/icassp27-spurious-onsets/releases/tag/results-2026-08-13) if run directly after setup. Results are saved in `runs/moshi/results.json` and `runs/personaplex/results.json`.

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

After downloading `mitigation_results.json` above, use the same split with your own 500-rollout results:

```bash
uv run plot_mitigation.py --raw runs/moshi/results.json runs/personaplex/results.json
```

Figures are saved in `figures/`.

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
