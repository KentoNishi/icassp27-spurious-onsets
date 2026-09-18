# Causal Analysis and Mitigation of Spurious Onsets in Full-Duplex Speech LLMs

Kento Nishi

[arXiv](https://arxiv.org/abs/2609.13445) / [PDF](https://arxiv.org/pdf/2609.13445)

## Usage

### Setup

Requires Linux, two CUDA GPUs, and access to [PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1).

```bash
./build
./fetch-noise
```

### Experiments

```bash
./calibrate
./run              # 40 trials per model
./run --extend 500 # full mitigation experiment
```

The extended run can also reuse released results, so you can run it directly after setup.

### Figures

Reproduce Figure 3 and Table 1 using `uv`:

```bash
curl -fLO https://github.com/KentoNishi/icassp27-spurious-onsets/releases/download/results-2026-09-16/mitigation_results.json
uv run plot_mitigation.py
```

Generate Figure 2 after running the experiments:

```bash
./.venv/moshi/bin/python plot_results.py runs/moshi/results.json runs/personaplex/results.json --mechanism-only
```

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
