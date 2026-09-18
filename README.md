# Causal Analysis and Mitigation of Spurious Onsets in Full-Duplex Speech LLMs

[Kento Nishi](https://kentonishi.com/)

[arXiv](https://arxiv.org/abs/2609.13445) / [PDF](https://arxiv.org/pdf/2609.13445)

Why do full-duplex speech models start talking when the user is silent? We investigate this behavior in Moshi and PersonaPlex, and introduce an inference-time method to suppress spurious onsets while preserving genuine responses, without retraining.

## Usage

### Setup

Use Linux with CUDA and two NVIDIA GPUs. Request access to [PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1), then run:

```bash
./build
./fetch-noise
```

Set GPU IDs and paths in [`config`](config).

### Run experiments

```bash
./calibrate       # calibrate onset qualification
./run             # 40 trials per model
./run --extend 500 # full mitigation experiment
```

To reuse the [released calibration and initial results](https://github.com/KentoNishi/icassp27-spurious-onsets/releases/tag/results-2026-08-13), run `./run --extend 500` directly after setup. Results are saved in `runs/{moshi,personaplex}/results.json`.

### Reproduce figures and results

Plot the mechanism traces (Figure 2):

```bash
./.venv/moshi/bin/python plot_results.py runs/moshi/results.json runs/personaplex/results.json --mechanism-only
```

Reproduce Figure 3 and Table 1 from the released results with `uv`:

```bash
curl -fLO https://github.com/KentoNishi/icassp27-spurious-onsets/releases/download/results-2026-09-16/mitigation_results.json
uv run plot_mitigation.py
```

Or analyze your own 500-rollout results:

```bash
uv run plot_mitigation.py --raw runs/moshi/results.json runs/personaplex/results.json
```

The [analysis](plot_mitigation.py) uses 100 rollouts for threshold calibration and 400 for evaluation. Figures are saved in `figures/`.

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
