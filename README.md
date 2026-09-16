# Causal Analysis and Mitigation of Spurious Onsets in Full-Duplex Speech LLMs

Code for reproducing our Moshi and PersonaPlex experiments.

## Setup

Linux, CUDA, two NVIDIA GPUs, and Hugging Face access to `nvidia/personaplex-7b-v1` are required.

```bash
./build
./fetch-noise
```

## Experiments

```bash
./calibrate
./run
```

`calibrate` generates 40 responses per model and derives each onset qualification boundary from the longest nonlexical gap. `run` performs the silence, counterfactual, threshold, and runtime experiments, with 40 trials per model by default. Raw results are written to `runs/`; aggregate results are written to `results.json`.

For full paper replication, follow with `./run --extend 500` to reach 500 mitigation rollouts per model. Alternatively, run it directly after setup to reuse the [released calibration and 40-trial results](https://github.com/KentoNishi/icassp27-spurious-onsets/releases/tag/results-2026-08-13). This reuses the onset qualification boundary; the final analysis recalibrates decision thresholds on 100 rollouts and evaluates the other 400. Updated results are written to `runs/moshi/results.json` and `runs/personaplex/results.json`.

```bash
./.venv/moshi/bin/python plot_results.py runs/moshi/results.json runs/personaplex/results.json --mechanism-only
```

For Figure 3 and Table 1, use the released scores and frozen split (requires `uv`, no GPU):

```bash
curl -fLO https://github.com/KentoNishi/icassp27-spurious-onsets/releases/download/results-2026-09-16/mitigation_results.json
uv run plot_mitigation.py
```

To analyze your own 500-rollout results with that split, append `--raw runs/moshi/results.json runs/personaplex/results.json`. Percentages are one-sided 95% exact binomial lower bounds. Figure 3 omits low-density tails for display only; `--density-floor 0` shows all scores. Figures are written to `figures/`.

Paths, GPUs, and the PyTorch wheel index are defined in `config`.
