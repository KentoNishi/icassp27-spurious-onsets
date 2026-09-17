# Causal Analysis and Mitigation of Spurious Onsets in Full-Duplex Speech LLMs

Kento Nishi

[arXiv](https://arxiv.org/abs/2609.13445)

## Setup

Linux, CUDA, two NVIDIA GPUs, and Hugging Face access to `nvidia/personaplex-7b-v1` are required.
Paths, GPUs, and the PyTorch wheel index are set in `config`.

```bash
./build
./fetch-noise
```

## Experiments

```bash
./calibrate
./run
```

`calibrate` derives onset qualification boundaries from the longest nonlexical gap across 40 responses per model. `run` performs the silence, counterfactual, threshold, and runtime experiments with 40 trials per model by default. Raw results go to `runs/`; summaries go to `results.json`.

For full replication, use `./run --extend 500` to reach 500 mitigation rollouts per model. Running it directly after setup reuses the [released calibration and 40-trial results](https://github.com/KentoNishi/icassp27-spurious-onsets/releases/tag/results-2026-08-13). Onset qualification boundaries stay fixed; final analysis calibrates decision thresholds on 100 rollouts and evaluates the other 400. Results go to `runs/{moshi,personaplex}/results.json`.

## Figures and results

Figure 2:

```bash
./.venv/moshi/bin/python plot_results.py runs/moshi/results.json runs/personaplex/results.json --mechanism-only
```

Figure 3 and Table 1 from released scores and the frozen split (`uv` required; no GPU):

```bash
curl -fLO https://github.com/KentoNishi/icassp27-spurious-onsets/releases/download/results-2026-09-16/mitigation_results.json
uv run plot_mitigation.py
```

To analyze your own 500-rollout results with the same split, append `--raw runs/moshi/results.json runs/personaplex/results.json` to the plotting command. Percentages are one-sided 95% exact binomial lower bounds. Figure 3 omits low-density tails for display only; `--density-floor 0` shows all scores. Figures go to `figures/`.

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
