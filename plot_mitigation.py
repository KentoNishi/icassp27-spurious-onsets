# /// script
# requires-python = ">=3.9,<3.13"
# dependencies = ["numpy==1.26.4", "scipy==1.13.1", "matplotlib==3.9.4"]
# ///
"""Regenerate Figure 3 and verify its frozen 100/400 split and table values.

Run: uv run plot_mitigation.py
Optional raw-data check: --verify-zip /path/to/runs.zip
Analyze new 500-rollout results: --raw runs/moshi/results.json runs/personaplex/results.json
The released score-only data preserve all 500 seeds per model, including
null scores. Calibration includes the original 40 mitigation seeds plus 60
sampled seeds; the other 400 seeds are held out. No cached decisions are reused.
For display only, points and KDE tails below 5% of each group's peak density
are omitted. KDEs use all observed scores and Scott's bandwidth; constant
groups are retained without a KDE. Use --density-floor 0 for the full view.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.mlab import GaussianKDE
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent
WIDTH = 86 / 25.4 * 72
FONT_SIZE = 7
MOSHI, PERSONAPLEX = "#d62728", "#76b900"
THRESHOLD_BLUE = "#79add2"  # Figure 1's blue mixed with 40% white.
GRID, TEXT = "#dddddd", "#333333"
POINTS = ("Conservative", "Balanced", "Aggressive")
DENSITY_FLOOR = 0.05


def decision_boundaries(idle, speech):
    """Original protocol's threshold definitions, evaluated on calibration."""
    scores = sorted(set(idle + speech))
    boundaries = [np.nextafter(scores[0], -np.inf)]
    boundaries += [(left + right) / 2 for left, right in zip(scores, scores[1:])]
    boundaries.append(scores[-1])

    def accuracy(boundary):
        return (np.mean([s <= boundary for s in idle])
                + np.mean([s > boundary for s in speech])) / 2

    conservative = max((s for s in idle if s < min(speech)), default=0.0)
    next_speech = [s for s in speech if s > max(idle)]
    aggressive = np.nextafter(min(next_speech), -np.inf) if next_speech else max(idle)
    return conservative, float(max(boundaries, key=accuracy)), float(aggressive)


def raw_scores(paths):
    """Read first-onset scores, never the generator's cached gate decisions."""
    scores = {}
    for model, path in zip(("moshi", "personaplex"), paths):
        raw = json.loads(path.read_text())
        onsets = raw["microphone_rollouts"]
        responses = raw["microphone_responses"]
        for rows in (onsets, responses):
            if len(rows) != 500 or {r["seed"] for r in rows} != set(range(10000, 10500)):
                raise ValueError(f"{model}: expected 500 unique seeds, 10000 through 10499")
        onset_scores = {r["seed"]: r["scores"][0]["score"] if r["scores"] else None
                        for r in onsets}
        response_scores = {r["seed"]: r["score"] for r in responses}
        scores[model] = [[seed, onset_scores[seed], response_scores[seed]]
                         for seed in sorted(onset_scores)]
    return scores


def verify(data, archive=None, scores=None):
    cal = set(data["split"]["calibration_seeds"])
    held = set(data["split"]["evaluation_seeds"])
    assert len(cal) == 100 and len(held) == 400 and not cal & held
    assert cal | held == set(range(10000, 10500))
    sampled = np.random.default_rng(data["split"]["rng_seed"]).choice(
        np.arange(10040, 10500), 60, replace=False)
    assert cal == set(range(10000, 10040)) | set(sampled.tolist())

    if archive is not None:
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == data["source_zip_sha256"]
        with zipfile.ZipFile(archive) as bundle:
            for model in ("moshi", "personaplex"):
                raw = json.load(bundle.open(f"runs/{model}/results.json"))
                onsets = {r["seed"]: r["scores"][0]["score"] if r["scores"] else None
                          for r in raw["microphone_rollouts"]}
                responses = {r["seed"]: r["score"] for r in raw["microphone_responses"]}
                assert set(onsets) == set(responses) == cal | held
                assert data["scores"][model] == [
                    [seed, onsets[seed], responses[seed]] for seed in sorted(onsets)]

    prepared = []
    for model in ("moshi", "personaplex"):
        rows = (data["scores"] if scores is None else scores)[model]
        assert len(rows) == 500 and {r[0] for r in rows} == cal | held
        assert [r[0] for r in rows] == sorted(cal | held)
        if any(value is not None and (not np.isfinite(value) or value < 0)
               for row in rows for value in row[1:]):
            raise ValueError(f"{model}: scores must be finite, nonnegative, or null")
        calibration = [[r[i] for r in rows if r[0] in cal and r[i] is not None]
                       for i in (1, 2)]
        evaluation = [[r[i] for r in rows if r[0] in held and r[i] is not None]
                      for i in (1, 2)]
        if not all(calibration):
            raise ValueError(f"{model}: calibration needs both spurious and response scores")
        thresholds = decision_boundaries(*calibration)
        expected = data["models"][model] if scores is None else None
        if expected is not None:
            assert len(evaluation[0]) == expected["evaluation_spurious_onsets"]
            assert len(evaluation[1]) == expected["evaluation_observed_responses"]
            assert max(evaluation[0]) < min(evaluation[1])
        print(f"{model}: {len(evaluation[0])} spurious onsets; "
              f"{len(evaluation[1])}/{len(held)} observed responses")
        for name, threshold in zip(POINTS, thresholds):
            point = expected["operating_points"][name] if expected is not None else None
            if point is not None:
                assert threshold == point["threshold"]
            counts = (sum(s <= threshold for s in evaluation[0]),
                      sum(s > threshold for s in evaluation[1]))
            bounds = []
            for metric, numerator, denominator in zip(
                    ("suppression", "responses"), counts, (len(evaluation[0]), len(held))):
                lower = (None if denominator == 0 else 0.0 if numerator == 0 else
                         100 * beta.ppf(.05, numerator, denominator - numerator + 1))
                if point is not None:
                    result = point[metric]
                    assert (numerator, denominator) == (result["numerator"], result["denominator"])
                    assert abs(lower - result["lower_95_percent"]) < 1e-10
                bounds.append("n/a" if lower is None else f"{lower:.2f}%")
            print(f'{model} {name}: '
                  f'{bounds[0]} suppressed; {bounds[1]} preserved '
                  '(one-sided 95% lower bounds)')
        prepared.append((evaluation, thresholds))
    return prepared


def paper_axes(height, boxes):
    """Place plot interiors in the manuscript's final, top-origin PDF geometry."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelpad": 5,
        }
    )
    fig = plt.figure(figsize=(WIDTH / 72, height / 72))
    axes = [
        fig.add_axes(
            (
                left / WIDTH,
                (height - bottom) / height,
                (right - left) / WIDTH,
                (bottom - top) / height,
            )
        )
        for left, top, right, bottom in boxes
    ]
    return fig, axes


def model_title(fig, ax, name, height):
    fig.text(
        (ax.get_position().x0 + ax.get_position().x1) / 2,
        1 - 10 / height,
        name,
        ha="center",
        va="baseline",
        fontsize=FONT_SIZE,
        fontweight="bold",
        color=TEXT,
    )


def style(ax, *, grid_axis="both"):
    for spine in ax.spines.values():
        spine.set_color(TEXT)
        spine.set_linewidth(0.65)
    ax.tick_params(colors=TEXT, labelsize=FONT_SIZE, width=0.65, length=2, pad=2)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def density_display(values, floor):
    """Estimate on all scores; the mask affects drawing, never verification."""
    if not 0 <= floor < 1:
        raise ValueError("Density floor must be in [0, 1).")
    if len(values) < 3 or np.ptp(values) == 0:
        return np.ones(len(values), dtype=bool), None
    kde = GaussianKDE(values, bw_method="scott")
    grid = np.linspace(values.min(), values.max(), 512)
    density = kde.evaluate(grid)
    peak = density.max()
    keep = kde.evaluate(values) >= floor * peak
    return keep, (grid, density / peak)


def plot(prepared, output, density_floor=DENSITY_FLOOR):
    models = [(name, color, tuple(np.asarray(s) for s in scores), thresholds)
              for name, color, (scores, thresholds) in zip(
                  ("Moshi", "PersonaPlex"), (MOSHI, PERSONAPLEX), prepared)]
    styles = ((0, (1, 1.5)), "-", (0, (5, 2)))
    labels = ("Conservative", "Balanced", "Aggressive")
    line_order = (1, 0, 2)
    fig, axes = paper_axes(96, ((36, 14, 133, 59), (148, 14, 242, 59)))
    positions = (0, 0.72)

    for model_index, (ax, (name, color, scores, thresholds)) in enumerate(
        zip(axes, models)
    ):
        visible_extents = []
        for score_index, (y, values) in enumerate(zip(positions, scores)):
            keep, profile = density_display(values, density_floor)
            if profile is not None:
                grid, relative_density = profile
                visible = relative_density >= density_floor
                ax.fill_between(
                    grid, y, y + 0.25 * relative_density, where=visible,
                    facecolor=color, edgecolor=color, alpha=0.22,
                )
                visible_extents.append(grid[visible].max())
            if keep.any():
                visible_extents.append(values[keep].max())
            print(f"Display only: {name} {('spurious', 'response')[score_index]}: "
                  f"{int((~keep).sum())}/{len(values)} points omitted "
                  f"at {100 * density_floor:g}% of peak KDE density.")
            jitter = np.random.default_rng(10 * model_index + score_index).uniform(
                -0.14, -0.05, len(values)
            )
            ax.scatter(
                values[keep],
                y + jitter[keep],
                color=color,
                s=2,
                alpha=0.4,
                edgecolors="none",
            )
        for index in line_order:
            ax.axvline(
                thresholds[index],
                color=THRESHOLD_BLUE,
                linestyle=styles[index],
                linewidth=1.4 if index == 1 else 0.8,
                zorder=4,
            )
        model_title(fig, ax, name, 96)
        upper = (
            np.ceil(
                10
                * max(
                    *visible_extents,
                    *thresholds,
                    0.1,
                )
            )
            / 10
        )
        ax.set_xlim(-0.015, upper)
        ax.set_ylim(-0.24, 1.02)
        ax.set_yticks(positions, ("spurious", "response"))
        ax.set_xticks(np.arange(0, upper + 0.001, 0.1 if upper <= 0.2 else 0.2))
        if np.isclose(ax.get_xticks()[-1], upper):
            ax.get_xticklabels()[-1].set_horizontalalignment("right")
        style(ax, grid_axis="x")
        ax.tick_params(axis="y", length=0)
        if model_index:
            ax.tick_params(axis="y", labelleft=False)

    fig.text(
        0.5,
        1 - 81 / 96,
        "Divergence (Dₜ)",
        ha="center",
        va="baseline",
        fontsize=FONT_SIZE,
        fontweight="bold",
        color=TEXT,
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=THRESHOLD_BLUE,
                linestyle=styles[index],
                linewidth=1.0,
                label=label,
            )
            for index, label in enumerate(labels)
        ],
        frameon=False,
        fontsize=FONT_SIZE,
        labelcolor=TEXT,
        handlelength=1.6,
        loc="lower left",
        bbox_to_anchor=(2 / WIDTH, 0.5 / 96, (WIDTH - 4) / WIDTH, 1),
        bbox_transform=fig.transFigure,
        mode="expand",
        ncol=3,
        borderaxespad=0,
        borderpad=0,
    )
    fig.savefig(output, metadata={"CreationDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "mitigation_results.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "figures" / "counterfactual-readable.pdf")
    raw_input = parser.add_mutually_exclusive_group()
    raw_input.add_argument("--verify-zip", type=Path)
    raw_input.add_argument("--raw", type=Path, nargs=2, metavar=("MOSHI", "PERSONAPLEX"),
                           help="Analyze new 500-rollout results using the released split.")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--density-floor", type=float, default=DENSITY_FLOOR,
                        help="Display-only fraction of each KDE's peak density; 0 shows all scores.")
    args = parser.parse_args()
    if not 0 <= args.density_floor < 1:
        parser.error("--density-floor must be in [0, 1).")
    prepared = verify(json.loads(args.data.read_text()), args.verify_zip,
                      raw_scores(args.raw) if args.raw else None)
    if not args.verify_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plot(prepared, args.output, args.density_floor)


if __name__ == "__main__":
    main()
