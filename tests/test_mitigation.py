import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import plot_mitigation as mitigation


def fixture():
    """Synthetic scores only; the split matches the released protocol."""
    calibration = set(range(10000, 10040)) | set(
        np.random.default_rng(20260916).choice(np.arange(10040, 10500), 60, replace=False)
    )
    data = {"split": {
        "rng_seed": 20260916,
        "calibration_seeds": sorted(calibration),
        "evaluation_seeds": sorted(set(range(10000, 10500)) - calibration),
    }}
    scores = {model: [[seed, 0.0, 0.2 + (seed % 10) / 100]
                      for seed in range(10000, 10500)]
              for model in ("moshi", "personaplex")}
    return data, scores


class MitigationTests(unittest.TestCase):
    def analyze(self, data, scores):
        with redirect_stdout(io.StringIO()) as output:
            prepared = mitigation.verify(data, scores=scores)
        return prepared, output.getvalue()

    def test_evaluation_never_changes_calibrated_thresholds(self):
        data, scores = fixture()
        before, _ = self.analyze(data, scores)
        held = set(data["split"]["evaluation_seeds"])
        for rows in scores.values():
            for row in rows:
                if row[0] in held:
                    row[1:] = [0.5, 0.0]
        after, report = self.analyze(data, scores)
        self.assertEqual([x[1] for x in before], [x[1] for x in after])
        self.assertIn("0.00% suppressed; 0.00% preserved", report)

    def test_missing_responses_stay_in_denominator(self):
        data, scores = fixture()
        seed = data["split"]["evaluation_seeds"][0]
        scores["moshi"][seed - 10000][2] = None
        _, report = self.analyze(data, scores)
        self.assertIn("399/400 observed responses", report)
        self.assertIn("98.82% preserved", report)
        self.assertIn("99.25% preserved", report)

    def test_empty_held_out_onset_group_is_not_reported_as_success(self):
        data, scores = fixture()
        held = set(data["split"]["evaluation_seeds"])
        for rows in scores.values():
            for row in rows:
                if row[0] in held:
                    row[1] = None
        _, report = self.analyze(data, scores)
        self.assertIn("n/a suppressed", report)

    def test_raw_input_requires_exactly_500_unique_seeds(self):
        _, scores = fixture()
        rows = scores["moshi"]
        raw = {
            "microphone_rollouts": [
                {"seed": seed, "scores": [{"score": onset}]} for seed, onset, _ in rows
            ],
            "microphone_responses": [
                {"seed": seed, "score": response} for seed, _, response in rows
            ],
            "boundaries": {"ignored": 999},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps(raw))
            self.assertEqual(mitigation.raw_scores([path, path]), scores)
            raw["microphone_responses"][-1] = raw["microphone_responses"][0]
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                mitigation.raw_scores([path, path])

    def test_display_filter_does_not_change_scores_and_can_be_disabled(self):
        values = np.r_[np.linspace(0.1, 0.2, 399), 5.0]
        original = values.copy()
        keep, _ = mitigation.density_display(values, 0.05)
        self.assertFalse(keep[-1])
        self.assertTrue(mitigation.density_display(values, 0)[0].all())
        np.testing.assert_array_equal(values, original)
        for values in (np.array([]), np.zeros(5)):
            self.assertTrue(mitigation.density_display(values, 0.05)[0].all())
        for floor in (-0.1, 1, np.nan):
            with self.assertRaises(ValueError):
                mitigation.density_display(original, floor)

    def test_paper_colors_legend_and_geometry(self):
        data, scores = fixture()
        prepared, _ = self.analyze(data, scores)
        original = copy.deepcopy(prepared)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "figure.pdf"
            with patch.object(plt, "close"), redirect_stdout(io.StringIO()):
                mitigation.plot(prepared, output)
                fig = plt.gcf()
            self.addCleanup(plt.close, fig)
            self.assertGreater(output.stat().st_size, 1000)
        self.assertEqual(prepared, original)
        np.testing.assert_allclose(fig.get_size_inches() * 72, [mitigation.WIDTH, 96])
        self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()],
                         list(mitigation.POINTS))
        lines = [line for ax in fig.axes for line in ax.lines]
        lines += fig.legends[0].get_lines()
        self.assertEqual(len(lines), 9)
        self.assertTrue(all(line.get_color() == "#79add2" for line in lines))


if __name__ == "__main__":
    unittest.main()
