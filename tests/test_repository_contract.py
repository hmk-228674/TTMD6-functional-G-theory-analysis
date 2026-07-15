from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_reproduction_and_figure_qa_pass(self) -> None:
        status = json.loads(
            (ROOT / "reference_results/reproduction/REPRODUCTION_STATUS.json").read_text()
        )
        figure_qa = json.loads((ROOT / "figure_source_data/Figure_QA.json").read_text())
        self.assertEqual(status["status"], "PASS")
        self.assertTrue(all(value == "PASS" or value == "ABSENT" for value in status["regression_checks"].values()))
        self.assertTrue(figure_qa["hard_checks_pass"])
        self.assertEqual(figure_qa["text_boundary_violations_total"], 0)

    def test_primary_n90_regression_values(self) -> None:
        table = pd.read_csv(ROOT / "reference_results/primary/Table_R2_FullBalanced_Integrated.csv")
        expected = {
            "racket": [6, 7, 8, 17, 27, 10],
            "body_configuration": [6, 6, 4, 7, 8, 4],
        }
        for waveform, values in expected.items():
            observed = (
                table.loc[table.waveform == waveform]
                .sort_values("action_id")["required_n_R_L2_90"]
                .astype(int)
                .tolist()
            )
            self.assertEqual(observed, values)

    def test_all_figure_formats_present(self) -> None:
        bases = {
            path.stem
            for path in (ROOT / "figures").glob("*.png")
        }
        self.assertEqual(len(bases), 7)
        for base in bases:
            for suffix in (".png", ".pdf", ".svg", ".tif"):
                self.assertTrue((ROOT / "figures" / f"{base}{suffix}").is_file())


if __name__ == "__main__":
    unittest.main()
