from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.FixedValidation import (
    flatten_fixed_validation_sets,
    load_or_create_fixed_validation_sets,
    summarize_fixed_validation_sets,
)


class FixedValidationTest(unittest.TestCase):
    def test_files_are_created_once_and_reloaded_identically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = load_or_create_fixed_validation_sets(
                dataset_dir="instance/Data",
                validation_dir=root,
                problem_sizes=(20, 40),
                per_size=2,
                seed=24610,
            )
            first_summary = summarize_fixed_validation_sets(first)
            mtimes = {path.name: path.stat().st_mtime_ns for path in root.glob("J*.pt")}
            second = load_or_create_fixed_validation_sets(
                dataset_dir="instance/Data",
                validation_dir=root,
                problem_sizes=(20, 40),
                per_size=2,
                seed=999999,
            )
            second_summary = summarize_fixed_validation_sets(second)
            self.assertEqual(first_summary, second_summary)
            self.assertEqual(
                mtimes,
                {path.name: path.stat().st_mtime_ns for path in root.glob("J*.pt")},
            )
            self.assertEqual(first_summary["counts"], {"20": 2, "40": 2})
            self.assertEqual(first_summary["total"], 4)
            self.assertEqual(len(flatten_fixed_validation_sets(second)), 4)
            self.assertEqual(second_summary["shapes"]["20"]["targets"], [2, 20, 2])


if __name__ == "__main__":
    unittest.main()
