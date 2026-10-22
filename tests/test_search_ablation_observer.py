from pathlib import Path
import tempfile
import unittest

from scripts.watch_search_ablation import completed_metric_rows


class ObserverCSVTests(unittest.TestCase):
    def test_partial_epoch_is_ignored_then_visible_after_append(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text("epoch,target_mean_objective\n1,22.6\n2,21.")
            self.assertEqual(completed_metric_rows(path), [
                {"epoch": "1", "target_mean_objective": "22.6"}])
            with path.open("a") as handle:
                handle.write("5\n")
            self.assertEqual(completed_metric_rows(path)[-1],
                             {"epoch": "2", "target_mean_objective": "21.5"})

    def test_missing_empty_and_partial_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            self.assertEqual(completed_metric_rows(path), [])
            for text in ("", "epoch,target_mean", "epoch,target_mean_objective\n"):
                path.write_text(text)
                self.assertEqual(completed_metric_rows(path), [])


if __name__ == "__main__":
    unittest.main()
