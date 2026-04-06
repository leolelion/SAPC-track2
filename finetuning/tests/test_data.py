#!/usr/bin/env python3
"""
test_data.py — Test SAPC data preparation and lhotse CutSet loading.

Run: python3 finetuning/tests/test_data.py [--data-root /workspace/SAPC2]
"""

import argparse
import sys
import unittest
from pathlib import Path


# Pod default
DEFAULT_DATA_ROOT = "/workspace/SAPC2"
DEFAULT_CUTS_DIR  = "/workspace/finetune/data"

data_root = Path(DEFAULT_DATA_ROOT)
cuts_dir  = Path(DEFAULT_CUTS_DIR)


class TestSAPCData(unittest.TestCase):

    def test_manifest_csvs_exist(self):
        for split in ["Train", "Dev"]:
            p = data_root / "manifest" / f"{split}.csv"
            self.assertTrue(p.exists(), f"Missing: {p}")
            print(f"\n  Found: {p}")

    def test_manifest_csv_columns(self):
        import pandas as pd
        df = pd.read_csv(data_root / "manifest" / "Dev.csv")
        required = {"id", "speaker", "etiology", "audio_filepath",
                    "duration", "norm_text_without_disfluency"}
        missing = required - set(df.columns)
        self.assertFalse(missing, f"Missing columns: {missing}")
        print(f"\n  Dev.csv: {len(df)} rows, columns OK")

    def test_audio_files_accessible(self):
        import pandas as pd
        df = pd.read_csv(data_root / "manifest" / "Dev.csv")
        # Check first 10 rows only
        missing = []
        for _, row in df.head(10).iterrows():
            p = data_root / row["audio_filepath"]
            if not p.exists():
                missing.append(str(p))
        self.assertFalse(missing, f"Missing audio files:\n" + "\n".join(missing[:5]))
        print(f"\n  First 10 Dev audio files: all found")

    def test_cutset_files_exist(self):
        for fname in ["sapc2_train_cuts.jsonl.gz", "sapc2_dev_cuts.jsonl.gz"]:
            p = cuts_dir / fname
            self.assertTrue(p.exists(), f"Missing CutSet: {p}\n"
                            "Run: python3 finetuning/prepare_sapc_lhotse.py --data-root {data_root}")
            print(f"\n  Found: {p}")

    def test_cutset_loadable(self):
        from lhotse import load_manifest_lazy
        dev_cuts_path = cuts_dir / "sapc2_dev_cuts.jsonl.gz"
        self.assertTrue(dev_cuts_path.exists(), f"Missing: {dev_cuts_path}")
        cuts = load_manifest_lazy(str(dev_cuts_path))
        sample = next(iter(cuts))
        self.assertIsNotNone(sample.supervisions[0].text)
        print(f"\n  Dev cuts loadable. Sample text: '{sample.supervisions[0].text[:60]}'")

    def test_cutset_durations(self):
        from lhotse import load_manifest_lazy
        dev_cuts_path = cuts_dir / "sapc2_dev_cuts.jsonl.gz"
        if not dev_cuts_path.exists():
            self.skipTest("Dev CutSet not found")
        cuts = list(load_manifest_lazy(str(dev_cuts_path)))
        durations = [c.duration for c in cuts]
        total_hours = sum(durations) / 3600
        avg = sum(durations) / len(durations)
        self.assertGreater(len(cuts), 100, "Expected >100 Dev utterances")
        self.assertGreater(total_hours, 0.5, "Expected >0.5h of Dev audio")
        print(f"\n  Dev: {len(cuts)} cuts, {total_hours:.2f}h, avg {avg:.1f}s")

    def test_train_cutset_stats(self):
        from lhotse import load_manifest_lazy
        train_cuts_path = cuts_dir / "sapc2_train_cuts.jsonl.gz"
        if not train_cuts_path.exists():
            self.skipTest("Train CutSet not found")
        cuts = list(load_manifest_lazy(str(train_cuts_path)))
        total_hours = sum(c.duration for c in cuts) / 3600
        self.assertGreater(total_hours, 1.0, "Expected >1h of Train audio")
        print(f"\n  Train: {len(cuts)} cuts, {total_hours:.2f}h")

    def test_prepare_script_dry_run(self):
        """Run prepare_sapc_lhotse.py on a 5-row subset to verify it works end-to-end."""
        import subprocess, tempfile, textwrap, os
        script = Path(__file__).resolve().parents[2] / "finetuning" / "prepare_sapc_lhotse.py"
        if not script.exists():
            self.skipTest(f"prepare_sapc_lhotse.py not found at {script}")

        dev_csv = data_root / "manifest" / "Dev.csv"
        if not dev_csv.exists():
            self.skipTest(f"Dev.csv not found at {dev_csv}")

        import pandas as pd
        df = pd.read_csv(dev_csv).head(5)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write mini CSV
            mini_csv_dir = Path(tmpdir) / "manifest"
            mini_csv_dir.mkdir()
            # Fix audio_filepath to absolute paths for this mini CSV
            df = df.copy()
            df["audio_filepath"] = df["audio_filepath"].apply(
                lambda p: str(data_root / p) if not Path(p).is_absolute() else p
            )
            mini_csv = mini_csv_dir / "Dev.csv"
            df.to_csv(mini_csv, index=False)

            out_dir = Path(tmpdir) / "cuts"
            result = subprocess.run(
                [sys.executable, str(script),
                 "--data-root", tmpdir,
                 "--output-dir", str(out_dir)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                print(f"\nSTDOUT:\n{result.stdout[-2000:]}")
                print(f"STDERR:\n{result.stderr[-2000:]}")
            self.assertEqual(result.returncode, 0, f"prepare_sapc_lhotse.py failed:\n{result.stderr[-1000:]}")
            self.assertTrue((out_dir / "cuts_dev.jsonl.gz").exists())
            print(f"\n  prepare_sapc_lhotse.py dry run: OK (5 utterances)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cuts-dir",  default=DEFAULT_CUTS_DIR)
    known, remaining = parser.parse_known_args()
    data_root = Path(known.data_root)
    cuts_dir  = Path(known.cuts_dir)
    sys.argv = [sys.argv[0]] + remaining
    unittest.main(verbosity=2)
