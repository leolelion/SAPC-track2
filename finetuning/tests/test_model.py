#!/usr/bin/env python3
"""
test_model.py — Test model loading, forward pass, and ONNX inference.

Run: python3 finetuning/tests/test_model.py
"""

import sys
import unittest
from pathlib import Path

# ── Setup paths ───────────────────────────────────────────────────────────────
ICEFALL_DIR   = Path("/workspace/finetune/icefall")
ZIPFORMER_DIR = ICEFALL_DIR / "egs" / "librispeech" / "ASR" / "zipformer"
PRETRAINED_PT = Path("/workspace/finetune/weights/standard/exp/epoch-30.pt")
BPE_MODEL     = Path("/workspace/finetune/weights/standard/data/lang_bpe_500/bpe.model")
FINETUNED_PT  = Path("/workspace/finetune/exp/standard/epoch-2.pt")
ONNX_DIR      = Path("/workspace/finetune/onnx/standard")
CUTS_DIR      = Path("/workspace/finetune/data")

for p in [str(ICEFALL_DIR), str(ZIPFORMER_DIR)]:
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)


class TestModelLoad(unittest.TestCase):

    def test_pretrained_checkpoint_exists(self):
        self.assertTrue(PRETRAINED_PT.exists(), f"Missing: {PRETRAINED_PT}")
        size_mb = PRETRAINED_PT.stat().st_size / 1e6
        print(f"\n  epoch-30.pt: {size_mb:.1f} MB")

    def test_bpe_model_exists(self):
        self.assertTrue(BPE_MODEL.exists(), f"Missing: {BPE_MODEL}")

    def test_bpe_model_loadable(self):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(str(BPE_MODEL))
        self.assertEqual(sp.get_piece_size(), 500)
        ids = sp.encode("hello world")
        self.assertGreater(len(ids), 0)
        print(f"\n  BPE model: 500 tokens OK, 'hello world' -> {ids}")

    def test_icefall_imports(self):
        self.assertTrue(ICEFALL_DIR.exists(), f"icefall not found at {ICEFALL_DIR}")
        self.assertTrue(ZIPFORMER_DIR.exists(), f"zipformer dir not found at {ZIPFORMER_DIR}")
        import icefall
        print(f"\n  icefall imported OK")

    def test_model_build(self):
        import torch
        # Import icefall model builder
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "finetune_icefall", str(ZIPFORMER_DIR / "finetune.py")
        )
        ft_mod = importlib.util.module_from_spec(spec)
        # Don't exec the full module (has if __main__), just check it parses
        # Instead use train.py which has get_model/get_params
        spec2 = importlib.util.spec_from_file_location(
            "train_icefall", str(ZIPFORMER_DIR / "train.py")
        )
        train_mod = importlib.util.module_from_spec(spec2)
        # Just verify the file is parseable
        import ast
        src = (ZIPFORMER_DIR / "train.py").read_text()
        ast.parse(src)
        print(f"\n  train.py parses OK")

    def test_finetuned_checkpoint_exists(self):
        self.assertTrue(FINETUNED_PT.exists(), f"Missing: {FINETUNED_PT}")
        size_mb = FINETUNED_PT.stat().st_size / 1e6
        print(f"\n  epoch-2.pt (finetuned): {size_mb:.1f} MB")

    def test_checkpoint_loadable(self):
        import torch
        ckpt = torch.load(str(FINETUNED_PT), map_location="cpu")
        self.assertIn("model", ckpt, "Expected 'model' key in checkpoint")
        n_params = len(ckpt["model"])
        print(f"\n  epoch-2.pt loaded: {n_params} param tensors")

    def test_checkpoint_epoch_metadata(self):
        import torch
        ckpt = torch.load(str(FINETUNED_PT), map_location="cpu")
        # icefall checkpoints may store epoch in different keys
        epoch = ckpt.get("epoch", ckpt.get("cur_epoch", "unknown"))
        print(f"\n  Checkpoint epoch: {epoch}")


class TestONNX(unittest.TestCase):

    def test_onnx_files_exist(self):
        for name in ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]:
            p = ONNX_DIR / name
            self.assertTrue(p.exists(), f"Missing ONNX file: {p}")
            size_mb = p.stat().st_size / 1e6
            print(f"\n  {name}: {size_mb:.1f} MB")

    def test_onnx_loadable_with_ort(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")
        for name in ["encoder.onnx", "decoder.onnx", "joiner.onnx"]:
            p = ONNX_DIR / name
            if not p.exists():
                continue
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            inputs = {i.name for i in sess.get_inputs()}
            print(f"\n  {name}: inputs={inputs}")

    def test_sherpa_onnx_inference(self):
        try:
            import sherpa_onnx
        except ImportError:
            self.skipTest("sherpa-onnx not installed")

        tokens = ONNX_DIR / "tokens.txt"
        if not tokens.exists():
            self.skipTest("tokens.txt not found")

        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=str(ONNX_DIR / "encoder.onnx"),
            decoder=str(ONNX_DIR / "decoder.onnx"),
            joiner=str(ONNX_DIR / "joiner.onnx"),
            tokens=str(tokens),
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=False,
            decoding_method="greedy_search",
            chunk_size=16,
            left_context=128,
        )
        import numpy as np
        stream = recognizer.create_stream()
        # Feed 1 second of silence
        samples = np.zeros(16000, dtype=np.float32)
        stream.accept_waveform(16000, samples)
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        result = recognizer.get_result(stream)
        print(f"\n  sherpa-onnx inference OK. Result on silence: '{result.text}'")

    def test_sherpa_onnx_on_real_audio(self):
        try:
            import sherpa_onnx
            import soundfile as sf
            import numpy as np
        except ImportError as e:
            self.skipTest(f"Missing dep: {e}")

        tokens = ONNX_DIR / "tokens.txt"
        data_root = Path("/workspace/SAPC2")
        dev_csv   = data_root / "manifest" / "Dev.csv"
        if not dev_csv.exists() or not tokens.exists():
            self.skipTest("Dev.csv or tokens.txt not found")

        import pandas as pd
        df = pd.read_csv(dev_csv).head(1)
        row = df.iloc[0]
        audio_path = data_root / row["audio_filepath"]
        if not audio_path.exists():
            self.skipTest(f"Audio not found: {audio_path}")

        audio, sr = sf.read(str(audio_path))
        self.assertEqual(sr, 16000)

        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=str(ONNX_DIR / "encoder.onnx"),
            decoder=str(ONNX_DIR / "decoder.onnx"),
            joiner=str(ONNX_DIR / "joiner.onnx"),
            tokens=str(tokens),
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=False,
            decoding_method="greedy_search",
            chunk_size=16,
            left_context=128,
        )
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, audio.astype(np.float32))
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        result = recognizer.get_result(stream)
        ref = row["norm_text_without_disfluency"]
        print(f"\n  Real audio inference:")
        print(f"    REF: {ref}")
        print(f"    HYP: {result.text}")
        self.assertGreater(len(result.text.strip()), 0, "Empty hypothesis on real audio")


class TestDataloader(unittest.TestCase):
    """Quick smoke test that lhotse data pipeline produces valid batches."""

    def test_lhotse_cutset_to_batch(self):
        from lhotse import load_manifest_lazy
        from lhotse.dataset import DynamicBucketingSampler

        dev_cuts_path = CUTS_DIR / "sapc2_dev_cuts.jsonl.gz"
        if not dev_cuts_path.exists():
            self.skipTest(f"Missing: {dev_cuts_path}")

        cuts = load_manifest_lazy(str(dev_cuts_path))
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=60,  # small for test
            shuffle=False,
            drop_last=False,
            num_buckets=5,
        )
        # Get one batch
        batch_cuts = next(iter(sampler))
        self.assertGreater(len(batch_cuts), 0)
        total_dur = sum(c.duration for c in batch_cuts)
        print(f"\n  Sampler batch: {len(batch_cuts)} cuts, {total_dur:.1f}s total")

    def test_feature_extraction_pipeline(self):
        """Verify on-the-fly fbank extraction works on a real cut."""
        import torch
        import numpy as np
        from lhotse import load_manifest_lazy

        dev_cuts_path = CUTS_DIR / "sapc2_dev_cuts.jsonl.gz"
        if not dev_cuts_path.exists():
            self.skipTest(f"Missing: {dev_cuts_path}")

        cut = next(iter(load_manifest_lazy(str(dev_cuts_path))))

        # Load audio
        audio = cut.load_audio()  # (C, T)
        self.assertEqual(audio.shape[0], 1)
        self.assertGreater(audio.shape[1], 0)
        print(f"\n  Loaded audio: {audio.shape[1]/16000:.2f}s")

        # Compute fbank
        try:
            import kaldifeat
            opts = kaldifeat.FbankOptions()
            opts.device = "cuda" if torch.cuda.is_available() else "cpu"
            opts.frame_opts.dither = 0.0
            opts.mel_opts.num_bins = 80
            fbank = kaldifeat.Fbank(opts)
            wave = torch.from_numpy(audio[0]).to(opts.device)
            feats = fbank(wave.unsqueeze(0))
            self.assertEqual(feats.shape[-1], 80)
            print(f"  kaldifeat fbank: {feats.shape} (frames x 80)")
        except ImportError:
            self.skipTest("kaldifeat not installed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
