#!/usr/bin/env python3
"""
test_env.py — Verify the training environment is correctly set up.

Run: python3 finetuning/tests/test_env.py
"""

import sys
import unittest


class TestEnvironment(unittest.TestCase):

    def test_torch_available(self):
        import torch
        print(f"\n  PyTorch {torch.__version__}")
        self.assertTrue(torch.__version__.startswith("2."), f"Expected PyTorch 2.x, got {torch.__version__}")

    def test_cuda_available(self):
        import torch
        self.assertTrue(torch.cuda.is_available(), "CUDA not available — GPU required for training")
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")

    def test_k2_importable(self):
        import k2
        ver = getattr(k2, '__dev_version__', getattr(k2, '__version__', 'unknown'))
        print(f"\n  k2 {ver}")

    def test_k2_cuda(self):
        import k2
        import torch
        # k2 ops require CUDA — verify a basic op works on GPU
        fsa = k2.Fsa.from_str("0 1 1 0.5\n1")
        self.assertIsNotNone(fsa)

    def test_kaldifeat_importable(self):
        import kaldifeat
        print(f"\n  kaldifeat installed")

    def test_kaldifeat_gpu(self):
        import kaldifeat
        import torch
        opts = kaldifeat.FbankOptions()
        opts.device = "cuda"
        opts.mel_opts.num_bins = 80
        fbank = kaldifeat.Fbank(opts)
        # 0.1s of silence
        wave = torch.zeros(1600, device="cuda")
        feats = fbank(wave.unsqueeze(0))
        self.assertEqual(feats.shape[-1], 80)
        print(f"\n  kaldifeat GPU fbank OK: {feats.shape}")

    def test_lhotse_importable(self):
        import lhotse
        print(f"\n  lhotse {lhotse.__version__}")

    def test_sentencepiece_importable(self):
        import sentencepiece
        print(f"\n  sentencepiece {sentencepiece.__version__}")

    def test_omegaconf_importable(self):
        import omegaconf
        print(f"\n  omegaconf {omegaconf.__version__}")

    def test_tensorboard_importable(self):
        from torch.utils.tensorboard import SummaryWriter
        print(f"\n  tensorboard OK")

    def test_icefall_importable(self):
        import sys
        from pathlib import Path
        # Try both pod path and local dev path
        for candidate in [
            "/workspace/finetune/icefall",
            str(Path(__file__).resolve().parents[2] / "finetuning" / "icefall"),
        ]:
            if Path(candidate).exists() and candidate not in sys.path:
                sys.path.insert(0, candidate)
                zipformer_dir = str(Path(candidate) / "egs" / "librispeech" / "ASR" / "zipformer")
                if Path(zipformer_dir).exists():
                    sys.path.insert(0, zipformer_dir)
                break
        import icefall
        print(f"\n  icefall OK")

    def test_icefall_zipformer_train_importable(self):
        import sys
        from pathlib import Path
        zipformer_dir = "/workspace/finetune/icefall/egs/librispeech/ASR/zipformer"
        if Path(zipformer_dir).exists() and zipformer_dir not in sys.path:
            sys.path.insert(0, zipformer_dir)
        # The finetune.py (not train.py) is the finetuning entry point in icefall
        import importlib.util
        for script in ["finetune", "train"]:
            spec = importlib.util.spec_from_file_location(
                script, f"{zipformer_dir}/{script}.py"
            )
            if spec:
                print(f"\n  Found icefall zipformer/{script}.py")
                return
        self.fail("Neither finetune.py nor train.py found in icefall zipformer dir")


if __name__ == "__main__":
    unittest.main(verbosity=2)
