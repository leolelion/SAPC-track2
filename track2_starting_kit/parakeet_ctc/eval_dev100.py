#!/usr/bin/env python3
"""Eval Parakeet CTC (fine-tuned) on dev100 with dual-reference scoring.

Uses direct encoder+decoder inference to avoid NeMo's lhotse dataloader
compatibility issues. Same dual-reference scoring as eval_parakeet_full_dev.py.
"""
import argparse, csv, os, re, sys, time, json, glob as _glob

# ── Inject parakeet_ctc venv into sys.path ──────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_venv_candidates = _glob.glob(
    os.path.join(_DIR, "venv", "lib", "python3.*", "site-packages")
) or _glob.glob(
    os.path.join(_DIR, "venv", "*", "lib", "python3.*", "site-packages")
)
if _venv_candidates:
    sys.path.insert(0, _venv_candidates[0])


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def cer(ref, hyp):
    if not ref:
        return 1.0 if hyp else 0.0
    a, b = ref, hyp
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]; dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1] / len(ref)


def wer(ref, hyp):
    r = ref.split(); h = hyp.split()
    if not r:
        return 1.0 if h else 0.0
    dp = list(range(len(h) + 1))
    for i, ra in enumerate(r, 1):
        prev = dp[0]; dp[0] = i
        for j, hb in enumerate(h, 1):
            cur = dp[j]
            dp[j] = prev if ra == hb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1] / len(r)


def transcribe_direct(model, wav_paths, batch_size, device):
    """Transcribe by directly running preprocessor -> encoder -> decoder.

    Avoids NeMo's model.transcribe() which uses lhotse and can have
    compatibility issues across NeMo/lhotse versions.
    """
    import torch
    import soundfile as sf
    import numpy as np

    vocabulary = list(model.decoder.vocabulary)
    blank_id = model.decoder.num_classes_with_blank - 1

    def ctc_greedy_decode(log_probs):
        token_ids = torch.argmax(log_probs, dim=-1).tolist()
        prev = -1
        tokens = []
        for tid in token_ids:
            if tid != prev:
                if tid != blank_id:
                    tokens.append(tid)
                prev = tid
        pieces = [vocabulary[t] for t in tokens if t < len(vocabulary)]
        return "".join(pieces).replace("\u2581", " ").strip()

    hyps = []
    for i in range(0, len(wav_paths), batch_size):
        batch_paths = wav_paths[i:i + batch_size]
        # Load and pad audio
        audios = []
        for p in batch_paths:
            audio, sr = sf.read(p, dtype="float32")
            if sr != 16000:
                raise ValueError(f"Expected 16kHz, got {sr}Hz for {p}")
            audios.append(torch.from_numpy(audio))
        lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
        max_len = lengths.max().item()
        padded = torch.zeros(len(audios), max_len)
        for j, a in enumerate(audios):
            padded[j, :a.shape[0]] = a
        padded = padded.to(device)
        lengths = lengths.to(device)

        with torch.inference_mode():
            features, feat_len = model.preprocessor(input_signal=padded, length=lengths)
            encoded, enc_len = model.encoder(audio_signal=features, length=feat_len)
            log_probs = model.decoder(encoder_output=encoded)

        for j in range(len(batch_paths)):
            seq_len = enc_len[j].item()
            hyps.append(ctc_greedy_decode(log_probs[j, :seq_len]))
        print(f"  {min(i + batch_size, len(wav_paths))}/{len(wav_paths)}", end="\r")
    print()
    return hyps


def main():
    p = argparse.ArgumentParser(description="Eval Parakeet CTC on dev100")
    p.add_argument("--model", default=os.path.join(_DIR, "weights", "final.nemo"),
                   help="Path to .nemo model (default: weights/final.nemo)")
    p.add_argument("--dev_csv", default=os.path.join(_DIR, "..", "..", "dev100_bundle", "Dev_100_local.csv"),
                   help="Path to Dev_100_local.csv")
    p.add_argument("--audio_root", default=os.path.join(_DIR, "..", "..", "dev100_bundle"),
                   help="Root dir for audio_filepath column")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default=os.path.join(_DIR, "dev100_results.jsonl"))
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()

    # Resolve relative paths
    args.dev_csv = os.path.abspath(args.dev_csv)
    args.audio_root = os.path.abspath(args.audio_root)
    args.model = os.path.abspath(args.model)
    args.out = os.path.abspath(args.out)

    print(f"Model:      {args.model}")
    print(f"Dev CSV:    {args.dev_csv}")
    print(f"Audio root: {args.audio_root}")

    import torch
    import nemo.collections.asr as nemo_asr
    print(f"Loading model ...")
    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.model)
    model.eval()
    device = "cpu"
    if args.device == "cuda" and torch.cuda.is_available():
        model.cuda()
        device = "cuda"
    else:
        print("Using CPU")

    rows = []
    with open(args.dev_csv) as f:
        for r in csv.DictReader(f):
            d = float(r["duration"])
            if d <= 0:
                continue
            rows.append(r)
    print(f"Loaded {len(rows)} dev100 samples")

    wavs = [os.path.join(args.audio_root, r["audio_filepath"]) for r in rows]

    # Verify audio files exist
    missing = [w for w in wavs if not os.path.exists(w)]
    if missing:
        print(f"WARNING: {len(missing)} audio files missing, e.g.: {missing[0]}")

    print(f"Transcribing (batch_size={args.batch}) ...")
    t0 = time.time()
    hyps = transcribe_direct(model, wavs, args.batch, device)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Dual-reference scoring: min CER/WER over (with, without) disfluency refs
    cers, wers = [], []
    per_etio = {}
    with open(args.out, "w") as f:
        for r, h in zip(rows, hyps):
            ref_wo = normalize(r.get("norm_text_without_disfluency", ""))
            ref_w = normalize(r.get("norm_text_with_disfluency", ""))
            hyp = normalize(h)
            candidates = [x for x in [ref_wo, ref_w] if x]
            if not candidates:
                continue
            c = min(cer(ref, hyp) for ref in candidates)
            w = min(wer(ref, hyp) for ref in candidates)
            cers.append(c); wers.append(w)
            etio = r.get("etiology", "?")
            per_etio.setdefault(etio, []).append(c)
            f.write(json.dumps({"id": r.get("id", ""), "etiology": etio,
                                "ref": ref_wo or ref_w, "hyp": hyp,
                                "cer": c, "wer": w}) + "\n")

    print(f"\n=== DEV100 RESULTS (Parakeet CTC) ===")
    print(f"Samples:  {len(cers)}")
    print(f"CER:      {sum(cers)/len(cers)*100:.2f}%")
    print(f"WER:      {sum(wers)/len(wers)*100:.2f}%")
    print(f"Time:     {elapsed:.1f}s")
    total_dur = sum(float(r["duration"]) for r in rows)
    print(f"RTF:      {elapsed/total_dur:.4f}")
    print(f"\nPer etiology CER:")
    for etio, clist in sorted(per_etio.items(), key=lambda x: -sum(x[1]) / len(x[1])):
        print(f"  {etio:<20} n={len(clist):>3}  CER={sum(clist)/len(clist)*100:.2f}%")
    print(f"\nResults written to: {args.out}")


if __name__ == "__main__":
    main()
