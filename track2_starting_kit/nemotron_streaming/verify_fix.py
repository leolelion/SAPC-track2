#!/usr/bin/env python3
"""Verify the feat_len fix eliminates trailing-token divergences."""
import csv
import os
import re
import sys
import wave

import numpy as np
import torch

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600


def read_wav(path):
    with wave.open(path, "rb") as f:
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    submission_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, submission_dir)

    # First: confirm feat_len vs shape difference with preprocessor
    from nemo.collections.asr.models import ASRModel
    model = ASRModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b", map_location="cpu"
    )
    model.to("cpu"); model.eval()
    model.encoder.set_default_att_context_size([70, 1])
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    test_path = "/workspace/SAPC2/processed/Dev/ea9825d9-3c46-49ed-b21d-08dc6a2b69be/ea9825d9-3c46-49ed-b21d-08dc6a2b69be_60_7474.wav"
    samples = read_wav(test_path)
    audio_t = torch.from_numpy(samples).unsqueeze(0)
    audio_len = torch.tensor([len(samples)], dtype=torch.long)
    with torch.no_grad():
        features, feat_len = model.preprocessor(input_signal=audio_t, length=audio_len)
    print(f"Preprocessor check: shape[2]={features.shape[2]}, feat_len={feat_len.item()}")
    print(f"  Extra padding frames: {features.shape[2] - feat_len.item()}")

    del model
    import gc; gc.collect()

    # Now test our fixed Model class on all 8 previously-divergent utterances
    print("\n=== Testing fixed Model class ===")
    from model import Model
    m = Model()

    # Load reference harness predictions for comparison
    ref_preds = {}
    with open("/workspace/nemotron_streaming/results/reference_predict.csv") as f:
        for row in csv.DictReader(f):
            ref_preds[row["id"]] = row["raw_hypos"]

    # The 8 divergent UIDs from prior analysis
    divergent_uids = [
        "02005a84-8847-4ef7-7b99-08dc286c108f_16310_3770",
        "15dec664-dc2b-4a05-4a54-08dc3c7b3134_46_5431",
        "51451a02-b769-42db-fd6d-08dcb5d1edd7_1072_8219",
        "9d33c23d-5a77-45dc-c547-08dbb7ad5db3_554_2318",
        "de99f6a4-1fca-498b-f781-08dcb8bbedb8_29094_12491",
        "ea9825d9-3c46-49ed-b21d-08dc6a2b69be_60_7474",
        "ee4627de-c026-4d38-e164-08dc4555b94c_1064_4636",
        "f0472e92-85eb-46f5-dcfe-08dc1694d265_81_4711",
    ]

    # Load manifest for audio paths
    manifest = {}
    with open("/workspace/SAPC2/manifest/Dev_streaming.csv") as f:
        for row in csv.DictReader(f):
            manifest[row["id"]] = row

    fixed_match = 0
    for uid in divergent_uids:
        row = manifest.get(uid)
        if not row:
            print(f"  {uid}: NOT IN MANIFEST")
            continue

        audio_path = os.path.join("/workspace/SAPC2", row["audio_filepath"])
        samples = read_wav(audio_path)

        m.reset()
        m.set_partial_callback(lambda _: None)
        for start in range(0, len(samples), CHUNK_SIZE):
            m.accept_chunk(samples[start:start + CHUNK_SIZE])
        result = m.input_finished()

        ref = ref_preds.get(uid, "")
        match = normalize(result) == normalize(ref)
        if match:
            fixed_match += 1
        status = "MATCH" if match else "STILL DIVERGENT"
        print(f"  {status}: {uid[:50]}")
        if not match:
            print(f"    Ref:  \"{ref}\"")
            print(f"    Ours: \"{result}\"")

    print(f"\nFixed: {fixed_match}/{len(divergent_uids)} previously-divergent utterances now match")

    # Also run full 123-utt batch to count total divergences
    print("\n=== Full 123-utt batch check ===")
    total_match = 0
    total_divergent = 0
    for uid, row in manifest.items():
        audio_path = os.path.join("/workspace/SAPC2", row["audio_filepath"])
        if not os.path.exists(audio_path):
            continue
        samples = read_wav(audio_path)
        m.reset()
        m.set_partial_callback(lambda _: None)
        for start in range(0, len(samples), CHUNK_SIZE):
            m.accept_chunk(samples[start:start + CHUNK_SIZE])
        result = m.input_finished()
        ref = ref_preds.get(uid, "")
        if normalize(result) == normalize(ref):
            total_match += 1
        else:
            total_divergent += 1
            if total_divergent <= 3:
                print(f"  DIVERGENT: {uid[:50]}")
                print(f"    Ref:  \"{ref[:70]}\"")
                print(f"    Ours: \"{result[:70]}\"")

    print(f"\nTotal: {total_match} match, {total_divergent} divergent out of {total_match + total_divergent}")


if __name__ == "__main__":
    main()
