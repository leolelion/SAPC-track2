# Scoring source locations

| Component | Path |
|---|---|
| Canonical SAPC2 scoring (upstream HEAD) | `github.com/xiuwenz2/SAPC-template/utils/` |
| Local copy (this fork, slightly older) | `utils/` |
| Codabench runtime image | `xiuwenz2/sapc2-runtime:latest` (contains `utils/` and likely a wrapper `scoring.py`; not extracted) |
| Track2 spec reference | `Track2 (Streaming ASR Track).md` cites `utils/metrics/cer.py`, `utils/metrics/wer.py`, `utils/compute_latency.py` as the canonical implementations |

Upstream cloned to `/tmp/SAPC-template-upstream/` for diff inspection on 2026-05-28. Upstream HEAD: `c48d945 update compute_metrics`.

Codabench `scoring.py` (the bundle entry point) was not extracted from the Docker image (Docker unavailable in this RunPod container). Mention of it in `utils/normalize_hyp.py` and `utils/compute_metrics.py` as `scoring.py (bundle)` confirms it is a thin wrapper that calls into the same `normalize_text` + `compute_from_sgml` library API as our local `evaluate.py`. The evaluation logic lives in `utils/`, not in the wrapper.
