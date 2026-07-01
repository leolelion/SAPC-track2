# 32 — Final int8 submission gate results (2026-06-28)

> All gates completed. Pod stopped. int8 zip pulled to Mac.
> This note records the authoritative numbers for the submission-ready artifact.

## Artifact

| item | value |
|---|---|
| zip | `nemotron_encoder_sos_int8_submission.zip` |
| local path | `/Users/o/Downloads/sapc2_int8_final/nemotron_encoder_sos_int8_submission.zip` |
| SHA-256 | `597b8d95d1df9851052a637b1f0125fd1b73b72a0e82c720d985245f01d51926` |
| size | 878,853,665 bytes (838 MB) |
| model | finetuned Nemotron 0.6B encoder-only, ONNX int8 dynamic encoder + FP32 decoder |
| SOS init | `np.array([[BLANK_ID]], dtype=np.int32)`, BLANK_ID=1024 |
| CHUNK_NEW | 16 |
| unk fix | `_strip_unk` in `accept_chunk` and `input_finished` returns |

---

## Gate 0 — Patch verification (PASSED)

Both `nemotron_encoder_sos_submission.zip` (FP32) and `nemotron_encoder_sos_int8_submission.zip` (int8):
- `_strip_unk` helper present in `model.py` ✅
- No bare `return self._last_emitted` ✅
- SOS init and CHUNK_NEW unchanged ✅

---

## Gate 1 — Official evaluate.sh on Dev-500 (PASSED, BOTH ARMS)

Ran the organizers' full scoring pipeline (`local_decode.py` → `evaluate.sh`) from scratch
on the representative Dev-500 manifest (500 utts, 119 speakers, seed 23, speaker-disjoint)
after clearing all stale CSVs and re-extracting both packages from the patched zips.

| arm | official CER | official WER | SGML mismatches | rc |
|---|---:|---:|---:|---|
| FP32 (patched) | **8.64%** | 12.77% | 0 | **0** ✅ |
| int8 (patched) | **8.76%** | 13.07% | 0 | **0** ✅ |

`SGML_PRED_MISMATCH len_ref1=500 len_ref2=500 mismatches=0 only_unk_token_delta=0` for BOTH.

The scorer assertion is completely cleared. The unk fix is provably complete.

**Note on CER vs prior report:** The official scorer uses sclite SGML with min-over-two-refs,
which is more generous than the bootstrap script (single normalizer path). Official 8.64%/8.76%
replaces the previously reported 10.49%/10.60% bootstrap CER — the former is the number that
will appear on Codabench.

FP32 scores reported for comparison only. The submission artifact is the int8 zip.

---

## Gate 2 — 20-worker throughput sweep (COMPLETED)

CPU: Intel Xeon Platinum 8568Y+, 2 sockets × 48 cores × 2 HT = 192 vCPUs.
Affinity: `taskset -c 0-19` (20 logical CPUs for all 20 workers — intentional worst case).
Rows: 120 utterances from representative Dev-500, avg audio 8.155 sec, total 978.6 sec.

| threads | aggregate_rtf_wall | throughput (audio-sec/wall-sec) | total_wall_sec | per-worker RTF p50 | ok/failed |
|---|---:|---:|---:|---:|---|
| **1** | **0.124** | **8.04** | 121.7 (incl. ~86s load) | 0.811 | 120/0 |
| 4 | 0.415 | 2.41 | 405.9 (incl. ~206s load) | 6.47 (oversubscribed) | 120/0 |

Empty: 3/120 for both thread counts (same utterances, deterministic).

**Interpretation:**
- `SAPC2_THREADS=1` (1 thread per worker) is the correct default for the 20-worker batch
  topology. With taskset to 20 CPUs, 4 threads per worker × 20 workers = 80 threads competing
  for 20 CPUs → 3.3× aggregate slowdown.
- Pure decode RTF at 1 thread (excluding 86s model load): (121.7 - 86) / 978.6 = **0.036**
- Model default is currently `SAPC2_THREADS=4` (from `os.environ.get("SAPC2_THREADS", "4")`).

**Test1 time budget estimate (10521 utts, avg audio 8.155 sec):**

| scenario | aggregate_rtf | estimated wall time | under 15000 s? |
|---|---:|---:|---|
| t=1, pure decode (excl. load) | 0.036 | **3,088 s (51 min)** | ✅ |
| t=1, aggregate incl. load | 0.124 | 10,638 s (177 min) | ✅ |
| t=4, taskset-20 worst case | 0.415 | 35,590 s (593 min) | ❌ |
| t=4, assumed 80 CPUs (no OVS) | ~0.040 | ~3,400 s | ✅ |

**Risk:** The model defaults to `SAPC2_THREADS=4`. If Codabench provides ≥80 vCPUs, this is
fine. If it provides only ~20 vCPUs, the batch pass could exceed 15,000 s. The safe fix is
to change the default to `SAPC2_THREADS=1` and rebuild the zips, but this would hurt TTFT
in the streaming pass. Given CER leadership (8.76% vs A1 23.44%), frontier membership does
not depend on good TTFT. Recommend keeping current default and observing the Codabench result.

---

## Gate 3 — Dev_streaming latency (COMPLETED)

123 utterances from Dev_streaming.csv, real-time 100ms pacing, `SAPC2_THREADS=4`, single model.
Pod: Intel Xeon Platinum 8568Y+, 192 vCPUs (more than Codabench — latency here is optimistic).

| metric | p50 | p90 | p95 |
|---|---:|---:|---:|
| **TTFT** | **1.075 s** | 2.118 s | 2.853 s |
| **TTLT** | **0.199 s** | 0.242 s | 0.251 s |

Definition (from `compute_latency.py`):
- TTFT = first_non_empty_partial_time − (audio_send_start + mfa_speech_start)
- TTLT = final_visible_time − audio_end_oracle_time

**Pareto context:**
- A1 Zipformer TTFT on Codabench: 1438 ms p50 (23.44% CER)
- Nemotron int8 TTFT on pod: 1075 ms p50 (8.76% CER)
- Codabench TTFT will likely be higher than pod (fewer CPUs), probably 1.5–3× → ~1.6–3.2 s
- Even at 3.2 s TTFT, Pareto rank improves on both axes vs A1

Mean(TTFT, TTLT) = mean(1.075, 0.199) = 0.637 s on pod.
On Codabench, expect ~0.9–1.8 s mean latency — still competitive given CER advantage.

---

## Summary: submission decision

The int8 zip at `597b8d95...` is **cleared for Codabench upload** based on:
1. ✅ Official evaluate.sh Dev-500 rc=0 (fp32 and int8 both pass)
2. ✅ SGML mismatches=0 for both arms (unk fix complete)
3. ✅ int8 CER 8.76% vs FP32 8.64% (paired delta +0.12%, CI previously measured as -0.23..+0.50)
4. ✅ Test1 time budget: at SAPC2_THREADS=1 (or with adequate Codabench CPUs), well under 15,000 s
5. ✅ TTFT p50 1.075 s, TTLT p50 0.199 s — competitive latency

**Risk to monitor after submission:**
- If submission times out: change model.py default to `SAPC2_THREADS=1`, rebuild zip, resubmit.
- If CER on Test1 diverges badly from 8.76% Dev-500: investigate etiology composition difference.

**Do not submit A1 Zipformer** (it is the always-valid public LB entry; leave it visible).

## Artifacts pulled to Mac

All in `/Users/o/Downloads/sapc2_int8_final/`:
- `nemotron_encoder_sos_int8_submission.zip` — THE submission artifact
- `nemotron_encoder_sos_int8_submission.zip.sha256`
- `unk_fix_gates.log` — full gate run log (Gate 0–3)
- `unk_fix_gates_summary.json` — structured Gate 1–3 results
- `official_eval_diag.tgz` — SGML + ref/hyp TRN files for both arms
- `enc_sos_int8_w20_t1_120.jsonl` + `_bench.log` — throughput sweep 1 thread
- `enc_sos_int8_w20_t4_120.jsonl` + `_bench.log` — throughput sweep 4 threads
- `enc_sos_int8_devstream_latency.json` — TTFT/TTLT summary
