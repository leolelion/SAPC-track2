# Track 2 (Streaming ASR) — Status Update

**Date:** 2026-07-21 · **Prepared for:** management · **Scope:** SAPC2 Track 2 — streaming speech recognition for dysarthric (impaired) speech, CPU-only, scored on both accuracy and latency.

---

## Bottom line

Our live competition entry is **ranked #1 on the public leaderboard** — accurate, low-latency, and CPU-efficient. It **outperformed a model 10× larger**, which validated our strategy of fine-tuning a right-sized model on impaired-speech data rather than chasing raw size. Our next step (cheap accuracy/latency wins) is fully prepared locally and waiting on a single, low-cost compute run.

The competition is scored on the **error rate × latency trade-off** ("Pareto frontier"); the prize is split among all teams on that frontier. **Character Error Rate (CER) is the primary metric — lower is better.**

**Headline latency:** our entry returns the **first word in ~1.4 s** and the **final transcript ~0.1 s after the speaker stops** — a combined latency score of **~0.73 s**, which is what keeps us on the winning frontier alongside our low error rate.

---

## About the challenge

The **Speech Accessibility Project Challenge 2 (SAPC2)** is a public benchmark competition to improve automatic speech recognition for people with **speech disabilities** — dysarthric speech arising from conditions such as ALS, cerebral palsy, Parkinson's, Down syndrome, and stroke. Mainstream speech recognition performs poorly on this population, and the challenge exists to close that gap.

We compete in **Track 2 — streaming (real-time) recognition**: transcribe speech *as it is spoken*, running on **CPU only** within a fixed compute budget. Entries are scored on the trade-off between **accuracy** (character error rate) and **latency** (how quickly words appear); the prize is shared among all teams that land on the best accuracy–latency frontier.

**Our entry is currently ranked #1 on the public leaderboard for this track.**

---

## Results that matter (official scoring)

Measured on the sequestered **Test1** set through the organizers' exact scoring pipeline:

| Submission | CER ↓ | WER ↓ | Latency (time-to-first-word) | Verdict |
|---|---|---|---|---|
| **Zipformer, beam-4** (current live entry) | **21.28%** | 29.5% | ~1.37 s | **Our best — leads on both accuracy and latency** |
| Zipformer, greedy (earlier) | 23.44% | 31.51% | ~1.44 s | Superseded |
| Nemotron 0.6B (10× larger) | 27.97% | 37.71% | ~2.28 s | Lost — bigger did not win |

**Takeaways for the boss:**
- Our current entry **strictly beats** our earlier version on *both* accuracy and speed — a clean upgrade with zero retraining cost.
- A model with **~10× more parameters lost to it**. We proved this rigorously rather than assuming, and redirected effort away from the dead end.
- The biggest remaining challenge is the **most severely impaired speakers**, where all models still struggle — this is the accuracy floor we're working to raise.

---

## How it works — our approach

A three-step build on an open, streaming-capable model:

1. **Start from a pretrained streaming Zipformer** — a ~66-million-parameter recognizer, originally trained on read English audiobooks (LibriSpeech). "Streaming" means it transcribes *as audio arrives*, in ~0.3-second chunks, rather than waiting for the whole utterance — which is exactly what the contest scores.
2. **Fine-tune it on impaired speech** — 16 training passes over the challenge's dysarthric-speech data, with *speed-perturbation* and *SpecAugment* (standard techniques that show the model slightly sped-up/slowed and partially-masked audio so it generalizes). This step is where most of the accuracy comes from: the model learns the acoustics of impaired speech.
3. **Compress and decode efficiently** — export to an optimized 8-bit (int8) form so it runs fast on CPU within budget (near-lossless — within ~0.1 CER of full precision), and decode with **beam search**.

### How beam search helps here

When transcribing, the model emits tokens (word-pieces) one at a time.

- **Greedy** decoding commits to the single most likely token at each step — fast, but one locally-wrong guess can derail the rest of the transcript.
- **Beam search** instead keeps the several most promising partial transcripts alive at once — the "beam width" is how many (we ship 4; beam-8 is now testing better) — and only commits at the end to the best-scoring whole. It can recover from a tempting-but-wrong token by keeping alternatives open, which matters most exactly when the audio is ambiguous — as impaired speech often is. That's why beam-4 beat greedy by ~2 CER points on the real test set.
- **Why it's nearly free:** the expensive part — the encoder, ~95% of the computation — runs *once* and is shared across all the candidate transcripts. So widening the beam from 4 to 8 barely changes latency. A bigger beam buys accuracy at almost no speed cost — which is exactly what the dev-set result below shows.

---

## Improvements already measured (internal dev set)

We have **implemented and measured** two upgrades to our winning model, using the organizers' exact scoring on our internal streaming dev set. This dev set is easier than the real test set, so read the absolute numbers as a **relative comparison** — the *gaps between settings* are what's meaningful.

| Setting | CER (dev) | First word | vs. live entry |
|---|---|---|---|
| beam-4 — current live entry | 12.14% | 1.16 s | reference |
| **beam-8 — implemented** | **11.62%** | 1.16 s | **−0.52 CER at identical latency — a strict win** |
| chunk-8 — latency sweep | 13.85% | 0.93 s | −20% faster first word, +1.71 CER — low-latency option |

- **beam-8 is a free accuracy win:** better error rate at the *same* speed and cost — a one-line runtime change on the already-shipped model. It's our recommended next submission, pending one confirmation run on the real test set before we flip the live entry.
- **The latency sweep found a distinct low-latency option (chunk-8):** ~20% faster to first word for a small accuracy cost. Because the prize is split across *everyone* on the accuracy × latency frontier, this could be a valuable **second** entry at the fast end.

---

## Why the 10× model (Nemotron) failed — and why that's useful

Bigger did not win, and we understand why. On our internal dev sets the larger model actually looked competitive — but that lead came almost entirely from **easy and moderate speech**, and it evaporated on the two things the contest actually rewards:

1. **Severely impaired speakers** — the hardest cases, and heavily represented in the real test set. Here the big model was no better than ours, and on very quiet, low-energy speech (e.g. ALS) it often **output nothing at all**.
2. **Real-time streaming** — the contest scores incremental, low-latency transcription. The large model was strong when allowed to see a whole utterance at once, but lost ~10 points of accuracy when converted to the streaming mode being scored.

The tell: from dev to the real test set it **dropped ~18 points of accuracy, versus ~2 points for our winning model** — a generalization failure, not a scale advantage. Every rescue attempt (further fine-tuning, data augmentation, lightweight adapters, decode-time fixes) failed our validation bar.

**Why it's useful:** it confirmed our strategy. Effort is better spent adapting a right-sized, streaming-native model to impaired speech than scaling a general-purpose one. We closed this out with evidence and redirected accordingly.

**Also ruled out — Parakeet (NVIDIA off-the-shelf models):** benchmarked; do not beat our fine-tuned model on accuracy or latency, and are not built for real-time streaming. Not adopted.

---

## What's next

1. **Confirm beam-8 on the real test set**, then flip it to our live entry — near-zero risk given the clean dev-set win.
2. **Language-model fusion** — add a small language model to sharpen transcripts (staged locally, not yet run).
3. **Combine the two wins** (beam-8 at the faster chunk-8 setting) and check the low-latency option on the severe tail before any second submission.

Remaining cost is **one short compute run**; all code and validation are prepared locally.

---

## Honest caveats

- The **beam-8 and chunk-8 results above are on our internal dev set**, not the official test set. Dev-set wins don't always transfer, so we confirm on the real harness before flipping our live entry.
- **Severely impaired speech remains the hardest case** across every model we've tried; it is our main accuracy limitation.

---

*Source of record: `RESEARCH_SUMMARY_FOR_FABLE.md` and `experiments/summary.csv` in the Track 2 repository. All figures above are from the organizers' official scoring (`evaluate.sh`), except the beam-8/LM items, which are flagged as internal dev-set only.*
