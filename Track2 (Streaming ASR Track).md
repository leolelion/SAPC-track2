

Participants submit streaming ASR models that process audio chunk-by-chunk in real time, aiming to advance the state of the art in dysarthric speech recognition with low-latency systems.

---

## Resources & Data

|   |   |
|---|---|
|**Challenge website**|[https://xiuwenz2.github.io/SAPC2-website/](https://xiuwenz2.github.io/SAPC2-website/)|
|**Team registration**|[Registration form](https://docs.google.com/forms/d/e/1FAIpQLSff_gBsA_mhIuSV70V4d8n9X_mRxEGGvzKAsfzKeKw63p1VKA/viewform?pli=1)|
|**Starting kit**|[https://github.com/xiuwenz2/SAPC-template](https://github.com/xiuwenz2/SAPC-template)|
|**Data access**|Request the SAP corpus via the [official website](https://speechaccessibilityproject.beckman.illinois.edu/conduct-research-through-the-project).|

Note: 1. Only Codabench accounts registered with the primary email address listed in your team’s registration form will be automatically admitted to the challenge.

2. [test1.tsv and test2.tsv of SAPC](https://github.com/speechaccessibility/SplitTrainTest/tree/main/SAPC1) are now included in the training split of SAPC2. If you would like to benchmark with them, you can locate them there accordingly.

3. Local decoding script is now available ([link](https://github.com/xiuwenz2/SAPC-template/tree/main/track2_starting_kit#local-dev-test-with-dev-set)). In addition, a Dev_streaming subset for latency measurement is shared through Box, and the corresponding code has been added to [preprocess.sh](https://github.com/xiuwenz2/SAPC-template/blob/aac08c079ed610e3f83bc2d8d72dc42c94d81d33/preprocess.sh#L94).

---

## Submission

### Interface

Your submission must implement a **streaming** `Model` class with 5 methods:

|Method|Called|Purpose|
|---|---|---|
|`__init__()`|Once at startup|Load model weights, tokenizer, etc.|
|`set_partial_callback(fn)`|Once per evaluation pass|Register a callback that receives partial transcription text|
|`reset()`|Once per audio file|Reset internal streaming state|
|`accept_chunk(chunk)`|Many times per file|Feed 100ms audio chunk (1600 samples at 16kHz), return partial result|
|`input_finished()`|Once per file|Signal end of audio, return final transcription|

### How Ingestion Works

The ingestion program evaluates each audio file in **two passes**:

1. **Pass 1 — Batch (accuracy)**: multiprocess (parallel workers), no delay, feeds all chunks as fast as possible. Within each worker, chunks are processed sequentially, and `set_partial_callback` is set to a no-op.
2. **Pass 2 — Streaming (latency)**: two-thread real-time simulation. The **Audio Sender thread** sends 100ms chunks at real-time pace; the **Decoder thread** calls your model's `accept_chunk()` / `input_finished()` and collects partial results via the callback.

All model methods are called **from the Decoder thread only** — your model does **not** need to handle thread safety.

### Environment & Scoring

Competitors will submit trained model parameters and inference code through **Codabench**, subject to a maximum number of permitted submissions.

- Docker image: `xiuwenz2/sapc2-runtime:latest`
- Pre-installed: **PyTorch 2.5.0+cu124**, torchaudio, torchvision
- GPU: Track 2 is evaluated exclusively on CPUs
    
- Time limit: 15000 seconds per submission
- If a `setup.sh` is provided, it runs **before** your model is loaded. Use it to install system packages and Python dependencies.
- If a `requirements.txt` is provided, dependencies are auto-installed via `pip install -r requirements.txt` after `setup.sh`.

Results on **test1 (10521 samples)** will be released within **three days** of submission. Results on **test2 (8304 samples)** will be released **after the close of the competition**.

We allow participants to add or remove submissions from the public leaderboard. However, teams must ensure that **at least one valid submission** remains publicly visible on the leaderboard at all times. We reserve the right to review leaderboard visibility periodically.

---

## Evaluation Metrics

### Accuracy Metrics

Accuracy transcripts are normalized with a fully formatted normalizer adapted from the HuggingFace ASR leaderboard. Implementation details are available in: [`cer.py`](https://github.com/xiuwenz2/SAPC-template/blob/main/utils/metrics/cer.py) and [`wer.py`](https://github.com/xiuwenz2/SAPC-template/blob/main/utils/metrics/wer.py).

- **Character Error Rate (CER)**  
    _Primary metric_, chosen for its better correlation with human judgments and its sensitivity to pronunciation variations in dysarthric speech.
    
- **Word Error Rate (WER)**  
    _Secondary metric_, reported for comparison with prior work and related literature.
    

CER/WER are clipped to **100%** at the utterance level. Accuracy scores are computed using two references (with and without disfluencies), and the lower error is selected per utterance.

### Latency Metrics

Latency is computed from streaming partial results on the streaming manifest (`*_streaming.csv`) and reported as median (**P50**, in ms). We report: Reference implementation: [`compute_latency.py`](https://github.com/xiuwenz2/SAPC-template/blob/main/utils/compute_latency.py).

- **Time To First Token (TTFT, P50, ms)**  
    Per-utterance TTFT is defined as: `first_non_empty_partial_time - (audio_send_start_time + mfa_speech_start)`. Here, `mfa_speech_start` is read from the manifest column with the same name (seconds, relative to the start of the waveform). `first_non_empty_partial_time` is the timestamp of the first partial result whose text is non-empty after stripping whitespace. If all partial texts are empty, the timestamp of the last partial event is used as fallback.
    
- **Time To Last Token (TTLT, P50, ms)**  
    Per-utterance TTLT is defined as: `final_visible_time - audio_end_oracle_time`, where `audio_end_oracle_time = audio_send_start_time + audio_duration_sec`. `final_visible_time` is the timestamp when the final transcript becomes visible in ingestion output.
    

For robustness analysis, P90 latency may also be reported in detailed outputs. For Pareto comparison, final latency is defined as the average of TTFT and TTLT.

---

## Prizes & Publication

A total prize of **USD $10,000** will be divided equally among all teams with a system on the Pareto frontier of accuracy and latency, as measured on the sequestered **test2** set.

To clarify how winners are selected across tracks:

- **Track 1 (Unconstrained ASR):** submissions are non-streaming systems and are ranked by recognition accuracy. For Pareto comparison, Track 1 latency is set to **inf**. Exactly **one non-streaming ASR** system will win.
- **Track 2 (Streaming ASR):** submissions are ranked by the competition's accuracy-latency criteria, and **one or more** streaming systems may win.

Teams submitting to the competition will be invited to present their work at a competition workshop, scheduled in conjunction with a major conference (TBA).

---

## Acknowledgements

The Speech Accessibility Project is funded by a grant from the AI Accessibility Coalition. Computational resources for the challenge are provided by the National Center for Supercomputing Applications (NCSA). We would also like to thank Rob Kooper (NCSA), Wei Kang (Xiaomi Corp.), and Maisy Wieman (SoundHound AI) for their expertise and invaluable assistance in setting up the challenge.