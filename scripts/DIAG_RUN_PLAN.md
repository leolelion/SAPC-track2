# Full-picture Nemotron diagnostic — run plan (exp A–C + length×severity)

Goal: convert the "well-supported story" into a verdict. Decide whether the empty/near-fail tail is
**acoustic-domain** (→ finetuning) or **quantization / decode artifact** (→ partially fixable now), and
map failure vs **duration** and **speaker-severity**. All CPU; pod is GPU-reserved but we only use CPU.

Prereqs already staged in `scripts/`: `build_diag_manifest.py`, `diag_crosstab.py`,
`nemotron_blank_probe.py`, `setup_zf_sub.sh`, `nemotron_repro_run.sh`, `nemotron_repro_analyze.py`.

## 0. Connect (port changes per restart)
    runpodctl pod list   # get 38.80.152.249:<PORT> for tcp/22
    SSH="ssh -i /Users/o/.runpod/ssh/RunPod-Key-Go -p <PORT> -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    SCP="scp -i /Users/o/.runpod/ssh/RunPod-Key-Go -P <PORT> -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    $SCP scripts/{build_diag_manifest.py,diag_crosstab.py,nemotron_blank_probe.py,setup_zf_sub.sh} root@38.80.152.249:/workspace/

## 1. Build stratified manifest (length bins × worst speakers)
    $SSH 'python3 /workspace/build_diag_manifest.py /workspace/SAPC2/manifest/Dev.csv /workspace/SAPC2/manifest/Dev_diag.csv'
    # ~450 utts: all utts (cap 25) of 7 worst speakers + 50/duration-bin random. Prints composition.

## 2. Decode NEMOTRON (int8) on the diag set  [exp length×severity, and the failure pool]
    $SSH 'cd /workspace && nohup bash nemotron_repro_run.sh Dev_diag Dev_10 > /workspace/diag_nemo.log 2>&1 &'
    # output: /dev/shm/Dev_diag_repro.csv

## 3. (A) Decode finetuned ZIPFORMER on the SAME utts — does it rescue Nemo's failures?
    $SCP /tmp/a1_pkg/model.py root@.../workspace/finetune/zf_a1_sub/model.py     # A1 greedy model.py
    $SCP <tokens.txt from submission_a1_int8.zip> root@.../workspace/finetune/zf_a1_sub/tokens.txt
    $SSH 'bash /workspace/setup_zf_sub.sh'
    $SSH 'cd /workspace && nohup bash nemotron_repro_run.sh ... '   # NOTE: repro_run is nemo-specific;
    # for ZF use local_decode.py directly:
    $SSH 'cd /workspace/finetune/zf_a1_sub && DR=$(...) ; python3 /workspace/sapc-nemotron/track2_starting_kit/local_decode.py \
          --submission-dir /workspace/finetune/zf_a1_sub --manifest-csv /workspace/SAPC2/manifest/Dev_diag.csv \
          --streaming-manifest-csv /workspace/SAPC2/manifest/Dev_10.csv --data-root $DR \
          --out-csv /dev/shm/Dev_diag_zf.csv --out-partial-json /dev/shm/_zf.json'

## 4. Cross-tab both models (length × etiology + head-to-head rescue test)
    $SSH 'cd /workspace/sapc-nemotron/utils && python3 /workspace/diag_crosstab.py \
          /workspace/SAPC2/manifest/Dev_diag.csv nemo=/dev/shm/Dev_diag_repro.csv zf=/dev/shm/Dev_diag_zf.csv'
    # KEY: "ZF good on Nemo failures" => DOMAIN (finetune fixes). "ZF also fails" => audio genuinely hard.

## 5. (B) Blank-probability probe on ~4 Nemo empties (mix short + long)
    $SSH 'cd /workspace/finetune/nemo_submission && python3 /workspace/nemotron_blank_probe.py \
          /workspace/SAPC2/manifest/Dev_diag.csv $DR <emptyID_short> <emptyID_long> <goodID_control>'
    # blank>>nonblank stable margin = confident give-up (domain). erratic/NaN/saturated = artifact.

## 6. (C) fp32 vs int8 Nemotron on the failing utts (rule out quantization collapse)
    # check for fp32 encoder; danielbodart has an fp32 variant (pod has network):
    $SSH 'ls -la /workspace/finetune/nemo_submission/weights/*.onnx*'
    # if only int8: download fp32 encoder_model.onnx(.data) to weights_fp32/, point a copy of the sub at it,
    # re-decode the empties; if empties disappear under fp32 => quantization collapse (fixable w/o finetune).

## 7. STOP THE POD
    runpodctl pod stop 3dwiczo41jeg1y

## Interpretation matrix
- ZF rescues Nemo failures + blank-margin large + fp32==int8  => DOMAIN. Finetune Nemotron.
- fp32 fixes empties                                          => QUANTIZATION. Re-export better int8 / use fp32.
- blank margins erratic / decode-path specific               => CODE/DECODE artifact. Patch the loop.
- ZF ALSO fails on same utts                                 => audio genuinely unintelligible; neither helps.
- failure strongly rises with duration                       => streaming/O(N^2)/cache issue on long utts.
