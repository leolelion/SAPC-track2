# Step 4 — Line-by-line setup.sh comparison

baseline `setup.sh` (2924 B, mode 644, LF, trailing `\n`)
v7       `setup.sh` (2662 B, mode 755, LF, trailing `\n`)

## Literal text-level properties

| Property | baseline | v7 | Same? |
|---|---|---|---|
| Shebang | `#!/bin/bash` | `#!/bin/bash` | ✅ |
| `set -e` | yes (line 12) | yes (line 16) | ✅ |
| `set -u` | no | no | ✅ |
| `set -o pipefail` | no | no | ✅ |
| `DIR="$(cd "$(dirname "$0")" && pwd)"` | yes | yes | ✅ identical idiom |
| Final line | `echo "=== setup.sh complete ==="` | `echo "=== setup.sh complete ==="` | ✅ |
| Explicit `exit 0` | **no** | **no** | ✅ (relies on last cmd rc=0) |
| Trailing newline | yes (`0x0a`) | yes (`0x0a`) | ✅ |
| Line endings | LF | LF | ✅ |
| `file` classification | Bourne-Again shell script, UTF-8 | Bourne-Again shell script, UTF-8 | ✅ |
| Stdout/stderr redirection | none | none | ✅ |
| Stage banner echos | "Stage 1/2/3" | "Stage 1/2/3" | ✅ same pattern |

The *shape* of setup.sh is essentially identical. Both: bash, `set -e`, resolve
`$DIR`, three labeled stages, no explicit exit. These are the properties that
would matter to a pre-ingestion / shell-launch validator, and they match.

## Functional differences (these run DURING ingestion, not before it)

| Aspect | baseline | v7 |
|---|---|---|
| Installer | `pip install …` (plain) | `pip install --no-cache-dir --prefer-binary -c <constraint>` |
| Flags | `-q`, `--no-deps` (for wheels) | `-q`, `--no-cache-dir`, `--prefer-binary` |
| torch pinning | none | writes `/tmp/torch_pin_$$.txt` = `torch==<sys>`, passes via `-c` |
| Packages | numpy sentencepiece tqdm huggingface_hub graphviz lhotse omegaconf; k2 + kaldifeat wheels (`--no-deps`); icefall via `git clone … && pip install -e .` | omegaconf huggingface_hub onnxruntime onnx |
| Git clone | yes (icefall) | no |
| Direct wheel URLs (HF) | yes (k2, kaldifeat) | no |
| Weights download | `hf_hub_download` epoch-30.pt + bpe.model | `hf_hub_download` 4 ONNX files + tokens.txt (pinned revision) |
| `--user` / `--break-system-packages` | no | no |
| venv creation | no | no |
| Pre-flight import check | no | no |

### Notes on the functional deltas
- Both install into the **base** Python (neither creates a venv), matching each
  other. The spec *recommends* a venv but the working baseline does not use one —
  so "no venv" is not the gate.
- v7's installs are strictly lighter (4 pure/binary packages vs a heavy
  k2/kaldifeat/icefall stack). If anything v7's setup is *less* likely to fail.
- **Crucially:** every functional delta here executes inside `setup.sh`, which runs
  *during* ingestion. With v6/v7 ingestion tabs reported **empty** (setup.sh banner
  never appeared), these deltas cannot be what gated v6/v7 before ingestion — the
  script never ran. They remain candidates only if the empty-tab reading is wrong.

## Full text — baseline
```bash
#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
# Stage 1: detect TORCH_VER / PY_TAG via python3
# Stage 2: pip install numpy sentencepiece tqdm huggingface_hub graphviz lhotse omegaconf
#          pip install --no-deps <k2 wheel>; <kaldifeat wheel>
#          git clone --depth 1 icefall; pip install -e .
# Stage 3: hf_hub_download epoch-30.pt + data/lang_bpe_500/bpe.model
echo "=== setup.sh complete ==="
```

## Full text — v7
```bash
#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
# Stage 1: detect TORCH_VER / PY_TAG via python3
# Stage 2: SYS_TORCH pin -> /tmp/torch_pin_$$.txt; pip install -q --no-cache-dir
#          --prefer-binary -c <pin> omegaconf huggingface_hub onnxruntime onnx
# Stage 3: hf_hub_download 4 ONNX files + tokens.txt (repo rev pinned)
echo "=== setup.sh complete ==="
```

## Verdict (Step 4)
The text-level / launch-relevant shape of setup.sh is **identical** to baseline
(shebang, `set -e`, `$DIR`, final echo, no exit, LF, trailing newline). All deltas
are in package *contents*, which run during ingestion — irrelevant to an empty
ingestion tab. No pre-ingestion gate found in setup.sh.
