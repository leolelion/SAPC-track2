# Step 2 — Byte-level zip structure comparison

baseline = `streaming_zipformer.zip` (SHA256 f7876a00…, 7331 B, **works on Codabench**)
v7       = `nemotron_streaming_v7_minimal.zip` (SHA256 868b3732…, 11328 B, **empty ingestion tab**)

## Summary table

| Property | baseline | v7 | Same? |
|---|---|---|---|
| Wrapping directory | none — files flat at root | none — files flat at root | ✅ SAME |
| Central-dir entries | 3 | 4 (extra `README.md`) | ❌ v7 +1 |
| File order (central dir) | model.py, setup.sh, config.yaml | model.py, setup.sh, config.yaml, README.md | ✅ first 3 identical |
| Compression method (all files) | Deflate, normal | Deflate, normal | ✅ SAME |
| Stored/other methods | none | none | ✅ SAME |
| `__MACOSX/` entries | none | none | ✅ SAME |
| `.DS_Store` entries | none | none | ✅ SAME |
| Directory entries (e.g. `weights/`) | none | none | ✅ SAME |
| OS of origin | Unix | Unix | ✅ SAME |
| Version made by | 3.0 | 3.0 | ✅ SAME |
| Min version to extract | 2.0 | 2.0 | ✅ SAME |
| Extended local header (data descriptor) | no | no | ✅ SAME |
| Encryption | none | none | ✅ SAME |
| Extra fields per entry | 0x5455 (UT) + 0x7875 (Unix uid/gid) | 0x5455 (UT) + 0x7875 (Unix uid/gid) | ✅ SAME field IDs |
| Zip comment | none | none | ✅ SAME |
| File comments | none | none | ✅ SAME |
| Integrity (`unzip -t`) | OK | OK | ✅ both valid |

## Differences found (in detail)

### D1 — v7 contains an extra file: `README.md`
- baseline: 3 files (model.py, setup.sh, config.yaml).
- v7: 4 files (above + README.md, 2877 B uncompressed).
- README.md is appended last in the central directory (after config.yaml), so the
  first three entries keep identical ordering.

### D2 — setup.sh Unix permission bits differ
- baseline `setup.sh`: `100644` (`-rw-r--r--`) — **not** executable.
- v7 `setup.sh`: `100755` (`-rwxr-xr-x`) — executable.
- (model.py and config.yaml are `100644` in both.)

### D3 — config.yaml line endings differ
- baseline `config.yaml`: **CRLF** line terminators (`file` reports "with CRLF").
- v7 `config.yaml`: **LF** only.
- NOTE: the CRLF one is the *baseline that works*. So CRLF-vs-LF in config.yaml is
  demonstrably not an ingestion gate. setup.sh and model.py are LF in both.

### D4 — 0x7875 (Unix uid/gid) extra-field payload differs (cosmetic)
- baseline: `01 04 1e 0d 01 00 04 ca 00 00 00` → uid 68894, gid 202 (Linux build user).
- v7:       `01 04 f5 01 00 00 04 14 00 00 00` → uid 501, gid 20 (macOS first user).
- Indicates baseline was zipped on Linux, v7 on macOS. Both Info-ZIP, identical
  field structure. No functional impact on extraction.

### D5 — timestamps differ (cosmetic)
- baseline entries: 2026-02-17/19. v7 entries: 2026-06-09. Expected; no impact.

## Raw evidence

### baseline `unzip -l`
```
    16114  02-17-2026 21:23   model.py
     2924  02-19-2026 05:18   setup.sh
     1712  02-17-2026 21:23   config.yaml
    20750                     3 files
```

### v7 `unzip -l`
```
    25132  06-09-2026 15:57   model.py
     2662  06-09-2026 15:56   setup.sh
      775  06-09-2026 15:55   config.yaml
     2877  06-09-2026 15:58   README.md
    31446                     4 files
```

### Method/compression (`unzip -v`)
Both archives: every entry `Defl:N` (deflate, normal). No `Stored`, no other method.

## Verdict (Step 2)

The two archives are **structurally near-identical**. The only structural deltas are:
an extra README.md (D1), the setup.sh executable bit (D2), and cosmetic
metadata (D3 CRLF-on-baseline, D4 uid/gid, D5 timestamps).

**No wrapping directory. No macOS artifacts. No directory entries. No stored
files. Same version-made-by, same flags, both pass integrity.** None of the
structural deltas is a plausible Codabench pre-ingestion gate — and the one that
looks most "wrong" (CRLF) is on the *working* baseline, not on ours.
