# Baseline Reference — known-working streaming_zipformer.zip

Audit date: 2026-06-16
Purpose: pin the exact verified-working reference the organizer re-submitted
and confirmed runs on Codabench (per their 2026-06-09 reply).

## Reference artifact

| Field | Value |
|---|---|
| Source repo | https://github.com/xiuwenz2/SAPC-template (remote `upstream`) |
| Path | `track2_starting_kit/streaming_zipformer.zip` |
| Commit SHA (last touching this file) | `ce02f68023572e264bdf72c74173ea1dccc8e39b` |
| Commit timestamp | 2026-02-19 11:14:14 -0600 |
| Commit author | Xiuwen Zheng |
| Commit subject | `modify submission templates` |
| SHA256 (zip) | `f7876a006d9ba9730c6c38236b1556b005b8abfb293cf450cb72ea6c7f99aa45` |
| File size | 7331 bytes |

## Freshness check

- `git fetch upstream` run at audit time.
- `git log upstream/main --since="2026-06-02" -- track2_starting_kit/` → **no commits**.
- Last 3 commits touching the baseline:
  - `ce02f68` 2026-02-19 — modify submission templates
  - `95adacc` 2026-02-17 — add track2 starting kit
  - `44d9b68` 2026-02-17 — add track2 starting kit
- Conclusion: the organizer's "re-submitted the template today" was a **re-upload
  of the existing artifact** to Codabench, not a new commit. The byte-exact
  working reference is `ce02f68`'s `streaming_zipformer.zip` (hash above).

## Local copy verification

- Working-copy `track2_starting_kit/streaming_zipformer.zip` is **byte-identical**
  to `upstream/main` (`cmp` → IDENTICAL, same SHA256). Safe to diff against the
  local copy.

## Our latest submission under test

| Field | Value |
|---|---|
| Path | `track2_starting_kit/nemotron_streaming_v7_minimal.zip` |
| SHA256 | `868b37328b31519fab83229168807eb87acc904bb99805da539e98adb7714e37` |
| File size | 11328 bytes |
| Built | 2026-06-09 15:58 local |
