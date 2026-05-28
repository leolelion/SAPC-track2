# Step 2 — Empty-prediction characterization (Dev_10k Nemotron int8-static)

Total: 10521  |  Empty: 1463 (13.91%)  |  Non-empty: 9058

## 1. Audio-duration buckets — empty rate per bucket

| Bucket | N total | Empty | Empty rate |
|---|---:|---:|---:|
| <0.5s | 0 | 0 | 0.00% |
| 0.5-1s | 0 | 0 | 0.00% |
| 1-1.5s | 1 | 0 | 0.00% |
| 1.5-2s | 147 | 48 | 32.65% |
| 2-3s | 923 | 165 | 17.88% |
| 3-5s | 3221 | 463 | 14.37% |
| >5s | 6229 | 787 | 12.63% |

## 2. Speech-active duration (mfa_speech_end − mfa_speech_start), seconds

- Empty: (empty)
- Non-empty: (empty)

## 3. Reference word count

- Empty: n=1463 min=1.00 p10=1.00 p25=3.00 p50=4.00 mean=4.44 p75=6.00 p90=8.00 p99=15.00 max=20.00
- Non-empty: n=9058 min=1.00 p10=3.00 p25=4.00 p50=6.00 mean=9.91 p75=9.00 p90=18.00 p99=84.00 max=228.00

Distribution of empty-set reference word counts (low end):
| Words | Empty count | Empty share |
|---:|---:|---:|
| 1 | 172 | 11.76% |
| 2 | 153 | 10.46% |
| 3 | 301 | 20.57% |
| 4 | 240 | 16.40% |
| 5 | 185 | 12.65% |
| 6 | 170 | 11.62% |
| 7 | 69 | 4.72% |
| 8 | 73 | 4.99% |
| 9 | 40 | 2.73% |
| 10 | 9 | 0.62% |
| 11 | 13 | 0.89% |
| 12 | 7 | 0.48% |
| 13 | 13 | 0.89% |
| 14 | 2 | 0.14% |
| 15 | 7 | 0.48% |

## 4. Per-etiology empty rate

| Etiology | N | Empty | Empty rate |
|---|---:|---:|---:|
| ALS | 2204 | 316 | 14.34% |
| Cerebral Palsy | 2530 | 744 | 29.41% |
| Down Syndrome | 1711 | 218 | 12.74% |
| Parkinson's Disease | 3187 | 55 | 1.73% |
| Stroke | 889 | 130 | 14.62% |

## 5. Top-10 speakers by empty count

| Speaker (prefix) | Etiology | N utts | Empty | Empty rate |
|---|---|---:|---:|---:|
| `24e7fbed-4e8…` | Cerebral Palsy | 109 | 90 | 82.57% |
| `55c1784a-ece…` | ALS | 104 | 90 | 86.54% |
| `6b942f5f-0f1…` | Stroke | 95 | 80 | 84.21% |
| `d41daa38-6d4…` | Cerebral Palsy | 95 | 78 | 82.11% |
| `54618732-a2c…` | Cerebral Palsy | 96 | 75 | 78.12% |
| `5801631a-2f0…` | Cerebral Palsy | 92 | 70 | 76.09% |
| `b2f16a07-844…` | Cerebral Palsy | 99 | 67 | 67.68% |
| `fb9c683c-41a…` | Cerebral Palsy | 89 | 66 | 74.16% |
| `031e84ad-54f…` | ALS | 99 | 63 | 63.64% |
| `4a9f71ab-f3a…` | Down Syndrome | 88 | 51 | 57.95% |

## 6. Encoder-step count for empties (≈ floor(audio_duration / 0.56))

| Steps available | Empty count | Empty share |
|---|---:|---:|
| 2 chunk(s) | 22 | 1.50% |
| 3 chunk(s) | 55 | 3.76% |
| 4 chunk(s) | 91 | 6.22% |
| 5-9 chunks | 620 | 42.38% |
| 10+ chunks | 675 | 46.14% |

Counts (raw): T=0 (<25 ms): **0**, under 1 full chunk (25–560 ms): **0**, ≥1 full chunk: **1463**.
