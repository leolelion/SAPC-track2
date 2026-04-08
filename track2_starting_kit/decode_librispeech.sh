#!/bin/bash
# Decode dev100 with the librispeech streaming_pruned_stateless7 variant
set -e
cd "$(dirname "$0")"
mkdir -p local_results/streaming_pruned_stateless7_librispeech
python3 local_decode.py \
    --submission-dir ./streaming_pruned_stateless7 \
    --manifest-csv /Users/o/Development/SAPC-template/dev100_bundle/Dev_100_local.csv \
    --data-root /Users/o/Development/SAPC-template/dev100_bundle \
    --out-csv ./local_results/streaming_pruned_stateless7_librispeech/Dev.predict.csv \
    --out-partial-json ./local_results/streaming_pruned_stateless7_librispeech/Dev.partial.json
