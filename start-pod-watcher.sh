#!/bin/bash

POD_IDS=("1ppb7l0i5xuna8" "3dwiczo41jeg1y")
NOTIFY_TOPIC="runpod-free-gpu-alert"   # ← CHANGE THIS (see below)

echo "🚀 Starting watcher for pods ${POD_IDS[*]} (retry every 30s)..."

while true; do
    for POD_ID in "${POD_IDS[@]}"; do
        # Run the exact command you showed
        if output=$(runpodctl pod start "$POD_ID" 2>&1); then
            echo "✅ SUCCESS at $(date) for pod $POD_ID!"
            echo "$output"

            # === PING YOU HERE ===
            curl -H "Title: Pod Started! 🎉" \
                 -H "Priority: high" \
                 -d "Pod $POD_ID is now RUNNING!\nCost: $(echo "$output" | grep -o '\$[0-9.]* / hr')" \
                 "https://ntfy.sh/$NOTIFY_TOPIC" >/dev/null 2>&1

            break 2   # Stop both loops once either works
        else
            echo "❌ Failed at $(date) – pod $POD_ID no free GPUs yet."
            # Optional: echo the error for logging
            # echo "$output"
        fi
    done
    sleep 30
done

echo "Watcher finished."
