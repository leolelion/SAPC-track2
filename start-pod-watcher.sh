#!/bin/bash

POD_ID="1ppb7l0i5xuna8"
NOTIFY_TOPIC="runpod-free-gpu-alert"   # ← CHANGE THIS (see below)

echo "🚀 Starting watcher for pod $POD_ID (retry every 30s)..."

while true; do
    # Run the exact command you showed
    if output=$(runpodctl pod start "$POD_ID" 2>&1); then
        echo "✅ SUCCESS at $(date)!"
        echo "$output"
        
        # === PING YOU HERE ===
        curl -H "Title: Pod Started! 🎉" \
             -H "Priority: high" \
             -d "Pod $POD_ID is now RUNNING!\nCost: $(echo "$output" | grep -o '\$[0-9.]* / hr')" \
             "https://ntfy.sh/$NOTIFY_TOPIC" >/dev/null 2>&1
        
        break   # Stop the loop once it works
    else
        echo "❌ Failed at $(date) – no free GPUs yet. Retrying in 60s..."
        # Optional: echo the error for logging
        # echo "$output"
    fi
    sleep 30
done

echo "Watcher finished."