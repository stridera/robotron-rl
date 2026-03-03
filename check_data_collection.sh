#!/bin/bash
# Check data collection progress

echo "=================================="
echo "DATA COLLECTION STATUS"
echo "=================================="
echo ""

# Check if process is running
PID=$(ps aux | grep "collect_detector_data" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ Data collection process not running"
    echo ""
    echo "Check log for errors:"
    echo "  tail -50 data_collection.log"
    exit 1
else
    echo "✅ Process running (PID: $PID)"
fi

# Get last line from log
LAST_LINE=$(tail -1 data_collection.log | grep "Episodes:")
if [ -n "$LAST_LINE" ]; then
    echo "📊 $LAST_LINE"
fi

# Extract episode count
EPISODES=$(echo "$LAST_LINE" | grep -oP '\d+/1000' | head -1)
if [ -n "$EPISODES" ]; then
    COMPLETED=$(echo "$EPISODES" | cut -d'/' -f1)
    PCT=$((COMPLETED * 100 / 1000))
    REMAINING=$((1000 - COMPLETED))

    # Estimate time remaining
    ELAPSED_SEC=$(($(date +%s) - $(stat -c %Y data_collection.log)))
    if [ $COMPLETED -gt 0 ]; then
        SEC_PER_EP=$((ELAPSED_SEC / COMPLETED))
        REMAINING_SEC=$((REMAINING * SEC_PER_EP))
        HOURS=$((REMAINING_SEC / 3600))
        MINS=$(((REMAINING_SEC % 3600) / 60))
        echo "⏱️  Progress: $COMPLETED/1000 episodes ($PCT%)"
        echo "⏳ Estimated time remaining: ${HOURS}h ${MINS}m"
    fi
fi

# Check if file exists and show size
if [ -f "detector_dataset.npz" ]; then
    SIZE=$(ls -lh detector_dataset.npz | awk '{print $5}')
    echo "💾 Dataset file: $SIZE"
else
    echo "💾 Dataset file: Not created yet (created after all episodes complete)"
fi

echo ""
echo "=================================="
echo "COMMANDS"
echo "=================================="
echo "Monitor live: tail -f data_collection.log"
echo "Kill process: kill $PID"
echo "Check this status: bash check_data_collection.sh"
