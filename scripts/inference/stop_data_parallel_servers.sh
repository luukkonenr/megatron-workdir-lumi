#!/bin/bash
# Stop all data parallel inference servers
# Usage: bash stop_data_parallel_servers.sh

PIDS_FILE="/tmp/inference_servers.pids"

if [ ! -f "$PIDS_FILE" ]; then
    echo "No PID file found at $PIDS_FILE"
    echo "Servers may not be running or were started differently."
    exit 1
fi

echo "Stopping all inference servers..."
echo ""

while read PID; do
    if ps -p $PID > /dev/null 2>&1; then
        echo "Killing server with PID: $PID"
        kill $PID
    else
        echo "PID $PID not found (already stopped)"
    fi
done < "$PIDS_FILE"

# Clean up PID file
rm -f "$PIDS_FILE"

echo ""
echo "✅ All servers stopped!"
