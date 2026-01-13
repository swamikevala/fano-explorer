#!/bin/bash
# Restart the pool server
cd "$(dirname "$0")/.."
FANO_ROOT=$(pwd)

# Find and kill existing pool process
PID=$(netstat -ano 2>/dev/null | grep ":9000.*LISTEN" | awk '{print $NF}')
if [ -n "$PID" ]; then
    echo "Stopping pool (PID: $PID)..."
    cmd //c "taskkill /PID $PID /F" 2>/dev/null || kill $PID 2>/dev/null
    sleep 2
fi

# Start pool
echo "Starting pool..."
cd "$FANO_ROOT"
python pool/run_pool.py > pool_output.log 2>&1 &
sleep 5

# Verify
if curl -s http://127.0.0.1:9000/health | grep -q "ok"; then
    echo "Pool started successfully"
    curl -s http://127.0.0.1:9000/health
else
    echo "Pool may have failed to start. Check pool_output.log"
fi
