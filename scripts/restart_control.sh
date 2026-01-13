#!/bin/bash
# Restart the control server
cd "$(dirname "$0")/.."
FANO_ROOT=$(pwd)

# Find and kill existing control server
for PORT in 8765 8080 5000; do
    PID=$(netstat -ano 2>/dev/null | grep ":$PORT.*LISTEN" | awk '{print $NF}')
    if [ -n "$PID" ]; then
        echo "Stopping control server on port $PORT (PID: $PID)..."
        cmd //c "taskkill /PID $PID /F" 2>/dev/null || kill $PID 2>/dev/null
        sleep 1
    fi
done

sleep 2

# Start control server
echo "Starting control server..."
cd "$FANO_ROOT"
python -m control.server > control/restart.log 2>&1 &
sleep 3

# Verify
for PORT in 8765 8080 5000; do
    if curl -s "http://127.0.0.1:$PORT/" | grep -q "Fano"; then
        echo "Control server started on port $PORT"
        exit 0
    fi
done

echo "Control server may have failed to start. Check control/restart.log"
