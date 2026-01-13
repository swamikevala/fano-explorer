#!/bin/bash
# Service management for Fano components
# Usage: svc.sh [start|stop|restart|status] [pool|control|explorer|all]

cd "$(dirname "$0")/.."
FANO_ROOT=$(pwd)

get_pid_on_port() {
    netstat -ano 2>/dev/null | grep ":$1.*LISTEN" | awk '{print $NF}' | head -1
}

kill_pid() {
    if [ -n "$1" ]; then
        cmd //c "taskkill /PID $1 /F" 2>/dev/null || kill $1 2>/dev/null
    fi
}

start_pool() {
    PID=$(get_pid_on_port 9000)
    if [ -n "$PID" ]; then
        echo "Pool already running (PID: $PID)"
        return
    fi
    echo "Starting pool..."
    cd "$FANO_ROOT"
    python pool/run_pool.py > pool_output.log 2>&1 &
    sleep 5
    if curl -s http://127.0.0.1:9000/health | grep -q "ok"; then
        echo "Pool started"
    else
        echo "Pool may have failed - check pool_output.log"
    fi
}

stop_pool() {
    PID=$(get_pid_on_port 9000)
    if [ -n "$PID" ]; then
        echo "Stopping pool (PID: $PID)..."
        kill_pid $PID
        sleep 2
        echo "Pool stopped"
    else
        echo "Pool not running"
    fi
}

start_control() {
    for PORT in 8080 8765 5000; do
        PID=$(get_pid_on_port $PORT)
        if [ -n "$PID" ]; then
            echo "Control already running on port $PORT (PID: $PID)"
            return
        fi
    done
    echo "Starting control..."
    cd "$FANO_ROOT"
    python -m control.server > control/restart.log 2>&1 &
    sleep 3
    for PORT in 8080 8765 5000; do
        if curl -s "http://127.0.0.1:$PORT/" 2>/dev/null | grep -q "Fano"; then
            echo "Control started on port $PORT"
            return
        fi
    done
    echo "Control may have failed - check control/restart.log"
}

stop_control() {
    for PORT in 8080 8765 5000; do
        PID=$(get_pid_on_port $PORT)
        if [ -n "$PID" ]; then
            echo "Stopping control on port $PORT (PID: $PID)..."
            kill_pid $PID
        fi
    done
    sleep 1
    echo "Control stopped"
}

status() {
    echo "=== Service Status ==="

    # Pool
    PID=$(get_pid_on_port 9000)
    if [ -n "$PID" ]; then
        echo "Pool: RUNNING (PID: $PID)"
        curl -s http://127.0.0.1:9000/health 2>/dev/null | head -1
    else
        echo "Pool: STOPPED"
    fi

    # Control
    for PORT in 8080 8765 5000; do
        PID=$(get_pid_on_port $PORT)
        if [ -n "$PID" ]; then
            echo "Control: RUNNING on port $PORT (PID: $PID)"
            break
        fi
    done
    [ -z "$PID" ] && echo "Control: STOPPED"
}

ACTION=$1
SERVICE=$2

case "$ACTION" in
    start)
        case "$SERVICE" in
            pool) start_pool ;;
            control) start_control ;;
            all) start_pool; start_control ;;
            *) echo "Usage: svc.sh start [pool|control|all]" ;;
        esac
        ;;
    stop)
        case "$SERVICE" in
            pool) stop_pool ;;
            control) stop_control ;;
            all) stop_pool; stop_control ;;
            *) echo "Usage: svc.sh stop [pool|control|all]" ;;
        esac
        ;;
    restart)
        case "$SERVICE" in
            pool) stop_pool; sleep 1; start_pool ;;
            control) stop_control; sleep 1; start_control ;;
            all) stop_pool; stop_control; sleep 1; start_pool; start_control ;;
            *) echo "Usage: svc.sh restart [pool|control|all]" ;;
        esac
        ;;
    status)
        status
        ;;
    *)
        echo "Fano Service Manager"
        echo "Usage: svc.sh [start|stop|restart|status] [pool|control|all]"
        echo ""
        echo "Examples:"
        echo "  svc.sh status           - Show status of all services"
        echo "  svc.sh restart pool     - Restart the pool server"
        echo "  svc.sh restart all      - Restart all services"
        ;;
esac
