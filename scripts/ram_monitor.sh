#!/bin/bash

# --- Argument check ---------------------------------------------------------
if [ $# -lt 2 ]; then
    echo "Usage: $0 <log_file> <command-to-run>"
    echo "Example: $0 mem_log.csv python -m sea"
    exit 1
fi

LOG_FILE="$1"
shift
CMD=("$@")   # Keep command as array (safer for args with spaces)

# --- Write CSV header -------------------------------------------------------
echo "Timestamp,%MEM,RSS_KB,VSZ_KB,Command" > "$LOG_FILE"

# --- Launch command in background ------------------------------------------
"${CMD[@]}" &
PID=$!

echo "Launched '${CMD[*]}' with PID $PID"
echo "Logging memory usage to '$LOG_FILE'"

# --- Handle Ctrl-C ---------------------------------------------------------
cleanup() {
    echo ""
    echo "Ctrl-C pressed — killing PID $PID …"
    kill -INT "$PID" 2>/dev/null
    wait "$PID" 2>/dev/null
    echo "Exiting."
    exit 130
}

trap cleanup INT

# --- Monitoring loop --------------------------------------------------------
while kill -0 "$PID" 2>/dev/null; do
    MEM_LINE=$(ps -p "$PID" -o %mem,rss,vsz,comm --no-headers | awk '{$1=$1; gsub(/ +/,","); print}')
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$MEM_LINE" >> "$LOG_FILE"
    sleep 1
done

wait "$PID" 2>/dev/null

echo "Process $PID finished. Memory log saved to '$LOG_FILE'"
