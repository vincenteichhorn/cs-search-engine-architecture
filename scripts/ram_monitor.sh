#!/bin/bash

# Check if at least 2 arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <log_file> <command-to-run>"
    echo "Example: $0 mem_log.csv python -m sea"
    exit 1
fi

# First argument is the log file
LOG_FILE="$1"
shift  # Remove first argument so $@ is the command

# Remaining arguments are the command to run
CMD="$@"

# Write CSV header to log file
echo "Timestamp,%MEM,RSS_KB,VSZ_KB,Command" > "$LOG_FILE"

# Launch the command in the background
$CMD < /dev/tty &
PID=$!

echo "Launched '$CMD' with PID $PID"
echo "Logging memory usage to '$LOG_FILE'"

# Monitor memory usage while the process is running
while kill -0 $PID 2>/dev/null; do
    MEM_LINE=$(ps -p $PID -o %mem,rss,vsz,comm --no-headers | awk '{$1=$1; gsub(/ +/,","); print}')
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$MEM_LINE" >> "$LOG_FILE"
    sleep 1
done

echo "Process $PID finished. Memory log saved to '$LOG_FILE'"
