#!/bin/zsh
#
# train.sh - Wrapper for train.py that handles logging and checkpoint directories
#
# Usage: ./train.sh [all train.py arguments]
#
# This script:
# - Creates a timestamped checkpoint directory (pt_<input_basename>_YYYY_MM_DD_HHMM/)
# - Creates a timestamped log file in terminal_logs/
# - Runs train.py in background with output redirected to the log file
# - Runs tail -f on the log file so you can observe progress
#
# Ctrl+C will stop tail but leave training running in background.

# Extract input filename from arguments to create meaningful names
INPUT_FILE=""
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--input" ]]; then
        INPUT_FILE="$arg"
        break
    fi
    prev_arg="$arg"
done

if [[ -z "$INPUT_FILE" ]]; then
    echo "Error: --input argument is required"
    echo "Usage: $0 --input <file> [other train.py arguments]"
    exit 1
fi

# Extract basename without extension
INPUT_BASENAME=$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')

# Generate timestamp (YYYY_MM_DD_HHMM)
TIMESTAMP=$(date +"%Y_%m_%d_%H%M")

# Create directory names
PT_DIR="pt"
LOG_DIR="terminal_logs"
LOG_FILE="${LOG_DIR}/terminal_log_for_${INPUT_BASENAME}_${TIMESTAMP}.txt"

# Create directories
mkdir -p "$PT_DIR"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Training wrapper started"
echo "Input file: $INPUT_FILE"
echo "Checkpoint directory: $PT_DIR"
echo "Log file: $LOG_FILE"
echo "========================================"
echo ""

# Log the command line first
echo "Command: python -u py/train.py --checkpoints_to \"$PT_DIR\" $@" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Run train.py with output appended to log file
python -u py/train.py --checkpoints_to "$PT_DIR" "$@" >> "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

# Kill training process if user presses Ctrl+C
trap "echo ''; echo 'Stopping training (PID $TRAIN_PID)...'; kill $TRAIN_PID 2>/dev/null; exit" INT TERM

echo "Training started with PID $TRAIN_PID"
echo "Output is being logged to: $LOG_FILE"
echo ""
echo "Press Ctrl+C to stop both tail AND training"
echo ""

# Wait for log file to be created
while [[ ! -f "$LOG_FILE" ]]; do
    sleep 0.1
done

# Follow the log file
tail -f "$LOG_FILE"
