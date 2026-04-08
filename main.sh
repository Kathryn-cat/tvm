#!/bin/bash
# Run printer validation on all test suite splits.
# Usage: bash main.sh
# Expects testsuite/ splits in ../tvm-ffi/testsuite/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUITE_DIR="$SCRIPT_DIR/../tvm-ffi/testsuite"
LOGFILE="$SCRIPT_DIR/validation_results.log"

# Check suite dir exists
if [ ! -d "$SUITE_DIR" ]; then
    echo "ERROR: Suite dir not found: $SUITE_DIR"
    echo "Trying fallback..."
    SUITE_DIR="/home/scratch.kathrync_sw/work/tvm-ffi/testsuite"
fi

files=("$SUITE_DIR"/suite_*.jsonl)
if [ ! -f "${files[0]}" ]; then
    echo "ERROR: No suite files found in $SUITE_DIR"
    exit 1
fi

echo "=== Printer Validation ===" | tee "$LOGFILE"
echo "Suite dir: $SUITE_DIR" | tee -a "$LOGFILE"
echo "Files: ${#files[@]}" | tee -a "$LOGFILE"
echo "Start: $(date)" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

total_v1=0
total_crash=0
total_mismatch=0
total_examples=0

for f in "${files[@]}"; do
    name=$(basename "$f")
    result=$(python "$SCRIPT_DIR/main.py" "$f" 2>&1 | grep "^V1 OK")
    echo "$name: $result" | tee -a "$LOGFILE"

    # Parse counts
    v1=$(echo "$result" | sed 's/.*V1 OK: \([0-9]*\).*/\1/')
    cr=$(echo "$result" | sed 's/.*V2 crashes: \([0-9]*\).*/\1/')
    mm=$(echo "$result" | sed 's/.*mismatch: \([0-9]*\).*/\1/')
    tot=$(echo "$result" | sed 's/.*Total: \([0-9]*\).*/\1/')

    total_v1=$((total_v1 + v1))
    total_crash=$((total_crash + cr))
    total_mismatch=$((total_mismatch + mm))
    total_examples=$((total_examples + tot))
done

echo "" | tee -a "$LOGFILE"
echo "=== TOTALS ===" | tee -a "$LOGFILE"
echo "V1 OK:        $total_v1" | tee -a "$LOGFILE"
echo "V2 crashes:   $total_crash" | tee -a "$LOGFILE"
echo "V2 mismatch:  $total_mismatch" | tee -a "$LOGFILE"
echo "Total:        $total_examples" | tee -a "$LOGFILE"
echo "End: $(date)" | tee -a "$LOGFILE"
