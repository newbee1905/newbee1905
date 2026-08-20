#!/usr/bin/env bash
# Run the same encrypted ReBoot configuration under each KeyMemRT key mode and
# collect the resource traces, the way KeyMemRT's own benchmarks compare its
# modes against the all-keys-resident baseline.
#
#   scripts/compare_modes.sh [extra reboot_train flags...]
#
# Everything after the first argument is passed through, so the model and CKKS
# parameters are chosen once and applied to every mode.
set -euo pipefail

BIN=${BIN:-./build/reboot_train}
KEYDIR=${KEYDIR:-./keys}
RESULTDIR=${RESULTDIR:-./results}
MODES=${MODES:-"ignore imperative prefetch"}

mkdir -p "$RESULTDIR"

for mode in $MODES; do
  echo "=== key mode: $mode ==="
  rm -rf "$KEYDIR"
  mkdir -p "$KEYDIR"
  SERIALIZED_DATA_DIR="$KEYDIR" "$BIN" \
    --key-mode "$mode" \
    --input-dir "$KEYDIR" \
    --result-dir "$RESULTDIR" \
    --log-level error \
    "$@" | tee "$RESULTDIR/reboot_${mode}.out"
  echo
done

echo "resource traces written to $RESULTDIR"
