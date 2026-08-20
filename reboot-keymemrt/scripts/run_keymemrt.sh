#!/usr/bin/env bash
#
# Emit -> compile -> run, end to end.
#
#   1. reboot_emit           ckks-dialect MLIR for the training step
#   2. keymemrt-opt          places the KeyMemRT key management
#   3. keymemrt-translate    OpenFHE C++
#   4. g++                   links that with drivers/reboot_runner.cc
#   5. run                   trains under the requested key mode
#
# Required:
#   KEYMEMRT_COMPILER  a KeyMemRT-Compiler checkout with the tools built
#                      (bazel build //tools:keymemrt-opt //tools:keymemrt-translate)
#   KEYMEMRT           a KeyMemRT checkout (for include/KeyMemRT.hpp and friends)
#
# The model flags are passed to reboot_emit and reboot_runner alike: the runner
# rebuilds the same graph to learn the argument order and the packing, so the
# two must agree.
set -euo pipefail

: "${KEYMEMRT_COMPILER:?set KEYMEMRT_COMPILER to a KeyMemRT-Compiler checkout}"
: "${KEYMEMRT:?set KEYMEMRT to a KeyMemRT checkout}"

MODEL_FLAGS=${MODEL_FLAGS:-"--hidden 32,16 --input-dim 16 --classes 4 --batch-size 1 --log-n 13"}
KEY_MODE=${KEY_MODE:-imperative}
STEPS=${STEPS:-4}
OUT=${OUT:-build/generated}
KEYS=${KEYS:-$PWD/keys}
RESULTS=${RESULTS:-$PWD/results}

OPT=${OPT:-$KEYMEMRT_COMPILER/bazel-bin/tools/keymemrt-opt}
TRANSLATE=${TRANSLATE:-$KEYMEMRT_COMPILER/bazel-bin/tools/keymemrt-translate}

mkdir -p "$OUT" "$KEYS" "$RESULTS"

echo "==> 1/5 emitting MLIR"
./build/reboot_emit $MODEL_FLAGS --stats -o "$OUT/reboot_train_step.mlir"

echo "==> 2/5 keymemrt-opt"
"$OPT" \
  --ckks-to-lwe \
  --lwe-to-openfhe \
  --annotate-module="backend=openfhe scheme=ckks" \
  --openfhe-configure-crypto-context \
  --kmrt-merge-rotation-keys \
  --bootstrap-rotation-analysis \
  --cse \
  --openfhe-insert-clear-ops \
  "$OUT/reboot_train_step.mlir" > "$OUT/reboot_train_step.opt.mlir"

# Balanced mode: let the runtime prefetch ahead of the rotations.
if [ "${PREFETCH:-0}" = "1" ]; then
  "$OPT" --kmrt-key-prefetching="runtime-delegated=1" --lower-affine \
    "$OUT/reboot_train_step.opt.mlir" > "$OUT/reboot_train_step.prefetch.mlir"
  mv "$OUT/reboot_train_step.prefetch.mlir" "$OUT/reboot_train_step.opt.mlir"
fi

echo "==> 3/5 keymemrt-translate"
"$TRANSLATE" --emit-openfhe-pke "$OUT/reboot_train_step.opt.mlir" \
  > "$OUT/reboot_train_step.cc"

echo "==> 4/5 building the host runner"
g++ -std=c++17 -O2 -fopenmp \
  -I include -I "$KEYMEMRT/include" \
  -I /usr/local/include/openfhe -I /usr/local/include/openfhe/core \
  -I /usr/local/include/openfhe/pke -I /usr/local/include/openfhe/binfhe \
  -I /usr/local/include -I /usr \
  drivers/reboot_runner.cc "$OUT/reboot_train_step.cc" \
  build/libreboot.a \
  -o "$OUT/reboot_runner" \
  -L /usr/local/lib -lOPENFHEpke -lOPENFHEcore -lOPENFHEbinfhe -lfmt

echo "==> 5/5 running (key mode: $KEY_MODE)"
SERIALIZED_DATA_DIR="$KEYS" "$OUT/reboot_runner" \
  --key-mode "$KEY_MODE" --input-dir "$KEYS" --result-dir "$RESULTS" \
  --log-level error --steps "$STEPS" $MODEL_FLAGS
