#!/usr/bin/env bash
#
# Build and run the rotation-key memory benchmark.
#
#   KEYMEMRT=/path/to/KeyMemRT ./scripts/measure_key_memory.sh --log-n 16 --depth 24
#
# Needs the forked OpenFHE installed (see the README) and a KeyMemRT checkout
# for include/KeyMemRT.hpp.  The frontend library must already be built:
# cmake -B build -S . && cmake --build build -j
set -euo pipefail

: "${KEYMEMRT:?set KEYMEMRT to a KeyMemRT checkout}"
OPENFHE_PREFIX=${OPENFHE_PREFIX:-/usr/local}
OUT=${OUT:-build/measure_key_memory}
KEYS=${KEYS:-$PWD/build/keys_measure}

mkdir -p "$(dirname "$OUT")" "$KEYS"

g++ -std=c++17 -O2 -fopenmp \
  -I include -I "$KEYMEMRT/include" \
  -I "$OPENFHE_PREFIX/include/openfhe" \
  -I "$OPENFHE_PREFIX/include/openfhe/core" \
  -I "$OPENFHE_PREFIX/include/openfhe/pke" \
  -I "$OPENFHE_PREFIX/include/openfhe/binfhe" \
  -I "$OPENFHE_PREFIX/include" \
  tools/measure_key_memory.cc build/libreboot.a \
  -o "$OUT" \
  -L "$OPENFHE_PREFIX/lib" -lOPENFHEpke -lOPENFHEcore -lOPENFHEbinfhe -lfmt

"$OUT" --key-dir "$KEYS" "$@"
