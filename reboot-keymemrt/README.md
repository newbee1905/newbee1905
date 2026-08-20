# ReBoot on KeyMemRT

A rewrite of the **ReBoot** encrypted-training method
([AAAI-26](https://ojs.aaai.org/index.php/AAAI/article/view/39670),
[arXiv:2506.19693](https://arxiv.org/abs/2506.19693),
[code](https://github.com/AI-Tech-Research-Lab/ReBoot)) so that it runs on the
**KeyMemRT** runtime ([KeyMemRT](https://github.com/eymay/KeyMemRT),
[KeyMemRT-Compiler](https://github.com/eymay/KeyMemRT-Compiler)) and its forked
OpenFHE ([eymay/openfhe-development](https://github.com/eymay/openfhe-development)).

ReBoot is the first framework for fully encrypted, non-interactive training of
MLPs under CKKS: local-loss blocks keep the multiplicative depth of a training
step independent of network depth, a two-format packing scheme keeps every
tensor in one ciphertext, and the weights are bootstrapped after every step.
KeyMemRT attacks the other half of the FHE cost model — rotation keys, which
dominate memory — by paging each key in from disk for the rotation that needs
it, at the compression level that rotation needs, and dropping it afterwards.

Putting the two together is not a packaging exercise: the stacks conflict, and
the conflict is structural rather than incidental. This directory contains the
port, the reasoning, and the tests that check it.

---

## 1. Why the original stack cannot be reused

### 1.1 Two different OpenFHE builds, one process

ReBoot is a Python library (`lib/`) over a pybind11 extension (`cpp/`,
module `reboot_cpp`). Its README pins **Python 3.10.12, OpenFHE 1.2.1,
openfhe-python 0.8.9**. Every entry point in `cpp/bindings.cpp` takes OpenFHE
objects by value:

```cpp
m.def("encrypt_array", [](vector<vector<double>> array,
                          const CryptoContext<DCRTPoly>& cc,
                          const KeyPair<DCRTPoly>& key_pair, int level) { ... });
```

Those arguments only convert because the *same* `CryptoContext<DCRTPoly>` type
is registered in the interpreter by `openfhe-python`. Two extension modules
therefore have to agree on the OpenFHE headers, the C++ ABI and the pybind11
type registry. KeyMemRT builds against `eymay/openfhe-development` (currently
**1.2.3**), not upstream 1.2.1, so `reboot_cpp` and `openfhe-python` cannot both
be right at once.

### 1.2 The fork's feature is exactly the one Python cannot reach

The fork is not a cosmetic branch. It adds a dynamic tower count to evaluation
keys:

```
src/pke/include/key/evalkey.h      virtual void   SetDynamicQSize(size_t)
src/pke/include/key/evalkeyrelin.h size_t GetDynamicQSize() const
src/pke/lib/keyswitch/keyswitch-hybrid.cpp:452   size_t sizeQ = evalKey->GetDynamicQSize();
```

That field is what lets hybrid key switching use a key that has been *truncated
to fewer towers* — the trick behind `RNSKeyCompressor` in
`KeyMemRT/include/KeyCompression.hpp` and behind `serializeKeysAtLevel()` /
`RestoreDynamicQSize()` in `KeyMemRT.hpp`. Upstream OpenFHE 1.2.1, which
openfhe-python 0.8.9 wraps, has no such field: a compressed key deserialized
into it would be rejected or silently wrong. Rebuilding openfhe-python against
the fork is possible in principle, but it buys nothing, because of §1.3.

### 1.3 ReBoot's rotations are invisible to KeyMemRT

ReBoot's two matrix products are built on `EvalSumRows` and `EvalSumCols`:

```cpp
// ReBoot cpp/lib/packing.cpp
out[i] = X[i] * W;
out[i] = out[i].sumRows(row_size);   // cc->EvalSumRows(ct, rowSize, sumRowsKeys)
```

Those helpers perform their rotations *inside OpenFHE*, against the **EvalSum
key map** created by `EvalSumRowsKeyGen` / `EvalSumColsKeyGen`. KeyMemRT manages
the **automorphism key map** (`GetEvalAutomorphismKeyMap(keyTag)`), one rotation
index at a time — `deserializeKey(index, depth)` … `EvalRotate` …
`clearKey(index)`. It has no hook into the EvalSum path, and the individual
indices are never named by the program.

So even a perfectly rebuilt Python stack would hand KeyMemRT nothing to manage:
the entire rotation key set stays resident for the whole run, which is precisely
the behaviour KeyMemRT exists to remove.

### 1.4 A Python-driven runtime measures the wrong thing anyway

KeyMemRT is a stateful C++ runtime: it mutates the crypto context's key map,
runs a background deserialisation thread with a tower budget
(`--prefetch-sat`), and `ResourceMonitor` reports process RSS. Driving it from
Python breaks all three. `clearKey()` erases the map entry, but any
`Ciphertext`/`EvalKey` still referenced by a Python object keeps its
`shared_ptr` alive, so the memory is not returned; the GIL serialises the
prefetch overlap that PREFETCH mode is built around; and the resident set now
includes the interpreter, NumPy and every intermediate the Python layer holds.

**Conclusion.** The port has to be native C++, and the packing primitives have
to name their rotations explicitly. That is what this directory is.

---

## 2. What the port does

The training method is unchanged — same architecture, same local error signals,
same packing formats, same Nesterov update, same bootstrapping schedule. What
changed is everything about *how the rotations happen* and *where the code
lives*.

| ReBoot (Python + `reboot_cpp`) | This port | Why |
| --- | --- | --- |
| `lib/cryptocontext.py` | `include/reboot/CkksContext.hpp` | context, parameters and client-side key provisioning in C++ |
| `cpp/lib/packing.cpp` (`EvalSumRows`/`EvalSumCols`) | `include/reboot/LinAlg.hpp` | explicit rotate-and-add trees, one named index per rotation |
| `cpp/lib/encrypted_value.cpp` | `include/reboot/CkksBackend.hpp` | ciphertext ops, each rotation bracketed by KeyMemRT load/clear |
| `lib/layers/linear.py`, `activations.py` | `include/reboot/Layers.hpp` | packed linear layer, PolyReLU, square |
| `lib/optim/optimizers.py` | `include/reboot/Optim.hpp` | encrypted Nesterov SGD, cosine schedule |
| `lib/blocks/*`, `lib/models/local_loss_models.py` | `include/reboot/Model.hpp` | local-loss blocks and the eMLP builder |
| `lib/utils/train.py`, `experiments/3_training_encrypted/` | `include/reboot/Trainer.hpp`, `drivers/reboot_train.cpp` | training loop and benchmark driver in the KeyMemRT driver style |
| — | `include/reboot/KeyPlan.hpp` | rotation-key plan: the dynamic stand-in for the compiler's static `kmrt.load_key` analysis |
| `lib/models/models.py` (plain vs encrypted paths) | `include/reboot/Backend.hpp` | one backend-generic implementation, run either encrypted or on plain slots |

### 2.1 Rotations, spelled out

`EvalSumRows` and `EvalSumCols` are replaced by rotate-and-add trees over
explicit indices, with the same depth as the OpenFHE originals:

```
sumRows(x)  for k = cols, 2*cols, 4*cols ... < slots:  x += rot(x, k)
            -> column sums replicated down every row (Repeated), no extra level

sumCols(x)  for k = 1, 2, 4 ... < cols:                x += rot(x, k)
            x *= mask(first column of each row)        <- the one level EvalSumCols also spends
            for k = 1, 2, 4 ... < cols:                x += rot(x, -k)
            -> row sums replicated across every row (Expanded)
```

Every one of those rotations goes through the backend, and the CKKS backend
wraps it exactly the way KeyMemRT-Compiler emits `openfhe.rot`:

```cpp
const RotKey rk = runtime_->deserializeKey(index, keyDepthFor(index, lvl));
Ct out = cc_->EvalRotate(a, index);
runtime_->clearKey(rk);
```

The weight gradient needs no rotations at all: one operand is Repeated and the
other Expanded, so their elementwise product is already the outer product in the
layer's own weight layout.

### 2.2 A key plan instead of a compiler pass

KeyMemRT-Compiler decides statically which key to page in and at which
compression level (`kmrt.load_key %idx {key_depth = L}`). A training loop with
persistent weight ciphertexts and per-step bootstrapping is not expressible in
that IR today, so the port recovers the same information dynamically:

1. **Plan** — a plaintext pass over the same code (the `PlainBackend` tracks
   levels exactly like CKKS) reports the rotation indices and the depth of the
   level schedule. No crypto, milliseconds.
2. **Calibrate** — one client-side CKKS run with all keys resident records the
   real `(rotation index, ciphertext level)` pairs, for the cold first step and
   for the steady state after the first weight bootstrap.
3. **Provision** — for each level, the keys are regenerated, compressed with
   `serializeKeysAtLevel()` and written one file each; then everything is
   dropped from the context. Level 0 is always provisioned as an uncompressed
   fallback, so correctness never depends on the plan being complete.
4. **Train** — the server runs with `--key-mode imperative|prefetch|speculative`;
   in PREFETCH mode the recorded trace drives `enqueueKey()` a configurable
   number of rotations ahead of the cursor.

### 2.3 Bootstrapping keys

`EvalBootstrap` drives its rotations inside OpenFHE, so its keys cannot be paged
one at a time. They are staged as a single bundle around the bootstrap phase of
each step (`BootstrapKeyBundle::load()` / `clear()`), which still keeps them out
of memory for the rest of the step — where ReBoot spends most of its time. If
the bootstrap rotation indices are wanted under per-key management, the
`--bootstrap-rotation-analysis` pass of KeyMemRT-Compiler already computes them
statically and they can be fed to `keymem_rt.addRotIndices()`.

### 2.4 Two fixes relative to the reference implementation

* **Labels.** ReBoot's blocks re-encrypt the *plaintext* labels during the
  backward pass (`compute_local_loss` calls `repeat_and_encrypt(y_onehot)`), so
  the "server" is holding cleartext labels mid-training. Here the client
  encrypts the one-hot labels once per batch in both packings and the server
  uses only ciphertexts, which is what non-interactive training is supposed to
  mean.
* **Layout.** `get_recommended_parameters` sizes `row_size`/`col_size` from a
  heuristic that assigns some layers' output widths to the wrong dimension
  (conservative in the paper's configurations, wrong in general).
  `LocalLossMLP::recommendLayout` derives the constraint per layer from its own
  packing — row-packed needs `in <= rows, out <= cols`, column-packed the
  reverse — then inflates the rows to fill the ciphertext.

---

## 3. Layout of this directory

```
include/reboot/
  Layout.hpp        slot geometry, Repeated/Expanded packing, weight packing
  Backend.hpp       backend concept + plaintext slot backend with CKKS level accounting
  CkksBackend.hpp   OpenFHE backend; every rotation goes through KeyMemRT
  CkksContext.hpp   CKKS parameters, key provisioning, bootstrap key bundle
  KeyPlan.hpp       rotation-key plan, persistence, prefetch cursor
  LinAlg.hpp        sumRows / sumCols / RE-Matmul / CE-Matmul / outer product
  Layers.hpp        packed linear layer, PolyReLU, square, RSS gradient
  Optim.hpp         encrypted Nesterov SGD, cosine LR
  Model.hpp         local-loss blocks, eMLP builder, the training step
  Trainer.hpp       batching, evaluation, training loop
  Data.hpp          synthetic blobs and a CSV loader
drivers/
  reboot_plain.cpp  plaintext driver and key planner (no OpenFHE needed)
  reboot_train.cpp  encrypted driver on KeyMemRT
tests/
  test_linalg.cpp   packing algebra vs. direct matrix math
  test_training.cpp packed training step vs. an independent unpacked reference
```

---

## 4. Building

### Plaintext driver and tests — no dependencies

```shell
cmake -B build -S . && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/reboot_plain --hidden 32,16 --epochs 3 --lr 0.01
```

### Encrypted driver

Needs the forked OpenFHE and a KeyMemRT checkout:

```shell
git clone https://github.com/eymay/openfhe-development && cd openfhe-development
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
cmake --build build -j && sudo cmake --install build && sudo ldconfig

git clone https://github.com/eymay/KeyMemRT
cmake -B build -S . -DKEYMEMRT_DIR=/path/to/KeyMemRT
cmake --build build -j
```

## 5. Running

```shell
mkdir -p keys results

# imperative: load one key per rotation, clear it straight after
./build/reboot_train --key-mode imperative --input-dir ./keys --result-dir ./results \
    --hidden 32,16 --dim 16 --classes 4 --samples 64 --batch-size 2 --epochs 1 \
    --ring-dim 65536 --level-budget 3,3

# prefetch: same, with the recorded plan driving background loads
./build/reboot_train --key-mode prefetch --prefetch-sat 250 --lookahead 8 \
    --input-dir ./keys --reuse-plan ...

# ignore: baseline with every key resident, for comparison
./build/reboot_train --key-mode ignore ...
```

The paper's networks map onto `--hidden` as
`eMLP-1: --hidden 32`, `eMLP-2: --hidden 64,32`, `eMLP-3: --hidden 128,64,32`;
the packing alternates row/column down the network exactly as in the paper, and
`reboot_plain` prints which layer got which.

`--help` lists everything; the KeyMemRT flags (`--key-mode`, `--input-dir`,
`--prefetch-sat`, `--log-level`, `--log-file`) are parsed by the runtime itself,
so they behave exactly as in the KeyMemRT benchmarks. `ResourceMonitor` writes a
memory/time trace to `--result-dir`.

Start from `reboot_plain` to size things: it prints the architecture, the slot
layout, the rotation indices, the level schedule and the required compute depth
in under a second.

---

## 6. What has been verified

* `test_linalg` — the rotate-and-add rewrite reproduces the matrix algebra
  exactly, in both packings, forwards and backwards, including batched weight
  gradients; every index it uses is declared by `rotationIndices()`; the depth
  of RE-Matmul (1) and CE-Matmul (2) matches the paper.
* `test_training` — five packed training steps agree with an independent
  unpacked reference implementation of the same local-loss algorithm to 1e-9 on
  all four layers, and training converges on a separable synthetic task.
* End to end under CKKS, on the forked OpenFHE 1.2.3 with
  `--key-mode imperative`: keys are provisioned per level, paged in per
  rotation and cleared, and the run finishes with nothing resident
  (`KeyMemRT Stats: Keys loaded: 0`).

Not yet done: a like-for-like memory and latency comparison against ReBoot's own
numbers at the paper's parameters (N = 2^16/2^17, eMLP-1/2/3, MNIST) — that
needs a machine with enough RAM to hold the `ignore`-mode baseline, which is the
whole point of the comparison.

## 7. Possible next steps

* Feed the block's forward and backward passes through KeyMemRT-Compiler as
  `ckks` dialect functions so `--kmrt-merge-rotation-keys` and
  `--kmrt-key-prefetching` schedule the keys statically, and keep only the
  optimiser and the bootstrap staging in the driver.
* Register the bootstrap rotation indices (from `--bootstrap-rotation-analysis`)
  with `keymem_rt` so the bundle becomes per-key managed too.
* Baby-step/giant-step the `sumCols` replication tree to trade rotations for
  depth on wide layouts.

## Licence

The ReBoot algorithm and the structure being ported are GPL-3.0 (see the
[upstream repository](https://github.com/AI-Tech-Research-Lab/ReBoot)); this
port keeps that licence.
