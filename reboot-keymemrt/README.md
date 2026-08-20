# ReBoot → MLIR for KeyMemRT

A pure C++ reimplementation of the **ReBoot** encrypted-training method
([AAAI-26](https://ojs.aaai.org/index.php/AAAI/article/view/39670),
[arXiv:2506.19693](https://arxiv.org/abs/2506.19693),
[code](https://github.com/AI-Tech-Research-Lab/ReBoot)) that builds the network,
**differentiates the training step itself**, lowers ReBoot's packing to explicit
slot operations, and emits `ckks`-dialect **MLIR** for
[KeyMemRT-Compiler](https://github.com/eymay/KeyMemRT-Compiler) — which then
generates the OpenFHE code and the KeyMemRT key management against
[eymay/openfhe-development](https://github.com/eymay/openfhe-development).

```
                 reboot_emit (this directory)
  ModelConfig ──► tensor graph ──► autograd ──► Nesterov + bootstrap
                                                     │
                                       lower packing to slot ops
                                                     │
                                        ckks-dialect MLIR
                                                     │
                                              keymemrt-opt
                          --ckks-to-lwe --lwe-to-openfhe
                          --kmrt-merge-rotation-keys --kmrt-key-prefetching
                                                     │
                                          keymemrt-translate
                                                     │
                              OpenFHE C++ with keymem_rt.deserializeKey /
                              EvalRotate / keymem_rt.clearKey around every
                              rotation
```

## 1. Why ReBoot cannot simply run on KeyMemRT

ReBoot is a Python library over a pybind11 module (`reboot_cpp`) pinned to
**OpenFHE 1.2.1 + openfhe-python 0.8.9**. Three things stop that stack from
driving KeyMemRT, and they are structural rather than packaging problems:

1. **Two OpenFHE builds, one interpreter.** Every `reboot_cpp` entry point takes
   `CryptoContext<DCRTPoly>`, `KeyPair<DCRTPoly>` and friends by value, so it
   only works if `openfhe-python` registered those exact types. KeyMemRT builds
   against the fork (currently **1.2.3**); both cannot be right at once.
2. **The fork's feature is unreachable from Python.** The fork adds
   `Get/SetDynamicQSize` on evaluation keys, used by
   `keyswitch-hybrid.cpp:452` — that is what lets hybrid key switching consume a
   key truncated to fewer towers, which is the whole basis of
   `RNSKeyCompressor` and `serializeKeysAtLevel()` in KeyMemRT.
3. **ReBoot's rotations are invisible.** Its matrix products call
   `EvalSumRows`/`EvalSumCols`, whose rotations happen inside OpenFHE against
   the **EvalSum** key map. KeyMemRT pages keys out of the **automorphism** key
   map one index at a time; it has no hook there, and the indices are never
   named by the program.

The fix for (3) is also the fix for (1) and (2): name every rotation in the IR
and let the compiler place the key management. That is what this frontend does.

## 2. What it does

### 2.1 Autograd, not a hand-written backward pass

`tensor_graph.h` is a small tensor-level IR — vectors and weight matrices, not
slot vectors — so `autograd.cc` is ordinary reverse-mode differentiation. The
model builder assembles only the *forward* network and one loss seed per local
classifier; the backward pass is derived.

ReBoot's published backward rules fall out of the standard vector-Jacobian
products:

| forward | VJP | what it lowers to |
| --- | --- | --- |
| `matmul(x, W)` | `∂x = matmul_t(g, W)` | the same weight ciphertext, summed the *other* way — no transpose, no repacking |
| `matmul(x, W)` | `∂W = outer(x, g)` | a bare elementwise product: one operand is Repeated and the other Expanded, so the product already *is* the outer product in the weight layout — zero rotations |
| `poly_relu(x)` | `g · (2x + 1)` | one level, as in the paper |

Each block ends in a `stop_gradient`, so its error signal never leaves the
block. `test_autograd` checks all of this against central finite differences,
including that a block's gradient equals the gradient of *its own* loss (4e-12)
and differs from that of the full objective (2.5e-2) — the difference being
exactly what keeps the depth of a step independent of network depth.

The optimiser is emitted from the same graph. Because the velocities are
function arguments, passing zeros on the first step reproduces ReBoot's separate
"initialise the velocity" branch exactly, so there is only one update rule:

```
g' = g + wd·W;   Δ = lr·g' + m·lr·g' + m²·lr·V;   W' = W − Δ;   V' = m·V + g'
```

All three factors are `ckks.mul_scalar`, which costs no level.

### 2.2 Packing lowered to named rotations

`slot_graph.cc` turns each tensor op into the `ckks` ops that exist, expanding
the two summations into rotate-and-add trees:

```
sum_rows  for k = cols, 2·cols, ... < slots:  x += rot(x, k)      (no extra level)
sum_cols  for k = 1, 2, ... < cols:           x += rot(x, k)
          x *= mask(first column of each row)                     (one level, as EvalSumCols)
          for k = 1, 2, ... < cols:           x += rot(x, −k)
```

`test_lowering` runs the tensor graph and the slot graph on the same inputs and
compares every result of the step — updated weights, velocities and predictions
— for three architectures; agreement is ~1e-17.

### 2.3 The emitted module

`mlir_emitter.cc` prints a module the KeyMemRT pipeline consumes: NTT-friendly
primes generated for the requested `logN` and depth, spelled-out
`!lwe.lwe_ciphertext` types, and one `func.func` whose arguments are the
weights, velocities, inputs and encrypted labels and whose results are the
refreshed state plus the predictions.

```mlir
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 12, Q = [...], P = [...], logDefaultScale = 26>} {
  func.func @reboot_train_step(%arg0: !ct {reboot.name = "w_head"}, ... ) -> (!ct, ...) {
    %0 = ckks.mul %arg4, %arg0 : (!ct, !ct) -> !ct_d3
    %1 = ckks.relinearize %0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_d3) -> !ct
    %2 = ckks.rotate %1 {static_shift = 8 : i64} : !ct
    %3 = ckks.add %1, %2 : (!ct, !ct) -> !ct
    ...
    %n = ckks.bootstrap %m : !ct -> !ct      // w_head_next
    return ...
  }
}
```

The `static_shift` attribute is the point of the exercise: `--ckks-to-lwe` and
`--lwe-to-openfhe` rewrite each `ckks.rotate` into `kmrt.load_key` /
`openfhe.rot` / `kmrt.clear_key`, so the rotation keys of the *training* step
come under KeyMemRT's per-key management without anything here knowing about
key files, compression levels or prefetch queues.

Scale handling: every ciphertext is emitted at the top of the chain and each
product is relinearised straight back to the canonical basis, leaving rescaling
to OpenFHE's `FLEXIBLEAUTO` at run time. The depth the graph actually needs is
computed from the graph and sizes the modulus chain.

## 3. Layout

```
include/reboot/
  layout.h         slot geometry: Repeated / Expanded packing, weight packing
  tensor_graph.h   tensor-level IR with shapes and packings
  autograd.h       reverse-mode differentiation
  reboot_model.h   the eMLP, its local losses, the optimiser and bootstrapping
  slot_graph.h     lowering to ckks-level slot operations
  ckks_params.h    ring dimension, NTT prime chain, scheme parameters
  mlir_emitter.h   ckks-dialect MLIR text
  interpreter.h    plaintext evaluators for both levels (the test oracles)
  data.h           synthetic blobs and a CSV loader
lib/*.cc           implementations
drivers/
  reboot_emit.cc   build, differentiate, lower, emit
  reboot_eval.cc   run the emitted step on plaintext slots
tests/
  test_autograd.cc gradients vs finite differences; gradient locality
  test_lowering.cc packed lowering vs tensor semantics
  test_emitter.cc  op counts, rotation attributes, signature
```

## 4. Building and running

Needs only a C++17 compiler and [{fmt}](https://fmt.dev/12.0/) 12:

```shell
cmake -B build -S . && cmake --build build -j
ctest --test-dir build --output-on-failure
```

Emit a training step:

```shell
./build/reboot_emit --hidden 64,32 --input-dim 64 --classes 10 \
                    --batch-size 1 --log-n 13 --stats -o reboot_train_step.mlir
```

Then run it through the compiler (see the KeyMemRT-Compiler README for the full
pipeline):

```shell
keymemrt-opt --ckks-to-lwe --lwe-to-openfhe \
             --annotate-module="backend=openfhe scheme=ckks" \
             --openfhe-configure-crypto-context --kmrt-merge-rotation-keys \
             --bootstrap-rotation-analysis --openfhe-insert-clear-ops \
             --kmrt-key-prefetching="runtime-delegated=1" --lower-affine \
             reboot_train_step.mlir > reboot_train_step.opt.mlir
keymemrt-translate --emit-openfhe-pke reboot_train_step.opt.mlir > reboot_train_step.cpp
```

Check what the step does before compiling it:

```shell
./build/reboot_eval --hidden 16,8 --dim 8 --classes 3 --samples 128 --epochs 3
# epoch  0 | loss   0.5457 | accuracy  84.4%
# epoch  1 | loss   0.1585 | accuracy 100.0%
# epoch  2 | loss   0.0576 | accuracy 100.0%
```

The paper's networks map onto `--hidden` as `eMLP-1: 32`, `eMLP-2: 64,32`,
`eMLP-3: 128,64,32`.

## 5. What has been verified

* **Autograd** — gradients match central finite differences to ~1e-11; a
  block's gradient matches the finite difference of its own loss and not of the
  whole objective; batch gradients accumulate correctly.
* **Lowering** — the slot graph reproduces the tensor semantics to ~1e-17 for
  eMLP-1/2/3, all rotation indices are powers of two inside the slot vector,
  and the column summation is the only source of right rotations.
* **Emitter** — op counts match the slot graph, every `ckks.rotate` carries a
  `static_shift`, the emitted index set equals the graph's, every product is
  relinearised, and all results are returned.
* **End to end in plaintext** — the emitted step trains: 84% → 100% on a
  separable synthetic task in three epochs, executing the exact graph that gets
  printed as MLIR.

Not verified here: the emitted text has not been run through `keymemrt-opt` in
this environment (building it needs Bazel and a full MLIR/LLVM tree). The op
names, attribute names and assembly formats are taken from the fork's own
TableGen definitions — `CKKSOps.td` (`static_shift`, `from_basis`/`to_basis`,
`scalar`), `LWETypes.td` (`lwe_plaintext` takes only `plaintext_space` in this
fork, unlike the Orion translator's output) and `LWEOps.td` (`rlwe_encode`) —
and the structural tests check the emitter against them, but a `keymemrt-opt`
round trip is the obvious next step.

## 6. Style

Google C++ style with tab indentation, snake_case functions, variables and file
names, CamelCase types, and trailing-underscore members; `.clang-format` is in
this directory. All formatting and output goes through {fmt} 12.

## Licence

The ReBoot algorithm being reimplemented is GPL-3.0 (see the
[upstream repository](https://github.com/AI-Tech-Research-Lab/ReBoot)); this
work keeps that licence.
