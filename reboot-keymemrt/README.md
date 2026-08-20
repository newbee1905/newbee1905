# ReBoot → MLIR for KeyMemRT

A pure C++ reimplementation of the **ReBoot** encrypted-training method
([AAAI-26](https://ojs.aaai.org/index.php/AAAI/article/view/39670),
[arXiv:2506.19693](https://arxiv.org/abs/2506.19693),
[code](https://github.com/AI-Tech-Research-Lab/ReBoot)) that builds the network,
**differentiates the training step itself**, lowers ReBoot's packing to explicit
slot operations, and emits `ckks`-dialect **MLIR** for
[KeyMemRT-Compiler](https://github.com/eymay/KeyMemRT-Compiler). The compiler
generates the OpenFHE code and the KeyMemRT key management against
[eymay/openfhe-development](https://github.com/eymay/openfhe-development).

```
                 reboot_emit (this directory)
  model_config_t ─► tensor graph ─► autograd ─► Nesterov + bootstrap
                                                     │
                                       lower packing to slot ops
                                                     │
                                        ckks-dialect MLIR  (+ manifest)
                                                     │
                                              keymemrt-opt
                          --ckks-to-lwe --lwe-to-openfhe
                          --kmrt-merge-rotation-keys --kmrt-key-prefetching
                                                     │
                                          keymemrt-translate
                                                     │
                              OpenFHE C++ with keymem_rt.deserializeKey /
                              EvalRotate / keymem_rt.clearKey around every
                              rotation, linked with drivers/reboot_runner.cc
```

At N = 2¹⁶ the rotation-key working set drops from **2871 MB to 99 MB**.
Method and caveats in [§6](#6-rotation-key-memory-measured).

---

## Contents

1. [What you need, and why](#1-what-you-need-and-why)
2. [Setup with Nix](#2-setup-with-nix)
3. [Setup on Ubuntu or Debian](#3-setup-on-ubuntu-or-debian)
4. [Build and test the frontend](#4-build-and-test-the-frontend)
5. [The full pipeline: emit → compile → run](#5-the-full-pipeline-emit--compile--run)
6. [Rotation-key memory, measured](#6-rotation-key-memory-measured)
7. [Troubleshooting](#7-troubleshooting)
8. [How it works](#8-how-it-works)
9. [Why ReBoot cannot simply run on KeyMemRT](#9-why-reboot-cannot-simply-run-on-keymemrt)
10. [Repository layout](#10-repository-layout)
11. [Style](#11-style)

---

## 1. What you need, and why

The frontend and the runtime have very different dependency needs. **You only
need the first row to emit MLIR**; everything else is for compiling and running
what it emits.

| Component | Version | Needed for | Where from |
| --- | --- | --- | --- |
| C++17 compiler, CMake ≥ 3.16 | GCC 13 / Clang 18 tested | everything | distro |
| [{fmt}](https://fmt.dev/12.0/) | **12.0.0** | all formatting and output | source (see below) |
| [eymay/openfhe-development](https://github.com/eymay/openfhe-development) | 1.2.3 fork | running generated code, key-memory benchmark | source |
| [KeyMemRT](https://github.com/eymay/KeyMemRT) | header-only | `KeyMemRT.hpp`, `ResourceMonitor.hpp` | source |
| [KeyMemRT-Compiler](https://github.com/eymay/KeyMemRT-Compiler) | Bazel 8.1.0 | `keymemrt-opt`, `keymemrt-translate` | source |
| cereal | any | generated code includes it | distro (`libcereal-dev`) |

Two versions are fixed, and substituting either one breaks the build:

- **{fmt} 12.** `CMakeLists.txt` asks for `find_package(fmt 12 REQUIRED)`.
  Debian and Ubuntu ship 9 or 10, so §3.2 builds it from source.
- **The eymay fork of OpenFHE.** The fork adds `Get/SetDynamicQSize` to
  evaluation keys, read by `keyswitch-hybrid.cpp:452`. Hybrid key switching
  needs that field to accept a key truncated to fewer towers, and KeyMemRT's
  `RNSKeyCompressor` and `serializeKeysAtLevel()` are built on it. Upstream
  OpenFHE lacks the field, so compressed keys fail against it.

Revisions this was developed and measured against:

```
openfhe-development  6b8bc1162b65e61bf7b2c76fe7c088aad0026231   (reports 1.2.3)
KeyMemRT             04020dfa6d03af53bc6d571043ae12519799c6e5
KeyMemRT-Compiler    2632a234b412db40e15e58f894e772074903a024   (.bazelversion 8.1.0)
fmt                  12.0.0
```

---

## 2. Setup with Nix

`flake.nix` pins the fork, KeyMemRT and {fmt} 12 as flake inputs, so there are
no sha256 hashes to maintain and `flake.lock` records exactly what you built
against.

```shell
# Everything: frontend deps, OpenFHE fork, KeyMemRT headers, bazelisk.
nix develop

# Inside the shell, $KEYMEMRT and $OPENFHE_PREFIX are set for you:
cmake -B build -S . -DKEYMEMRT_DIR=$KEYMEMRT
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Build the frontend as a package, tests and all:

```shell
nix build          # result/bin/reboot_emit, result/bin/reboot_eval
nix run . -- --help
```

Other outputs:

```shell
nix build .#openfhe   # the fork, if you want it on its own
nix build .#fmt12
```

Without flakes enabled, `shell.nix` evaluates the same definitions through
flake-compat:

```shell
nix-shell
```

There is an `.envrc` for direnv users (`direnv allow`).

**Two things Nix does not do here:**

- **`keymemrt-opt` and `keymemrt-translate` are not packaged.** They are Bazel
  targets over a full MLIR/LLVM tree, and Bazel fetches dependencies from the
  network, which a Nix build sandbox forbids. `bazelisk` ships in the dev shell,
  so build them yourself: [§3.5](#35-keymemrt-compiler) works unchanged inside
  `nix develop`.
- **The expressions are unverified.** They were written from the manual setup
  below, which does work, but no Nix was available in the environment where this
  was developed. If `nix develop` fails, that is a bug and not a local
  misconfiguration. Suspect `?submodules=1` on the OpenFHE input first (see
  [§7](#7-troubleshooting)).

---

## 3. Setup on Ubuntu or Debian

Tested on Ubuntu 24.04, GCC 13.3, 4 cores. Timings are from that machine.

### 3.1 Distribution packages

```shell
sudo apt update && sudo apt install -y \
    build-essential cmake ninja-build git python3 \
    clang lld libomp-dev zlib1g-dev libcereal-dev wget
```

`libcereal-dev` matters later: the code `keymemrt-translate` generates includes
cereal headers for weight loading.

### 3.2 {fmt} 12

```shell
git clone --depth 1 --branch 12.0.0 https://github.com/fmtlib/fmt
cd fmt
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
      -DFMT_TEST=OFF -DFMT_DOC=OFF -DBUILD_SHARED_LIBS=ON
cmake --build build -j$(nproc)
sudo cmake --install build
sudo ldconfig
```

About a minute.

### 3.3 The OpenFHE fork

```shell
git clone https://github.com/eymay/openfhe-development
cd openfhe-development
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED=ON \
      -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
cmake --build build -j$(nproc)
sudo cmake --install build
sudo ldconfig
```

Disabling the unit tests, examples and benchmarks cuts the build from close to
an hour down to about **5 minutes** on 4 cores. Installs to
`/usr/local/include/openfhe` and `/usr/local/lib`.

Verify the fork, not upstream, is what you have:

```shell
grep -r GetDynamicQSize /usr/local/include/openfhe/pke/key/evalkey.h
# virtual size_t GetDynamicQSize() const { ...
```

If that prints nothing you have upstream OpenFHE installed and compressed
rotation keys will not work.

### 3.4 KeyMemRT

Header-only for this purpose, so a checkout is all that is required:

```shell
git clone https://github.com/eymay/KeyMemRT
export KEYMEMRT=$PWD/KeyMemRT      # worth putting in ~/.bashrc
```

### 3.5 KeyMemRT-Compiler

Only needed to compile the emitted MLIR. This is the long one: a full MLIR/LLVM
build, tens of minutes to hours on a first run, and it wants a lot of disk.

```shell
wget -c https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
mkdir -p ~/bin && mv bazelisk-linux-amd64 ~/bin/bazel && chmod +x ~/bin/bazel
export PATH=$PATH:~/bin

git clone https://github.com/eymay/KeyMemRT-Compiler
cd KeyMemRT-Compiler
bazel build //tools:keymemrt-opt //tools:keymemrt-translate
export KEYMEMRT_COMPILER=$PWD
```

Bazelisk reads `.bazelversion` (8.1.0) and fetches that Bazel for you, so this
step needs network access.

---

## 4. Build and test the frontend

```shell
git clone <this repository> && cd reboot-keymemrt
cmake -B build -S .                       # frontend only: needs {fmt} 12
cmake -B build -S . -DKEYMEMRT_DIR=$KEYMEMRT   # ...and the key-memory benchmark
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

```
Test #1: test_autograd ....   Passed
Test #2: test_lowering ....   Passed
Test #3: test_emitter .....   Passed
Test #4: test_options .....   Passed
```

The frontend links against no FHE library at all. It builds a graph,
differentiates it and prints text. `-DKEYMEMRT_DIR` adds one extra target,
`measure_key_memory`.

Sanity check, no crypto involved, about a second:

```shell
./build/reboot_eval --hidden 16,8 --input-dim 8 --classes 3 --samples 128 --epochs 3
# epoch  0 | loss   0.5457 | accuracy  84.4%
# epoch  1 | loss   0.1585 | accuracy 100.0%
# epoch  2 | loss   0.0576 | accuracy 100.0%
```

This runs the *same graph* that `reboot_emit` prints as MLIR, with ciphertexts
replaced by slot vectors. Convergence here means the emitted program really is a
training step; what the FHE backend adds on top is approximation noise and cost,
not different arithmetic.

---

## 5. The full pipeline: emit → compile → run

### 5.1 All at once

```shell
export KEYMEMRT_COMPILER=/path/to/KeyMemRT-Compiler
export KEYMEMRT=/path/to/KeyMemRT
export MODEL_FLAGS="--hidden 32,16 --input-dim 16 --classes 4 --batch-size 1 --log-n 13"

KEY_MODE=imperative STEPS=4 ./scripts/run_keymemrt.sh
```

`PREFETCH=1` adds `--kmrt-key-prefetching="runtime-delegated=1" --lower-affine`
for KeyMemRT's Balanced mode.

### 5.2 Step by step

**1. Emit.**

```shell
./build/reboot_emit --hidden 64,32 --input-dim 64 --classes 10 \
                    --batch-size 1 --log-n 13 --stats \
                    -o reboot_train_step.mlir
```

`--stats` reports the graph and what it costs:

```
  layout          : 64x64 (4096 slots)
  trainable params: 4
  required depth  : 9 levels
  rotate     72
  distinct rotation indices: 18
```

Alongside the `.mlir` it writes `reboot_train_step.mlir.manifest`, recording the
model, the layout, and the argument and result order of the generated function.
The runner reads that file, so nobody has to retype the flags
([§5.3](#53-the-manifest)).

**2. Place the key management.**

```shell
keymemrt-opt --ckks-to-lwe --lwe-to-openfhe \
             --annotate-module="backend=openfhe scheme=ckks" \
             --openfhe-configure-crypto-context --kmrt-merge-rotation-keys \
             --bootstrap-rotation-analysis --cse --openfhe-insert-clear-ops \
             reboot_train_step.mlir > reboot_train_step.opt.mlir
```

Every `ckks.rotate {static_shift = k}` becomes `kmrt.load_key` / `openfhe.rot` /
`kmrt.clear_key`. This rewrite is why the packing is lowered to explicit
rotations: `EvalSumRows` and `EvalSumCols` name no indices for the pass to
match on.

**3. Translate.**

```shell
keymemrt-translate --emit-openfhe-pke reboot_train_step.opt.mlir \
                   > reboot_train_step.cc
```

**4. Link with a host.** The generated file is a *library*. It expects two
globals, `KeyMemRT keymem_rt;` and `std::unique_ptr<ResourceMonitor> monitor;`
(the second from `generic_header.h`), and exposes three entry points:

```cpp
CryptoContextT reboot_train_step__generate_crypto_context();
CryptoContextT reboot_train_step__configure_crypto_context(CryptoContextT, PrivateKeyT);
std::vector<CiphertextT> reboot_train_step(CryptoContextT, std::vector<CiphertextT>);
```

`drivers/reboot_runner.cc` is that host. Build it against the generated file:

```shell
g++ -std=c++17 -O2 -fopenmp \
  -I include -I $KEYMEMRT/include \
  -I /usr/local/include/openfhe -I /usr/local/include/openfhe/core \
  -I /usr/local/include/openfhe/pke -I /usr/local/include/openfhe/binfhe \
  -I /usr/local/include -I /usr \
  drivers/reboot_runner.cc reboot_train_step.cc build/libreboot.a \
  -o reboot_runner \
  -L /usr/local/lib -lOPENFHEpke -lOPENFHEcore -lOPENFHEbinfhe -lfmt
```

The `-I /usr` is deliberate: generated code includes cereal as
`include/cereal/...`, so `/usr` must be on the include path for that spelling to
resolve to `/usr/include/cereal/...`.

**5. Run.**

```shell
mkdir -p keys results
SERIALIZED_DATA_DIR=./keys ./reboot_runner \
    --manifest reboot_train_step.mlir.manifest \
    --key-mode imperative --input-dir ./keys --result-dir ./results \
    --steps 4
```

`--key-mode` picks the strategy:

| mode | behaviour |
| --- | --- |
| `ignore` | every key resident for the entire run; the baseline |
| `imperative` | one key paged in per rotation, dropped after |
| `prefetch` | background loads overlap the computation, bounded by `--prefetch-sat` |
| `speculative` | as imperative, but waits for keys to arrive (cold start) |

`ResourceMonitor` writes a memory and time trace to `--result-dir`.

### 5.3 The manifest

`reboot_runner` takes `--manifest`, not model flags. The generated function has a
fixed argument order; the runner rebuilds the graph to learn the shapes and
packings, then checks that order against the manifest and refuses to run on a
mismatch:

```
error: the model does not match the manifest: argument count is '6' but the
module was emitted with '11'.  Re-emit the module, or point --manifest at the
one written beside it.
```

Retyping the same flags for both binaries would not be a contract. One mistyped
width builds a different argument order, and the run then produces wrong numbers
instead of failing.

---

## 6. Rotation-key memory, measured

`tools/measure_key_memory.cc` builds one context and populates it two ways: the
key set ReBoot's `lib/cryptocontext.py` builds (`EvalMultKeyGen`,
`EvalSumKeyGen`, `EvalSumRowsKeyGen(col_size)`, `EvalSumColsKeyGen`), and the
rotation indices the emitted module names, which are generated, compressed,
written one file each, and dropped.

```shell
KEYMEMRT=$KEYMEMRT ./scripts/measure_key_memory.sh --log-n 16 --depth 24
```

Measured on the fork, `HEStd_NotSet`, 50-bit scaling modulus. Sizes come from
walking each key's RNS limbs, with keys shared between the EvalSum and
automorphism maps counted once:

| | N = 2¹⁴, depth 11 | N = 2¹⁶, depth 24 |
| --- | --- | --- |
| ReBoot: keys resident | 25 | 29 |
| ReBoot: key material | **300 MB** | **2871 MB** |
| KeyMemRT: indices named by the step | 18 | 20 |
| KeyMemRT: keys resident | 1 | 1 |
| KeyMemRT: key material | **12 MB** | **99 MB** |
| reduction | **25×** | **29×** |

Two effects compound. Naming the rotations shrinks the set that has to exist at
all, since `EvalSumKeyGen` provisions every power-of-two rotation whether the
program uses it or not. Paging then drops the resident set to one. On top of
that, KeyMemRT stores each key truncated to the level it is used at:

```
key size against the level it is used at   (N = 2^14, depth 11)
  level  1: 12.0 MB      level  6:  8.2 MB
  level  3: 10.5 MB      level  7:  7.5 MB
  level  4:  9.8 MB      level  8:  6.8 MB
  level  5:  9.0 MB      level 10:  5.2 MB
```

The figures cover **rotation keys**: the memory KeyMemRT manages, and the
dominant term in an FHE process. They exclude the bootstrapping key set, which
both sides must hold and which KeyMemRT stages as one bundle around the weight
refresh. Process RSS appears alongside as a cross-check, but it is the weaker
number because the allocator does not return freed pages. The key-material
figures are exact.

---

## 7. Troubleshooting

**`fatal error: fmt/format.h: No such file`** or
**`Could not find a configuration file for package "fmt" ... version 12`**
Your distro's {fmt} is too old. Build 12.0.0 from source ([§3.2](#32-fmt-12)),
then `sudo ldconfig`.

**`error while loading shared libraries: libOPENFHEpke.so.1`**
The installed libraries are not on the loader path:

```shell
echo /usr/local/lib | sudo tee /etc/ld.so.conf.d/openfhe.conf
sudo ldconfig
```

**`DropLastElement: Removing last element of DCRTPoly renders it invalid`**
The modulus chain ran out of levels. The step needs more depth than the context
has. `reboot_emit --stats` prints `required depth`; pass `--levels` (or
`--depth` to the benchmark) at least that high. Without bootstrapping the levels
only ever go down, so a run of more than one step needs roughly twice the
single-step depth.

**`reference to 'Format' is ambiguous`** in your own host code
OpenFHE declares a global `enum Format` in `core/utils/inttypes.h`. This project
calls its own `pack_format_t` for exactly that reason; if you add code that does
`using namespace lbcrypto;` alongside your own `Format`, rename yours.

**`Only functions with a single return type are supported`** from
`keymemrt-translate`
The module returns several values. The emitter avoids this by bundling both
sides of the signature into one `tensor<Nx!ct>`; if you have hand-edited the
MLIR, that is what broke.

**Nix: OpenFHE fails at `git submodule update` / cereal headers missing**
The `openfhe-fork` input needs `?submodules=1`. Its CMake shells out to `git
submodule update`, which cannot reach the network inside the build sandbox, so
the submodules have to arrive with the source.

**Bazel build of the compiler is enormous or fails on network access**
Expected: it builds MLIR and LLVM. A Nix sandbox cannot run it, so build it in
an ordinary shell. `nix develop` works, because the sandbox applies only to
`nix build`.

**The generated function's signature does not match the runner's declarations**
`drivers/reboot_runner.cc` declares the three entry points near the top against
what `keymemrt-translate` is expected to emit. If the translator spells them
differently, that block is the single place to adjust.

---

## 8. How it works

### 8.1 Autograd, not a hand-written backward pass

`tensor_graph.h` is a small tensor-level IR holding vectors and weight matrices
instead of slot vectors, which leaves `autograd.cc` as ordinary reverse-mode
differentiation. The model builder assembles only the *forward* network and one
loss seed per local classifier. Differentiation supplies the backward pass.

ReBoot's published backward rules fall out of the standard vector-Jacobian
products:

| forward | VJP | what it lowers to |
| --- | --- | --- |
| `matmul(x, W)` | `∂x = matmul_transposed(g, W)` | the same weight ciphertext, summed the *other* way: no transpose, no repacking |
| `matmul(x, W)` | `∂W = outer(x, g)` | a bare elementwise product: one operand is Repeated and the other Expanded, so the product already *is* the outer product in the weight layout, at zero rotations |
| `poly_relu(x)` | `g · (2x + 1)` | one level, as in the paper |

Each block ends in a `stop_gradient`, so its error signal never leaves the
block. `test_autograd` checks this against central finite differences: a block's
gradient matches the gradient of *its own* loss to 4e-12 and departs from the
gradient of the full objective by 2.5e-2. That confinement keeps the depth of a
step independent of network depth.

The optimiser comes from the same graph. Because the velocities are function
arguments, passing zeros on the first step reproduces ReBoot's separate
"initialise the velocity" branch exactly, so there is one update rule:

```
g' = g + wd·W;   Δ = lr·g' + m·lr·g' + m²·lr·V;   W' = W − Δ;   V' = m·V + g'
```

All three factors are `ckks.mul_scalar`, which costs no level.

### 8.2 RE and CE blocks

Packing alternates down the network, which is what RE and CE *are* in the paper.
`--hidden 128,64,32` gives the paper's eMLP-3, `RE(128) → CE(64) → RE(32)`:

```
w_fwd_0    [64x128]   row-packed      <- RE-Block(128)
w_lrn_0    [128x10]   column-packed
w_fwd_1    [128x64]   column-packed   <- CE-Block(64)
w_lrn_1    [64x10]    row-packed
w_head     [64x32]    row-packed      <- RE(32)
w_out      [32x10]    column-packed
```

Each block's classifier takes the opposite packing of its forward layer, so
nothing ever repacks. The depth asymmetry survives as well: `matmul_re` costs one
level and `matmul_ce` costs two, the extra level being the mask in the column
summation. That matches the paper's τ=2 against τ=3.

### 8.3 Packing lowered to named rotations

```
sum_rows  for k = cols, 2·cols, ... < slots:  x += rot(x, k)      (no extra level)
sum_cols  for k = 1, 2, ... < cols:           x += rot(x, k)
          x *= mask(first column of each row)                     (one level, as EvalSumCols)
          for k = 1, 2, ... < cols:           x += rot(x, −k)
```

`test_lowering` runs the tensor graph and the slot graph on the same inputs and
compares every result of the step (updated weights, velocities and predictions)
across three architectures. Agreement is ~1e-17.

### 8.4 The emitted module

```mlir
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 12, Q = [...], P = [...], logDefaultScale = 26>} {
  func.func @reboot_train_step(%args: tensor<6x!ct> {reboot.argument_names = ["w_head", "v_w_head", "w_out", "v_w_out", "x_0", "y_expanded_0"]})
      -> tensor<5x!ct> attributes {reboot.result_names = [...]} {
    %idx0 = arith.constant 0 : index
    %arg0 = tensor.extract %args[%idx0] : tensor<6x!ct>  // w_head
    ...
    %0 = ckks.mul %arg4, %arg0 : (!ct, !ct) -> !ct_d3
    %1 = ckks.relinearize %0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : (!ct_d3) -> !ct
    %2 = ckks.rotate %1 {static_shift = 8 : i64} : !ct
    %3 = ckks.add %1, %2 : (!ct, !ct) -> !ct
    ...
    %n = ckks.bootstrap %m : !ct -> !ct      // w_head_next
    %res_init = tensor.empty() : tensor<5x!ct>
    %res_0 = tensor.insert %n into %res_init[%idx0] : tensor<5x!ct>
    ...
    return %res_4 : tensor<5x!ct>
  }
}
```

NTT-friendly primes are generated for the requested `logN` and depth. Every
ciphertext is emitted at the top of the chain, and each product is relinearised
straight back to the canonical basis, which leaves rescaling to OpenFHE's
`FLEXIBLEAUTO` at run time. The depth the graph needs is computed from the graph
itself and sizes the modulus chain.

---

## 9. Why ReBoot cannot simply run on KeyMemRT

ReBoot is a Python library over a pybind11 module (`reboot_cpp`) pinned to
**OpenFHE 1.2.1 + openfhe-python 0.8.9**. Three structural obstacles:

1. **Two OpenFHE builds, one interpreter.** Every `reboot_cpp` entry point takes
   `CryptoContext<DCRTPoly>` and friends by value, so it only works if
   `openfhe-python` registered those exact types. KeyMemRT builds against the
   fork; both cannot be right at once.
2. **The fork's feature is unreachable from Python.** Compressed rotation keys
   depend on `Get/SetDynamicQSize`, and upstream 1.2.1, the version
   openfhe-python wraps, does not define it.
3. **ReBoot's rotations are invisible.** Its matrix products call
   `EvalSumRows`/`EvalSumCols`, whose rotations happen inside OpenFHE against
   the **EvalSum** key map. KeyMemRT pages keys out of the **automorphism** key
   map one index at a time; it has no hook there, and the indices are never
   named by the program.

One change addresses all three: name every rotation in the IR and let the
compiler place the key management.

---

## 10. Repository layout

```
include/reboot/
  options.h        declarative command line, shared by every binary
  manifest.h       the emit -> run contract: model, layout and ABI order
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
  reboot_runner.cc host driver for the generated OpenFHE code
tools/
  measure_key_memory.cc  rotation-key working set, ReBoot vs KeyMemRT
scripts/
  run_keymemrt.sh        emit -> keymemrt-opt -> keymemrt-translate -> build -> run
  measure_key_memory.sh  build and run the memory benchmark
docs/
  optimizations.md       every optimisation, its cost, and how it was measured
tests/
  test_autograd.cc gradients vs finite differences; gradient locality
  test_lowering.cc packed lowering vs tensor semantics
  test_emitter.cc  op counts, rotation attributes, signature
  test_options.cc  command-line parsing and the manifest guard
flake.nix          Nix dev shell and packages
shell.nix          the same, for setups without flakes
```

---

## 11. Style

Snake case throughout: functions, variables and file names, and types with a
`_t` suffix (`option_parser_t`, `slot_graph_t`, `pack_format_t`). Enumerators are
plain snake case inside their scoped enum (`pack_format_t::repeated`), class
members carry a trailing underscore, and indentation is tabs; `.clang-format`
is in this directory.

The remaining CamelCase belongs to other people's API: OpenFHE's `CiphertextT`,
KeyMemRT's `deserializeKey`. Those keep their original spelling so the generated
code and the runtime headers still read as theirs.

Options are registered once per binary in a table that binds each flag to the
variable it sets and carries its help text, so `--help` and the parser come from
the same place. `--name value` and `--name=value` both work, and a missing
value, a malformed number or an unknown flag is reported against the flag that
caused it.

## Licence

The ReBoot algorithm being reimplemented is GPL-3.0 (see the
[upstream repository](https://github.com/AI-Tech-Research-Lab/ReBoot)); this
work keeps that licence.
