# Optimizations

Every change made for speed or memory, with what it cost, what it bought, and
how that was established. Entries stay here after the fact, including the ones
that were investigated and rejected, so the reasoning survives the decision.

Measurements come from Ubuntu 24.04, GCC 13.3, 4 cores, 15 GB RAM, on the
`eymay/openfhe-development` fork reporting 1.2.3.

---

## 1. Weight gradients cost zero rotations

**Change.** The vector-Jacobian product for a weight matrix is
`∂W = outer(x, g)`, lowered to a single `ckks.mul`.

**Why it is free.** ReBoot's packing stores one operand Repeated
(`A[i][j] = v[j]`) and the other Expanded (`A[i][j] = v[i]`). Their elementwise
product is `x_i · g_j` at slot `(i, j)`, which already is the outer product laid
out in the layer's own weight packing. No summation, so no rotation.

**Cost.** One multiplicative level for the product, which the gradient needs
anyway.

**Effect.** For eMLP-3 at N = 2¹⁴, the six weight gradients account for 6 of the
24 ciphertext products in a step and 0 of its 120 rotations.

**Where.** `lower_to_slots`, `tensor_op_t::outer` case in `lib/slot_graph.cc`.
`tests/test_lowering.cc` checks the packed result against the unpacked outer
product.

---

## 2. Transposed matmul reuses the weight ciphertext

**Change.** `∂x = matmul_transposed(g, W)` lowers to the same elementwise
product as the forward pass, summed in the other direction: `sum_cols` where the
forward used `sum_rows`, and the reverse for a column-packed layer.

**Alternative rejected.** Materialising `Wᵀ` as a second ciphertext. That would
double the weight state the client encrypts, double what the step bootstraps,
and add a transpose whose cost under this packing is a full
rotate-mask-accumulate over the layout.

**Cost.** Nothing beyond the summation the gradient needs.

**Effect.** Weight state stays at one ciphertext per layer. A step bootstraps
`2 × layers` ciphertexts (weights and velocities) instead of `3 ×`.

**Where.** `tensor_op_t::matmul_transposed` in `lib/slot_graph.cc`.

---

## 3. Optimiser scalars consume no level

**Change.** Learning rate, momentum and weight decay multiply through
`ckks.mul_scalar`, which the fork documents as multiplying by a real constant
without rescaling.

**Consequence.** The Nesterov update

```
g' = g + wd·W;   Δ = lr·g' + m·lr·g' + m²·lr·V;   W' = W − Δ;   V' = m·V + g'
```

adds no depth to a training step, matching the remark in ReBoot's README that
weight decay and momentum are free. A single rule covers both the first step and
the steady state, because passing `V = 0` reproduces ReBoot's separate
initialisation branch.

**Effect.** The modulus chain is sized by the forward and backward passes alone.
eMLP-2 at N = 2¹³ consumes 10 levels, none of them from the optimiser, so the
emitter provisions a 12-prime chain.

**Where.** `build_train_step` in `lib/reboot_model.cc`;
`slot_graph_t::mul_scalar` in `lib/slot_graph.cc`.

---

## 4. Summation trees at OpenFHE's own depth

**Change.** `EvalSumRows` and `EvalSumCols` are replaced by explicit
rotate-and-add trees.

```
sum_rows  for k = cols, 2·cols, ... < slots:  x += rot(x, k)
sum_cols  for k = 1, 2, ... < cols:           x += rot(x, k)
          x *= mask(first column of each row)
          for k = 1, 2, ... < cols:           x += rot(x, −k)
```

**Cost accounting.** `sum_rows` performs log₂(rows) rotations and spends no
level. `sum_cols` performs 2·log₂(cols) rotations and spends one level on the
mask, which is the same level OpenFHE's own `EvalSumCols` spends. Depth per
training step is therefore unchanged from the paper: τ=2 for an RE-Block forward
and τ=3 for a CE-Block.

**What it buys.** The rotation indices appear in the IR, where
`--ckks-to-lwe` and `--lwe-to-openfhe` can rewrite each one into
`kmrt.load_key` / `openfhe.rot` / `kmrt.clear_key`. Section 5 below quantifies
the result.

**Where.** `slot_graph_t::sum_rows` and `sum_cols` in `lib/slot_graph.cc`.
`tests/test_lowering.cc` verifies the algebra to ~1e-17 and checks that every
index used is a power of two inside the slot vector.

---

## 5. Rotation keys: 29× less resident memory

**Measured**, not estimated. `tools/measure_key_memory.cc` builds one context and
populates it two ways, then walks each key's RNS limbs to size it, counting keys
shared between the EvalSum and automorphism maps once.

| | N = 2¹⁴, depth 11 | N = 2¹⁶, depth 24 |
| --- | --- | --- |
| ReBoot: keys resident | 25 | 29 |
| ReBoot: key material | 300 MB | 2871 MB |
| KeyMemRT: indices named by the step | 18 | 20 |
| KeyMemRT: keys resident | 1 | 1 |
| KeyMemRT: key material | 12 MB | 99 MB |
| reduction | 25× | 29× |

Two effects compound.

1. **Naming shrinks the set.** `EvalSumKeyGen` provisions every power-of-two
   rotation whether the program uses it or not, so ReBoot holds 25 or 29 keys
   where the step rotates over 18 or 20 distinct indices.
2. **Paging shrinks residency to one.** Each key is loaded for its rotation and
   cleared afterwards.

A third effect rides along: KeyMemRT stores each key truncated to the level it is
used at, so a key used deep in the step is smaller.

```
key size against the level it is used at   (N = 2^14, depth 11)
  level  1: 12.0 MB      level  6:  8.2 MB
  level  3: 10.5 MB      level  7:  7.5 MB
  level  4:  9.8 MB      level  8:  6.8 MB
  level  5:  9.0 MB      level 10:  5.2 MB
```

**Scope.** Rotation keys only. The bootstrapping key set is common to both sides
and is staged as one bundle around the weight refresh. Process RSS is recorded
as a cross-check, but it overstates residency because the allocator does not
return freed pages to the OS. The key-material figures are the exact ones.

**A correction worth recording.** The first run of this benchmark reported
5742 MB for ReBoot at N = 2¹⁶, exactly double the true figure.
`EvalSumRowsKeyGen` and `EvalSumColsKeyGen` place their keys in both the EvalSum
map and the automorphism map, and the two maps share the objects, so summing
both maps counted every key twice. The tool now takes the union by object
identity. The published number is 2871 MB.

---

## 6. Investigated and rejected

### 6.1 SIMD-aligned or custom allocator

**Finding.** OpenFHE already ships a fixed-block allocator
(`core/utils/blockAllocator/`, `xvector`), and limb storage selects between it
and `std::vector` at compile time:

```cpp
#if BLOCK_VECTOR_ALLOCATION != 1
    std::vector<IntegerType> m_data{};
#else
    xvector<IntegerType> m_data{};
#endif
```

`BLOCK_VECTOR_ALLOCATION` is defined nowhere in the fork's build files, so the
block allocator is dormant and plain `std::vector` is in use.

**Decision.** Writing an allocator here would be misdirected. This frontend
allocates graph nodes and slot vectors, none on a hot path: emitting eMLP-3 at
N = 2¹⁵ takes **6 ms** and produces 1278 slot values. The allocations that
matter belong to OpenFHE's limb buffers, inside types that are not templated on
an allocator. The available experiment is rebuilding the fork with
`BLOCK_VECTOR_ALLOCATION=1` and measuring, not writing new code.

**On alignment.** `std::vector<uint64_t>` is already 16-byte aligned by default,
and OpenFHE's NTT kernels manage their own layout. Over-aligned allocation pays
off alongside an AVX-512 backend such as HEXL, which is a separate decision.

### 6.2 Arena allocator per ReBoot block

**The idea.** A local-loss block has a bounded working set that dies at the
block boundary, which is arena-shaped.

**Why it is not implemented.** Two obstacles.

- *Plumbing.* The objects are `DCRTPoly` limb vectors inside OpenFHE. Reaching
  them means the `BLOCK_VECTOR_ALLOCATION` path above, which is a global
  fixed-block pool and not a per-block arena.
- *Target.* Section 5 measured rotation keys at 2871 MB against ~99 MB for a
  single key. Ciphertext temporaries are small next to that, so an arena would
  tidy allocator churn without moving the number that matters.

**The version worth building instead.** The slot graph knows each value's last
use, so the emitter can compute liveness, report the true peak ciphertext
working set, and emit a clear at each last use. `--openfhe-insert-clear-ops`
already exists in the compiler; feeding it precise liveness beats letting it
infer. Not implemented yet.

### 6.3 ATen tensors from libtorch

**Decision.** No for the core, with one bounded exception.

The frontend never multiplies a tensor. It builds a graph of about 1300 nodes
and prints text, in 6 ms. Torch's autograd also has the wrong shape for this
job: it executes a backward pass, whereas the emitter needs to *emit* one as IR.
Reaching that through `torch::jit` tracing and then mapping the traced graph onto
packed CKKS operations is more work than the ~120-line reverse-mode pass in
`lib/autograd.cc`, on top of a multi-gigabyte dependency for a project whose only
dependency is {fmt}.

**The exception.** `tests/test_autograd.cc` currently checks gradients against
central finite differences, accurate to ~1e-11. Building the same architecture in
libtorch and comparing against `torch.autograd` would catch a systematically
wrong VJP that stays smooth, which finite differences can miss. That is a
test-only, optional dependency.

---

## 7. Frontend cost, for reference

Emitting is not a bottleneck, which is why none of the above targets it.

| configuration | slot values | rotations | emit time |
| --- | --- | --- | --- |
| eMLP-1, N = 2¹² | 96 | 28 | 3 ms |
| eMLP-2, N = 2¹³, batch 1 | 223 | 72 | 3 ms |
| eMLP-3, N = 2¹⁵, batch 4 | 1278 | 504 | 6 ms |

The emitted text is dominated by one constant: the first-column mask is a dense
`tensor<Nxf64>` written out in full, which is most of a 76 KB module at
N = 2¹³. It is emitted once and reused by every column summation.
