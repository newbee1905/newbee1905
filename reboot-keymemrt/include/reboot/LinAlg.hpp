// SPDX-License-Identifier: GPL-3.0-or-later
//
// Slot-level primitives of ReBoot, rewritten on explicit rotations.
//
// The reference ReBoot implementation builds its two matrix products out of
// OpenFHE's EvalSumRows / EvalSumCols.  Those helpers hide their rotations
// inside the library and draw on the EvalSum key map, which KeyMemRT does not
// manage: it pages keys in and out of the *automorphism* key map, one rotation
// index at a time.  Using them would leave every rotation key of the training
// step resident for the whole run, which is exactly the memory behaviour
// KeyMemRT exists to remove.
//
// So the two summations are spelled out here as rotate-and-add trees over
// explicit indices.  Each rotation goes through the backend, and the CKKS
// backend brackets it with keymem_rt.deserializeKey() / keymem_rt.clearKey(),
// the same shape the KeyMemRT compiler emits for `openfhe.rot`.
//
//   sumRows: sum down the columns, result replicated in every row
//            (Repeated output)   log2(rows) rotations, no extra depth
//   sumCols: sum along each row, result replicated across the row
//            (Expanded output)   2*log2(cols) rotations, one masking level
//
// The depth of both matches OpenFHE's own EvalSumRows / EvalSumCols, so the
// multiplicative depth of a training step is unchanged from the paper.

#ifndef REBOOT_LINALG_HPP_
#define REBOOT_LINALG_HPP_

#include <vector>

#include "reboot/Backend.hpp"
#include "reboot/Layout.hpp"

namespace reboot {

// Every rotation index the training step can ask for, given a layout.
// Used to size key generation on the client.
inline std::vector<int32_t> rotationIndices(const Layout &layout) {
  std::vector<int32_t> idx;
  for (int k = layout.cols; k < layout.slots(); k <<= 1) idx.push_back(k);
  for (int k = 1; k < layout.cols; k <<= 1) {
    idx.push_back(k);
    idx.push_back(-k);
  }
  return idx;
}

// Sum over rows: out[i][j] = sum_i' x[i'][j], replicated down every row.
// Valid because `rows` is a power of two and the ciphertext is fully packed.
template <class B>
typename B::Ct sumRows(B &be, const typename B::Ct &x, const Layout &layout) {
  typename B::Ct acc = x;
  for (int k = layout.cols; k < layout.slots(); k <<= 1)
    acc = be.add(acc, be.rotate(acc, k));
  return acc;
}

// Sum along rows: out[i][j] = sum_j' x[i][j'], replicated across every row.
//
// Three stages, mirroring OpenFHE's EvalSumCols:
//   1. rotate-and-add by 1, 2, ... cols/2 -> slot s holds the cyclic window sum
//      x[s] + ... + x[s + cols - 1]; only the row starts are the true row sums,
//   2. mask the row starts (this is the one level EvalSumCols also spends),
//   3. rotate-and-add right by 1, 2, ... cols/2 to replicate the row start
//      across its row without bleeding into the neighbouring row.
template <class B>
typename B::Ct sumCols(B &be, const typename B::Ct &x, const Layout &layout,
                       const Mask &firstCol) {
  typename B::Ct acc = x;
  for (int k = 1; k < layout.cols; k <<= 1)
    acc = be.add(acc, be.rotate(acc, k));
  acc = be.mulMask(acc, firstCol);
  for (int k = 1; k < layout.cols; k <<= 1)
    acc = be.add(acc, be.rotate(acc, -k));
  return acc;
}

// RE-Matmul: Expanded input x Repeated-producing weights.
//   z_j = sum_i a_i W[i][j]        (a Expanded, W row-packed, z Repeated)
template <class B>
typename B::Ct matmulRE(B &be, const typename B::Ct &a, const typename B::Ct &w,
                        const Layout &layout) {
  return sumRows(be, be.mul(a, w), layout);
}

// CE-Matmul: Repeated input x Expanded-producing weights.
//   y_i = sum_j a_j W_col[j][i]    (a Repeated, W column-packed, y Expanded)
template <class B>
typename B::Ct matmulCE(B &be, const typename B::Ct &a, const typename B::Ct &w,
                        const Layout &layout, const Mask &firstCol) {
  return sumCols(be, be.mul(a, w), layout, firstCol);
}

// Outer product of the saved input and the incoming error signal, accumulated
// over the batch.  One of the two operands is Expanded and the other Repeated,
// so the elementwise product already lands in the layer's weight layout - no
// rotation is needed, which is why the weight gradient costs no rotation keys.
template <class B>
typename B::Ct outerProductSum(B &be, const std::vector<typename B::Ct> &x,
                               const std::vector<typename B::Ct> &delta) {
  if (x.empty() || x.size() != delta.size())
    throw std::invalid_argument("batch size mismatch in outerProductSum");
  typename B::Ct acc = be.mul(x[0], delta[0]);
  for (size_t i = 1; i < x.size(); ++i)
    acc = be.add(acc, be.mul(x[i], delta[i]));
  return acc;
}

// Sum of all valid slots, replicated everywhere.  Only used for the encrypted
// loss read-out, which is diagnostic; the training step itself never needs it.
template <class B>
typename B::Ct sumAll(B &be, const typename B::Ct &x, const Layout &layout) {
  typename B::Ct acc = x;
  for (int k = 1; k < layout.slots(); k <<= 1)
    acc = be.add(acc, be.rotate(acc, k));
  return acc;
}

}  // namespace reboot

#endif  // REBOOT_LINALG_HPP_
