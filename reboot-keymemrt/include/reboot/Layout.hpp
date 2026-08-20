// SPDX-License-Identifier: GPL-3.0-or-later
//
// Slot layout and packing for the ReBoot-on-KeyMemRT port.
//
// ReBoot packs every tensor into a single CKKS ciphertext holding a
// `rows x cols` matrix in row-major order (rows*cols == number of slots).
// Vectors live in one of two formats (Section 4 of the ReBoot paper):
//
//   Repeated  A[i][j] = v[j]   the vector lies along a row, replicated downwards
//                              (a|b|c || a|b|c || a|b|c)
//   Expanded  A[i][j] = v[i]   the vector lies along a column, replicated rightwards
//                              (a|a|a || b|b|b || c|c|c)
//
// A row-packed layer consumes an Expanded vector and produces a Repeated one;
// a column-packed layer does the opposite.  Alternating the two formats across
// layers removes any need for repacking between layers.

#ifndef REBOOT_LAYOUT_HPP_
#define REBOOT_LAYOUT_HPP_

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace reboot {

inline int nextPowerOfTwo(int x) {
  int p = 1;
  while (p < x) p <<= 1;
  return p;
}

inline bool isPowerOfTwo(int x) { return x > 0 && (x & (x - 1)) == 0; }

// Geometry of the packed matrix held by one ciphertext.
struct Layout {
  int rows = 0;  // number of rows      (`row_size` in the reference ReBoot code)
  int cols = 0;  // length of one row   (`col_size` in the reference ReBoot code)

  Layout() = default;
  Layout(int r, int c) : rows(r), cols(c) {
    if (!isPowerOfTwo(rows) || !isPowerOfTwo(cols))
      throw std::invalid_argument("Layout dimensions must be powers of two");
  }

  int slots() const { return rows * cols; }
  std::string str() const {
    return std::to_string(rows) + "x" + std::to_string(cols);
  }
};

enum class Format { Repeated, Expanded };

inline const char *formatName(Format f) {
  return f == Format::Repeated ? "repeated" : "expanded";
}

// The format a packed linear layer expects on its input / produces on its
// output.  `rowPacking` selects between the two encodings of the weight matrix.
inline Format inputFormat(bool rowPacking) {
  return rowPacking ? Format::Expanded : Format::Repeated;
}
inline Format outputFormat(bool rowPacking) {
  return rowPacking ? Format::Repeated : Format::Expanded;
}

// Pack a plain vector into `layout.slots()` slots using `f`.
inline std::vector<double> packVector(const std::vector<double> &v, Format f,
                                      const Layout &layout) {
  std::vector<double> out(static_cast<size_t>(layout.slots()), 0.0);
  if (f == Format::Repeated) {
    if (static_cast<int>(v.size()) > layout.cols)
      throw std::invalid_argument("vector longer than layout.cols");
    for (int i = 0; i < layout.rows; ++i)
      for (size_t j = 0; j < v.size(); ++j)
        out[static_cast<size_t>(i) * layout.cols + j] = v[j];
  } else {
    if (static_cast<int>(v.size()) > layout.rows)
      throw std::invalid_argument("vector longer than layout.rows");
    for (size_t i = 0; i < v.size(); ++i)
      for (int j = 0; j < layout.cols; ++j)
        out[i * layout.cols + static_cast<size_t>(j)] = v[i];
  }
  return out;
}

// Read the first `n` entries of a vector back out of a packed slot vector.
inline std::vector<double> unpackVector(const std::vector<double> &slots,
                                        Format f, const Layout &layout, int n) {
  std::vector<double> out(static_cast<size_t>(n), 0.0);
  for (int k = 0; k < n; ++k) {
    size_t idx = (f == Format::Repeated)
                     ? static_cast<size_t>(k)                       // row 0
                     : static_cast<size_t>(k) * layout.cols;        // column 0
    out[static_cast<size_t>(k)] = slots[idx];
  }
  return out;
}

// Pack a weight matrix `w` of shape (inFeatures x outFeatures), given in
// row-major order, into the layout.
//
//   rowPacking  : pad to (rows x cols) and flatten            -> W_row
//   !rowPacking : pad to (cols x rows), transpose and flatten -> W_col
//
// Both encodings end up as a `rows x cols` slot matrix, so the same slot-level
// primitives serve both.
inline std::vector<double> packWeights(const std::vector<double> &w,
                                       int inFeatures, int outFeatures,
                                       bool rowPacking, const Layout &layout) {
  if (w.size() != static_cast<size_t>(inFeatures) * outFeatures)
    throw std::invalid_argument("weight size does not match its shape");
  const int padRows = rowPacking ? layout.rows : layout.cols;
  const int padCols = rowPacking ? layout.cols : layout.rows;
  if (inFeatures > padRows || outFeatures > padCols)
    throw std::invalid_argument("weights do not fit into the layout");

  std::vector<double> out(static_cast<size_t>(layout.slots()), 0.0);
  for (int i = 0; i < inFeatures; ++i) {
    for (int j = 0; j < outFeatures; ++j) {
      const double value = w[static_cast<size_t>(i) * outFeatures + j];
      // (i, j) in the padded matrix; transposed when column-packed.
      const int r = rowPacking ? i : j;
      const int c = rowPacking ? j : i;
      out[static_cast<size_t>(r) * layout.cols + c] = value;
    }
  }
  return out;
}

// Inverse of packWeights, used by the tests and by weight read-back.
inline std::vector<double> unpackWeights(const std::vector<double> &slots,
                                         int inFeatures, int outFeatures,
                                         bool rowPacking, const Layout &layout) {
  std::vector<double> out(static_cast<size_t>(inFeatures) * outFeatures, 0.0);
  for (int i = 0; i < inFeatures; ++i) {
    for (int j = 0; j < outFeatures; ++j) {
      const int r = rowPacking ? i : j;
      const int c = rowPacking ? j : i;
      out[static_cast<size_t>(i) * outFeatures + j] =
          slots[static_cast<size_t>(r) * layout.cols + c];
    }
  }
  return out;
}

// Mask that keeps the first column of every row (slot index % cols == 0).
// Used by the column-summation primitive.
inline std::vector<double> firstColumnMask(const Layout &layout) {
  std::vector<double> mask(static_cast<size_t>(layout.slots()), 0.0);
  for (int i = 0; i < layout.rows; ++i)
    mask[static_cast<size_t>(i) * layout.cols] = 1.0;
  return mask;
}

}  // namespace reboot

#endif  // REBOOT_LAYOUT_HPP_
