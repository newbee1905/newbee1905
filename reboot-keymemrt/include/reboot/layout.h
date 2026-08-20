// SPDX-License-Identifier: GPL-3.0-or-later
//
// Slot layout and packing for the ReBoot-on-KeyMemRT port.
//
// ReBoot packs every tensor into a single CKKS ciphertext holding a
// `rows x cols` matrix in row-major order (rows*cols == number of slots).
// Vectors live in one of two formats (Section 4 of the ReBoot paper):
//
//   Repeated  A[i][j] = v[j]  the vector lies along a row, replicated down
//                             (a|b|c || a|b|c || a|b|c)
//   Expanded  A[i][j] = v[i]  the vector lies along a column, replicated right
//                             (a|a|a || b|b|b || c|c|c)
//
// A row-packed layer consumes an Expanded vector and produces a Repeated one;
// a column-packed layer does the opposite.  Alternating the two formats across
// layers removes any need for repacking between layers.

#ifndef REBOOT_LAYOUT_H_
#define REBOOT_LAYOUT_H_

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace reboot {

inline int next_power_of_two(int x) {
	int p = 1;
	while (p < x) p <<= 1;
	return p;
}

inline bool is_power_of_two(int x) {
	return x > 0 && (x & (x - 1)) == 0;
}

// Geometry of the packed matrix held by one ciphertext.
struct Layout {
	int rows =
		0;	// number of rows      (`row_size` in the reference ReBoot code)
	int cols =
		0;	// length of one row   (`col_size` in the reference ReBoot code)

	Layout() = default;
	Layout(int r, int c) : rows(r), cols(c) {
		if (!is_power_of_two(rows) || !is_power_of_two(cols))
			throw std::invalid_argument(
				"Layout dimensions must be powers of two");
	}

	int slots() const { return rows * cols; }
	std::string str() const {
		return std::to_string(rows) + "x" + std::to_string(cols);
	}
};

enum class PackFormat { kRepeated, kExpanded };

inline const char *format_name(PackFormat f) {
	return f == PackFormat::kRepeated ? "repeated" : "expanded";
}

// The format a packed linear layer expects on its input / produces on its
// output.  `row_packing` selects between the two encodings of the weight
// matrix.
inline PackFormat input_format(bool row_packing) {
	return row_packing ? PackFormat::kExpanded : PackFormat::kRepeated;
}
inline PackFormat output_format(bool row_packing) {
	return row_packing ? PackFormat::kRepeated : PackFormat::kExpanded;
}

// Pack a plain vector into `layout.slots()` slots using `f`.
inline std::vector<double> pack_vector(const std::vector<double> &v, PackFormat f,
									   const Layout &layout) {
	std::vector<double> out(static_cast<size_t>(layout.slots()), 0.0);
	if (f == PackFormat::kRepeated) {
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
inline std::vector<double> unpack_vector(const std::vector<double> &slots,
										 PackFormat f, const Layout &layout,
										 int n) {
	std::vector<double> out(static_cast<size_t>(n), 0.0);
	for (int k = 0; k < n; ++k) {
		size_t idx = (f == PackFormat::kRepeated)
						 ? static_cast<size_t>(k)				  // row 0
						 : static_cast<size_t>(k) * layout.cols;  // column 0
		out[static_cast<size_t>(k)] = slots[idx];
	}
	return out;
}

// Pack a weight matrix `w` of shape (in_features x out_features), given in
// row-major order, into the layout.
//
//   row_packing  : pad to (rows x cols) and flatten            -> W_row
//   !row_packing : pad to (cols x rows), transpose and flatten -> W_col
//
// Both encodings end up as a `rows x cols` slot matrix, so the same slot-level
// primitives serve both.
inline std::vector<double> pack_weights(const std::vector<double> &w,
										int in_features, int out_features,
										bool row_packing,
										const Layout &layout) {
	if (w.size() != static_cast<size_t>(in_features) * out_features)
		throw std::invalid_argument("weight size does not match its shape");
	const int pad_rows = row_packing ? layout.rows : layout.cols;
	const int pad_cols = row_packing ? layout.cols : layout.rows;
	if (in_features > pad_rows || out_features > pad_cols)
		throw std::invalid_argument("weights do not fit into the layout");

	std::vector<double> out(static_cast<size_t>(layout.slots()), 0.0);
	for (int i = 0; i < in_features; ++i) {
		for (int j = 0; j < out_features; ++j) {
			const double value = w[static_cast<size_t>(i) * out_features + j];
			// (i, j) in the padded matrix; transposed when column-packed.
			const int r = row_packing ? i : j;
			const int c = row_packing ? j : i;
			out[static_cast<size_t>(r) * layout.cols + c] = value;
		}
	}
	return out;
}

// Inverse of pack_weights, used by the tests and by weight read-back.
inline std::vector<double> unpack_weights(const std::vector<double> &slots,
										  int in_features, int out_features,
										  bool row_packing,
										  const Layout &layout) {
	std::vector<double> out(static_cast<size_t>(in_features) * out_features,
							0.0);
	for (int i = 0; i < in_features; ++i) {
		for (int j = 0; j < out_features; ++j) {
			const int r = row_packing ? i : j;
			const int c = row_packing ? j : i;
			out[static_cast<size_t>(i) * out_features + j] =
				slots[static_cast<size_t>(r) * layout.cols + c];
		}
	}
	return out;
}

// Mask that keeps the first column of every row (slot index % cols == 0).
// Used by the column-summation primitive.
inline std::vector<double> first_column_mask(const Layout &layout) {
	std::vector<double> mask(static_cast<size_t>(layout.slots()), 0.0);
	for (int i = 0; i < layout.rows; ++i)
		mask[static_cast<size_t>(i) * layout.cols] = 1.0;
	return mask;
}

}  // namespace reboot

#endif	// REBOOT_LAYOUT_H_
