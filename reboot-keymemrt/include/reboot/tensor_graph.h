// SPDX-License-Identifier: GPL-3.0-or-later
//
// The tensor-level graph that ReBoot is built on and differentiated in.
//
// Nodes carry logical shapes (a vector of length n, or an n x m weight matrix)
// rather than slot vectors, so reverse-mode differentiation is ordinary tensor
// calculus.  Each node also carries the packing it will occupy once lowered:
// a row-packed weight matrix consumes an Expanded vector and produces a
// Repeated one, a column-packed matrix does the opposite.  Recording the
// packing here means the lowering to slot operations is a local rewrite and the
// builder can reject a network whose packings do not alternate.
//
// The interesting consequence is that ReBoot's hand-derived backward pass falls
// out of the standard rules: the vector-Jacobian product of `matmul` is
// `matmul_transposed`, which in this packing is the same elementwise product summed in
// the other direction - the same weight ciphertext, no transposition, no
// repacking.

#ifndef REBOOT_TENSOR_GRAPH_H_
#define REBOOT_TENSOR_GRAPH_H_

#include <string>
#include <vector>

#include "reboot/layout.h"

namespace reboot {

using value_id_t = int;
inline constexpr value_id_t no_value = -1;

enum class tensor_op_t {
	input,	 // encrypted activation or label supplied by the client
	param,	 // trainable weight matrix (encrypted, persists across steps)
	matmul,	 // (1 x n) . (n x m) -> (1 x m)
	matmul_transposed,	// (1 x m) . (n x m)^T -> (1 x n)
	outer,				// (1 x n) (x) (1 x m) -> (n x m)
	add,				// elementwise, same shape
	sub,				// elementwise, same shape
	hadamard,			// elementwise product, same shape
	scale,				// multiply by a public scalar
	add_scalar,			// add a public scalar to every entry
	square,				// elementwise square
	poly_relu,			// x^2 + x, the degree-two ReLU surrogate ReBoot uses
	poly_relu_grad,		// g * (2x + 1), the VJP of poly_relu kept as one node
	stop_gradient,		// block boundary: values pass forward, gradients do not
	bootstrap,			// refresh a ciphertext that survives into the next step
};

const char *tensor_op_name(tensor_op_t op);

// A logical shape.  Vectors are 1 x cols; weight matrices are rows x cols.
struct shape_t {
	int rows = 1;
	int cols = 0;

	bool is_vector() const { return rows == 1; }
	bool operator==(const shape_t &other) const {
		return rows == other.rows && cols == other.cols;
	}
	bool operator!=(const shape_t &other) const { return !(*this == other); }
	std::string str() const;
};

struct tensor_value_t {
	value_id_t id = no_value;
	tensor_op_t op = tensor_op_t::input;
	shape_t shape;
	std::vector<value_id_t> inputs;
	double scalar = 0.0;  // scale, add_scalar
	std::string name;	  // leaves and anything worth reading in a dump

	// Packing: `format` applies to vectors, `row_packing` to matrices.
	pack_format_t format = pack_format_t::repeated;
	bool row_packing = true;
};

class tensor_graph_t {
   public:
	// ---- leaves ------------------------------------------------------------
	value_id_t input(const std::string &name, int length, pack_format_t format);
	value_id_t param(const std::string &name, int rows, int cols,
					 bool row_packing);

	// ---- linear algebra ----------------------------------------------------
	// x is a vector whose format must match the weight packing.
	value_id_t matmul(value_id_t x, value_id_t w);
	// Gradient towards the input of a matmul: the same weights, summed the
	// other way round.
	value_id_t matmul_transposed(value_id_t g, value_id_t w);
	// Gradient towards the weights of a matmul.
	value_id_t outer(value_id_t x, value_id_t g);

	// ---- elementwise -------------------------------------------------------
	value_id_t add(value_id_t a, value_id_t b);
	value_id_t sub(value_id_t a, value_id_t b);
	value_id_t hadamard(value_id_t a, value_id_t b);
	value_id_t scale(value_id_t a, double s);
	value_id_t add_scalar(value_id_t a, double s);
	value_id_t square(value_id_t a);
	value_id_t poly_relu(value_id_t a);
	value_id_t poly_relu_grad(value_id_t g, value_id_t x);

	// ---- structure ---------------------------------------------------------
	value_id_t stop_gradient(value_id_t a);
	value_id_t bootstrap(value_id_t a);

	// ---- accessors ---------------------------------------------------------
	const tensor_value_t &value(value_id_t id) const { return values_.at(id); }
	const std::vector<tensor_value_t> &values() const { return values_; }
	size_t size() const { return values_.size(); }
	void set_name(value_id_t id, const std::string &name);

	// Values in dependency order (the graph is built acyclically, so index
	// order already is one; this makes the intent explicit).
	std::vector<value_id_t> topological_order() const;

	std::string dump() const;

   private:
	value_id_t push(tensor_value_t v);

	std::vector<tensor_value_t> values_;
};

}  // namespace reboot

#endif	// REBOOT_TENSOR_GRAPH_H_
