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
// `matmul_t`, which in this packing is the same elementwise product summed in
// the other direction - the same weight ciphertext, no transposition, no
// repacking.

#ifndef REBOOT_TENSOR_GRAPH_H_
#define REBOOT_TENSOR_GRAPH_H_

#include <string>
#include <vector>

#include "reboot/layout.h"

namespace reboot {

using ValueId = int;
inline constexpr ValueId kNoValue = -1;

enum class TensorOp {
	kInput,		 // encrypted activation or label supplied by the client
	kParam,		 // trainable weight matrix (encrypted, persists across steps)
	kMatMul,	 // (1 x n) . (n x m) -> (1 x m)
	kMatMulT,	 // (1 x m) . (n x m)^T -> (1 x n)
	kOuter,		 // (1 x n) (x) (1 x m) -> (n x m)
	kAdd,		 // elementwise, same shape
	kSub,		 // elementwise, same shape
	kHadamard,	 // elementwise product, same shape
	kScale,		 // multiply by a public scalar
	kAddScalar,	 // add a public scalar to every entry
	kSquare,	 // elementwise square
	kPolyRelu,	 // x^2 + x, the degree-two ReLU surrogate ReBoot uses
	kPolyReluGrad,	// g * (2x + 1), the VJP of kPolyRelu kept as one node
	kStopGradient,	// block boundary: values pass forward, gradients do not
	kBootstrap,		// refresh a ciphertext that survives into the next step
};

const char *tensor_op_name(TensorOp op);

// A logical shape.  Vectors are 1 x cols; weight matrices are rows x cols.
struct Shape {
	int rows = 1;
	int cols = 0;

	bool is_vector() const { return rows == 1; }
	bool operator==(const Shape &other) const {
		return rows == other.rows && cols == other.cols;
	}
	bool operator!=(const Shape &other) const { return !(*this == other); }
	std::string str() const;
};

struct TensorValue {
	ValueId id = kNoValue;
	TensorOp op = TensorOp::kInput;
	Shape shape;
	std::vector<ValueId> inputs;
	double scalar = 0.0;  // kScale, kAddScalar
	std::string name;	  // leaves and anything worth reading in a dump

	// Packing: `format` applies to vectors, `row_packing` to matrices.
	PackFormat format = PackFormat::kRepeated;
	bool row_packing = true;
};

class TensorGraph {
   public:
	// ---- leaves ------------------------------------------------------------
	ValueId input(const std::string &name, int length, PackFormat format);
	ValueId param(const std::string &name, int rows, int cols,
				  bool row_packing);

	// ---- linear algebra ----------------------------------------------------
	// x is a vector whose format must match the weight packing.
	ValueId matmul(ValueId x, ValueId w);
	// Gradient towards the input of a matmul: the same weights, summed the
	// other way round.
	ValueId matmul_t(ValueId g, ValueId w);
	// Gradient towards the weights of a matmul.
	ValueId outer(ValueId x, ValueId g);

	// ---- elementwise -------------------------------------------------------
	ValueId add(ValueId a, ValueId b);
	ValueId sub(ValueId a, ValueId b);
	ValueId hadamard(ValueId a, ValueId b);
	ValueId scale(ValueId a, double s);
	ValueId add_scalar(ValueId a, double s);
	ValueId square(ValueId a);
	ValueId poly_relu(ValueId a);
	ValueId poly_relu_grad(ValueId g, ValueId x);

	// ---- structure ---------------------------------------------------------
	ValueId stop_gradient(ValueId a);
	ValueId bootstrap(ValueId a);

	// ---- accessors ---------------------------------------------------------
	const TensorValue &value(ValueId id) const { return values_.at(id); }
	const std::vector<TensorValue> &values() const { return values_; }
	size_t size() const { return values_.size(); }
	void set_name(ValueId id, const std::string &name);

	// Values in dependency order (the graph is built acyclically, so index
	// order already is one; this makes the intent explicit).
	std::vector<ValueId> topological_order() const;

	std::string dump() const;

   private:
	ValueId push(TensorValue v);

	std::vector<TensorValue> values_;
};

}  // namespace reboot

#endif	// REBOOT_TENSOR_GRAPH_H_
