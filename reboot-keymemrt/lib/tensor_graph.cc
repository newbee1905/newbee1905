// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/tensor_graph.h"

#include <fmt/format.h>

#include <numeric>
#include <stdexcept>

namespace reboot {

const char *tensor_op_name(TensorOp op) {
	switch (op) {
		case TensorOp::kInput:
			return "input";
		case TensorOp::kParam:
			return "param";
		case TensorOp::kMatMul:
			return "matmul";
		case TensorOp::kMatMulT:
			return "matmul_t";
		case TensorOp::kOuter:
			return "outer";
		case TensorOp::kAdd:
			return "add";
		case TensorOp::kSub:
			return "sub";
		case TensorOp::kHadamard:
			return "hadamard";
		case TensorOp::kScale:
			return "scale";
		case TensorOp::kAddScalar:
			return "add_scalar";
		case TensorOp::kSquare:
			return "square";
		case TensorOp::kPolyRelu:
			return "poly_relu";
		case TensorOp::kPolyReluGrad:
			return "poly_relu_grad";
		case TensorOp::kStopGradient:
			return "stop_gradient";
		case TensorOp::kBootstrap:
			return "bootstrap";
	}
	return "unknown";
}

std::string Shape::str() const {
	return is_vector() ? fmt::format("[{}]", cols)
					   : fmt::format("[{}x{}]", rows, cols);
}

ValueId TensorGraph::push(TensorValue v) {
	v.id = static_cast<ValueId>(values_.size());
	values_.push_back(std::move(v));
	return values_.back().id;
}

void TensorGraph::set_name(ValueId id, const std::string &name) {
	values_.at(id).name = name;
}

ValueId TensorGraph::input(const std::string &name, int length, Format format) {
	TensorValue v;
	v.op = TensorOp::kInput;
	v.shape = Shape{1, length};
	v.name = name;
	v.format = format;
	return push(std::move(v));
}

ValueId TensorGraph::param(const std::string &name, int rows, int cols,
						   bool row_packing) {
	TensorValue v;
	v.op = TensorOp::kParam;
	v.shape = Shape{rows, cols};
	v.name = name;
	v.row_packing = row_packing;
	return push(std::move(v));
}

ValueId TensorGraph::matmul(ValueId x, ValueId w) {
	const TensorValue &xv = value(x);
	const TensorValue &wv = value(w);
	if (!xv.shape.is_vector() || wv.shape.is_vector())
		throw std::invalid_argument("matmul expects a vector and a matrix");
	if (xv.shape.cols != wv.shape.rows)
		throw std::invalid_argument(fmt::format(
			"matmul shape mismatch: {} . {}", xv.shape.str(), wv.shape.str()));
	// A row-packed weight matrix consumes an Expanded vector and produces a
	// Repeated one; a column-packed matrix does the opposite.
	if (xv.format != input_format(wv.row_packing))
		throw std::invalid_argument(
			fmt::format("matmul format mismatch: {} is {} but {} needs {}",
						xv.name, format_name(xv.format), wv.name,
						format_name(input_format(wv.row_packing))));

	TensorValue v;
	v.op = TensorOp::kMatMul;
	v.shape = Shape{1, wv.shape.cols};
	v.inputs = {x, w};
	v.format = output_format(wv.row_packing);
	return push(std::move(v));
}

ValueId TensorGraph::matmul_t(ValueId g, ValueId w) {
	const TensorValue &gv = value(g);
	const TensorValue &wv = value(w);
	if (!gv.shape.is_vector() || wv.shape.is_vector())
		throw std::invalid_argument("matmul_t expects a vector and a matrix");
	if (gv.shape.cols != wv.shape.cols)
		throw std::invalid_argument(
			fmt::format("matmul_t shape mismatch: {} . {}^T", gv.shape.str(),
						wv.shape.str()));
	if (gv.format != output_format(wv.row_packing))
		throw std::invalid_argument("matmul_t format mismatch");

	TensorValue v;
	v.op = TensorOp::kMatMulT;
	v.shape = Shape{1, wv.shape.rows};
	v.inputs = {g, w};
	v.format = input_format(wv.row_packing);
	return push(std::move(v));
}

ValueId TensorGraph::outer(ValueId x, ValueId g) {
	const TensorValue &xv = value(x);
	const TensorValue &gv = value(g);
	if (!xv.shape.is_vector() || !gv.shape.is_vector())
		throw std::invalid_argument("outer expects two vectors");
	if (xv.format == gv.format)
		throw std::invalid_argument(
			"outer needs one Expanded and one Repeated operand; that is what "
			"makes the elementwise product land in the weight layout");

	TensorValue v;
	v.op = TensorOp::kOuter;
	v.shape = Shape{xv.shape.cols, gv.shape.cols};
	v.inputs = {x, g};
	// x Expanded (down the rows) and g Repeated (along the row) is exactly the
	// row-packed weight layout; the mirrored case is the column-packed one.
	v.row_packing = xv.format == Format::kExpanded;
	return push(std::move(v));
}

namespace {

void require_same(const TensorValue &a, const TensorValue &b,
				  const char *what) {
	if (a.shape != b.shape)
		throw std::invalid_argument(fmt::format("{}: shape {} vs {}", what,
												a.shape.str(), b.shape.str()));
	if (a.shape.is_vector() && a.format != b.format)
		throw std::invalid_argument(fmt::format("{}: format {} vs {}", what,
												format_name(a.format),
												format_name(b.format)));
	if (!a.shape.is_vector() && a.row_packing != b.row_packing)
		throw std::invalid_argument(fmt::format("{}: packing mismatch", what));
}

}  // namespace

ValueId TensorGraph::add(ValueId a, ValueId b) {
	require_same(value(a), value(b), "add");
	TensorValue v = value(a);
	v.op = TensorOp::kAdd;
	v.inputs = {a, b};
	v.name.clear();
	return push(std::move(v));
}

ValueId TensorGraph::sub(ValueId a, ValueId b) {
	require_same(value(a), value(b), "sub");
	TensorValue v = value(a);
	v.op = TensorOp::kSub;
	v.inputs = {a, b};
	v.name.clear();
	return push(std::move(v));
}

ValueId TensorGraph::hadamard(ValueId a, ValueId b) {
	require_same(value(a), value(b), "hadamard");
	TensorValue v = value(a);
	v.op = TensorOp::kHadamard;
	v.inputs = {a, b};
	v.name.clear();
	return push(std::move(v));
}

ValueId TensorGraph::scale(ValueId a, double s) {
	TensorValue v = value(a);
	v.op = TensorOp::kScale;
	v.inputs = {a};
	v.scalar = s;
	v.name.clear();
	return push(std::move(v));
}

ValueId TensorGraph::add_scalar(ValueId a, double s) {
	TensorValue v = value(a);
	v.op = TensorOp::kAddScalar;
	v.inputs = {a};
	v.scalar = s;
	v.name.clear();
	return push(std::move(v));
}

ValueId TensorGraph::square(ValueId a) {
	TensorValue v = value(a);
	v.op = TensorOp::kSquare;
	v.inputs = {a};
	v.name.clear();
	return push(std::move(v));
}

ValueId TensorGraph::poly_relu(ValueId a) {
	TensorValue v = value(a);
	v.op = TensorOp::kPolyRelu;
	v.inputs = {a};
	v.name.clear();
	return push(std::move(v));
}

ValueId TensorGraph::poly_relu_grad(ValueId g, ValueId x) {
	require_same(value(g), value(x), "poly_relu_grad");
	TensorValue v = value(g);
	v.op = TensorOp::kPolyReluGrad;
	v.inputs = {g, x};
	v.name.clear();
	return push(std::move(v));
}

ValueId TensorGraph::stop_gradient(ValueId a) {
	TensorValue v = value(a);
	v.op = TensorOp::kStopGradient;
	v.inputs = {a};
	v.name.clear();
	return push(std::move(v));
}

ValueId TensorGraph::bootstrap(ValueId a) {
	TensorValue v = value(a);
	v.op = TensorOp::kBootstrap;
	v.inputs = {a};
	v.name.clear();
	return push(std::move(v));
}

std::vector<ValueId> TensorGraph::topological_order() const {
	std::vector<ValueId> order(values_.size());
	std::iota(order.begin(), order.end(), 0);
	return order;
}

std::string TensorGraph::dump() const {
	std::string out;
	for (const TensorValue &v : values_) {
		out += fmt::format("  %{:<4} = {:<15} {}", v.id, tensor_op_name(v.op),
						   v.shape.str());
		for (ValueId in : v.inputs) out += fmt::format(" %{}", in);
		if (v.op == TensorOp::kScale || v.op == TensorOp::kAddScalar)
			out += fmt::format(" ({})", v.scalar);
		if (!v.name.empty()) out += fmt::format("  // {}", v.name);
		out += "\n";
	}
	return out;
}

}  // namespace reboot
