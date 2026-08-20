// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/tensor_graph.h"

#include <fmt/format.h>

#include <numeric>
#include <stdexcept>

namespace reboot {

const char *tensor_op_name(tensor_op_t op) {
	switch (op) {
		case tensor_op_t::input:
			return "input";
		case tensor_op_t::param:
			return "param";
		case tensor_op_t::matmul:
			return "matmul";
		case tensor_op_t::matmul_transposed:
			return "matmul_transposed";
		case tensor_op_t::outer:
			return "outer";
		case tensor_op_t::add:
			return "add";
		case tensor_op_t::sub:
			return "sub";
		case tensor_op_t::hadamard:
			return "hadamard";
		case tensor_op_t::scale:
			return "scale";
		case tensor_op_t::add_scalar:
			return "add_scalar";
		case tensor_op_t::square:
			return "square";
		case tensor_op_t::poly_relu:
			return "poly_relu";
		case tensor_op_t::poly_relu_grad:
			return "poly_relu_grad";
		case tensor_op_t::stop_gradient:
			return "stop_gradient";
		case tensor_op_t::bootstrap:
			return "bootstrap";
	}
	return "unknown";
}

std::string shape_t::str() const {
	return is_vector() ? fmt::format("[{}]", cols)
					   : fmt::format("[{}x{}]", rows, cols);
}

value_id_t tensor_graph_t::push(tensor_value_t v) {
	v.id = static_cast<value_id_t>(values_.size());
	values_.push_back(std::move(v));
	return values_.back().id;
}

void tensor_graph_t::set_name(value_id_t id, const std::string &name) {
	values_.at(id).name = name;
}

value_id_t tensor_graph_t::input(const std::string &name, int length,
								 pack_format_t format) {
	tensor_value_t v;
	v.op = tensor_op_t::input;
	v.shape = shape_t{1, length};
	v.name = name;
	v.format = format;
	return push(std::move(v));
}

value_id_t tensor_graph_t::param(const std::string &name, int rows, int cols,
								 bool row_packing) {
	tensor_value_t v;
	v.op = tensor_op_t::param;
	v.shape = shape_t{rows, cols};
	v.name = name;
	v.row_packing = row_packing;
	return push(std::move(v));
}

value_id_t tensor_graph_t::matmul(value_id_t x, value_id_t w) {
	const tensor_value_t &xv = value(x);
	const tensor_value_t &wv = value(w);
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

	tensor_value_t v;
	v.op = tensor_op_t::matmul;
	v.shape = shape_t{1, wv.shape.cols};
	v.inputs = {x, w};
	v.format = output_format(wv.row_packing);
	return push(std::move(v));
}

value_id_t tensor_graph_t::matmul_transposed(value_id_t g, value_id_t w) {
	const tensor_value_t &gv = value(g);
	const tensor_value_t &wv = value(w);
	if (!gv.shape.is_vector() || wv.shape.is_vector())
		throw std::invalid_argument(
			"matmul_transposed expects a vector and a matrix");
	if (gv.shape.cols != wv.shape.cols)
		throw std::invalid_argument(
			fmt::format("matmul_transposed shape mismatch: {} . {}^T",
						gv.shape.str(), wv.shape.str()));
	if (gv.format != output_format(wv.row_packing))
		throw std::invalid_argument("matmul_transposed format mismatch");

	tensor_value_t v;
	v.op = tensor_op_t::matmul_transposed;
	v.shape = shape_t{1, wv.shape.rows};
	v.inputs = {g, w};
	v.format = input_format(wv.row_packing);
	return push(std::move(v));
}

value_id_t tensor_graph_t::outer(value_id_t x, value_id_t g) {
	const tensor_value_t &xv = value(x);
	const tensor_value_t &gv = value(g);
	if (!xv.shape.is_vector() || !gv.shape.is_vector())
		throw std::invalid_argument("outer expects two vectors");
	if (xv.format == gv.format)
		throw std::invalid_argument(
			"outer needs one Expanded and one Repeated operand; that is what "
			"makes the elementwise product land in the weight layout");

	tensor_value_t v;
	v.op = tensor_op_t::outer;
	v.shape = shape_t{xv.shape.cols, gv.shape.cols};
	v.inputs = {x, g};
	// x Expanded (down the rows) and g Repeated (along the row) is exactly the
	// row-packed weight layout; the mirrored case is the column-packed one.
	v.row_packing = xv.format == pack_format_t::expanded;
	return push(std::move(v));
}

namespace {

void require_same(const tensor_value_t &a, const tensor_value_t &b,
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

value_id_t tensor_graph_t::add(value_id_t a, value_id_t b) {
	require_same(value(a), value(b), "add");
	tensor_value_t v = value(a);
	v.op = tensor_op_t::add;
	v.inputs = {a, b};
	v.name.clear();
	return push(std::move(v));
}

value_id_t tensor_graph_t::sub(value_id_t a, value_id_t b) {
	require_same(value(a), value(b), "sub");
	tensor_value_t v = value(a);
	v.op = tensor_op_t::sub;
	v.inputs = {a, b};
	v.name.clear();
	return push(std::move(v));
}

value_id_t tensor_graph_t::hadamard(value_id_t a, value_id_t b) {
	require_same(value(a), value(b), "hadamard");
	tensor_value_t v = value(a);
	v.op = tensor_op_t::hadamard;
	v.inputs = {a, b};
	v.name.clear();
	return push(std::move(v));
}

value_id_t tensor_graph_t::scale(value_id_t a, double s) {
	tensor_value_t v = value(a);
	v.op = tensor_op_t::scale;
	v.inputs = {a};
	v.scalar = s;
	v.name.clear();
	return push(std::move(v));
}

value_id_t tensor_graph_t::add_scalar(value_id_t a, double s) {
	tensor_value_t v = value(a);
	v.op = tensor_op_t::add_scalar;
	v.inputs = {a};
	v.scalar = s;
	v.name.clear();
	return push(std::move(v));
}

value_id_t tensor_graph_t::square(value_id_t a) {
	tensor_value_t v = value(a);
	v.op = tensor_op_t::square;
	v.inputs = {a};
	v.name.clear();
	return push(std::move(v));
}

value_id_t tensor_graph_t::poly_relu(value_id_t a) {
	tensor_value_t v = value(a);
	v.op = tensor_op_t::poly_relu;
	v.inputs = {a};
	v.name.clear();
	return push(std::move(v));
}

value_id_t tensor_graph_t::poly_relu_grad(value_id_t g, value_id_t x) {
	require_same(value(g), value(x), "poly_relu_grad");
	tensor_value_t v = value(g);
	v.op = tensor_op_t::poly_relu_grad;
	v.inputs = {g, x};
	v.name.clear();
	return push(std::move(v));
}

value_id_t tensor_graph_t::stop_gradient(value_id_t a) {
	tensor_value_t v = value(a);
	v.op = tensor_op_t::stop_gradient;
	v.inputs = {a};
	v.name.clear();
	return push(std::move(v));
}

value_id_t tensor_graph_t::bootstrap(value_id_t a) {
	tensor_value_t v = value(a);
	v.op = tensor_op_t::bootstrap;
	v.inputs = {a};
	v.name.clear();
	return push(std::move(v));
}

std::vector<value_id_t> tensor_graph_t::topological_order() const {
	std::vector<value_id_t> order(values_.size());
	std::iota(order.begin(), order.end(), 0);
	return order;
}

std::string tensor_graph_t::dump() const {
	std::string out;
	for (const tensor_value_t &v : values_) {
		out += fmt::format("  %{:<4} = {:<15} {}", v.id, tensor_op_name(v.op),
						   v.shape.str());
		for (value_id_t in : v.inputs) out += fmt::format(" %{}", in);
		if (v.op == tensor_op_t::scale || v.op == tensor_op_t::add_scalar)
			out += fmt::format(" ({})", v.scalar);
		if (!v.name.empty()) out += fmt::format("  // {}", v.name);
		out += "\n";
	}
	return out;
}

}  // namespace reboot
