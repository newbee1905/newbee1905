// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/interpreter.h"

#include <fmt/format.h>

#include <stdexcept>

namespace reboot {
namespace {

dense_value_t zip(const dense_value_t &a, const dense_value_t &b, double sign) {
	if (a.size() != b.size())
		throw std::runtime_error("elementwise operands differ in size");
	dense_value_t out(a.size());
	for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + sign * b[i];
	return out;
}

dense_value_t product(const dense_value_t &a, const dense_value_t &b) {
	if (a.size() != b.size())
		throw std::runtime_error("elementwise operands differ in size");
	dense_value_t out(a.size());
	for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] * b[i];
	return out;
}

}  // namespace

std::vector<dense_value_t> evaluate(const tensor_graph_t &graph,
									const tensor_inputs_t &inputs) {
	std::vector<dense_value_t> values(graph.size());

	for (value_id_t id : graph.topological_order()) {
		const tensor_value_t &v = graph.value(id);
		switch (v.op) {
			case tensor_op_t::input:
			case tensor_op_t::param: {
				auto it = inputs.find(id);
				if (it == inputs.end())
					throw std::runtime_error(
						fmt::format("no value supplied for leaf '{}'", v.name));
				const size_t expected =
					static_cast<size_t>(v.shape.rows) * v.shape.cols;
				if (it->second.size() != expected)
					throw std::runtime_error(
						fmt::format("leaf '{}' expects {} entries, got {}",
									v.name, expected, it->second.size()));
				values[id] = it->second;
				break;
			}

			case tensor_op_t::matmul: {
				const dense_value_t &x = values[v.inputs[0]];
				const dense_value_t &w = values[v.inputs[1]];
				const tensor_value_t &wv = graph.value(v.inputs[1]);
				dense_value_t out(static_cast<size_t>(wv.shape.cols), 0.0);
				for (int j = 0; j < wv.shape.cols; ++j)
					for (int i = 0; i < wv.shape.rows; ++i)
						out[static_cast<size_t>(j)] +=
							x[static_cast<size_t>(i)] *
							w[static_cast<size_t>(i) * wv.shape.cols + j];
				values[id] = std::move(out);
				break;
			}

			case tensor_op_t::matmul_transposed: {
				const dense_value_t &g = values[v.inputs[0]];
				const dense_value_t &w = values[v.inputs[1]];
				const tensor_value_t &wv = graph.value(v.inputs[1]);
				dense_value_t out(static_cast<size_t>(wv.shape.rows), 0.0);
				for (int i = 0; i < wv.shape.rows; ++i)
					for (int j = 0; j < wv.shape.cols; ++j)
						out[static_cast<size_t>(i)] +=
							w[static_cast<size_t>(i) * wv.shape.cols + j] *
							g[static_cast<size_t>(j)];
				values[id] = std::move(out);
				break;
			}

			case tensor_op_t::outer: {
				const dense_value_t &a = values[v.inputs[0]];
				const dense_value_t &b = values[v.inputs[1]];
				dense_value_t out(a.size() * b.size(), 0.0);
				for (size_t i = 0; i < a.size(); ++i)
					for (size_t j = 0; j < b.size(); ++j)
						out[i * b.size() + j] = a[i] * b[j];
				values[id] = std::move(out);
				break;
			}

			case tensor_op_t::add:
				values[id] = zip(values[v.inputs[0]], values[v.inputs[1]], 1.0);
				break;
			case tensor_op_t::sub:
				values[id] =
					zip(values[v.inputs[0]], values[v.inputs[1]], -1.0);
				break;
			case tensor_op_t::hadamard:
				values[id] = product(values[v.inputs[0]], values[v.inputs[1]]);
				break;
			case tensor_op_t::scale: {
				dense_value_t out = values[v.inputs[0]];
				for (double &x : out) x *= v.scalar;
				values[id] = std::move(out);
				break;
			}
			case tensor_op_t::add_scalar: {
				dense_value_t out = values[v.inputs[0]];
				for (double &x : out) x += v.scalar;
				values[id] = std::move(out);
				break;
			}
			case tensor_op_t::square:
				values[id] = product(values[v.inputs[0]], values[v.inputs[0]]);
				break;
			case tensor_op_t::poly_relu: {
				dense_value_t out = values[v.inputs[0]];
				for (double &x : out) x = x * x + x;
				values[id] = std::move(out);
				break;
			}
			case tensor_op_t::poly_relu_grad: {
				const dense_value_t &g = values[v.inputs[0]];
				const dense_value_t &x = values[v.inputs[1]];
				dense_value_t out(g.size());
				for (size_t i = 0; i < g.size(); ++i)
					out[i] = g[i] * (2.0 * x[i] + 1.0);
				values[id] = std::move(out);
				break;
			}
			case tensor_op_t::stop_gradient:
			case tensor_op_t::bootstrap:
				values[id] = values[v.inputs[0]];
				break;
		}
	}
	return values;
}

std::vector<dense_value_t> evaluate(const slot_graph_t &graph,
									const slot_inputs_t &inputs) {
	const int slots = graph.layout().slots();
	std::vector<dense_value_t> values(graph.size());

	for (const slot_value_t &v : graph.values()) {
		switch (v.op) {
			case slot_op_t::argument: {
				auto it = inputs.find(v.id);
				if (it == inputs.end())
					throw std::runtime_error(fmt::format(
						"no value supplied for argument '{}'", v.name));
				dense_value_t value = it->second;
				value.resize(static_cast<size_t>(slots), 0.0);
				values[v.id] = std::move(value);
				break;
			}
			case slot_op_t::add:
				values[v.id] =
					zip(values[v.inputs[0]], values[v.inputs[1]], 1.0);
				break;
			case slot_op_t::sub:
				values[v.id] =
					zip(values[v.inputs[0]], values[v.inputs[1]], -1.0);
				break;
			case slot_op_t::mul:
				values[v.id] =
					product(values[v.inputs[0]], values[v.inputs[1]]);
				break;
			case slot_op_t::mul_plain:
				values[v.id] = product(values[v.inputs[0]],
									   graph.constants().at(v.constant).values);
				break;
			case slot_op_t::add_plain:
				values[v.id] =
					zip(values[v.inputs[0]],
						graph.constants().at(v.constant).values, 1.0);
				break;
			case slot_op_t::mul_scalar: {
				dense_value_t out = values[v.inputs[0]];
				for (double &x : out) x *= v.scalar;
				values[v.id] = std::move(out);
				break;
			}
			case slot_op_t::rotate: {
				const dense_value_t &in = values[v.inputs[0]];
				dense_value_t out(in.size());
				const int n = static_cast<int>(in.size());
				const int shift = ((v.rotation % n) + n) % n;
				for (int i = 0; i < n; ++i)
					out[static_cast<size_t>(i)] =
						in[static_cast<size_t>((i + shift) % n)];
				values[v.id] = std::move(out);
				break;
			}
			case slot_op_t::bootstrap:
				values[v.id] = values[v.inputs[0]];
				break;
		}
	}
	return values;
}

}  // namespace reboot
