// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/interpreter.h"

#include <fmt/format.h>

#include <stdexcept>

namespace reboot {
namespace {

std::vector<double> zip(const std::vector<double> &a,
						const std::vector<double> &b, double sign) {
	if (a.size() != b.size())
		throw std::runtime_error("elementwise operands differ in size");
	std::vector<double> out(a.size());
	for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + sign * b[i];
	return out;
}

std::vector<double> product(const std::vector<double> &a,
							const std::vector<double> &b) {
	if (a.size() != b.size())
		throw std::runtime_error("elementwise operands differ in size");
	std::vector<double> out(a.size());
	for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] * b[i];
	return out;
}

}  // namespace

std::vector<std::vector<double>> evaluate(const tensor_graph_t &graph,
										  const tensor_inputs_t &inputs) {
	std::vector<std::vector<double>> values(graph.size());

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
				const std::vector<double> &x = values[v.inputs[0]];
				const std::vector<double> &w = values[v.inputs[1]];
				const tensor_value_t &wv = graph.value(v.inputs[1]);
				std::vector<double> out(static_cast<size_t>(wv.shape.cols),
										0.0);
				for (int j = 0; j < wv.shape.cols; ++j)
					for (int i = 0; i < wv.shape.rows; ++i)
						out[static_cast<size_t>(j)] +=
							x[static_cast<size_t>(i)] *
							w[static_cast<size_t>(i) * wv.shape.cols + j];
				values[id] = std::move(out);
				break;
			}

			case tensor_op_t::matmul_transposed: {
				const std::vector<double> &g = values[v.inputs[0]];
				const std::vector<double> &w = values[v.inputs[1]];
				const tensor_value_t &wv = graph.value(v.inputs[1]);
				std::vector<double> out(static_cast<size_t>(wv.shape.rows),
										0.0);
				for (int i = 0; i < wv.shape.rows; ++i)
					for (int j = 0; j < wv.shape.cols; ++j)
						out[static_cast<size_t>(i)] +=
							w[static_cast<size_t>(i) * wv.shape.cols + j] *
							g[static_cast<size_t>(j)];
				values[id] = std::move(out);
				break;
			}

			case tensor_op_t::outer: {
				const std::vector<double> &a = values[v.inputs[0]];
				const std::vector<double> &b = values[v.inputs[1]];
				std::vector<double> out(a.size() * b.size(), 0.0);
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
				std::vector<double> out = values[v.inputs[0]];
				for (double &x : out) x *= v.scalar;
				values[id] = std::move(out);
				break;
			}
			case tensor_op_t::add_scalar: {
				std::vector<double> out = values[v.inputs[0]];
				for (double &x : out) x += v.scalar;
				values[id] = std::move(out);
				break;
			}
			case tensor_op_t::square:
				values[id] = product(values[v.inputs[0]], values[v.inputs[0]]);
				break;
			case tensor_op_t::poly_relu: {
				std::vector<double> out = values[v.inputs[0]];
				for (double &x : out) x = x * x + x;
				values[id] = std::move(out);
				break;
			}
			case tensor_op_t::poly_relu_grad: {
				const std::vector<double> &g = values[v.inputs[0]];
				const std::vector<double> &x = values[v.inputs[1]];
				std::vector<double> out(g.size());
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

std::vector<std::vector<double>> evaluate(const slot_graph_t &graph,
										  const slot_inputs_t &inputs) {
	const int slots = graph.layout().slots();
	std::vector<std::vector<double>> values(graph.size());

	for (const slot_value_t &v : graph.values()) {
		switch (v.op) {
			case slot_op_t::argument: {
				auto it = inputs.find(v.id);
				if (it == inputs.end())
					throw std::runtime_error(fmt::format(
						"no value supplied for argument '{}'", v.name));
				std::vector<double> value = it->second;
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
				std::vector<double> out = values[v.inputs[0]];
				for (double &x : out) x *= v.scalar;
				values[v.id] = std::move(out);
				break;
			}
			case slot_op_t::rotate: {
				const std::vector<double> &in = values[v.inputs[0]];
				std::vector<double> out(in.size());
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
