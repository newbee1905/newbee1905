// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/interpreter.h"

#include <fmt/format.h>

#include <stdexcept>

namespace reboot {
namespace {

DenseValue zip(const DenseValue &a, const DenseValue &b, double sign) {
	if (a.size() != b.size())
		throw std::runtime_error("elementwise operands differ in size");
	DenseValue out(a.size());
	for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + sign * b[i];
	return out;
}

DenseValue product(const DenseValue &a, const DenseValue &b) {
	if (a.size() != b.size())
		throw std::runtime_error("elementwise operands differ in size");
	DenseValue out(a.size());
	for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] * b[i];
	return out;
}

}  // namespace

std::vector<DenseValue> evaluate(const TensorGraph &graph,
								 const TensorInputs &inputs) {
	std::vector<DenseValue> values(graph.size());

	for (ValueId id : graph.topological_order()) {
		const TensorValue &v = graph.value(id);
		switch (v.op) {
			case TensorOp::kInput:
			case TensorOp::kParam: {
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

			case TensorOp::kMatMul: {
				const DenseValue &x = values[v.inputs[0]];
				const DenseValue &w = values[v.inputs[1]];
				const TensorValue &wv = graph.value(v.inputs[1]);
				DenseValue out(static_cast<size_t>(wv.shape.cols), 0.0);
				for (int j = 0; j < wv.shape.cols; ++j)
					for (int i = 0; i < wv.shape.rows; ++i)
						out[static_cast<size_t>(j)] +=
							x[static_cast<size_t>(i)] *
							w[static_cast<size_t>(i) * wv.shape.cols + j];
				values[id] = std::move(out);
				break;
			}

			case TensorOp::kMatMulT: {
				const DenseValue &g = values[v.inputs[0]];
				const DenseValue &w = values[v.inputs[1]];
				const TensorValue &wv = graph.value(v.inputs[1]);
				DenseValue out(static_cast<size_t>(wv.shape.rows), 0.0);
				for (int i = 0; i < wv.shape.rows; ++i)
					for (int j = 0; j < wv.shape.cols; ++j)
						out[static_cast<size_t>(i)] +=
							w[static_cast<size_t>(i) * wv.shape.cols + j] *
							g[static_cast<size_t>(j)];
				values[id] = std::move(out);
				break;
			}

			case TensorOp::kOuter: {
				const DenseValue &a = values[v.inputs[0]];
				const DenseValue &b = values[v.inputs[1]];
				DenseValue out(a.size() * b.size(), 0.0);
				for (size_t i = 0; i < a.size(); ++i)
					for (size_t j = 0; j < b.size(); ++j)
						out[i * b.size() + j] = a[i] * b[j];
				values[id] = std::move(out);
				break;
			}

			case TensorOp::kAdd:
				values[id] = zip(values[v.inputs[0]], values[v.inputs[1]], 1.0);
				break;
			case TensorOp::kSub:
				values[id] =
					zip(values[v.inputs[0]], values[v.inputs[1]], -1.0);
				break;
			case TensorOp::kHadamard:
				values[id] = product(values[v.inputs[0]], values[v.inputs[1]]);
				break;
			case TensorOp::kScale: {
				DenseValue out = values[v.inputs[0]];
				for (double &x : out) x *= v.scalar;
				values[id] = std::move(out);
				break;
			}
			case TensorOp::kAddScalar: {
				DenseValue out = values[v.inputs[0]];
				for (double &x : out) x += v.scalar;
				values[id] = std::move(out);
				break;
			}
			case TensorOp::kSquare:
				values[id] = product(values[v.inputs[0]], values[v.inputs[0]]);
				break;
			case TensorOp::kPolyRelu: {
				DenseValue out = values[v.inputs[0]];
				for (double &x : out) x = x * x + x;
				values[id] = std::move(out);
				break;
			}
			case TensorOp::kPolyReluGrad: {
				const DenseValue &g = values[v.inputs[0]];
				const DenseValue &x = values[v.inputs[1]];
				DenseValue out(g.size());
				for (size_t i = 0; i < g.size(); ++i)
					out[i] = g[i] * (2.0 * x[i] + 1.0);
				values[id] = std::move(out);
				break;
			}
			case TensorOp::kStopGradient:
			case TensorOp::kBootstrap:
				values[id] = values[v.inputs[0]];
				break;
		}
	}
	return values;
}

std::vector<DenseValue> evaluate(const SlotGraph &graph,
								 const SlotInputs &inputs) {
	const int slots = graph.layout().slots();
	std::vector<DenseValue> values(graph.size());

	for (const SlotValue &v : graph.values()) {
		switch (v.op) {
			case SlotOp::kArgument: {
				auto it = inputs.find(v.id);
				if (it == inputs.end())
					throw std::runtime_error(fmt::format(
						"no value supplied for argument '{}'", v.name));
				DenseValue value = it->second;
				value.resize(static_cast<size_t>(slots), 0.0);
				values[v.id] = std::move(value);
				break;
			}
			case SlotOp::kAdd:
				values[v.id] =
					zip(values[v.inputs[0]], values[v.inputs[1]], 1.0);
				break;
			case SlotOp::kSub:
				values[v.id] =
					zip(values[v.inputs[0]], values[v.inputs[1]], -1.0);
				break;
			case SlotOp::kMul:
				values[v.id] =
					product(values[v.inputs[0]], values[v.inputs[1]]);
				break;
			case SlotOp::kMulPlain:
				values[v.id] = product(values[v.inputs[0]],
									   graph.constants().at(v.constant).values);
				break;
			case SlotOp::kAddPlain:
				values[v.id] =
					zip(values[v.inputs[0]],
						graph.constants().at(v.constant).values, 1.0);
				break;
			case SlotOp::kMulScalar: {
				DenseValue out = values[v.inputs[0]];
				for (double &x : out) x *= v.scalar;
				values[v.id] = std::move(out);
				break;
			}
			case SlotOp::kRotate: {
				const DenseValue &in = values[v.inputs[0]];
				DenseValue out(in.size());
				const int n = static_cast<int>(in.size());
				const int shift = ((v.rotation % n) + n) % n;
				for (int i = 0; i < n; ++i)
					out[static_cast<size_t>(i)] =
						in[static_cast<size_t>((i + shift) % n)];
				values[v.id] = std::move(out);
				break;
			}
			case SlotOp::kBootstrap:
				values[v.id] = values[v.inputs[0]];
				break;
		}
	}
	return values;
}

}  // namespace reboot
