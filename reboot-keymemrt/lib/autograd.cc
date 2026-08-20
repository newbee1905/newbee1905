// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/autograd.h"

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>

namespace reboot {

ValueId rss_seed(TensorGraph &graph, ValueId prediction, ValueId target) {
	return graph.sub(prediction, target);
}

namespace {

// Adjoints accumulate: a value feeding several consumers sums their
// contributions, which is what makes a shared weight collect the gradient of
// every sample in the batch.
void accumulate(TensorGraph &graph, GradientMap &grads, ValueId value,
				ValueId contribution) {
	auto it = grads.find(value);
	if (it == grads.end()) {
		grads.emplace(value, contribution);
	} else {
		it->second = graph.add(it->second, contribution);
	}
}

}  // namespace

GradientMap backward(TensorGraph &graph,
					 const std::vector<GradientSeed> &seeds) {
	GradientMap grads;
	for (const GradientSeed &seed : seeds) {
		if (seed.value == kNoValue || seed.seed == kNoValue)
			throw std::invalid_argument("backward: incomplete gradient seed");
		accumulate(graph, grads, seed.value, seed.seed);
	}

	// The graph is built in dependency order, so walking indices downwards
	// visits every consumer before its producers.  New nodes created while
	// walking are gradient nodes with larger ids; they are never differentiated
	// again, so the bound is captured up front.
	const ValueId last = static_cast<ValueId>(graph.size()) - 1;
	for (ValueId id = last; id >= 0; --id) {
		auto it = grads.find(id);
		if (it == grads.end()) continue;
		const ValueId g = it->second;
		// Copy: the vector backing the graph reallocates as nodes are added.
		const TensorValue node = graph.value(id);

		switch (node.op) {
			case TensorOp::kInput:
			case TensorOp::kParam:
			case TensorOp::kStopGradient:
				// Leaves and block boundaries absorb the adjoint.
				break;

			case TensorOp::kMatMul: {
				const ValueId x = node.inputs[0];
				const ValueId w = node.inputs[1];
				accumulate(graph, grads, x, graph.matmul_t(g, w));
				accumulate(graph, grads, w, graph.outer(x, g));
				break;
			}

			case TensorOp::kMatMulT: {
				const ValueId y = node.inputs[0];
				const ValueId w = node.inputs[1];
				accumulate(graph, grads, y, graph.matmul(g, w));
				accumulate(graph, grads, w, graph.outer(g, y));
				break;
			}

			case TensorOp::kOuter: {
				const ValueId a = node.inputs[0];
				const ValueId b = node.inputs[1];
				accumulate(graph, grads, a, graph.matmul_t(b, g));
				accumulate(graph, grads, b, graph.matmul(a, g));
				break;
			}

			case TensorOp::kAdd:
				accumulate(graph, grads, node.inputs[0], g);
				accumulate(graph, grads, node.inputs[1], g);
				break;

			case TensorOp::kSub:
				accumulate(graph, grads, node.inputs[0], g);
				accumulate(graph, grads, node.inputs[1], graph.scale(g, -1.0));
				break;

			case TensorOp::kHadamard:
				accumulate(graph, grads, node.inputs[0],
						   graph.hadamard(g, node.inputs[1]));
				accumulate(graph, grads, node.inputs[1],
						   graph.hadamard(g, node.inputs[0]));
				break;

			case TensorOp::kScale:
				accumulate(graph, grads, node.inputs[0],
						   graph.scale(g, node.scalar));
				break;

			case TensorOp::kAddScalar:
				accumulate(graph, grads, node.inputs[0], g);
				break;

			case TensorOp::kSquare:
				accumulate(graph, grads, node.inputs[0],
						   graph.hadamard(g, graph.scale(node.inputs[0], 2.0)));
				break;

			case TensorOp::kPolyRelu:
				// One node rather than the composition of square and add, so
				// the backward pass costs a single multiplicative level, as in
				// the paper.
				accumulate(graph, grads, node.inputs[0],
						   graph.poly_relu_grad(g, node.inputs[0]));
				break;

			case TensorOp::kPolyReluGrad:
				throw std::invalid_argument(
					"poly_relu_grad appears in a forward graph; it is only "
					"produced by differentiation and is not differentiable "
					"again");

			case TensorOp::kBootstrap:
				// Bootstrapping is the identity on the message.
				accumulate(graph, grads, node.inputs[0], g);
				break;
		}
	}
	return grads;
}

}  // namespace reboot
