// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/autograd.h"

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>

namespace reboot {

value_id_t rss_seed(tensor_graph_t &graph, value_id_t prediction,
					value_id_t target) {
	return graph.sub(prediction, target);
}

namespace {

// Adjoints accumulate: a value feeding several consumers sums their
// contributions, which is what makes a shared weight collect the gradient of
// every sample in the batch.
void accumulate(tensor_graph_t &graph, gradient_map_t &grads, value_id_t value,
				value_id_t contribution) {
	auto it = grads.find(value);
	if (it == grads.end()) {
		grads.emplace(value, contribution);
	} else {
		it->second = graph.add(it->second, contribution);
	}
}

}  // namespace

gradient_map_t backward(tensor_graph_t &graph,
						const std::vector<gradient_seed_t> &seeds) {
	gradient_map_t grads;
	for (const gradient_seed_t &seed : seeds) {
		if (seed.value == no_value || seed.seed == no_value)
			throw std::invalid_argument("backward: incomplete gradient seed");
		accumulate(graph, grads, seed.value, seed.seed);
	}

	// The graph is built in dependency order, so walking indices downwards
	// visits every consumer before its producers.  New nodes created while
	// walking are gradient nodes with larger ids; they are never differentiated
	// again, so the bound is captured up front.
	const value_id_t last = static_cast<value_id_t>(graph.size()) - 1;
	for (value_id_t id = last; id >= 0; --id) {
		auto it = grads.find(id);
		if (it == grads.end()) continue;
		const value_id_t g = it->second;
		// Copy: the vector backing the graph reallocates as nodes are added.
		const tensor_value_t node = graph.value(id);

		switch (node.op) {
			case tensor_op_t::input:
			case tensor_op_t::param:
			case tensor_op_t::stop_gradient:
				// Leaves and block boundaries absorb the adjoint.
				break;

			case tensor_op_t::matmul: {
				const value_id_t x = node.inputs[0];
				const value_id_t w = node.inputs[1];
				accumulate(graph, grads, x, graph.matmul_transposed(g, w));
				accumulate(graph, grads, w, graph.outer(x, g));
				break;
			}

			case tensor_op_t::matmul_transposed: {
				const value_id_t y = node.inputs[0];
				const value_id_t w = node.inputs[1];
				accumulate(graph, grads, y, graph.matmul(g, w));
				accumulate(graph, grads, w, graph.outer(g, y));
				break;
			}

			case tensor_op_t::outer: {
				const value_id_t a = node.inputs[0];
				const value_id_t b = node.inputs[1];
				accumulate(graph, grads, a, graph.matmul_transposed(b, g));
				accumulate(graph, grads, b, graph.matmul(a, g));
				break;
			}

			case tensor_op_t::add:
				accumulate(graph, grads, node.inputs[0], g);
				accumulate(graph, grads, node.inputs[1], g);
				break;

			case tensor_op_t::sub:
				accumulate(graph, grads, node.inputs[0], g);
				accumulate(graph, grads, node.inputs[1], graph.scale(g, -1.0));
				break;

			case tensor_op_t::hadamard:
				accumulate(graph, grads, node.inputs[0],
						   graph.hadamard(g, node.inputs[1]));
				accumulate(graph, grads, node.inputs[1],
						   graph.hadamard(g, node.inputs[0]));
				break;

			case tensor_op_t::scale:
				accumulate(graph, grads, node.inputs[0],
						   graph.scale(g, node.scalar));
				break;

			case tensor_op_t::add_scalar:
				accumulate(graph, grads, node.inputs[0], g);
				break;

			case tensor_op_t::square:
				accumulate(graph, grads, node.inputs[0],
						   graph.hadamard(g, graph.scale(node.inputs[0], 2.0)));
				break;

			case tensor_op_t::poly_relu:
				// One node rather than the composition of square and add, so
				// the backward pass costs a single multiplicative level, as in
				// the paper.
				accumulate(graph, grads, node.inputs[0],
						   graph.poly_relu_grad(g, node.inputs[0]));
				break;

			case tensor_op_t::poly_relu_grad:
				throw std::invalid_argument(
					"poly_relu_grad appears in a forward graph; it is only "
					"produced by differentiation and is not differentiable "
					"again");

			case tensor_op_t::bootstrap:
				// Bootstrapping is the identity on the message.
				accumulate(graph, grads, node.inputs[0], g);
				break;
		}
	}
	return grads;
}

}  // namespace reboot
