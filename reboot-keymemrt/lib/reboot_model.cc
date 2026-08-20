// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/reboot_model.h"

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>

namespace reboot {
namespace {

struct layer_spec_t {
	std::string name;
	int in_features = 0;
	int out_features = 0;
	bool row_packing = true;
};

// The layer list in build order: for every block its forward layer and its
// local classifier, then the head layer, then the output layer.  Packing
// alternates block by block, and a block's classifier takes the opposite
// packing of its forward layer because it consumes that layer's output.
std::vector<layer_spec_t> layer_specs(const model_config_t &config) {
	std::vector<layer_spec_t> specs;
	const int num_blocks =
		config.hidden.empty() ? 0 : static_cast<int>(config.hidden.size()) - 1;
	bool row_packing = true;

	for (int i = 0; i < num_blocks; ++i) {
		const int in = (i == 0) ? config.input_dim : config.hidden[i - 1];
		specs.push_back(
			{fmt::format("w_fwd_{}", i), in, config.hidden[i], row_packing});
		specs.push_back({fmt::format("w_lrn_{}", i), config.hidden[i],
						 config.num_classes, !row_packing});
		row_packing = !row_packing;
	}
	if (!config.hidden.empty()) {
		const int in = config.hidden.size() == 1
						   ? config.input_dim
						   : config.hidden[config.hidden.size() - 2];
		specs.push_back({"w_head", in, config.hidden.back(), row_packing});
		row_packing = !row_packing;
	}
	const int last_in =
		config.hidden.empty() ? config.input_dim : config.hidden.back();
	specs.push_back({"w_out", last_in, config.num_classes,
					 config.hidden.empty() ? true : row_packing});
	return specs;
}

// Level accounting mirroring CKKS: products cost a level, additions take the
// deeper of their operands, bootstrapping resets to zero.  Scalar
// multiplication is free because ckks.mul_scalar does not rescale.
int compute_depth(const tensor_graph_t &graph) {
	std::vector<int> level(graph.size(), 0);
	int deepest = 0;
	for (value_id_t id : graph.topological_order()) {
		const tensor_value_t &v = graph.value(id);
		int base = 0;
		for (value_id_t in : v.inputs) base = std::max(base, level[in]);
		switch (v.op) {
			case tensor_op_t::matmul:
			case tensor_op_t::matmul_transposed:
			case tensor_op_t::outer:
			case tensor_op_t::hadamard:
			case tensor_op_t::square:
				level[id] = base + 1;
				break;
			case tensor_op_t::poly_relu:
			case tensor_op_t::poly_relu_grad:
				level[id] = base + 1;
				break;
			case tensor_op_t::bootstrap:
				level[id] = 0;
				break;
			default:
				level[id] = base;
				break;
		}
		deepest = std::max(deepest, level[id]);
	}
	// A column-packed product spends one extra level on the masking step of the
	// column summation; count it once so the modulus chain is not too short.
	return deepest + 1;
}

}  // namespace

layout_t recommend_layout(const model_config_t &config, int num_slots) {
	int rows = 1, cols = 1;
	for (const layer_spec_t &spec : layer_specs(config)) {
		if (spec.row_packing) {
			rows = std::max(rows, spec.in_features);
			cols = std::max(cols, spec.out_features);
		} else {
			cols = std::max(cols, spec.in_features);
			rows = std::max(rows, spec.out_features);
		}
	}
	rows = next_power_of_two(rows);
	cols = next_power_of_two(cols);
	if (rows * cols > num_slots)
		throw std::invalid_argument(
			fmt::format("network needs {} slots but the ring provides {}",
						rows * cols, num_slots));
	rows = num_slots / cols;
	return layout_t(rows, cols);
}

std::string train_step_t::describe() const {
	std::string out = fmt::format(
		"ReBoot training step\n"
		"  layout          : {} ({} slots)\n"
		"  batch size      : {}\n"
		"  trainable params: {}\n"
		"  graph nodes     : {}\n"
		"  required depth  : {} levels\n",
		layout.str(), layout.slots(), config.batch_size, params.size(),
		graph.size(), required_depth);
	for (const param_binding_t &p : params) {
		const tensor_value_t &w = graph.value(p.weight);
		out += fmt::format("  {:<10} {:<10} {}\n", p.name, w.shape.str(),
						   w.row_packing ? "row-packed" : "column-packed");
	}
	return out;
}

train_step_t build_train_step(const model_config_t &config,
							  const layout_t &layout) {
	if (config.batch_size < 1)
		throw std::invalid_argument("batch size must be positive");

	train_step_t step;
	step.layout = layout;
	step.config = config;
	tensor_graph_t &g = step.graph;

	const std::vector<layer_spec_t> specs = layer_specs(config);
	const int num_blocks =
		config.hidden.empty() ? 0 : static_cast<int>(config.hidden.size()) - 1;

	// ---- parameters --------------------------------------------------------
	// Weights and velocities are both encrypted state that persists across
	// steps, so both are arguments and both are returned.
	std::map<std::string, size_t> param_index;
	for (const layer_spec_t &spec : specs) {
		param_binding_t binding;
		binding.name = spec.name;
		binding.weight = g.param(spec.name, spec.in_features, spec.out_features,
								 spec.row_packing);
		binding.velocity =
			g.param(fmt::format("v_{}", spec.name), spec.in_features,
					spec.out_features, spec.row_packing);
		param_index[spec.name] = step.params.size();
		step.params.push_back(binding);
	}
	for (const param_binding_t &p : step.params) {
		step.arguments.push_back(p.weight);
		step.arguments.push_back(p.velocity);
	}

	// ---- inputs ------------------------------------------------------------
	const pack_format_t first_format = input_format(specs.front().row_packing);
	std::vector<value_id_t> x(config.batch_size);
	for (int b = 0; b < config.batch_size; ++b) {
		x[b] = g.input(fmt::format("x_{}", b), config.input_dim, first_format);
		step.arguments.push_back(x[b]);
	}

	// Labels are encrypted once per batch in each packing a classifier needs.
	// The reference implementation re-encrypts the plaintext labels inside
	// every block, which a non-interactive server could not do.
	std::map<std::pair<int, int>, value_id_t> labels;
	auto label = [&](int sample, pack_format_t format) {
		const std::pair<int, int> key{sample, static_cast<int>(format)};
		auto it = labels.find(key);
		if (it != labels.end()) return it->second;
		const value_id_t id =
			g.input(fmt::format("y_{}_{}", format_name(format), sample),
					config.num_classes, format);
		labels.emplace(key, id);
		step.arguments.push_back(id);
		return id;
	};

	// ---- forward pass and local losses -------------------------------------
	std::vector<gradient_seed_t> seeds;
	for (int b = 0; b < config.batch_size; ++b) {
		value_id_t h = x[b];
		for (int i = 0; i < num_blocks; ++i) {
			const param_binding_t &fwd =
				step.params[param_index.at(fmt::format("w_fwd_{}", i))];
			const param_binding_t &lrn =
				step.params[param_index.at(fmt::format("w_lrn_{}", i))];

			h = g.poly_relu(g.matmul(h, fwd.weight));
			const value_id_t local = g.matmul(h, lrn.weight);
			const value_id_t local_label = label(b, g.value(local).format);
			seeds.push_back({local, rss_seed(g, local, local_label)});
			step.losses.push_back(
				{fmt::format("block_{}_sample_{}", i, b), local, local_label});
			// The block's error signal stops here: the next block sees the
			// activation but no gradient flows back through it.
			h = g.stop_gradient(h);
		}
		if (!config.hidden.empty()) {
			const param_binding_t &head = step.params[param_index.at("w_head")];
			h = g.poly_relu(g.matmul(h, head.weight));
		}
		const param_binding_t &out = step.params[param_index.at("w_out")];
		const value_id_t prediction = g.matmul(h, out.weight);
		g.set_name(prediction, fmt::format("y_hat_{}", b));
		step.predictions.push_back(prediction);
		const value_id_t top_label = label(b, g.value(prediction).format);
		seeds.push_back({prediction, rss_seed(g, prediction, top_label)});
		step.losses.push_back(
			{fmt::format("output_sample_{}", b), prediction, top_label});
	}

	// ---- differentiate -----------------------------------------------------
	const gradient_map_t grads = backward(g, seeds);

	// ---- Nesterov update and refresh ---------------------------------------
	const double lr = config.learning_rate;
	const double m = config.momentum;
	for (param_binding_t &p : step.params) {
		auto it = grads.find(p.weight);
		if (it == grads.end())
			throw std::runtime_error(
				fmt::format("no gradient reached parameter {}", p.name));
		p.gradient = it->second;

		value_id_t grad = p.gradient;
		if (config.weight_decay != 0.0)
			grad = g.add(grad, g.scale(p.weight, config.weight_decay));

		// update = lr*g + m*lr*g + m^2*lr*v   (all public scalars, no depth)
		const value_id_t update =
			g.add(g.add(g.scale(grad, lr), g.scale(grad, m * lr)),
				  g.scale(p.velocity, m * m * lr));
		value_id_t weight_out = g.sub(p.weight, update);
		value_id_t velocity_out = g.add(g.scale(p.velocity, m), grad);

		if (config.bootstrap) {
			weight_out = g.bootstrap(weight_out);
			velocity_out = g.bootstrap(velocity_out);
		}
		g.set_name(weight_out, fmt::format("{}_next", p.name));
		g.set_name(velocity_out, fmt::format("v_{}_next", p.name));
		p.weight_out = weight_out;
		p.velocity_out = velocity_out;
	}

	for (const param_binding_t &p : step.params) {
		step.results.push_back(p.weight_out);
		step.results.push_back(p.velocity_out);
	}
	for (value_id_t prediction : step.predictions)
		step.results.push_back(prediction);

	step.required_depth = compute_depth(g);
	return step;
}

}  // namespace reboot
