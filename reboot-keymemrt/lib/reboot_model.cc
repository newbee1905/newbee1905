// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/reboot_model.h"

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>

namespace reboot {
namespace {

struct LayerSpec {
	std::string name;
	int in_features = 0;
	int out_features = 0;
	bool row_packing = true;
};

// The layer list in build order: for every block its forward layer and its
// local classifier, then the head layer, then the output layer.  Packing
// alternates block by block, and a block's classifier takes the opposite
// packing of its forward layer because it consumes that layer's output.
std::vector<LayerSpec> layer_specs(const ModelConfig &config) {
	std::vector<LayerSpec> specs;
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
int compute_depth(const TensorGraph &graph) {
	std::vector<int> level(graph.size(), 0);
	int deepest = 0;
	for (ValueId id : graph.topological_order()) {
		const TensorValue &v = graph.value(id);
		int base = 0;
		for (ValueId in : v.inputs) base = std::max(base, level[in]);
		switch (v.op) {
			case TensorOp::kMatMul:
			case TensorOp::kMatMulT:
			case TensorOp::kOuter:
			case TensorOp::kHadamard:
			case TensorOp::kSquare:
				level[id] = base + 1;
				break;
			case TensorOp::kPolyRelu:
			case TensorOp::kPolyReluGrad:
				level[id] = base + 1;
				break;
			case TensorOp::kBootstrap:
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

Layout recommend_layout(const ModelConfig &config, int num_slots) {
	int rows = 1, cols = 1;
	for (const LayerSpec &spec : layer_specs(config)) {
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
	return Layout(rows, cols);
}

std::string TrainStep::describe() const {
	std::string out = fmt::format(
		"ReBoot training step\n"
		"  layout          : {} ({} slots)\n"
		"  batch size      : {}\n"
		"  trainable params: {}\n"
		"  graph nodes     : {}\n"
		"  required depth  : {} levels\n",
		layout.str(), layout.slots(), config.batch_size, params.size(),
		graph.size(), required_depth);
	for (const ParamBinding &p : params) {
		const TensorValue &w = graph.value(p.weight);
		out += fmt::format("  {:<10} {:<10} {}\n", p.name, w.shape.str(),
						   w.row_packing ? "row-packed" : "column-packed");
	}
	return out;
}

TrainStep build_train_step(const ModelConfig &config, const Layout &layout) {
	if (config.batch_size < 1)
		throw std::invalid_argument("batch size must be positive");

	TrainStep step;
	step.layout = layout;
	step.config = config;
	TensorGraph &g = step.graph;

	const std::vector<LayerSpec> specs = layer_specs(config);
	const int num_blocks =
		config.hidden.empty() ? 0 : static_cast<int>(config.hidden.size()) - 1;

	// ---- parameters --------------------------------------------------------
	// Weights and velocities are both encrypted state that persists across
	// steps, so both are arguments and both are returned.
	std::map<std::string, size_t> param_index;
	for (const LayerSpec &spec : specs) {
		ParamBinding binding;
		binding.name = spec.name;
		binding.weight = g.param(spec.name, spec.in_features, spec.out_features,
								 spec.row_packing);
		binding.velocity =
			g.param(fmt::format("v_{}", spec.name), spec.in_features,
					spec.out_features, spec.row_packing);
		param_index[spec.name] = step.params.size();
		step.params.push_back(binding);
	}
	for (const ParamBinding &p : step.params) {
		step.arguments.push_back(p.weight);
		step.arguments.push_back(p.velocity);
	}

	// ---- inputs ------------------------------------------------------------
	const Format first_format = input_format(specs.front().row_packing);
	std::vector<ValueId> x(config.batch_size);
	for (int b = 0; b < config.batch_size; ++b) {
		x[b] = g.input(fmt::format("x_{}", b), config.input_dim, first_format);
		step.arguments.push_back(x[b]);
	}

	// Labels are encrypted once per batch in each packing a classifier needs.
	// The reference implementation re-encrypts the plaintext labels inside
	// every block, which a non-interactive server could not do.
	std::map<std::pair<int, int>, ValueId> labels;
	auto label = [&](int sample, Format format) {
		const std::pair<int, int> key{sample, static_cast<int>(format)};
		auto it = labels.find(key);
		if (it != labels.end()) return it->second;
		const ValueId id =
			g.input(fmt::format("y_{}_{}", format_name(format), sample),
					config.num_classes, format);
		labels.emplace(key, id);
		step.arguments.push_back(id);
		return id;
	};

	// ---- forward pass and local losses -------------------------------------
	std::vector<GradientSeed> seeds;
	for (int b = 0; b < config.batch_size; ++b) {
		ValueId h = x[b];
		for (int i = 0; i < num_blocks; ++i) {
			const ParamBinding &fwd =
				step.params[param_index.at(fmt::format("w_fwd_{}", i))];
			const ParamBinding &lrn =
				step.params[param_index.at(fmt::format("w_lrn_{}", i))];

			h = g.poly_relu(g.matmul(h, fwd.weight));
			const ValueId local = g.matmul(h, lrn.weight);
			const ValueId local_label = label(b, g.value(local).format);
			seeds.push_back({local, rss_seed(g, local, local_label)});
			step.losses.push_back(
				{fmt::format("block_{}_sample_{}", i, b), local, local_label});
			// The block's error signal stops here: the next block sees the
			// activation but no gradient flows back through it.
			h = g.stop_gradient(h);
		}
		if (!config.hidden.empty()) {
			const ParamBinding &head = step.params[param_index.at("w_head")];
			h = g.poly_relu(g.matmul(h, head.weight));
		}
		const ParamBinding &out = step.params[param_index.at("w_out")];
		const ValueId prediction = g.matmul(h, out.weight);
		g.set_name(prediction, fmt::format("y_hat_{}", b));
		step.predictions.push_back(prediction);
		const ValueId top_label = label(b, g.value(prediction).format);
		seeds.push_back({prediction, rss_seed(g, prediction, top_label)});
		step.losses.push_back(
			{fmt::format("output_sample_{}", b), prediction, top_label});
	}

	// ---- differentiate -----------------------------------------------------
	const GradientMap grads = backward(g, seeds);

	// ---- Nesterov update and refresh ---------------------------------------
	const double lr = config.learning_rate;
	const double m = config.momentum;
	for (ParamBinding &p : step.params) {
		auto it = grads.find(p.weight);
		if (it == grads.end())
			throw std::runtime_error(
				fmt::format("no gradient reached parameter {}", p.name));
		p.gradient = it->second;

		ValueId grad = p.gradient;
		if (config.weight_decay != 0.0)
			grad = g.add(grad, g.scale(p.weight, config.weight_decay));

		// update = lr*g + m*lr*g + m^2*lr*v   (all public scalars, no depth)
		const ValueId update =
			g.add(g.add(g.scale(grad, lr), g.scale(grad, m * lr)),
				  g.scale(p.velocity, m * m * lr));
		ValueId weight_out = g.sub(p.weight, update);
		ValueId velocity_out = g.add(g.scale(p.velocity, m), grad);

		if (config.bootstrap) {
			weight_out = g.bootstrap(weight_out);
			velocity_out = g.bootstrap(velocity_out);
		}
		g.set_name(weight_out, fmt::format("{}_next", p.name));
		g.set_name(velocity_out, fmt::format("v_{}_next", p.name));
		p.weight_out = weight_out;
		p.velocity_out = velocity_out;
	}

	for (const ParamBinding &p : step.params) {
		step.results.push_back(p.weight_out);
		step.results.push_back(p.velocity_out);
	}
	for (ValueId prediction : step.predictions)
		step.results.push_back(prediction);

	step.required_depth = compute_depth(g);
	return step;
}

}  // namespace reboot
