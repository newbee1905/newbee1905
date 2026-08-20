// SPDX-License-Identifier: GPL-3.0-or-later
//
// Run the emitted training step on plaintext slots.
//
// This executes the same graph `reboot_emit` prints as MLIR - same packing,
// same rotations, same differentiated update - with the ciphertexts replaced by
// slot vectors.  If the network learns here, the emitted program is the
// training step it claims to be; what the FHE backend then adds is
// approximation noise and cost, not different arithmetic.

#include <fmt/format.h>

#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "reboot/data.h"
#include "reboot/interpreter.h"
#include "reboot/layout.h"
#include "reboot/options.h"
#include "reboot/reboot_model.h"
#include "reboot/slot_graph.h"

using namespace reboot;

namespace {

int argmax(const dense_value_t &v) {
	int best = 0;
	for (size_t i = 1; i < v.size(); ++i)
		if (v[i] > v[static_cast<size_t>(best)]) best = static_cast<int>(i);
	return best;
}

}  // namespace

int main(int argc, char **argv) {
	model_config_t config;
	config.hidden = {32, 16};
	config.input_dim = 16;
	config.num_classes = 4;
	config.batch_size = 4;
	config.learning_rate = 0.01;

	int samples = 512, epochs = 4, log_n = 13;
	unsigned seed_value = 1;
	int seed = 1;
	std::string csv;

	option_parser_t parser(
		"reboot_eval",
		"run the emitted ReBoot training step on plaintext slots");
	add_model_options(parser, config, log_n);
	parser.section("Data")
		.add("--samples", samples, "synthetic samples")
		.add("--csv", csv, "CSV dataset instead (last column = label)");
	parser.section("Run")
		.add("--epochs", epochs, "epochs")
		.add("--seed", seed, "RNG seed");

	try {
		if (!parser.parse(argc, argv)) return 0;
		seed_value = static_cast<unsigned>(seed);

		dataset_t data = csv.empty()
							 ? make_blobs(samples, config.input_dim,
										  config.num_classes, /*seed=*/5)
							 : load_csv(csv);
		if (!csv.empty()) {
			normalise(data);
			config.input_dim = data.dim;
			config.num_classes = data.num_classes;
		}

		const int num_slots = 1 << (log_n - 1);
		const layout_t layout = recommend_layout(config, num_slots);
		const train_step_t step = build_train_step(config, layout);
		const lowered_step_t lowered = lower_to_slots(step);
		const tensor_graph_t &g = step.graph;

		fmt::print("{}", step.describe());
		fmt::print("{}", lowered.graph.statistics());
		fmt::print("dataset: {} samples, dim {}, {} classes\n\n", data.size(),
				   data.dim, data.num_classes);

		// Argument lookup by the name the model gave it.
		std::map<std::string, size_t> argument_index;
		for (size_t i = 0; i < lowered.argument_names.size(); ++i)
			argument_index[lowered.argument_names[i]] = i;

		// Encrypted state: weights initialised Xavier-uniform, velocities zero.
		std::mt19937 rng(seed_value);
		std::map<std::string, dense_value_t> state;
		for (const param_binding_t &p : step.params) {
			const tensor_value_t &meta = g.value(p.weight);
			const double bound =
				std::sqrt(1.0 / (meta.shape.rows + meta.shape.cols));
			std::uniform_real_distribution<double> dist(-bound, bound);
			dense_value_t weights(static_cast<size_t>(meta.shape.rows) *
								  meta.shape.cols);
			for (double &w : weights) w = dist(rng);
			state[p.name] =
				pack_weights(weights, meta.shape.rows, meta.shape.cols,
							 meta.row_packing, layout);
			state[fmt::format("v_{}", p.name)] =
				dense_value_t(static_cast<size_t>(layout.slots()), 0.0);
		}

		const pack_format_t prediction_format =
			g.value(step.predictions.front()).format;
		std::mt19937 shuffle_rng(seed_value);

		for (int epoch = 0; epoch < epochs; ++epoch) {
			shuffle(data, shuffle_rng);
			double epoch_loss = 0.0;
			int correct = 0, seen = 0;

			for (size_t start = 0;
				 start + static_cast<size_t>(config.batch_size) <= data.size();
				 start += static_cast<size_t>(config.batch_size)) {
				slot_inputs_t inputs;
				for (const auto &[name, value] : state)
					inputs[lowered.arguments[argument_index.at(name)]] = value;

				for (int b = 0; b < config.batch_size; ++b) {
					const size_t index = start + static_cast<size_t>(b);
					const dense_value_t one_hot_label =
						one_hot(data.y[index], config.num_classes);
					const std::string x_name = fmt::format("x_{}", b);
					inputs[lowered.arguments[argument_index.at(x_name)]] =
						pack_vector(
							data.features[index],
							g.value(step.arguments[argument_index.at(x_name)])
								.format,
							layout);
					for (pack_format_t format :
						 {pack_format_t::repeated, pack_format_t::expanded}) {
						const std::string y_name =
							fmt::format("y_{}_{}", format_name(format), b);
						auto it = argument_index.find(y_name);
						if (it == argument_index.end()) continue;
						inputs[lowered.arguments[it->second]] =
							pack_vector(one_hot_label, format, layout);
					}
				}

				const std::vector<dense_value_t> values =
					evaluate(lowered.graph, inputs);

				// The step returns the refreshed state; feed it back in.
				for (size_t i = 0; i < step.params.size(); ++i) {
					const param_binding_t &p = step.params[i];
					state[p.name] = values[lowered.results[2 * i]];
					state[fmt::format("v_{}", p.name)] =
						values[lowered.results[2 * i + 1]];
				}

				const size_t prediction_base = 2 * step.params.size();
				for (int b = 0; b < config.batch_size; ++b) {
					const dense_value_t scores = unpack_vector(
						values[lowered.results[prediction_base +
											   static_cast<size_t>(b)]],
						prediction_format, layout, config.num_classes);
					const int label = data.y[start + static_cast<size_t>(b)];
					for (int c = 0; c < config.num_classes; ++c) {
						const double target = (c == label) ? 1.0 : 0.0;
						const double diff =
							scores[static_cast<size_t>(c)] - target;
						epoch_loss += diff * diff;
					}
					if (argmax(scores) == label) ++correct;
					++seen;
				}
			}
			fmt::print("epoch {:2d} | loss {:8.4f} | accuracy {:5.1f}%\n",
					   epoch, epoch_loss / std::max(1, seen),
					   100.0 * correct / std::max(1, seen));
		}
	} catch (const std::exception &error) {
		fmt::print(stderr, "error: {}\n", error.what());
		return 1;
	}
	return 0;
}
