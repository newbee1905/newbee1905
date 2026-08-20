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
#include "reboot/reboot_model.h"
#include "reboot/slot_graph.h"

using namespace reboot;

namespace {

std::vector<int> parse_int_list(const std::string &text) {
	std::vector<int> out;
	size_t pos = 0;
	while (pos < text.size()) {
		size_t comma = text.find(',', pos);
		if (comma == std::string::npos) comma = text.size();
		out.push_back(std::stoi(text.substr(pos, comma - pos)));
		pos = comma + 1;
	}
	return out;
}

void print_help() {
	fmt::print(
		"reboot_eval - run the emitted ReBoot training step on plaintext "
		"slots\n\n"
		"  --hidden a,b        hidden layer widths (default 32,16)\n"
		"  --classes n         number of classes (4)\n"
		"  --dim n             input dimension (16)\n"
		"  --samples n         synthetic samples (512)\n"
		"  --csv path          CSV dataset instead (last column = label)\n"
		"  --batch-size n      samples per step (4)\n"
		"  --epochs n          epochs (4)\n"
		"  --lr f              learning rate (0.01)\n"
		"  --momentum f        Nesterov momentum (0.9)\n"
		"  --weight-decay f    weight decay (0.0)\n"
		"  --log-n n           log2 of the ring dimension (13)\n"
		"  --seed n            RNG seed (1)\n");
}

int argmax(const DenseValue &v) {
	int best = 0;
	for (size_t i = 1; i < v.size(); ++i)
		if (v[i] > v[static_cast<size_t>(best)]) best = static_cast<int>(i);
	return best;
}

}  // namespace

int main(int argc, char **argv) {
	ModelConfig config;
	config.hidden = {32, 16};
	config.input_dim = 16;
	config.num_classes = 4;
	config.batch_size = 4;
	config.learning_rate = 0.01;

	int samples = 512, epochs = 4, log_n = 13;
	unsigned seed = 1;
	std::string csv;

	try {
		for (int i = 1; i < argc; ++i) {
			const std::string arg = argv[i];
			auto next = [&]() { return std::string(argv[++i]); };
			if (arg == "--hidden")
				config.hidden = parse_int_list(next());
			else if (arg == "--classes")
				config.num_classes = std::stoi(next());
			else if (arg == "--dim")
				config.input_dim = std::stoi(next());
			else if (arg == "--samples")
				samples = std::stoi(next());
			else if (arg == "--csv")
				csv = next();
			else if (arg == "--batch-size")
				config.batch_size = std::stoi(next());
			else if (arg == "--epochs")
				epochs = std::stoi(next());
			else if (arg == "--lr")
				config.learning_rate = std::stod(next());
			else if (arg == "--momentum")
				config.momentum = std::stod(next());
			else if (arg == "--weight-decay")
				config.weight_decay = std::stod(next());
			else if (arg == "--log-n")
				log_n = std::stoi(next());
			else if (arg == "--seed")
				seed = static_cast<unsigned>(std::stoi(next()));
			else if (arg == "--help" || arg == "-h") {
				print_help();
				return 0;
			} else {
				fmt::print(stderr, "unknown option {}\n", arg);
				return 1;
			}
		}

		Dataset data = csv.empty() ? make_blobs(samples, config.input_dim,
												config.num_classes, /*seed=*/5)
								   : load_csv(csv);
		if (!csv.empty()) {
			normalise(data);
			config.input_dim = data.dim;
			config.num_classes = data.num_classes;
		}

		const int num_slots = 1 << (log_n - 1);
		const Layout layout = recommend_layout(config, num_slots);
		const TrainStep step = build_train_step(config, layout);
		const LoweredStep lowered = lower_to_slots(step);
		const TensorGraph &g = step.graph;

		fmt::print("{}", step.describe());
		fmt::print("{}", lowered.graph.statistics());
		fmt::print("dataset: {} samples, dim {}, {} classes\n\n", data.size(),
				   data.dim, data.num_classes);

		// Argument lookup by the name the model gave it.
		std::map<std::string, size_t> argument_index;
		for (size_t i = 0; i < lowered.argument_names.size(); ++i)
			argument_index[lowered.argument_names[i]] = i;

		// Encrypted state: weights initialised Xavier-uniform, velocities zero.
		std::mt19937 rng(seed);
		std::map<std::string, DenseValue> state;
		for (const ParamBinding &p : step.params) {
			const TensorValue &meta = g.value(p.weight);
			const double bound =
				std::sqrt(1.0 / (meta.shape.rows + meta.shape.cols));
			std::uniform_real_distribution<double> dist(-bound, bound);
			DenseValue weights(static_cast<size_t>(meta.shape.rows) *
							   meta.shape.cols);
			for (double &w : weights) w = dist(rng);
			state[p.name] =
				pack_weights(weights, meta.shape.rows, meta.shape.cols,
							 meta.row_packing, layout);
			state[fmt::format("v_{}", p.name)] =
				DenseValue(static_cast<size_t>(layout.slots()), 0.0);
		}

		const PackFormat prediction_format =
			g.value(step.predictions.front()).format;
		std::mt19937 shuffle_rng(seed);

		for (int epoch = 0; epoch < epochs; ++epoch) {
			shuffle(data, shuffle_rng);
			double epoch_loss = 0.0;
			int correct = 0, seen = 0;

			for (size_t start = 0;
				 start + static_cast<size_t>(config.batch_size) <= data.size();
				 start += static_cast<size_t>(config.batch_size)) {
				SlotInputs inputs;
				for (const auto &[name, value] : state)
					inputs[lowered.arguments[argument_index.at(name)]] = value;

				for (int b = 0; b < config.batch_size; ++b) {
					const size_t index = start + static_cast<size_t>(b);
					const DenseValue one_hot_label =
						one_hot(data.y[index], config.num_classes);
					const std::string x_name = fmt::format("x_{}", b);
					inputs[lowered.arguments[argument_index.at(x_name)]] =
						pack_vector(
							data.features[index],
							g.value(step.arguments[argument_index.at(x_name)])
								.format,
							layout);
					for (PackFormat format :
						 {PackFormat::kRepeated, PackFormat::kExpanded}) {
						const std::string y_name =
							fmt::format("y_{}_{}", format_name(format), b);
						auto it = argument_index.find(y_name);
						if (it == argument_index.end()) continue;
						inputs[lowered.arguments[it->second]] =
							pack_vector(one_hot_label, format, layout);
					}
				}

				const std::vector<DenseValue> values =
					evaluate(lowered.graph, inputs);

				// The step returns the refreshed state; feed it back in.
				for (size_t i = 0; i < step.params.size(); ++i) {
					const ParamBinding &p = step.params[i];
					state[p.name] = values[lowered.results[2 * i]];
					state[fmt::format("v_{}", p.name)] =
						values[lowered.results[2 * i + 1]];
				}

				const size_t prediction_base = 2 * step.params.size();
				for (int b = 0; b < config.batch_size; ++b) {
					const DenseValue scores = unpack_vector(
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
