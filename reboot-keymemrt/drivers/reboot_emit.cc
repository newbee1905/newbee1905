// SPDX-License-Identifier: GPL-3.0-or-later
//
// Build a ReBoot network, differentiate its training step and print the
// resulting `ckks`-dialect MLIR for KeyMemRT-Compiler.

#include <fmt/format.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "reboot/ckks_params.h"
#include "reboot/mlir_emitter.h"
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
		"reboot_emit - emit a ReBoot training step as ckks-dialect MLIR\n\n"
		"Model:\n"
		"  --hidden a,b        hidden layer widths (default 32,16)\n"
		"  --input-dim n       input dimension (16)\n"
		"  --classes n         number of classes (4)\n"
		"  --batch-size n      samples per step (1)\n"
		"  --lr f              learning rate (0.005)\n"
		"  --momentum f        Nesterov momentum (0.9)\n"
		"  --weight-decay f    weight decay (0.0)\n"
		"  --no-bootstrap      leave the weights unrefreshed (inspection "
		"only)\n"
		"\nCKKS:\n"
		"  --log-n n           log2 of the ring dimension (13)\n"
		"  --log-scale n       log2 of the scaling factor (26)\n"
		"  --levels n          modulus chain length (default: from the graph)\n"
		"\nOutput:\n"
		"  --function name     emitted function name (reboot_train_step)\n"
		"  -o path             write MLIR here instead of stdout\n"
		"  --dump-graph        print the tensor graph before lowering\n"
		"  --stats             print graph statistics to stderr\n");
}

}  // namespace

int main(int argc, char **argv) {
	ModelConfig config;
	config.hidden = {32, 16};
	config.input_dim = 16;
	config.num_classes = 4;
	config.batch_size = 1;

	EmitOptions options;
	options.params.log_n = 13;
	options.params.log_scale = 26;
	int levels_override = 0;
	std::string output_path;
	bool dump_graph = false, stats = false;

	try {
		for (int i = 1; i < argc; ++i) {
			const std::string arg = argv[i];
			auto next = [&]() { return std::string(argv[++i]); };
			if (arg == "--hidden")
				config.hidden = parse_int_list(next());
			else if (arg == "--input-dim")
				config.input_dim = std::stoi(next());
			else if (arg == "--classes")
				config.num_classes = std::stoi(next());
			else if (arg == "--batch-size")
				config.batch_size = std::stoi(next());
			else if (arg == "--lr")
				config.learning_rate = std::stod(next());
			else if (arg == "--momentum")
				config.momentum = std::stod(next());
			else if (arg == "--weight-decay")
				config.weight_decay = std::stod(next());
			else if (arg == "--no-bootstrap")
				config.bootstrap = false;
			else if (arg == "--log-n")
				options.params.log_n = std::stoi(next());
			else if (arg == "--log-scale")
				options.params.log_scale = std::stoi(next());
			else if (arg == "--levels")
				levels_override = std::stoi(next());
			else if (arg == "--function")
				options.function_name = next();
			else if (arg == "-o")
				output_path = next();
			else if (arg == "--dump-graph")
				dump_graph = true;
			else if (arg == "--stats")
				stats = true;
			else if (arg == "--help" || arg == "-h") {
				print_help();
				return 0;
			} else {
				fmt::print(stderr, "unknown option {}\n", arg);
				return 1;
			}
		}

		const Layout layout =
			recommend_layout(config, options.params.num_slots());
		const TrainStep step = build_train_step(config, layout);
		const LoweredStep lowered = lower_to_slots(step);

		// The modulus chain has to be long enough for the levels the step
		// consumes; the slot graph knows exactly how many that is.
		options.params.levels = levels_override > 0
									? levels_override
									: lowered.graph.max_level() + 1;

		if (dump_graph) fmt::print(stderr, "{}", step.graph.dump());
		if (stats) {
			fmt::print(stderr, "{}", step.describe());
			fmt::print(stderr, "{}", lowered.graph.statistics());
			fmt::print(stderr, "  modulus chain            : {} primes\n",
					   options.params.levels + 1);
		}

		const std::string mlir = emit_mlir(lowered, options);
		if (output_path.empty()) {
			fmt::print("{}", mlir);
		} else {
			std::ofstream file(output_path);
			if (!file) {
				fmt::print(stderr, "cannot write {}\n", output_path);
				return 1;
			}
			file << mlir;
			fmt::print(stderr, "wrote {} ({} bytes)\n", output_path,
					   mlir.size());
		}
	} catch (const std::exception &error) {
		fmt::print(stderr, "error: {}\n", error.what());
		return 1;
	}
	return 0;
}
