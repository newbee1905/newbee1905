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
#include "reboot/manifest.h"
#include "reboot/mlir_emitter.h"
#include "reboot/options.h"
#include "reboot/reboot_model.h"
#include "reboot/slot_graph.h"

using namespace reboot;

int main(int argc, char **argv) {
	ModelConfig config;
	config.hidden = {32, 16};
	config.input_dim = 16;
	config.num_classes = 4;
	config.batch_size = 1;

	EmitOptions options;
	int log_n = 13;
	int levels_override = 0;
	std::string output_path, manifest_path;
	bool dump_graph = false, stats = false;

	OptionParser parser("reboot_emit",
						"emit a ReBoot training step as ckks-dialect MLIR");
	add_model_options(parser, config, log_n);
	parser.section("CKKS")
		.add("--log-scale", options.params.log_scale,
			 "log2 of the scaling factor")
		.add("--levels", levels_override,
			 "modulus chain length (default: from the graph)");
	parser.section("Output")
		.add("--function", options.function_name, "emitted function name")
		.add("-o", output_path, "write MLIR here instead of stdout")
		.add("--manifest", manifest_path,
			 "write the run manifest here (default: <output>.manifest)")
		.add_switch("--dump-graph", dump_graph, true,
					"print the tensor graph before lowering")
		.add_switch("--stats", stats, true, "print graph statistics to stderr");

	try {
		if (!parser.parse(argc, argv)) return 0;
		options.params.log_n = log_n;

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

		// The manifest is what lets the runner check it is calling the module
		// it thinks it is, instead of trusting that the flags matched.
		if (manifest_path.empty() && !output_path.empty())
			manifest_path = output_path + ".manifest";
		if (!manifest_path.empty()) {
			make_manifest(options.function_name, config, log_n, layout, lowered)
				.save(manifest_path);
			fmt::print(stderr, "wrote {}\n", manifest_path);
		}
	} catch (const std::exception &error) {
		fmt::print(stderr, "error: {}\n", error.what());
		return 1;
	}
	return 0;
}
