// SPDX-License-Identifier: GPL-3.0-or-later
//
// Structural checks on the emitted module.
//
// The pipeline in the KeyMemRT-Compiler README consumes this text, so what
// matters is that the ops, their attributes and the function signature are the
// ones those passes look for - above all that every rotation carries a static
// index, because `--ckks-to-lwe` / `--lwe-to-openfhe` turn exactly that into
// kmrt.load_key / openfhe.rot / kmrt.clear_key.

#include <fmt/format.h>

#include <regex>
#include <set>
#include <string>
#include <vector>

#include "reboot/mlir_emitter.h"
#include "reboot/reboot_model.h"
#include "reboot/slot_graph.h"

using namespace reboot;

namespace {

int failures = 0;

void check(bool condition, const std::string &what) {
	if (!condition) {
		fmt::print("  FAIL: {}\n", what);
		++failures;
	}
}

size_t count_occurrences(const std::string &text, const std::string &needle) {
	size_t count = 0, pos = 0;
	while ((pos = text.find(needle, pos)) != std::string::npos) {
		++count;
		pos += needle.size();
	}
	return count;
}

size_t count_op(const SlotGraph &graph, SlotOp op) {
	size_t count = 0;
	for (const SlotValue &v : graph.values())
		if (v.op == op) ++count;
	return count;
}

}  // namespace

int main() {
	ModelConfig config;
	config.input_dim = 6;
	config.hidden = {8, 4};
	config.num_classes = 3;
	config.batch_size = 2;
	config.weight_decay = 0.01;

	EmitOptions options;
	options.params.log_n = 12;
	options.params.log_scale = 26;

	const Layout layout = recommend_layout(config, options.params.num_slots());
	const TrainStep step = build_train_step(config, layout);
	const LoweredStep lowered = lower_to_slots(step);
	options.params.levels = lowered.graph.max_level() + 1;
	const std::string mlir = emit_mlir(lowered, options);

	fmt::print("emitted module: {} bytes, {} slot values\n", mlir.size(),
			   lowered.graph.size());

	fmt::print("module structure\n");
	check(mlir.find("ckks.schemeParam = #ckks.scheme_param<logN = 12") !=
			  std::string::npos,
		  "scheme parameters name the ring dimension");
	check(mlir.find("func.func @reboot_train_step(") != std::string::npos,
		  "the training step is a func.func");
	check(count_occurrences(mlir, "!lwe.lwe_ciphertext<") >= 2,
		  "ciphertext types are spelled out for the pipeline");

	fmt::print("operation counts match the slot graph\n");
	const size_t muls = count_op(lowered.graph, SlotOp::kMul);
	check(count_occurrences(mlir, "ckks.mul ") == muls, "ckks.mul count");
	check(count_occurrences(mlir, "ckks.relinearize ") == muls,
		  "every product is relinearised back to the canonical basis");
	check(count_occurrences(mlir, "ckks.add ") ==
			  count_op(lowered.graph, SlotOp::kAdd),
		  "ckks.add count");
	check(count_occurrences(mlir, "ckks.sub ") ==
			  count_op(lowered.graph, SlotOp::kSub),
		  "ckks.sub count");
	check(count_occurrences(mlir, "ckks.mul_scalar ") ==
			  count_op(lowered.graph, SlotOp::kMulScalar),
		  "ckks.mul_scalar count");
	check(count_occurrences(mlir, "ckks.bootstrap ") ==
			  count_op(lowered.graph, SlotOp::kBootstrap),
		  "ckks.bootstrap count");
	check(count_occurrences(mlir, "lwe.rlwe_encode ") ==
			  lowered.graph.constants().size(),
		  "one encoded plaintext per constant");

	fmt::print("rotations carry the index KeyMemRT needs\n");
	const size_t rotates = count_op(lowered.graph, SlotOp::kRotate);
	check(count_occurrences(mlir, "ckks.rotate ") == rotates,
		  "ckks.rotate count");
	check(count_occurrences(mlir, "static_shift") == rotates,
		  "every rotation has a static_shift attribute");

	std::set<int> emitted;
	const std::regex shift_pattern("static_shift = (-?[0-9]+) : i64");
	for (auto it =
			 std::sregex_iterator(mlir.begin(), mlir.end(), shift_pattern);
		 it != std::sregex_iterator(); ++it)
		emitted.insert(std::stoi((*it)[1]));
	const std::vector<int> expected = lowered.graph.rotation_indices();
	check(emitted == std::set<int>(expected.begin(), expected.end()),
		  "the emitted rotation indices are exactly the graph's");
	fmt::print("  {} rotations over {} distinct indices\n", rotates,
			   emitted.size());

	fmt::print("signature and results\n");
	check(count_occurrences(mlir, "reboot.name = ") == lowered.arguments.size(),
		  "every argument is named for the runner");
	for (const std::string &name :
		 {"w_fwd_0", "v_w_fwd_0", "w_out", "x_0", "y_repeated_0"})
		check(mlir.find(fmt::format("reboot.name = \"{}\"", name)) !=
				  std::string::npos,
			  fmt::format("argument {} is present", name));

	const size_t return_pos = mlir.rfind("return ");
	check(return_pos != std::string::npos, "the function returns");
	const std::string tail = mlir.substr(return_pos);
	check(count_occurrences(tail, "%") == lowered.results.size(),
		  "all updated weights, velocities and predictions are returned");

	fmt::print("no placeholders left behind\n");
	check(mlir.find("%-1") == std::string::npos, "no unresolved values");
	check(mlir.find("kNoSlot") == std::string::npos, "no debug spelling");

	if (failures == 0) {
		fmt::print("all emitter tests passed\n");
		return 0;
	}
	fmt::print("{} failure(s)\n", failures);
	return 1;
}
