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

size_t count_op(const slot_graph_t &graph, slot_op_t op) {
	size_t count = 0;
	for (const slot_value_t &v : graph.values())
		if (v.op == op) ++count;
	return count;
}

}  // namespace

int main() {
	model_config_t config;
	config.input_dim = 6;
	config.hidden = {8, 4};
	config.num_classes = 3;
	config.batch_size = 2;
	config.weight_decay = 0.01;

	emit_options_t options;
	options.params.log_n = 12;
	options.params.log_scale = 26;

	const layout_t layout =
		recommend_layout(config, options.params.num_slots());
	const train_step_t step = build_train_step(config, layout);
	const lowered_step_t lowered = lower_to_slots(step);
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
	const size_t muls = count_op(lowered.graph, slot_op_t::mul);
	check(count_occurrences(mlir, "ckks.mul ") == muls, "ckks.mul count");
	check(count_occurrences(mlir, "ckks.relinearize ") == muls,
		  "every product is relinearised back to the canonical basis");
	check(count_occurrences(mlir, "ckks.add ") ==
			  count_op(lowered.graph, slot_op_t::add),
		  "ckks.add count");
	check(count_occurrences(mlir, "ckks.sub ") ==
			  count_op(lowered.graph, slot_op_t::sub),
		  "ckks.sub count");
	check(count_occurrences(mlir, "ckks.mul_scalar ") ==
			  count_op(lowered.graph, slot_op_t::mul_scalar),
		  "ckks.mul_scalar count");
	check(count_occurrences(mlir, "ckks.bootstrap ") ==
			  count_op(lowered.graph, slot_op_t::bootstrap),
		  "ckks.bootstrap count");
	check(count_occurrences(mlir, "lwe.rlwe_encode ") ==
			  lowered.graph.constants().size(),
		  "one encoded plaintext per constant");

	fmt::print("rotations carry the index KeyMemRT needs\n");
	const size_t rotates = count_op(lowered.graph, slot_op_t::rotate);
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

	// keymemrt-translate only emits C++ for single-result functions, so both
	// sides of the ABI travel as one tensor of ciphertexts, which converts to
	// std::vector<CiphertextT>.
	fmt::print("tensor ABI is translatable\n");
	check(mlir.find(fmt::format("(%args: tensor<{}x!ct>",
								lowered.arguments.size())) != std::string::npos,
		  "arguments arrive as one tensor");
	check(mlir.find(fmt::format("-> tensor<{}x!ct>", lowered.results.size())) !=
			  std::string::npos,
		  "results leave as one tensor");
	check(
		count_occurrences(mlir, "tensor.extract ") == lowered.arguments.size(),
		"every argument is extracted once");
	check(count_occurrences(mlir, "tensor.insert ") == lowered.results.size(),
		  "every result is inserted once");
	check(count_occurrences(mlir, "tensor.empty()") == 1,
		  "one result tensor is allocated");
	check(count_occurrences(mlir, "return ") == 1,
		  "a single value is returned");

	fmt::print("the ABI names travel with the module\n");
	for (const std::string &name :
		 {"w_fwd_0", "v_w_fwd_0", "w_out", "x_0", "y_repeated_0"})
		check(mlir.find(fmt::format("\"{}\"", name)) != std::string::npos,
			  fmt::format("argument {} is named", name));
	check(mlir.find("reboot.argument_names = [") != std::string::npos,
		  "argument order is recorded for the runner");
	check(mlir.find("reboot.result_names = [") != std::string::npos,
		  "result order is recorded for the runner");

	fmt::print("no placeholders left behind\n");
	check(mlir.find("%-1") == std::string::npos, "no unresolved values");
	check(mlir.find("no_slot") == std::string::npos, "no debug spelling");

	if (failures == 0) {
		fmt::print("all emitter tests passed\n");
		return 0;
	}
	fmt::print("{} failure(s)\n", failures);
	return 1;
}
