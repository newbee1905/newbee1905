// SPDX-License-Identifier: GPL-3.0-or-later
//
// The lowering turns tensor operations into slot operations: a matrix product
// becomes an elementwise multiply plus a rotate-and-add tree, and the transpose
// product becomes the same multiply summed the other way.  This test runs both
// levels on the same inputs and compares every result of the training step -
// updated weights, updated velocities and predictions.
//
// A wrong rotation index, a wrong summation direction or a mask that clips the
// wrong slots all show up here, which is what makes it safe to emit the slot
// graph as MLIR without an FHE library in the loop.

#include <fmt/format.h>

#include <cmath>
#include <random>
#include <vector>

#include "reboot/interpreter.h"
#include "reboot/layout.h"
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

DenseValue random_dense(size_t n, std::mt19937 &rng, double scale) {
	std::uniform_real_distribution<double> dist(-scale, scale);
	DenseValue v(n);
	for (double &x : v) x = dist(rng);
	return v;
}

// Pack a tensor value the way the emitted program expects to receive it.
DenseValue pack(const TensorValue &meta, const DenseValue &dense,
				const Layout &layout) {
	if (meta.shape.is_vector()) return pack_vector(dense, meta.format, layout);
	return pack_weights(dense, meta.shape.rows, meta.shape.cols,
						meta.row_packing, layout);
}

DenseValue unpack(const TensorValue &meta, const DenseValue &slots,
				  const Layout &layout) {
	if (meta.shape.is_vector())
		return unpack_vector(slots, meta.format, layout, meta.shape.cols);
	return unpack_weights(slots, meta.shape.rows, meta.shape.cols,
						  meta.row_packing, layout);
}

double max_deviation(const DenseValue &a, const DenseValue &b) {
	double worst = 0.0;
	for (size_t i = 0; i < a.size(); ++i)
		worst = std::max(worst, std::fabs(a[i] - b[i]));
	return worst;
}

void run_case(const std::string &title, const ModelConfig &config,
			  int num_slots) {
	fmt::print("{}\n", title);

	const Layout layout = recommend_layout(config, num_slots);
	const TrainStep step = build_train_step(config, layout);
	const LoweredStep lowered = lower_to_slots(step);
	const TensorGraph &g = step.graph;

	check(lowered.arguments.size() == step.arguments.size(),
		  "every argument survives the lowering");
	check(lowered.results.size() == step.results.size(),
		  "every result survives the lowering");

	std::mt19937 rng(17);
	TensorInputs dense_inputs;
	SlotInputs slot_inputs;
	for (size_t i = 0; i < step.arguments.size(); ++i) {
		const ValueId id = step.arguments[i];
		const TensorValue &meta = g.value(id);
		const size_t n = static_cast<size_t>(meta.shape.rows) * meta.shape.cols;
		DenseValue dense = meta.name.rfind("v_", 0) == 0
							   ? DenseValue(n, 0.0)
							   : random_dense(n, rng, 0.4);
		slot_inputs[lowered.arguments[i]] = pack(meta, dense, layout);
		dense_inputs[id] = std::move(dense);
	}

	const std::vector<DenseValue> dense_values = evaluate(g, dense_inputs);
	const std::vector<DenseValue> slot_values =
		evaluate(lowered.graph, slot_inputs);

	double worst = 0.0;
	for (size_t i = 0; i < step.results.size(); ++i) {
		const TensorValue &meta = g.value(step.results[i]);
		const DenseValue got =
			unpack(meta, slot_values[lowered.results[i]], layout);
		worst =
			std::max(worst, max_deviation(got, dense_values[step.results[i]]));
	}
	fmt::print("  {} results, {} slot values, {} rotations\n",
			   step.results.size(), lowered.graph.size(),
			   lowered.graph.rotation_indices().size());
	fmt::print("  largest deviation: {:.3e}\n", worst);
	check(worst < 1e-9, "packed evaluation matches the tensor semantics");
}

void test_rotation_indices() {
	fmt::print("rotation indices are powers of two within the layout\n");

	ModelConfig config;
	config.input_dim = 6;
	config.hidden = {8, 4};
	config.num_classes = 3;
	config.batch_size = 1;

	const Layout layout = recommend_layout(config, 1024);
	const LoweredStep lowered =
		lower_to_slots(build_train_step(config, layout));

	for (int index : lowered.graph.rotation_indices()) {
		const int magnitude = std::abs(index);
		check(magnitude > 0 && (magnitude & (magnitude - 1)) == 0,
			  fmt::format("rotation {} is a power of two", index));
		check(magnitude < layout.slots(),
			  fmt::format("rotation {} stays inside the slot vector", index));
	}
	// The column summation is the only source of right rotations.
	bool has_negative = false;
	for (int index : lowered.graph.rotation_indices())
		has_negative = has_negative || index < 0;
	check(has_negative, "the column summation replicates with right rotations");
}

}  // namespace

int main() {
	ModelConfig single;
	single.input_dim = 6;
	single.hidden = {8};
	single.num_classes = 3;
	single.batch_size = 1;
	run_case("eMLP-1: head and output only", single, 512);

	ModelConfig blocks;
	blocks.input_dim = 5;
	blocks.hidden = {6, 4};
	blocks.num_classes = 3;
	blocks.batch_size = 2;
	run_case("eMLP-2: one local-loss block, batch of two", blocks, 1024);

	ModelConfig deep;
	deep.input_dim = 4;
	deep.hidden = {6, 5, 4};
	deep.num_classes = 2;
	deep.batch_size = 1;
	deep.weight_decay = 0.01;
	run_case("eMLP-3: two blocks with weight decay", deep, 1024);

	test_rotation_indices();

	if (failures == 0) {
		fmt::print("all lowering tests passed\n");
		return 0;
	}
	fmt::print("{} failure(s)\n", failures);
	return 1;
}
