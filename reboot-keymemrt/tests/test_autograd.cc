// SPDX-License-Identifier: GPL-3.0-or-later
//
// The training step is differentiated, not written by hand, so the thing worth
// testing is the differentiation itself.
//
//  1. Gradients of a small network against central finite differences.
//  2. Gradient locality: in a two-block ReBoot network the gradient the
//     autograd pass produces for a block's forward weights matches the finite
//     difference of that block's own loss, and differs from the finite
//     difference of the whole objective.  That difference is precisely what
//     kStopGradient buys - and it is why the multiplicative depth of a step
//     does not grow with the number of blocks.

#include <fmt/format.h>

#include <cmath>
#include <random>
#include <vector>

#include "reboot/autograd.h"
#include "reboot/interpreter.h"
#include "reboot/reboot_model.h"
#include "reboot/tensor_graph.h"

using namespace reboot;

namespace {

int failures = 0;

void check(bool condition, const std::string &what) {
	if (!condition) {
		fmt::print("  FAIL: {}\n", what);
		++failures;
	}
}

DenseValue random_dense(size_t n, std::mt19937 &rng, double scale = 0.5) {
	std::uniform_real_distribution<double> dist(-scale, scale);
	DenseValue v(n);
	for (double &x : v) x = dist(rng);
	return v;
}

double rss(const DenseValue &prediction, const DenseValue &label) {
	double sum = 0.0;
	for (size_t i = 0; i < prediction.size(); ++i) {
		const double d = prediction[i] - label[i];
		sum += d * d;
	}
	return 0.5 * sum;
}

// ---------------------------------------------------------------------------

void test_chain_rule() {
	fmt::print("gradients against finite differences\n");

	TensorGraph g;
	const ValueId x = g.input("x", 4, PackFormat::kExpanded);
	const ValueId w1 = g.param("w1", 4, 3, /*row_packing=*/true);
	const ValueId w2 = g.param("w2", 3, 2, /*row_packing=*/false);
	const ValueId h = g.poly_relu(g.matmul(x, w1));
	const ValueId prediction = g.matmul(h, w2);
	const ValueId y = g.input("y", 2, g.value(prediction).format);

	const GradientMap grads =
		backward(g, {{prediction, rss_seed(g, prediction, y)}});
	check(grads.count(w1) == 1, "w1 has a gradient");
	check(grads.count(w2) == 1, "w2 has a gradient");

	std::mt19937 rng(5);
	TensorInputs inputs;
	inputs[x] = random_dense(4, rng);
	inputs[y] = random_dense(2, rng);
	inputs[w1] = random_dense(12, rng);
	inputs[w2] = random_dense(6, rng);

	const std::vector<DenseValue> values = evaluate(g, inputs);
	const DenseValue analytic_w1 = values[grads.at(w1)];
	const DenseValue analytic_w2 = values[grads.at(w2)];

	const double eps = 1e-6;
	auto finite_difference = [&](ValueId param, size_t index) {
		TensorInputs plus = inputs, minus = inputs;
		plus[param][index] += eps;
		minus[param][index] -= eps;
		const double loss_plus =
			rss(evaluate(g, plus)[prediction], inputs.at(y));
		const double loss_minus =
			rss(evaluate(g, minus)[prediction], inputs.at(y));
		return (loss_plus - loss_minus) / (2.0 * eps);
	};

	double worst = 0.0;
	for (size_t i = 0; i < analytic_w1.size(); ++i)
		worst = std::max(worst,
						 std::fabs(analytic_w1[i] - finite_difference(w1, i)));
	for (size_t i = 0; i < analytic_w2.size(); ++i)
		worst = std::max(worst,
						 std::fabs(analytic_w2[i] - finite_difference(w2, i)));
	fmt::print("  largest deviation over 18 entries: {:.3e}\n", worst);
	check(worst < 1e-6, "autograd matches finite differences");
}

void test_gradient_locality() {
	fmt::print("gradient locality of the local-loss blocks\n");

	ModelConfig config;
	config.input_dim = 5;
	config.hidden = {6, 4, 3};	// two blocks + head
	config.num_classes = 3;
	config.batch_size = 1;
	config.bootstrap = false;

	const Layout layout = recommend_layout(config, 1024);
	const TrainStep step = build_train_step(config, layout);
	const TensorGraph &g = step.graph;

	check(step.losses.size() == 3,
		  "three loss terms: two blocks and the output");

	std::mt19937 rng(11);
	TensorInputs inputs;
	for (ValueId id : step.arguments) {
		const TensorValue &v = g.value(id);
		const size_t n = static_cast<size_t>(v.shape.rows) * v.shape.cols;
		// Velocities start at zero, which is what makes a single Nesterov rule
		// reproduce ReBoot's separate first-step branch.
		inputs[id] = v.name.rfind("v_", 0) == 0 ? DenseValue(n, 0.0)
												: random_dense(n, rng, 0.4);
	}

	const std::vector<DenseValue> values = evaluate(g, inputs);

	// The gradient the autograd pass produced for the first block's forward
	// weights.
	const ParamBinding &block0 = step.params.at(0);
	check(block0.name == "w_fwd_0", "first parameter is the first block");
	const DenseValue analytic = values[block0.gradient];

	const double eps = 1e-6;
	auto loss_of = [&](const TensorInputs &in, size_t which) {
		const std::vector<DenseValue> v = evaluate(g, in);
		return rss(v[step.losses[which].prediction],
				   in.at(step.losses[which].label));
	};
	auto total_loss = [&](const TensorInputs &in) {
		double sum = 0.0;
		for (size_t i = 0; i < step.losses.size(); ++i) sum += loss_of(in, i);
		return sum;
	};

	double worst_local = 0.0, worst_total = 0.0;
	for (size_t i = 0; i < analytic.size(); ++i) {
		TensorInputs plus = inputs, minus = inputs;
		plus[block0.weight][i] += eps;
		minus[block0.weight][i] -= eps;
		const double local =
			(loss_of(plus, 0) - loss_of(minus, 0)) / (2.0 * eps);
		const double total =
			(total_loss(plus) - total_loss(minus)) / (2.0 * eps);
		worst_local = std::max(worst_local, std::fabs(analytic[i] - local));
		worst_total = std::max(worst_total, std::fabs(analytic[i] - total));
	}
	fmt::print("  deviation from the block's own loss : {:.3e}\n", worst_local);
	fmt::print("  deviation from the whole objective  : {:.3e}\n", worst_total);
	check(worst_local < 1e-6,
		  "block gradient equals the gradient of its own loss");
	check(worst_total > 1e-4,
		  "block gradient deliberately ignores the downstream losses");
}

void test_output_layer_gradient() {
	fmt::print("output layer gradient\n");

	ModelConfig config;
	config.input_dim = 4;
	config.hidden = {5};
	config.num_classes = 2;
	config.batch_size = 2;
	config.bootstrap = false;

	const Layout layout = recommend_layout(config, 512);
	const TrainStep step = build_train_step(config, layout);
	const TensorGraph &g = step.graph;

	std::mt19937 rng(3);
	TensorInputs inputs;
	for (ValueId id : step.arguments) {
		const TensorValue &v = g.value(id);
		const size_t n = static_cast<size_t>(v.shape.rows) * v.shape.cols;
		inputs[id] = v.name.rfind("v_", 0) == 0 ? DenseValue(n, 0.0)
												: random_dense(n, rng, 0.4);
	}

	const ParamBinding &out = step.params.back();
	check(out.name == "w_out", "last parameter is the output layer");
	const DenseValue analytic = evaluate(g, inputs)[out.gradient];

	// The output weights only take part in the top loss, and both samples of
	// the batch contribute - so this also checks gradient accumulation.
	const double eps = 1e-6;
	auto top_loss = [&](const TensorInputs &in) {
		const std::vector<DenseValue> v = evaluate(g, in);
		double sum = 0.0;
		for (const LossTerm &term : step.losses)
			if (term.name.rfind("output_", 0) == 0)
				sum += rss(v[term.prediction], in.at(term.label));
		return sum;
	};

	double worst = 0.0;
	for (size_t i = 0; i < analytic.size(); ++i) {
		TensorInputs plus = inputs, minus = inputs;
		plus[out.weight][i] += eps;
		minus[out.weight][i] -= eps;
		worst = std::max(
			worst, std::fabs(analytic[i] -
							 (top_loss(plus) - top_loss(minus)) / (2.0 * eps)));
	}
	fmt::print("  largest deviation over a batch of 2: {:.3e}\n", worst);
	check(worst < 1e-6, "output gradient accumulates over the batch");
}

}  // namespace

int main() {
	test_chain_rule();
	test_gradient_locality();
	test_output_layer_gradient();

	if (failures == 0) {
		fmt::print("all autograd tests passed\n");
		return 0;
	}
	fmt::print("{} failure(s)\n", failures);
	return 1;
}
