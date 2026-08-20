// SPDX-License-Identifier: GPL-3.0-or-later
//
// The ReBoot network and its training step, expressed as a tensor graph.
//
// Architecture (eMLP-h of the paper).  For h hidden layers:
//
//   h-1 local-loss blocks    [ linear -> PolyReLU ] + a local classifier
//   1   hidden linear + PolyReLU
//   1   output linear
//
// Each block ends in a kStopGradient, so its error signal never leaves the
// block: that is what keeps the multiplicative depth of a step independent of
// network depth.  Packing alternates row / column down the network, so no layer
// ever repacks its input.
//
// Nothing here writes a backward pass.  The builder assembles the forward graph
// and the loss seeds, calls the autograd pass, and then appends the optimiser
// and the bootstrapping of everything that survives into the next step.

#ifndef REBOOT_REBOOT_MODEL_H_
#define REBOOT_REBOOT_MODEL_H_

#include <map>
#include <string>
#include <vector>

#include "reboot/autograd.h"
#include "reboot/layout.h"
#include "reboot/tensor_graph.h"

namespace reboot {

struct ModelConfig {
	int input_dim = 0;
	std::vector<int> hidden;  // one entry per hidden layer
	int num_classes = 0;
	int batch_size = 1;
	double learning_rate = 0.005;
	double momentum = 0.9;
	double weight_decay = 0.0;
	// Bootstrap the weights and velocities at the end of the step.  Off is only
	// useful for inspecting a single step.
	bool bootstrap = true;
};

// Everything the emitter needs about one trainable tensor: the incoming state,
// the gradient the autograd pass produced, and the refreshed state to return.
struct ParamBinding {
	std::string name;
	ValueId weight = kNoValue;
	ValueId velocity = kNoValue;
	ValueId gradient = kNoValue;
	ValueId weight_out = kNoValue;
	ValueId velocity_out = kNoValue;
};

// One term of the training objective: 1/2 |prediction - label|^2.  A block's
// term is the only one that reaches its weights, which is what "local error
// signal" means and what the gradient locality test checks.
struct LossTerm {
	std::string name;
	ValueId prediction = kNoValue;
	ValueId label = kNoValue;
};

struct TrainStep {
	TensorGraph graph;
	Layout layout;
	ModelConfig config;

	std::vector<ValueId> arguments;	 // function arguments, in order
	std::vector<ValueId> results;	 // function results, in order
	std::vector<ParamBinding> params;
	std::vector<ValueId> predictions;  // top-level prediction per sample
	std::vector<LossTerm> losses;

	// Level the deepest value reaches, from the same accounting CKKS uses:
	// every ciphertext-ciphertext or ciphertext-plaintext product costs one.
	int required_depth = 0;

	std::string describe() const;
};

// Smallest layout that holds every layer, then inflated so that rows * cols
// equals the slot count: the rotate-and-add trees run over the whole slot
// vector, and a partially used one would let the cyclic rotations mix in the
// unused region.
Layout recommend_layout(const ModelConfig &config, int num_slots);

// Build the forward graph, differentiate it, and append the Nesterov update and
// the bootstrapping.  The velocities arrive as arguments; passing zeros on the
// first step reproduces ReBoot's separate "initialise the velocity" branch
// exactly, so there is only one update rule to emit.
TrainStep build_train_step(const ModelConfig &config, const Layout &layout);

}  // namespace reboot

#endif	// REBOOT_REBOOT_MODEL_H_
