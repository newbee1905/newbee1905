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
// Each block ends in a stop_gradient, so its error signal never leaves the
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

struct model_config_t {
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
struct param_binding_t {
	std::string name;
	value_id_t weight = no_value;
	value_id_t velocity = no_value;
	value_id_t gradient = no_value;
	value_id_t weight_out = no_value;
	value_id_t velocity_out = no_value;
};

// One term of the training objective: 1/2 |prediction - label|^2.  A block's
// term is the only one that reaches its weights, which is what "local error
// signal" means and what the gradient locality test checks.
struct loss_term_t {
	std::string name;
	value_id_t prediction = no_value;
	value_id_t label = no_value;
};

struct train_step_t {
	tensor_graph_t graph;
	layout_t layout;
	model_config_t config;

	std::vector<value_id_t> arguments;	// function arguments, in order
	std::vector<value_id_t> results;	// function results, in order
	std::vector<param_binding_t> params;
	std::vector<value_id_t> predictions;  // top-level prediction per sample
	std::vector<loss_term_t> losses;

	// Level the deepest value reaches, from the same accounting CKKS uses:
	// every ciphertext-ciphertext or ciphertext-plaintext product costs one.
	int required_depth = 0;

	std::string describe() const;
};

// Smallest layout that holds every layer, then inflated so that rows * cols
// equals the slot count: the rotate-and-add trees run over the whole slot
// vector, and a partially used one would let the cyclic rotations mix in the
// unused region.
layout_t recommend_layout(const model_config_t &config, int num_slots);

// Build the forward graph, differentiate it, and append the Nesterov update and
// the bootstrapping.  The velocities arrive as arguments; passing zeros on the
// first step reproduces ReBoot's separate "initialise the velocity" branch
// exactly, so there is only one update rule to emit.
train_step_t build_train_step(const model_config_t &config,
							  const layout_t &layout);

}  // namespace reboot

#endif	// REBOOT_REBOOT_MODEL_H_
