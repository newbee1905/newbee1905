// SPDX-License-Identifier: GPL-3.0-or-later
//
// Reverse-mode differentiation of the tensor graph.
//
// The training step is never written by hand: the forward network is built,
// each local loss contributes a seed, and this pass walks the graph backwards
// accumulating adjoints.  ReBoot's published backward rules come out of the
// standard vector-Jacobian products - including the two that matter for
// packing, `d/dx matmul(x, W) = matmul_t(g, W)` and `d/dW matmul(x, W) =
// outer(x, g)`, which lower to the same weight ciphertext summed the other way
// and to a bare elementwise product.
//
// The seeds are given rather than derived from a scalar loss node, because
// ReBoot never materialises the loss: for the residual sum of squares
// L = 1/2 |y_hat - y|^2 the seed is simply y_hat - y, and computing it costs
// one subtraction instead of a full slot reduction.

#ifndef REBOOT_AUTOGRAD_H_
#define REBOOT_AUTOGRAD_H_

#include <map>
#include <vector>

#include "reboot/tensor_graph.h"

namespace reboot {

// One term of the objective: the adjoint `seed` is dL/d(value).
struct GradientSeed {
	ValueId value = kNoValue;
	ValueId seed = kNoValue;
};

// Seed for 1/2 |prediction - target|^2 with respect to `prediction`.
ValueId rss_seed(TensorGraph &graph, ValueId prediction, ValueId target);

// Accumulated adjoints, keyed by the value they belong to.  Values the
// objective does not depend on are absent rather than zero, so a caller asking
// for the gradient of an untouched parameter gets kNoValue and can say so.
using GradientMap = std::map<ValueId, ValueId>;

// Differentiate the objective described by `seeds`.  Nodes behind a
// kStopGradient are reached in the forward direction but never receive an
// adjoint, which is what confines each block's error signal to that block.
GradientMap backward(TensorGraph &graph,
					 const std::vector<GradientSeed> &seeds);

}  // namespace reboot

#endif	// REBOOT_AUTOGRAD_H_
