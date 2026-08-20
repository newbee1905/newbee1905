// SPDX-License-Identifier: GPL-3.0-or-later
//
// Plaintext interpreters for both graph levels.
//
// These are not a second implementation of ReBoot: they execute the very graph
// the emitter turns into MLIR, with the crypto removed.  That makes them
// oracles.  The tensor interpreter checks the differentiated graph against
// finite differences, and the slot interpreter checks that lowering the packing
// preserves the tensor semantics - which is where a rotate-and-add tree with a
// wrong index would show up.
//
// The slot interpreter also reproduces CKKS level accounting, so it reports the
// depth the emitted module needs without encrypting anything.

#ifndef REBOOT_INTERPRETER_H_
#define REBOOT_INTERPRETER_H_

#include <map>
#include <vector>

#include "reboot/slot_graph.h"
#include "reboot/tensor_graph.h"

namespace reboot {

// A dense node value is a plain vector of doubles: a vector of length cols, or
// a rows x cols matrix in row-major order.
using tensor_inputs_t = std::map<value_id_t, std::vector<double>>;
using slot_inputs_t = std::map<slot_id_t, std::vector<double>>;

// Evaluate every node of the tensor graph; leaves must appear in `inputs`.
std::vector<std::vector<double>> evaluate(const tensor_graph_t &graph,
										  const tensor_inputs_t &inputs);

// Evaluate every node of the slot graph; arguments must appear in `inputs`.
std::vector<std::vector<double>> evaluate(const slot_graph_t &graph,
										  const slot_inputs_t &inputs);

}  // namespace reboot

#endif	// REBOOT_INTERPRETER_H_
