// SPDX-License-Identifier: GPL-3.0-or-later
//
// The contract between the emitted module and whoever runs it.
//
// The generated function takes one tensor of ciphertexts, so its caller has to
// know what goes in which slot of that tensor, which packing each one is in,
// and which model produced the module.  Passing "the same flags" to
// `reboot_emit` and `reboot_runner` and hoping they match is not a contract: a
// single mistyped width silently builds a different argument order and the run
// is wrong rather than broken.
//
// So the emitter writes this manifest next to the .mlir and the runner reads
// it.  The runner still rebuilds the graph - it needs the shapes and formats to
// pack the data - but then checks the rebuilt argument and result order against
// the manifest and refuses to run on a mismatch.

#ifndef REBOOT_MANIFEST_H_
#define REBOOT_MANIFEST_H_

#include <string>
#include <vector>

#include "reboot/layout.h"
#include "reboot/reboot_model.h"
#include "reboot/slot_graph.h"

namespace reboot {

struct manifest_t {
	std::string function_name;
	model_config_t config;
	int log_n = 0;
	layout_t layout;
	std::vector<std::string> argument_names;
	std::vector<std::string> result_names;

	// Written next to the module, `<module>.manifest`.
	void save(const std::string &path) const;
	static manifest_t load(const std::string &path);

	// Throws when a freshly built step does not match what the module was
	// emitted from, naming the first difference.
	void verify(const lowered_step_t &step) const;

	std::string describe() const;
};

manifest_t make_manifest(const std::string &function_name,
						 const model_config_t &config, int log_n,
						 const layout_t &layout, const lowered_step_t &step);

}  // namespace reboot

#endif	// REBOOT_MANIFEST_H_
