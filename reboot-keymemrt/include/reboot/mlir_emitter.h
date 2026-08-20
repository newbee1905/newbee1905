// SPDX-License-Identifier: GPL-3.0-or-later
//
// Emits the lowered training step as `ckks`-dialect MLIR for KeyMemRT-Compiler.
//
// The output is the input format of the pipeline in the KeyMemRT-Compiler
// README: a module carrying `ckks.schemeParam`, with one `func.func` whose
// arguments and results are `!lwe.lwe_ciphertext` values.  Running
//
//   keymemrt-opt --ckks-to-lwe --lwe-to-openfhe --annotate-module=... \
//                --openfhe-configure-crypto-context --kmrt-merge-rotation-keys \
//                --bootstrap-rotation-analysis --openfhe-insert-clear-ops \
//                --kmrt-key-prefetching=... reboot_train_step.mlir
//
// turns every `ckks.rotate` into `kmrt.load_key` / `openfhe.rot` /
// `kmrt.clear_key`, which is how the rotation keys of the training step end up
// under KeyMemRT's per-key management.  That is the entire reason the packing
// is lowered to explicit rotations rather than left as EvalSumRows/EvalSumCols.
//
// Scale handling: every ciphertext is emitted at the top of the modulus chain
// and multiplications are followed by a relinearisation, leaving the rescaling
// to OpenFHE's FLEXIBLEAUTO at run time.  The depth the graph actually needs is
// reported separately and sizes the modulus chain.

#ifndef REBOOT_MLIR_EMITTER_H_
#define REBOOT_MLIR_EMITTER_H_

#include <string>

#include "reboot/ckks_params.h"
#include "reboot/slot_graph.h"

namespace reboot {

struct EmitOptions {
	std::string function_name = "reboot_train_step";
	CkksParams params;
	// Emit each value's origin (the tensor node it came from) as a comment.
	bool annotate = true;
};

// Render the lowered step as an MLIR module.
std::string emit_mlir(const LoweredStep &step, const EmitOptions &options);

}  // namespace reboot

#endif	// REBOOT_MLIR_EMITTER_H_
