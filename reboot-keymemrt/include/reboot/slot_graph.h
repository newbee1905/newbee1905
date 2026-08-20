// SPDX-License-Identifier: GPL-3.0-or-later
//
// Slot-level graph: the differentiated tensor graph lowered to the operations
// the `ckks` dialect actually has.
//
// Every tensor value becomes one ciphertext holding a rows x cols slot matrix.
// The two matrix products collapse into an elementwise multiply followed by a
// summation in one of the two directions, and each summation is spelled out as
// a rotate-and-add tree over explicit indices:
//
//   sum_rows  for k = cols, 2*cols, ... < slots:  x += rot(x, k)
//             column sums replicated down every row, no extra level
//   sum_cols  for k = 1, 2, ... < cols:           x += rot(x, k)
//             x *= mask(first column of each row)
//             for k = 1, 2, ... < cols:           x += rot(x, -k)
//             row sums replicated across every row, one masking level
//
// Spelling the rotations out is the whole point: KeyMemRT pages rotation keys
// one index at a time, so the indices have to appear in the IR.  OpenFHE's
// EvalSumRows / EvalSumCols, which the reference implementation calls, hide
// them inside the library and draw on the EvalSum key map that KeyMemRT does
// not manage.

#ifndef REBOOT_SLOT_GRAPH_H_
#define REBOOT_SLOT_GRAPH_H_

#include <string>
#include <vector>

#include "reboot/layout.h"
#include "reboot/reboot_model.h"

namespace reboot {

using SlotId = int;
inline constexpr SlotId kNoSlot = -1;

enum class SlotOp {
	kArgument,	 // ciphertext function argument
	kAdd,		 // ckks.add
	kSub,		 // ckks.sub
	kMul,		 // ckks.mul + ckks.relinearize
	kMulPlain,	 // ckks.mul_plain against an encoded constant
	kAddPlain,	 // ckks.add_plain against an encoded constant
	kMulScalar,	 // ckks.mul_scalar, no level consumed
	kRotate,	 // ckks.rotate {static_shift}
	kBootstrap,	 // ckks.bootstrap
};

const char *slot_op_name(SlotOp op);

struct SlotValue {
	SlotId id = kNoSlot;
	SlotOp op = SlotOp::kArgument;
	std::vector<SlotId> inputs;
	int rotation = 0;	  // kRotate
	double scalar = 0.0;  // kMulScalar
	int constant = -1;	  // kMulPlain, kAddPlain: index into constants()
	int level = 0;		  // levels consumed on the path to this value
	std::string name;	  // arguments and results
};

// A public constant vector, encoded once and reused.
struct SlotConstant {
	std::string name;
	std::vector<double> values;
	bool splat = false;	 // every slot identical: emitted as dense<c>
};

class SlotGraph {
   public:
	explicit SlotGraph(const Layout &layout) : layout_(layout) {}

	const Layout &layout() const { return layout_; }
	const std::vector<SlotValue> &values() const { return values_; }
	const std::vector<SlotConstant> &constants() const { return constants_; }
	const SlotValue &value(SlotId id) const { return values_.at(id); }
	size_t size() const { return values_.size(); }

	SlotId argument(const std::string &name);
	SlotId add(SlotId a, SlotId b);
	SlotId sub(SlotId a, SlotId b);
	SlotId mul(SlotId a, SlotId b);
	SlotId mul_plain(SlotId a, int constant);
	SlotId add_plain(SlotId a, int constant);
	SlotId mul_scalar(SlotId a, double s);
	SlotId rotate(SlotId a, int shift);
	SlotId bootstrap(SlotId a);

	// Summation trees.
	SlotId sum_rows(SlotId a);
	SlotId sum_cols(SlotId a);

	// Constant pool.
	int splat_constant(double value);
	int first_column_mask_constant();

	void set_name(SlotId id, const std::string &name);

	// Every rotation index the graph uses, sorted.
	std::vector<int> rotation_indices() const;
	int max_level() const;
	std::string statistics() const;

   private:
	SlotId push(SlotValue v);

	Layout layout_;
	std::vector<SlotValue> values_;
	std::vector<SlotConstant> constants_;
	int mask_constant_ = -1;
};

struct LoweredStep {
	SlotGraph graph;
	std::vector<SlotId> arguments;
	std::vector<std::string> argument_names;
	std::vector<SlotId> results;
	std::vector<std::string> result_names;

	explicit LoweredStep(const Layout &layout) : graph(layout) {}
};

// Lower a differentiated training step to slot operations.
LoweredStep lower_to_slots(const TrainStep &step);

}  // namespace reboot

#endif	// REBOOT_SLOT_GRAPH_H_
