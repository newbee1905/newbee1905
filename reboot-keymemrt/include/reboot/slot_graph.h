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

using slot_id_t = int;
inline constexpr slot_id_t no_slot = -1;

enum class slot_op_t {
	argument,	 // ciphertext function argument
	add,		 // ckks.add
	sub,		 // ckks.sub
	mul,		 // ckks.mul + ckks.relinearize
	mul_plain,	 // ckks.mul_plain against an encoded constant
	add_plain,	 // ckks.add_plain against an encoded constant
	mul_scalar,	 // ckks.mul_scalar, no level consumed
	rotate,		 // ckks.rotate {static_shift}
	bootstrap,	 // ckks.bootstrap
};

const char *slot_op_name(slot_op_t op);

struct slot_value_t {
	slot_id_t id = no_slot;
	slot_op_t op = slot_op_t::argument;
	std::vector<slot_id_t> inputs;
	int rotation = 0;	  // rotate
	double scalar = 0.0;  // mul_scalar
	int constant = -1;	  // mul_plain, add_plain: index into constants()
	int level = 0;		  // levels consumed on the path to this value
	std::string name;	  // arguments and results
};

// A public constant vector, encoded once and reused.
struct slot_constant_t {
	std::string name;
	std::vector<double> values;
	bool splat = false;	 // every slot identical: emitted as dense<c>
};

class slot_graph_t {
   public:
	explicit slot_graph_t(const layout_t &layout) : layout_(layout) {}

	const layout_t &layout() const { return layout_; }
	const std::vector<slot_value_t> &values() const { return values_; }
	const std::vector<slot_constant_t> &constants() const { return constants_; }
	const slot_value_t &value(slot_id_t id) const { return values_.at(id); }
	size_t size() const { return values_.size(); }

	slot_id_t argument(const std::string &name);
	slot_id_t add(slot_id_t a, slot_id_t b);
	slot_id_t sub(slot_id_t a, slot_id_t b);
	slot_id_t mul(slot_id_t a, slot_id_t b);
	slot_id_t mul_plain(slot_id_t a, int constant);
	slot_id_t add_plain(slot_id_t a, int constant);
	slot_id_t mul_scalar(slot_id_t a, double s);
	slot_id_t rotate(slot_id_t a, int shift);
	slot_id_t bootstrap(slot_id_t a);

	// Summation trees.
	slot_id_t sum_rows(slot_id_t a);
	slot_id_t sum_cols(slot_id_t a);

	// Constant pool.
	int splat_constant(double value);
	int first_column_mask_constant();

	void set_name(slot_id_t id, const std::string &name);

	// Every rotation index the graph uses, sorted.
	std::vector<int> rotation_indices() const;
	int max_level() const;
	std::string statistics() const;

   private:
	slot_id_t push(slot_value_t v);

	layout_t layout_;
	std::vector<slot_value_t> values_;
	std::vector<slot_constant_t> constants_;
	int mask_constant_ = -1;
};

struct lowered_step_t {
	slot_graph_t graph;
	std::vector<slot_id_t> arguments;
	std::vector<std::string> argument_names;
	std::vector<slot_id_t> results;
	std::vector<std::string> result_names;

	explicit lowered_step_t(const layout_t &layout) : graph(layout) {}
};

// Lower a differentiated training step to slot operations.
lowered_step_t lower_to_slots(const train_step_t &step);

}  // namespace reboot

#endif	// REBOOT_SLOT_GRAPH_H_
