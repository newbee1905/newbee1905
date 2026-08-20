// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/slot_graph.h"

#include <fmt/format.h>

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace reboot {

const char *slot_op_name(SlotOp op) {
	switch (op) {
		case SlotOp::kArgument:
			return "argument";
		case SlotOp::kAdd:
			return "add";
		case SlotOp::kSub:
			return "sub";
		case SlotOp::kMul:
			return "mul";
		case SlotOp::kMulPlain:
			return "mul_plain";
		case SlotOp::kAddPlain:
			return "add_plain";
		case SlotOp::kMulScalar:
			return "mul_scalar";
		case SlotOp::kRotate:
			return "rotate";
		case SlotOp::kBootstrap:
			return "bootstrap";
	}
	return "unknown";
}

SlotId SlotGraph::push(SlotValue v) {
	v.id = static_cast<SlotId>(values_.size());
	values_.push_back(std::move(v));
	return values_.back().id;
}

void SlotGraph::set_name(SlotId id, const std::string &name) {
	values_.at(id).name = name;
}

SlotId SlotGraph::argument(const std::string &name) {
	SlotValue v;
	v.op = SlotOp::kArgument;
	v.name = name;
	return push(std::move(v));
}

namespace {

int deeper_of(const std::vector<SlotValue> &values,
			  const std::vector<SlotId> &inputs) {
	int level = 0;
	for (SlotId in : inputs) level = std::max(level, values.at(in).level);
	return level;
}

}  // namespace

SlotId SlotGraph::add(SlotId a, SlotId b) {
	SlotValue v;
	v.op = SlotOp::kAdd;
	v.inputs = {a, b};
	v.level = deeper_of(values_, v.inputs);
	return push(std::move(v));
}

SlotId SlotGraph::sub(SlotId a, SlotId b) {
	SlotValue v;
	v.op = SlotOp::kSub;
	v.inputs = {a, b};
	v.level = deeper_of(values_, v.inputs);
	return push(std::move(v));
}

SlotId SlotGraph::mul(SlotId a, SlotId b) {
	SlotValue v;
	v.op = SlotOp::kMul;
	v.inputs = {a, b};
	v.level = deeper_of(values_, v.inputs) + 1;
	return push(std::move(v));
}

SlotId SlotGraph::mul_plain(SlotId a, int constant) {
	SlotValue v;
	v.op = SlotOp::kMulPlain;
	v.inputs = {a};
	v.constant = constant;
	v.level = values_.at(a).level + 1;
	return push(std::move(v));
}

SlotId SlotGraph::add_plain(SlotId a, int constant) {
	SlotValue v;
	v.op = SlotOp::kAddPlain;
	v.inputs = {a};
	v.constant = constant;
	v.level = values_.at(a).level;
	return push(std::move(v));
}

SlotId SlotGraph::mul_scalar(SlotId a, double s) {
	// ckks.mul_scalar multiplies by a real constant without rescaling, so the
	// optimiser's learning rate and momentum factors cost no depth.
	SlotValue v;
	v.op = SlotOp::kMulScalar;
	v.inputs = {a};
	v.scalar = s;
	v.level = values_.at(a).level;
	return push(std::move(v));
}

SlotId SlotGraph::rotate(SlotId a, int shift) {
	if (shift == 0) return a;
	SlotValue v;
	v.op = SlotOp::kRotate;
	v.inputs = {a};
	v.rotation = shift;
	v.level = values_.at(a).level;
	return push(std::move(v));
}

SlotId SlotGraph::bootstrap(SlotId a) {
	SlotValue v;
	v.op = SlotOp::kBootstrap;
	v.inputs = {a};
	v.level = 0;
	return push(std::move(v));
}

SlotId SlotGraph::sum_rows(SlotId a) {
	SlotId acc = a;
	for (int k = layout_.cols; k < layout_.slots(); k <<= 1)
		acc = add(acc, rotate(acc, k));
	return acc;
}

SlotId SlotGraph::sum_cols(SlotId a) {
	SlotId acc = a;
	for (int k = 1; k < layout_.cols; k <<= 1) acc = add(acc, rotate(acc, k));
	acc = mul_plain(acc, first_column_mask_constant());
	for (int k = 1; k < layout_.cols; k <<= 1) acc = add(acc, rotate(acc, -k));
	return acc;
}

int SlotGraph::splat_constant(double value) {
	for (size_t i = 0; i < constants_.size(); ++i) {
		if (constants_[i].splat && !constants_[i].values.empty() &&
			constants_[i].values.front() == value)
			return static_cast<int>(i);
	}
	SlotConstant c;
	c.name = fmt::format("splat_{}", constants_.size());
	c.values.assign(static_cast<size_t>(layout_.slots()), value);
	c.splat = true;
	constants_.push_back(std::move(c));
	return static_cast<int>(constants_.size()) - 1;
}

int SlotGraph::first_column_mask_constant() {
	if (mask_constant_ >= 0) return mask_constant_;
	SlotConstant c;
	c.name = "first_column_mask";
	c.values = first_column_mask(layout_);
	constants_.push_back(std::move(c));
	mask_constant_ = static_cast<int>(constants_.size()) - 1;
	return mask_constant_;
}

std::vector<int> SlotGraph::rotation_indices() const {
	std::set<int> indices;
	for (const SlotValue &v : values_)
		if (v.op == SlotOp::kRotate) indices.insert(v.rotation);
	return std::vector<int>(indices.begin(), indices.end());
}

int SlotGraph::max_level() const {
	int level = 0;
	for (const SlotValue &v : values_) level = std::max(level, v.level);
	return level;
}

std::string SlotGraph::statistics() const {
	std::map<std::string, int> counts;
	for (const SlotValue &v : values_) ++counts[slot_op_name(v.op)];
	std::string out = fmt::format("slot graph: {} values, {} constants\n",
								  values_.size(), constants_.size());
	for (const auto &[name, count] : counts)
		out += fmt::format("  {:<10} {}\n", name, count);
	const std::vector<int> indices = rotation_indices();
	out += fmt::format("  distinct rotation indices: {}\n", indices.size());
	out += fmt::format("  levels consumed          : {}\n", max_level());
	return out;
}

LoweredStep lower_to_slots(const TrainStep &step) {
	LoweredStep lowered(step.layout);
	SlotGraph &sg = lowered.graph;
	const TensorGraph &tg = step.graph;

	std::vector<SlotId> mapping(tg.size(), kNoSlot);

	// Arguments first and in the order the model declared them, so the emitted
	// function signature is stable.
	for (ValueId id : step.arguments) {
		const TensorValue &v = tg.value(id);
		mapping[id] = sg.argument(v.name);
		lowered.arguments.push_back(mapping[id]);
		lowered.argument_names.push_back(v.name);
	}

	for (ValueId id : tg.topological_order()) {
		if (mapping[id] != kNoSlot) continue;
		const TensorValue &v = tg.value(id);
		SlotId out = kNoSlot;

		switch (v.op) {
			case TensorOp::kInput:
			case TensorOp::kParam:
				throw std::runtime_error(fmt::format(
					"leaf '{}' is not among the declared arguments", v.name));

			case TensorOp::kMatMul: {
				const SlotId product =
					sg.mul(mapping[v.inputs[0]], mapping[v.inputs[1]]);
				out = tg.value(v.inputs[1]).row_packing ? sg.sum_rows(product)
														: sg.sum_cols(product);
				break;
			}

			case TensorOp::kMatMulT: {
				// The transpose is the same product summed the other way: no
				// transposition of the weight ciphertext, no repacking.
				const SlotId product =
					sg.mul(mapping[v.inputs[0]], mapping[v.inputs[1]]);
				out = tg.value(v.inputs[1]).row_packing ? sg.sum_cols(product)
														: sg.sum_rows(product);
				break;
			}

			case TensorOp::kOuter:
				// One operand is Expanded and the other Repeated, so their
				// elementwise product already is the outer product in the
				// weight layout - no rotation at all.
				out = sg.mul(mapping[v.inputs[0]], mapping[v.inputs[1]]);
				break;

			case TensorOp::kAdd:
				out = sg.add(mapping[v.inputs[0]], mapping[v.inputs[1]]);
				break;
			case TensorOp::kSub:
				out = sg.sub(mapping[v.inputs[0]], mapping[v.inputs[1]]);
				break;
			case TensorOp::kHadamard:
				out = sg.mul(mapping[v.inputs[0]], mapping[v.inputs[1]]);
				break;
			case TensorOp::kScale:
				out = sg.mul_scalar(mapping[v.inputs[0]], v.scalar);
				break;
			case TensorOp::kAddScalar:
				out = sg.add_plain(mapping[v.inputs[0]],
								   sg.splat_constant(v.scalar));
				break;
			case TensorOp::kSquare:
				out = sg.mul(mapping[v.inputs[0]], mapping[v.inputs[0]]);
				break;
			case TensorOp::kPolyRelu: {
				const SlotId x = mapping[v.inputs[0]];
				out = sg.add(sg.mul(x, x), x);
				break;
			}
			case TensorOp::kPolyReluGrad: {
				const SlotId g = mapping[v.inputs[0]];
				const SlotId x = mapping[v.inputs[1]];
				const SlotId shifted =
					sg.add_plain(sg.mul_scalar(x, 2.0), sg.splat_constant(1.0));
				out = sg.mul(g, shifted);
				break;
			}
			case TensorOp::kStopGradient:
				// Forward-transparent: the block boundary only matters to the
				// autograd pass, and by now differentiation has happened.
				out = mapping[v.inputs[0]];
				break;
			case TensorOp::kBootstrap:
				out = sg.bootstrap(mapping[v.inputs[0]]);
				break;
		}

		mapping[id] = out;
		if (!v.name.empty()) sg.set_name(out, v.name);
	}

	for (ValueId id : step.results) {
		lowered.results.push_back(mapping[id]);
		lowered.result_names.push_back(tg.value(id).name);
	}
	return lowered;
}

}  // namespace reboot
