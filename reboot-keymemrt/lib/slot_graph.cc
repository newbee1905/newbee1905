// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/slot_graph.h"

#include <fmt/format.h>

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace reboot {

const char *slot_op_name(slot_op_t op) {
	switch (op) {
		case slot_op_t::argument:
			return "argument";
		case slot_op_t::add:
			return "add";
		case slot_op_t::sub:
			return "sub";
		case slot_op_t::mul:
			return "mul";
		case slot_op_t::mul_plain:
			return "mul_plain";
		case slot_op_t::add_plain:
			return "add_plain";
		case slot_op_t::mul_scalar:
			return "mul_scalar";
		case slot_op_t::rotate:
			return "rotate";
		case slot_op_t::bootstrap:
			return "bootstrap";
	}
	return "unknown";
}

slot_id_t slot_graph_t::push(slot_value_t v) {
	v.id = static_cast<slot_id_t>(values_.size());
	values_.push_back(std::move(v));
	return values_.back().id;
}

void slot_graph_t::set_name(slot_id_t id, const std::string &name) {
	values_.at(id).name = name;
}

slot_id_t slot_graph_t::argument(const std::string &name) {
	slot_value_t v;
	v.op = slot_op_t::argument;
	v.name = name;
	return push(std::move(v));
}

namespace {

int deeper_of(const std::vector<slot_value_t> &values,
			  const std::vector<slot_id_t> &inputs) {
	int level = 0;
	for (slot_id_t in : inputs) level = std::max(level, values.at(in).level);
	return level;
}

}  // namespace

slot_id_t slot_graph_t::add(slot_id_t a, slot_id_t b) {
	slot_value_t v;
	v.op = slot_op_t::add;
	v.inputs = {a, b};
	v.level = deeper_of(values_, v.inputs);
	return push(std::move(v));
}

slot_id_t slot_graph_t::sub(slot_id_t a, slot_id_t b) {
	slot_value_t v;
	v.op = slot_op_t::sub;
	v.inputs = {a, b};
	v.level = deeper_of(values_, v.inputs);
	return push(std::move(v));
}

slot_id_t slot_graph_t::mul(slot_id_t a, slot_id_t b) {
	slot_value_t v;
	v.op = slot_op_t::mul;
	v.inputs = {a, b};
	v.level = deeper_of(values_, v.inputs) + 1;
	return push(std::move(v));
}

slot_id_t slot_graph_t::mul_plain(slot_id_t a, int constant) {
	slot_value_t v;
	v.op = slot_op_t::mul_plain;
	v.inputs = {a};
	v.constant = constant;
	v.level = values_.at(a).level + 1;
	return push(std::move(v));
}

slot_id_t slot_graph_t::add_plain(slot_id_t a, int constant) {
	slot_value_t v;
	v.op = slot_op_t::add_plain;
	v.inputs = {a};
	v.constant = constant;
	v.level = values_.at(a).level;
	return push(std::move(v));
}

slot_id_t slot_graph_t::mul_scalar(slot_id_t a, double s) {
	// ckks.mul_scalar multiplies by a real constant without rescaling, so the
	// optimiser's learning rate and momentum factors cost no depth.
	slot_value_t v;
	v.op = slot_op_t::mul_scalar;
	v.inputs = {a};
	v.scalar = s;
	v.level = values_.at(a).level;
	return push(std::move(v));
}

slot_id_t slot_graph_t::rotate(slot_id_t a, int shift) {
	if (shift == 0) return a;
	slot_value_t v;
	v.op = slot_op_t::rotate;
	v.inputs = {a};
	v.rotation = shift;
	v.level = values_.at(a).level;
	return push(std::move(v));
}

slot_id_t slot_graph_t::bootstrap(slot_id_t a) {
	slot_value_t v;
	v.op = slot_op_t::bootstrap;
	v.inputs = {a};
	v.level = 0;
	return push(std::move(v));
}

slot_id_t slot_graph_t::sum_rows(slot_id_t a) {
	slot_id_t acc = a;
	for (int k = layout_.cols; k < layout_.slots(); k <<= 1)
		acc = add(acc, rotate(acc, k));
	return acc;
}

slot_id_t slot_graph_t::sum_cols(slot_id_t a) {
	slot_id_t acc = a;
	for (int k = 1; k < layout_.cols; k <<= 1) acc = add(acc, rotate(acc, k));
	acc = mul_plain(acc, first_column_mask_constant());
	for (int k = 1; k < layout_.cols; k <<= 1) acc = add(acc, rotate(acc, -k));
	return acc;
}

int slot_graph_t::splat_constant(double value) {
	for (size_t i = 0; i < constants_.size(); ++i) {
		if (constants_[i].splat && !constants_[i].values.empty() &&
			constants_[i].values.front() == value)
			return static_cast<int>(i);
	}
	slot_constant_t c;
	c.name = fmt::format("splat_{}", constants_.size());
	c.values.assign(static_cast<size_t>(layout_.slots()), value);
	c.splat = true;
	constants_.push_back(std::move(c));
	return static_cast<int>(constants_.size()) - 1;
}

int slot_graph_t::first_column_mask_constant() {
	if (mask_constant_ >= 0) return mask_constant_;
	slot_constant_t c;
	c.name = "first_column_mask";
	c.values = first_column_mask(layout_);
	constants_.push_back(std::move(c));
	mask_constant_ = static_cast<int>(constants_.size()) - 1;
	return mask_constant_;
}

std::vector<int> slot_graph_t::rotation_indices() const {
	std::set<int> indices;
	for (const slot_value_t &v : values_)
		if (v.op == slot_op_t::rotate) indices.insert(v.rotation);
	return std::vector<int>(indices.begin(), indices.end());
}

int slot_graph_t::max_level() const {
	int level = 0;
	for (const slot_value_t &v : values_) level = std::max(level, v.level);
	return level;
}

std::string slot_graph_t::statistics() const {
	std::map<std::string, int> counts;
	for (const slot_value_t &v : values_) ++counts[slot_op_name(v.op)];
	std::string out = fmt::format("slot graph: {} values, {} constants\n",
								  values_.size(), constants_.size());
	for (const auto &[name, count] : counts)
		out += fmt::format("  {:<10} {}\n", name, count);
	const std::vector<int> indices = rotation_indices();
	out += fmt::format("  distinct rotation indices: {}\n", indices.size());
	out += fmt::format("  levels consumed          : {}\n", max_level());
	return out;
}

lowered_step_t lower_to_slots(const train_step_t &step) {
	lowered_step_t lowered(step.layout);
	slot_graph_t &sg = lowered.graph;
	const tensor_graph_t &tg = step.graph;

	std::vector<slot_id_t> mapping(tg.size(), no_slot);

	// Arguments first and in the order the model declared them, so the emitted
	// function signature is stable.
	for (value_id_t id : step.arguments) {
		const tensor_value_t &v = tg.value(id);
		mapping[id] = sg.argument(v.name);
		lowered.arguments.push_back(mapping[id]);
		lowered.argument_names.push_back(v.name);
	}

	for (value_id_t id : tg.topological_order()) {
		if (mapping[id] != no_slot) continue;
		const tensor_value_t &v = tg.value(id);
		slot_id_t out = no_slot;

		switch (v.op) {
			case tensor_op_t::input:
			case tensor_op_t::param:
				throw std::runtime_error(fmt::format(
					"leaf '{}' is not among the declared arguments", v.name));

			case tensor_op_t::matmul: {
				const slot_id_t product =
					sg.mul(mapping[v.inputs[0]], mapping[v.inputs[1]]);
				out = tg.value(v.inputs[1]).row_packing ? sg.sum_rows(product)
														: sg.sum_cols(product);
				break;
			}

			case tensor_op_t::matmul_transposed: {
				// The transpose is the same product summed the other way: no
				// transposition of the weight ciphertext, no repacking.
				const slot_id_t product =
					sg.mul(mapping[v.inputs[0]], mapping[v.inputs[1]]);
				out = tg.value(v.inputs[1]).row_packing ? sg.sum_cols(product)
														: sg.sum_rows(product);
				break;
			}

			case tensor_op_t::outer:
				// One operand is Expanded and the other Repeated, so their
				// elementwise product already is the outer product in the
				// weight layout - no rotation at all.
				out = sg.mul(mapping[v.inputs[0]], mapping[v.inputs[1]]);
				break;

			case tensor_op_t::add:
				out = sg.add(mapping[v.inputs[0]], mapping[v.inputs[1]]);
				break;
			case tensor_op_t::sub:
				out = sg.sub(mapping[v.inputs[0]], mapping[v.inputs[1]]);
				break;
			case tensor_op_t::hadamard:
				out = sg.mul(mapping[v.inputs[0]], mapping[v.inputs[1]]);
				break;
			case tensor_op_t::scale:
				out = sg.mul_scalar(mapping[v.inputs[0]], v.scalar);
				break;
			case tensor_op_t::add_scalar:
				out = sg.add_plain(mapping[v.inputs[0]],
								   sg.splat_constant(v.scalar));
				break;
			case tensor_op_t::square:
				out = sg.mul(mapping[v.inputs[0]], mapping[v.inputs[0]]);
				break;
			case tensor_op_t::poly_relu: {
				const slot_id_t x = mapping[v.inputs[0]];
				out = sg.add(sg.mul(x, x), x);
				break;
			}
			case tensor_op_t::poly_relu_grad: {
				const slot_id_t g = mapping[v.inputs[0]];
				const slot_id_t x = mapping[v.inputs[1]];
				const slot_id_t shifted =
					sg.add_plain(sg.mul_scalar(x, 2.0), sg.splat_constant(1.0));
				out = sg.mul(g, shifted);
				break;
			}
			case tensor_op_t::stop_gradient:
				// Forward-transparent: the block boundary only matters to the
				// autograd pass, and by now differentiation has happened.
				out = mapping[v.inputs[0]];
				break;
			case tensor_op_t::bootstrap:
				out = sg.bootstrap(mapping[v.inputs[0]]);
				break;
		}

		mapping[id] = out;
		if (!v.name.empty()) sg.set_name(out, v.name);
	}

	for (value_id_t id : step.results) {
		lowered.results.push_back(mapping[id]);
		lowered.result_names.push_back(tg.value(id).name);
	}
	return lowered;
}

}  // namespace reboot
