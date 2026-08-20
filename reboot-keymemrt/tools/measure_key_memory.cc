// SPDX-License-Identifier: GPL-3.0-or-later
//
// Measures the rotation-key working set of a ReBoot training step two ways.
//
//   ReBoot   the key set lib/cryptocontext.py builds - EvalMultKeyGen,
//            EvalSumKeyGen, EvalSumRowsKeyGen(col_size), EvalSumColsKeyGen -
//            all of it resident for the whole run, because the rotations
//            happen inside OpenFHE and nothing names their indices.
//
//   KeyMemRT the indices the emitted module names, generated, compressed to
//            the level they are used at, written one file each and dropped;
//            at run time one key is resident at a time.
//
// Both cases use the same context, so the comparison is the key material
// itself.  Sizes are measured by walking each key's RNS limbs rather than
// estimated, and process RSS is sampled around each phase as a cross-check.
//
// Build: see scripts/measure_key_memory.sh (needs OpenFHE and KeyMemRT).

#include <fmt/format.h>

#include <algorithm>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "KeyMemRT.hpp"
#include "openfhe.h"
#include "reboot/options.h"
#include "reboot/reboot_model.h"
#include "reboot/slot_graph.h"

using namespace lbcrypto;
using namespace reboot;

// The generated code refers to these; a measurement binary still has to define
// them because it links KeyMemRT.
KeyMemRT keymem_rt;

namespace {

double rss_mb() {
	std::ifstream status("/proc/self/status");
	std::string line;
	while (std::getline(status, line)) {
		if (line.rfind("VmRSS:", 0) == 0) {
			return std::stod(line.substr(6)) / 1024.0;
		}
	}
	return 0.0;
}

double peak_rss_mb() {
	std::ifstream status("/proc/self/status");
	std::string line;
	while (std::getline(status, line)) {
		if (line.rfind("VmHWM:", 0) == 0) {
			return std::stod(line.substr(6)) / 1024.0;
		}
	}
	return 0.0;
}

// Bytes a switching key actually occupies: every RNS limb of every digit, in
// both the a and b vectors.
size_t key_bytes(const EvalKey<DCRTPoly> &key) {
	size_t bytes = 0;
	for (const auto &vector : {key->GetAVector(), key->GetBVector()})
		for (const DCRTPoly &poly : vector)
			for (const auto &limb : poly.GetAllElements())
				bytes += limb.GetLength() * sizeof(uint64_t);
	return bytes;
}

size_t map_bytes(const std::map<usint, EvalKey<DCRTPoly>> &keys) {
	size_t bytes = 0;
	for (const auto &[index, key] : keys)
		if (key) bytes += key_bytes(key);
	return bytes;
}

// EvalSumRowsKeyGen and EvalSumColsKeyGen put their keys in the EvalSum map and
// in the automorphism map, and the two maps share the objects - so the union
// has to be taken by identity, not by summing both maps.
struct key_census_t {
	size_t bytes = 0;
	size_t count = 0;
};

key_census_t census(
	const std::vector<const std::map<usint, EvalKey<DCRTPoly>> *> &maps) {
	std::set<const void *> seen;
	key_census_t result;
	for (const auto *keys : maps)
		for (const auto &[index, key] : *keys) {
			if (!key || !seen.insert(key.get()).second) continue;
			result.bytes += key_bytes(key);
			++result.count;
		}
	return result;
}

double mb(size_t bytes) {
	return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char **argv) {
	model_config_t config;
	config.hidden = {32, 16};
	config.input_dim = 16;
	config.num_classes = 4;
	config.batch_size = 1;

	int log_n = 14, depth_override = 0, scaling_mod = 50;
	std::string key_dir = "./keys_measure";

	option_parser_t parser(
		"measure_key_memory",
		"compare the rotation-key working set of ReBoot and KeyMemRT");
	add_model_options(parser, config, log_n);
	parser.section("CKKS")
		.add("--depth", depth_override,
			 "multiplicative depth (default: from the graph)")
		.add("--scaling-mod", scaling_mod, "scaling modulus size in bits");
	parser.section("Output").add("--key-dir", key_dir,
								 "where the serialised keys go");

	try {
		if (!parser.parse(argc, argv)) return 0;
	} catch (const std::exception &error) {
		fmt::print(stderr, "error: {}\n", error.what());
		return 1;
	}

	// ---- the step, and the rotation indices it names -----------------------
	const int num_slots = 1 << (log_n - 1);
	const layout_t layout = recommend_layout(config, num_slots);
	const train_step_t step = build_train_step(config, layout);
	const lowered_step_t lowered = lower_to_slots(step);
	const std::vector<int> indices = lowered.graph.rotation_indices();
	const int depth =
		depth_override > 0 ? depth_override : lowered.graph.max_level() + 1;

	fmt::print("{}", step.describe());
	fmt::print(
		"rotations in one step   : {}\n"
		"distinct indices        : {}\n"
		"multiplicative depth    : {}\n\n",
		[&] {
			size_t n = 0;
			for (const slot_value_t &v : lowered.graph.values())
				if (v.op == slot_op_t::rotate) ++n;
			return n;
		}(),
		indices.size(), depth);

	// ---- context -----------------------------------------------------------
	CCParams<CryptoContextCKKSRNS> params;
	params.SetMultiplicativeDepth(static_cast<usint>(depth));
	params.SetScalingModSize(static_cast<usint>(scaling_mod));
	params.SetFirstModSize(static_cast<usint>(scaling_mod + 1));
	params.SetScalingTechnique(FLEXIBLEAUTO);
	params.SetRingDim(static_cast<usint>(1 << log_n));
	params.SetSecurityLevel(HEStd_NotSet);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	KeyPair<DCRTPoly> key_pair = cc->KeyGen();
	const std::string tag = key_pair.secretKey->GetKeyTag();

	const double rss_context = rss_mb();
	fmt::print(
		"context: ring {}, {} slots, depth {}\n"
		"  RSS after key generation setup: {:.1f} MB\n\n",
		cc->GetRingDimension(), num_slots, depth, rss_context);

	// ---- ReBoot: everything resident ---------------------------------------
	// The calls are the ones in ReBoot's lib/cryptocontext.py, in the same
	// order, with the same row size.
	cc->EvalMultKeyGen(key_pair.secretKey);
	cc->EvalSumKeyGen(key_pair.secretKey);
	cc->EvalSumRowsKeyGen(key_pair.secretKey, nullptr,
						  static_cast<usint>(layout.cols));
	cc->EvalSumColsKeyGen(key_pair.secretKey);

	const auto &sum_keys = CryptoContextImpl<DCRTPoly>::GetEvalSumKeyMap(tag);
	const auto &auto_keys =
		CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(tag);
	const key_census_t reboot = census({&sum_keys, &auto_keys});
	const size_t reboot_bytes = reboot.bytes;
	const size_t reboot_keys = reboot.count;
	const double rss_reboot = rss_mb();

	fmt::print(
		"ReBoot (EvalSumRows/EvalSumCols, all keys resident)\n"
		"  keys resident         : {}\n"
		"  key material          : {:.1f} MB\n"
		"  process RSS           : {:.1f} MB (+{:.1f} MB)\n\n",
		reboot_keys, mb(reboot_bytes), rss_reboot, rss_reboot - rss_context);

	// Per-key size, for the extrapolation to the paper's parameters.
	size_t one_key = 0;
	for (const auto &[index, key] : auto_keys)
		if (key) {
			one_key = key_bytes(key);
			break;
		}

	// ---- KeyMemRT: one key at a time ---------------------------------------
	cc->ClearEvalAutomorphismKeys(tag);
	keymem_rt.setCryptoContext(cc);
	keymem_rt.setKeyTag(tag);
	keymem_rt.setMultDepth(depth);
	keymem_rt.setKeyMemMode(KeyMemMode::IMPERATIVE);
	BenchmarkCLI::setInputDir(key_dir);

	std::vector<int32_t> rotation_indices(indices.begin(), indices.end());
	keymem_rt.addRotIndices(rotation_indices);
	cc->EvalRotateKeyGen(key_pair.secretKey, rotation_indices);
	const size_t named_bytes =
		map_bytes(CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(tag));
	keymem_rt.serializeKeysAtLevel(rotation_indices, 0);
	keymem_rt.clearAllKeys();
	const double rss_cleared = rss_mb();

	// One rotation, the way the generated code does it.
	const RotKey loaded = keymem_rt.deserializeKey(rotation_indices.front(), 0);
	const size_t resident_bytes =
		map_bytes(CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(tag));
	const double rss_one_key = rss_mb();
	keymem_rt.clearKey(loaded);

	fmt::print(
		"KeyMemRT (named rotations, one key paged in at a time)\n"
		"  keys the step names   : {} ({:.1f} MB if they were all resident)\n"
		"  keys resident         : 1\n"
		"  key material          : {:.1f} MB\n"
		"  process RSS           : {:.1f} MB (cleared: {:.1f} MB)\n\n",
		rotation_indices.size(), mb(named_bytes), mb(resident_bytes),
		rss_one_key, rss_cleared);

	// ---- what level compression buys ---------------------------------------
	// KeyMemRT stores each key truncated to the ciphertext level it is used at,
	// which is the fork's dynamic-Q-size feature.  Deeper in the step means
	// fewer towers and a smaller key.
	std::set<int> levels;
	for (const slot_value_t &v : lowered.graph.values())
		if (v.op == slot_op_t::rotate) levels.insert(v.level);

	fmt::print("key size against the level it is used at\n");
	size_t compressed_total = 0;
	for (int level : levels) {
		cc->ClearEvalAutomorphismKeys(tag);
		cc->EvalRotateKeyGen(key_pair.secretKey, {rotation_indices.front()});
		keymem_rt.serializeKeysAtLevel({rotation_indices.front()}, level);
		const size_t bytes = map_bytes(
			CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(tag));
		fmt::print("  level {:2d}: {:6.1f} MB\n", level, mb(bytes));
		compressed_total = std::max(compressed_total, bytes);
	}
	cc->ClearEvalAutomorphismKeys(tag);
	fmt::print("  worst case resident   : {:.1f} MB\n\n", mb(compressed_total));

	fmt::print(
		"comparison\n"
		"  one switching key     : {:.1f} MB\n"
		"  ReBoot resident       : {:.1f} MB over {} keys\n"
		"  KeyMemRT resident     : {:.1f} MB over 1 key\n"
		"  reduction             : {:.1f}x\n"
		"  peak process RSS      : {:.1f} MB\n",
		mb(one_key), mb(reboot_bytes), reboot_keys, mb(resident_bytes),
		resident_bytes > 0 ? static_cast<double>(reboot_bytes) /
								 static_cast<double>(resident_bytes)
						   : 0.0,
		peak_rss_mb());
	return 0;
}
