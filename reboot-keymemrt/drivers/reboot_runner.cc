// SPDX-License-Identifier: GPL-3.0-or-later
//
// Host driver for the generated code.
//
// `reboot_emit` produces MLIR, `keymemrt-opt` places the key management and
// `keymemrt-translate` turns it into a C++ translation unit.  That unit is a
// library: it needs a host that builds the crypto context, generates and
// serialises the keys, encrypts the batch in the right packing, and drives the
// step in a loop.  This is that host.
//
// It links against the same frontend library that emitted the module, so the
// argument order, the slot layout and the packing of every tensor come from one
// place instead of being duplicated by hand.  Pass it the *same* model flags
// used for `reboot_emit`; scripts/run_keymemrt.sh does that for you.
//
// Build: this file is not part of the default CMake build, because it only
// compiles once the generated .cc exists.  See scripts/run_keymemrt.sh.

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "KeyMemRT.hpp"
#include "ResourceMonitor.hpp"
#include "openfhe.h"
#include "reboot/data.h"
#include "reboot/layout.h"
#include "reboot/manifest.h"
#include "reboot/options.h"
#include "reboot/reboot_model.h"
#include "reboot/slot_graph.h"

using namespace lbcrypto;
using namespace reboot;

using CiphertextT = ConstCiphertext<DCRTPoly>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;

// Globals the generated translation unit refers to: `extern KeyMemRT
// keymem_rt;` comes from the module prelude, `monitor` from generic_header.h.
KeyMemRT keymem_rt;
std::unique_ptr<ResourceMonitor> monitor;

// Emitted by keymemrt-translate.  A ckks function taking and returning one
// tensor of ciphertexts becomes std::vector<CiphertextT>; the two context
// helpers come from --openfhe-configure-crypto-context.  If the translator
// spells any of these differently, this is the one place to adjust.
extern "C++" {
CryptoContextT reboot_train_step__generate_crypto_context();
CryptoContextT reboot_train_step__configure_crypto_context(CryptoContextT cc,
														   PrivateKeyT sk);
std::vector<CiphertextT> reboot_train_step(CryptoContextT cc,
										   std::vector<CiphertextT> args);
}

namespace {

int argmax(const std::vector<double> &v) {
	int best = 0;
	for (size_t i = 1; i < v.size(); ++i)
		if (v[i] > v[static_cast<size_t>(best)]) best = static_cast<int>(i);
	return best;
}

}  // namespace

int main(int argc, char **argv) {
	keymem_rt.initFromArgs(argc, argv);

	std::string manifest_path, csv, result_dir = "./results";
	int samples = 64, steps = 4;

	OptionParser parser(
		"reboot_runner",
		"run the generated ReBoot training step under KeyMemRT");
	parser.section("Module").add(
		"--manifest", manifest_path,
		"manifest reboot_emit wrote beside the .mlir (required)");
	parser.section("Run")
		.add("--samples", samples, "synthetic samples")
		.add("--csv", csv, "CSV dataset instead (last column = label)")
		.add("--steps", steps, "training steps to run")
		.add("--result-dir", result_dir, "resource monitor output directory");
	// KeyMemRT reads --key-mode, --input-dir, --prefetch-sat and friends
	// straight from argv; they are declared here so they are skipped rather
	// than reported as unknown.
	ignore_keymemrt_options(parser);

	Manifest manifest;
	try {
		if (!parser.parse(argc, argv)) return 0;
		if (manifest_path.empty())
			throw std::invalid_argument(
				"--manifest is required: it is what pins this run to the "
				"module that was generated, so the argument order cannot "
				"silently disagree");
		manifest = Manifest::load(manifest_path);
	} catch (const std::exception &error) {
		fmt::print(stderr, "error: {}\n", error.what());
		return 1;
	}

	// The model comes from the manifest, never from flags: the generated
	// function has a fixed argument order and a mistyped width here would
	// build a different one.
	ModelConfig config = manifest.config;
	const int log_n = manifest.log_n;
	const Layout layout = manifest.layout;
	fmt::print("{}", manifest.describe());

	// ---- schema: the same graph the emitted module was built from ----------
	Dataset data = csv.empty() ? make_blobs(samples, config.input_dim,
											config.num_classes, /*seed=*/5)
							   : load_csv(csv);
	if (!csv.empty()) {
		normalise(data);
		if (data.dim != config.input_dim ||
			data.num_classes != config.num_classes) {
			fmt::print(stderr,
					   "error: {} has dim {} and {} classes, but the module "
					   "was generated for dim {} and {} classes\n",
					   csv, data.dim, data.num_classes, config.input_dim,
					   config.num_classes);
			return 1;
		}
	}

	const int num_slots = 1 << (log_n - 1);
	const TrainStep step = build_train_step(config, layout);
	const LoweredStep lowered = lower_to_slots(step);
	const TensorGraph &g = step.graph;
	try {
		manifest.verify(lowered);
	} catch (const std::exception &error) {
		fmt::print(stderr, "error: {}\n", error.what());
		return 1;
	}
	fmt::print("{}", step.describe());

	// ---- context and keys --------------------------------------------------
	// The generated helpers own the parameters and the key schedule; the driver
	// only supplies the secret key and tells KeyMemRT which side it is on.
	CryptoContextT cc = reboot_train_step__generate_crypto_context();
	KeyPair<DCRTPoly> key_pair = cc->KeyGen();
	const PublicKeyT public_key = key_pair.publicKey;
	const PrivateKeyT secret_key = key_pair.secretKey;

	keymem_rt.setCryptoContext(cc);
	keymem_rt.setKeyTag(secret_key->GetKeyTag());
	keymem_rt.setMultDepth(step.required_depth);
	keymem_rt.setPlatform(Platform::CLIENT);

	fmt::print("generating and serialising keys ...\n");
	cc = reboot_train_step__configure_crypto_context(cc, secret_key);
	keymem_rt.setPlatform(Platform::SERVER);

	// ---- encrypted state ---------------------------------------------------
	auto encrypt_slots = [&](const std::vector<double> &slots) {
		std::vector<double> padded = slots;
		padded.resize(static_cast<size_t>(num_slots), 0.0);
		return cc->Encrypt(public_key, cc->MakeCKKSPackedPlaintext(padded));
	};
	auto decrypt_slots = [&](const CiphertextT &ct) {
		Plaintext pt;
		cc->Decrypt(secret_key, ct, &pt);
		pt->SetLength(static_cast<size_t>(num_slots));
		return pt->GetRealPackedValue();
	};

	std::mt19937 rng(1);
	std::map<std::string, Ciphertext<DCRTPoly>> state;
	for (const ParamBinding &p : step.params) {
		const TensorValue &meta = g.value(p.weight);
		const double bound =
			std::sqrt(1.0 / (meta.shape.rows + meta.shape.cols));
		std::uniform_real_distribution<double> dist(-bound, bound);
		std::vector<double> weights(static_cast<size_t>(meta.shape.rows) *
									meta.shape.cols);
		for (double &w : weights) w = dist(rng);
		state[p.name] = encrypt_slots(pack_weights(weights, meta.shape.rows,
												   meta.shape.cols,
												   meta.row_packing, layout));
		// Zero velocities on the first step reproduce ReBoot's separate
		// "initialise the velocity" branch, so only one update rule is emitted.
		state[fmt::format("v_{}", p.name)] = encrypt_slots(
			std::vector<double>(static_cast<size_t>(num_slots), 0.0));
	}

	monitor = std::make_unique<ResourceMonitor>(true);
	const std::string trace =
		fmt::format("{}/reboot_{}_{}.csv", result_dir,
					getModeString(keymem_rt.getOperationMode()), layout.str());
	monitor->start(trace);

	// ---- training loop -----------------------------------------------------
	const PackFormat prediction_format =
		g.value(step.predictions.front()).format;
	const size_t batch = static_cast<size_t>(config.batch_size);

	for (int s = 0; s < steps; ++s) {
		const size_t start = (static_cast<size_t>(s) * batch) % data.size();
		if (start + batch > data.size()) break;

		// Arguments go in the order the module recorded in
		// reboot.argument_names, which is the order this schema produced.
		std::vector<CiphertextT> args(lowered.argument_names.size());
		for (size_t i = 0; i < lowered.argument_names.size(); ++i) {
			const std::string &name = lowered.argument_names[i];
			auto it = state.find(name);
			if (it != state.end()) {
				args[i] = it->second;
				continue;
			}
			// Inputs and labels: encrypted fresh for this batch.
			const TensorValue &meta = g.value(step.arguments[i]);
			const size_t sample = static_cast<size_t>(
				std::stoi(name.substr(name.rfind('_') + 1)));
			const std::vector<double> plain =
				name.rfind("x_", 0) == 0
					? data.features[start + sample]
					: one_hot(data.y[start + sample], config.num_classes);
			args[i] = encrypt_slots(pack_vector(plain, meta.format, layout));
		}

		const auto t0 = std::chrono::high_resolution_clock::now();
		const std::vector<CiphertextT> results = reboot_train_step(cc, args);
		const auto t1 = std::chrono::high_resolution_clock::now();

		for (size_t i = 0; i < step.params.size(); ++i) {
			state[step.params[i].name] =
				std::const_pointer_cast<CiphertextImpl<DCRTPoly>>(
					results[2 * i]);
			state[fmt::format("v_{}", step.params[i].name)] =
				std::const_pointer_cast<CiphertextImpl<DCRTPoly>>(
					results[2 * i + 1]);
		}

		double loss = 0.0;
		int correct = 0;
		const size_t prediction_base = 2 * step.params.size();
		for (size_t b = 0; b < batch; ++b) {
			const std::vector<double> scores =
				unpack_vector(decrypt_slots(results[prediction_base + b]),
							  prediction_format, layout, config.num_classes);
			const int label = data.y[start + b];
			for (int c = 0; c < config.num_classes; ++c) {
				const double diff =
					scores[static_cast<size_t>(c)] - (c == label ? 1.0 : 0.0);
				loss += diff * diff;
			}
			if (argmax(scores) == label) ++correct;
		}

		fmt::print(
			"step {:3d} | loss {:8.4f} | accuracy {:5.1f}% | {:8.1f} ms\n", s,
			loss / static_cast<double>(batch),
			100.0 * correct / static_cast<double>(batch),
			std::chrono::duration<double, std::milli>(t1 - t0).count());
	}

	monitor->stop();
	monitor->save_to_file(trace);
	fmt::print("\nresource trace: {}\n", trace);
	keymem_rt.printKeyStats();
	return 0;
}
