// SPDX-License-Identifier: GPL-3.0-or-later
//
// CKKS context construction and client-side key provisioning.
//
// Provisioning is the piece KeyMemRT-Compiler normally emits from the IR
// (`openfhe.gen_rot_key`, `openfhe.gen_rot_key_depth`, `openfhe.gen_bootstrap_key`).
// A hand-written training loop has to do the same three things itself:
//
//   1. generate the rotation keys the step needs,
//   2. compress each key to the ciphertext level it will be used at and write
//      it to its own file, so the runtime can page it in individually,
//   3. drop every key from the context, leaving the server to load them on
//      demand.
//
// Which (index, level) pairs to provision comes from a calibration run - see
// KeyPlan.hpp.  Level 0 is always provisioned as an uncompressed fallback.

#ifndef REBOOT_CKKSCONTEXT_HPP_
#define REBOOT_CKKSCONTEXT_HPP_

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "KeyMemRT.hpp"
#include "openfhe.h"
#include "reboot/KeyPlan.hpp"

namespace reboot {

struct CkksParams {
  int computeDepth = 20;   // levels the training step itself consumes
  int ringDim = 1 << 16;
  int scalingModSize = 59;
  int firstModSize = 60;
  bool bootstrap = true;
  std::vector<uint32_t> levelBudget = {2, 2};
  int bsIterations = 2;
  int bsPrecision = 18;
  bool standardSecurity = false;  // HEStd_128_classic when true
};

struct CkksSetup {
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
  lbcrypto::KeyPair<lbcrypto::DCRTPoly> keyPair;
  int numSlots = 0;
  int multDepth = 0;
  int bootstrapDepth = 0;
};

inline CkksSetup makeContext(const CkksParams &p) {
  using namespace lbcrypto;

  CCParams<CryptoContextCKKSRNS> params;
  int multDepth = p.computeDepth;
  int bootstrapDepth = 0;

  if (p.bootstrap) {
    params.SetSecretKeyDist(SPARSE_TERNARY);
    bootstrapDepth = static_cast<int>(
        FHECKKSRNS::GetBootstrapDepth(p.levelBudget, SPARSE_TERNARY));
    multDepth += bootstrapDepth + p.bsIterations;
  }

  params.SetMultiplicativeDepth(static_cast<usint>(multDepth));
  params.SetScalingModSize(static_cast<usint>(p.scalingModSize));
  params.SetFirstModSize(static_cast<usint>(p.firstModSize));
  params.SetScalingTechnique(FLEXIBLEAUTO);
  params.SetRingDim(static_cast<usint>(p.ringDim));
  params.SetSecurityLevel(p.standardSecurity ? HEStd_128_classic : HEStd_NotSet);

  CkksSetup setup;
  setup.cc = GenCryptoContext(params);
  setup.cc->Enable(PKE);
  setup.cc->Enable(KEYSWITCH);
  setup.cc->Enable(LEVELEDSHE);
  setup.cc->Enable(ADVANCEDSHE);
  if (p.bootstrap) setup.cc->Enable(FHE);

  setup.keyPair = setup.cc->KeyGen();
  setup.cc->EvalMultKeyGen(setup.keyPair.secretKey);
  setup.numSlots = static_cast<int>(setup.cc->GetRingDimension() / 2);
  setup.multDepth = multDepth;
  setup.bootstrapDepth = bootstrapDepth;

  std::cout << "CKKS parameters\n"
            << "  ring dimension     : " << setup.cc->GetRingDimension() << "\n"
            << "  slots              : " << setup.numSlots << "\n"
            << "  multiplicative depth: " << multDepth << " (compute "
            << p.computeDepth << " + bootstrap " << bootstrapDepth << " + "
            << p.bsIterations << " iterations)\n"
            << "  scaling mod size   : " << p.scalingModSize << "\n"
            << "  security           : "
            << (p.standardSecurity ? "HEStd_128_classic" : "HEStd_NotSet")
            << "\n";
  return setup;
}

// The bootstrapping key set is loaded and dropped as one bundle.  OpenFHE
// performs the rotations of EvalBootstrap internally, so a per-key handshake is
// impossible there; staging the bundle around the (single) bootstrap phase of a
// training step still keeps it out of memory for the rest of the step, which is
// where ReBoot spends most of its time.
class BootstrapKeyBundle {
 public:
  BootstrapKeyBundle(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
                     std::string keyTag, std::string path)
      : cc_(std::move(cc)), keyTag_(std::move(keyTag)), path_(std::move(path)) {}

  bool serialize() const {
    std::ofstream f(path_, std::ios::binary);
    if (!f) return false;
    return cc_->SerializeEvalAutomorphismKey(f, lbcrypto::SerType::BINARY,
                                             keyTag_);
  }

  bool load() const {
    std::ifstream f(path_, std::ios::binary);
    if (!f) return false;
    return cc_->DeserializeEvalAutomorphismKey(f, lbcrypto::SerType::BINARY);
  }

  void clear() const { cc_->ClearEvalAutomorphismKeys(keyTag_); }

 private:
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc_;
  std::string keyTag_;
  std::string path_;
};

// Generate, compress and serialize the rotation keys named by `plan`.
// Mirrors the emitted GenRotKeyOp / GenRotKeyDepthOp sequence: generate at a
// level, hand the keys to KeyMemRT to compress and write out, then clear.
inline bool provisionRotationKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
                                  const lbcrypto::PrivateKey<lbcrypto::DCRTPoly> &sk,
                                  KeyMemRT &runtime, const KeyPlan &plan) {
  const auto all = plan.indices();
  if (all.empty()) {
    std::cerr << "provisionRotationKeys: empty plan\n";
    return false;
  }
  runtime.addRotIndices(all);

  bool ok = true;
  // Level 0: the uncompressed fallback, provisioned for every index.
  cc->EvalRotateKeyGen(sk, all);
  ok &= runtime.serializeKeysAtLevel(all, 0);
  runtime.clearAllKeys();

  for (int level : plan.levels()) {
    if (level == 0) continue;
    const auto indices = plan.indicesAtLevel(level);
    if (indices.empty()) continue;
    cc->EvalRotateKeyGen(sk, indices);
    ok &= runtime.serializeKeysAtLevel(indices, level);
    runtime.clearAllKeys();
  }
  return ok;
}

}  // namespace reboot

#endif  // REBOOT_CKKSCONTEXT_HPP_
