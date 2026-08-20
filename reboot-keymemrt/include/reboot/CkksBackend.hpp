// SPDX-License-Identifier: GPL-3.0-or-later
//
// CKKS backend: OpenFHE (eymay fork) with every rotation routed through the
// KeyMemRT runtime.
//
// This is the whole point of the port.  In the reference ReBoot the rotations
// happen inside OpenFHE's EvalSumRows / EvalSumCols, called from Python through
// openfhe-python, with the complete rotation key set resident for the entire
// run.  Here each rotation is explicit, so it can be bracketed exactly the way
// KeyMemRT-Compiler brackets `openfhe.rot`:
//
//     RotKey rk = keymem_rt.deserializeKey(index, keyDepth);
//     auto out  = cc->EvalRotate(ct, index);
//     keymem_rt.clearKey(rk);
//
// The key is paged in from disk at the compression level that matches the
// ciphertext, used once and dropped, so the resident key working set is one
// key instead of the full set.  In PREFETCH mode the upcoming keys of the
// recorded plan are enqueued first, letting the background thread overlap the
// deserialisation with the computation.

#ifndef REBOOT_CKKSBACKEND_HPP_
#define REBOOT_CKKSBACKEND_HPP_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "KeyMemRT.hpp"
#include "openfhe.h"
#include "reboot/Backend.hpp"
#include "reboot/KeyPlan.hpp"

namespace reboot {

class CkksBackend {
 public:
  using Ct = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
  using Pt = lbcrypto::Plaintext;

  CkksBackend(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
              lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey,
              lbcrypto::PrivateKey<lbcrypto::DCRTPoly> secretKey,
              KeyMemRT *runtime)
      : cc_(std::move(cc)), publicKey_(std::move(publicKey)),
        secretKey_(std::move(secretKey)), runtime_(runtime) {
    slots_ = static_cast<int>(cc_->GetRingDimension() / 2);
  }

  // ---- configuration -------------------------------------------------------
  // Plan being recorded during a calibration run (may be null).
  void setRecordingPlan(KeyPlan *plan) { recording_ = plan; }
  // Plan that was actually provisioned on disk; requests outside it fall back
  // to the uncompressed (depth 0) key.
  void setProvisionedPlan(const KeyPlan *plan) { provisioned_ = plan; }
  void setPrefetch(const KeyPlan *plan, int lookahead) {
    cursor_ = PrefetchCursor(plan, lookahead);
  }
  void setBootstrapParams(int numIterations, int precision) {
    bsIterations_ = numIterations;
    bsPrecision_ = precision;
  }
  void resetCursor() { cursor_.reset(); }

  long rotationCount() const { return rotations_; }
  long bootstrapCount() const { return bootstraps_; }

  // ---- backend interface ---------------------------------------------------
  int slots() const { return slots_; }
  int level(const Ct &c) const { return static_cast<int>(c->GetLevel()); }

  Ct add(const Ct &a, const Ct &b) { return cc_->EvalAdd(a, b); }
  Ct sub(const Ct &a, const Ct &b) { return cc_->EvalSub(a, b); }
  Ct mul(const Ct &a, const Ct &b) { return cc_->EvalMult(a, b); }
  Ct mulScalar(const Ct &a, double s) { return cc_->EvalMult(a, s); }
  Ct addScalar(const Ct &a, double s) { return cc_->EvalAdd(a, s); }
  Ct square(const Ct &a) { return cc_->EvalSquare(a); }

  // Masking needs the constant encoded at the ciphertext's level and noise
  // scale; the encodings are cached because the same handful of (mask, level)
  // pairs recur on every sample of every batch.
  Ct mulMask(const Ct &a, const Mask &m) {
    return cc_->EvalMult(a, encodeAt(m, a));
  }

  Ct rotate(const Ct &a, int index) {
    if (index == 0) return a;
    const int lvl = level(a);
    if (recording_) recording_->record(index, lvl);

    if (cursor_.active() && runtime_) {
      for (const auto &req : cursor_.advance(index, lvl))
        runtime_->enqueueKey(req.index, keyDepthFor(req.index, req.level));
    }

    ++rotations_;
    if (!runtime_) return cc_->EvalRotate(a, index);

    const RotKey rk = runtime_->deserializeKey(index, keyDepthFor(index, lvl));
    Ct out = cc_->EvalRotate(a, index);
    runtime_->clearKey(rk);
    return out;
  }

  Ct bootstrap(const Ct &a) {
    ++bootstraps_;
    return cc_->EvalBootstrap(a, bsIterations_, bsPrecision_);
  }

  Ct encryptSlots(const std::vector<double> &v) {
    std::vector<double> padded = v;
    padded.resize(static_cast<size_t>(slots_), 0.0);
    Pt pt = cc_->MakeCKKSPackedPlaintext(padded);
    return cc_->Encrypt(publicKey_, pt);
  }

  std::vector<double> decryptSlots(const Ct &c) const {
    Pt pt;
    cc_->Decrypt(secretKey_, c, &pt);
    pt->SetLength(static_cast<size_t>(slots_));
    return pt->GetRealPackedValue();
  }

  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context() const { return cc_; }

 private:
  // KeyMemRT stores each key compressed for one ciphertext level.  A level the
  // provisioning step did not cover falls back to the uncompressed key, which
  // is always valid - correctness never depends on the plan being complete.
  int keyDepthFor(int index, int lvl) const {
    if (lvl <= 0) return 0;
    if (!provisioned_) return lvl;
    return provisioned_->distinct().count(KeyRequest{index, lvl}) ? lvl : 0;
  }

  Pt encodeAt(const Mask &m, const Ct &reference) {
    const auto key = std::make_tuple(static_cast<const void *>(&m),
                                     reference->GetLevel(),
                                     reference->GetNoiseScaleDeg());
    auto it = maskCache_.find(key);
    if (it != maskCache_.end()) return it->second;
    Pt pt = cc_->MakeCKKSPackedPlaintext(m.values, reference->GetNoiseScaleDeg(),
                                         reference->GetLevel(), nullptr,
                                         static_cast<usint>(slots_));
    maskCache_.emplace(key, pt);
    return pt;
  }

  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc_;
  lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey_;
  lbcrypto::PrivateKey<lbcrypto::DCRTPoly> secretKey_;
  KeyMemRT *runtime_ = nullptr;

  KeyPlan *recording_ = nullptr;
  const KeyPlan *provisioned_ = nullptr;
  PrefetchCursor cursor_{nullptr, 0};

  int slots_ = 0;
  int bsIterations_ = 1;
  int bsPrecision_ = 0;
  long rotations_ = 0;
  long bootstraps_ = 0;

  std::map<std::tuple<const void *, size_t, size_t>, Pt> maskCache_;
};

}  // namespace reboot

#endif  // REBOOT_CKKSBACKEND_HPP_
