// SPDX-License-Identifier: GPL-3.0-or-later
//
// ReBoot layers: the CKKS-packed linear layer and the HE-friendly activations.
//
// A batch is a vector of ciphertexts, one per sample; the weights are a single
// ciphertext holding the whole (padded) weight matrix.  Forward and backward
// are the RE-/CE-Matmul pair from LinAlg.hpp, so a row-packed layer maps
// Expanded -> Repeated and a column-packed layer maps Repeated -> Expanded.
// Alternating packings across layers keeps the pipeline free of repacking.

#ifndef REBOOT_LAYERS_HPP_
#define REBOOT_LAYERS_HPP_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "reboot/Backend.hpp"
#include "reboot/Layout.hpp"
#include "reboot/LinAlg.hpp"
#include "reboot/Optim.hpp"

namespace reboot {

template <class B>
class PackedLinear {
 public:
  using Ct = typename B::Ct;

  PackedLinear(int inFeatures, int outFeatures, bool rowPacking,
               const Layout &layout, double momentum, double weightDecay,
               bool trainable, bool propagateBackward, std::string name)
      : inFeatures_(inFeatures), outFeatures_(outFeatures),
        rowPacking_(rowPacking), layout_(layout), trainable_(trainable),
        propagateBackward_(propagateBackward), name_(std::move(name)),
        optimizer_(momentum, weightDecay) {}

  const std::string &name() const { return name_; }
  bool rowPacking() const { return rowPacking_; }
  bool trainable() const { return trainable_; }
  Format inFormat() const { return inputFormat(rowPacking_); }
  Format outFormat() const { return outputFormat(rowPacking_); }
  const Ct &weights() const { return weights_; }
  void setWeights(const Ct &w) { weights_ = w; }
  NesterovSGD<B> &optimizer() { return optimizer_; }

  // Xavier-uniform initialisation, matching the reference implementation.
  void initWeights(std::mt19937 &rng) {
    const double bound = std::sqrt(1.0 / (inFeatures_ + outFeatures_));
    std::uniform_real_distribution<double> dist(-bound, bound);
    plainWeights_.resize(static_cast<size_t>(inFeatures_) * outFeatures_);
    for (auto &w : plainWeights_) w = dist(rng);
  }

  const std::vector<double> &plainWeights() const { return plainWeights_; }

  void encrypt(B &be) {
    weights_ = be.encryptSlots(
        packWeights(plainWeights_, inFeatures_, outFeatures_, rowPacking_, layout_));
  }

  std::vector<double> decryptWeights(B &be) const {
    return unpackWeights(be.decryptSlots(weights_), inFeatures_, outFeatures_,
                         rowPacking_, layout_);
  }

  std::vector<Ct> forward(B &be, const std::vector<Ct> &x, const Mask &firstCol,
                          bool training) {
    if (training) lastInput_ = x;
    std::vector<Ct> out;
    out.reserve(x.size());
    for (const auto &sample : x) {
      out.push_back(rowPacking_ ? matmulRE(be, sample, weights_, layout_)
                                : matmulCE(be, sample, weights_, layout_, firstCol));
    }
    return out;
  }

  // Error signal towards the input.  The transpose of the forward product is
  // obtained by swapping the summation direction, reusing the same weight
  // ciphertext - no transposition, no repacking.
  std::vector<Ct> backwardDelta(B &be, const std::vector<Ct> &delta,
                                const Mask &firstCol) {
    std::vector<Ct> out;
    out.reserve(delta.size());
    for (const auto &d : delta) {
      auto prod = be.mul(d, weights_);
      out.push_back(rowPacking_ ? sumCols(be, prod, layout_, firstCol)
                                : sumRows(be, prod, layout_));
    }
    return out;
  }

  // Full backward pass: gradient towards the input (optional) plus the weight
  // update.  Returns an empty vector when the layer does not propagate.
  std::vector<Ct> backward(B &be, const std::vector<Ct> &delta, double lr,
                           const Mask &firstCol) {
    std::vector<Ct> newDelta;
    if (propagateBackward_) newDelta = backwardDelta(be, delta, firstCol);

    if (trainable_) {
      Ct gradient = outerProductSum(be, lastInput_, delta);
      Ct update = optimizer_.computeUpdate(be, gradient, weights_, lr);
      weights_ = be.sub(weights_, update);
    }
    return newDelta;
  }

 private:
  int inFeatures_;
  int outFeatures_;
  bool rowPacking_;
  Layout layout_;
  bool trainable_;
  bool propagateBackward_;
  std::string name_;

  std::vector<double> plainWeights_;
  Ct weights_;
  std::vector<Ct> lastInput_;
  NesterovSGD<B> optimizer_;
};

// PolyReLU(x) = x^2 + x, the degree-two ReLU surrogate of Ali et al. used by
// ReBoot.  Forward and backward each cost one level.
template <class B>
class PolyReLU {
 public:
  using Ct = typename B::Ct;

  std::vector<Ct> forward(B &be, const std::vector<Ct> &x, bool training) {
    if (training) lastInput_ = x;
    std::vector<Ct> out;
    out.reserve(x.size());
    for (const auto &v : x) out.push_back(be.add(be.square(v), v));
    return out;
  }

  std::vector<Ct> backward(B &be, const std::vector<Ct> &delta) {
    std::vector<Ct> out;
    out.reserve(delta.size());
    for (size_t i = 0; i < delta.size(); ++i)
      out.push_back(
          be.mul(delta[i], be.addScalar(be.mulScalar(lastInput_[i], 2.0), 1.0)));
    return out;
  }

 private:
  std::vector<Ct> lastInput_;
};

// Square activation, kept for parity with the reference library.
template <class B>
class SquareActivation {
 public:
  using Ct = typename B::Ct;

  std::vector<Ct> forward(B &be, const std::vector<Ct> &x, bool training) {
    if (training) lastInput_ = x;
    std::vector<Ct> out;
    out.reserve(x.size());
    for (const auto &v : x) out.push_back(be.square(v));
    return out;
  }

  std::vector<Ct> backward(B &be, const std::vector<Ct> &delta) {
    std::vector<Ct> out;
    out.reserve(delta.size());
    for (size_t i = 0; i < delta.size(); ++i)
      out.push_back(be.mul(delta[i], be.mulScalar(lastInput_[i], 2.0)));
    return out;
  }

 private:
  std::vector<Ct> lastInput_;
};

// Gradient of the residual-sum-of-squares loss: y_hat - y.
template <class B>
std::vector<typename B::Ct> rssLossGradient(B &be,
                                            const std::vector<typename B::Ct> &yPred,
                                            const std::vector<typename B::Ct> &yTrue) {
  std::vector<typename B::Ct> out;
  out.reserve(yPred.size());
  for (size_t i = 0; i < yPred.size(); ++i)
    out.push_back(be.sub(yPred[i], yTrue[i]));
  return out;
}

}  // namespace reboot

#endif  // REBOOT_LAYERS_HPP_
