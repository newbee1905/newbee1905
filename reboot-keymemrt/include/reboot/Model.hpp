// SPDX-License-Identifier: GPL-3.0-or-later
//
// The ReBoot network: local-loss blocks plus a regular head, and the encrypted
// training step.
//
// Architecture (eMLP-h of the paper).  For h hidden layers:
//
//   h-1 local-loss blocks    [ linear -> PolyReLU ] + local classifier
//   1   hidden linear + PolyReLU
//   1   output linear
//
// Each block carries its own classifier and its own RSS loss, so its error
// signal never leaves the block.  That is what keeps the multiplicative depth
// of a training step independent of the network depth - the property that makes
// non-interactive encrypted training feasible at all.
//
// Packing alternates block by block (row, column, row, ...), so no layer ever
// has to repack its input.

#ifndef REBOOT_MODEL_HPP_
#define REBOOT_MODEL_HPP_

#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "reboot/Backend.hpp"
#include "reboot/Layers.hpp"
#include "reboot/Layout.hpp"
#include "reboot/Optim.hpp"

namespace reboot {

struct ModelConfig {
  int inputDim = 0;
  std::vector<int> hidden;   // one entry per hidden layer
  int numClasses = 0;
  double momentum = 0.9;
  double weightDecay = 0.0;
  unsigned seed = 42;
};

// A batch as it arrives from the client: inputs in Expanded form (the first
// layer is row-packed) and the one-hot labels in both packings, because the
// local classifiers of the blocks alternate their output format.  Encrypting
// the labels in both formats up front is what keeps the training step
// non-interactive; the reference implementation re-encrypts the plaintext
// labels inside every block instead, which a real server could not do.
template <class B>
struct EncryptedBatch {
  std::vector<typename B::Ct> x;
  std::vector<typename B::Ct> yRepeated;
  std::vector<typename B::Ct> yExpanded;

  const std::vector<typename B::Ct> &labels(Format f) const {
    return f == Format::Repeated ? yRepeated : yExpanded;
  }
  size_t size() const { return x.size(); }
};

template <class B>
class LocalLossMLP {
 public:
  using Ct = typename B::Ct;

  // Smallest layout that holds every layer of the network, then inflated so
  // that rows*cols equals the number of CKKS slots (the rotate-and-add trees
  // operate on the full slot vector).
  static Layout recommendLayout(const ModelConfig &cfg, int numSlots) {
    int rows = 1, cols = 1;
    bool rowPacking = true;
    const int numBlocks = cfg.hidden.empty() ? 0 : static_cast<int>(cfg.hidden.size()) - 1;

    auto require = [&](int in, int out, bool rp) {
      if (rp) {
        rows = std::max(rows, in);
        cols = std::max(cols, out);
      } else {
        cols = std::max(cols, in);
        rows = std::max(rows, out);
      }
    };

    for (int i = 0; i < numBlocks; ++i) {
      const int in = (i == 0) ? cfg.inputDim : cfg.hidden[i - 1];
      require(in, cfg.hidden[i], rowPacking);            // forward layer
      require(cfg.hidden[i], cfg.numClasses, !rowPacking);  // local classifier
      rowPacking = !rowPacking;
    }
    if (!cfg.hidden.empty()) {
      const int in = cfg.hidden.size() == 1
                         ? cfg.inputDim
                         : cfg.hidden[cfg.hidden.size() - 2];
      require(in, cfg.hidden.back(), rowPacking);
      rowPacking = !rowPacking;
    }
    const int lastIn = cfg.hidden.empty() ? cfg.inputDim : cfg.hidden.back();
    require(lastIn, cfg.numClasses, cfg.hidden.empty() ? true : rowPacking);

    rows = nextPowerOfTwo(rows);
    cols = nextPowerOfTwo(cols);
    if (rows * cols > numSlots)
      throw std::invalid_argument(
          "network needs " + std::to_string(rows * cols) +
          " slots but the context provides " + std::to_string(numSlots));
    // Fill the ciphertext: the extra rows are zero and do not disturb the
    // column sums, while a partially used slot vector would let the cyclic
    // rotations mix in the unused region.
    rows = numSlots / cols;
    return Layout(rows, cols);
  }

  LocalLossMLP(const ModelConfig &cfg, const Layout &layout)
      : cfg_(cfg), layout_(layout), firstColMask_{"first_column",
                                                  firstColumnMask(layout)} {
    const int numBlocks =
        cfg.hidden.empty() ? 0 : static_cast<int>(cfg.hidden.size()) - 1;
    bool rowPacking = true;

    for (int i = 0; i < numBlocks; ++i) {
      const int in = (i == 0) ? cfg.inputDim : cfg.hidden[i - 1];
      Block b;
      b.forward = std::make_unique<PackedLinear<B>>(
          in, cfg.hidden[i], rowPacking, layout, cfg.momentum, cfg.weightDecay,
          /*trainable=*/true, /*propagateBackward=*/false,
          "linear_fwd_" + std::to_string(i));
      b.classifier = std::make_unique<PackedLinear<B>>(
          cfg.hidden[i], cfg.numClasses, !rowPacking, layout, cfg.momentum,
          cfg.weightDecay, /*trainable=*/true, /*propagateBackward=*/true,
          "linear_lrn_" + std::to_string(i));
      blocks_.push_back(std::move(b));
      rowPacking = !rowPacking;
    }

    if (!cfg.hidden.empty()) {
      const int in = cfg.hidden.size() == 1
                         ? cfg.inputDim
                         : cfg.hidden[cfg.hidden.size() - 2];
      head_ = std::make_unique<PackedLinear<B>>(
          in, cfg.hidden.back(), rowPacking, layout, cfg.momentum,
          cfg.weightDecay, /*trainable=*/true, /*propagateBackward=*/false,
          "linear_head");
      rowPacking = !rowPacking;
    }

    const int lastIn = cfg.hidden.empty() ? cfg.inputDim : cfg.hidden.back();
    output_ = std::make_unique<PackedLinear<B>>(
        lastIn, cfg.numClasses, cfg.hidden.empty() ? true : rowPacking, layout,
        cfg.momentum, cfg.weightDecay, /*trainable=*/true,
        /*propagateBackward=*/!cfg.hidden.empty(), "linear_out");
  }

  const Layout &layout() const { return layout_; }
  const Mask &firstColumnMaskRef() const { return firstColMask_; }
  Format inputFormat() const {
    return blocks_.empty()
               ? (head_ ? head_->inFormat() : output_->inFormat())
               : blocks_.front().forward->inFormat();
  }
  Format outputFormat() const { return output_->outFormat(); }

  void initWeights() {
    std::mt19937 rng(cfg_.seed);
    for (auto *layer : allLayers()) layer->initWeights(rng);
  }

  void encrypt(B &be) {
    for (auto *layer : allLayers()) layer->encrypt(be);
  }

  std::vector<PackedLinear<B> *> allLayers() {
    std::vector<PackedLinear<B> *> out;
    for (auto &b : blocks_) {
      out.push_back(b.forward.get());
      out.push_back(b.classifier.get());
    }
    if (head_) out.push_back(head_.get());
    out.push_back(output_.get());
    return out;
  }

  // ---- inference -----------------------------------------------------------
  std::vector<Ct> forward(B &be, const std::vector<Ct> &x, bool training) {
    std::vector<Ct> h = x;
    for (auto &b : blocks_) {
      h = b.forward->forward(be, h, firstColMask_, training);
      h = b.activation.forward(be, h, training);
      if (training) b.lastActivation = h;
    }
    if (head_) {
      h = head_->forward(be, h, firstColMask_, training);
      h = headActivation_.forward(be, h, training);
    }
    return output_->forward(be, h, firstColMask_, training);
  }

  // ---- training ------------------------------------------------------------
  // One ReBoot step: a global forward pass, backpropagation restricted to the
  // head, then one independent local update per block.
  std::vector<Ct> trainStep(B &be, const EncryptedBatch<B> &batch, double lr) {
    auto pred = forward(be, batch.x, /*training=*/true);
    auto delta = rssLossGradient(be, pred, batch.labels(output_->outFormat()));

    // Head: output linear -> activation -> hidden linear.
    delta = output_->backward(be, delta, lr, firstColMask_);
    if (head_) {
      delta = headActivation_.backward(be, delta);
      head_->backward(be, delta, lr, firstColMask_);
    }

    // Blocks, deepest first.  Each one regenerates its own error signal from
    // its stored activation and the encrypted labels; nothing flows between
    // blocks, which is what bounds the depth of the step.
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
      Block &b = *it;
      auto local = b.classifier->forward(be, b.lastActivation, firstColMask_,
                                         /*training=*/true);
      auto localDelta =
          rssLossGradient(be, local, batch.labels(b.classifier->outFormat()));
      auto d = b.classifier->backward(be, localDelta, lr, firstColMask_);
      d = b.activation.backward(be, d);
      b.forward->backward(be, d, lr, firstColMask_);
    }
    return pred;
  }

  // Refresh every trainable ciphertext that survives across steps.  Without
  // this the weights would run out of levels after a handful of updates; it is
  // the "Re" in ReBoot.
  void bootstrapParameters(B &be) {
    for (auto *layer : allLayers()) {
      layer->setWeights(be.bootstrap(layer->weights()));
      if (layer->optimizer().hasVelocity())
        layer->optimizer().setVelocity(be.bootstrap(layer->optimizer().velocity()));
    }
  }

  std::string describe() const {
    std::string s = "LocalLossMLP(layout=" + layout_.str() + ")\n";
    for (const auto &b : blocks_)
      s += "  block: " + b.forward->name() + " (" +
           (b.forward->rowPacking() ? "row" : "col") + ") + PolyReLU + " +
           b.classifier->name() + " (" +
           (b.classifier->rowPacking() ? "row" : "col") + ")\n";
    if (head_)
      s += "  head:  " + head_->name() + " (" +
           (head_->rowPacking() ? "row" : "col") + ") + PolyReLU\n";
    s += "  out:   " + output_->name() + " (" +
         (output_->rowPacking() ? "row" : "col") + ")\n";
    return s;
  }

 private:
  struct Block {
    std::unique_ptr<PackedLinear<B>> forward;
    PolyReLU<B> activation;
    std::unique_ptr<PackedLinear<B>> classifier;
    std::vector<Ct> lastActivation;
  };

  ModelConfig cfg_;
  Layout layout_;
  Mask firstColMask_;
  std::vector<Block> blocks_;
  std::unique_ptr<PackedLinear<B>> head_;
  PolyReLU<B> headActivation_;
  std::unique_ptr<PackedLinear<B>> output_;
};

// Decode the class scores of one sample from a packed prediction.
inline std::vector<double> readPrediction(const std::vector<double> &slots,
                                          Format f, const Layout &layout,
                                          int numClasses) {
  return unpackVector(slots, f, layout, numClasses);
}

inline int argmax(const std::vector<double> &v) {
  int best = 0;
  for (size_t i = 1; i < v.size(); ++i)
    if (v[i] > v[static_cast<size_t>(best)]) best = static_cast<int>(i);
  return best;
}

}  // namespace reboot

#endif  // REBOOT_MODEL_HPP_
