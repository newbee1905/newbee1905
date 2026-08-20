// SPDX-License-Identifier: GPL-3.0-or-later
//
// End-to-end check of the ReBoot training step.
//
// The packed implementation (Model.hpp, running on the plaintext slot backend)
// is compared step by step against an independent, unpacked reference written
// directly in terms of matrices: same architecture, same local-loss rule, same
// Nesterov update.  If the two agree to machine precision after several steps,
// the packed rewrite is faithful to the algorithm - and because Model.hpp is
// backend-generic, the CKKS run executes the very same sequence of operations.
//
// A short convergence run follows, so the test also fails if the training rule
// itself is broken rather than just self-consistent.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "reboot/Backend.hpp"
#include "reboot/Data.hpp"
#include "reboot/Model.hpp"

using namespace reboot;

namespace {

int failures = 0;
using Matrix = std::vector<double>;              // row-major
using Batch = std::vector<std::vector<double>>;  // one vector per sample

void check(bool cond, const char *what) {
  if (!cond) {
    std::printf("  FAIL: %s\n", what);
    ++failures;
  }
}

// --------------------------------------------------------------------------
// Unpacked reference implementation
// --------------------------------------------------------------------------

struct RefLinear {
  int in = 0, out = 0;
  Matrix w;
  Matrix velocity;
  bool hasVelocity = false;
  double momentum = 0.9, weightDecay = 0.0;
  Batch lastInput;

  Batch forward(const Batch &x) {
    lastInput = x;
    Batch out_(x.size(), std::vector<double>(static_cast<size_t>(out), 0.0));
    for (size_t b = 0; b < x.size(); ++b)
      for (int j = 0; j < out; ++j)
        for (int i = 0; i < in; ++i)
          out_[b][static_cast<size_t>(j)] +=
              x[b][static_cast<size_t>(i)] * w[static_cast<size_t>(i) * out + j];
    return out_;
  }

  Batch backwardDelta(const Batch &delta) const {
    Batch out_(delta.size(), std::vector<double>(static_cast<size_t>(in), 0.0));
    for (size_t b = 0; b < delta.size(); ++b)
      for (int i = 0; i < in; ++i)
        for (int j = 0; j < out; ++j)
          out_[b][static_cast<size_t>(i)] +=
              w[static_cast<size_t>(i) * out + j] * delta[b][static_cast<size_t>(j)];
    return out_;
  }

  void update(const Batch &delta, double lr) {
    Matrix grad(w.size(), 0.0);
    for (size_t b = 0; b < delta.size(); ++b)
      for (int i = 0; i < in; ++i)
        for (int j = 0; j < out; ++j)
          grad[static_cast<size_t>(i) * out + j] +=
              lastInput[b][static_cast<size_t>(i)] * delta[b][static_cast<size_t>(j)];

    if (weightDecay != 0.0)
      for (size_t k = 0; k < grad.size(); ++k) grad[k] += weightDecay * w[k];

    Matrix update_(w.size(), 0.0);
    if (!hasVelocity) {
      velocity = grad;
      hasVelocity = true;
      for (size_t k = 0; k < grad.size(); ++k)
        update_[k] = lr * grad[k] + momentum * lr * grad[k];
    } else {
      for (size_t k = 0; k < grad.size(); ++k)
        update_[k] = lr * grad[k] + momentum * lr * grad[k] +
                     momentum * momentum * lr * velocity[k];
      for (size_t k = 0; k < grad.size(); ++k)
        velocity[k] = momentum * velocity[k] + grad[k];
    }
    for (size_t k = 0; k < w.size(); ++k) w[k] -= update_[k];
  }
};

Batch polyReLU(const Batch &x) {
  Batch out(x);
  for (auto &row : out)
    for (auto &v : row) v = v * v + v;
  return out;
}

Batch polyReLUBackward(const Batch &delta, const Batch &preAct) {
  Batch out(delta);
  for (size_t b = 0; b < delta.size(); ++b)
    for (size_t i = 0; i < delta[b].size(); ++i)
      out[b][i] = delta[b][i] * (2.0 * preAct[b][i] + 1.0);
  return out;
}

struct RefModel {
  struct Block {
    RefLinear forward;
    RefLinear classifier;
    Batch preActivation;
    Batch activation;
  };
  std::vector<Block> blocks;
  bool hasHead = false;
  RefLinear head;
  Batch headPreActivation;
  RefLinear output;

  Batch forward(const Batch &x) {
    Batch h = x;
    for (auto &b : blocks) {
      b.preActivation = b.forward.forward(h);
      h = polyReLU(b.preActivation);
      b.activation = h;
    }
    if (hasHead) {
      headPreActivation = head.forward(h);
      h = polyReLU(headPreActivation);
    }
    return output.forward(h);
  }

  void step(const Batch &x, const Batch &y, double lr) {
    Batch pred = forward(x);
    Batch delta(pred);
    for (size_t b = 0; b < pred.size(); ++b)
      for (size_t j = 0; j < pred[b].size(); ++j) delta[b][j] -= y[b][j];

    Batch d = output.backwardDelta(delta);
    output.update(delta, lr);
    if (hasHead) {
      d = polyReLUBackward(d, headPreActivation);
      head.update(d, lr);
    }
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
      Batch local = it->classifier.forward(it->activation);
      Batch localDelta(local);
      for (size_t b = 0; b < local.size(); ++b)
        for (size_t j = 0; j < local[b].size(); ++j) localDelta[b][j] -= y[b][j];
      Batch dd = it->classifier.backwardDelta(localDelta);
      it->classifier.update(localDelta, lr);
      dd = polyReLUBackward(dd, it->preActivation);
      it->forward.update(dd, lr);
    }
  }
};

// --------------------------------------------------------------------------

double maxAbsDiff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

void testAgainstReference() {
  std::printf("packed step vs. unpacked reference\n");

  ModelConfig cfg;
  cfg.inputDim = 6;
  cfg.hidden = {8, 5};  // one local-loss block + head
  cfg.numClasses = 3;
  cfg.momentum = 0.9;
  cfg.weightDecay = 0.01;
  cfg.seed = 11;

  const int numSlots = 256;
  Layout layout = LocalLossMLP<PlainBackend>::recommendLayout(cfg, numSlots);
  PlainBackend be(numSlots);
  LocalLossMLP<PlainBackend> model(cfg, layout);
  model.initWeights();
  model.encrypt(be);

  // Mirror the same initial weights into the reference model.
  auto layers = model.allLayers();
  RefModel ref;
  ref.blocks.resize(1);
  ref.hasHead = true;
  RefLinear *refLayers[] = {&ref.blocks[0].forward, &ref.blocks[0].classifier,
                            &ref.head, &ref.output};
  const int dims[4][2] = {{cfg.inputDim, cfg.hidden[0]},
                          {cfg.hidden[0], cfg.numClasses},
                          {cfg.hidden[0], cfg.hidden[1]},
                          {cfg.hidden[1], cfg.numClasses}};
  for (int i = 0; i < 4; ++i) {
    refLayers[i]->in = dims[i][0];
    refLayers[i]->out = dims[i][1];
    refLayers[i]->w = layers[static_cast<size_t>(i)]->plainWeights();
    refLayers[i]->momentum = cfg.momentum;
    refLayers[i]->weightDecay = cfg.weightDecay;
  }

  // A small batch, encrypted in the formats the network consumes.
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> dist(-0.6, 0.6);
  const int batchSize = 3;
  Batch x(batchSize, std::vector<double>(static_cast<size_t>(cfg.inputDim)));
  Batch y(batchSize, std::vector<double>(static_cast<size_t>(cfg.numClasses), 0.0));
  for (int b = 0; b < batchSize; ++b) {
    for (auto &v : x[static_cast<size_t>(b)]) v = dist(rng);
    y[static_cast<size_t>(b)][static_cast<size_t>(b % cfg.numClasses)] = 1.0;
  }

  EncryptedBatch<PlainBackend> batch;
  for (int b = 0; b < batchSize; ++b) {
    batch.x.push_back(be.encryptSlots(
        packVector(x[static_cast<size_t>(b)], model.inputFormat(), layout)));
    batch.yRepeated.push_back(be.encryptSlots(
        packVector(y[static_cast<size_t>(b)], Format::Repeated, layout)));
    batch.yExpanded.push_back(be.encryptSlots(
        packVector(y[static_cast<size_t>(b)], Format::Expanded, layout)));
  }

  const double lr = 0.05;
  for (int step = 0; step < 5; ++step) {
    model.trainStep(be, batch, lr);
    ref.step(x, y, lr);

    for (int i = 0; i < 4; ++i) {
      auto got = layers[static_cast<size_t>(i)]->decryptWeights(be);
      double diff = maxAbsDiff(got, refLayers[i]->w);
      if (diff > 1e-9) {
        std::printf("  FAIL: step %d layer %s diverges by %g\n", step,
                    layers[static_cast<size_t>(i)]->name().c_str(), diff);
        ++failures;
      }
    }
  }
  std::printf("  5 steps, 4 layers: packed and unpacked weights agree\n");
}

void testConvergence() {
  std::printf("training converges on synthetic blobs\n");

  Dataset ds = makeBlobs(240, 8, 3, /*seed=*/5, /*spread=*/0.25);
  ModelConfig cfg;
  cfg.inputDim = ds.dim;
  cfg.hidden = {16, 8};
  cfg.numClasses = ds.numClasses;
  cfg.momentum = 0.9;
  cfg.weightDecay = 0.0;
  cfg.seed = 1;

  const int numSlots = 1024;
  Layout layout = LocalLossMLP<PlainBackend>::recommendLayout(cfg, numSlots);
  PlainBackend be(numSlots);
  LocalLossMLP<PlainBackend> model(cfg, layout);
  model.initWeights();
  model.encrypt(be);

  auto accuracy = [&]() {
    int correct = 0;
    for (size_t i = 0; i < ds.size(); ++i) {
      std::vector<PlainValue> single{
          be.encryptSlots(packVector(ds.X[i], model.inputFormat(), layout))};
      auto pred = model.forward(be, single, /*training=*/false);
      auto scores = readPrediction(be.decryptSlots(pred[0]),
                                   model.outputFormat(), layout, cfg.numClasses);
      if (argmax(scores) == ds.y[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(ds.size());
  };

  const double before = accuracy();

  const int batchSize = 8;
  const double lr = 0.02;
  for (int epoch = 0; epoch < 6; ++epoch) {
    for (size_t start = 0; start + batchSize <= ds.size(); start += batchSize) {
      EncryptedBatch<PlainBackend> batch;
      for (int b = 0; b < batchSize; ++b) {
        const size_t i = start + static_cast<size_t>(b);
        auto oh = oneHot(ds.y[i], cfg.numClasses);
        batch.x.push_back(
            be.encryptSlots(packVector(ds.X[i], model.inputFormat(), layout)));
        batch.yRepeated.push_back(
            be.encryptSlots(packVector(oh, Format::Repeated, layout)));
        batch.yExpanded.push_back(
            be.encryptSlots(packVector(oh, Format::Expanded, layout)));
      }
      model.trainStep(be, batch, lr);
      model.bootstrapParameters(be);  // level refresh, a no-op on plain values
    }
  }

  const double after = accuracy();
  std::printf("  accuracy %.3f -> %.3f\n", before, after);
  check(after > 0.85, "training should reach a high accuracy on separable blobs");
  check(after > before, "training should improve on the initial model");
}

}  // namespace

int main() {
  testAgainstReference();
  testConvergence();
  if (failures == 0) {
    std::printf("all training tests passed\n");
    return 0;
  }
  std::printf("%d failure(s)\n", failures);
  return 1;
}
