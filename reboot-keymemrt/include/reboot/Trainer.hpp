// SPDX-License-Identifier: GPL-3.0-or-later
//
// Batch preparation, evaluation and the training loop, shared by the plaintext
// and the encrypted driver.

#ifndef REBOOT_TRAINER_HPP_
#define REBOOT_TRAINER_HPP_

#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "reboot/Data.hpp"
#include "reboot/Layout.hpp"
#include "reboot/Model.hpp"

namespace reboot {

// Encrypt one batch the way a client would: inputs in the format the first
// layer consumes and one-hot labels in both packings (the local classifiers
// alternate their output format down the network).
template <class B>
EncryptedBatch<B> encryptBatch(B &be, const LocalLossMLP<B> &model,
                               const Layout &layout, const Dataset &ds,
                               size_t start, size_t count, int numClasses) {
  EncryptedBatch<B> batch;
  for (size_t k = 0; k < count; ++k) {
    const size_t i = start + k;
    const auto oh = oneHot(ds.y[i], numClasses);
    batch.x.push_back(
        be.encryptSlots(packVector(ds.X[i], model.inputFormat(), layout)));
    batch.yRepeated.push_back(
        be.encryptSlots(packVector(oh, Format::Repeated, layout)));
    batch.yExpanded.push_back(
        be.encryptSlots(packVector(oh, Format::Expanded, layout)));
  }
  return batch;
}

// Class scores for one sample, decrypted and unpacked.
template <class B>
std::vector<double> predictOne(B &be, LocalLossMLP<B> &model,
                               const Layout &layout,
                               const std::vector<double> &x, int numClasses) {
  std::vector<typename B::Ct> single{
      be.encryptSlots(packVector(x, model.inputFormat(), layout))};
  auto pred = model.forward(be, single, /*training=*/false);
  return readPrediction(be.decryptSlots(pred[0]), model.outputFormat(), layout,
                        numClasses);
}

template <class B>
double evaluate(B &be, LocalLossMLP<B> &model, const Layout &layout,
                const Dataset &ds, int numClasses, size_t limit) {
  const size_t n = std::min(limit, ds.size());
  int correct = 0;
  for (size_t i = 0; i < n; ++i) {
    auto scores = predictOne(be, model, layout, ds.X[i], numClasses);
    if (argmax(scores) == ds.y[i]) ++correct;
  }
  return n == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(n);
}

struct TrainOptions {
  int epochs = 1;
  int batchSize = 8;
  double lr = 0.02;
  bool bootstrapEachStep = true;
  size_t evalSamples = 0;  // 0 = evaluate on the whole training set
  bool verbose = true;
};

struct TrainHooks {
  std::function<void()> beforeStep;
  std::function<void()> beforeBootstrap;
  std::function<void()> afterBootstrap;
  std::function<void(int epoch, int batch, double loss, double acc, double ms)>
      onBatch;
};

// Residual-sum-of-squares loss and accuracy of a decrypted prediction batch.
inline void batchMetrics(const std::vector<std::vector<double>> &preds,
                         const std::vector<int> &labels, int numClasses,
                         double &loss, double &acc) {
  loss = 0.0;
  int correct = 0;
  for (size_t b = 0; b < preds.size(); ++b) {
    for (int j = 0; j < numClasses; ++j) {
      const double target = (labels[b] == j) ? 1.0 : 0.0;
      const double diff = preds[b][static_cast<size_t>(j)] - target;
      loss += diff * diff;
    }
    if (argmax(preds[b]) == labels[b]) ++correct;
  }
  acc = preds.empty() ? 0.0
                      : static_cast<double>(correct) /
                            static_cast<double>(preds.size());
}

template <class B>
void runTraining(B &be, LocalLossMLP<B> &model, Dataset &ds,
                 const ModelConfig &cfg, const Layout &layout,
                 const TrainOptions &opt, const TrainHooks &hooks = {}) {
  std::mt19937 rng(cfg.seed);
  const size_t batchSize = static_cast<size_t>(opt.batchSize);

  for (int epoch = 0; epoch < opt.epochs; ++epoch) {
    shuffle(ds, rng);
    double epochLoss = 0.0, epochAcc = 0.0;
    int batches = 0;

    for (size_t start = 0; start + batchSize <= ds.size(); start += batchSize) {
      if (hooks.beforeStep) hooks.beforeStep();
      auto batch =
          encryptBatch(be, model, layout, ds, start, batchSize, cfg.numClasses);

      // Timed from here: encrypting the batch is the client's work, the step
      // and the weight refresh are the server's.
      const auto t0 = std::chrono::high_resolution_clock::now();
      auto pred = model.trainStep(be, batch, opt.lr);

      if (opt.bootstrapEachStep) {
        if (hooks.beforeBootstrap) hooks.beforeBootstrap();
        model.bootstrapParameters(be);
        if (hooks.afterBootstrap) hooks.afterBootstrap();
      }

      const auto t1 = std::chrono::high_resolution_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();

      std::vector<std::vector<double>> decoded;
      std::vector<int> labels;
      for (size_t k = 0; k < batchSize; ++k) {
        decoded.push_back(readPrediction(be.decryptSlots(pred[k]),
                                         model.outputFormat(), layout,
                                         cfg.numClasses));
        labels.push_back(ds.y[start + k]);
      }
      double loss = 0.0, acc = 0.0;
      batchMetrics(decoded, labels, cfg.numClasses, loss, acc);
      epochLoss += loss;
      epochAcc += acc;
      ++batches;

      if (hooks.onBatch) hooks.onBatch(epoch, batches - 1, loss, acc, ms);
      if (opt.verbose)
        std::printf("  epoch %2d batch %3d | loss %8.4f | acc %5.1f%% | %8.1f ms\n",
                    epoch, batches - 1, loss, 100.0 * acc, ms);
    }

    if (batches > 0)
      std::printf("epoch %2d | mean loss %8.4f | mean acc %5.1f%%\n", epoch,
                  epochLoss / batches, 100.0 * epochAcc / batches);
  }
}

}  // namespace reboot

#endif  // REBOOT_TRAINER_HPP_
