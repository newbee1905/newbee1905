// SPDX-License-Identifier: GPL-3.0-or-later
//
// Plaintext driver and key planner.
//
// Runs the exact code path of the encrypted driver on the plaintext slot
// backend.  Two uses:
//
//   * planning  - a couple of steps with a KeyPlan attached report the rotation
//                 indices the network needs and the level schedule, which sizes
//                 the multiplicative depth of the CKKS context and the set of
//                 keys to provision.  No crypto, no OpenFHE, seconds instead of
//                 hours.
//   * reference - training accuracy without encryption noise, to compare
//                 against an encrypted run of the same configuration.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "reboot/Backend.hpp"
#include "reboot/Data.hpp"
#include "reboot/KeyPlan.hpp"
#include "reboot/Model.hpp"
#include "reboot/Trainer.hpp"

using namespace reboot;

namespace {

std::vector<int> parseIntList(const std::string &s) {
  std::vector<int> out;
  size_t pos = 0;
  while (pos < s.size()) {
    size_t comma = s.find(',', pos);
    if (comma == std::string::npos) comma = s.size();
    out.push_back(std::atoi(s.substr(pos, comma - pos).c_str()));
    pos = comma + 1;
  }
  return out;
}

void printHelp() {
  std::printf(
      "reboot_plain - plaintext ReBoot training and rotation-key planner\n\n"
      "  --hidden a,b        hidden layer widths (default 32,16)\n"
      "  --dim n             input dimension of the synthetic dataset (16)\n"
      "  --classes n         number of classes (4)\n"
      "  --samples n         number of synthetic samples (512)\n"
      "  --csv path          load a CSV dataset instead (last column = label)\n"
      "  --ring-dim n        ring dimension used to size the layout (8192)\n"
      "  --epochs n          training epochs (3)\n"
      "  --batch-size n      batch size (8)\n"
      "  --lr f              learning rate (0.01)\n"
      "  --plan-out path     write the rotation key plan to this file\n"
      "  --quiet             only print epoch summaries\n");
}

}  // namespace

int main(int argc, char **argv) {
  std::vector<int> hidden{32, 16};
  int dim = 16, classes = 4, samples = 512, ringDim = 8192;
  int epochs = 3, batchSize = 8;
  double lr = 0.01;
  std::string csv, planOut;
  bool quiet = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--hidden") hidden = parseIntList(next());
    else if (a == "--dim") dim = std::atoi(next().c_str());
    else if (a == "--classes") classes = std::atoi(next().c_str());
    else if (a == "--samples") samples = std::atoi(next().c_str());
    else if (a == "--csv") csv = next();
    else if (a == "--ring-dim") ringDim = std::atoi(next().c_str());
    else if (a == "--epochs") epochs = std::atoi(next().c_str());
    else if (a == "--batch-size") batchSize = std::atoi(next().c_str());
    else if (a == "--lr") lr = std::atof(next().c_str());
    else if (a == "--plan-out") planOut = next();
    else if (a == "--quiet") quiet = true;
    else if (a == "--help" || a == "-h") { printHelp(); return 0; }
    else { std::printf("unknown option %s\n", a.c_str()); return 1; }
  }

  Dataset ds = csv.empty() ? makeBlobs(samples, dim, classes, /*seed=*/5)
                           : loadCsv(csv);
  if (!csv.empty()) normalise(ds);

  ModelConfig cfg;
  cfg.inputDim = ds.dim;
  cfg.hidden = hidden;
  cfg.numClasses = ds.numClasses;
  cfg.momentum = 0.9;
  cfg.weightDecay = 0.0;
  cfg.seed = 1;

  const int numSlots = ringDim / 2;
  Layout layout = LocalLossMLP<PlainBackend>::recommendLayout(cfg, numSlots);
  std::printf("dataset: %zu samples, dim %d, %d classes\n", ds.size(), ds.dim,
              ds.numClasses);

  // ---- planning pass -------------------------------------------------------
  {
    PlainBackend be(numSlots);
    KeyPlan plan;
    be.setKeyPlan(&plan);
    LocalLossMLP<PlainBackend> model(cfg, layout);
    model.initWeights();
    model.encrypt(be);
    std::printf("%s", model.describe().c_str());

    auto batch = encryptBatch(be, model, layout, ds, 0, 1, cfg.numClasses);
    for (int step = 0; step < 2; ++step) {
      model.trainStep(be, batch, lr);
      model.bootstrapParameters(be);
    }
    std::printf("\n%s", plan.summary().c_str());
    std::printf(
        "rotations per training step (batch of 1): %zu\n"
        "compute depth required     : %d levels\n\n",
        plan.numRotations() / 2, plan.maxLevel());
    if (!planOut.empty() && plan.save(planOut))
      std::printf("wrote key plan to %s\n\n", planOut.c_str());
  }

  // ---- training ------------------------------------------------------------
  PlainBackend be(numSlots);
  LocalLossMLP<PlainBackend> model(cfg, layout);
  model.initWeights();
  model.encrypt(be);

  TrainOptions opt;
  opt.epochs = epochs;
  opt.batchSize = batchSize;
  opt.lr = lr;
  opt.bootstrapEachStep = true;
  opt.verbose = !quiet;

  std::printf("accuracy before training: %.1f%%\n",
              100.0 * evaluate(be, model, layout, ds, cfg.numClasses, ds.size()));
  runTraining(be, model, ds, cfg, layout, opt);
  std::printf("accuracy after training : %.1f%%\n",
              100.0 * evaluate(be, model, layout, ds, cfg.numClasses, ds.size()));
  return 0;
}
