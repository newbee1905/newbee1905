// SPDX-License-Identifier: GPL-3.0-or-later
//
// Encrypted ReBoot training on the KeyMemRT runtime.
//
// The run has four phases, matching what KeyMemRT-Compiler emits for an
// inference program but driven at run time because the training loop is not
// expressible in the compiler's IR today:
//
//   1. plan        - a plaintext pass over the same code records which
//                    rotations the step performs and how deep the level
//                    schedule goes, which sizes the CKKS parameters.
//   2. calibrate   - one client-side run with all keys resident records the
//                    real (rotation index, ciphertext level) pairs.
//   3. provision   - keys are generated, compressed per level, written one file
//                    each, and dropped from the context; the bootstrapping key
//                    set is written as one bundle.
//   4. train       - the server trains with KeyMemRT paging keys in and out.
//
// The whole thing is a single process, like KeyMemRT's own benchmark drivers:
// Platform::CLIENT covers phases 1-3, Platform::SERVER phase 4.

#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "KeyMemRT.hpp"
#include "ResourceMonitor.hpp"
#include "openfhe.h"
#include "reboot/Backend.hpp"
#include "reboot/CkksBackend.hpp"
#include "reboot/CkksContext.hpp"
#include "reboot/Data.hpp"
#include "reboot/KeyPlan.hpp"
#include "reboot/Model.hpp"
#include "reboot/Trainer.hpp"

using namespace reboot;
using namespace lbcrypto;

// The KeyMemRT runtime instance.  Generated KeyMemRT code refers to a global of
// exactly this name; keeping the convention lets compiler-emitted kernels be
// dropped into this driver unchanged.
KeyMemRT keymem_rt;

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
      "reboot_train - encrypted ReBoot training on KeyMemRT\n\n"
      "KeyMemRT options (parsed by the runtime):\n"
      "  --key-mode <ignore|imperative|prefetch|speculative>\n"
      "  --input-dir <dir>      where serialized keys live\n"
      "  --prefetch-sat <n>     KeyMemRT tower budget for PREFETCH\n"
      "  --log-level <level>    debug|info|warning|error|off\n\n"
      "Model and training:\n"
      "  --hidden a,b           hidden layer widths (default 32,16)\n"
      "  --dim n                synthetic input dimension (16)\n"
      "  --classes n            number of classes (4)\n"
      "  --samples n            number of synthetic samples (64)\n"
      "  --csv path             CSV dataset instead (last column = label)\n"
      "  --epochs n             epochs (1)\n"
      "  --batch-size n         batch size (2)\n"
      "  --lr f                 learning rate (0.005)\n\n"
      "CKKS:\n"
      "  --ring-dim n           ring dimension (8192)\n"
      "  --depth n              compute depth override (default: from the plan)\n"
      "  --scaling-mod n        scaling modulus size (50)\n"
      "  --no-bootstrap         skip weight bootstrapping (short runs only)\n"
      "  --level-budget a,b     bootstrapping level budget (2,2)\n"
      "  --bs-iterations n      bootstrapping iterations (1)\n"
      "  --bs-precision n       iterative bootstrapping precision (0)\n"
      "  --secure               use HEStd_128_classic instead of HEStd_NotSet\n\n"
      "Key management:\n"
      "  --lookahead n          rotations to prefetch ahead (8)\n"
      "  --plan path            key plan file (default <input-dir>/reboot.plan)\n"
      "  --reuse-plan           load the plan instead of calibrating\n"
      "  --result-dir dir       resource monitor output directory (./results)\n");
}

bool ensureDir(const std::string &path) {
  if (path.empty()) return true;
  struct stat st;
  if (stat(path.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
  return mkdir(path.c_str(), 0755) == 0;
}

}  // namespace

int main(int argc, char **argv) {
  // KeyMemRT parses its own flags (mode, directories, logging, prefetch budget).
  keymem_rt.initFromArgs(argc, argv);

  std::vector<int> hidden{32, 16};
  std::vector<int> levelBudget{2, 2};
  int dim = 16, classes = 4, samples = 64;
  int epochs = 1, batchSize = 2, ringDim = 8192;
  int depthOverride = 0, scalingMod = 50, lookahead = 8;
  int bsIterations = 1, bsPrecision = 0;
  double lr = 0.005;
  bool bootstrap = true, reusePlan = false, secure = false;
  std::string csv, planPath, resultDir = "./results";

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--hidden") hidden = parseIntList(next());
    else if (a == "--dim") dim = std::atoi(next().c_str());
    else if (a == "--classes") classes = std::atoi(next().c_str());
    else if (a == "--samples") samples = std::atoi(next().c_str());
    else if (a == "--csv") csv = next();
    else if (a == "--epochs") epochs = std::atoi(next().c_str());
    else if (a == "--batch-size") batchSize = std::atoi(next().c_str());
    else if (a == "--lr") lr = std::atof(next().c_str());
    else if (a == "--ring-dim") ringDim = std::atoi(next().c_str());
    else if (a == "--depth") depthOverride = std::atoi(next().c_str());
    else if (a == "--scaling-mod") scalingMod = std::atoi(next().c_str());
    else if (a == "--no-bootstrap") bootstrap = false;
    else if (a == "--level-budget") levelBudget = parseIntList(next());
    else if (a == "--bs-iterations") bsIterations = std::atoi(next().c_str());
    else if (a == "--bs-precision") bsPrecision = std::atoi(next().c_str());
    else if (a == "--secure") secure = true;
    else if (a == "--lookahead") lookahead = std::atoi(next().c_str());
    else if (a == "--plan") planPath = next();
    else if (a == "--reuse-plan") reusePlan = true;
    else if (a == "--result-dir") resultDir = next();
    else if (a == "--help" || a == "-h") { printHelp(); return 0; }
    // Flags consumed by KeyMemRT's own parser are skipped here.
    else if (a == "--key-mode" || a == "--input-dir" || a == "--output-dir" ||
             a == "--prefetch-sat" || a == "--log-level" || a == "--log-file" ||
             a == "--output-base") { ++i; }
    else if (a == "--verbose" || a == "-v" || a == "--ser-single-file" ||
             a == "--log-console-off") { /* KeyMemRT flag */ }
    else { std::printf("unknown option %s\n", a.c_str()); return 1; }
  }

  const std::string keyDir = BenchmarkCLI::getInputDir();
  if (!ensureDir(keyDir)) {
    std::printf("cannot create key directory %s\n", keyDir.c_str());
    return 1;
  }
  ensureDir(resultDir);
  if (planPath.empty()) planPath = keyDir + "/reboot.plan";

  // ---- data and architecture ------------------------------------------------
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
  const Layout layout = LocalLossMLP<PlainBackend>::recommendLayout(cfg, numSlots);

  // ---- phase 1: plaintext plan ---------------------------------------------
  KeyPlan plainPlan;
  {
    PlainBackend be(numSlots);
    be.setKeyPlan(&plainPlan);
    LocalLossMLP<PlainBackend> model(cfg, layout);
    model.initWeights();
    model.encrypt(be);
    std::printf("%s", model.describe().c_str());
    auto batch = encryptBatch(be, model, layout, ds, 0, 1, cfg.numClasses);
    for (int step = 0; step < 2; ++step) {
      model.trainStep(be, batch, lr);
      model.bootstrapParameters(be);
    }
    std::printf("\n%s\n", plainPlan.summary().c_str());
  }

  CkksParams cparams;
  // With bootstrapping the level schedule is periodic, so one step's depth plus
  // a small margin is enough.  Without it every step starts lower than the last
  // and even the closing evaluation pass has to fit, hence the doubled budget.
  cparams.computeDepth = depthOverride > 0 ? depthOverride
                         : bootstrap      ? plainPlan.maxLevel() + 2
                                          : 2 * plainPlan.maxLevel() + 4;
  cparams.ringDim = ringDim;
  cparams.scalingModSize = scalingMod;
  cparams.firstModSize = scalingMod + 1;
  cparams.bootstrap = bootstrap;
  cparams.levelBudget.assign(levelBudget.begin(), levelBudget.end());
  cparams.bsIterations = bsIterations;
  cparams.bsPrecision = bsPrecision;
  cparams.standardSecurity = secure;

  CkksSetup setup = makeContext(cparams);
  const std::string keyTag = setup.keyPair.secretKey->GetKeyTag();

  keymem_rt.setCryptoContext(setup.cc);
  keymem_rt.setKeyTag(keyTag);
  keymem_rt.setMultDepth(setup.multDepth);
  keymem_rt.setPlatform(Platform::CLIENT);

  BootstrapKeyBundle bundle(setup.cc, keyTag, keyDir + "/bootstrap_keys.bin");

  // ---- phase 2/3: calibrate and provision -----------------------------------
  const KeyMemMode requestedMode = keymem_rt.getOperationMode();
  KeyPlan plan;

  keymem_rt.setKeyMemMode(KeyMemMode::IGNORE);  // client: all keys resident

  // The bootstrapping keys are generated first and written as one bundle, so
  // the bundle holds only them; the rotation keys are provisioned afterwards,
  // one file per (index, level).
  if (bootstrap) {
    std::printf("bootstrapping setup ...\n");
    setup.cc->EvalBootstrapSetup(cparams.levelBudget, {0, 0},
                                 static_cast<usint>(numSlots));
    setup.cc->EvalBootstrapKeyGen(setup.keyPair.secretKey,
                                  static_cast<usint>(numSlots));
    if (!bundle.serialize())
      std::printf("warning: could not serialize the bootstrap key bundle\n");
  }

  if (reusePlan && plan.load(planPath)) {
    std::printf("reusing key plan from %s\n%s\n", planPath.c_str(),
                plan.summary().c_str());
  } else {
    std::printf("calibration run (all keys resident) ...\n");
    setup.cc->EvalRotateKeyGen(setup.keyPair.secretKey, plainPlan.indices());

    CkksBackend calib(setup.cc, setup.keyPair.publicKey,
                      setup.keyPair.secretKey, &keymem_rt);
    calib.setBootstrapParams(bsIterations, bsPrecision);
    calib.setRecordingPlan(&plan);
    LocalLossMLP<CkksBackend> model(cfg, layout);
    model.initWeights();
    model.encrypt(calib);
    auto batch = encryptBatch(calib, model, layout, ds, 0, 1, cfg.numClasses);
    // Two steps when bootstrapping: the cold first step and the steady state
    // that follows the first weight refresh use different weight levels.
    // Without bootstrapping there is no steady state - the levels only ever go
    // down - so one step is all that can be calibrated.
    const int calibrationSteps = bootstrap ? 2 : 1;
    for (int step = 0; step < calibrationSteps; ++step) {
      model.trainStep(calib, batch, lr);
      if (bootstrap) model.bootstrapParameters(calib);
    }
    std::printf("\n%s", plan.summary().c_str());
    plan.save(planPath);
  }

  // Provisioning always runs: the key files belong to this run's secret key,
  // so only the plan itself is worth reusing across runs.
  keymem_rt.clearAllKeys();
  std::printf("provisioning rotation keys into %s ...\n", keyDir.c_str());
  keymem_rt.setKeyMemMode(requestedMode == KeyMemMode::IGNORE
                              ? KeyMemMode::IMPERATIVE
                              : requestedMode);
  if (!provisionRotationKeys(setup.cc, setup.keyPair.secretKey, keymem_rt, plan))
    std::printf("warning: some rotation keys failed to serialize\n");

  // ---- phase 4: encrypted training on the server ----------------------------
  keymem_rt.setKeyMemMode(requestedMode);
  keymem_rt.setPlatform(Platform::SERVER);
  keymem_rt.addRotIndices(plan.indices());

  // IGNORE is the baseline the paper compares against: no paging at all, so
  // every key has to be resident for the whole run.
  if (requestedMode == KeyMemMode::IGNORE) {
    std::printf("ignore mode: loading the full key set up front\n");
    setup.cc->EvalRotateKeyGen(setup.keyPair.secretKey, plan.indices());
    if (bootstrap && !bundle.load())
      std::printf("warning: could not load the bootstrap key bundle\n");
  }

  CkksBackend be(setup.cc, setup.keyPair.publicKey, setup.keyPair.secretKey,
                 &keymem_rt);
  be.setBootstrapParams(bsIterations, bsPrecision);
  be.setProvisionedPlan(&plan);
  if (requestedMode == KeyMemMode::PREFETCH) be.setPrefetch(&plan, lookahead);

  LocalLossMLP<CkksBackend> model(cfg, layout);
  model.initWeights();
  model.encrypt(be);

  ResourceMonitor monitor(true);
  const std::string csvPath = resultDir + "/reboot_" +
                              getModeString(requestedMode) + "_" +
                              layout.str() + ".csv";
  monitor.start(csvPath);

  TrainOptions opt;
  opt.epochs = epochs;
  opt.batchSize = batchSize;
  opt.lr = lr;
  opt.bootstrapEachStep = bootstrap;
  opt.verbose = true;

  TrainHooks hooks;
  hooks.beforeStep = [&]() { be.resetCursor(); };
  if (bootstrap) {
    // The bootstrapping keys are staged as one bundle: OpenFHE drives those
    // rotations internally, so they cannot be paged one at a time, but they
    // stay out of memory for the rest of the step.
    hooks.beforeBootstrap = [&]() {
      if (requestedMode != KeyMemMode::IGNORE && !bundle.load())
        std::printf("warning: could not load the bootstrap key bundle\n");
    };
    // Unstaging goes through KeyMemRT rather than ClearEvalAutomorphismKeys so
    // that its PREFETCH bookkeeping (ready set, tower budget) is reset with the
    // key map instead of being left describing keys that no longer exist.
    hooks.afterBootstrap = [&]() {
      if (requestedMode != KeyMemMode::IGNORE) keymem_rt.clearAllKeys();
    };
  }

  const auto t0 = std::chrono::high_resolution_clock::now();
  runTraining(be, model, ds, cfg, layout, opt, hooks);
  const auto t1 = std::chrono::high_resolution_clock::now();

  monitor.stop();
  monitor.save_to_file(csvPath);

  // Counted before the evaluation pass, which is client-side book-keeping
  // rather than part of the training work being measured.
  const long rotations = be.rotationCount();
  const long bootstraps = be.bootstrapCount();
  const double accuracy = evaluate(be, model, layout, ds, cfg.numClasses,
                                   std::min<size_t>(ds.size(), 32));

  std::printf(
      "\nencrypted training done\n"
      "  key mode        : %s\n"
      "  rotations       : %ld\n"
      "  bootstraps      : %ld\n"
      "  wall clock      : %.1f s\n"
      "  accuracy        : %.1f%%\n"
      "  resource trace  : %s\n",
      getModeString(requestedMode).c_str(), rotations, bootstraps,
      std::chrono::duration<double>(t1 - t0).count(), 100.0 * accuracy,
      csvPath.c_str());
  keymem_rt.printKeyStats();
  return 0;
}
