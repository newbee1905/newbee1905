// SPDX-License-Identifier: GPL-3.0-or-later
//
// Checks that the rotate-and-add rewrite of EvalSumRows / EvalSumCols computes
// exactly the matrix algebra ReBoot expects, in both packings and in both
// directions, and that the weight gradient lands in the layer's own layout.
//
// The tests run on the plaintext slot backend, which shares its code path with
// the CKKS backend, so a failure here is a failure of the packing algebra
// rather than of the crypto.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "reboot/Backend.hpp"
#include "reboot/Layers.hpp"
#include "reboot/Layout.hpp"
#include "reboot/LinAlg.hpp"

using namespace reboot;

namespace {

int failures = 0;

void check(bool cond, const char *what) {
  if (!cond) {
    std::printf("  FAIL: %s\n", what);
    ++failures;
  }
}

void checkClose(const std::vector<double> &got, const std::vector<double> &want,
                const char *what, double tol = 1e-9) {
  if (got.size() != want.size()) {
    std::printf("  FAIL: %s (size %zu vs %zu)\n", what, got.size(), want.size());
    ++failures;
    return;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::fabs(got[i] - want[i]) > tol) {
      std::printf("  FAIL: %s at %zu: %g vs %g\n", what, i, got[i], want[i]);
      ++failures;
      return;
    }
  }
}

std::vector<double> randomVector(size_t n, std::mt19937 &rng) {
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  std::vector<double> v(n);
  for (auto &x : v) x = d(rng);
  return v;
}

// out[j] = sum_i a[i] * W[i][j]
std::vector<double> matvec(const std::vector<double> &a,
                           const std::vector<double> &w, int in, int out) {
  std::vector<double> r(static_cast<size_t>(out), 0.0);
  for (int j = 0; j < out; ++j)
    for (int i = 0; i < in; ++i)
      r[static_cast<size_t>(j)] += a[static_cast<size_t>(i)] *
                                   w[static_cast<size_t>(i) * out + j];
  return r;
}

// out[i] = sum_j W[i][j] * d[j]
std::vector<double> matvecT(const std::vector<double> &d,
                            const std::vector<double> &w, int in, int out) {
  std::vector<double> r(static_cast<size_t>(in), 0.0);
  for (int i = 0; i < in; ++i)
    for (int j = 0; j < out; ++j)
      r[static_cast<size_t>(i)] += w[static_cast<size_t>(i) * out + j] *
                                   d[static_cast<size_t>(j)];
  return r;
}

void testPackingRoundTrip(std::mt19937 &rng) {
  std::printf("packing round trip\n");
  Layout layout(8, 4);
  auto v = randomVector(4, rng);
  auto rep = packVector(v, Format::Repeated, layout);
  checkClose(unpackVector(rep, Format::Repeated, layout, 4), v, "repeated");
  auto u = randomVector(8, rng);
  auto exp = packVector(u, Format::Expanded, layout);
  checkClose(unpackVector(exp, Format::Expanded, layout, 8), u, "expanded");

  auto w = randomVector(5 * 3, rng);
  auto packedRow = packWeights(w, 5, 3, true, layout);
  checkClose(unpackWeights(packedRow, 5, 3, true, layout), w, "weights row");
  auto w2 = randomVector(3 * 6, rng);
  auto packedCol = packWeights(w2, 3, 6, false, layout);
  checkClose(unpackWeights(packedCol, 3, 6, false, layout), w2, "weights col");
}

void testRowPackedLayer(std::mt19937 &rng) {
  std::printf("row-packed layer (Expanded -> Repeated)\n");
  Layout layout(8, 4);
  PlainBackend be(layout.slots());
  Mask mask{"first_column", firstColumnMask(layout)};

  const int in = 5, out = 3;
  auto a = randomVector(in, rng);
  auto w = randomVector(static_cast<size_t>(in) * out, rng);

  auto aCt = be.encryptSlots(packVector(a, Format::Expanded, layout));
  auto wCt = be.encryptSlots(packWeights(w, in, out, true, layout));

  auto z = matmulRE(be, aCt, wCt, layout);
  checkClose(unpackVector(be.decryptSlots(z), Format::Repeated, layout, out),
             matvec(a, w, in, out), "forward");

  // Backward: Repeated delta -> Expanded gradient towards the input.
  auto d = randomVector(out, rng);
  auto dCt = be.encryptSlots(packVector(d, Format::Repeated, layout));
  auto g = sumCols(be, be.mul(dCt, wCt), layout, mask);
  checkClose(unpackVector(be.decryptSlots(g), Format::Expanded, layout, in),
             matvecT(d, w, in, out), "backward");

  // Weight gradient: outer product in the layer's own packing.
  auto grad = outerProductSum(be, {aCt}, {dCt});
  std::vector<double> want(static_cast<size_t>(in) * out);
  for (int i = 0; i < in; ++i)
    for (int j = 0; j < out; ++j)
      want[static_cast<size_t>(i) * out + j] = a[static_cast<size_t>(i)] *
                                               d[static_cast<size_t>(j)];
  checkClose(unpackWeights(be.decryptSlots(grad), in, out, true, layout), want,
             "weight gradient");
}

void testColPackedLayer(std::mt19937 &rng) {
  std::printf("column-packed layer (Repeated -> Expanded)\n");
  Layout layout(8, 4);
  PlainBackend be(layout.slots());
  Mask mask{"first_column", firstColumnMask(layout)};

  const int in = 3, out = 6;  // in <= cols, out <= rows
  auto a = randomVector(in, rng);
  auto w = randomVector(static_cast<size_t>(in) * out, rng);

  auto aCt = be.encryptSlots(packVector(a, Format::Repeated, layout));
  auto wCt = be.encryptSlots(packWeights(w, in, out, false, layout));

  auto y = matmulCE(be, aCt, wCt, layout, mask);
  checkClose(unpackVector(be.decryptSlots(y), Format::Expanded, layout, out),
             matvec(a, w, in, out), "forward");

  auto d = randomVector(out, rng);
  auto dCt = be.encryptSlots(packVector(d, Format::Expanded, layout));
  auto g = sumRows(be, be.mul(dCt, wCt), layout);
  checkClose(unpackVector(be.decryptSlots(g), Format::Repeated, layout, in),
             matvecT(d, w, in, out), "backward");

  auto grad = outerProductSum(be, {aCt}, {dCt});
  std::vector<double> want(static_cast<size_t>(in) * out);
  for (int i = 0; i < in; ++i)
    for (int j = 0; j < out; ++j)
      want[static_cast<size_t>(i) * out + j] = a[static_cast<size_t>(i)] *
                                               d[static_cast<size_t>(j)];
  checkClose(unpackWeights(be.decryptSlots(grad), in, out, false, layout), want,
             "weight gradient");
}

void testBatchAccumulation(std::mt19937 &rng) {
  std::printf("gradient accumulation over a batch\n");
  Layout layout(16, 8);
  PlainBackend be(layout.slots());
  const int in = 6, out = 5, batch = 4;

  std::vector<PlainValue> xs, ds;
  std::vector<std::vector<double>> xp, dp;
  for (int b = 0; b < batch; ++b) {
    xp.push_back(randomVector(in, rng));
    dp.push_back(randomVector(out, rng));
    xs.push_back(be.encryptSlots(packVector(xp.back(), Format::Expanded, layout)));
    ds.push_back(be.encryptSlots(packVector(dp.back(), Format::Repeated, layout)));
  }
  auto grad = outerProductSum(be, xs, ds);
  std::vector<double> want(static_cast<size_t>(in) * out, 0.0);
  for (int b = 0; b < batch; ++b)
    for (int i = 0; i < in; ++i)
      for (int j = 0; j < out; ++j)
        want[static_cast<size_t>(i) * out + j] +=
            xp[static_cast<size_t>(b)][static_cast<size_t>(i)] *
            dp[static_cast<size_t>(b)][static_cast<size_t>(j)];
  checkClose(unpackWeights(be.decryptSlots(grad), in, out, true, layout), want,
             "batched weight gradient");
}

void testRotationBudget() {
  std::printf("rotation index budget\n");
  Layout layout(64, 16);
  PlainBackend be(layout.slots());
  KeyPlan plan;
  be.setKeyPlan(&plan);
  Mask mask{"first_column", firstColumnMask(layout)};

  auto x = be.encryptSlots(std::vector<double>(layout.slots(), 1.0));
  sumRows(be, x, layout);
  sumCols(be, x, layout, mask);
  sumAll(be, x, layout);

  const auto declared = rotationIndices(layout);
  for (int idx : plan.indices()) {
    bool found = false;
    for (int d : declared) found = found || d == idx;
    check(found, "rotation index used but not declared by rotationIndices()");
  }
  // sumRows uses log2(rows) rotations, sumCols 2*log2(cols).
  check(plan.numRotations() ==
            6 /* sumRows: 64 rows */ + 8 /* sumCols: 2*log2(16) */ +
                10 /* sumAll: log2(1024) */,
        "unexpected rotation count");
}

void testLevelModel() {
  std::printf("level accounting\n");
  Layout layout(8, 4);
  PlainBackend be(layout.slots());
  Mask mask{"first_column", firstColumnMask(layout)};
  auto x = be.encryptSlots(std::vector<double>(layout.slots(), 1.0));

  // RE-Matmul spends one level (the product), CE-Matmul two (product + mask).
  check(be.level(matmulRE(be, x, x, layout)) == 1, "RE-Matmul depth");
  check(be.level(matmulCE(be, x, x, layout, mask)) == 2, "CE-Matmul depth");
  check(be.level(be.rotate(x, 4)) == 0, "rotation is level preserving");
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  testPackingRoundTrip(rng);
  testRowPackedLayer(rng);
  testColPackedLayer(rng);
  testBatchAccumulation(rng);
  testRotationBudget();
  testLevelModel();

  if (failures == 0) {
    std::printf("all linalg tests passed\n");
    return 0;
  }
  std::printf("%d failure(s)\n", failures);
  return 1;
}
