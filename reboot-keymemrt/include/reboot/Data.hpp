// SPDX-License-Identifier: GPL-3.0-or-later
//
// Small data helpers for the benchmark driver: a synthetic classification task
// that needs no download, plus a CSV loader for the tabular datasets the ReBoot
// experiments use (last column = integer label).

#ifndef REBOOT_DATA_HPP_
#define REBOOT_DATA_HPP_

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace reboot {

struct Dataset {
  std::vector<std::vector<double>> X;
  std::vector<int> y;
  int dim = 0;
  int numClasses = 0;

  size_t size() const { return X.size(); }
};

// Gaussian blobs with one centre per class, scaled into a range where the
// degree-two PolyReLU stays well conditioned.
inline Dataset makeBlobs(int numSamples, int dim, int numClasses, unsigned seed,
                         double spread = 0.35) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> noise(0.0, spread);
  std::uniform_real_distribution<double> centreDist(-1.0, 1.0);

  std::vector<std::vector<double>> centres(numClasses,
                                           std::vector<double>(dim, 0.0));
  for (auto &c : centres)
    for (auto &v : c) v = centreDist(rng);

  Dataset ds;
  ds.dim = dim;
  ds.numClasses = numClasses;
  ds.X.resize(numSamples, std::vector<double>(dim, 0.0));
  ds.y.resize(numSamples);
  std::uniform_int_distribution<int> classDist(0, numClasses - 1);
  for (int i = 0; i < numSamples; ++i) {
    const int c = classDist(rng);
    ds.y[static_cast<size_t>(i)] = c;
    for (int j = 0; j < dim; ++j)
      ds.X[static_cast<size_t>(i)][static_cast<size_t>(j)] =
          centres[static_cast<size_t>(c)][static_cast<size_t>(j)] + noise(rng);
  }
  return ds;
}

inline Dataset loadCsv(const std::string &path, bool hasHeader = true) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open dataset: " + path);
  Dataset ds;
  std::string line;
  if (hasHeader) std::getline(f, line);
  int maxLabel = -1;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::stringstream ls(line);
    std::string cell;
    std::vector<double> row;
    while (std::getline(ls, cell, ',')) row.push_back(std::stod(cell));
    if (row.size() < 2) continue;
    const int label = static_cast<int>(row.back());
    row.pop_back();
    maxLabel = std::max(maxLabel, label);
    ds.dim = static_cast<int>(row.size());
    ds.X.push_back(std::move(row));
    ds.y.push_back(label);
  }
  ds.numClasses = maxLabel + 1;
  return ds;
}

// Scale every feature to [-1, 1] using the training range.
inline void normalise(Dataset &ds) {
  if (ds.X.empty()) return;
  std::vector<double> lo(ds.X[0]), hi(ds.X[0]);
  for (const auto &row : ds.X)
    for (int j = 0; j < ds.dim; ++j) {
      lo[static_cast<size_t>(j)] = std::min(lo[static_cast<size_t>(j)], row[static_cast<size_t>(j)]);
      hi[static_cast<size_t>(j)] = std::max(hi[static_cast<size_t>(j)], row[static_cast<size_t>(j)]);
    }
  for (auto &row : ds.X)
    for (int j = 0; j < ds.dim; ++j) {
      const double span = hi[static_cast<size_t>(j)] - lo[static_cast<size_t>(j)];
      row[static_cast<size_t>(j)] =
          span > 1e-12 ? 2.0 * (row[static_cast<size_t>(j)] - lo[static_cast<size_t>(j)]) / span - 1.0
                       : 0.0;
    }
}

inline std::vector<double> oneHot(int label, int numClasses) {
  std::vector<double> v(static_cast<size_t>(numClasses), 0.0);
  v[static_cast<size_t>(label)] = 1.0;
  return v;
}

inline void shuffle(Dataset &ds, std::mt19937 &rng) {
  std::vector<size_t> order(ds.size());
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<std::vector<double>> X;
  std::vector<int> y;
  X.reserve(ds.size());
  y.reserve(ds.size());
  for (size_t i : order) {
    X.push_back(ds.X[i]);
    y.push_back(ds.y[i]);
  }
  ds.X = std::move(X);
  ds.y = std::move(y);
}

}  // namespace reboot

#endif  // REBOOT_DATA_HPP_
