// SPDX-License-Identifier: GPL-3.0-or-later
//
// Small data helpers for the benchmark driver: a synthetic classification task
// that needs no download, plus a CSV loader for the tabular datasets the ReBoot
// experiments use (last column = integer label).

#ifndef REBOOT_DATA_H_
#define REBOOT_DATA_H_

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
	std::vector<std::vector<double>> features;
	std::vector<int> y;
	int dim = 0;
	int num_classes = 0;

	size_t size() const { return features.size(); }
};

// Gaussian blobs with one centre per class, scaled into a range where the
// degree-two PolyReLU stays well conditioned.
inline Dataset make_blobs(int num_samples, int dim, int num_classes,
						  unsigned seed, double spread = 0.35) {
	std::mt19937 rng(seed);
	std::normal_distribution<double> noise(0.0, spread);
	std::uniform_real_distribution<double> centre_dist(-1.0, 1.0);

	std::vector<std::vector<double>> centres(num_classes,
											 std::vector<double>(dim, 0.0));
	for (auto &c : centres)
		for (auto &v : c) v = centre_dist(rng);

	Dataset ds;
	ds.dim = dim;
	ds.num_classes = num_classes;
	ds.features.resize(num_samples, std::vector<double>(dim, 0.0));
	ds.y.resize(num_samples);
	std::uniform_int_distribution<int> class_dist(0, num_classes - 1);
	for (int i = 0; i < num_samples; ++i) {
		const int c = class_dist(rng);
		ds.y[static_cast<size_t>(i)] = c;
		for (int j = 0; j < dim; ++j)
			ds.features[static_cast<size_t>(i)][static_cast<size_t>(j)] =
				centres[static_cast<size_t>(c)][static_cast<size_t>(j)] +
				noise(rng);
	}
	return ds;
}

inline Dataset load_csv(const std::string &path, bool has_header = true) {
	std::ifstream f(path);
	if (!f) throw std::runtime_error("cannot open dataset: " + path);
	Dataset ds;
	std::string line;
	if (has_header) std::getline(f, line);
	int max_label = -1;
	while (std::getline(f, line)) {
		if (line.empty()) continue;
		std::stringstream ls(line);
		std::string cell;
		std::vector<double> row;
		while (std::getline(ls, cell, ',')) row.push_back(std::stod(cell));
		if (row.size() < 2) continue;
		const int label = static_cast<int>(row.back());
		row.pop_back();
		max_label = std::max(max_label, label);
		ds.dim = static_cast<int>(row.size());
		ds.features.push_back(std::move(row));
		ds.y.push_back(label);
	}
	ds.num_classes = max_label + 1;
	return ds;
}

// Scale every feature to [-1, 1] using the training range.
inline void normalise(Dataset &ds) {
	if (ds.features.empty()) return;
	std::vector<double> lo(ds.features[0]), hi(ds.features[0]);
	for (const auto &row : ds.features)
		for (int j = 0; j < ds.dim; ++j) {
			lo[static_cast<size_t>(j)] = std::min(lo[static_cast<size_t>(j)],
												  row[static_cast<size_t>(j)]);
			hi[static_cast<size_t>(j)] = std::max(hi[static_cast<size_t>(j)],
												  row[static_cast<size_t>(j)]);
		}
	for (auto &row : ds.features)
		for (int j = 0; j < ds.dim; ++j) {
			const double span =
				hi[static_cast<size_t>(j)] - lo[static_cast<size_t>(j)];
			row[static_cast<size_t>(j)] =
				span > 1e-12 ? 2.0 *
									   (row[static_cast<size_t>(j)] -
										lo[static_cast<size_t>(j)]) /
									   span -
								   1.0
							 : 0.0;
		}
}

inline std::vector<double> one_hot(int label, int num_classes) {
	std::vector<double> v(static_cast<size_t>(num_classes), 0.0);
	v[static_cast<size_t>(label)] = 1.0;
	return v;
}

inline void shuffle(Dataset &ds, std::mt19937 &rng) {
	std::vector<size_t> order(ds.size());
	std::iota(order.begin(), order.end(), 0);
	std::shuffle(order.begin(), order.end(), rng);
	std::vector<std::vector<double>> features;
	std::vector<int> labels;
	features.reserve(ds.size());
	labels.reserve(ds.size());
	for (size_t i : order) {
		features.push_back(ds.features[i]);
		labels.push_back(ds.y[i]);
	}
	ds.features = std::move(features);
	ds.y = std::move(labels);
}

}  // namespace reboot

#endif	// REBOOT_DATA_H_
