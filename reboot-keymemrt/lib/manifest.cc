// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/manifest.h"

#include <fmt/format.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace reboot {
namespace {

constexpr int kManifestVersion = 1;

std::string join(const std::vector<int> &values) {
	std::string out;
	for (size_t i = 0; i < values.size(); ++i)
		out += fmt::format("{}{}", i ? "," : "", values[i]);
	return out;
}

std::vector<int> split(const std::string &text) {
	std::vector<int> values;
	size_t pos = 0;
	while (pos <= text.size() && !text.empty()) {
		size_t comma = text.find(',', pos);
		if (comma == std::string::npos) comma = text.size();
		values.push_back(std::stoi(text.substr(pos, comma - pos)));
		if (comma == text.size()) break;
		pos = comma + 1;
	}
	return values;
}

}  // namespace

Manifest make_manifest(const std::string &function_name,
					   const ModelConfig &config, int log_n,
					   const Layout &layout, const LoweredStep &step) {
	Manifest manifest;
	manifest.function_name = function_name;
	manifest.config = config;
	manifest.log_n = log_n;
	manifest.layout = layout;
	manifest.argument_names = step.argument_names;
	manifest.result_names = step.result_names;
	return manifest;
}

void Manifest::save(const std::string &path) const {
	std::ofstream file(path);
	if (!file)
		throw std::runtime_error(fmt::format("cannot write manifest {}", path));

	file << fmt::format("reboot_manifest {}\n", kManifestVersion);
	file << fmt::format("function {}\n", function_name);
	file << fmt::format("log_n {}\n", log_n);
	file << fmt::format("layout {} {}\n", layout.rows, layout.cols);
	file << fmt::format("input_dim {}\n", config.input_dim);
	file << fmt::format("hidden {}\n", join(config.hidden));
	file << fmt::format("classes {}\n", config.num_classes);
	file << fmt::format("batch_size {}\n", config.batch_size);
	file << fmt::format("learning_rate {}\n", config.learning_rate);
	file << fmt::format("momentum {}\n", config.momentum);
	file << fmt::format("weight_decay {}\n", config.weight_decay);
	file << fmt::format("bootstrap {}\n", config.bootstrap ? 1 : 0);
	for (const std::string &name : argument_names)
		file << fmt::format("argument {}\n", name);
	for (const std::string &name : result_names)
		file << fmt::format("result {}\n", name);
}

Manifest Manifest::load(const std::string &path) {
	std::ifstream file(path);
	if (!file)
		throw std::runtime_error(fmt::format("cannot read manifest {}", path));

	Manifest manifest;
	bool seen_header = false;
	int rows = 0, cols = 0;
	std::string line;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') continue;
		std::istringstream fields(line);
		std::string key;
		fields >> key;

		if (key == "reboot_manifest") {
			int version = 0;
			fields >> version;
			if (version != kManifestVersion)
				throw std::runtime_error(
					fmt::format("manifest {} is version {}, expected {}", path,
								version, kManifestVersion));
			seen_header = true;
		} else if (key == "function") {
			fields >> manifest.function_name;
		} else if (key == "log_n") {
			fields >> manifest.log_n;
		} else if (key == "layout") {
			fields >> rows >> cols;
		} else if (key == "input_dim") {
			fields >> manifest.config.input_dim;
		} else if (key == "hidden") {
			std::string list;
			fields >> list;
			manifest.config.hidden = split(list);
		} else if (key == "classes") {
			fields >> manifest.config.num_classes;
		} else if (key == "batch_size") {
			fields >> manifest.config.batch_size;
		} else if (key == "learning_rate") {
			fields >> manifest.config.learning_rate;
		} else if (key == "momentum") {
			fields >> manifest.config.momentum;
		} else if (key == "weight_decay") {
			fields >> manifest.config.weight_decay;
		} else if (key == "bootstrap") {
			int value = 1;
			fields >> value;
			manifest.config.bootstrap = value != 0;
		} else if (key == "argument") {
			std::string name;
			fields >> name;
			manifest.argument_names.push_back(name);
		} else if (key == "result") {
			std::string name;
			fields >> name;
			manifest.result_names.push_back(name);
		} else {
			throw std::runtime_error(
				fmt::format("manifest {}: unknown key '{}'", path, key));
		}
	}

	if (!seen_header)
		throw std::runtime_error(
			fmt::format("{} is not a reboot manifest", path));
	manifest.layout = Layout(rows, cols);
	return manifest;
}

void Manifest::verify(const LoweredStep &step) const {
	auto mismatch = [&](const std::string &what, const std::string &expected,
						const std::string &got) {
		throw std::runtime_error(fmt::format(
			"the model does not match the manifest: {} is '{}' but the module "
			"was emitted with '{}'.  Re-emit the module, or point --manifest "
			"at "
			"the one written beside it.",
			what, got, expected));
	};

	if (step.argument_names.size() != argument_names.size())
		mismatch("argument count", std::to_string(argument_names.size()),
				 std::to_string(step.argument_names.size()));
	for (size_t i = 0; i < argument_names.size(); ++i)
		if (step.argument_names[i] != argument_names[i])
			mismatch(fmt::format("argument {}", i), argument_names[i],
					 step.argument_names[i]);

	if (step.result_names.size() != result_names.size())
		mismatch("result count", std::to_string(result_names.size()),
				 std::to_string(step.result_names.size()));
	for (size_t i = 0; i < result_names.size(); ++i)
		if (step.result_names[i] != result_names[i])
			mismatch(fmt::format("result {}", i), result_names[i],
					 step.result_names[i]);
}

std::string Manifest::describe() const {
	return fmt::format(
		"manifest: {} over layout {}, ring 2^{}\n"
		"  {} arguments, {} results\n",
		function_name, layout.str(), log_n, argument_names.size(),
		result_names.size());
}

}  // namespace reboot
