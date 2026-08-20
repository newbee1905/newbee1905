// SPDX-License-Identifier: GPL-3.0-or-later
//
// The command line and the manifest are the two places a mistake is silent
// rather than loud: a flag that reads past the end of argv, or a runner that
// calls the generated module with the argument order of a different model.
// Both are checked here.

#include <fmt/format.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "reboot/manifest.h"
#include "reboot/options.h"
#include "reboot/reboot_model.h"
#include "reboot/slot_graph.h"

using namespace reboot;

namespace {

int failures = 0;

void check(bool condition, const std::string &what) {
	if (!condition) {
		fmt::print("  FAIL: {}\n", what);
		++failures;
	}
}

// parse() takes argc/argv, so tests hand it a real one.
bool run(OptionParser &parser, const std::vector<std::string> &arguments) {
	std::vector<std::string> storage{"program"};
	storage.insert(storage.end(), arguments.begin(), arguments.end());
	std::vector<char *> argv;
	for (std::string &argument : storage) argv.push_back(argument.data());
	return parser.parse(static_cast<int>(argv.size()), argv.data());
}

// Returns the error message, or "" when the parse succeeded.
std::string error_of(OptionParser &parser,
					 const std::vector<std::string> &arguments) {
	try {
		run(parser, arguments);
		return "";
	} catch (const std::exception &error) {
		return error.what();
	}
}

bool contains(const std::string &haystack, const std::string &needle) {
	return haystack.find(needle) != std::string::npos;
}

void test_values() {
	fmt::print("values, in both spellings\n");
	int count = 0;
	double rate = 0.0;
	std::string path;
	std::vector<int> widths;
	OptionParser parser("test", "values");
	parser.add("--count", count, "")
		.add("--rate", rate, "")
		.add("--path", path, "")
		.add("--widths", widths, "");

	check(run(parser, {"--count", "7", "--rate=0.25", "--path", "out.mlir",
					   "--widths", "64,32,16"}),
		  "parse succeeds");
	check(count == 7, "--count value");
	check(rate == 0.25, "--rate=value");
	check(path == "out.mlir", "--path value");
	check(widths == std::vector<int>({64, 32, 16}), "--widths list");

	std::vector<int> single;
	OptionParser one("test", "values");
	one.add("--widths", single, "");
	run(one, {"--widths=32"});
	check(single == std::vector<int>({32}), "single-element list");
}

void test_switches_and_ignored() {
	fmt::print("switches and borrowed flags\n");
	bool bootstrap = true;
	OptionParser parser("test", "switches");
	parser.add_switch("--no-bootstrap", bootstrap, false, "");
	ignore_keymemrt_options(parser);

	// KeyMemRT's own flags must be skipped, with and without their values,
	// without swallowing the flag that follows.
	check(run(parser, {"--key-mode", "imperative", "--verbose",
					   "--no-bootstrap", "--prefetch-sat", "250"}),
		  "parse succeeds alongside the runtime's flags");
	check(bootstrap == false, "switch applied");

	int steps = 0;
	OptionParser second("test", "switches");
	second.add("--steps", steps, "");
	ignore_keymemrt_options(second);
	run(second, {"--log-level", "error", "--steps", "9"});
	check(steps == 9, "a value flag after an ignored one still parses");
}

void test_errors() {
	fmt::print("errors name the flag that caused them\n");
	int count = 0;
	bool flag = false;
	OptionParser parser("test", "errors");
	parser.add("--count", count, "").add_switch("--flag", flag, true, "");

	// The old hand-rolled loops read argv[++i] here, past the end.
	const std::string missing = error_of(parser, {"--count"});
	check(
		contains(missing, "--count") && contains(missing, "expects a value"),
		"a trailing flag reports a missing value instead of reading past argv");

	const std::string bad = error_of(parser, {"--count", "twelve"});
	check(contains(bad, "--count") && contains(bad, "integer"),
		  "a malformed number names the flag");

	const std::string trailing = error_of(parser, {"--count", "12x"});
	check(contains(trailing, "--count"), "trailing characters are rejected");

	const std::string unknown = error_of(parser, {"--nope"});
	check(contains(unknown, "--nope"),
		  "unknown flags are rejected, not ignored");

	const std::string valued_switch = error_of(parser, {"--flag=1"});
	check(contains(valued_switch, "does not take a value"),
		  "a switch rejects a value");

	check(error_of(parser, {"--count", "12"}).empty(), "valid input parses");
}

void test_help() {
	fmt::print("help comes from the table\n");
	int count = 0;
	OptionParser parser("test", "a summary");
	parser.section("Model").add("--count", count, "how many");
	check(!run(parser, {"--help"}), "--help stops the parse");

	const std::string help = parser.help();
	check(contains(help, "a summary"), "summary appears");
	check(contains(help, "Model:"), "section heading appears");
	check(contains(help, "--count") && contains(help, "how many"),
		  "the option and its help appear");
	check(contains(help, "--help"), "--help documents itself");
}

ModelConfig sample_config() {
	ModelConfig config;
	config.input_dim = 6;
	config.hidden = {8, 4};
	config.num_classes = 3;
	config.batch_size = 2;
	config.learning_rate = 0.005;
	config.weight_decay = 0.01;
	return config;
}

void test_manifest() {
	fmt::print("manifest round trip and verification\n");

	const ModelConfig config = sample_config();
	const Layout layout = recommend_layout(config, 1024);
	const LoweredStep lowered =
		lower_to_slots(build_train_step(config, layout));
	const Manifest manifest =
		make_manifest("reboot_train_step", config, 11, layout, lowered);

	const std::string path = "test_options.manifest";
	manifest.save(path);
	const Manifest loaded = Manifest::load(path);

	check(loaded.function_name == "reboot_train_step",
		  "function name survives");
	check(loaded.log_n == 11, "ring dimension survives");
	check(
		loaded.layout.rows == layout.rows && loaded.layout.cols == layout.cols,
		"layout survives");
	check(loaded.config.hidden == config.hidden, "hidden widths survive");
	check(loaded.config.batch_size == config.batch_size, "batch size survives");
	check(loaded.config.weight_decay == config.weight_decay,
		  "weight decay survives");
	check(loaded.argument_names == lowered.argument_names,
		  "argument order survives");
	check(loaded.result_names == lowered.result_names, "result order survives");

	// The same model verifies.
	bool accepted = true;
	try {
		loaded.verify(lowered);
	} catch (const std::exception &) {
		accepted = false;
	}
	check(accepted, "a matching model verifies");

	// A different one must not: this is the failure the manifest exists to
	// catch, where the runner would otherwise call the module with the
	// argument order of another network.
	ModelConfig other = config;
	other.hidden = {8, 4, 4};
	const Layout other_layout = recommend_layout(other, 1024);
	const LoweredStep other_step =
		lower_to_slots(build_train_step(other, other_layout));
	std::string rejection;
	try {
		loaded.verify(other_step);
	} catch (const std::exception &error) {
		rejection = error.what();
	}
	check(!rejection.empty(), "a different model is rejected");
	check(contains(rejection, "manifest"), "the rejection explains itself");

	std::remove(path.c_str());
}

}  // namespace

int main() {
	try {
		test_values();
		test_switches_and_ignored();
		test_errors();
		test_help();
		test_manifest();
	} catch (const std::exception &error) {
		fmt::print("unexpected exception: {}\n", error.what());
		return 1;
	}

	if (failures == 0) {
		fmt::print("all option and manifest tests passed\n");
		return 0;
	}
	fmt::print("{} failure(s)\n", failures);
	return 1;
}
