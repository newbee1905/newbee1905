// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/options.h"

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>

namespace reboot {
namespace {

[[noreturn]] void fail(const std::string &message) {
	throw std::invalid_argument(message);
}

int to_int(const std::string &name, const std::string &text) {
	try {
		size_t consumed = 0;
		const int value = std::stoi(text, &consumed);
		if (consumed != text.size()) throw std::invalid_argument("trailing");
		return value;
	} catch (const std::exception &) {
		fail(fmt::format("{} expects an integer, got '{}'", name, text));
	}
}

double to_double(const std::string &name, const std::string &text) {
	try {
		size_t consumed = 0;
		const double value = std::stod(text, &consumed);
		if (consumed != text.size()) throw std::invalid_argument("trailing");
		return value;
	} catch (const std::exception &) {
		fail(fmt::format("{} expects a number, got '{}'", name, text));
	}
}

std::vector<int> to_int_list(const std::string &name, const std::string &text) {
	if (text.empty()) fail(fmt::format("{} expects a non-empty list", name));
	std::vector<int> values;
	size_t pos = 0;
	while (pos <= text.size()) {
		size_t comma = text.find(',', pos);
		if (comma == std::string::npos) comma = text.size();
		values.push_back(to_int(name, text.substr(pos, comma - pos)));
		if (comma == text.size()) break;
		pos = comma + 1;
	}
	return values;
}

}  // namespace

option_parser_t &option_parser_t::push(option_t option) {
	option.section = current_section_;
	if (!option.section.empty() &&
		std::find(section_order_.begin(), section_order_.end(),
				  option.section) == section_order_.end())
		section_order_.push_back(option.section);
	options_.push_back(std::move(option));
	return *this;
}

option_parser_t &option_parser_t::section(const std::string &title) {
	current_section_ = title;
	return *this;
}

option_parser_t &option_parser_t::add(const std::string &name, int &target,
									  const std::string &help) {
	return push(
		{name, help, "", true, false,
		 [&target, name](const std::string &v) { target = to_int(name, v); }});
}

option_parser_t &option_parser_t::add(const std::string &name, double &target,
									  const std::string &help) {
	return push(
		{name, help, "", true, false, [&target, name](const std::string &v) {
			 target = to_double(name, v);
		 }});
}

option_parser_t &option_parser_t::add(const std::string &name,
									  std::string &target,
									  const std::string &help) {
	return push({name, help, "", true, false,
				 [&target](const std::string &v) { target = v; }});
}

option_parser_t &option_parser_t::add(const std::string &name,
									  std::vector<int> &target,
									  const std::string &help) {
	return push(
		{name, help, "", true, false, [&target, name](const std::string &v) {
			 target = to_int_list(name, v);
		 }});
}

option_parser_t &option_parser_t::add_switch(const std::string &name,
											 bool &target, bool value,
											 const std::string &help) {
	return push({name, help, "", false, false,
				 [&target, value](const std::string &) { target = value; }});
}

option_parser_t &option_parser_t::ignore(const std::string &name,
										 bool takes_value) {
	return push({name, "", "", takes_value, true, nullptr});
}

const option_parser_t::option_t *option_parser_t::find(
	const std::string &name) const {
	for (const option_t &option : options_)
		if (option.name == name) return &option;
	return nullptr;
}

bool option_parser_t::parse(int argc, char **argv) {
	for (int i = 1; i < argc; ++i) {
		std::string argument = argv[i];
		if (argument == "--help" || argument == "-h") {
			fmt::print("{}", help());
			return false;
		}

		// `--name=value` carries its own value; `--name value` takes the next
		// argument, which has to exist.
		std::string value;
		bool inline_value = false;
		const size_t equals = argument.find('=');
		if (equals != std::string::npos) {
			value = argument.substr(equals + 1);
			argument = argument.substr(0, equals);
			inline_value = true;
		}

		const option_t *option = find(argument);
		if (option == nullptr)
			fail(fmt::format("unknown option '{}'; try --help", argument));

		if (!option->takes_value) {
			if (inline_value)
				fail(fmt::format("{} does not take a value", argument));
		} else if (!inline_value) {
			if (i + 1 >= argc)
				fail(fmt::format("{} expects a value", argument));
			value = argv[++i];
		}

		if (option->apply) option->apply(value);
	}
	return true;
}

std::string option_parser_t::help() const {
	std::string out = fmt::format("{} - {}\n", program_, summary_);

	size_t width = 0;
	for (const option_t &option : options_)
		if (!option.hidden) width = std::max(width, option.name.size());

	std::vector<std::string> sections = section_order_;
	if (std::find(sections.begin(), sections.end(), std::string()) ==
		sections.end())
		sections.insert(sections.begin(), std::string());

	for (const std::string &section : sections) {
		bool printed_heading = false;
		for (const option_t &option : options_) {
			if (option.hidden || option.section != section) continue;
			if (!printed_heading) {
				out += section.empty() ? "\n" : fmt::format("\n{}:\n", section);
				printed_heading = true;
			}
			out +=
				fmt::format("  {:<{}}  {}\n", option.name, width, option.help);
		}
	}
	out +=
		fmt::format("\n  {:<{}}  {}\n", "--help", width, "show this message");
	return out;
}

void add_model_options(option_parser_t &parser, model_config_t &config,
					   int &log_n) {
	parser.section("Model")
		.add("--hidden", config.hidden, "hidden layer widths, comma separated")
		.add("--input-dim", config.input_dim, "input dimension")
		.add("--classes", config.num_classes, "number of classes")
		.add("--batch-size", config.batch_size, "samples per step")
		.add("--lr", config.learning_rate, "learning rate")
		.add("--momentum", config.momentum, "Nesterov momentum")
		.add("--weight-decay", config.weight_decay, "weight decay")
		.add_switch("--no-bootstrap", config.bootstrap, false,
					"leave the weights unrefreshed (inspection only)")
		.add("--log-n", log_n, "log2 of the ring dimension");
}

void ignore_keymemrt_options(option_parser_t &parser) {
	for (const char *name :
		 {"--key-mode", "--input-dir", "--output-dir", "--output-base",
		  "--prefetch-sat", "--log-level", "--log-file"})
		parser.ignore(name, /*takes_value=*/true);
	for (const char *name :
		 {"--verbose", "-v", "--ser-single-file", "--log-console-off"})
		parser.ignore(name, /*takes_value=*/false);
}

}  // namespace reboot
