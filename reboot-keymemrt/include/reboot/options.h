// SPDX-License-Identifier: GPL-3.0-or-later
//
// A declarative command line, shared by every binary here.
//
// Each option is registered once, bound to the variable it sets, with its help
// text attached; the parser and `--help` both come from that table.  This
// replaces four hand-written if/else chains that between them re-implemented
// the same flags, skipped KeyMemRT's flags by hardcoded name, and read
// `argv[++i]` without checking whether an argument was actually there.
//
// Supports `--name value` and `--name=value`, reports a missing value or a
// malformed number against the flag that caused it, and rejects unknown flags
// instead of ignoring them.

#ifndef REBOOT_OPTIONS_H_
#define REBOOT_OPTIONS_H_

#include <functional>
#include <string>
#include <vector>

#include "reboot/reboot_model.h"

namespace reboot {

class option_parser_t {
   public:
	option_parser_t(std::string program, std::string summary)
		: program_(std::move(program)), summary_(std::move(summary)) {}

	// Value options, bound by reference.  The referenced variable must outlive
	// the parse, which is the usual case: locals in main().
	option_parser_t &add(const std::string &name, int &target,
						 const std::string &help);
	option_parser_t &add(const std::string &name, double &target,
						 const std::string &help);
	option_parser_t &add(const std::string &name, std::string &target,
						 const std::string &help);
	// Comma-separated list, as in `--hidden 64,32`.
	option_parser_t &add(const std::string &name, std::vector<int> &target,
						 const std::string &help);

	// Valueless switch: presence sets `target` to `value`.
	option_parser_t &add_switch(const std::string &name, bool &target,
								bool value, const std::string &help);

	// A flag another parser owns - KeyMemRT reads its own from argv - so that
	// it is skipped rather than reported as unknown.
	option_parser_t &ignore(const std::string &name, bool takes_value);

	// Blank line plus a heading in the generated help.
	option_parser_t &section(const std::string &title);

	// Returns false when --help was given, so main() can exit cleanly.
	// Throws std::invalid_argument with a usable message on anything wrong.
	bool parse(int argc, char **argv);

	std::string help() const;

   private:
	struct option_t {
		std::string name;
		std::string help;
		std::string section;
		bool takes_value = true;
		bool hidden = false;  // ignored flags do not appear in --help
		std::function<void(const std::string &)> apply;
	};

	option_parser_t &push(option_t option);
	const option_t *find(const std::string &name) const;

	std::string program_;
	std::string summary_;
	std::string current_section_;
	std::vector<option_t> options_;
	std::vector<std::string> section_order_;
};

// The flags that describe a network.  Registered from one place so the emitter,
// the plaintext evaluator and the measurement tool cannot drift apart.
void add_model_options(option_parser_t &parser, model_config_t &config,
					   int &log_n);

// The flags KeyMemRT's own parser consumes.
void ignore_keymemrt_options(option_parser_t &parser);

}  // namespace reboot

#endif	// REBOOT_OPTIONS_H_
