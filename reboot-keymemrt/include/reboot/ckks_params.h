// SPDX-License-Identifier: GPL-3.0-or-later
//
// CKKS scheme parameters for the emitted module.
//
// The KeyMemRT pipeline starts from `ckks`-dialect IR whose module carries a
// `ckks.schemeParam` attribute and whose ciphertext types spell out the RNS
// modulus chain.  Both need concrete NTT-friendly primes, so this header
// generates them: a chain of distinct primes q = 1 (mod 2N) of roughly the
// requested bit width, plus one larger first modulus and one special modulus P
// for hybrid key switching.

#ifndef REBOOT_CKKS_PARAMS_H_
#define REBOOT_CKKS_PARAMS_H_

#include <cstdint>
#include <string>
#include <vector>

namespace reboot {

struct CkksParams {
	// Ring dimension is 2^log_n; a fully packed ciphertext holds 2^(log_n-1)
	// slots.
	int log_n = 13;
	// Number of levels the training step consumes, i.e. how many primes the
	// chain needs beyond the first modulus.
	int levels = 5;
	int log_scale = 26;		 // log2 of the default scaling factor
	int log_first_mod = 30;	 // log2 of the first (largest) modulus
	int log_p = 30;			 // log2 of the special modulus for key switching

	int ring_dim() const { return 1 << log_n; }
	int num_slots() const { return 1 << (log_n - 1); }
};

// A generated modulus chain: q[0] is the first modulus, q[1..] the rescaling
// primes, and p the special modulus.
struct ModulusChain {
	std::vector<uint64_t> q;
	std::vector<uint64_t> p;
};

// Generate `params.levels + 1` distinct primes congruent to 1 mod 2N, the first
// of width log_first_mod and the rest of width log_scale, plus one special
// prime of width log_p.  Throws when no prime of the requested width exists.
ModulusChain generate_modulus_chain(const CkksParams &params);

// `#ckks.scheme_param<...>` attribute text for the emitted module.
std::string scheme_param_attr(const CkksParams &params,
							  const ModulusChain &chain);

}  // namespace reboot

#endif	// REBOOT_CKKS_PARAMS_H_
