// SPDX-License-Identifier: GPL-3.0-or-later

#include "reboot/ckks_params.h"

#include <fmt/format.h>

#include <stdexcept>

namespace reboot {
namespace {

uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t m) {
	return static_cast<uint64_t>((static_cast<__uint128_t>(a) * b) % m);
}

uint64_t pow_mod(uint64_t base, uint64_t exp, uint64_t m) {
	uint64_t result = 1;
	base %= m;
	while (exp > 0) {
		if (exp & 1) result = mul_mod(result, base, m);
		base = mul_mod(base, base, m);
		exp >>= 1;
	}
	return result;
}

// Deterministic Miller-Rabin; the listed bases are exact for all 64-bit inputs.
bool is_prime(uint64_t n) {
	if (n < 2) return false;
	for (uint64_t small : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull,
						   23ull, 29ull, 31ull, 37ull}) {
		if (n % small == 0) return n == small;
	}
	uint64_t d = n - 1;
	int r = 0;
	while ((d & 1) == 0) {
		d >>= 1;
		++r;
	}
	for (uint64_t a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull,
					   23ull, 29ull, 31ull, 37ull}) {
		uint64_t x = pow_mod(a, d, n);
		if (x == 1 || x == n - 1) continue;
		bool composite = true;
		for (int i = 1; i < r; ++i) {
			x = mul_mod(x, x, n);
			if (x == n - 1) {
				composite = false;
				break;
			}
		}
		if (composite) return false;
	}
	return true;
}

// Largest prime below 2^bits that is congruent to 1 modulo `modulus` and not
// already in `used`.  Walking downwards keeps the primes close to the requested
// width, which is what the scaling factor assumes.
uint64_t find_prime(int bits, uint64_t modulus,
					const std::vector<uint64_t> &used) {
	if (bits < 10 || bits > 62)
		throw std::invalid_argument(
			fmt::format("modulus width {} bits is out of range", bits));
	const uint64_t limit = 1ull << bits;
	uint64_t candidate = ((limit - 1) / modulus) * modulus + 1;
	if (candidate >= limit) candidate -= modulus;
	while (candidate > modulus) {
		bool taken = false;
		for (uint64_t u : used) taken = taken || u == candidate;
		if (!taken && is_prime(candidate)) return candidate;
		candidate -= modulus;
	}
	throw std::runtime_error(
		fmt::format("no {}-bit prime congruent to 1 mod {}", bits, modulus));
}

}  // namespace

modulus_chain_t generate_modulus_chain(const ckks_params_t &params) {
	const uint64_t two_n = 2ull * static_cast<uint64_t>(params.ring_dim());
	modulus_chain_t chain;
	std::vector<uint64_t> used;

	chain.q.push_back(find_prime(params.log_first_mod, two_n, used));
	used.push_back(chain.q.back());
	for (int i = 0; i < params.levels; ++i) {
		chain.q.push_back(find_prime(params.log_scale, two_n, used));
		used.push_back(chain.q.back());
	}
	chain.p.push_back(find_prime(params.log_p, two_n, used));
	return chain;
}

std::string scheme_param_attr(const ckks_params_t &params,
							  const modulus_chain_t &chain) {
	std::string q;
	for (size_t i = 0; i < chain.q.size(); ++i)
		q += fmt::format("{}{}", i ? ", " : "", chain.q[i]);
	std::string p;
	for (size_t i = 0; i < chain.p.size(); ++i)
		p += fmt::format("{}{}", i ? ", " : "", chain.p[i]);
	return fmt::format(
		"#ckks.scheme_param<logN = {}, Q = [{}], P = [{}], logDefaultScale = "
		"{}>",
		params.log_n, q, p, params.log_scale);
}

}  // namespace reboot
