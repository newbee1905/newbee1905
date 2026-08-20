// SPDX-License-Identifier: GPL-3.0-or-later
//
// Backend interface plus the plaintext slot backend.
//
// Every ReBoot primitive in this port (packing, layers, optimiser, training
// loop) is written once against a backend concept, so the identical code path
// runs either on CKKS ciphertexts under KeyMemRT (`CkksBackend`) or on plain
// slot vectors (`PlainBackend`).  The plaintext backend is not a second
// implementation of the algorithm: it is the same algorithm with the crypto
// removed, which makes it usable as
//
//   * a correctness oracle for the packing / rotation algebra,
//   * a fast planner that reproduces the level schedule (and hence the rotation
//     key depths) without paying for encryption,
//   * a way to run the whole benchmark on a machine without OpenFHE.
//
// A backend provides:
//
//   using Ct;                                 // ciphertext-like value
//   using Pt;                                 // encoded constant
//   int   slots() const;
//   int   level(const Ct&) const;
//   Ct    add(const Ct&, const Ct&);
//   Ct    sub(const Ct&, const Ct&);
//   Ct    mul(const Ct&, const Ct&);          // consumes one level
//   Ct    mulScalar(const Ct&, double);       // consumes one level
//   Ct    addScalar(const Ct&, double);
//   Ct    square(const Ct&);                  // consumes one level
//   Ct    mulMask(const Ct&, const Mask&);    // consumes one level
//   Ct    rotate(const Ct&, int index);       // left rotation, negative = right
//   Ct    bootstrap(const Ct&);
//   Ct    encryptSlots(const std::vector<double>&);
//   std::vector<double> decryptSlots(const Ct&) const;

#ifndef REBOOT_BACKEND_HPP_
#define REBOOT_BACKEND_HPP_

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "reboot/KeyPlan.hpp"

namespace reboot {

// A named constant slot vector.  Backends cache the encoding of a mask per
// level, keyed by the mask's address, so masks must outlive the backend use.
struct Mask {
  std::string name;
  std::vector<double> values;
};

// ---------------------------------------------------------------------------
// Plaintext slot backend
// ---------------------------------------------------------------------------

// Mirrors CKKS level accounting: a multiplication (by a ciphertext, a plaintext
// or a scalar) consumes one level, additions take the maximum of their inputs,
// rotations are level-preserving and bootstrapping resets to `bootstrapLevel`.
struct PlainValue {
  std::vector<double> slots;
  int level = 0;
};

class PlainBackend {
 public:
  using Ct = PlainValue;
  using Pt = std::vector<double>;

  explicit PlainBackend(int numSlots, int bootstrapLevel = 0)
      : slots_(numSlots), bootstrapLevel_(bootstrapLevel) {}

  int slots() const { return slots_; }
  int level(const Ct &c) const { return c.level; }

  void setKeyPlan(KeyPlan *plan) { plan_ = plan; }
  KeyPlan *keyPlan() const { return plan_; }

  Ct encryptSlots(const std::vector<double> &v) {
    Ct c;
    c.slots = v;
    c.slots.resize(static_cast<size_t>(slots_), 0.0);
    c.level = 0;
    return c;
  }

  std::vector<double> decryptSlots(const Ct &c) const { return c.slots; }

  Ct add(const Ct &a, const Ct &b) const { return zip(a, b, std::plus<double>()); }
  Ct sub(const Ct &a, const Ct &b) const { return zip(a, b, std::minus<double>()); }

  Ct mul(const Ct &a, const Ct &b) const {
    Ct out = zip(a, b, std::multiplies<double>());
    out.level = std::max(a.level, b.level) + 1;
    return out;
  }

  Ct mulScalar(const Ct &a, double s) const {
    Ct out = a;
    for (auto &v : out.slots) v *= s;
    out.level = a.level + 1;
    return out;
  }

  Ct addScalar(const Ct &a, double s) const {
    Ct out = a;
    for (auto &v : out.slots) v += s;
    return out;
  }

  Ct square(const Ct &a) const { return mul(a, a); }

  Ct mulMask(const Ct &a, const Mask &m) const {
    Ct out = a;
    for (size_t i = 0; i < out.slots.size(); ++i) out.slots[i] *= m.values[i];
    out.level = a.level + 1;
    return out;
  }

  // Left rotation by `index` slots; negative indices rotate right.  The
  // rotation is recorded in the key plan exactly as the CKKS backend does, so
  // the plan produced here matches the one the encrypted run needs.
  Ct rotate(const Ct &a, int index) {
    if (plan_) plan_->record(index, a.level);
    Ct out = a;
    const int n = static_cast<int>(a.slots.size());
    if (n == 0) return out;
    int shift = ((index % n) + n) % n;
    for (int i = 0; i < n; ++i)
      out.slots[static_cast<size_t>(i)] =
          a.slots[static_cast<size_t>((i + shift) % n)];
    return out;
  }

  Ct bootstrap(const Ct &a) const {
    Ct out = a;
    out.level = bootstrapLevel_;
    ++numBootstraps_;
    return out;
  }

  long numBootstraps() const { return numBootstraps_; }

 private:
  template <class Op>
  Ct zip(const Ct &a, const Ct &b, Op op) const {
    if (a.slots.size() != b.slots.size())
      throw std::invalid_argument("slot count mismatch");
    Ct out;
    out.slots.resize(a.slots.size());
    for (size_t i = 0; i < a.slots.size(); ++i)
      out.slots[i] = op(a.slots[i], b.slots[i]);
    out.level = std::max(a.level, b.level);
    return out;
  }

  int slots_;
  int bootstrapLevel_;
  KeyPlan *plan_ = nullptr;
  mutable long numBootstraps_ = 0;
};

}  // namespace reboot

#endif  // REBOOT_BACKEND_HPP_
