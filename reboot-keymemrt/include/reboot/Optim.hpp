// SPDX-License-Identifier: GPL-3.0-or-later
//
// Encrypted Nesterov SGD and the cosine learning-rate schedule.
//
// Both the weights and the velocity stay encrypted for the whole run; the
// learning rate, momentum and weight decay are public scalars, so the update is
// a sequence of scalar multiplications and additions.  Neither weight decay nor
// momentum adds multiplicative depth (Section "Remark" of the ReBoot README):
// the two momentum terms are formed from the gradient at the same level and
// summed.
//
// This matches init_nesterov / compute_nesterov of the reference implementation
// exactly, including the deliberate reuse of the pre-decay gradient in the
// velocity recursion.

#ifndef REBOOT_OPTIM_HPP_
#define REBOOT_OPTIM_HPP_

#include <cmath>
#include <memory>

namespace reboot {

template <class B>
class NesterovSGD {
 public:
  using Ct = typename B::Ct;

  NesterovSGD(double momentum, double weightDecay)
      : momentum_(momentum), weightDecay_(weightDecay) {}

  bool hasVelocity() const { return static_cast<bool>(velocity_); }
  const Ct &velocity() const { return *velocity_; }
  void setVelocity(const Ct &v) { velocity_ = std::make_shared<Ct>(v); }

  // Returns the update to subtract from the weights.
  Ct computeUpdate(B &be, const Ct &gradient, const Ct &weights, double lr) {
    Ct g = gradient;
    if (weightDecay_ != 0.0)
      g = be.add(g, be.mulScalar(weights, weightDecay_));

    Ct update;
    if (!velocity_) {
      // First step: the velocity starts at the gradient itself.
      velocity_ = std::make_shared<Ct>(g);
      update = be.add(be.mulScalar(g, lr), be.mulScalar(g, momentum_ * lr));
    } else {
      update = be.add(be.add(be.mulScalar(g, lr), be.mulScalar(g, momentum_ * lr)),
                      be.mulScalar(*velocity_, momentum_ * momentum_ * lr));
      velocity_ = std::make_shared<Ct>(
          be.add(be.mulScalar(*velocity_, momentum_), g));
    }
    return update;
  }

 private:
  double momentum_;
  double weightDecay_;
  std::shared_ptr<Ct> velocity_;
};

// Cosine schedule with a warm-up plateau, as used by the ReBoot experiments.
class CosineLR {
 public:
  CosineLR(double baseLr, int stepsBeforeRestart, int fromStep)
      : baseLr_(baseLr), stepsBeforeRestart_(stepsBeforeRestart),
        fromStep_(fromStep) {}

  void step() { ++step_; }

  double lr() const {
    if (step_ < fromStep_) return baseLr_;
    const int t = (step_ - fromStep_) % stepsBeforeRestart_;
    return 0.5 * baseLr_ *
           (1.0 + std::cos(M_PI * static_cast<double>(t) / stepsBeforeRestart_));
  }

 private:
  double baseLr_;
  int stepsBeforeRestart_;
  int fromStep_;
  int step_ = 0;
};

}  // namespace reboot

#endif  // REBOOT_OPTIM_HPP_
