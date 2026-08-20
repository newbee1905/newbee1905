// SPDX-License-Identifier: GPL-3.0-or-later
//
// Rotation-key plan.
//
// KeyMemRT-Compiler decides statically, for every `openfhe.rot` in the program,
// which rotation key to page in and at which compression level (`kmrt.load_key
// %idx {key_depth = L}`).  A hand-written training loop has no such static IR,
// so this file recovers the same information dynamically: a calibration run
// records every (rotation index, ciphertext level) pair the training step asks
// for, in order.  The recorded plan then drives
//
//   * provisioning  - which keys to generate, compress and serialize per level,
//   * prefetching   - what to enqueue ahead of the current rotation in
//                     KeyMemMode::PREFETCH,
//   * reporting     - the working set the runtime has to hold.
//
// The plan is deterministic: the training graph is identical from one step to
// the next, so a two-step calibration covers both the cold first step and the
// steady state that follows the first weight bootstrap.

#ifndef REBOOT_KEYPLAN_HPP_
#define REBOOT_KEYPLAN_HPP_

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace reboot {

struct KeyRequest {
  int index = 0;  // rotation index (negative rotates right)
  int level = 0;  // level of the ciphertext being rotated == key compression depth

  bool operator<(const KeyRequest &o) const {
    return index != o.index ? index < o.index : level < o.level;
  }
  bool operator==(const KeyRequest &o) const {
    return index == o.index && level == o.level;
  }
};

class KeyPlan {
 public:
  // ---- recording -----------------------------------------------------------
  void record(int index, int level) {
    if (index == 0) return;  // a no-op rotation needs no key
    trace_.push_back(KeyRequest{index, level});
    distinct_.insert(KeyRequest{index, level});
    indices_.insert(index);
    maxLevel_ = std::max(maxLevel_, level);
  }

  void clearTrace() { trace_.clear(); }

  // ---- queries -------------------------------------------------------------
  const std::vector<KeyRequest> &trace() const { return trace_; }
  const std::set<KeyRequest> &distinct() const { return distinct_; }
  int maxLevel() const { return maxLevel_; }

  std::vector<int32_t> indices() const {
    return std::vector<int32_t>(indices_.begin(), indices_.end());
  }

  // Rotation indices that are used at a given ciphertext level.
  std::vector<int32_t> indicesAtLevel(int level) const {
    std::vector<int32_t> out;
    for (const auto &r : distinct_)
      if (r.level == level) out.push_back(r.index);
    return out;
  }

  std::vector<int> levels() const {
    std::set<int> ls;
    for (const auto &r : distinct_) ls.insert(r.level);
    return std::vector<int>(ls.begin(), ls.end());
  }

  size_t numRotations() const { return trace_.size(); }

  std::string summary() const {
    std::ostringstream os;
    os << "KeyPlan: " << trace_.size() << " rotations, " << indices_.size()
       << " distinct indices, " << distinct_.size()
       << " (index, level) pairs, max level " << maxLevel_ << "\n";
    for (int level : levels()) {
      auto idx = indicesAtLevel(level);
      os << "  level " << level << ": " << idx.size() << " keys [";
      for (size_t i = 0; i < idx.size() && i < 12; ++i)
        os << (i ? ", " : "") << idx[i];
      if (idx.size() > 12) os << ", ...";
      os << "]\n";
    }
    return os.str();
  }

  // ---- persistence ---------------------------------------------------------
  // The plan is written next to the serialized keys so that the provisioning
  // step and the training step can run as separate processes (client / server).
  bool save(const std::string &path) const {
    std::ofstream f(path);
    if (!f) return false;
    f << "# reboot key plan: <rotation index> <key depth>\n";
    for (const auto &r : distinct_) f << r.index << " " << r.level << "\n";
    return true;
  }

  bool load(const std::string &path) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ls(line);
      int index = 0, level = 0;
      if (!(ls >> index >> level)) continue;
      distinct_.insert(KeyRequest{index, level});
      indices_.insert(index);
      maxLevel_ = std::max(maxLevel_, level);
    }
    return true;
  }

 private:
  std::vector<KeyRequest> trace_;
  std::set<KeyRequest> distinct_;
  std::set<int> indices_;
  int maxLevel_ = 0;
};

// Replays a recorded trace alongside the live execution so that PREFETCH mode
// can enqueue the keys the step is about to need.  `lookahead` bounds how far
// ahead of the cursor keys are requested; KeyMemRT's own tower budget
// (--prefetch-sat) still decides when the loads actually happen.
class PrefetchCursor {
 public:
  PrefetchCursor(const KeyPlan *plan, int lookahead)
      : plan_(plan), lookahead_(lookahead) {}

  bool active() const { return plan_ != nullptr && lookahead_ > 0; }

  // Advance the cursor past the request that is about to be executed and return
  // the requests that should be enqueued now.
  std::vector<KeyRequest> advance(int index, int level) {
    std::vector<KeyRequest> toEnqueue;
    if (!active()) return toEnqueue;
    const auto &trace = plan_->trace();
    if (trace.empty()) return toEnqueue;

    // Re-synchronise if the live execution diverged from the recorded trace
    // (for example because the first step differs from the steady state).
    if (pos_ >= trace.size() || !(trace[pos_] == KeyRequest{index, level})) {
      auto it = std::find(trace.begin() + std::min(pos_, trace.size()),
                          trace.end(), KeyRequest{index, level});
      if (it == trace.end())
        it = std::find(trace.begin(), trace.end(), KeyRequest{index, level});
      if (it == trace.end()) return toEnqueue;  // unknown request: no prefetch
      pos_ = static_cast<size_t>(it - trace.begin());
    }

    for (size_t k = pos_ + 1;
         k < trace.size() && k <= pos_ + static_cast<size_t>(lookahead_); ++k)
      toEnqueue.push_back(trace[k]);
    ++pos_;
    return toEnqueue;
  }

  void reset() { pos_ = 0; }

 private:
  const KeyPlan *plan_ = nullptr;
  int lookahead_ = 0;
  size_t pos_ = 0;
};

}  // namespace reboot

#endif  // REBOOT_KEYPLAN_HPP_
