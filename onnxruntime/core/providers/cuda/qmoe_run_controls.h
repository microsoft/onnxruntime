// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Per-run QMoE offload controls.
//
// An application steers the prefill/decode hand-off by passing run config entries to Run(). The values live on the
// session (QMoESessionController::RunControls) instead of in the process environment, so a session never observes or
// consumes another session's signals, and the library never calls setenv while other threads read the environment.
//
// A Run() that replays a captured CUDA graph skips the execution provider's OnRunStart and therefore cannot carry
// these entries.
#pragma once

#include <atomic>
#include <charconv>
#include <string>
#include <system_error>
#include <utility>

namespace onnxruntime::cuda {

// "1" marks the Run as a warm-up Run: it does not feed expert scores or promotions. Applies to that Run only.
inline constexpr const char* kQMoERunInWarmupConfig = "ep.cuda.qmoe_in_warmup";
// "1" marks the Run as the first of a new conversation turn: prefill state left over from the previous turn is
// finalized and reset once. Applies to that Run only.
inline constexpr const char* kQMoERunPrefillNewTurnConfig = "ep.cuda.qmoe_prefill_new_turn";
// Total prompt tokens of the turn being prefilled. Persists on the session until a later Run supplies a new value.
inline constexpr const char* kQMoERunPrefillTotalTokensConfig = "ep.cuda.qmoe_prefill_total_tokens";
// "1" marks the Run as the last prefill Run of a turn. Once it completes successfully, the hand-off to decode is
// pending until the next prefill-to-decode finalization consumes it.
inline constexpr const char* kQMoERunPrefillHandoffDoneConfig = "ep.cuda.qmoe_prefill_handoff_done";

class QMoERunControls {
 public:
  // Applies the controls carried by one Run(). lookup(key, value) returns true and fills `value` when the Run supplies
  // `key`. If an entry is malformed nothing is applied, false is returned and `error` names the entry.
  template <typename LookupFn>
  bool Apply(LookupFn&& lookup, std::string& error) {
    bool in_warmup = false;
    bool new_turn = false;
    bool handoff_done = false;
    int total_tokens = -1;
    std::string value;

    auto read_flag = [&](const char* key, bool& out) {
      value.clear();
      return !lookup(key, value) || ParseFlag(key, value, out, error);
    };
    if (!read_flag(kQMoERunInWarmupConfig, in_warmup) ||
        !read_flag(kQMoERunPrefillNewTurnConfig, new_turn) ||
        !read_flag(kQMoERunPrefillHandoffDoneConfig, handoff_done)) {
      return false;
    }
    value.clear();
    if (lookup(kQMoERunPrefillTotalTokensConfig, value) &&
        !ParseNonNegativeInt(kQMoERunPrefillTotalTokensConfig, value, total_tokens, error)) {
      return false;
    }

    in_warmup_.store(in_warmup, std::memory_order_release);
    new_turn_.store(new_turn, std::memory_order_release);
    handoff_requested_.store(handoff_done, std::memory_order_release);
    if (total_tokens >= 0) {
      prefill_total_tokens_.store(total_tokens, std::memory_order_release);
    }
    return true;
  }

  // Called when the Run ends. The Run-scoped controls lapse, and a hand-off request becomes pending only when the Run
  // completed successfully.
  void EndRun(bool run_completed) noexcept {
    in_warmup_.store(false, std::memory_order_release);
    new_turn_.store(false, std::memory_order_release);
    if (handoff_requested_.exchange(false, std::memory_order_acq_rel) && run_completed) {
      handoff_done_.store(true, std::memory_order_release);
    }
  }

  bool InWarmup() const noexcept {
    return in_warmup_.load(std::memory_order_acquire);
  }

  // Returns true once per request, so exactly one consumer acts on a new turn.
  bool ConsumePrefillNewTurn() noexcept {
    return new_turn_.exchange(false, std::memory_order_acq_rel);
  }

  int PrefillTotalTokens(int default_value) const noexcept {
    const int value = prefill_total_tokens_.load(std::memory_order_acquire);
    return value >= 0 ? value : default_value;
  }

  bool HandoffDonePending() const noexcept {
    return handoff_done_.load(std::memory_order_acquire);
  }

  // Returns true once per hand-off.
  bool ConsumeHandoffDone() noexcept {
    return handoff_done_.exchange(false, std::memory_order_acq_rel);
  }

 private:
  static bool ParseFlag(const char* key, const std::string& value, bool& out, std::string& error) {
    if (value == "0" || value == "1") {
      out = (value == "1");
      return true;
    }
    error = std::string("Run config entry '") + key + "' must be 0 or 1, got '" + value + "'.";
    return false;
  }

  static bool ParseNonNegativeInt(const char* key, const std::string& value, int& out, std::string& error) {
    const char* const first = value.data();
    const char* const last = first + value.size();
    int parsed = 0;
    const auto result = std::from_chars(first, last, parsed);
    if (result.ec != std::errc() || result.ptr != last || parsed < 0) {
      error = std::string("Run config entry '") + key + "' must be a non-negative 32-bit integer, got '" + value + "'.";
      return false;
    }
    out = parsed;
    return true;
  }

  std::atomic<bool> in_warmup_{false};
  std::atomic<bool> new_turn_{false};
  std::atomic<bool> handoff_requested_{false};
  std::atomic<bool> handoff_done_{false};
  std::atomic<int> prefill_total_tokens_{-1};
};

}  // namespace onnxruntime::cuda
