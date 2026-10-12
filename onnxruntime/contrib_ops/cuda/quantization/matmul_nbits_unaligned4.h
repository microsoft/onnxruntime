// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <atomic>
#include <cstdio>
#include <string>

#include "core/common/parse_string.h"
#include "core/common/string_utils.h"

namespace onnxruntime::contrib::cuda {

constexpr const char* kUnaligned4FastPathEnvVar = "ORT_MATMULNBITS_UNALIGNED4_FAST_PATH";

// The kernels support M <= 8, but for M in [5, 8] the register-batched tail kernel measured slower than the
// dequantize + GEMM fallback on H200 (N=324, K=10240: ~39-43 us vs ~26 us), so dispatch stops at M = 4.
constexpr int kUnaligned4MaxM = 4;

inline bool ParseUnaligned4FastPathEnabled(const std::string& value) {
  const auto trimmed = onnxruntime::utils::TrimString(value);
  int enabled = 0;
  ORT_ENFORCE(trimmed.empty() || TryParseStringWithClassicLocale(trimmed, enabled),
              "Invalid ", kUnaligned4FastPathEnvVar, ": expected 0 or 1.");
  ORT_ENFORCE(enabled == 0 || enabled == 1, "Invalid ", kUnaligned4FastPathEnvVar, ": expected 0 or 1.");
  return enabled == 1;
}

inline bool IsUnaligned4FastPathEligible(
    bool enabled, bool fp16, int m, int n, int k, int bits, int block_size,
    bool column_wise, bool zero_points, bool bias, bool g_idx, bool optimized,
    bool force_chunked) {
  return enabled && fp16 && m >= 1 && m <= kUnaligned4MaxM && n >= 4 && n % 8 == 4 &&
         k > 0 && k % 32 == 0 && bits == 4 && block_size == 32 && column_wise &&
         !zero_points && !bias && !g_idx && !optimized && !force_chunked;
}

// The explicitly enabled selector emits one actual successful dispatch receipt per kernel branch.
class Unaligned4DispatchLog {
 public:
  bool ShouldLog(bool m1) const {
    return !(m1 ? m1_logged_ : batched_logged_).load(std::memory_order_relaxed);
  }
  void Log(bool m1, const std::string& node, int m, int k, int n) const {
    auto& logged = m1 ? m1_logged_ : batched_logged_;
    bool expected = false;
    if (logged.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
      std::fprintf(stderr,
                   "[MatMulNBitsUnaligned4] node=\"%s\" M=%d K=%d N=%d type=FP16 bits=4 branch=%s\n",
                   node.c_str(), m, k, n, m1 ? "M1_SHARED_TAIL" : "BATCHED_REGISTER_TAIL");
    }
  }

 private:
  mutable std::atomic<bool> m1_logged_{false};
  mutable std::atomic<bool> batched_logged_{false};
};

}  // namespace onnxruntime::contrib::cuda
