/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include <initializer_list>

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

enum class KernelType {
  FP16Int8Groupwise,
  FP16Int4Groupwise,
  FP16Int2Groupwise,
  FP16Int8PerChannel,
  FP16Int4PerChannel,
  FP16Int2PerChannel,
  BF16Int8Groupwise,
  BF16Int4Groupwise,
  BF16Int2Groupwise,
  BF16Int8PerChannel,
  BF16Int4PerChannel,
  BF16Int2PerChannel
};

inline std::pair<int, int> GetInt4DecodeGeometry(int variant) {
  switch (variant) {
    case 2:
      return {2, 128};
    case 3:
      return {2, 256};
    case 4:
      return {4, 128};
    case 5:
      return {4, 256};
    case 6:
      return {8, 128};
    case 7:
      return {8, 256};
    default:
      return {0, 0};
  }
}

inline bool IsInt4DecodeGeometryLegal(int variant, int n, int k, int interleave) {
  const auto [tile, threads] = GetInt4DecodeGeometry(variant);
  return tile != 0 && threads != 0 && interleave == 4 && n >= 64 && n % 64 == 0 &&
         k >= 64 && k % 64 == 0 && n % (tile * interleave) == 0 && n / (tile * interleave) <= 65535;
}

// A device whose M = 8 tile windows were measured: SM 8.0 with 108 SMs (A100 40 GB / 80 GB, PCIe or SXM).
// The SM count separates it from the 56-SM A30, which is also SM 8.0.
inline bool IsA100Device(int device_sm, int device_sm_count) {
  return device_sm == 80 && device_sm_count == 108;
}

// SM 9.0 with 132 SMs: the H200 that the H200 M = 8 tile rule was measured on. H100 SXM, H800 and GH200
// report the same pair and therefore share the rule, but were not measured. The SM count separates the
// group from the 114-SM H100 PCIe and the 78-SM H20.
inline bool IsH200Device(int device_sm, int device_sm_count) {
  return device_sm == 90 && device_sm_count == 132;
}

// Devices with their own measured M = 8 tile rule; every other device keeps the device-independent cutoffs.
enum class M8GemvDevice {
  kUnmeasured,
  kA100,
  kH200,
};

inline M8GemvDevice GetM8GemvDevice(int device_sm, int device_sm_count) {
  if (IsA100Device(device_sm, device_sm_count)) {
    return M8GemvDevice::kA100;
  }
  return IsH200Device(device_sm, device_sm_count) ? M8GemvDevice::kH200 : M8GemvDevice::kUnmeasured;
}

// Bytes one INT4 launch streams: the packed weights plus one 16-bit scale per column and group.
inline int64_t Int4WeightAndScaleBytes(int n, int k, int group_size) {
  return static_cast<int64_t>(n) * k / 2 + static_cast<int64_t>(n) * (k / group_size) * 2;
}

// H200 INT4 rule (FP16 and BF16, group size 32 / 64 / 128). A narrow block (CtaN = 2) covers 8 columns and 5
// blocks are resident per SM (92 registers); a default block (CtaN = 4) covers 16 columns and 3 are resident
// (142 - 146 registers). The default tile therefore holds 6336 columns per wave and the narrow tile 5280
// (= 132 * 5 * 8).
//   * N <= 4224 (= 132 * 32): neither tile gives the busiest SM more than 32 columns, so mostly latency
//     differs. The narrow tile gains 2 - 5% on average while the weight stream is small and loses it (BF16
//     first) as the stream grows.
//   * 4224 < N <= 5280: the default tile leaves some SMs with 48 columns against 40 for the narrow tile, a
//     7 - 18% gain that fades as the stream grows and is gone at about 44 MB.
//   * N > 5280: the narrow tile needs a second wave and is 6 - 83% slower up to N = 6144. Further up (group size
//     32 only) it wins by at most 5% in scattered windows that the rule does not chase.
// A bias epilogue makes the narrow K loop 6 - 32% slower than the default one (the penalty grows with K). Only
// the wave gain of the 4224 < N <= 5280 window survives it, up to 34 MB.
// The fade follows the bytes of weights plus scales: group sizes 32, 64 and 128 collapse onto one curve in
// bytes but not in weight elements. The caps stay below the measured crossover; in the rotating-weights
// sweeps the shapes they select were at most 1.8% slower than the default tile.
inline bool IsH200Int4M8NarrowTile(int n, int k, int group_size, bool has_bias) {
  constexpr int kTieMaxN = 4224;
  constexpr int kOneWaveMaxN = 5280;
  constexpr int64_t kTieMaxBytes = 28'000'000;
  constexpr int64_t kOneWaveMaxBytes = 42'000'000;
  constexpr int64_t kBiasOneWaveMaxBytes = 34'000'000;
  if (n > kOneWaveMaxN || group_size <= 0) {
    return false;
  }
  const int64_t bytes = Int4WeightAndScaleBytes(n, k, group_size);
  if (has_bias) {
    return n > kTieMaxN && bytes <= kBiasOneWaveMaxBytes;
  }
  return bytes <= (n <= kTieMaxN ? kTieMaxBytes : kOneWaveMaxBytes);
}

// The A100 and H200 rules were timed with FP16 / BF16, group size 32 / 64 / 128, no zero points, activation
// scales or alpha-in-advance; the dispatcher keeps the default tile on those devices for anything else. A bias
// epilogue is decided here because the H200 rule has its own window for it. Every other device keeps the
// device-independent cutoffs.
inline int PickM8GemvCtaN(M8GemvDevice device, int m, int n, int k, int group_size, int interleave,
                          bool has_bias, int base_cta_n) {
  if (m != 8 || base_cta_n != 4 || (interleave != 2 && interleave != 4) ||
      n <= 0 || n % ((base_cta_n / 2) * interleave) != 0) {
    return base_cta_n;
  }
  bool narrow;
  switch (device) {
    case M8GemvDevice::kA100:
      // A100 timings need both lower and upper bounds: more blocks help only until the narrow tile crosses
      // a resident-block wave boundary. They were taken with the scale-only epilogue.
      narrow = !has_bias && (interleave == 4
                                 ? ((n > 3456 && n <= 4320) || (k <= 3072 && n > 1728 && n <= 2560))
                                 : (n > 864 && n <= 1304));
      break;
    case M8GemvDevice::kH200:
      // INT8 (interleave 2) never beat the default tile by a margin that survives K, with or without a bias:
      // the narrow tile slows down steadily with K and only small, non-monotonic windows remain.
      narrow = interleave == 4 && IsH200Int4M8NarrowTile(n, k, group_size, has_bias);
      break;
    default: {
      const int max_grid_y = interleave == 2 ? 160 : 320;
      narrow = n / (base_cta_n * interleave) <= max_grid_y;
      break;
    }
  }
  return narrow ? base_cta_n / 2 : base_cta_n;
}

// SM count that the wave-aware tile selection (PickGemvCtaN) runs with: the one the caller reports in
// Params::device_sm_count if the device is sm_12x (consumer / workstation Blackwell), else 0. A caller that does
// not report its device (0) gets 0 as well, so it never gets a wave-aware tile.
inline int GemvPickSmCount(int device_sm, int device_sm_count) {
  return device_sm / 10 == 12 && device_sm_count > 0 ? device_sm_count : 0;
}

// Picks the dense GEMV column tile (CtaN) that fills the GPU's block slots best.
//
// The M = 8 kernel is register-limited to 4 / 3 / 2 resident 128-thread blocks per SM for
// CtaN = 2 / 4 / 8 (fp16 int4 on sm_120: 110 / 140 / 250 registers), and a decode projection
// launches only n / (CtaN * interleave) blocks, so the last wave is often mostly empty (for
// example N = 10240 at CtaN = 4 runs 640 blocks on the 510 slots of an RTX 5090). Returns the
// candidate with the highest wave efficiency, and only leaves `base_cta_n` when the gain is clear.
// The slot counts and the gain were measured on sm_120 only, so `sm_count` (GemvPickSmCount) is 0 elsewhere.
inline int PickGemvCtaN(bool wave_aware, int m, int n, int interleave, int base_cta_n, int sm_count) {
  if (!wave_aware || m != 8 || sm_count <= 0) {
    return base_cta_n;
  }
  auto efficiency = [&](int cta_n, int blocks_per_sm) {
    int const cols = cta_n * interleave;
    if (n % cols != 0) {
      return 0.0;
    }
    long long const blocks = n / cols;
    long long const slots = static_cast<long long>(sm_count) * blocks_per_sm;
    long long const waves = (blocks + slots - 1) / slots;
    return static_cast<double>(blocks) / static_cast<double>(waves * slots);
  };
  auto blocks_per_sm = [](int cta_n) { return cta_n <= 2 ? 4 : (cta_n <= 4 ? 3 : 2); };
  int best = base_cta_n;
  double best_eff = efficiency(base_cta_n, blocks_per_sm(base_cta_n)) * 1.05;
  for (int cta_n : {base_cta_n / 2, base_cta_n * 2}) {
    double const eff = efficiency(cta_n, blocks_per_sm(cta_n));
    if (eff > best_eff) {
      best = cta_n;
      best_eff = eff;
    }
  }
  return best;
}

// The tensor-core INT8 GEMV (int8_tc_gemv.cuh) is on unless ORT_DISABLE_FPA_INTB_GEMV_TC=1; the callers that build
// Params copy it into Params::tensor_core. The variable is read directly, as cuda_runtime_utils.h does: this leaf header
// can be included before provider_api.h, where core/platform/env_var_utils.h clashes with the provider logging stubs.
// std::getenv raises C4996 on MSVC, so _dupenv_s is used there.
inline bool GemvTensorCoreEnabled() {
  static const bool enabled = [] {
#if defined(_WIN32)
    char* value = nullptr;
    size_t value_len = 0;
    const bool disabled = _dupenv_s(&value, &value_len, "ORT_DISABLE_FPA_INTB_GEMV_TC") == 0 && value != nullptr &&
                          value[0] == '1' && value[1] == '\0';
    std::free(value);
    return !disabled;
#else
    const char* value = std::getenv("ORT_DISABLE_FPA_INTB_GEMV_TC");
    return value == nullptr || value[0] != '1' || value[1] != '\0';
#endif
  }();
  return enabled;
}

struct Params {
  using Pointer = void*;
  using ConstPointer = void const*;
  Pointer act;
  Pointer act_scale;
  Pointer weight;
  Pointer scales;
  Pointer zeros;
  Pointer bias;
  Pointer out;
  float alpha;
  int m;
  int n;
  int k;
  int groupsize;
  KernelType type;
  bool apply_alpha_in_advance;
  // Selects the paired-K fp16 int4 kernel (M = 5..8). Ignored when the kernel does not support it.
  bool paired_k = false;
  bool wave_aware = false;
  bool debug = false;
  // Allows the tensor-core INT8 kernel for BF16 activations with symmetric group-wise INT8 weights, M = 1..8, on
  // SM 9.0 (device_sm and device_sm_count must be reported). Other launches ignore it.
  bool tensor_core = false;
  int decode_variant = 0;
  // Device that runs the launch, reported by the caller (0 = unknown, which selects the device-independent
  // cutoffs and no wave-aware tile). The raw device values, not the packing arch passed to kernel_launcher. The
  // dispatcher keys its M = 8 tile choices (A100 windows, wave-aware SM count) on them instead of querying the
  // CUDA runtime on every launch.
  int device_sm = 0;        // major * 10 + minor
  int device_sm_count = 0;  // multiProcessorCount

  Params(ConstPointer _act, ConstPointer _act_scale, ConstPointer _weight, ConstPointer _scales, ConstPointer _zeros,
         ConstPointer _bias, Pointer _out, float _alpha, int _m, int _n, int _k, int _groupsize, KernelType _type,
         bool _apply_alpha_in_advance = false)
      : act(const_cast<Pointer>(_act)),
        act_scale(const_cast<Pointer>(_act_scale)),
        weight(const_cast<Pointer>(_weight)),
        scales(const_cast<Pointer>(_scales)),
        zeros(const_cast<Pointer>(_zeros)),
        bias(const_cast<Pointer>(_bias)),
        out(_out),
        alpha(_alpha),
        m(_m),
        n(_n),
        k(_k),
        groupsize(_groupsize),
        type(_type),
        apply_alpha_in_advance(_apply_alpha_in_advance) {
  }
};

void kernel_launcher(int kernel_arch, Params& params, cudaStream_t s);

bool is_supported(int device_arch, int kernel_arch, KernelType kernel_type);

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
