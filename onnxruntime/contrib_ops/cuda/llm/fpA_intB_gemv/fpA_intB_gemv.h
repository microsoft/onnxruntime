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

// Picks the dense GEMV column tile (CtaN) that fills the GPU's block slots best.
//
// The M = 8 kernel is register-limited to 4 / 3 / 2 resident 128-thread blocks per SM for
// CtaN = 2 / 4 / 8 (fp16 int4 on sm_120: 110 / 140 / 250 registers), and a decode projection
// launches only n / (CtaN * interleave) blocks, so the last wave is often mostly empty (for
// example N = 10240 at CtaN = 4 runs 640 blocks on the 510 slots of an RTX 5090). Returns the
// candidate with the highest wave efficiency, and only leaves `base_cta_n` when the gain is clear.
// The slot counts and the gain were measured on sm_120 only, so `sm_count` is 0 elsewhere.
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
