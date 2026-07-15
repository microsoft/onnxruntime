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

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

enum class KernelType {
  FP16Int8Groupwise,
  FP16Int4Groupwise,
  FP16Int8PerChannel,
  FP16Int4PerChannel,
  BF16Int8Groupwise,
  BF16Int4Groupwise,
  BF16Int8PerChannel,
  BF16Int4PerChannel
};

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

void kernel_launcher(int arch, Params& params, cudaStream_t s);

bool is_supported(int arch, KernelType kernel_type);

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
