/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdint>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

enum Data_type {
  DATA_TYPE_BOOL,
  DATA_TYPE_E8M10,
  DATA_TYPE_E8M7,
  DATA_TYPE_FP16,
  DATA_TYPE_FP32,
  DATA_TYPE_INT4,
  DATA_TYPE_INT8,
  DATA_TYPE_INT32
};

constexpr int32_t kSM_70 = 70;
constexpr int32_t kSM_75 = 75;
constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;
constexpr int32_t kSM_89 = 89;

void set_alpha_fp16(uint32_t& alpha, float norm);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
