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
#include "fmha_cross_attention.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

int32_t runFMHCAKernel(void const* devQ, void const* devKV, void* cuSeqlensQ, void* cuSeqlensKV, void* devOutput,
                       int32_t sm, FusedMultiHeadCrossAttentionKernel const* kernels,
                       int32_t b = 2, int32_t h = 8, int32_t d = 64,
                       int32_t seqQ = 4096, int32_t seqKV = 77,
                       cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
