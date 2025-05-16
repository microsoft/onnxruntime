/*
 * Copyright (c) 2017-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

/*! \file
  \brief Functor performing linear combination with a maximum operation used by epilogues.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/functional.h"
#include "cutlass/half.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace epilogue
{
namespace thread
{

/////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__forceinline__ __device__ float tanh_opt(float x)
{
#if (__CUDACC_VER_MAJOR__ < 11) || (__CUDA_ARCH__ < 750)
    float const exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#else
    return fast_tanh(x);
#endif
}

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
