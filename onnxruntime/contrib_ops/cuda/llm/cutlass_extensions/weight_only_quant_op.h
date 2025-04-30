/*
 * Copyright (c) 2017-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

namespace cutlass
{

enum class WeightOnlyQuantOp
{
    UNDEFINED,
    PER_COLUMN_SCALE_ONLY,
    FINEGRAINED_SCALE_ONLY,
    FINEGRAINED_SCALE_AND_ZEROS
};

constexpr bool isFinegrained(WeightOnlyQuantOp op)
{
    return op == WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS || op == WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;
}

constexpr bool hasZero(WeightOnlyQuantOp op)
{
    return op == WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS;
}

} // namespace cutlass
