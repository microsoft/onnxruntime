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
    \brief Templates exposing architecture support for multiply-add operations
*/

#pragma once
#include "contrib_ops/cuda/llm/cutlass_extensions/weight_only_quant_op.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

// Tag which triggers MMA which will trigger
struct OpMultiplyAddDequantizeInterleavedBToA;

/*
  Below we have extra tags to signal what kind of dequantization we want to do
  (per col, scale only fine grained, finegrained with zero). This still lets us
  the existing template infrastructure (incl. that in CUTLASS). However, we
  split out the template below into OpMultiplyAddDequantizeInterleavedBToA along
  with the quantization op before instantiating the GEMM pieces.

  Note that this is somewhat of a hack, but it SIGNIFICANTLY reduces the amount of
  code we need to duplicate.
 */
struct OpMultiplyAddDequantizeInterleavedBToA_percol_scale;
struct OpMultiplyAddDequantizeInterleavedBToA_fine_scale;
struct OpMultiplyAddDequantizeInterleavedBToA_fine_scalebias;

// The default just forwards the original operator
template <typename MmaOp, WeightOnlyQuantOp QuantOp_>
struct TagOperator {
  using TaggedOperator = MmaOp;
};

// Specializations below attach more information to the operator
template <>
struct TagOperator<OpMultiplyAddDequantizeInterleavedBToA, WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY> {
  using TaggedOperator = OpMultiplyAddDequantizeInterleavedBToA_percol_scale;
};

template <>
struct TagOperator<OpMultiplyAddDequantizeInterleavedBToA, WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> {
  using TaggedOperator = OpMultiplyAddDequantizeInterleavedBToA_fine_scale;
};

template <>
struct TagOperator<OpMultiplyAddDequantizeInterleavedBToA, WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS> {
  using TaggedOperator = OpMultiplyAddDequantizeInterleavedBToA_fine_scalebias;
};

// Here we instantiate some structs to "detag" the tagged operator. It splits it back to the original
// operator + the extra information. If no extra info was tagged, the dequant op per column scaling
// as a default.
template <typename TaggedMmaOp>
struct DetagOperator {
  using Operator = TaggedMmaOp;
  static constexpr WeightOnlyQuantOp QuantOp = WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;
};

template <>
struct DetagOperator<OpMultiplyAddDequantizeInterleavedBToA_percol_scale> {
  using Operator = OpMultiplyAddDequantizeInterleavedBToA;
  static constexpr WeightOnlyQuantOp QuantOp = WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;
};

template <>
struct DetagOperator<OpMultiplyAddDequantizeInterleavedBToA_fine_scale> {
  using Operator = OpMultiplyAddDequantizeInterleavedBToA;
  static constexpr WeightOnlyQuantOp QuantOp = WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;
};

template <>
struct DetagOperator<OpMultiplyAddDequantizeInterleavedBToA_fine_scalebias> {
  using Operator = OpMultiplyAddDequantizeInterleavedBToA;
  static constexpr WeightOnlyQuantOp QuantOp = WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS;
};

}  // namespace arch
}  // namespace cutlass
