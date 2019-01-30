// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/conv_base.h"
#include "core/util/gemmlowp_common_wrapper.h"

namespace onnxruntime {
namespace contrib {
class QLinearConv : public OpKernel, public ConvBase {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info), ConvBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

  void QuantizeMultiplier(float fp_multiplier, std::int32_t* integer_multiplier, int* right_shift) const;

  void ScaleAndZeropointPairValidationHelper(const Tensor* scale, const Tensor* zeropoint) const;  
};

typedef gemmlowp::VectorMap<const std::int32_t, gemmlowp::VectorShape::Col> ColVectorMap;

inline std::tuple<gemmlowp::OutputStageBiasAddition<ColVectorMap>,
                  gemmlowp::OutputStageQuantizeDownInt32ByFixedPoint,
                  gemmlowp::OutputStageSaturatingCastToUint8>
MakeOutputPipelineWithBias(const int32_t* bias,
                           int rows,
                           std::int32_t result_offset,
                           std::int32_t result_mult_int,
                           std::int32_t result_shift) {
  ColVectorMap bias_vector(bias, rows);
  gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
  bias_addition_stage.bias_vector = bias_vector;
  gemmlowp::OutputStageQuantizeDownInt32ByFixedPoint quantize_down_stage;
  quantize_down_stage.result_offset_after_shift = result_offset;
  quantize_down_stage.result_fixedpoint_multiplier = result_mult_int;
  quantize_down_stage.result_shift = result_shift;
  gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
  return std::make_tuple(bias_addition_stage, quantize_down_stage, saturating_cast_stage);
}

inline std::tuple<gemmlowp::OutputStageQuantizeDownInt32ByFixedPoint,
                  gemmlowp::OutputStageSaturatingCastToUint8>
MakeOutputPipelineWithOutBias(std::int32_t result_offset,
                              std::int32_t result_mult_int,
                              std::int32_t result_shift) {
  gemmlowp::OutputStageQuantizeDownInt32ByFixedPoint quantize_down_stage;
  quantize_down_stage.result_offset_after_shift = result_offset;
  quantize_down_stage.result_fixedpoint_multiplier = result_mult_int;
  quantize_down_stage.result_shift = result_shift;
  gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
  return std::make_tuple(quantize_down_stage, saturating_cast_stage);
}
}
}  // namespace onnxruntime
