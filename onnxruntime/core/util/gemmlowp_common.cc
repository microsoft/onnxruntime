// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/util/gemmlowp_common_wrapper.h"
#include "core/util/gemmlowp_common.h"

namespace onnxruntime {

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

void GemmlowpMultiplyu8u8_u8(const uint8_t* lhs_data, const uint8_t* rhs_data, uint8_t* result_data,
                        const int lhs_offset, const int rhs_offset, const int result_offset,
                        int m, int n, int k, int32_t int_multiplier, int32_t right_shift, const int32_t* bias) {
  // TODO exp ColMajor order for rhs and result. That may be faster
  const auto matOrder = gemmlowp::MapOrder::RowMajor;
  gemmlowp::MatrixMap<const uint8_t, matOrder> lhs(lhs_data, m, k);
  gemmlowp::MatrixMap<const uint8_t, matOrder> rhs(rhs_data, k, n);
  gemmlowp::MatrixMap<std::uint8_t, matOrder> result(result_data, m, n);

  gemmlowp::GemmContext gemm_context;

  if (bias == nullptr) {
    auto output_pipeline = MakeOutputPipelineWithOutBias(result_offset,
                                                         int_multiplier, right_shift);
    gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t, gemmlowp::DefaultL8R8BitDepthParams>(
        &gemm_context, lhs, rhs, &result, -lhs_offset, -rhs_offset, output_pipeline);
  } else {
    auto output_pipeline = MakeOutputPipelineWithBias(bias, m, result_offset, int_multiplier, right_shift);
    gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t,
                                     gemmlowp::DefaultL8R8BitDepthParams>(
        &gemm_context, lhs, rhs, &result, -lhs_offset, -rhs_offset, output_pipeline);
  }
}

void GemmlowpMultiplyu8u8_s32(const uint8_t* lhs_data, const uint8_t* rhs_data, int32_t* result_data,
                                const int lhs_offset, const int rhs_offset, int m, int n, int k, concurrency::ThreadPool* ) {

  const auto matOrder = gemmlowp::MapOrder::RowMajor;
  gemmlowp::MatrixMap<const uint8_t, matOrder> lhs(lhs_data, m, k);
  gemmlowp::MatrixMap<const uint8_t, matOrder> rhs(rhs_data, k, n);
  gemmlowp::MatrixMap<std::int32_t, matOrder> result(result_data, m, n);

  gemmlowp::GemmContext gemm_context;

  const std::tuple<> empty_pipeline = {};

  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t, gemmlowp::DefaultL8R8BitDepthParams>(
      &gemm_context, lhs, rhs, &result, -lhs_offset, -rhs_offset, empty_pipeline);
}

}
