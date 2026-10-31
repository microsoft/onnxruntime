// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/subgroup_matrix_pad_b.h"

#include <limits>

#include "core/common/narrow.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/tensor/pad.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

Status PadSubgroupMatrixWeightColumns(ComputeContext& context, const Tensor& b, const TensorShape& b_shape,
                                      uint32_t N, uint32_t n_b, std::unique_ptr<Tensor>& padded) {
  TensorShapeVector padded_dims{b_shape.GetDims().begin(), b_shape.GetDims().end()};
  padded_dims.back() = static_cast<int64_t>(n_b);
  const TensorShape padded_shape{padded_dims};
  const int64_t output_size_i64 = padded_shape.Size();
  ORT_RETURN_IF_NOT(output_size_i64 <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                    "Cannot pad the subgroup-matrix weight because the padded tensor has ", output_size_i64,
                    " elements, exceeding uint32_t shader indexing range.");
  const uint32_t output_size = narrow<uint32_t>(output_size_i64);

  padded = std::make_unique<Tensor>(context.CreateGPUTensor(b.DataType(), padded_shape));
  // A zero-element padded tensor needs no pad pass -- dispatching 0 workgroups is
  // pointless and some drivers reject it.
  if (output_size == 0) {
    return Status::OK();
  }

  // Reuses the Pad op's program as a 2D constant pad: B viewed as [rows, N] gets
  // N_b - N zero columns appended on the right ([rows, N_b]).
  const int64_t rows = b_shape.Size() / N;
  const TensorShape input_2d{rows, static_cast<int64_t>(N)};
  const TensorShape output_2d{rows, static_cast<int64_t>(n_b)};
  const std::vector<int32_t> lower_pads{0, 0};
  constexpr uint32_t kZeroPadValue = 0;  // +0.0 bit pattern for f16 and f32

  PadProgram program{Mode::Constant, /*dim_value_zero=*/false, /*is_float16=*/b.IsDataType<MLFloat16>()};
  program.AddInput({&b, ProgramTensorMetadataDependency::Rank, input_2d, 1})
      .AddOutput({padded.get(), ProgramTensorMetadataDependency::TypeAndRank, output_2d, 1})
      .SetDispatchGroupSize(CeilDiv<uint32_t>(output_size, WORKGROUP_SIZE))
      .CacheHint(std::to_string(static_cast<int>(Mode::Constant)), false)
      .AddUniformVariables({{gsl::span<const int32_t>(lower_pads.data(), lower_pads.size())},
                            {output_size},
                            {kZeroPadValue}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
