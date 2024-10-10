// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/tensor/dynamic_time_warping.h"
#include "core/providers/cpu/tensor/utils.h"

#include <vector>
#include <numeric>

using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    DynamicTimeWarping,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("F", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int32_t>()),
    DynamicTimeWarping);

Status DynamicTimeWarping::Compute(OpKernelContext* ctx) const {
  const Tensor& input_tensor = *ctx->Input<Tensor>(0);
  const auto& input_dims = input_tensor.Shape().GetDims();
  int rank = SafeInt<int>(input_dims.size());
  ORT_ENFORCE(rank == 2 || (rank == 3 && input_dims[0] == 1),
              "Currently input rank must be 2, or (3 with first dim equal to 1), but got:", rank);

  const size_t rows = SafeInt<size_t>(input_dims[rank == 3 ? 1 : 0]);
  const size_t cols = SafeInt<size_t>(input_dims[rank == 3 ? 2 : 1]);

  std::vector<std::vector<float>> cost(rows + 1, std::vector<float>(cols + 1, std::numeric_limits<float>::infinity()));
  std::vector<std::vector<int8_t>> trace(rows + 1, std::vector<int8_t>(cols + 1, -1));
  std::vector<std::vector<int32_t>> path_helper;

  // Compute the cost and trace matrices
  cost[0][0] = 0;
  for (size_t j = 1; j <= cols; ++j) {
    for (size_t i = 1; i <= rows; ++i) {
      const float c0 = cost[i - 1][j - 1];
      const float c1 = cost[i - 1][j];
      const float c2 = cost[i][j - 1];

      float cur_cost;
      int8_t cur_trace;
      if (c0 < c1 && c0 < c2) {
        cur_cost = c0;
        cur_trace = 0;
      } else if (c1 < c0 && c1 < c2) {
        cur_cost = c1;
        cur_trace = 1;
      } else {
        cur_cost = c2;
        cur_trace = 2;
      }

      cost[i][j] = cur_cost + input_tensor.Data<float>()[(i - 1) * cols + j - 1];
      trace[i][j] = cur_trace;
    }
  }

  // Back-tracing to find the optimal path
  int i = static_cast<int>(rows);
  int j = static_cast<int>(cols);
  int result_len = 0;
  while (i > 0 && j > 0) {
    path_helper.push_back({i - 1, j - 1});
    ++result_len;
    int8_t cur_trace = trace[i][j];
    switch (cur_trace) {
      case 0:
        --i;
        --j;
        break;
      case 1:
        --i;
        break;
      case 2:
        --j;
        break;
      default:
        ORT_THROW("Invalid trace value: ", cur_trace);
    }
  }

  // Update the output tensor
  Tensor* output_tensor = ctx->Output(0, TensorShape{2LL, SafeInt<int64_t>(result_len)});
  auto* output_data = output_tensor->MutableData<int32_t>();
  for (int k = 0; k < result_len; ++k) {
    output_data[k] = path_helper[result_len - k - 1][0];
    output_data[k + result_len] = path_helper[result_len - k - 1][1];
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
