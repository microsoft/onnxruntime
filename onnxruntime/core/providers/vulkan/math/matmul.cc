
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/math/matmul.h"

#include "ncnn-src/src/layer/vulkan/gemm_vulkan.h"          // all other MatMul
#include "ncnn-src/src/layer/vulkan/innerproduct_vulkan.h"  // 2D MatMul

#include "core/common/safeint.h"
#include "core/optimizer/initializer.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/vulkan/vulkan_utils.h"
#include "core/providers/vulkan/shape_utils.h"

namespace onnxruntime {
namespace vulkan {

MatMulKernel::MatMulKernel(const VulkanExecutionProvider& vulkan_ep,
                           const GraphViewer& graph_viewer,
                           const onnxruntime::Node& node)
    : VulkanKernel{vulkan_ep, node} {
  auto input_defs = Node().InputDefs();

  constant_A_ = graph_viewer.IsConstantInitializer(input_defs[0]->Name(), /* check_outer_scope */ true);
  constant_B_ = graph_viewer.IsConstantInitializer(input_defs[1]->Name(), /* check_outer_scope */ true);
}

Status MatMulKernel::SetupParamDict(const GraphViewer& graph_viewer, ncnn::ParamDict& params) {
  const auto& logger = Logger();
  auto input_defs = Node().InputDefs();

  std::vector<int64_t> shape_A, shape_B;
  bool have_shape_A = GetShape(*input_defs[0], shape_A, logger);
  bool have_shape_B = GetShape(*input_defs[1], shape_B, logger);

  // User InnerProduct if B is constant with rank 2
  // same logic as https://github.com/Tencent/ncnn/blob/92e0b8253bc9d16b0d77bd17693fe9a72fb64b64/tools/onnx/onnx2ncnn.cpp#L5111
  use_inner_product_ = constant_B_ && shape_B.size() == 2;  // constant initializer always has a shape

  if (use_inner_product_) {
    /* InnerProduct params
      num_output = pd.get(0, 0);
      bias_term = pd.get(1, 0);
      weight_data_size = pd.get(2, 0);  num elements in B
      int8_scale_term = pd.get(8, 0);
      activation_type = pd.get(9, 0);      // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
      activation_params = pd.get(10, Mat());
    */
    params.set(0, narrow<int32_t>(shape_B[1]));                                      // num_output
    params.set(2, static_cast<int32_t>(SafeInt<int32_t>(shape_B[0]) * shape_B[1]));  // weight_data_size

  } else {
    /* Gemm params
      alpha = pd.get(0, 1.f);
      beta = pd.get(1, 1.f);
      transA = pd.get(2, 0);
      transB = pd.get(3, 0);
      constantA = pd.get(4, 0);
      constantB = pd.get(5, 0);
      constantC = pd.get(6, 0);
      constantM = pd.get(7, 0);
      constantN = pd.get(8, 0);
      constantK = pd.get(9, 0);
      constant_broadcast_type_C = pd.get(10, 0);
      output_N1M = pd.get(11, 0);
      output_elempack = pd.get(12, 0);
      output_elemtype = pd.get(13, 0);
      output_transpose = pd.get(14, 0);
      constant_TILE_M = pd.get(20, 0);
      constant_TILE_N = pd.get(21, 0);
      constant_TILE_K = pd.get(22, 0);
    */
    params.set(4, constant_A_ ? 1 : 0);
    params.set(5, constant_B_ ? 1 : 0);
    params.set(6, 1);  // constantC: no bias in MatMul so C is always a constant with the default value
  }

  bool constant_k = false;

  if (std::vector<int64_t> shape_a; GetShape(*input_defs[0], shape_a, logger)) {
    if (shape_a[0] != -1) {
      params.set(7, 1);  // constantM
    }
    constant_k = shape_a[1] != -1;
  }

  if (std::vector<int64_t> shape_b; GetShape(*input_defs[0], shape_b, logger)) {
    if (shape_b[1] != -1) {
      params.set(8, 1);  // constantN
    }

    constant_k |= shape_b[0] != -1;
  }

  params.set(9, constant_k ? 1 : 0);  // constantK
}

Status MatMulKernel::SetupConstantInitializers(const GraphViewer& graph_viewer, ncnn::Layer& layer) {
  const auto& node = Node();
  auto input_defs = node.InputDefs();
  const auto& logger = Logger();

  std::vector<int64_t> shape_A, shape_B;
  bool have_shape_A = GetShape(*input_defs[0], shape_A, logger);
  bool have_shape_B = GetShape(*input_defs[1], shape_B, logger);

  if (use_inner_product_) {
    // need to transpose from K, N to N, K when setting up constant initializers
    const auto& b_tensorproto = *graph_viewer.GetConstantInitializer(input_defs[1]->Name());
    Initializer b_data(b_tensorproto);
    std::vector<int64_t> target_shape{shape_B[1], shape_B[0]};
    std::vector<size_t> stride{narrow<size_t>(shape_B[0])};
    auto src_data = b_data.DataAsByteSpan();

    std::vector<float> target_data;
    target_data.resize(b_data.size());

    DoTransposeEltWise(shape_B.size(), target_shape, narrow<size_t>(shape_B[1]),
                       stride, src_data.data(), reinterpret_cast<uint8_t*>(target_data.data()), sizeof(float));

    transposed_b_ = std::move(target_data);
  } else {
  }
}

Status MatMulKernel::UploadConstantInitializers(ncnn::VkTransfer& cmd, ncnn::Option& upload_options) {
  ORT_RETURN_IF_ERROR(VulkanKernel::UploadConstantInitializers(cmd, upload_options));

  transposed_b_ = std::nullopt;  // free any temporary data

  return Status::OK();
}

}  // namespace vulkan
}  // namespace onnxruntime
