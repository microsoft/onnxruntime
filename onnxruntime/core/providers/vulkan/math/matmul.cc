
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/math/matmul.h"

// 2D MatMul -> InnerProduct
// Everything else -> Gemm
#include "include/ncnn/layer/gemm.h"
#include "include/ncnn/layer/vulkan/gemm_vulkan.h"
#include "include/ncnn/layer/vulkan/innerproduct_vulkan.h"

#include "core/common/safeint.h"
#include "core/optimizer/initializer.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/vulkan/vulkan_utils.h"
#include "core/providers/vulkan/shape_utils.h"

namespace onnxruntime {
namespace vulkan {

MatMulKernel::InputInfo::InputInfo(const GraphViewer& graph_viewer, const onnxruntime::Node& node,
                                   const logging::Logger& logger) {
  auto input_defs = node.InputDefs();

  constant_A = graph_viewer.IsConstantInitializer(input_defs[0]->Name(), /* check_outer_scope */ true);
  constant_B = graph_viewer.IsConstantInitializer(input_defs[1]->Name(), /* check_outer_scope */ true);

  have_shape_A = GetShape(*input_defs[0], shape_A, logger);
  have_shape_B = GetShape(*input_defs[1], shape_B, logger);
}

/* static */
bool MatMulKernel::IsSupported(const GraphViewer& graph_viewer, const onnxruntime::Node& node,
                               const logging::Logger& logger) {
  // start with InnerProduct.
  InputInfo info(graph_viewer, node, logger);

  // User InnerProduct if B is constant and is 2D
  // https://github.com/Tencent/ncnn/blob/92e0b8253bc9d16b0d77bd17693fe9a72fb64b64/tools/onnx/onnx2ncnn.cpp#L5111
  return info.constant_B && info.shape_B.size() == 2;  // constant initializer always has a shape
}

MatMulKernel::MatMulKernel(const VulkanExecutionProvider& vulkan_ep,
                           const GraphViewer& graph_viewer,
                           const onnxruntime::Node& node)
    : VulkanKernel{vulkan_ep, node},
      input_info_{graph_viewer, Node(), Logger()},
      use_inner_product_{input_info_.constant_B && input_info_.shape_B.size() == 2} {
}

Status MatMulKernel::SetupParamDict(const GraphViewer& /*graph_viewer*/, ncnn::ParamDict& params) {
  // const auto& logger = Logger();

  if (use_inner_product_) {
    /* InnerProduct params
      num_output = pd.get(0, 0);
      bias_term = pd.get(1, 0);
      weight_data_size = pd.get(2, 0);
      int8_scale_term = pd.get(8, 0);
      activation_type = pd.get(9, 0);      // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
      activation_params = pd.get(10, Mat());
    */
    int32_t num_elements = SafeInt<int32_t>(input_info_.shape_B[0]) * input_info_.shape_B[1];

    params.set(0, narrow<int32_t>(input_info_.shape_B[1]));  // num_output
    params.set(2, num_elements);                             // weight_data_size
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
    params.set(4, input_info_.constant_A ? 1 : 0);
    params.set(5, input_info_.constant_B ? 1 : 0);
    params.set(6, 1);  // constantC: no bias in MatMul so C is always a constant with the default value

    bool constant_k = false;

    if (input_info_.have_shape_A) {
      if (input_info_.shape_A[0] != -1) {
        params.set(7, 1);  // constantM
      }

      constant_k = input_info_.shape_A[1] != -1;
    }

    if (input_info_.have_shape_B) {
      if (input_info_.shape_B[1] != -1) {
        params.set(8, 1);  // constantN
      }

      constant_k |= input_info_.shape_B[0] != -1;
    }

    params.set(9, constant_k ? 1 : 0);  // constantK

    // TODO: Other params
  }

  return Status::OK();
}

Status MatMulKernel::SetupConstantInitializers(const GraphViewer& graph_viewer, ValueIndexes& value_indexes) {
  // const auto& logger = Logger();
  const auto& node = Node();
  ncnn::Layer& layer = Layer();
  const auto& input_defs = node.InputDefs();

  if (use_inner_product_) {
    // need to transpose from K, N to N, K when setting up constant initializers
    const auto& shape_B = input_info_.shape_B;

    const auto& tensorproto_B = *graph_viewer.GetConstantInitializer(input_defs[1]->Name());
    Initializer data_B(tensorproto_B);
    auto src_data = data_B.DataAsSpan<float>();

    auto cur_K = narrow<int32_t>(shape_B[0]);
    auto cur_M = narrow<int32_t>(shape_B[1]);

    // there's no existing way to access the ORT CPU EP allocator to use it with the Mat allocations, but it probably
    // doesn't matter given a) we shouldn't need to transpose too many intializers and b) the memory is freed once
    // we upload to GPU, all of which happens during model loading.
    ncnn::Mat transposed_data(cur_M, cur_K);
    float* dst_data = static_cast<float*>(transposed_data.data);

    for (size_t x = 0; x < cur_K; x++) {
      for (size_t y = 0; y < cur_M; y++) {
        dst_data[y * cur_K + x] = src_data[x * cur_M + y];
      }
    }

    ncnn::InnerProduct& inner_product = static_cast<ncnn::InnerProduct&>(layer);

    // FIXME: This setup is inefficient. InnerProduct_vulkan does another round of packing in
    // create_pipeline so we have the ONNX weights, the InnerProduct.weight_data copy, and another temporary one in
    // InnerProduct_vulkan.weight_data_packed before we upload to GPU. The latter gets released after upload_model,
    // but the value in InnerProduct.weight_data does not. We can free that manually at least after we create the
    // pipeline. Ideally we do the packing into InnerProduct_vulkan.weight_data_packed directly and never use
    // InnerProduct.weight_data, but overriding InnerProduct_vulkan::create_pipeline to do that would be non-trivial.
    inner_product.weight_data = std::move(transposed_data);

    value_indexes.Add(*input_defs[1]);
  } else {
    ORT_NOT_IMPLEMENTED("Requires gemm");
  }

  return Status::OK();
}

Status MatMulKernel::UploadConstantInitializers(ncnn::VkTransfer& cmd, ncnn::Option& upload_options) {
  ORT_RETURN_IF_ERROR(VulkanKernel::UploadConstantInitializers(cmd, upload_options));

  // was using this to free data but using CreatePipeline to cleanup data in InnerProduct, and
  // upload_model cleans up data in InnerProduct_vulkan
  return Status::OK();
}

Status MatMulKernel::CreatePipeline() {
  ORT_RETURN_IF_ERROR(VulkanKernel::CreatePipeline());
  if (use_inner_product_) {
    ncnn::InnerProduct& inner_product = static_cast<ncnn::InnerProduct&>(Layer());
    inner_product.weight_data.release();
  } else {
    ORT_NOT_IMPLEMENTED("Requires gemm");
  }

  return Status::OK();
}

}  // namespace vulkan
}  // namespace onnxruntime
