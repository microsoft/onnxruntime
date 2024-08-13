
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/math/matmul.h"

// 2D MatMul -> InnerProduct
// Everything else -> Gemm
#include "include/ncnn/layer/gemm.h"
#include "include/ncnn/layer/vulkan/gemm_vulkan.h"
#include "include/ncnn/layer/vulkan/innerproduct_vulkan.h"

#include "core/common/safeint.h"
#include "core/framework/transpose_helper.h"
#include "core/optimizer/initializer.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/vulkan/vulkan_utils.h"
#include "core/providers/vulkan/shape_utils.h"

#include "core/providers/vulkan/shaders/include/shaderncnn.innerproduct_gemm.hpp"

namespace onnxruntime {
namespace vulkan {

#define REGISTER_VERSIONED_KERNEL(op, since_version, end_version)                                    \
  REGISTER_ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL(                                                    \
      op, since_version, end_version,                                                                \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      op);

#define REGISTER_KERNEL(op, since_version)                                                           \
  REGISTER_ONNX_OPERATOR_VULKAN_KERNEL(                                                              \
      op, since_version,                                                                             \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      op);

REGISTER_VERSIONED_KERNEL(MatMul, 6, 12);
REGISTER_KERNEL(MatMul, 13);

namespace {
static void TransposeB(const GraphViewer& graph_viewer, const std::string& input_name,
                       const std::vector<int64_t>& shape_B, gsl::span<float> dst_data) {
  const auto& tensorproto_B = *graph_viewer.GetConstantInitializer(input_name);
  Initializer data_B(tensorproto_B);
  auto src_data = data_B.DataAsSpan<float>();

  auto cur_K = narrow<int32_t>(shape_B[0]);
  auto cur_M = narrow<int32_t>(shape_B[1]);

  for (size_t x = 0; x < cur_K; x++) {
    for (size_t y = 0; y < cur_M; y++) {
      dst_data[y * cur_K + x] = src_data[x * cur_M + y];
    }
  }
}
}  // namespace
MatMulKernel::InputInfo::InputInfo(const GraphViewer& graph_viewer, const onnxruntime::Node& node,
                                   const logging::Logger& logger) {
  auto input_defs = node.InputDefs();

  constant_A = graph_viewer.IsConstantInitializer(input_defs[0]->Name(), /* check_outer_scope */ true);
  constant_B = graph_viewer.IsConstantInitializer(input_defs[1]->Name(), /* check_outer_scope */ true);

  have_shape_A = GetShape(*input_defs[0], shape_A, logger);
  have_shape_B = GetShape(*input_defs[1], shape_B, logger);
}

/* static */
bool MatMulKernel::IsSupported(bool /*use_kompute*/, const GraphViewer& graph_viewer, const onnxruntime::Node& node,
                               const logging::Logger& logger) {
  // start with InnerProduct.
  InputInfo info(graph_viewer, node, logger);

  // User InnerProduct if B is constant and is 2D
  // https://github.com/Tencent/ncnn/blob/92e0b8253bc9d16b0d77bd17693fe9a72fb64b64/tools/onnx/onnx2ncnn.cpp#L5111
  // A can be 1D or 2D. the NCNN implementation doesn't handle the batches if > 2D.
  // We could manually do that but not worth it for our testing purposes.
  bool a_ok = info.have_shape_A && info.shape_A.size() < 3 && !DoesShapeSpecifyZeroElements(info.shape_A);
  bool b_ok = info.constant_B && info.shape_B.size() == 2 && !DoesShapeSpecifyZeroElements(info.shape_B);

  // same support for Kompute and NCNN in current implementation
  return a_ok && b_ok;
}

MatMulKernel::MatMulKernel(const VulkanExecutionProvider& vulkan_ep,
                           bool use_kompute,
                           const GraphViewer& graph_viewer,
                           const onnxruntime::Node& node)
    : VulkanKernel{vulkan_ep, use_kompute, node},
      input_info_{graph_viewer, Node(), Logger()},
      use_inner_product_{input_info_.constant_B && input_info_.shape_B.size() == 2} {
}

Status MatMulKernel::ComputeImpl(OpKernelContext& context) const {
  // const auto& logger = Logger();
  const auto& input_tensor_A = *context.Input<Tensor>(0);
  ORT_ENFORCE(transposed_b_tensor_);  // should have been set in PrePack
  TensorShape output_shape{input_tensor_A.Shape()[0], transposed_b_tensor_->Shape()[0]};

  const auto& output_tensor = *context.Output(0, output_shape);
  ORT_UNUSED_PARAMETER(output_tensor);

  // TODO: Create the algo in the ctor and use here.
  // Call set_tensors to plug in the values for this execution.
  // TODO: Is this whole setup thread safe or do we need a way for the tensors to be plugged in during record
  // instead of being Algorithm class members?

  return Status::OK();
}

Status MatMulKernel::SetupNcnnParamDict(const GraphViewer& /*graph_viewer*/, ncnn::ParamDict& params) {
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

Status MatMulKernel::SetupNcnnConstantInitializers(const GraphViewer& graph_viewer, ValueIndexes& /*value_indexes*/) {
  // const auto& logger = Logger();
  const auto& node = Node();
  ncnn::Layer& layer = NcnnLayer();
  const auto& input_defs = node.InputDefs();

  if (use_inner_product_) {
    // need to transpose from K, M to M, K when setting up constant initializers
    const auto& shape_B = input_info_.shape_B;
    auto cur_K = narrow<int32_t>(shape_B[0]);
    auto cur_M = narrow<int32_t>(shape_B[1]);

    // there's no existing way to access the ORT CPU EP allocator to use it with the Mat allocations, but it probably
    // doesn't matter given a) we shouldn't need to transpose too many initializers and b) the memory is freed once
    // we upload to GPU, all of which happens during model loading.
    ncnn::Mat transposed_data(/* w */ cur_K, /* h */ cur_M);
    gsl::span<float> dst_data = gsl::make_span(static_cast<float*>(transposed_data.data), size_t(cur_K * cur_M));

    TransposeB(graph_viewer, input_defs[1]->Name(), shape_B, dst_data);

    ncnn::InnerProduct& inner_product = static_cast<ncnn::InnerProduct&>(layer);

    // FIXME: This setup is inefficient. InnerProduct_vulkan does another round of packing in
    // create_pipeline so we have the ONNX weights, the InnerProduct.weight_data copy, and another temporary one in
    // InnerProduct_vulkan.weight_data_packed before we upload to GPU. The latter gets released after upload_model,
    // but the value in InnerProduct.weight_data does not. We can free that manually at least after we create the
    // pipeline. Ideally we do the packing into InnerProduct_vulkan.weight_data_packed directly and never use
    // InnerProduct.weight_data, but overriding InnerProduct_vulkan::create_pipeline to do that would be non-trivial.
    inner_product.weight_data = std::move(transposed_data);
  } else {
    ORT_NOT_IMPLEMENTED("Requires gemm");
  }

  // if there are constant initializers that are NOT directly added to the layer we need to add the to value_indexes
  // here so they get assigned an index number.

  return Status::OK();
}

Status MatMulKernel::UploadNcnnConstantInitializers(ncnn::VkTransfer& cmd, ncnn::Option& upload_options) {
  ORT_RETURN_IF_ERROR(VulkanKernel::UploadNcnnConstantInitializers(cmd, upload_options));

  // was using this to free data but using CreatePipeline to cleanup data in InnerProduct, and
  // upload_model cleans up data in InnerProduct_vulkan
  return Status::OK();
}

Status MatMulKernel::CreateNcnnPipeline() {
  ORT_RETURN_IF_ERROR(VulkanKernel::CreateNcnnPipeline());
  if (use_inner_product_) {
    // This happens in InnerProduct_vulkan::create_pipeline when lightmode is on (it is by default)
    // ncnn::InnerProduct& inner_product = static_cast<ncnn::InnerProduct&>(Layer());
    // inner_product.weight_data.release();
  } else {
    ORT_NOT_IMPLEMENTED("Requires gemm");
  }

  return Status::OK();
}

void MatMulKernel::KomputeProcessConstantInitializers(
    const GraphViewer& graph_viewer, kp::Manager& manager,
    std::unordered_map<const NodeArg*, std::shared_ptr<kp::Tensor>>& initializers_to_upload) const {
  // we only support usage with a constant B currently
  const NodeArg* const_B = Node().InputDefs()[1];

  // as we're going to transform the data to match the shader we don't expect the initializer to already exist.
  // it's valid if multiple MatMul nodes share the same initializer, but not if a non-MatMul node is using it.
  ORT_ENFORCE(initializers_to_upload.count(const_B) == 0,
              "Constant initializer was already added. Layout may differ.");

  size_t num_elements = input_info_.shape_B[0] * input_info_.shape_B[1];
  std::vector<float> data(num_elements);
  TransposeB(graph_viewer, const_B->Name(), input_info_.shape_B, data);

  initializers_to_upload[const_B] = manager.tensor(data);
}

Status MatMulKernel::KomputeExecute(kp::Manager& manager, kp::Sequence& sequence,
                                    std::unordered_map<const NodeArg*, std::shared_ptr<kp::Tensor>>& values) const {
  auto tensorInA = values.at(Node().InputDefs()[0]);
  auto tensorInB = values.at(Node().InputDefs()[1]);

  // this is stupidly inefficient as it allocates device and staging buffers and copies data into the staging buffer
  auto num_output_elements = input_info_.shape_A[0] * input_info_.shape_B[1];
  auto tensorOutA = manager.tensor(std::vector<float>(num_output_elements, 0.f));
  values[Node().OutputDefs()[0]] = tensorOutA;

  auto shader_bytes = gsl::make_span(kp::shader_data::compiled_shaders_ncnn_innerproduct_gemm_comp_spv,
                                     kp::shader_data::compiled_shaders_ncnn_innerproduct_gemm_comp_spv_len);
  // TODO: Can we avoid this copy? Seems unnecessary
  std::vector<uint32_t> spirv(shader_bytes.begin(), shader_bytes.end());

  /*
  NCNN source
        std::vector<vk_specialization_type> specializations(4 + 10);
        specializations[0].i = bias_term;
        specializations[1].i = activation_type;
        specializations[2].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[4 + 0].i = shape.dims;
        specializations[4 + 1].i = shape.w;
        specializations[4 + 2].i = shape.h;
        specializations[4 + 3].i = shape.c;
        specializations[4 + 4].i = shape.cstep;
        specializations[4 + 5].i = out_shape.dims;
        specializations[4 + 6].i = out_shape.w;
        specializations[4 + 7].i = out_shape.h;
        specializations[4 + 8].i = out_shape.c;
        specializations[4 + 9].i = out_shape.cstep;

        Mat local_size_xyz(std::min(16, num_output / out_elempack), 4, 1, (void*)0);
        if (out_shape.dims != 0)
        {
            local_size_xyz.w = std::min(16, out_shape.w / out_elempack);
            local_size_xyz.h = std::min(4, out_shape.h);
            local_size_xyz.c = 1;
        }

  */

  // assuming ordering is the same as NCNN but have not validated.
  // num_output == M
  kp::Workgroup workgroup({std::min(16U, narrow<uint32_t>(input_info_.shape_B[1])),
                           std::min(4U, narrow<uint32_t>(input_info_.shape_B[0])),
                           1});
  std::vector<float> specializations(14, 0.f);
  specializations[4] = 2.f;
  specializations[5] = narrow<float>(input_info_.shape_A[1]);
  specializations[6] = narrow<float>(input_info_.shape_A[0]);
  specializations[7] = 1.f;
  specializations[8] = narrow<float>(input_info_.shape_A[0] * input_info_.shape_A[1]);
  specializations[9] = 2.f;
  specializations[10] = narrow<float>(input_info_.shape_B[1]);
  specializations[11] = narrow<float>(input_info_.shape_A[0]);
  specializations[12] = 1.f;
  specializations[13] = narrow<float>(num_output_elements);

  // only using static inputs at the moment so all values are set by specializations and none by push constants
  std::vector<float> pushConstsA(14, 0.f);

  std::vector<std::shared_ptr<kp::Tensor>> bindings = {tensorInA, tensorInB, tensorOutA};

  // TODO: This should be created upfront so we're re-using for multiple Run calls. Looks like the intended
  // usage is to call `rebuild` to plugin different inputs
  auto algo = manager.algorithm(bindings, spirv, workgroup, specializations, pushConstsA);

  sequence.record<kp::OpAlgoDispatch>(algo);
}

Status MatMul::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                       /*out*/ bool& is_packed, /*out*/ PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;

  if (input_idx == 1) {
    auto orig_shape = tensor.Shape();

    InlinedVector<size_t> perm{1, 0};
    TensorShapeVector new_dims{orig_shape[1], orig_shape[0]};

    auto packed_b = std::make_unique<Tensor>(tensor.DataType(), TensorShape(new_dims), std::move(alloc));

    SingleAxisTranspose(perm, tensor, *packed_b, /*from*/ 0, /*to*/ 1);

    matmul_kernel_->SetPrepackedB(std::move(packed_b));

    is_packed = true;
  }

  return Status::OK();
}
}  // namespace vulkan
}  // namespace onnxruntime
