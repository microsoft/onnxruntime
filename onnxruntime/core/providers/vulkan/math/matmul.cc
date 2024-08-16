
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/math/matmul.h"

#include "core/common/safeint.h"
#include "core/framework/transpose_helper.h"
#include "core/optimizer/initializer.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/vulkan/vulkan_utils.h"
#include "core/providers/vulkan/shape_utils.h"

#include "core/providers/vulkan/shaders/include/shaderncnn.innerproduct_gemm.hpp"
#include "core/providers/vulkan/shaders/include/shaderncnn.innerproduct_gemm_wp4.hpp"

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
  arg_A = input_defs[0];
  arg_B = input_defs[1];

  constant_A = graph_viewer.IsConstantInitializer(arg_A->Name(), /* check_outer_scope */ true);
  constant_B = graph_viewer.IsConstantInitializer(arg_A->Name(), /* check_outer_scope */ true);

  have_shape_A = GetShape(*arg_A, shape_A, logger);
  have_shape_B = GetShape(*arg_B, shape_B, logger);
}

/* static */
bool MatMulKernel::IsSupported(const GraphViewer& graph_viewer, const onnxruntime::Node& node,
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
                           const GraphViewer& graph_viewer,
                           const onnxruntime::Node& node)
    : VulkanKernel{vulkan_ep, node},
      input_info_{graph_viewer, Node(), Logger()} {
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

void MatMulKernel::ProcessConstantInitializers(const GraphViewer& graph_viewer, kp::Manager& manager,
                                               NodeArgToKpTensorMap& initializers_to_upload) const {
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

Status MatMulKernel::CreateKernel(kp::Manager& manager, NodeArgToKpTensorMap& initializers) {
  // current WIP implementation requires the shape of A to work as it assumes M is known
  ORT_ENFORCE(input_info_.have_shape_A);
  auto dummy_tensor = manager.tensor(std::vector<float>{0.f, 2.f});
  auto tensorInB = initializers.at(input_info_.arg_B);  // might as well use the existing tensor for B

  const auto N = input_info_.shape_A[0];
  const auto M = input_info_.shape_B[1];

  auto num_output_elements = N * M;
  bool use_pack4 = input_info_.have_shape_A && N % 4 == 0 && M % 4 == 0;  // fixed inputs so safe to use pack4
  const unsigned char* spirv_bytes = use_pack4
                                         ? kp::shader_data::compiled_shaders_ncnn_innerproduct_gemm_wp4_comp_spv
                                         : kp::shader_data::compiled_shaders_ncnn_innerproduct_gemm_comp_spv;
  const size_t spirv_num_bytes = use_pack4
                                     ? kp::shader_data::compiled_shaders_ncnn_innerproduct_gemm_wp4_comp_spv_len
                                     : kp::shader_data::compiled_shaders_ncnn_innerproduct_gemm_comp_spv_len;
  assert(spirv_num_bytes % sizeof(uint32_t) == 0);
  const uint32_t* spirv_uint32_data = reinterpret_cast<const uint32_t*>(spirv_bytes);
  const size_t spirv_uint32_num_elements = spirv_num_bytes / sizeof(uint32_t);
  // unnecessary copy to vector as it's copied again into the Algorithm instance.
  // TODO: Update Kompute to take a span
  std::vector<uint32_t> spirv(spirv_uint32_data, spirv_uint32_data + spirv_uint32_num_elements);

  /*
  NCNN source for calling the shader we are using
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

  kp::Workgroup workgroup({std::min(16U, narrow<uint32_t>(M)),
                           std::min(4U, narrow<uint32_t>(N)),
                           1});

  // these are setup to handle the shape of A being unknown but other code isn't
  std::vector<float> specializations(14, 0.f);
  specializations[4] = 2.f;
  specializations[5] = input_info_.have_shape_A ? narrow<float>(input_info_.shape_A[1]) : 0.f;
  specializations[6] = input_info_.have_shape_A ? narrow<float>(input_info_.shape_A[0]) : 0.f;
  specializations[7] = 1.f;
  specializations[8] = input_info_.have_shape_A ? narrow<float>(input_info_.shape_A[0] * input_info_.shape_A[1]) : 0.f;
  specializations[9] = 2.f;
  specializations[10] = narrow<float>(input_info_.shape_B[1]);
  specializations[11] = input_info_.have_shape_A ? narrow<float>(input_info_.shape_A[0]) : 0.f;
  specializations[12] = 1.f;
  specializations[13] = narrow<float>(num_output_elements);

  // only using static inputs at the moment so all values are set by specializations and none by push constants
  std::vector<float> pushConstsA(14, 0.f);

  std::vector<std::shared_ptr<kp::Tensor>> bindings = {dummy_tensor, tensorInB, dummy_tensor};

  kompute_kernel_ = manager.algorithm(bindings, spirv, workgroup, specializations, pushConstsA);

  return Status::OK();
}

Status MatMulKernel::Execute(kp::Manager& manager, kp::Sequence& sequence, NodeArgToKpTensorMap& values) const {
  auto tensorInA = values.at(input_info_.arg_A);
  auto tensorInB = values.at(input_info_.arg_B);

  const auto N = input_info_.shape_A[0];
  const auto M = input_info_.shape_B[1];
  auto num_output_elements = N * M;
  // this is unnecessarily inefficient as it allocates device and staging buffers and copies data into the
  // staging buffer. at the very least that copy is pointless. and if this is an integrated GPU we can potentially
  // use the ORT CPU Tensor buffer as the staging buffer.
  auto tensorOutA = manager.tensor(std::vector<float>(num_output_elements, 0.f));
  values[Node().OutputDefs()[0]] = tensorOutA;

  // only using static inputs at the moment so all values are set by specializations and none by push constants
  // TODO: have base set of push constants which are a copy of the specializations and override values as needed.
  // the NCNN kernel always prefers specialization constants so it doesn't matter if we set the same value in both
  // std::vector<float> push_constants(14, 0.f);

  // update the Algorithm instance to use the new data.
  // NOTE: this is obviously not threadsafe and not clear what would be required to make it so.
  std::vector<std::shared_ptr<kp::Tensor>> bindings = {tensorInA, tensorInB, tensorOutA};
  kompute_kernel_->setTensors(bindings);

  // if the input sizes changed we would also need to call setWorkgroup
  // kp::Workgroup workgroup({std::min(16U, narrow<uint32_t>(M)),
  //                          std::min(4U, narrow<uint32_t>(N)),
  //                          1});
  // kompute_kernel_->setWorkgroup(workgroup);

  sequence.record<kp::OpAlgoDispatch>(kompute_kernel_ /*, push_constants */);

  return Status::OK();
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
