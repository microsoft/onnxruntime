// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_plugin_kernels.h"
#include "cuda_stream_plugin.h"
#include "cuda_kernel_adapter.h"
#include "core/common/narrow.h"
#include "core/providers/cuda/activation/activations.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/math/clip.h"
#include "core/providers/cuda/math/softmax.h"
#include "core/providers/cuda/math/unary_elementwise_ops.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "core/providers/cuda/tensor/concat.h"
#include "core/providers/cuda/tensor/cast_op.h"
#include "core/providers/cuda/tensor/gather.h"
#include "core/providers/cuda/tensor/split.h"
#include "core/providers/cuda/tensor/where.h"
#include "contrib_ops/cuda/bert/decoder_masked_multihead_attention.h"
#include "contrib_ops/cuda/bert/embed_layer_norm.h"
#include "contrib_ops/cuda/bert/fast_gelu.h"
#include "contrib_ops/cuda/bert/gemma_rotary_emb.h"
#include "contrib_ops/cuda/bert/group_query_attention.h"
#include "contrib_ops/cuda/bert/multihead_attention.h"
#include "contrib_ops/cuda/bert/rotary_embedding.h"
#include "contrib_ops/cuda/bert/skip_layer_norm.h"
#include "contrib_ops/cuda/bert/attention.h"
#include "contrib_ops/cuda/moe/moe.h"
#include "contrib_ops/cuda/quantization/gather_block_quantized.h"
#include "contrib_ops/cuda/quantization/matmul_nbits.h"
#include "contrib_ops/cuda/quantization/moe_quantization.h"

#include <cstring>
#include <map>
#include <set>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace onnxruntime {
namespace cuda_plugin {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

CudaSyncStream* GetCudaSyncStream(const Ort::KernelContext& ctx) {
  void* stream = ctx.GetGPUComputeStream();
  if (!stream) return nullptr;
  return CudaSyncStream::FromCudaStream(static_cast<cudaStream_t>(stream));
}

// Resolve mutable aliases by schema names rather than hardcoded indices.
// This keeps alias intent explicit and avoids scattering magic numbers.
void AddMutableAliasesBySchemaName(const std::string& domain,
                                   const std::string& op_type,
                                   int max_inclusive_version,
                                   Ort::KernelDefBuilder& builder) {
  if (!g_host) {
    return;
  }
  const auto* schema = g_host->GetSchema(op_type, max_inclusive_version, domain);
  if (!schema) {
    return;
  }

  std::unordered_map<std::string_view, int> input_index_by_name;
  std::unordered_map<std::string_view, int> output_index_by_name;
  input_index_by_name.reserve(schema->inputs_size());
  output_index_by_name.reserve(schema->outputs_size());

  for (size_t i = 0; i < schema->inputs_size(); ++i) {
    input_index_by_name.emplace(schema->inputs__GetName(i), gsl::narrow_cast<int>(i));
  }
  for (size_t i = 0; i < schema->outputs_size(); ++i) {
    output_index_by_name.emplace(schema->outputs__GetName(i), gsl::narrow_cast<int>(i));
  }

  const auto add_alias = [&](std::string_view input_name, std::string_view output_name) {
    const auto in_it = input_index_by_name.find(input_name);
    const auto out_it = output_index_by_name.find(output_name);
    if (in_it != input_index_by_name.end() && out_it != output_index_by_name.end()) {
      builder.AddInputOutputMutableAlias(in_it->second, out_it->second);
    }
  };

  if (domain == "com.microsoft" && op_type == "GroupQueryAttention") {
    // present_key aliases past_key, present_value aliases past_value.
    add_alias("past_key", "present_key");
    add_alias("past_value", "present_value");
  } else if (domain == "com.microsoft" && op_type == "DecoderMaskedMultiHeadAttention") {
    add_alias("past_key", "present_key");
    add_alias("past_value", "present_value");
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Generic Adapter Kernel — wraps any cuda::CudaKernel-derived class
// ---------------------------------------------------------------------------

template <typename KernelT>
struct AdapterKernelImpl : public OrtKernelImpl {
  std::unique_ptr<KernelT> kernel;

  explicit AdapterKernelImpl(const OrtKernelInfo* info) : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;

    const auto& adapter_info = *reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
    kernel = std::make_unique<KernelT>(adapter_info);
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr,
                                             OrtKernelContext* context) noexcept {
    EXCEPTION_TO_STATUS_BEGIN

    auto* self = static_cast<AdapterKernelImpl*>(this_ptr);
    auto* adapter_ctx = reinterpret_cast<onnxruntime::OpKernelContext*>(context);
    Status status = self->kernel->Compute(adapter_ctx);
    if (!status.IsOK()) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL, status.ErrorMessage().c_str());
    }
    return nullptr;

    EXCEPTION_TO_STATUS_END
  }

  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<AdapterKernelImpl*>(this_ptr);
  }
};

using ReluKernelImpl = AdapterKernelImpl<cuda::Relu<float>>;

// Macro to define a type-dispatching create function for activation ops.
// At kernel creation time, ORT tells us the resolved type via the input tensor.
// We inspect this and dispatch to the right template instantiation.
#define DEFINE_ADAPTER_CREATE_FN_TYPED(OpName)                                             \
  OrtStatus* ORT_API_CALL Create##OpName##Kernel(void* /*state*/,                          \
                                                 const OrtKernelInfo* info,                \
                                                 OrtKernelImpl** kernel_out) noexcept {    \
    EXCEPTION_TO_STATUS_BEGIN                                                              \
    Ort::ConstKernelInfo ki(info);                                                         \
    auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType(); \
    switch (input_type) {                                                                  \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<float>>(info);                    \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:                                          \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<MLFloat16>>(info);                \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<double>>(info);                   \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:                                         \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<BFloat16>>(info);                 \
        break;                                                                             \
      default:                                                                             \
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,                                     \
                                          (std::string(#OpName) + ": unsupported type " +  \
                                           std::to_string(input_type))                     \
                                              .c_str());                                   \
    }                                                                                      \
    return nullptr;                                                                        \
    EXCEPTION_TO_STATUS_END                                                                \
  }

DEFINE_ADAPTER_CREATE_FN_TYPED(Elu)
DEFINE_ADAPTER_CREATE_FN_TYPED(HardSigmoid)
DEFINE_ADAPTER_CREATE_FN_TYPED(HardSwish)
DEFINE_ADAPTER_CREATE_FN_TYPED(LeakyRelu)
DEFINE_ADAPTER_CREATE_FN_TYPED(Selu)
DEFINE_ADAPTER_CREATE_FN_TYPED(Sigmoid)
DEFINE_ADAPTER_CREATE_FN_TYPED(Softplus)
DEFINE_ADAPTER_CREATE_FN_TYPED(Softsign)
DEFINE_ADAPTER_CREATE_FN_TYPED(Tanh)
DEFINE_ADAPTER_CREATE_FN_TYPED(ThresholdedRelu)

#define DEFINE_ADAPTER_CREATE_FN_TYPED_NUMERIC(OpName)                                     \
  OrtStatus* ORT_API_CALL Create##OpName##Kernel(void* /*state*/,                          \
                                                 const OrtKernelInfo* info,                \
                                                 OrtKernelImpl** kernel_out) noexcept {    \
    EXCEPTION_TO_STATUS_BEGIN                                                              \
    Ort::ConstKernelInfo ki(info);                                                         \
    auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType(); \
    switch (input_type) {                                                                  \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<uint8_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<uint16_t>>(info);                 \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<uint32_t>>(info);                 \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<uint64_t>>(info);                 \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:                                             \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int8_t>>(info);                   \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int16_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int32_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int64_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<float>>(info);                    \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:                                          \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<MLFloat16>>(info);                \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<double>>(info);                   \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:                                         \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<BFloat16>>(info);                 \
        break;                                                                             \
      default:                                                                             \
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,                                     \
                                          (std::string(#OpName) + ": unsupported type " +  \
                                           std::to_string(input_type))                     \
                                              .c_str());                                   \
    }                                                                                      \
    return nullptr;                                                                        \
    EXCEPTION_TO_STATUS_END                                                                \
  }

#define DEFINE_ADAPTER_CREATE_FN_TYPED_NUMERIC_OR_BOOL(OpName)                             \
  OrtStatus* ORT_API_CALL Create##OpName##Kernel(void* /*state*/,                          \
                                                 const OrtKernelInfo* info,                \
                                                 OrtKernelImpl** kernel_out) noexcept {    \
    EXCEPTION_TO_STATUS_BEGIN                                                              \
    Ort::ConstKernelInfo ki(info);                                                         \
    auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType(); \
    switch (input_type) {                                                                  \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<uint8_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<uint16_t>>(info);                 \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<uint32_t>>(info);                 \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<uint64_t>>(info);                 \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:                                             \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int8_t>>(info);                   \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int16_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int32_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int64_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<float>>(info);                    \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:                                          \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<MLFloat16>>(info);                \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<double>>(info);                   \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:                                         \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<BFloat16>>(info);                 \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:                                             \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<bool>>(info);                     \
        break;                                                                             \
      default:                                                                             \
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,                                     \
                                          (std::string(#OpName) + ": unsupported type " +  \
                                           std::to_string(input_type))                     \
                                              .c_str());                                   \
    }                                                                                      \
    return nullptr;                                                                        \
    EXCEPTION_TO_STATUS_END                                                                \
  }

#define DEFINE_ADAPTER_CREATE_FN_TYPED_SIGNED_NUMERIC(OpName)                              \
  OrtStatus* ORT_API_CALL Create##OpName##Kernel(void* /*state*/,                          \
                                                 const OrtKernelInfo* info,                \
                                                 OrtKernelImpl** kernel_out) noexcept {    \
    EXCEPTION_TO_STATUS_BEGIN                                                              \
    Ort::ConstKernelInfo ki(info);                                                         \
    auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType(); \
    switch (input_type) {                                                                  \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:                                             \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int8_t>>(info);                   \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int16_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int32_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int64_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<float>>(info);                    \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:                                          \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<MLFloat16>>(info);                \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<double>>(info);                   \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:                                         \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<BFloat16>>(info);                 \
        break;                                                                             \
      default:                                                                             \
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,                                     \
                                          (std::string(#OpName) + ": unsupported type " +  \
                                           std::to_string(input_type))                     \
                                              .c_str());                                   \
    }                                                                                      \
    return nullptr;                                                                        \
    EXCEPTION_TO_STATUS_END                                                                \
  }

#define DEFINE_ADAPTER_CREATE_FN_TYPED_HFD(OpName)                                         \
  OrtStatus* ORT_API_CALL Create##OpName##Kernel(void* /*state*/,                          \
                                                 const OrtKernelInfo* info,                \
                                                 OrtKernelImpl** kernel_out) noexcept {    \
    EXCEPTION_TO_STATUS_BEGIN                                                              \
    Ort::ConstKernelInfo ki(info);                                                         \
    auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType(); \
    switch (input_type) {                                                                  \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<float>>(info);                    \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:                                          \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<MLFloat16>>(info);                \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<double>>(info);                   \
        break;                                                                             \
      default:                                                                             \
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,                                     \
                                          (std::string(#OpName) + ": unsupported type " +  \
                                           std::to_string(input_type))                     \
                                              .c_str());                                   \
    }                                                                                      \
    return nullptr;                                                                        \
    EXCEPTION_TO_STATUS_END                                                                \
  }

#define DEFINE_ADAPTER_CREATE_FN_TYPED_HFDX(OpName)                                        \
  OrtStatus* ORT_API_CALL Create##OpName##Kernel(void* /*state*/,                          \
                                                 const OrtKernelInfo* info,                \
                                                 OrtKernelImpl** kernel_out) noexcept {    \
    EXCEPTION_TO_STATUS_BEGIN                                                              \
    Ort::ConstKernelInfo ki(info);                                                         \
    auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType(); \
    switch (input_type) {                                                                  \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<float>>(info);                    \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:                                          \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<MLFloat16>>(info);                \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<double>>(info);                   \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:                                         \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<BFloat16>>(info);                 \
        break;                                                                             \
      default:                                                                             \
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,                                     \
                                          (std::string(#OpName) + ": unsupported type " +  \
                                           std::to_string(input_type))                     \
                                              .c_str());                                   \
    }                                                                                      \
    return nullptr;                                                                        \
    EXCEPTION_TO_STATUS_END                                                                \
  }

#define DEFINE_ADAPTER_CREATE_FN_TYPED_I8U8I32I64HFD(OpName)                               \
  OrtStatus* ORT_API_CALL Create##OpName##Kernel(void* /*state*/,                          \
                                                 const OrtKernelInfo* info,                \
                                                 OrtKernelImpl** kernel_out) noexcept {    \
    EXCEPTION_TO_STATUS_BEGIN                                                              \
    Ort::ConstKernelInfo ki(info);                                                         \
    auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType(); \
    switch (input_type) {                                                                  \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:                                             \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int8_t>>(info);                   \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<uint8_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int32_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int64_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<float>>(info);                    \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:                                          \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<MLFloat16>>(info);                \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<double>>(info);                   \
        break;                                                                             \
      default:                                                                             \
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,                                     \
                                          (std::string(#OpName) + ": unsupported type " +  \
                                           std::to_string(input_type))                     \
                                              .c_str());                                   \
    }                                                                                      \
    return nullptr;                                                                        \
    EXCEPTION_TO_STATUS_END                                                                \
  }

#define DEFINE_ADAPTER_CREATE_FN_TYPED_I32I64HFDX(OpName)                                  \
  OrtStatus* ORT_API_CALL Create##OpName##Kernel(void* /*state*/,                          \
                                                 const OrtKernelInfo* info,                \
                                                 OrtKernelImpl** kernel_out) noexcept {    \
    EXCEPTION_TO_STATUS_BEGIN                                                              \
    Ort::ConstKernelInfo ki(info);                                                         \
    auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType(); \
    switch (input_type) {                                                                  \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int32_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int64_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<float>>(info);                    \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:                                          \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<MLFloat16>>(info);                \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<double>>(info);                   \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:                                         \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<BFloat16>>(info);                 \
        break;                                                                             \
      default:                                                                             \
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,                                     \
                                          (std::string(#OpName) + ": unsupported type " +  \
                                           std::to_string(input_type))                     \
                                              .c_str());                                   \
    }                                                                                      \
    return nullptr;                                                                        \
    EXCEPTION_TO_STATUS_END                                                                \
  }

#define DEFINE_ADAPTER_CREATE_FN_TYPED_I32HFDX(OpName)                                     \
  OrtStatus* ORT_API_CALL Create##OpName##Kernel(void* /*state*/,                          \
                                                 const OrtKernelInfo* info,                \
                                                 OrtKernelImpl** kernel_out) noexcept {    \
    EXCEPTION_TO_STATUS_BEGIN                                                              \
    Ort::ConstKernelInfo ki(info);                                                         \
    auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType(); \
    switch (input_type) {                                                                  \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<int32_t>>(info);                  \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:                                            \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<float>>(info);                    \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:                                          \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<MLFloat16>>(info);                \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:                                           \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<double>>(info);                   \
        break;                                                                             \
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:                                         \
        *kernel_out = new AdapterKernelImpl<cuda::OpName<BFloat16>>(info);                 \
        break;                                                                             \
      default:                                                                             \
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL,                                     \
                                          (std::string(#OpName) + ": unsupported type " +  \
                                           std::to_string(input_type))                     \
                                              .c_str());                                   \
    }                                                                                      \
    return nullptr;                                                                        \
    EXCEPTION_TO_STATUS_END                                                                \
  }

DEFINE_ADAPTER_CREATE_FN_TYPED_NUMERIC(Sub)
DEFINE_ADAPTER_CREATE_FN_TYPED_NUMERIC(Mul)
DEFINE_ADAPTER_CREATE_FN_TYPED_NUMERIC(Div)
OrtStatus* ORT_API_CALL CreateEqualKernel(void* /*state*/,
                                          const OrtKernelInfo* info,
                                          OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      *kernel_out = new AdapterKernelImpl<cuda::Equal<uint32_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      *kernel_out = new AdapterKernelImpl<cuda::Equal<uint64_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      *kernel_out = new AdapterKernelImpl<cuda::Equal<int32_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      *kernel_out = new AdapterKernelImpl<cuda::Equal<int64_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<cuda::Equal<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Equal<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      *kernel_out = new AdapterKernelImpl<cuda::Equal<double>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Equal<BFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      *kernel_out = new AdapterKernelImpl<cuda::Equal<bool>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        (std::string("Equal: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateGreaterKernel(void* /*state*/,
                                            const OrtKernelInfo* info,
                                            OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      *kernel_out = new AdapterKernelImpl<cuda::Greater<uint32_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      *kernel_out = new AdapterKernelImpl<cuda::Greater<uint64_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      *kernel_out = new AdapterKernelImpl<cuda::Greater<int32_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      *kernel_out = new AdapterKernelImpl<cuda::Greater<int64_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<cuda::Greater<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Greater<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      *kernel_out = new AdapterKernelImpl<cuda::Greater<double>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Greater<BFloat16>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        (std::string("Greater: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateLessKernel(void* /*state*/,
                                         const OrtKernelInfo* info,
                                         OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      *kernel_out = new AdapterKernelImpl<cuda::Less<uint32_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      *kernel_out = new AdapterKernelImpl<cuda::Less<uint64_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      *kernel_out = new AdapterKernelImpl<cuda::Less<int32_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      *kernel_out = new AdapterKernelImpl<cuda::Less<int64_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<cuda::Less<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Less<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      *kernel_out = new AdapterKernelImpl<cuda::Less<double>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Less<BFloat16>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        (std::string("Less: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

DEFINE_ADAPTER_CREATE_FN_TYPED_NUMERIC(Abs)
DEFINE_ADAPTER_CREATE_FN_TYPED_SIGNED_NUMERIC(Neg)
DEFINE_ADAPTER_CREATE_FN_TYPED_HFD(Floor)
DEFINE_ADAPTER_CREATE_FN_TYPED_HFD(Ceil)
DEFINE_ADAPTER_CREATE_FN_TYPED_HFDX(Sqrt)
DEFINE_ADAPTER_CREATE_FN_TYPED_HFDX(Exp)
DEFINE_ADAPTER_CREATE_FN_TYPED_HFD(Log)
DEFINE_ADAPTER_CREATE_FN_TYPED_HFD(ArgMax)
DEFINE_ADAPTER_CREATE_FN_TYPED_HFD(ArgMin)
DEFINE_ADAPTER_CREATE_FN_TYPED_I8U8I32I64HFD(ReduceMax)
DEFINE_ADAPTER_CREATE_FN_TYPED_I32HFDX(ReduceMean)
DEFINE_ADAPTER_CREATE_FN_TYPED_I8U8I32I64HFD(ReduceMin)
DEFINE_ADAPTER_CREATE_FN_TYPED_I32I64HFDX(ReduceSum)
DEFINE_ADAPTER_CREATE_FN_TYPED_HFDX(Softmax)

OrtStatus* ORT_API_CALL CreateLogSoftmaxKernel(void* /*state*/,
                                               const OrtKernelInfo* info,
                                               OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<cuda::Softmax<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Softmax<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      *kernel_out = new AdapterKernelImpl<cuda::Softmax<double>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Softmax<BFloat16>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        (std::string("LogSoftmax: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

// ---------------------------------------------------------------------------
// Add Kernel Implementation
// ---------------------------------------------------------------------------

struct AddKernelImpl : public OrtKernelImpl {
  AddKernelImpl() : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    flags = 0;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr,
                                             OrtKernelContext* context) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<AddKernelImpl*>(this_ptr);
  }
};

__global__ void AddKernelCuda(const float* a, const float* b, float* c, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    c[idx] = a[idx] + b[idx];
  }
}

/*static*/
OrtStatus* ORT_API_CALL AddKernelImpl::ComputeImpl(
    OrtKernelImpl* /*this_ptr*/, OrtKernelContext* context) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  Ort::KernelContext ctx{context};
  Ort::ConstValue input_a = ctx.GetInput(0);
  Ort::ConstValue input_b = ctx.GetInput(1);

  auto shape_info = input_a.GetTensorTypeAndShapeInfo();
  auto shape = shape_info.GetShape();
  size_t count = shape_info.GetElementCount();

  Ort::UnownedValue output = ctx.GetOutput(0, shape);

  const float* a_data = input_a.GetTensorData<float>();
  const float* b_data = input_b.GetTensorData<float>();
  float* c_data = output.GetTensorMutableData<float>();

  if (count > 0) {
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.GetGPUComputeStream());

    const int block_size = 256;
    const int grid_size = static_cast<int>((count + block_size - 1) / block_size);
    AddKernelCuda<<<grid_size, block_size, 0, stream>>>(a_data, b_data, c_data, count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, cudaGetErrorString(err));
    }
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

// ---------------------------------------------------------------------------
// MatMul Kernel Implementation
// ---------------------------------------------------------------------------

struct MatMulKernelImpl : public OrtKernelImpl {
  MatMulKernelImpl() : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    flags = 0;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr,
                                             OrtKernelContext* context) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<MatMulKernelImpl*>(this_ptr);
  }
};

/*static*/
OrtStatus* ORT_API_CALL MatMulKernelImpl::ComputeImpl(
    OrtKernelImpl* /*this_ptr*/, OrtKernelContext* context) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  Ort::KernelContext ctx{context};
  Ort::ConstValue input_a = ctx.GetInput(0);
  Ort::ConstValue input_b = ctx.GetInput(1);

  auto shape_info_a = input_a.GetTensorTypeAndShapeInfo();
  auto shape_info_b = input_b.GetTensorTypeAndShapeInfo();
  auto shape_a = shape_info_a.GetShape();
  auto shape_b = shape_info_b.GetShape();

  // MatMul: [M, K] x [K, N] -> [M, N]
  int M = static_cast<int>(shape_a[0]);
  int K = static_cast<int>(shape_a[1]);
  int N = static_cast<int>(shape_b[1]);

  std::vector<int64_t> output_shape = {M, N};
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);

  const float* a_data = input_a.GetTensorData<float>();
  const float* b_data = input_b.GetTensorData<float>();
  float* y_data = output.GetTensorMutableData<float>();

  if (M > 0 && N > 0 && K > 0) {
    CudaSyncStream* stream_impl = GetCudaSyncStream(ctx);
    if (!stream_impl) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "Failed to get CUDA stream");
    }

    cublasHandle_t cublas_handle = stream_impl->GetCublasHandle();

    float alpha = 1.0f;
    float beta = 0.0f;

    PL_CUBLAS_RETURN_IF_ERROR(cublasSgemm(cublas_handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          N, M, K,
                                          &alpha,
                                          b_data, N,
                                          a_data, K,
                                          &beta,
                                          y_data, N));
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

// ---------------------------------------------------------------------------
// Gemm Kernel Implementation
// ---------------------------------------------------------------------------

struct GemmKernelImpl : public OrtKernelImpl {
  GemmKernelImpl(const OrtKernelInfo* info) : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    flags = 0;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;

    OrtStatus* status = Ort::GetApi().KernelInfoGetAttribute_float(info, "alpha", &alpha_);
    if (status != nullptr) {
      alpha_ = 1.0f;
      Ort::GetApi().ReleaseStatus(status);
    }
    status = Ort::GetApi().KernelInfoGetAttribute_float(info, "beta", &beta_);
    if (status != nullptr) {
      beta_ = 1.0f;
      Ort::GetApi().ReleaseStatus(status);
    }
    int64_t tA = 0;
    status = Ort::GetApi().KernelInfoGetAttribute_int64(info, "transA", &tA);
    if (status != nullptr) {
      trans_a_ = 0;
      Ort::GetApi().ReleaseStatus(status);
    } else {
      trans_a_ = static_cast<int>(tA);
    }
    int64_t tB = 0;
    status = Ort::GetApi().KernelInfoGetAttribute_int64(info, "transB", &tB);
    if (status != nullptr) {
      trans_b_ = 0;
      Ort::GetApi().ReleaseStatus(status);
    } else {
      trans_b_ = static_cast<int>(tB);
    }
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr,
                                             OrtKernelContext* context) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<GemmKernelImpl*>(this_ptr);
  }

 private:
  float alpha_;
  float beta_;
  int trans_a_;
  int trans_b_;
};

__global__ void BroadcastBiasKernel(const float* c, float* y, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  if (idx < total) {
    int col = idx % N;
    y[idx] = c[col];
  }
}

/*static*/
OrtStatus* ORT_API_CALL GemmKernelImpl::ComputeImpl(
    OrtKernelImpl* this_ptr, OrtKernelContext* context) noexcept {
  auto* self = static_cast<GemmKernelImpl*>(this_ptr);
  EXCEPTION_TO_STATUS_BEGIN
  Ort::KernelContext ctx{context};
  Ort::ConstValue input_a = ctx.GetInput(0);
  Ort::ConstValue input_b = ctx.GetInput(1);

  auto shape_a = input_a.GetTensorTypeAndShapeInfo().GetShape();
  auto shape_b = input_b.GetTensorTypeAndShapeInfo().GetShape();

  int M = static_cast<int>(self->trans_a_ == 0 ? shape_a[0] : shape_a[1]);
  int K = static_cast<int>(self->trans_a_ == 0 ? shape_a[1] : shape_a[0]);
  int N = static_cast<int>(self->trans_b_ == 0 ? shape_b[1] : shape_b[0]);

  std::vector<int64_t> output_shape = {M, N};
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);

  const float* a_data = input_a.GetTensorData<float>();
  const float* b_data = input_b.GetTensorData<float>();
  float* y_data = output.GetTensorMutableData<float>();

  if (M > 0 && N > 0 && K > 0) {
    CudaSyncStream* stream_impl = GetCudaSyncStream(ctx);
    if (!stream_impl) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "Failed to get CUDA stream");
    }

    cublasHandle_t cublas_handle = stream_impl->GetCublasHandle();

    // Handle optional bias C
    if (ctx.GetInputCount() > 2) {
      Ort::ConstValue input_c = ctx.GetInput(2);
      const float* c_data = input_c.GetTensorData<float>();
      // Copy C to output initially if beta != 0
      if (self->beta_ != 0.0f) {
        // Gemm spec says C can be scalar, [N], [M, 1] or [M, N].
        // For now we assume [M, N] or [N] (broadcast row).
        // To simplify, we'll just support [M, N] or broadcast manually if needed.
        // cuBLAS sgemm does Y = alpha*op(A)*op(B) + beta*C.
        // If we want to use the output as C, we must initialize it with C data.
        auto shape_c = input_c.GetTensorTypeAndShapeInfo().GetShape();
        if (shape_c.size() == 2 && shape_c[0] == M && shape_c[1] == N) {
          PL_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(y_data, c_data, M * N * sizeof(float), cudaMemcpyDeviceToDevice, stream_impl->GetCudaStream()));
        } else if (shape_c.size() == 1 && shape_c[0] == N) {
          // Broadcast [N] to [M, N]
          int total = M * N;
          int threads = 256;
          int blocks = (total + threads - 1) / threads;
          BroadcastBiasKernel<<<blocks, threads, 0, stream_impl->GetCudaStream()>>>(c_data, y_data, M, N);
          PL_CUDA_RETURN_IF_ERROR(cudaGetLastError());
        } else {
          // Fallback - just zero if unsupported broadcast
          PL_CUDA_RETURN_IF_ERROR(cudaMemsetAsync(y_data, 0, M * N * sizeof(float), stream_impl->GetCudaStream()));
        }
      } else {
        PL_CUDA_RETURN_IF_ERROR(cudaMemsetAsync(y_data, 0, M * N * sizeof(float), stream_impl->GetCudaStream()));
      }
    } else {
      PL_CUDA_RETURN_IF_ERROR(cudaMemsetAsync(y_data, 0, M * N * sizeof(float), stream_impl->GetCudaStream()));
    }

    cublasOperation_t transA = self->trans_a_ == 0 ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transB = self->trans_b_ == 0 ? CUBLAS_OP_N : CUBLAS_OP_T;

    // Row-major A[M,K], B[K,N] -> C[M,N]
    // cuBLAS (col-major): C = alpha * op(B) * op(A) + beta * C
    // op(B) is [N, K] in col-major (if transB=0) or [K, N] (if transB=1)
    // op(A) is [K, M] in col-major (if transA=0) or [M, K] (if transA=1)

    int lda = (self->trans_a_ == 0) ? K : M;
    int ldb = (self->trans_b_ == 0) ? N : K;
    int ldc = N;

    PL_CUBLAS_RETURN_IF_ERROR(cublasSgemm(cublas_handle,
                                          transB, transA,
                                          N, M, K,
                                          &self->alpha_,
                                          b_data, ldb,
                                          a_data, lda,
                                          &self->beta_,
                                          y_data, ldc));
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

// ---------------------------------------------------------------------------
// Conv Kernel Implementation
// ---------------------------------------------------------------------------

struct ConvKernelImpl : public OrtKernelImpl {
  ConvKernelImpl(const OrtKernelInfo* info) : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    flags = 0;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;

    Ort::ConstKernelInfo k_info{info};
    try {
      pads_ = k_info.GetAttributes<int64_t>("pads");
    } catch (...) {
      pads_ = {0, 0, 0, 0};
    }
    try {
      strides_ = k_info.GetAttributes<int64_t>("strides");
    } catch (...) {
      strides_ = {1, 1};
    }
    try {
      dilations_ = k_info.GetAttributes<int64_t>("dilations");
    } catch (...) {
      dilations_ = {1, 1};
    }
    try {
      group_ = k_info.GetAttribute<int64_t>("group");
    } catch (...) {
      group_ = 1;
    }

    cudnnCreateTensorDescriptor(&x_desc_);
    cudnnCreateTensorDescriptor(&y_desc_);
    cudnnCreateFilterDescriptor(&w_desc_);
    cudnnCreateConvolutionDescriptor(&conv_desc_);
  }

  ~ConvKernelImpl() {
    cudnnDestroyTensorDescriptor(x_desc_);
    cudnnDestroyTensorDescriptor(y_desc_);
    cudnnDestroyFilterDescriptor(w_desc_);
    cudnnDestroyConvolutionDescriptor(conv_desc_);
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr,
                                             OrtKernelContext* context) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<ConvKernelImpl*>(this_ptr);
  }

 private:
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> dilations_;
  int64_t group_;

  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  cudnnFilterDescriptor_t w_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
};

/*static*/
OrtStatus* ORT_API_CALL ConvKernelImpl::ComputeImpl(
    OrtKernelImpl* this_ptr, OrtKernelContext* context) noexcept {
  auto* self = static_cast<ConvKernelImpl*>(this_ptr);
  EXCEPTION_TO_STATUS_BEGIN

  Ort::KernelContext ctx{context};
  Ort::ConstValue input_x = ctx.GetInput(0);
  Ort::ConstValue input_w = ctx.GetInput(1);

  auto shape_x = input_x.GetTensorTypeAndShapeInfo().GetShape();
  auto shape_w = input_w.GetTensorTypeAndShapeInfo().GetShape();

  int n = static_cast<int>(shape_x[0]);
  int c = static_cast<int>(shape_x[1]);
  int h = static_cast<int>(shape_x.size() > 2 ? shape_x[2] : 1);
  int w = static_cast<int>(shape_x.size() > 3 ? shape_x[3] : 1);

  int m = static_cast<int>(shape_w[0]);
  int wc = static_cast<int>(shape_w[1]);
  int kh = static_cast<int>(shape_w.size() > 2 ? shape_w[2] : 1);
  int kw = static_cast<int>(shape_w.size() > 3 ? shape_w[3] : 1);

  int out_h = static_cast<int>((h + self->pads_[0] + self->pads_[2] - self->dilations_[0] * (kh - 1) - 1) / self->strides_[0] + 1);
  int out_w = static_cast<int>((w + self->pads_[1] + self->pads_[3] - self->dilations_[1] * (kw - 1) - 1) / self->strides_[1] + 1);

  std::vector<int64_t> output_shape = {n, m, out_h, out_w};
  Ort::UnownedValue output_y = ctx.GetOutput(0, output_shape);

  CudaSyncStream* stream_impl = GetCudaSyncStream(ctx);
  if (!stream_impl) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "Failed to get CUDA stream");
  }

  cudnnHandle_t cudnn = stream_impl->GetCudnnHandle();

  cudnnSetTensor4dDescriptor(self->x_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
  cudnnSetTensor4dDescriptor(self->y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, m, out_h, out_w);
  cudnnSetFilter4dDescriptor(self->w_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, m, wc, kh, kw);

  cudnnSetConvolution2dDescriptor(self->conv_desc_,
                                  static_cast<int>(self->pads_[0]), static_cast<int>(self->pads_[1]),
                                  static_cast<int>(self->strides_[0]), static_cast<int>(self->strides_[1]),
                                  static_cast<int>(self->dilations_[0]), static_cast<int>(self->dilations_[1]),
                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
  cudnnSetConvolutionGroupCount(self->conv_desc_, static_cast<int>(self->group_));

  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  size_t workspace_size = 0;
  cudnnGetConvolutionForwardWorkspaceSize(cudnn, self->x_desc_, self->w_desc_, self->conv_desc_, self->y_desc_, algo, &workspace_size);

  void* workspace = nullptr;
  if (workspace_size > 0) {
    cudaMallocAsync(&workspace, workspace_size, stream_impl->GetCudaStream());
  }

  const float alpha = 1.0f, beta = 0.0f;
  const float* x_data = input_x.GetTensorData<float>();
  const float* w_data = input_w.GetTensorData<float>();
  float* y_data = output_y.GetTensorMutableData<float>();

  cudnnConvolutionForward(cudnn, &alpha, self->x_desc_, x_data,
                          self->w_desc_, w_data, self->conv_desc_, algo,
                          workspace, workspace_size, &beta, self->y_desc_, y_data);

  if (workspace_size > 0) {
    cudaFreeAsync(workspace, stream_impl->GetCudaStream());
  }

  if (ctx.GetInputCount() > 2) {
    Ort::ConstValue input_b = ctx.GetInput(2);
    const float* b_data = input_b.GetTensorData<float>();
    cudnnTensorDescriptor_t b_desc;
    cudnnCreateTensorDescriptor(&b_desc);
    cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, m, 1, 1);
    const float alpha_b = 1.0f, beta_b = 1.0f;
    cudnnAddTensor(cudnn, &alpha_b, b_desc, b_data, &beta_b, self->y_desc_, y_data);
    cudnnDestroyTensorDescriptor(b_desc);
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

namespace {

struct GeneratedKernelRegistration {
  const char* op_type;
  int since_version_start;
  int since_version_end;
  const char* domain;
  int registration_id;
  const char* constraint_name;
  ONNXTensorElementDataType type_constraint;
  int input_mem_type_indices[4];
  int num_input_mem_type_indices;
  int output_mem_type_indices[4];
  int num_output_mem_type_indices;
};

using PluginKernelCreateFn = OrtStatus*(ORT_API_CALL*)(void*, const OrtKernelInfo*, OrtKernelImpl**) noexcept;

OrtStatus* ORT_API_CALL CreateReluKernel(void* /*state*/,
                                         const OrtKernelInfo* info,
                                         OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<cuda::Relu<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Relu<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      *kernel_out = new AdapterKernelImpl<cuda::Relu<double>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Relu<BFloat16>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        (std::string("Relu: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateAddKernel(void* /*state*/,
                                        const OrtKernelInfo* info,
                                        OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      *kernel_out = new AdapterKernelImpl<cuda::Add<uint8_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      *kernel_out = new AdapterKernelImpl<cuda::Add<uint16_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      *kernel_out = new AdapterKernelImpl<cuda::Add<uint32_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      *kernel_out = new AdapterKernelImpl<cuda::Add<uint64_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      *kernel_out = new AdapterKernelImpl<cuda::Add<int8_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      *kernel_out = new AdapterKernelImpl<cuda::Add<int16_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      *kernel_out = new AdapterKernelImpl<cuda::Add<int32_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      *kernel_out = new AdapterKernelImpl<cuda::Add<int64_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<cuda::Add<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Add<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      *kernel_out = new AdapterKernelImpl<cuda::Add<double>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Add<BFloat16>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("Add: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateMatMulKernel(void* /*state*/,
                                           const OrtKernelInfo* info,
                                           OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    *kernel_out = new MatMulKernelImpl();
    return nullptr;
  }
  return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "MatMulKernelImpl (Stage 1 example) only supports float.");
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateGemmKernel(void* /*state*/,
                                         const OrtKernelInfo* info,
                                         OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    *kernel_out = new GemmKernelImpl(info);
    return nullptr;
  }
  return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "GemmKernelImpl (Stage 1 example) only supports float.");
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateConvKernel(void* /*state*/,
                                         const OrtKernelInfo* info,
                                         OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    *kernel_out = new ConvKernelImpl(info);
    return nullptr;
  }
  return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "ConvKernelImpl (Stage 1 example) only supports float.");
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreatePowKernel(void* /*state*/,
                                        const OrtKernelInfo* info,
                                        OrtKernelImpl** kernel_out) noexcept {
  *kernel_out = new AdapterKernelImpl<cuda::Pow>(info);
  return nullptr;
}

OrtStatus* ORT_API_CALL CreateClipKernel(void* /*state*/,
                                         const OrtKernelInfo* info,
                                         OrtKernelImpl** kernel_out) noexcept {
  float min_attr = 0.0f;
  float max_attr = 0.0f;
  OrtStatus* min_status = Ort::GetApi().KernelInfoGetAttribute_float(info, "min", &min_attr);
  OrtStatus* max_status = Ort::GetApi().KernelInfoGetAttribute_float(info, "max", &max_attr);
  const bool has_min_or_max_attr = (min_status == nullptr || max_status == nullptr);
  if (min_status) Ort::GetApi().ReleaseStatus(min_status);
  if (max_status) Ort::GetApi().ReleaseStatus(max_status);

  if (has_min_or_max_attr) {
    *kernel_out = new AdapterKernelImpl<cuda::Clip_6<float>>(info);
  } else {
    *kernel_out = new AdapterKernelImpl<cuda::Clip>(info);
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL CreateWhereKernel(void* /*state*/,
                                          const OrtKernelInfo* info,
                                          OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto x_type = ki.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetElementType();
  switch (x_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      *kernel_out = new AdapterKernelImpl<cuda::Where<uint8_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      *kernel_out = new AdapterKernelImpl<cuda::Where<int32_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      *kernel_out = new AdapterKernelImpl<cuda::Where<int64_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<cuda::Where<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      *kernel_out = new AdapterKernelImpl<cuda::Where<double>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Where<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Where<BFloat16>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        (std::string("Where: unsupported type ") + std::to_string(x_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateCastKernel(void* /*state*/,
                                         const OrtKernelInfo* info,
                                         OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<BFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<double>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<int8_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<int16_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<int32_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<int64_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<uint8_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<uint16_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<uint32_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<uint64_t>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<bool>>(info);
      break;
#if !defined(DISABLE_FLOAT8_TYPES)
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<Float8E4M3FN>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<Float8E5M2>>(info);
      break;
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1:
      *kernel_out = new AdapterKernelImpl<cuda::Cast<Float4E2M1x2>>(info);
      break;
#endif
    default:
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL,
                                        (std::string("Cast: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateConcatKernel(void* /*state*/,
                                           const OrtKernelInfo* info,
                                           OrtKernelImpl** kernel_out) noexcept {
  *kernel_out = new AdapterKernelImpl<cuda::Concat>(info);
  return nullptr;
}

OrtStatus* ORT_API_CALL CreateSplitKernel(void* /*state*/,
                                          const OrtKernelInfo* info,
                                          OrtKernelImpl** kernel_out) noexcept {
  int64_t num_outputs = -1;
  OrtStatus* status = Ort::GetApi().KernelInfoGetAttribute_int64(info, "num_outputs", &num_outputs);
  if (status == nullptr) {
    *kernel_out = new AdapterKernelImpl<cuda::Split_18>(info);
  } else {
    Ort::GetApi().ReleaseStatus(status);
    *kernel_out = new AdapterKernelImpl<cuda::Split_2_13>(info);
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL CreateGatherKernel(void* /*state*/,
                                           const OrtKernelInfo* info,
                                           OrtKernelImpl** kernel_out) noexcept {
  *kernel_out = new AdapterKernelImpl<cuda::Gather>(info);
  return nullptr;
}

// ---------------------------------------------------------------------------
// Shape-Only Kernels (Reshape, Squeeze, Unsqueeze, Flatten)
// These ops only change tensor shape metadata. On the GPU, we copy data from
// input to output using cudaMemcpyAsync since ORT may allocate a new buffer
// for the output with the target shape.
// ---------------------------------------------------------------------------

// Helper: compute output shape for Reshape from the shape tensor (input 1).
// Handles -1 (infer) and 0 (copy from input) dimensions per ONNX spec.
static std::vector<int64_t> ComputeReshapeOutputShape(
    const std::vector<int64_t>& input_dims,
    const int64_t* shape_data, size_t shape_len) {
  int64_t input_size = 1;
  for (auto d : input_dims) input_size *= d;

  std::vector<int64_t> output_dims(shape_data, shape_data + shape_len);
  int64_t inferred_idx = -1;
  int64_t known_size = 1;
  for (size_t i = 0; i < output_dims.size(); ++i) {
    if (output_dims[i] == 0 && i < input_dims.size()) {
      output_dims[i] = input_dims[i];
    }
    if (output_dims[i] == -1) {
      inferred_idx = static_cast<int64_t>(i);
    } else {
      known_size *= output_dims[i];
    }
  }
  if (inferred_idx >= 0 && known_size > 0) {
    output_dims[inferred_idx] = input_size / known_size;
  }
  return output_dims;
}

// Generic shape-copy compute: copy raw bytes from input to output.
static OrtStatus* ShapeCopyCompute(OrtKernelContext* context,
                                   const std::vector<int64_t>& output_shape) {
  Ort::KernelContext ctx{context};
  Ort::ConstValue input = ctx.GetInput(0);
  auto info = input.GetTensorTypeAndShapeInfo();
  size_t elem_count = info.GetElementCount();

  // Determine element size from type
  size_t elem_size = 0;
  auto type = info.GetElementType();
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      elem_size = 4;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      elem_size = 8;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      elem_size = 2;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      elem_size = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      elem_size = 2;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      elem_size = 4;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      elem_size = 8;
      break;
    default:
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("ShapeCopyCompute: unsupported element type ") +
           std::to_string(static_cast<int>(type)))
              .c_str());
  }

  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);

  if (elem_count > 0) {
    const void* src = input.GetTensorRawData();
    void* dst = output.GetTensorMutableRawData();
    if (src != dst) {
      cudaStream_t stream = static_cast<cudaStream_t>(ctx.GetGPUComputeStream());
      cudaError_t err = cudaMemcpyAsync(dst, src, elem_count * elem_size,
                                        cudaMemcpyDeviceToDevice, stream);
      if (err != cudaSuccess) {
        return Ort::GetApi().CreateStatus(ORT_EP_FAIL, cudaGetErrorString(err));
      }
    }
  }
  return nullptr;
}

// Reshape kernel: reads target shape from input 1 (CPU tensor).
struct ReshapeKernelImpl : public OrtKernelImpl {
  ReshapeKernelImpl() : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    flags = 0;
    Compute = ComputeImpl;
    Release = [](OrtKernelImpl* p) noexcept { delete static_cast<ReshapeKernelImpl*>(p); };
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl*, OrtKernelContext* context) noexcept {
    EXCEPTION_TO_STATUS_BEGIN
    Ort::KernelContext ctx{context};
    Ort::ConstValue input = ctx.GetInput(0);
    Ort::ConstValue shape_tensor = ctx.GetInput(1);
    auto input_info = input.GetTensorTypeAndShapeInfo();
    auto shape_info = shape_tensor.GetTensorTypeAndShapeInfo();
    const int64_t* shape_data = shape_tensor.GetTensorData<int64_t>();
    size_t shape_len = shape_info.GetElementCount();
    auto output_shape = ComputeReshapeOutputShape(
        input_info.GetShape(), shape_data, shape_len);
    return ShapeCopyCompute(context, output_shape);
    EXCEPTION_TO_STATUS_END
  }
};

// Squeeze kernel: removes dimensions of size 1.
struct SqueezeKernelImpl : public OrtKernelImpl {
  SqueezeKernelImpl() : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    flags = 0;
    Compute = ComputeImpl;
    Release = [](OrtKernelImpl* p) noexcept { delete static_cast<SqueezeKernelImpl*>(p); };
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl*, OrtKernelContext* context) noexcept {
    EXCEPTION_TO_STATUS_BEGIN
    Ort::KernelContext ctx{context};
    Ort::ConstValue input = ctx.GetInput(0);
    auto input_info = input.GetTensorTypeAndShapeInfo();
    auto input_shape = input_info.GetShape();

    // Squeeze axes from input 1 (opset 13+) or default to all size-1 dims.
    std::set<int64_t> axes_set;
    if (ctx.GetInputCount() > 1) {
      Ort::ConstValue axes_tensor = ctx.GetInput(1);
      if (axes_tensor) {
        auto axes_info = axes_tensor.GetTensorTypeAndShapeInfo();
        size_t n = axes_info.GetElementCount();
        const int64_t* axes_data = axes_tensor.GetTensorData<int64_t>();
        for (size_t i = 0; i < n; ++i) {
          int64_t a = axes_data[i];
          if (a < 0) a += static_cast<int64_t>(input_shape.size());
          axes_set.insert(a);
        }
      }
    }

    std::vector<int64_t> output_shape;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (axes_set.empty()) {
        if (input_shape[i] != 1) output_shape.push_back(input_shape[i]);
      } else {
        if (axes_set.find(static_cast<int64_t>(i)) == axes_set.end()) {
          output_shape.push_back(input_shape[i]);
        }
      }
    }
    if (output_shape.empty()) output_shape.push_back(1);  // scalar
    return ShapeCopyCompute(context, output_shape);
    EXCEPTION_TO_STATUS_END
  }
};

// Unsqueeze kernel: inserts dimensions of size 1.
struct UnsqueezeKernelImpl : public OrtKernelImpl {
  UnsqueezeKernelImpl() : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    flags = 0;
    Compute = ComputeImpl;
    Release = [](OrtKernelImpl* p) noexcept { delete static_cast<UnsqueezeKernelImpl*>(p); };
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl*, OrtKernelContext* context) noexcept {
    EXCEPTION_TO_STATUS_BEGIN
    Ort::KernelContext ctx{context};
    Ort::ConstValue input = ctx.GetInput(0);
    auto input_info = input.GetTensorTypeAndShapeInfo();
    auto input_shape = input_info.GetShape();

    // Read axes from input 1 (opset 13+).
    Ort::ConstValue axes_tensor = ctx.GetInput(1);
    auto axes_info = axes_tensor.GetTensorTypeAndShapeInfo();
    size_t n = axes_info.GetElementCount();
    const int64_t* axes_data = axes_tensor.GetTensorData<int64_t>();

    int64_t output_rank = static_cast<int64_t>(input_shape.size() + n);
    std::set<int64_t> axes_set;
    for (size_t i = 0; i < n; ++i) {
      int64_t a = axes_data[i];
      if (a < 0) a += output_rank;
      axes_set.insert(a);
    }

    std::vector<int64_t> output_shape;
    output_shape.reserve(output_rank);
    size_t input_idx = 0;
    for (int64_t i = 0; i < output_rank; ++i) {
      if (axes_set.count(i)) {
        output_shape.push_back(1);
      } else {
        output_shape.push_back(input_shape[input_idx++]);
      }
    }
    return ShapeCopyCompute(context, output_shape);
    EXCEPTION_TO_STATUS_END
  }
};

// Flatten kernel: reshapes to 2D based on the axis attribute.
struct FlattenKernelImpl : public OrtKernelImpl {
  int64_t axis = 1;

  explicit FlattenKernelImpl(const OrtKernelInfo* info) : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    flags = 0;
    Compute = ComputeImpl;
    Release = [](OrtKernelImpl* p) noexcept { delete static_cast<FlattenKernelImpl*>(p); };
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;
    Ort::ConstKernelInfo ki{info};
    try {
      axis = ki.GetAttribute<int64_t>("axis");
    } catch (...) {
      axis = 1;
    }
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* context) noexcept {
    EXCEPTION_TO_STATUS_BEGIN
    auto* self = static_cast<FlattenKernelImpl*>(this_ptr);
    Ort::KernelContext ctx{context};
    Ort::ConstValue input = ctx.GetInput(0);
    auto input_info = input.GetTensorTypeAndShapeInfo();
    auto input_shape = input_info.GetShape();

    int64_t ax = self->axis;
    if (ax < 0) ax += static_cast<int64_t>(input_shape.size());

    int64_t d0 = 1, d1 = 1;
    for (int64_t i = 0; i < ax; ++i) d0 *= input_shape[i];
    for (size_t i = static_cast<size_t>(ax); i < input_shape.size(); ++i) d1 *= input_shape[i];

    std::vector<int64_t> output_shape = {d0, d1};
    return ShapeCopyCompute(context, output_shape);
    EXCEPTION_TO_STATUS_END
  }
};

// Creation functions for shape-only ops
OrtStatus* ORT_API_CALL CreateReshapeKernel(void*, const OrtKernelInfo*, OrtKernelImpl** out) noexcept {
  *out = new ReshapeKernelImpl();
  return nullptr;
}
OrtStatus* ORT_API_CALL CreateSqueezeKernel(void*, const OrtKernelInfo*, OrtKernelImpl** out) noexcept {
  *out = new SqueezeKernelImpl();
  return nullptr;
}
OrtStatus* ORT_API_CALL CreateUnsqueezeKernel(void*, const OrtKernelInfo*, OrtKernelImpl** out) noexcept {
  *out = new UnsqueezeKernelImpl();
  return nullptr;
}
OrtStatus* ORT_API_CALL CreateFlattenKernel(void*, const OrtKernelInfo* info, OrtKernelImpl** out) noexcept {
  *out = new FlattenKernelImpl(info);
  return nullptr;
}

OrtStatus* ORT_API_CALL CreateRotaryEmbeddingKernel(void* /*state*/,
                                                    const OrtKernelInfo* info,
                                                    OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::RotaryEmbedding<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::RotaryEmbedding<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::RotaryEmbedding<BFloat16>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("RotaryEmbedding: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateGroupQueryAttentionKernel(void* /*state*/,
                                                        const OrtKernelInfo* info,
                                                        OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto query_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();

  // Resolve cache type by edge name instead of fixed index since GQA has optional
  // inputs and packed/non-packed forms that can shift effective input indices.
  ONNXTensorElementDataType cache_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  for (size_t i = 0; i < ki.GetInputCount(); ++i) {
    if (ki.GetInputName(i) == "past_key") {
      cache_type = ki.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
      break;
    }
  }
  if (cache_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
    for (size_t i = 0; i < ki.GetOutputCount(); ++i) {
      if (ki.GetOutputName(i) == "present_key") {
        cache_type = ki.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        break;
      }
    }
  }
  if (cache_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
    // Unquantized path without past/present cache tensors.
    cache_type = query_type;
  }
  if (query_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    switch (cache_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GroupQueryAttention<MLFloat16, MLFloat16>>(info);
        return nullptr;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GroupQueryAttention<MLFloat16, int8_t>>(info);
        return nullptr;
#ifdef USE_INT4_KV_CACHE
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GroupQueryAttention<MLFloat16, uint8_t>>(info);
        return nullptr;
#endif
#ifdef USE_FP8_KV_CACHE
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GroupQueryAttention<MLFloat16, Float8E4M3FN>>(info);
        return nullptr;
#endif
      default:
        break;
    }
  } else if (query_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    switch (cache_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GroupQueryAttention<BFloat16, BFloat16>>(info);
        return nullptr;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GroupQueryAttention<BFloat16, int8_t>>(info);
        return nullptr;
#ifdef USE_INT4_KV_CACHE
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GroupQueryAttention<BFloat16, uint8_t>>(info);
        return nullptr;
#endif
#ifdef USE_FP8_KV_CACHE
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GroupQueryAttention<BFloat16, Float8E4M3FN>>(info);
        return nullptr;
#endif
      default:
        break;
    }
  }

  return Ort::GetApi().CreateStatus(
      ORT_EP_FAIL,
      (std::string("GroupQueryAttention: unsupported type combination. query=") +
       std::to_string(query_type) + ", cache=" + std::to_string(cache_type))
          .c_str());
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateAttentionKernel(void* /*state*/,
                                              const OrtKernelInfo* info,
                                              OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::Attention<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::Attention<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::Attention<BFloat16>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("Attention: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateMultiHeadAttentionKernel(void* /*state*/,
                                                       const OrtKernelInfo* info,
                                                       OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  const auto t_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();

  ONNXTensorElementDataType qk_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  for (size_t i = 0; i < ki.GetOutputCount(); ++i) {
    if (ki.GetOutputName(i) == "qk") {
      qk_type = ki.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
      break;
    }
  }
  if (qk_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
    // qk is optional and can be omitted when output_qk == 0.
    qk_type = t_type;
  }

  if (t_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    if (qk_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::MultiHeadAttention<float, float>>(info);
      return nullptr;
    }
    if (qk_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::MultiHeadAttention<float, MLFloat16>>(info);
      return nullptr;
    }
    if (qk_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::MultiHeadAttention<float, BFloat16>>(info);
      return nullptr;
    }
  } else if (t_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    if (qk_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::MultiHeadAttention<MLFloat16, float>>(info);
      return nullptr;
    }
    if (qk_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::MultiHeadAttention<MLFloat16, MLFloat16>>(info);
      return nullptr;
    }
  } else if (t_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    if (qk_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::MultiHeadAttention<BFloat16, float>>(info);
      return nullptr;
    }
    if (qk_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::MultiHeadAttention<BFloat16, BFloat16>>(info);
      return nullptr;
    }
  }

  return Ort::GetApi().CreateStatus(
      ORT_EP_FAIL,
      (std::string("MultiHeadAttention: unsupported type combination T=") +
       std::to_string(t_type) + ", QK=" + std::to_string(qk_type))
          .c_str());
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateSkipLayerNormalizationKernel(void* /*state*/,
                                                           const OrtKernelInfo* info,
                                                           OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::SkipLayerNorm<float, false>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::SkipLayerNorm<MLFloat16, false>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::SkipLayerNorm<BFloat16, false>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("SkipLayerNormalization: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateSkipSimplifiedLayerNormalizationKernel(void* /*state*/,
                                                                     const OrtKernelInfo* info,
                                                                     OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::SkipLayerNorm<float, true>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::SkipLayerNorm<MLFloat16, true>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::SkipLayerNorm<BFloat16, true>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("SkipSimplifiedLayerNormalization: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateGemmaRotaryEmbeddingKernel(void* /*state*/,
                                                         const OrtKernelInfo* info,
                                                         OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto emb_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  auto data_type = ki.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetElementType();
  if (emb_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GemmaRotaryEmbedding<MLFloat16, float>>(info);
    return nullptr;
  }
  return Ort::GetApi().CreateStatus(
      ORT_EP_FAIL,
      (std::string("GemmaRotaryEmbedding: unsupported type combination emb=") +
       std::to_string(emb_type) + ", data=" + std::to_string(data_type))
          .c_str());
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateEmbedLayerNormalizationKernel(void* /*state*/,
                                                            const OrtKernelInfo* info,
                                                            OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto emb_type = ki.GetInputTypeInfo(2).GetTensorTypeAndShapeInfo().GetElementType();
  switch (emb_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::EmbedLayerNorm<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::EmbedLayerNorm<MLFloat16>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("EmbedLayerNormalization: unsupported type ") + std::to_string(emb_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateFastGeluKernel(void* /*state*/,
                                             const OrtKernelInfo* info,
                                             OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::FastGelu<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::FastGelu<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::FastGelu<BFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::FastGelu<double>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("FastGelu: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateDecoderMaskedMultiHeadAttentionKernel(void* /*state*/,
                                                                    const OrtKernelInfo* info,
                                                                    OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  const auto t_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();

  ONNXTensorElementDataType qk_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  for (size_t i = 0; i < ki.GetOutputCount(); ++i) {
    if (ki.GetOutputName(i) == "qk") {
      qk_type = ki.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
      break;
    }
  }
  if (qk_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
    // qk is optional and can be omitted when output_qk == 0.
    qk_type = t_type;
  }

  if (t_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    if (qk_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::DecoderMaskedMultiHeadAttention<float, float>>(info);
      return nullptr;
    }
    if (qk_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::DecoderMaskedMultiHeadAttention<float, MLFloat16>>(info);
      return nullptr;
    }
  } else if (t_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    if (qk_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::DecoderMaskedMultiHeadAttention<MLFloat16, float>>(info);
      return nullptr;
    }
    if (qk_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::DecoderMaskedMultiHeadAttention<MLFloat16, MLFloat16>>(info);
      return nullptr;
    }
  }

  return Ort::GetApi().CreateStatus(
      ORT_EP_FAIL,
      (std::string("DecoderMaskedMultiHeadAttention: unsupported type combination T=") +
       std::to_string(t_type) + ", QK=" + std::to_string(qk_type))
          .c_str());
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateMatMulNBitsKernel(void* /*state*/,
                                                const OrtKernelInfo* info,
                                                OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::MatMulNBits<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::MatMulNBits<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::MatMulNBits<BFloat16>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("MatMulNBits: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateMoEKernel(void* /*state*/,
                                        const OrtKernelInfo* info,
                                        OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::MoE<float>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::MoE<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::MoE<BFloat16>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("MoE: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateQMoEKernel(void* /*state*/,
                                         const OrtKernelInfo* info,
                                         OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  auto input_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  switch (input_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::QMoE<MLFloat16>>(info);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::QMoE<BFloat16>>(info);
      break;
    default:
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("QMoE: unsupported type ") + std::to_string(input_type)).c_str());
  }
  return nullptr;
  EXCEPTION_TO_STATUS_END
}

OrtStatus* ORT_API_CALL CreateGatherBlockQuantizedKernel(void* /*state*/,
                                                         const OrtKernelInfo* info,
                                                         OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_STATUS_BEGIN
  Ort::ConstKernelInfo ki(info);
  const auto data_type = ki.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  const auto index_type = ki.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetElementType();
  const auto dequantized_type = ki.GetInputTypeInfo(2).GetTensorTypeAndShapeInfo().GetElementType();

  if (index_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<uint8_t, float, int32_t>>(info);
        return nullptr;
      }
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<uint8_t, MLFloat16, int32_t>>(info);
        return nullptr;
      }
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<uint8_t, BFloat16, int32_t>>(info);
        return nullptr;
      }
    } else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4) {
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<UInt4x2, float, int32_t>>(info);
        return nullptr;
      }
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<UInt4x2, MLFloat16, int32_t>>(info);
        return nullptr;
      }
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<UInt4x2, BFloat16, int32_t>>(info);
        return nullptr;
      }
    } else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4) {
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<Int4x2, float, int32_t>>(info);
        return nullptr;
      }
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<Int4x2, MLFloat16, int32_t>>(info);
        return nullptr;
      }
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<Int4x2, BFloat16, int32_t>>(info);
        return nullptr;
      }
    }
  } else if (index_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<uint8_t, float, int64_t>>(info);
        return nullptr;
      }
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<uint8_t, MLFloat16, int64_t>>(info);
        return nullptr;
      }
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<uint8_t, BFloat16, int64_t>>(info);
        return nullptr;
      }
    } else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4) {
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<UInt4x2, float, int64_t>>(info);
        return nullptr;
      }
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<UInt4x2, MLFloat16, int64_t>>(info);
        return nullptr;
      }
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<UInt4x2, BFloat16, int64_t>>(info);
        return nullptr;
      }
    } else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4) {
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<Int4x2, float, int64_t>>(info);
        return nullptr;
      }
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<Int4x2, MLFloat16, int64_t>>(info);
        return nullptr;
      }
      if (dequantized_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        *kernel_out = new AdapterKernelImpl<onnxruntime::contrib::cuda::GatherBlockQuantized<Int4x2, BFloat16, int64_t>>(info);
        return nullptr;
      }
    }
  }

  return Ort::GetApi().CreateStatus(
      ORT_EP_FAIL,
      (std::string("GatherBlockQuantized: unsupported type combination data=") +
       std::to_string(data_type) + ", indices=" + std::to_string(index_type) +
       ", scales=" + std::to_string(dequantized_type))
          .c_str());
  EXCEPTION_TO_STATUS_END
}

PluginKernelCreateFn GetCreateFnForOp(std::string_view op_type, std::string_view domain = "") {
  if (domain == "com.microsoft") {
    if (op_type == "Attention") return CreateAttentionKernel;
    if (op_type == "DecoderMaskedMultiHeadAttention") return CreateDecoderMaskedMultiHeadAttentionKernel;
    if (op_type == "EmbedLayerNormalization") return CreateEmbedLayerNormalizationKernel;
    if (op_type == "FastGelu") return CreateFastGeluKernel;
    if (op_type == "GatherBlockQuantized") return CreateGatherBlockQuantizedKernel;
    if (op_type == "GemmaRotaryEmbedding") return CreateGemmaRotaryEmbeddingKernel;
    if (op_type == "GroupQueryAttention") return CreateGroupQueryAttentionKernel;
    if (op_type == "MatMulNBits") return CreateMatMulNBitsKernel;
    if (op_type == "MoE") return CreateMoEKernel;
    if (op_type == "MultiHeadAttention") return CreateMultiHeadAttentionKernel;
    if (op_type == "QMoE") return CreateQMoEKernel;
    if (op_type == "RotaryEmbedding") return CreateRotaryEmbeddingKernel;
    if (op_type == "SkipLayerNormalization") return CreateSkipLayerNormalizationKernel;
    if (op_type == "SkipSimplifiedLayerNormalization") return CreateSkipSimplifiedLayerNormalizationKernel;
  }

  if (op_type == "Relu") return CreateReluKernel;
  if (op_type == "Elu") return CreateEluKernel;
  if (op_type == "HardSigmoid") return CreateHardSigmoidKernel;
  if (op_type == "HardSwish") return CreateHardSwishKernel;
  if (op_type == "LeakyRelu") return CreateLeakyReluKernel;
  if (op_type == "Selu") return CreateSeluKernel;
  if (op_type == "Sigmoid") return CreateSigmoidKernel;
  if (op_type == "Softplus") return CreateSoftplusKernel;
  if (op_type == "Softsign") return CreateSoftsignKernel;
  if (op_type == "Tanh") return CreateTanhKernel;
  if (op_type == "ThresholdedRelu") return CreateThresholdedReluKernel;
  if (op_type == "Add") return CreateAddKernel;
  if (op_type == "MatMul") return CreateMatMulKernel;
  if (op_type == "Gemm") return CreateGemmKernel;
  if (op_type == "Conv") return CreateConvKernel;
  if (op_type == "Sub") return CreateSubKernel;
  if (op_type == "Mul") return CreateMulKernel;
  if (op_type == "Div") return CreateDivKernel;
  if (op_type == "Pow") return CreatePowKernel;
  if (op_type == "Equal") return CreateEqualKernel;
  if (op_type == "Greater") return CreateGreaterKernel;
  if (op_type == "Less") return CreateLessKernel;
  if (op_type == "Abs") return CreateAbsKernel;
  if (op_type == "Neg") return CreateNegKernel;
  if (op_type == "Floor") return CreateFloorKernel;
  if (op_type == "Ceil") return CreateCeilKernel;
  if (op_type == "Sqrt") return CreateSqrtKernel;
  if (op_type == "Exp") return CreateExpKernel;
  if (op_type == "Log") return CreateLogKernel;
  if (op_type == "Clip") return CreateClipKernel;
  if (op_type == "Where") return CreateWhereKernel;
  if (op_type == "Cast") return CreateCastKernel;
  if (op_type == "ArgMax") return CreateArgMaxKernel;
  if (op_type == "ArgMin") return CreateArgMinKernel;
  if (op_type == "ReduceMax") return CreateReduceMaxKernel;
  if (op_type == "ReduceMean") return CreateReduceMeanKernel;
  if (op_type == "ReduceMin") return CreateReduceMinKernel;
  if (op_type == "ReduceSum") return CreateReduceSumKernel;
  if (op_type == "Softmax") return CreateSoftmaxKernel;
  if (op_type == "LogSoftmax") return CreateLogSoftmaxKernel;
  if (op_type == "Concat") return CreateConcatKernel;
  if (op_type == "Split") return CreateSplitKernel;
  if (op_type == "Gather") return CreateGatherKernel;
  // Shape-only ops (Task 4.1)
  if (op_type == "Reshape") return CreateReshapeKernel;
  if (op_type == "Squeeze") return CreateSqueezeKernel;
  if (op_type == "Unsqueeze") return CreateUnsqueezeKernel;
  if (op_type == "Flatten") return CreateFlattenKernel;
  return nullptr;
}

bool IsRegistrationSupportedByCreateFn(const GeneratedKernelRegistration& reg) {
  const std::string_view op_type = reg.op_type ? reg.op_type : "";

  // Stage 1 custom kernels currently implement float-only paths.
  if ((op_type == "MatMul" || op_type == "Gemm" || op_type == "Conv") &&
      reg.type_constraint != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
    return reg.type_constraint == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  // Typed registration where the migration script could not map the type token
  // to a public ONNX data type enum. Skip it instead of creating an unconstrained
  // kernel def that would be too broad.
  if (reg.type_constraint == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED &&
      reg.constraint_name != nullptr && reg.constraint_name[0] != '\0') {
    return false;
  }

  return true;
}

constexpr GeneratedKernelRegistration kGeneratedKernelRegistrations[] = {
#include "core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc"
#include "core/providers/cuda/plugin/cuda_plugin_generated_contrib_registrations.inc"
};

}  // namespace

OrtStatus* CreateCudaKernelRegistry(const OrtEpApi& ep_api,
                                    const char* ep_name,
                                    void* /*create_kernel_state*/,
                                    OrtKernelRegistry** out_registry) {
  *out_registry = nullptr;

  EXCEPTION_TO_STATUS_BEGIN

  Ort::KernelRegistry registry;

  // Group registrations by (op, domain, version_start, version_end) to build
  // kernel defs with proper per-constraint type lists.
  struct KernelDefKey {
    std::string op_type;
    std::string domain;
    int since_version_start;
    int since_version_end;
    int registration_id;
    std::vector<int> input_mem_type_indices;
    std::vector<int> output_mem_type_indices;

    bool operator<(const KernelDefKey& other) const {
      return std::tie(op_type, domain, since_version_start, since_version_end, registration_id,
                      input_mem_type_indices, output_mem_type_indices) <
             std::tie(other.op_type, other.domain, other.since_version_start, other.since_version_end,
                      other.registration_id, other.input_mem_type_indices,
                      other.output_mem_type_indices);
    }
  };

  // Map: key -> { constraint_name -> set of type enums }
  std::map<KernelDefKey, std::map<std::string, std::set<ONNXTensorElementDataType>>> grouped;

  for (size_t i = 0; i < std::size(kGeneratedKernelRegistrations); ++i) {
    const auto& reg = kGeneratedKernelRegistrations[i];
    const std::string_view reg_domain = reg.domain ? reg.domain : "";
    PluginKernelCreateFn create_fn = GetCreateFnForOp(reg.op_type, reg_domain);
    if (!create_fn) {
      continue;
    }

    if (!IsRegistrationSupportedByCreateFn(reg)) {
      continue;
    }

    std::vector<int> in_mem, out_mem;
    for (int j = 0; j < reg.num_input_mem_type_indices; ++j) in_mem.push_back(reg.input_mem_type_indices[j]);
    for (int j = 0; j < reg.num_output_mem_type_indices; ++j) out_mem.push_back(reg.output_mem_type_indices[j]);

    KernelDefKey key{
        reg.op_type,
        reg.domain ? reg.domain : "",
        reg.since_version_start,
        reg.since_version_end,
        reg.registration_id,
        in_mem,
        out_mem};

    auto [it, inserted] = grouped.try_emplace(key);
    ORT_UNUSED_PARAMETER(inserted);

    if (reg.type_constraint != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      const std::string constraint_name =
          (reg.constraint_name && reg.constraint_name[0] != '\0') ? reg.constraint_name : "T";
      it->second[constraint_name].insert(reg.type_constraint);
    }
  }

  // Now register one KernelDef per grouped key, with all type constraints.
  for (const auto& [key, constraints] : grouped) {
    PluginKernelCreateFn create_fn = GetCreateFnForOp(key.op_type, key.domain);
    if (!create_fn) {
      continue;
    }

    Ort::KernelDefBuilder builder;
    builder.SetOperatorType(key.op_type.c_str())
        .SetDomain(key.domain.c_str())
        .SetSinceVersion(key.since_version_start, key.since_version_end)
        .SetExecutionProvider(ep_name);

    // For each input defined in the registration, apply the memory type constraint.
    // Index mapping matches the input positions defined for the operator.
    for (int idx : key.input_mem_type_indices) {
      builder.SetInputMemType(idx, OrtMemTypeCPUInput);
    }
    // For each output defined in the registration, apply the memory type constraint.
    // Index mapping matches the input positions defined for the operator.
    for (int idx : key.output_mem_type_indices) {
      builder.SetOutputMemType(idx, OrtMemTypeCPUOutput);
    }

    // Add mutable alias metadata using schema-name based resolution.
    AddMutableAliasesBySchemaName(key.domain, key.op_type, key.since_version_end, builder);

    for (const auto& [cname, types] : constraints) {
      std::vector<const OrtDataType*> type_list;
      for (ONNXTensorElementDataType t : types) {
        const OrtDataType* dt = nullptr;
        RETURN_IF_ERROR(ep_api.GetTensorDataType(t, &dt));
        type_list.push_back(dt);
      }
      builder.AddTypeConstraint(cname.c_str(), type_list);
    }

    Ort::KernelDef kernel_def = builder.Build();
    RETURN_IF_ERROR(registry.AddKernel(kernel_def.release(), create_fn, nullptr));
  }

  *out_registry = registry.release();
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
