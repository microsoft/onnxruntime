// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/fused_tensor_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "core/common/float8.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/scaled_silu_impl.h"
#include "contrib_ops/cuda/bert/hyper_connection_pre_mix_impl.h"
#include "contrib_ops/cuda/bert/branchwise_rms_norm_impl.h"
#include "core/providers/cuda/llm/rotary_embedding_impl.h"
#include "core/providers/cuda/reduction/reduction_functions.h"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"

namespace onnxruntime::contrib::cuda {

ONNX_OPERATOR_KERNEL_EX(
    RMSNormRotary, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("F", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("TI", DataTypeImpl::GetTensorType<int32_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 5),
    RMSNormRotary);

RMSNormRotary::RMSNormRotary(const OpKernelInfo& info)
    : CudaKernel(info),
      epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-6f)),
      input_start_(info.GetAttrOrDefault<int64_t>("input_start", 0)),
      input_width_(info.GetAttrOrDefault<int64_t>("input_width", 0)),
      rotary_dim_(info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0)),
      fp32_rotary_(info.GetAttrOrDefault<int64_t>("use_fp32_arithmetic", 1) == 1),
      flatten_batch_(info.GetAttrOrDefault<int64_t>("flatten_batch", 0) == 1) {
  const auto arithmetic = info.GetAttrOrDefault<int64_t>("use_fp32_arithmetic", 1);
  const auto flatten = info.GetAttrOrDefault<int64_t>("flatten_batch", 0);
  ORT_ENFORCE(std::isfinite(epsilon_) && epsilon_ >= 0 && input_start_ >= 0 && input_width_ >= 0 &&
                  rotary_dim_ > 0 && rotary_dim_ % 2 == 0 && (arithmetic == 0 || arithmetic == 1) &&
                  (flatten == 0 || flatten == 1),
              "RMSNormRotary requires finite nonnegative epsilon, nonnegative slice, positive even rotary width and mode 0/1");
}

Status RMSNormRotary::ComputeInternal(OpKernelContext* context) const {
  const auto* input = context->Input<Tensor>(0);
  const auto* scale = context->Input<Tensor>(1);
  const auto* positions = context->Input<Tensor>(2);
  const auto* cosine = context->Input<Tensor>(3);
  const auto* sine = context->Input<Tensor>(4);
  const auto* mode = context->Input<Tensor>(5);
  const auto& shape = input->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() == 4 && shape[2] > 0 && shape[3] > 0,
                    "RMSNormRotary input must have shape [B,S,heads,stride] with positive heads/stride");
  const int64_t width = input_width_ == 0 ? shape[3] : input_width_;
  ORT_RETURN_IF_NOT(width > 0 && input_start_ <= shape[3] && width <= shape[3] - input_start_ &&
                        rotary_dim_ <= width && scale->Shape() == TensorShape({width}),
                    "RMSNormRotary slice, rotary width or scale shape is incompatible");
  const auto& cache = cosine->Shape();
  ORT_RETURN_IF_NOT(cache.NumDimensions() == 2 && cache[0] > 0 && cache[1] == rotary_dim_ / 2 &&
                        sine->Shape() == cache,
                    "RMSNormRotary cosine/sine must have matching shape [cache_rows,rotary_width/2]");
  const bool scalar = positions->Shape().Size() == 1;
  const int64_t output_batch = flatten_batch_ ? 1 : shape[0];
  const int64_t output_sequence = flatten_batch_ ? static_cast<int64_t>(SafeInt<int64_t>(shape[0]) * shape[1]) : shape[1];
  ORT_RETURN_IF_NOT(scalar || positions->Shape() == TensorShape({output_batch, output_sequence}),
                    "RMSNormRotary positions must contain one base position or have shape [B,S]");
  ORT_RETURN_IF_NOT(!scalar || output_sequence <= cache[0], "RMSNormRotary sequence exceeds cache capacity");
  bool dense = false;
  if (mode != nullptr) {
    ORT_RETURN_IF_NOT(mode->Shape().Size() == 1 && (mode->Data<int32_t>()[0] == 0 || mode->Data<int32_t>()[0] == 1),
                      "RMSNormRotary prefill_numerics must contain one int32 value 0 or 1");
    dense = mode->Data<int32_t>()[0] == 1;
  }
  const TensorShape norm_shape({shape[0], shape[1], shape[2], width});
  const int groups = SafeInt<int>(norm_shape.Size() / width);
  const int heads = SafeInt<int>(shape[2]), stride = SafeInt<int>(shape[3]);
  const int sequence = SafeInt<int>(output_sequence), batch = SafeInt<int>(output_batch);
  const int dim = SafeInt<int>(width), rotary = SafeInt<int>(rotary_dim_), cache_rows = SafeInt<int>(cache[0]);
  auto* output = context->Output(0, TensorShape({output_batch, output_sequence, SafeInt<int64_t>(shape[2]) * width}));
  auto* normalized = context->OutputCount() > 1 ? context->Output(1, norm_shape) : nullptr;
  if (groups == 0) return Status::OK();
  const auto* x = reinterpret_cast<const uint16_t*>(input->Data<BFloat16>()) + input_start_;
  if (dim == 256 && groups > 1 && (dense || fp32_rotary_)) {
    return LaunchRMSNormRotary(
        Stream(context), x, scale->Data<float>(), positions->Data<int64_t>(),
        reinterpret_cast<const uint16_t*>(cosine->Data<BFloat16>()),
        reinterpret_cast<const uint16_t*>(sine->Data<BFloat16>()),
        reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()),
        normalized ? normalized->MutableData<float>() : nullptr,
        groups, heads, sequence, stride, rotary, cache_rows, scalar, epsilon_, dense, fp32_rotary_);
  }
  using CudaT = onnxruntime::cuda::OrtToCudaType<BFloat16>::type;
  auto prepared = GetScratchBuffer<CudaT>(SafeInt<size_t>(norm_shape.Size()), GetComputeStream(context));
  auto squares = GetScratchBuffer<float>(groups == 1 ? width : 0, GetComputeStream(context));
  auto sums = GetScratchBuffer<float>(groups, GetComputeStream(context));
  const auto buffer_size = onnxruntime::cuda::compute_reduce_matrix_columns_buffer_size<float>(groups, dim);
  auto buffer = GetScratchBuffer<uint8_t>(buffer_size, GetComputeStream(context));
  ORT_RETURN_IF_ERROR(LaunchStandardBranchwiseRMSNorm<CudaT>(
      Stream(context), reinterpret_cast<const CudaT*>(x), scale->Data<float>(),
      ONNX_NAMESPACE::TensorProto::FLOAT, prepared.get(), groups, heads, dim, true, epsilon_,
      normalized ? normalized->MutableData<float>() : nullptr, squares.get(), sums.get(),
      buffer.get(), buffer_size, stride, dense));
  const auto launch_rotary = [&]<typename RotaryT>() {
    return LaunchRotaryEmbeddingKernel<RotaryT>(
      Stream(context), reinterpret_cast<RotaryT*>(output->MutableData<BFloat16>()),
      reinterpret_cast<const RotaryT*>(prepared.get()),
      positions->Data<int64_t>(), nullptr,
      reinterpret_cast<const RotaryT*>(cosine->Data<BFloat16>()),
      reinterpret_cast<const RotaryT*>(sine->Data<BFloat16>()), batch, sequence, heads, dim, rotary,
      cache_rows, scalar ? 0 : 1, false, GetDeviceProp().maxThreadsPerBlock, false,
      !dense && fp32_rotary_);
  };
  if (dense) return launch_rotary.template operator()<CudaT>();
  return launch_rotary.template operator()<onnxruntime::cuda::ToCudaType<BFloat16>::MappedType>();
}

ONNX_OPERATOR_KERNEL_EX(
    PackedMtpMoe, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("W", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("F", DataTypeImpl::GetTensorType<float>()), PackedMtpMoe);

PackedMtpMoe::PackedMtpMoe(const OpKernelInfo& info)
    : CudaKernel(info), gate_up_scale_(info.GetAttrOrDefault<float>("gate_up_global_scale", 1.0f)),
      down_scale_(info.GetAttrOrDefault<float>("down_global_scale", 1.0f)) {
  ORT_ENFORCE(std::isfinite(gate_up_scale_) && gate_up_scale_ > 0 &&
                  std::isfinite(down_scale_) && down_scale_ > 0,
              "PackedMtpMoe requires positive finite global scales");
}

Status PackedMtpMoe::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router = context->Input<Tensor>(1);
  const Tensor* gate_up = context->Input<Tensor>(2);
  const Tensor* down = context->Input<Tensor>(3);
  const auto& shape = input->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() > 0 && shape[shape.NumDimensions() - 1] == 2560,
                    "PackedMtpMoe requires input [...,2560]");
  const int64_t rows = shape.Size() / 2560;
  auto router_shape = shape.AsShapeVector();
  router_shape.back() = 512;
  ORT_RETURN_IF_NOT(rows <= 65535 && router->Shape() == TensorShape(router_shape) &&
                        gate_up->Shape() == TensorShape({512, 1280, 1440}) &&
                        down->Shape() == TensorShape({512, 2560, 360}),
                    "PackedMtpMoe requires matching router, NVFP4 weights, and at most 65535 rows");
  Tensor* output = context->Output(0, shape);
  constexpr int64_t scratch_stride = 6432;
  Tensor* scratch_output = context->OutputCount() > 1
                              ? context->Output(1, TensorShape({rows, scratch_stride})) : nullptr;
  const size_t scratch_elements = SafeInt<size_t>(rows) * scratch_stride;
  auto scratch = GetScratchBuffer<float>(scratch_output == nullptr ? scratch_elements : 0,
                                         GetComputeStream(context));
  return LaunchPackedMtpMoe(Stream(context), reinterpret_cast<const uint16_t*>(input->Data<BFloat16>()),
      reinterpret_cast<const uint16_t*>(router->Data<BFloat16>()), gate_up->Data<uint8_t>(), down->Data<uint8_t>(),
      reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()),
      scratch_output == nullptr ? scratch.get() : scratch_output->MutableData<float>(),
      SafeInt<int>(rows), gate_up_scale_, down_scale_);
}

#define REGISTER_BF16_PROJECTION(name) \
  ONNX_OPERATOR_KERNEL_EX(name, kMSDomain, 1, kCudaExecutionProvider, \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>()), name);
REGISTER_BF16_PROJECTION(MtpInputProjection)
REGISTER_BF16_PROJECTION(HyperConnectionMixer)
#undef REGISTER_BF16_PROJECTION
ONNX_OPERATOR_KERNEL_EX(
    MtpAttentionProjection, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("F", DataTypeImpl::GetTensorType<float>()), MtpAttentionProjection);

namespace {
using ProjectionCudaT = onnxruntime::cuda::OrtToCudaType<BFloat16>::type;

Status ProjectionGemm(cublasHandle_t handle, const cudaDeviceProp& properties, bool tf32,
                      const BFloat16* input, const Tensor* weight, BFloat16* output,
                      int64_t rows, int64_t width) {
  ORT_RETURN_IF_NOT(weight->Shape().NumDimensions() == 2 && weight->Shape()[0] == width &&
                        weight->Shape()[1] > 0, "Projection weight must have shape [input_width, output_width]");
  if (rows == 0) return Status::OK();
  const int m = SafeInt<int>(rows), k = SafeInt<int>(width), n = SafeInt<int>(weight->Shape()[1]);
  const ProjectionCudaT one(1.0f), zero(0.0f);
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const ProjectionCudaT*>(weight->Data<BFloat16>()), n,
      reinterpret_cast<const ProjectionCudaT*>(input), k, &zero,
      reinterpret_cast<ProjectionCudaT*>(output), n, properties, tf32));
  return Status::OK();
}
}  // namespace

Status MtpInputProjection::ComputeInternal(OpKernelContext* context) const {
  const Tensor* hidden = context->Input<Tensor>(0);
  const Tensor* embedding = context->Input<Tensor>(1);
  const Tensor* hidden_weight = context->Input<Tensor>(2);
  const Tensor* embedding_weight = context->Input<Tensor>(3);
  const auto& hs = hidden->Shape();
  const auto& es = embedding->Shape();
  ORT_RETURN_IF_NOT(hs.NumDimensions() == 4 && es.NumDimensions() == 3 &&
                        hs[0] == es[0] && hs[1] == es[1] && hs[2] > 0 && hs[3] > 0 && es[2] > 0,
                    "MtpInputProjection expects hidden [B,S,C,H] and embedding [B,S,E]");
  ORT_RETURN_IF_NOT(hidden_weight->Shape() == TensorShape({hs[3], hs[3]}) &&
                        embedding_weight->Shape() == TensorShape({es[2], hs[3]}),
                    "MtpInputProjection weights must project to the hidden width");
  const int64_t rows = SafeInt<int64_t>(hs[0]) * hs[1];
  const int64_t count = SafeInt<int64_t>(rows) * hs[2] * hs[3];
  Tensor* output = context->Output(0, TensorShape({hs[0], hs[1], SafeInt<int64_t>(hs[2]) * hs[3]}));
  auto h = GetScratchBuffer<BFloat16>(SafeInt<size_t>(count), GetComputeStream(context));
  auto e = GetScratchBuffer<BFloat16>(SafeInt<size_t>(rows) * hs[3], GetComputeStream(context));
  ORT_RETURN_IF_ERROR(ProjectionGemm(GetCublasHandle(context), GetDeviceProp(), UseTF32(),
      hidden->Data<BFloat16>(), hidden_weight, h.get(), SafeInt<int64_t>(rows) * hs[2], hs[3]));
  ORT_RETURN_IF_ERROR(ProjectionGemm(GetCublasHandle(context), GetDeviceProp(), UseTF32(),
      embedding->Data<BFloat16>(), embedding_weight, e.get(), rows, es[2]));
  return LaunchMtpProjectionAdd(Stream(context), reinterpret_cast<const uint16_t*>(h.get()),
      reinterpret_cast<const uint16_t*>(e.get()), reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()),
      count, hs[3], hs[2]);
}

HyperConnectionMixer::HyperConnectionMixer(const OpKernelInfo& info)
    : CudaKernel(info), branches_(info.GetAttrOrDefault<int64_t>("num_branches", 4)),
      alpha_(info.GetAttrOrDefault<float>("alpha", 0.25f)),
      reduction_scale_(info.GetAttrOrDefault<float>("reduction_scale", 0.25f)) {
  ORT_ENFORCE(branches_ > 0 && std::isfinite(alpha_) && std::isfinite(reduction_scale_),
              "HyperConnectionMixer requires positive branches and finite scales");
}

Status HyperConnectionMixer::ComputeInternal(OpKernelContext* context) const {
  const Tensor* streams = context->Input<Tensor>(0);
  const Tensor* down = context->Input<Tensor>(1);
  const Tensor* up = context->Input<Tensor>(2);
  const auto& shape = streams->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() == 3 && shape[2] > 0 && shape[2] % branches_ == 0 &&
                        down->Shape().NumDimensions() == 2 && down->Shape()[0] == shape[2] &&
                        down->Shape()[1] > 0 && up->Shape() == TensorShape({down->Shape()[1], shape[2]}),
                    "HyperConnectionMixer requires [B,S,C*H], [C*H,R], [R,C*H]");
  const int64_t rows = SafeInt<int64_t>(shape[0]) * shape[1], rank = down->Shape()[1];
  auto d = GetScratchBuffer<BFloat16>(SafeInt<size_t>(rows) * rank, GetComputeStream(context));
  auto u = GetScratchBuffer<BFloat16>(SafeInt<size_t>(rows) * shape[2], GetComputeStream(context));
  Tensor* output = context->Output(0, TensorShape({shape[0], shape[1], shape[2] / branches_}));
  ORT_RETURN_IF_ERROR(ProjectionGemm(GetCublasHandle(context), GetDeviceProp(), UseTF32(),
      streams->Data<BFloat16>(), down, d.get(), rows, shape[2]));
  ORT_RETURN_IF_ERROR(LaunchScaledSiLU<ProjectionCudaT>(Stream(context),
      reinterpret_cast<const ProjectionCudaT*>(d.get()), nullptr, ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16,
      reinterpret_cast<ProjectionCudaT*>(d.get()), SafeInt<int64_t>(rows) * rank, alpha_));
  ORT_RETURN_IF_ERROR(ProjectionGemm(GetCublasHandle(context), GetDeviceProp(), UseTF32(),
      d.get(), up, u.get(), rows, rank));
  return LaunchHyperConnectionPreMix<ProjectionCudaT>(Stream(context),
      reinterpret_cast<const ProjectionCudaT*>(streams->Data<BFloat16>()), u.get(),
      ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16,
      reinterpret_cast<ProjectionCudaT*>(output->MutableData<BFloat16>()),
      SafeInt<int64_t>(rows) * (shape[2] / branches_), SafeInt<int>(branches_),
      SafeInt<int>(shape[2] / branches_), 3, reduction_scale_, 2);
}

MtpAttentionProjection::MtpAttentionProjection(const OpKernelInfo& info)
    : CudaKernel(info), heads_(info.GetAttrOrDefault<int64_t>("num_heads", 2)),
      rotary_dim_(info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 64)),
      epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-6f)) {
  ORT_ENFORCE(heads_ > 0 && rotary_dim_ > 0 && rotary_dim_ % 2 == 0 &&
                  std::isfinite(epsilon_) && epsilon_ >= 0,
              "MtpAttentionProjection requires positive heads/even rotary dimension and nonnegative epsilon");
}

Status MtpAttentionProjection::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const auto& shape = input->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() == 3 && shape[2] > 0,
                    "MtpAttentionProjection input must have shape [B,S,K]");
  const int64_t rows = SafeInt<int64_t>(shape[0]) * shape[1];
  const Tensor* weights[3] = {context->Input<Tensor>(1), context->Input<Tensor>(2), context->Input<Tensor>(3)};
  int64_t widths[3];
  for (int i = 0; i < 3; ++i) {
    ORT_RETURN_IF_NOT(weights[i]->Shape().NumDimensions() == 2 && weights[i]->Shape()[0] == shape[2] &&
                          weights[i]->Shape()[1] > 0, "MtpAttentionProjection weight shape mismatch");
    widths[i] = weights[i]->Shape()[1];
  }
  ORT_RETURN_IF_NOT(widths[1] % heads_ == 0 && widths[1] / heads_ <= 1024 &&
                        widths[1] / heads_ >= rotary_dim_,
                    "MtpAttentionProjection key head width must contain rotary dimension and be at most 1024");
  const int hidden = SafeInt<int>(widths[1] / heads_);
  const Tensor* scale = context->Input<Tensor>(4);
  const Tensor* cosine = context->Input<Tensor>(5);
  const Tensor* sine = context->Input<Tensor>(6);
  ORT_RETURN_IF_NOT(scale->Shape() == TensorShape({hidden}) &&
                        cosine->Shape() == TensorShape({shape[0], shape[1], rotary_dim_ / 2}) &&
                        sine->Shape() == cosine->Shape(), "MtpAttentionProjection norm/cache shape mismatch");
  Tensor* results[3];
  for (int i = 0; i < 3; ++i) results[i] = context->Output(i, TensorShape({shape[0], shape[1], widths[i]}));
  auto key = GetScratchBuffer<BFloat16>(SafeInt<size_t>(rows) * widths[1], GetComputeStream(context));
  for (int i = 0; i < 3; ++i) {
    ORT_RETURN_IF_ERROR(ProjectionGemm(GetCublasHandle(context), GetDeviceProp(), UseTF32(),
        input->Data<BFloat16>(), weights[i], i == 1 ? key.get() : results[i]->MutableData<BFloat16>(), rows, shape[2]));
  }
  if (rows == 0) return Status::OK();
  const int groups = SafeInt<int>(SafeInt<int64_t>(rows) * heads_);
  const auto buffer_size = onnxruntime::cuda::compute_reduce_matrix_columns_buffer_size<float>(groups, hidden);
  auto squares = GetScratchBuffer<float>(groups == 1 ? hidden : 0, GetComputeStream(context));
  auto sums = GetScratchBuffer<float>(groups, GetComputeStream(context));
  auto reduction = GetScratchBuffer<uint8_t>(buffer_size, GetComputeStream(context));
  ORT_RETURN_IF_ERROR(LaunchStandardBranchwiseRMSNorm<ProjectionCudaT>(
      Stream(context), reinterpret_cast<const ProjectionCudaT*>(key.get()), scale->DataRaw(),
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT, reinterpret_cast<ProjectionCudaT*>(key.get()),
      groups, SafeInt<int>(heads_), hidden, true, epsilon_, nullptr, squares.get(), sums.get(),
      reduction.get(), buffer_size, hidden, true));
  return onnxruntime::cuda::LaunchRotaryEmbeddingKernel<ProjectionCudaT>(
      Stream(context), reinterpret_cast<ProjectionCudaT*>(results[1]->MutableData<BFloat16>()),
      reinterpret_cast<const ProjectionCudaT*>(key.get()), nullptr,
      reinterpret_cast<const ProjectionCudaT*>(cosine->Data<BFloat16>()),
      reinterpret_cast<const ProjectionCudaT*>(sine->Data<BFloat16>()),
      SafeInt<int>(shape[0]), SafeInt<int>(shape[1]), SafeInt<int>(heads_), hidden,
      SafeInt<int>(rotary_dim_), SafeInt<int>(rows), 0, false, GetDeviceProp().maxThreadsPerBlock, false);
}

#define REGISTER_ARGMAX_LAST_AXIS(T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      ArgMaxLastAxis, kMSDomain, 1, T, kCudaExecutionProvider,             \
      (*KernelDefBuilder::Create())                                        \
            .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
            .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),  \
      ArgMaxLastAxis<T>);

REGISTER_ARGMAX_LAST_AXIS(float)
REGISTER_ARGMAX_LAST_AXIS(MLFloat16)
REGISTER_ARGMAX_LAST_AXIS(BFloat16)

#undef REGISTER_ARGMAX_LAST_AXIS

ONNX_OPERATOR_KERNEL_EX(
    EmbeddingRMSNorm, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
          .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
          .TypeConstraint("F", DataTypeImpl::GetTensorType<float>()),
    EmbeddingRMSNorm);

ONNX_OPERATOR_KERNEL_EX(
    RowSelect, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
          .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>()),
    RowSelect);

ONNX_OPERATOR_KERNEL_EX(
    DotProductGate, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>()),
    DotProductGate);

ONNX_OPERATOR_KERNEL_EX(
    SigmoidGateMul, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
          .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
          .TypeConstraint("TI", DataTypeImpl::GetTensorType<int32_t>())
          .InputMemoryType(OrtMemTypeCPUInput, 3),
    SigmoidGateMul);

#if !defined(DISABLE_FLOAT8_TYPES)
ONNX_OPERATOR_TYPED_KERNEL_EX(
    DequantizeLinear, kMSDomain, 1, Float8E4M3FN_BFloat16, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<Float8E4M3FN>())
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<BFloat16>())
          .TypeConstraint("TS", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinearFp8Bf16);
#endif

ONNX_OPERATOR_KERNEL_EX(
    UntileHyperConnectionWeights, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
          .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>()),
    UntileHyperConnectionWeights);

template <typename T>
ArgMaxLastAxis<T>::ArgMaxLastAxis(const OpKernelInfo& info)
    : CudaKernel(info), keepdims_(info.GetAttrOrDefault<int64_t>("keepdims", 0) != 0) {}

template <typename T>
Status ArgMaxLastAxis<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const auto& shape = input->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() >= 1 &&
                        shape[shape.NumDimensions() - 1] > 0,
                    "ArgMaxLastAxis requires a non-empty final dimension");
  const int64_t width = shape[shape.NumDimensions() - 1];
  const int64_t rows = shape.Size() / width;
  TensorShapeVector output_dims(shape.GetDims().begin(), shape.GetDims().end() - 1);
  if (keepdims_) output_dims.push_back(1);
  Tensor* output = context->Output(0, TensorShape(output_dims));
  const size_t partial_count = SafeInt<size_t>(rows) * GetArgMaxLastAxisPartitions(width);
  auto partial_values = GetScratchBuffer<float>(partial_count, GetComputeStream(context));
  auto partial_indices = GetScratchBuffer<int64_t>(partial_count, GetComputeStream(context));
  using CudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
  return LaunchArgMaxLastAxis(Stream(context), reinterpret_cast<const CudaT*>(input->Data<T>()),
                              input->GetElementType(), rows, width,
                              output->MutableData<int64_t>(), partial_values.get(), partial_indices.get());
}

EmbeddingRMSNorm::EmbeddingRMSNorm(const OpKernelInfo& info)
    : CudaKernel(info), epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-6f)) {
  ORT_ENFORCE(epsilon_ >= 0.0f, "EmbeddingRMSNorm epsilon must be non-negative");
}

Status EmbeddingRMSNorm::ComputeInternal(OpKernelContext* context) const {
  const Tensor* ids = context->Input<Tensor>(0);
  const Tensor* table = context->Input<Tensor>(1);
  const Tensor* weight = context->Input<Tensor>(2);
  ORT_RETURN_IF_NOT(table->Shape().NumDimensions() == 2,
                    "EmbeddingRMSNorm table must have shape [vocabulary, hidden]");
  const int64_t hidden_size = table->Shape()[1];
  ORT_RETURN_IF_NOT(weight->Shape().NumDimensions() == 1 &&
                        weight->Shape()[0] == hidden_size,
                    "EmbeddingRMSNorm weight must match the embedding hidden size");
  TensorShapeVector output_dims(ids->Shape().GetDims().begin(), ids->Shape().GetDims().end());
  output_dims.push_back(hidden_size);
  Tensor* output = context->Output(0, TensorShape(output_dims));
  return LaunchEmbeddingRMSNorm(Stream(context), ids->Data<int64_t>(),
                                reinterpret_cast<const uint16_t*>(table->Data<BFloat16>()),
                                weight->Data<float>(), epsilon_,
                                reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()),
                                ids->Shape().Size(), table->Shape()[0], hidden_size);
}

Status RowSelect::ComputeInternal(OpKernelContext* context) const {
  const Tensor* selection = context->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(selection->Shape().Size() == 1, "RowSelect selection must have one element");
  std::array<const uint16_t*, 3> inputs{};
  std::array<uint16_t*, 3> outputs{};
  std::array<int64_t, 3> row_sizes{};
  int64_t rows = 0;
  for (int i = 0; i < 3; ++i) {
    const Tensor* input = context->Input<Tensor>(i + 1);
    const auto& shape = input->Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() >= 2 && shape[0] == 1 && shape[1] >= 2 && shape[1] <= 4,
                      "RowSelect inputs must have shape [1, R, ...] with R in [2, 4]");
    if (i == 0) rows = shape[1];
    ORT_RETURN_IF_NOT(shape[1] == rows, "RowSelect inputs must have the same row count");
    row_sizes[i] = shape.Size() / rows;
    TensorShapeVector output_dims(shape.GetDims().begin(), shape.GetDims().end());
    output_dims[1] = 1;
    Tensor* output = context->Output(i, TensorShape(output_dims));
    inputs[i] = reinterpret_cast<const uint16_t*>(input->Data<BFloat16>());
    outputs[i] = reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>());
  }
  return LaunchRowSelect(Stream(context), selection->Data<int64_t>(), inputs.data(),
                         outputs.data(), row_sizes.data(), rows);
}

DotProductGate::DotProductGate(const OpKernelInfo& info)
    : CudaKernel(info), width_(info.GetAttrOrDefault<int64_t>("group_width", 0)),
      divisor_(info.GetAttrOrDefault<float>("divisor", 1.0f)),
      epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-6f)) {
  ORT_ENFORCE(width_ > 0 && std::isfinite(divisor_) && divisor_ > 0 &&
                  std::isfinite(epsilon_) && epsilon_ >= 0,
              "DotProductGate requires positive group_width/divisor and nonnegative epsilon");
}

Status DotProductGate::ComputeInternal(OpKernelContext* context) const {
  const Tensor* key = context->Input<Tensor>(0);
  const Tensor* query = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const auto& shape = key->Shape();
  ORT_RETURN_IF_NOT(shape == query->Shape() && shape.NumDimensions() > 0 &&
                        shape[shape.NumDimensions() - 1] > 0 &&
                        shape[shape.NumDimensions() - 1] % width_ == 0,
                    "DotProductGate key/query must have identical shapes with final dimension divisible by group_width");
  const auto& value_shape = value->Shape();
  ORT_RETURN_IF_NOT(value_shape.NumDimensions() == shape.NumDimensions(),
                    "DotProductGate value rank must match key");
  for (size_t i = 0; i + 1 < shape.NumDimensions(); ++i) {
    ORT_RETURN_IF_NOT(shape[i] == value_shape[i], "DotProductGate leading dimensions must match");
  }
  const int64_t last = shape[shape.NumDimensions() - 1];
  const int64_t value_last = value_shape[value_shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(value_last == width_ || value_last == last,
                    "DotProductGate value width must equal group_width or key width");
  Tensor* output = context->Output(0, shape);
  return LaunchDotProductGate(
      Stream(context), reinterpret_cast<const uint16_t*>(key->Data<BFloat16>()),
      reinterpret_cast<const uint16_t*>(query->Data<BFloat16>()),
      reinterpret_cast<const uint16_t*>(value->Data<BFloat16>()),
      reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()),
      shape.Size() / width_, width_, value_last == last ? 1 : last / width_, divisor_, epsilon_);
}

SigmoidGateMul::SigmoidGateMul(const OpKernelInfo& info)
    : CudaKernel(info),
      head_dim_(static_cast<int>(info.GetAttrOrDefault<int64_t>("head_dim", 0))),
      heads_(static_cast<int>(info.GetAttrOrDefault<int64_t>("num_heads", 0))),
      q_stride_(static_cast<int>(info.GetAttrOrDefault<int64_t>("q_stride", 0))),
      tok_stride_(info.GetAttrOrDefault<int64_t>("tok_stride", 0)),
      q_offset_(info.GetAttrOrDefault<int64_t>("q_offset", 0)),
      broadcast_gate_(info.GetAttrOrDefault<int64_t>("broadcast_gate", 0) == 1),
      sigmoid_in_fp32_(info.GetAttrOrDefault<int64_t>("sigmoid_in_fp32", 1) == 1),
      silu_gate_(info.GetAttrOrDefault<int64_t>("silu_gate", 0) == 1) {
  const int64_t broadcast = info.GetAttrOrDefault<int64_t>("broadcast_gate", 0);
  ORT_ENFORCE(broadcast == 0 || broadcast == 1, "SigmoidGateMul broadcast_gate must be zero or one");
  const int64_t fp32 = info.GetAttrOrDefault<int64_t>("sigmoid_in_fp32", 1);
  ORT_ENFORCE(fp32 == 0 || fp32 == 1, "SigmoidGateMul sigmoid_in_fp32 must be zero or one");
  const int64_t silu = info.GetAttrOrDefault<int64_t>("silu_gate", 0);
  ORT_ENFORCE(silu == 0 || silu == 1, "SigmoidGateMul silu_gate must be zero or one");
  ORT_ENFORCE(!(broadcast_gate_ && silu_gate_), "SigmoidGateMul broadcast_gate and silu_gate are exclusive");
  ORT_ENFORCE(broadcast_gate_ || silu_gate_ || (head_dim_ > 0 && heads_ > 0 &&
                  q_stride_ >= 2 * head_dim_ &&
                  tok_stride_ > 0 && q_offset_ >= 0),
              "SigmoidGateMul received invalid layout attributes");
  ORT_ENFORCE(!(broadcast_gate_ || silu_gate_) || (head_dim_ == 0 && heads_ == 0 && q_stride_ == 0 &&
                                      tok_stride_ == 0 && q_offset_ == 0),
              "SigmoidGateMul broadcast_gate/silu_gate cannot be combined with packed layout attributes");
}

Status SigmoidGateMul::ComputeInternal(OpKernelContext* context) const {
  const Tensor* numerics = context->Input<Tensor>(3);
  ORT_RETURN_IF_NOT(numerics == nullptr ||
                        (numerics->IsDataType<int32_t>() && numerics->Shape().Size() == 1 &&
                         (numerics->Data<int32_t>()[0] == 0 || numerics->Data<int32_t>()[0] == 1)),
                    "prefill_numerics must be an int32 scalar containing 0 or 1");
  const bool dense_prefill = numerics != nullptr && numerics->Data<int32_t>()[0] == 1;
  const Tensor* attention = context->Input<Tensor>(0);
  const Tensor* projection = context->Input<Tensor>(1);
  const Tensor* residual = context->Input<Tensor>(2);
  const Tensor* gate_weight = context->Input<Tensor>(4);
  const int64_t count = attention->Shape().Size();
  ORT_RETURN_IF_NOT(residual == nullptr || residual->Shape() == attention->Shape(),
                    "SigmoidGateMul residual must have the same shape as attention");
  if (gate_weight != nullptr) {
    ORT_RETURN_IF_NOT(broadcast_gate_ && !silu_gate_,
                      "SigmoidGateMul gate_weight requires broadcast_gate without silu_gate");
    const auto& shape = attention->Shape();
    const auto& activation_shape = projection->Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() >= 1 && shape[shape.NumDimensions() - 1] > 0 &&
                          activation_shape.NumDimensions() == shape.NumDimensions() &&
                          activation_shape[activation_shape.NumDimensions() - 1] > 0,
                      "SigmoidGateMul projected gate requires matching ranks and positive final dimensions");
    for (size_t axis = 0; axis + 1 < shape.NumDimensions(); ++axis) {
      ORT_RETURN_IF_NOT(shape[axis] == activation_shape[axis],
                        "SigmoidGateMul projected gate leading dimensions must match attention");
    }
    const int64_t input_width = activation_shape[activation_shape.NumDimensions() - 1];
    ORT_RETURN_IF_NOT(gate_weight->Shape().NumDimensions() == 2 &&
                          gate_weight->Shape()[0] == input_width && gate_weight->Shape()[1] == 1,
                      "SigmoidGateMul gate_weight must have shape [input_width, 1]");
    Tensor* output = context->Output(0, shape);
    return LaunchProjectedSigmoidGateMul(
        Stream(context), reinterpret_cast<const uint16_t*>(attention->Data<BFloat16>()),
        reinterpret_cast<const uint16_t*>(projection->Data<BFloat16>()),
        reinterpret_cast<const uint16_t*>(gate_weight->Data<BFloat16>()),
        residual == nullptr ? nullptr : reinterpret_cast<const uint16_t*>(residual->Data<BFloat16>()),
        reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()),
        count / shape[shape.NumDimensions() - 1], input_width, shape[shape.NumDimensions() - 1],
        dense_prefill ? false : sigmoid_in_fp32_);
  }
  int64_t gate_row_width = 0;
  if (silu_gate_) {
    ORT_RETURN_IF_NOT(projection->Shape() == attention->Shape(),
                      "SigmoidGateMul silu gate must have the same shape as attention");
  } else if (broadcast_gate_) {
    TensorShapeVector expected(attention->Shape().GetDims().begin(), attention->Shape().GetDims().end());
    ORT_RETURN_IF_NOT(!expected.empty() && expected.back() > 0,
                      "SigmoidGateMul broadcast_gate requires a positive final attention dimension");
    gate_row_width = expected.back();
    expected.back() = 1;
    ORT_RETURN_IF_NOT(projection->Shape() == TensorShape(expected),
                      "SigmoidGateMul broadcast gate must match attention with a singleton final dimension");
  } else {
    const int64_t row_width = static_cast<int64_t>(heads_) * head_dim_;
    ORT_RETURN_IF_NOT(count % row_width == 0,
                    "SigmoidGateMul attention size must be divisible by heads * head_dim");
    const int64_t tokens = count / row_width;
    ORT_RETURN_IF_NOT(projection->Shape().Size() >=
                        (tokens - 1) * tok_stride_ + q_offset_ +
                            static_cast<int64_t>(heads_) * q_stride_,
                    "SigmoidGateMul projection is too small for the configured layout");
  }
  Tensor* output = context->Output(0, attention->Shape());
  return LaunchSigmoidGateMul(
      Stream(context), reinterpret_cast<const uint16_t*>(attention->Data<BFloat16>()),
      reinterpret_cast<const uint16_t*>(projection->Data<BFloat16>()),
      reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()), count, head_dim_, heads_,
      q_stride_, tok_stride_, q_offset_, gate_row_width,
      residual == nullptr ? nullptr : reinterpret_cast<const uint16_t*>(residual->Data<BFloat16>()),
      dense_prefill ? false : sigmoid_in_fp32_, silu_gate_);
}

DequantizeLinearFp8Bf16::DequantizeLinearFp8Bf16(const OpKernelInfo& info)
    : CudaKernel(info), axis_(info.GetAttrOrDefault<int64_t>("axis", 1)) {}

Status DequantizeLinearFp8Bf16::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* scales = context->Input<Tensor>(1);
  const Tensor* zero_point = context->Input<Tensor>(2);
  const auto& input_shape = input->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() > 0,
                    "DequantizeLinear FP8/BF16 input must have rank at least one");
  int64_t axis = axis_ < 0 ? axis_ + static_cast<int64_t>(input_shape.NumDimensions()) : axis_;

  int64_t axis_size = 1;
  int64_t inner_size = 1;
  if (scales->Shape().Size() != 1) {
    ORT_RETURN_IF_NOT(axis >= 0 && axis < static_cast<int64_t>(input_shape.NumDimensions()),
                      "DequantizeLinear axis is out of range");
    ORT_RETURN_IF_NOT(scales->Shape().NumDimensions() == 1 &&
                          scales->Shape().Size() == input_shape[axis],
                      "DequantizeLinear scale must be scalar or match the selected axis");
    axis_size = scales->Shape().Size();
    inner_size = input_shape.SizeFromDimension(static_cast<size_t>(axis + 1));
  }
  if (zero_point != nullptr) {
    ORT_RETURN_IF_NOT(zero_point->Shape().Size() == 1 ||
                          zero_point->Shape() == scales->Shape(),
                      "DequantizeLinear zero point must be scalar or match the scale");
  }

  Tensor* output = context->Output(0, input_shape);
  return LaunchDequantizeLinearFp8Bf16(
      Stream(context), reinterpret_cast<const uint8_t*>(input->DataRaw()),
      zero_point == nullptr ? nullptr : reinterpret_cast<const uint8_t*>(zero_point->DataRaw()),
      scales->Data<float>(),
      reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()), input_shape.Size(),
      axis_size, inner_size, zero_point == nullptr || zero_point->Shape().Size() == 1);
}

UntileHyperConnectionWeights::UntileHyperConnectionWeights(const OpKernelInfo& info)
    : CudaKernel(info), kind_(static_cast<int>(info.GetAttrOrDefault<int64_t>("kind", -1))) {
  ORT_ENFORCE(kind_ == 0 || kind_ == 1, "UntileHyperConnectionWeights kind must be 0 or 1");
}

Status UntileHyperConnectionWeights::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(input->Shape() == TensorShape({160, 20480}),
                    "UntileHyperConnectionWeights input must have shape [160, 20480]");
  const TensorShape output_shape = kind_ == 0 ? TensorShape({10240, 320}) : TensorShape({320, 10240});
  Tensor* output = context->Output(0, output_shape);
  return LaunchUntileHyperConnectionWeights(
      Stream(context), reinterpret_cast<const uint16_t*>(input->Data<BFloat16>()),
      reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()), kind_);
}

template class ArgMaxLastAxis<float>;
template class ArgMaxLastAxis<MLFloat16>;
template class ArgMaxLastAxis<BFloat16>;

}  // namespace onnxruntime::contrib::cuda
