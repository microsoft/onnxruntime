// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp4.h"

#include <algorithm>
#include <type_traits>

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime::contrib::cuda {
using namespace onnxruntime::cuda;

ONNX_OPERATOR_KERNEL_EX(
    MatMulBlockQuantizedFp4Weight,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, BFloat16>())
        .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<uint8_t>())
        .TypeConstraint("T3", BuildKernelDefConstraints<float>()),
    MatMulBlockQuantizedFp4Weight);

namespace {

constexpr int kWeightScaleInputIndex = 2;

bool IsNativeSm120Fp4Enabled() {
  return ParseEnvironmentVariableWithDefault<bool>("ORT_MATMUL_BLOCK_SCALED_FP4_NATIVE_SM120", false);
}

int64_t RoundUp(int64_t value, int64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

MatMulBlockQuantizedFp4Weight::MatMulBlockQuantizedFp4Weight(const OpKernelInfo& info) : CudaKernel(info) {
  block_size_ = info.GetAttrOrDefault<int64_t>("block_size", static_cast<int64_t>(16));
  ORT_ENFORCE(block_size_ > 0, "block_size must be positive, got ", block_size_);
  sm_ = GetDeviceProp().major * 10 + GetDeviceProp().minor;
}

Status MatMulBlockQuantizedFp4Weight::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                              bool& is_packed, PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;

#if defined(ORT_ENABLE_BLOCKQUANT_SM120)
  if (input_idx != kWeightScaleInputIndex || !IsNativeSm120Fp4Enabled() || sm_ < 120 || sm_ >= 130 ||
      block_size_ != 16) {
    return Status::OK();
  }

  // N and K are derived from the scale shape [N, K / block_size]. This is exact only when K is a
  // multiple of block_size, which the native SM120 path requires anyway (K % 32 == 0).
  const auto& scale_shape = tensor.Shape();
  if (scale_shape.NumDimensions() != 2) {
    return Status::OK();
  }
  const int64_t n = scale_shape[0];
  const int64_t k_blocks = scale_shape[1];
  const int64_t k = k_blocks * block_size_;
  if (k % 32 != 0 || n % 32 != 0) {
    return Status::OK();
  }

  const int64_t rounded_k_blocks = RoundUp(k_blocks, 4);
  const int64_t rounded_n = RoundUp(n, 128);
  b_scale_prepacked_ = IAllocator::MakeUniquePtr<uint8_t>(
      alloc, SafeInt<size_t>(rounded_n) * SafeInt<size_t>(rounded_k_blocks), true);

  // PrePack runs once at session initialization and is not given an EP compute stream (the
  // OpKernel::PrePack contract has no stream parameter), so the repack is issued on the per-thread
  // default stream rather than on cudaStreamLegacy: the legacy default stream implicitly
  // synchronizes with every other blocking stream in the process, which would serialize unrelated
  // sessions initializing concurrently.
  //
  // The cudaStreamSynchronize below is required for correctness, not just ordering: the staging
  // buffer `weight_scale_device` and the source `tensor` are released as soon as this function
  // returns, and `b_scale_prepacked_` must be fully written before any Run() reads it from a
  // different stream. It costs one synchronization per session, outside the inference hot path.
  cudaStream_t stream = cudaStreamPerThread;
  const void* weight_scale = tensor.DataRaw();
  IAllocatorUniquePtr<uint8_t> weight_scale_device;
  if (tensor.Location().device.Type() != OrtDevice::GPU) {
    const size_t weight_scale_bytes = SafeInt<size_t>(n) * SafeInt<size_t>(k_blocks);
    weight_scale_device = IAllocator::MakeUniquePtr<uint8_t>(alloc, weight_scale_bytes, true);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(weight_scale_device.get(), weight_scale, weight_scale_bytes,
                                         cudaMemcpyDefault, stream));
    weight_scale = weight_scale_device.get();
  }

  ORT_RETURN_IF_ERROR(LaunchRepackWeightScaleNvFp4ForNativeSm120(
      b_scale_prepacked_.get(), weight_scale, SafeInt<int>(n), SafeInt<int>(k), SafeInt<int>(block_size_), stream));
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
#else
  ORT_UNUSED_PARAMETER(tensor);
  ORT_UNUSED_PARAMETER(input_idx);
  ORT_UNUSED_PARAMETER(alloc);
#endif

  return Status::OK();
}

template <typename T>
Status MatMulBlockQuantizedFp4Weight::ComputeImpl(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* a = context->Input<Tensor>(0);
  const Tensor* b = context->Input<Tensor>(1);
  const Tensor* weight_scale = context->Input<Tensor>(2);
  const Tensor* weight_scale_2 = context->Input<Tensor>(3);
  const Tensor* input_scale = context->Input<Tensor>(4);  // optional
  const Tensor* bias = context->Input<Tensor>(5);         // optional

  const auto& a_shape = a->Shape();
  ORT_ENFORCE(a_shape.NumDimensions() >= 1, "A must have rank at least 1.");

  // N and K are derived from the packed weight shape [N, K/2] instead of from attributes.
  const auto& b_shape = b->Shape();
  ORT_ENFORCE(b_shape.NumDimensions() == 2, "B must have shape [N, K/2], got ", b_shape.ToString(), ".");
  const int64_t n = b_shape[0];
  const int64_t k = b_shape[1] * 2;
  ORT_ENFORCE(a_shape[a_shape.NumDimensions() - 1] == k,
              "A's last dimension (", a_shape[a_shape.NumDimensions() - 1],
              ") must equal K (", k, ") derived from B's shape ", b_shape.ToString(), ".");

  const int64_t k_blocks = (k + block_size_ - 1) / block_size_;
  const auto& weight_scale_shape = weight_scale->Shape();
  ORT_ENFORCE(weight_scale_shape.NumDimensions() == 2 && weight_scale_shape[0] == n &&
                  weight_scale_shape[1] == k_blocks,
              "weight_scale must have shape [N, ceil(K/block_size)] = [", n, ", ", k_blocks, "], got ",
              weight_scale_shape.ToString(), ".");
  ORT_ENFORCE(weight_scale_2->Shape().Size() == 1, "weight_scale_2 must be a scalar.");
  if (input_scale != nullptr) {
    ORT_ENFORCE(input_scale->Shape().Size() == 1, "input_scale must be a scalar.");
    // input_scale is used only by the opt-in native NVFP4 x NVFP4 path. The default
    // weight-only FP16/BF16 activation path keeps full-precision activations.
  }
  if (bias != nullptr) {
    ORT_ENFORCE(bias->Shape().NumDimensions() == 1 && bias->Shape()[0] == n,
                "bias must have shape [N] = [", n, "], got ", bias->Shape().ToString(), ".");
  }

  constexpr bool transa = false;
  constexpr bool transb = true;
  MatMulComputeHelper helper;
  TensorShape b_logical_shape({n, k});
  ORT_RETURN_IF_ERROR(helper.Compute(a_shape, b_logical_shape, transa, transb));

  Tensor* Y = context->Output(0, helper.OutputShape());
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int m_i = SafeInt<int>(helper.M());
  const int n_i = SafeInt<int>(helper.N());
  const int k_i = SafeInt<int>(helper.K());

  // Degenerate K == 0 reduces to an empty sum, so Y is zero plus the optional bias. Handle it
  // here so that the specialized paths below can assume K > 0 (K % 32 == 0 is also true for 0).
  if (k_i == 0) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(Y->MutableDataRaw(), 0, Y->SizeInBytes(), Stream(context)));
    if (bias != nullptr) {
      ORT_RETURN_IF_ERROR(LaunchAddBiasNvFp4(
          Y->MutableDataRaw(),
          bias->DataRaw(),
          m_i,
          n_i,
          std::is_same<T, BFloat16>::value,
          Stream(context)));
    }
    return Status::OK();
  }

  // Decode fast path: for small M (autoregressive generation) this is a memory-bound GEMV.
  // A fused warp-per-column kernel reads the packed NVFP4 weight directly, avoiding both the
  // [N, K] dequant scratch buffer and the cuBLAS GEMM (which is underutilized at M == 1).
  constexpr int kGemvMaxM = 8;
  if (m_i > 0 && m_i <= kGemvMaxM && block_size_ == 16 && (k_i % 32 == 0)) {
    return LaunchMatMulBlockQuantizedFp4WeightGemv(
        Y->MutableDataRaw(),
        a->DataRaw(),
        b->DataRaw(),
        weight_scale->DataRaw(),
        weight_scale_2->Data<float>(),
        bias != nullptr ? bias->DataRaw() : nullptr,
        m_i,
        n_i,
        k_i,
        SafeInt<int>(block_size_),
        std::is_same<T, BFloat16>::value,
        GetDeviceProp(),
        Stream(context));
  }

#if defined(ORT_ENABLE_BLOCKQUANT_SM120)
  if (IsNativeSm120Fp4Enabled() && sm_ >= 120 && sm_ < 130 && block_size_ == 16 &&
      (k_i % 32 == 0) && (n_i % 32 == 0)) {
    constexpr int64_t kScaleVectorSize = 16;
    const int64_t k_scale_blocks = RoundUp(k_i / kScaleVectorSize, 4);
    const int64_t rounded_m = RoundUp(m_i, 128);
    const int64_t rounded_n = RoundUp(n_i, 128);

    auto* stream = GetComputeStream(context);
    auto a_packed = GetScratchBuffer<uint8_t>(SafeInt<size_t>(m_i) * SafeInt<size_t>(k_i / 2), stream);
    auto a_scale = GetScratchBuffer<uint8_t>(SafeInt<size_t>(rounded_m) * SafeInt<size_t>(k_scale_blocks), stream);
    IAllocatorUniquePtr<uint8_t> b_scale;
    const void* b_scale_data = b_scale_prepacked_.get();
    if (b_scale_data == nullptr) {
      b_scale = GetScratchBuffer<uint8_t>(SafeInt<size_t>(rounded_n) * SafeInt<size_t>(k_scale_blocks), stream);
      ORT_RETURN_IF_ERROR(LaunchRepackWeightScaleNvFp4ForNativeSm120(
          b_scale.get(), weight_scale->DataRaw(), n_i, k_i, SafeInt<int>(block_size_), Stream(context)));
      b_scale_data = b_scale.get();
    }
    auto alpha = GetScratchBuffer<float>(1, stream);
    const size_t workspace_size = GetMatMulBlockQuantizedFp4WeightNativeSm120WorkspaceSize(
        m_i, n_i, k_i, std::is_same<T, BFloat16>::value);
    auto workspace = GetScratchBuffer<uint8_t>(workspace_size, stream);

    ORT_RETURN_IF_ERROR(LaunchMatMulBlockQuantizedFp4WeightNativeSm120(
        Y->MutableDataRaw(),
        a->DataRaw(),
        b->DataRaw(),
        weight_scale_2->Data<float>(),
        input_scale != nullptr ? input_scale->Data<float>() : nullptr,
        a_packed.get(),
        a_scale.get(),
        b_scale_data,
        alpha.get(),
        m_i,
        n_i,
        k_i,
        SafeInt<int>(block_size_),
        std::is_same<T, BFloat16>::value,
        workspace.get(),
        workspace_size,
        Stream(context)));

    if (bias != nullptr) {
      ORT_RETURN_IF_ERROR(LaunchAddBiasNvFp4(
          Y->MutableDataRaw(),
          bias->DataRaw(),
          m_i,
          n_i,
          std::is_same<T, BFloat16>::value,
          Stream(context)));
    }

    return Status::OK();
  }
#endif

  // Dequantize the packed NVFP4 weight into a scratch [N, K] buffer of the activation type.
  IAllocatorUniquePtr<CudaT> b_dequant = GetScratchBuffer<CudaT>(SafeInt<size_t>(n) * SafeInt<size_t>(k),
                                                                 GetComputeStream(context));
  ORT_RETURN_IF_ERROR(LaunchDequantizeNvFp4(
      b_dequant.get(),
      b->DataRaw(),
      weight_scale->DataRaw(),
      weight_scale_2->Data<float>(),
      SafeInt<int>(n),
      SafeInt<int>(k),
      SafeInt<int>(block_size_),
      std::is_same<T, BFloat16>::value,
      Stream(context)));

  const CudaT alpha = ToCudaType<T>::FromFloat(1.f);
  const CudaT zero = ToCudaType<T>::FromFloat(0.f);

  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      GetCublasHandle(context),
      CUBLAS_OP_T,  // transB: dequantized weight is [N, K] row-major == K-major [K, N]
      CUBLAS_OP_N,  // transA
      n_i,
      m_i,
      k_i,
      &alpha,
      b_dequant.get(),
      helper.Ldb(transb),
      reinterpret_cast<const CudaT*>(a->DataRaw()),
      helper.Lda(transa),
      &zero,
      reinterpret_cast<CudaT*>(Y->MutableDataRaw()),
      helper.Ldc(),
      GetDeviceProp(),
      UseTF32()));

  if (bias != nullptr) {
    ORT_RETURN_IF_ERROR(LaunchAddBiasNvFp4(
        Y->MutableDataRaw(),
        bias->DataRaw(),
        m_i,
        n_i,
        std::is_same<T, BFloat16>::value,
        Stream(context)));
  }

  return Status::OK();
}

Status MatMulBlockQuantizedFp4Weight::ComputeInternal(OpKernelContext* context) const {
  const Tensor* a = context->Input<Tensor>(0);
  if (a->IsDataType<MLFloat16>()) {
    return ComputeImpl<MLFloat16>(context);
  }
  if (a->IsDataType<BFloat16>()) {
    return ComputeImpl<BFloat16>(context);
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "MatMulBlockQuantizedFp4Weight only supports FP16 or BF16 activations.");
}

}  // namespace onnxruntime::contrib::cuda
