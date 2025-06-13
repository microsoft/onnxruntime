// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_nbits.h"

#include <cstdint>

#include "core/common/status.h"
#include "core/framework/float16.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cpu/utils/dump_tensor.h"
#include "contrib_ops/cuda/quantization/matmul_nbits.cuh"
#include "contrib_ops/cuda/quantization/dequantize_blockwise.cuh"
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_adaptor.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors.h"
#include "contrib_ops/cpu/quantization/matmul_nbits_helper.h"

constexpr int MatMulNBits_Input_B = 1;
constexpr int MatMulNBits_Input_Scale = 2;
constexpr int MatMulNBits_Input_ZeroPoint = 3;

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;
using onnxruntime::llm::kernels::weight_only::GemmPluginProfilerManager;
using onnxruntime::llm::kernels::weight_only::WeightOnlyGroupwiseQuantGemmPluginProfiler;
using onnxruntime::llm::kernels::weight_only::WeightTypeId;
static GemmPluginProfilerManager<WeightOnlyGroupwiseQuantGemmPluginProfiler> s_profilerManager;

constexpr auto kScaleAndZeros = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS;
constexpr auto kScaleOnly = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;

template <typename T>
void MatMulNBits<T>::InitGemmProfiler(int sm) {
  gemmProfiler_ = s_profilerManager.createGemmPluginProfiler(/*inference*/ false);

  if constexpr (std::is_same_v<T, MLFloat16>) {
    if (has_zero_points_) {
      if (nbits_ == 8) {
        weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<half, uint8_t, kScaleAndZeros>>();
      } else if (nbits_ == 4) {
        weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, kScaleAndZeros>>();
      }
    } else {
      if (nbits_ == 8) {
        weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<half, uint8_t, kScaleOnly>>();
      } else if (nbits_ == 4) {
        weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, kScaleOnly>>();
      }
    }
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    if (has_zero_points_) {
      if (nbits_ == 8) {
        weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<__nv_bfloat16, uint8_t, kScaleAndZeros>>();
      } else if (nbits_ == 4) {
        weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<__nv_bfloat16, cutlass::uint4b_t, kScaleAndZeros>>();
      }
    } else {
      if (nbits_ == 8) {
        weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<__nv_bfloat16, uint8_t, kScaleOnly>>();
      } else if (nbits_ == 4) {
        weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<__nv_bfloat16, cutlass::uint4b_t, kScaleOnly>>();
      }
    }
  }

  using onnxruntime::llm::kernels::fpA_intB_gemv::KernelType;
  KernelType cuda_kernel_type = nbits_ == 8 ? KernelType::FP16Int8Groupwise : KernelType::FP16Int4Groupwise;
  gemmProfiler_->setCudaKernelType(cuda_kernel_type, sm);
  gemmProfiler_->setQuant(nbits_, has_bias_, has_zero_points_);
  gemmProfiler_->setGroupSize(block_size_);
}

template <typename T>
void MatMulNBits<T>::RunGemmProfile(bool hasWeightOnlyCudaKernel, int min_m, int max_m) {
  // Number of 16-bit elements after casting int8/int4 to fp16.
  int n_16b = N_ / (nbits_ == 8 ? 2 : 4);

  gemmId_ = GemmIdCore(n_16b, K_, onnxruntime::llm::nvinfer::DataType::kHALF);

  GemmDims dims = {min_m, max_m, n_16b, K_};
  gemmProfiler_->profileTactics(weightOnlyGemmRunner_, gemmId_.dtype, dims, gemmId_, hasWeightOnlyCudaKernel);
}

template <typename T>
Status MatMulNBits<T>::PrePack(const Tensor& /* tensor */, int /* input_idx */, AllocatorPtr /*alloc*/,
                               /*out*/ bool& is_packed,
                               /*out*/ PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;
  return Status::OK();
}

template <>
Status MatMulNBits<MLFloat16>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                       bool& is_packed,
                                       PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;
  if (has_fpA_intB_gemm_) {
    cudaStream_t stream = cudaStreamLegacy;  // Use default stream for prepacking.

#ifdef FPA_INTB_GEMM_LATENCY
    std::cout << "Prepack for input " << input_idx << ", N=" << N_ << ", K=" << K_ << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
#endif

    if (input_idx == MatMulNBits_Input_B) {
      ORT_RETURN_IF_ERROR(PrePack_B(tensor, alloc, stream));
      is_packed = true;
    } else if (input_idx == MatMulNBits_Input_Scale) {
      ORT_RETURN_IF_ERROR(PrePack_Scale(tensor, alloc, stream));
      is_packed = true;
    } else if (input_idx == MatMulNBits_Input_ZeroPoint) {
      if (has_zero_points_) {
        ORT_RETURN_IF_ERROR(PrePack_ZeroPoint(tensor, alloc, stream));
        is_packed = true;
      }
    }

#ifdef FPA_INTB_GEMM_LATENCY
    auto end = std::chrono::high_resolution_clock::now();
    auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Latency: " << latency_us << " microseconds" << K_ << std::endl;
#endif
  }

  return Status::OK();
}

template <typename T>
Status MatMulNBits<T>::PrePack_B([[maybe_unused]] const Tensor& tensor,
                                 [[maybe_unused]] AllocatorPtr alloc,
                                 [[maybe_unused]] cudaStream_t stream) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    size_t n = static_cast<size_t>(N_);
    size_t k = static_cast<size_t>(K_);

    size_t packed_weight_bytes = n * k / (8 / nbits_);

    // uint8 does not need to be packed so we do not need to allocate extra space.
    IAllocatorUniquePtr<void> packed_transposed_weight_space = this->GetTransientScratchBuffer<void>(packed_weight_bytes);
    int8_t* packed_transposed_weight = reinterpret_cast<int8_t*>(packed_transposed_weight_space.get());

    fpA_intB_weight_buffer_ = IAllocator::MakeUniquePtr<void>(alloc, packed_weight_bytes, true);  // Transient buffer.

    int8_t* preprocessed_weight = reinterpret_cast<int8_t*>(fpA_intB_weight_buffer_.get());

    const uint8_t* blob_data = tensor.Data<uint8_t>();
    if (nbits_ == 4) {
      // Transpose the weight and add default zero point.
      onnxruntime::llm::kernels::fpA_intB_gemv::unpack_uint4_transposed_to_int8_direct_cuda(
          stream, packed_transposed_weight, blob_data, n, k);
    } else {
      onnxruntime::llm::kernels::fpA_intB_gemv::transpose_uint8_matrix_and_convert_to_int8(
          stream, packed_transposed_weight, blob_data, n, k);
    }

    using onnxruntime::llm::kernels::weight_only::QuantType;
    QuantType quant_type = nbits_ == 4 ? QuantType::W4_A16 : QuantType::W8_A16;

    auto permutation_map_buffer = this->GetTransientScratchBuffer<int32_t>(32);
    onnxruntime::llm::kernels::weight_only::preprocess_weights_for_mixed_gemm_cuda(
        stream,
        sm_,
        preprocessed_weight,
        packed_transposed_weight,
        permutation_map_buffer.get(),
        {static_cast<size_t>(k), static_cast<size_t>(n)},
        quant_type);

    DUMP_TENSOR_INIT();
    DUMP_TENSOR_D("packed transposed_weight in GPU", packed_transposed_weight, k, n * nbits_ / 8);
    DUMP_TENSOR_D("preprocessed_weight", reinterpret_cast<uint8_t*>(preprocessed_weight), k, n * nbits_ / 8);
  }

  return Status::OK();
}

template <typename T>
Status MatMulNBits<T>::PrePack_Scale([[maybe_unused]] const Tensor& tensor,
                                     [[maybe_unused]] AllocatorPtr alloc,
                                     [[maybe_unused]] cudaStream_t stream) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    size_t n = static_cast<size_t>(N_);
    size_t k = static_cast<size_t>(K_);

    size_t k_blocks = (k + block_size_ - 1) / block_size_;
    size_t scale_bytes = n * k_blocks * sizeof(T);

    fpA_intB_scale_buffer_ = IAllocator::MakeUniquePtr<void>(alloc, scale_bytes, true);  // Transient buffer.

    typedef typename ToCudaType<T>::MappedType CudaT;
    CudaT* transposed_scales = reinterpret_cast<CudaT*>(fpA_intB_scale_buffer_.get());

    onnxruntime::llm::kernels::fpA_intB_gemv::launch_transpose_scale_kernel<CudaT>(
        stream, reinterpret_cast<const CudaT*>(tensor.Data<T>()), transposed_scales, n, k_blocks);
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

    DUMP_TENSOR_INIT();
    DUMP_TENSOR_D("transposed_scales", transposed_scales, k_blocks, n);
  }
  return Status::OK();
}

template <typename T>
Status MatMulNBits<T>::PrePack_ZeroPoint([[maybe_unused]] const Tensor& tensor,
                                         [[maybe_unused]] AllocatorPtr alloc,
                                         [[maybe_unused]] cudaStream_t stream) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    size_t n = static_cast<size_t>(N_);
    size_t k = static_cast<size_t>(K_);

    size_t k_blocks = (k + block_size_ - 1) / block_size_;
    size_t scale_bytes = n * k_blocks * sizeof(T);

    typedef typename ToCudaType<T>::MappedType CudaT;
    const CudaT* transposed_scales = reinterpret_cast<const CudaT*>(fpA_intB_scale_buffer_.get());

    fpA_intB_zero_buffer_ = IAllocator::MakeUniquePtr<void>(alloc, scale_bytes, true);  // Transient buffer.
    CudaT* scaled_zero_points = reinterpret_cast<CudaT*>(fpA_intB_zero_buffer_.get());

    constexpr float kDefaultZeroPoint4Bit = 8.0f;
    constexpr float kDefaultZeroPoint8Bit = 128.0f;
    const float default_zero_point = nbits_ == 4 ? kDefaultZeroPoint4Bit : kDefaultZeroPoint8Bit;
    const auto* zero_points_data = tensor.DataRaw();

    // The scaled zero point will be zero for the default zero point, so there is no need to scale when it is nullptr.
    if (!tensor.IsDataType<T>()) {  // zero point is uint8_t type
      if (nbits_ == 4) {
        onnxruntime::llm::kernels::fpA_intB_gemv::launch_scaled_zero_point_kernel<true, CudaT, uint8_t>(
            stream, reinterpret_cast<const uint8_t*>(zero_points_data),
            transposed_scales, scaled_zero_points, n, k_blocks, default_zero_point);
      } else {
        onnxruntime::llm::kernels::fpA_intB_gemv::launch_scaled_zero_point_kernel<false, CudaT, uint8_t>(
            stream, reinterpret_cast<const uint8_t*>(zero_points_data),
            transposed_scales, scaled_zero_points, n, k_blocks, default_zero_point);
      }
    } else {  // zero point is not uint8_t type
      onnxruntime::llm::kernels::fpA_intB_gemv::launch_scaled_zero_point_kernel<false, CudaT, CudaT>(
          stream, reinterpret_cast<const CudaT*>(zero_points_data),
          transposed_scales, scaled_zero_points, n, k_blocks, default_zero_point);
    }
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

    DUMP_TENSOR_INIT();
    DUMP_TENSOR_D("scaled_zero_points", scaled_zero_points, k_blocks, n);
  }
  return Status::OK();
}

  inline int nextPowerOfTwo(int v) {
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
  }

template <typename T>
Status MatMulNBits<T>::ComputeInternal(OpKernelContext* ctx) const {
  const bool is_prepacked = has_fpA_intB_gemm_;
  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b = is_prepacked ? nullptr : ctx->Input<Tensor>(1);
  const Tensor* scales = is_prepacked ? nullptr : ctx->Input<Tensor>(2);
  const Tensor* zero_points = is_prepacked ? nullptr : ctx->Input<Tensor>(3);
  const Tensor* reorder_idx = ctx->Input<Tensor>(4);
  const Tensor* bias = ctx->Input<Tensor>(5);

  if (bias != nullptr) {
    ORT_THROW("MatMulNBits does not support bias in CUDA kernel");
  }

  ORT_RETURN_IF_ERROR(matmul_nbits_helper::CheckInputs<Tensor>(
      a, b, scales, zero_points, reorder_idx, bias, N_, K_, block_size_, nbits_));

  const auto* a_data = a->Data<T>();
  const uint8_t* blob_data = is_prepacked ? nullptr : b->Data<uint8_t>();
  const auto* scales_data = is_prepacked ? nullptr : scales->Data<T>();
  const auto* zero_points_data = (is_prepacked || zero_points == nullptr) ? nullptr : zero_points->DataRaw();
  const auto* reorder_idx_data = reorder_idx == nullptr ? nullptr : reorder_idx->Data<int32_t>();
  const auto* bias_data = bias == nullptr ? nullptr : bias->Data<T>();

  constexpr bool transa = false;
  constexpr bool transb = true;
  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, transa, transb));

  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

  cudaStream_t stream = static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle());

  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT* out_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());

  int m = SafeInt<int>(helper.M());
  int n = SafeInt<int>(helper.N());
  int k = SafeInt<int>(helper.K());

  DUMP_TENSOR_INIT();

  if constexpr (std::is_same<T, MLFloat16>::value) {
    if (has_fpA_intB_gemm_) {
      if (m > max_m_) {
        auto next_m = nextPowerOfTwo(m);

#ifdef FPA_INTB_GEMM_LATENCY
        std::cout << "Gemm Profile for N=" << N_ << ", K=" << K_ << ", M=" << max_m_ << "~" << next_m << std::endl;
        auto latency_us = measure_latency([&]() {
#endif

        int n_16b = N_ / (nbits_ == 8 ? 2 : 4);
        GemmDims dims = {max_m_, next_m, n_16b, K_};
        gemmProfiler_->profileTactics(weightOnlyGemmRunner_, gemmId_.dtype, dims, gemmId_, has_fpA_intB_gemv_);

#ifdef FPA_INTB_GEMM_LATENCY
        });
        std::cout << "Latency: " << latency_us << " microseconds" << std::endl;
#endif

        max_m_ = next_m;
      }
      auto const& bestTactic = gemmProfiler_->getBestConfig(m, gemmId_);

      DUMP_STRING("Best tactic: m=", m, " n=", n, " k=", k, " group_size=", block_size_, bestTactic->toString());

      if (bestTactic->enableCudaKernel) {
        using onnxruntime::llm::kernels::fpA_intB_gemv::KernelType;
        KernelType cuda_kernel_type = (nbits_ == 8) ? KernelType::FP16Int8Groupwise : KernelType::FP16Int4Groupwise;

        void const* pre_quant_scale_ptr = nullptr;
        bool apply_alpha_in_advance = false;
        float alpha = 1.0f;
        onnxruntime::llm::kernels::fpA_intB_gemv::Params params(
            a_data, pre_quant_scale_ptr, fpA_intB_weight_buffer_.get(),
            fpA_intB_scale_buffer_.get(), has_zero_points_ ? fpA_intB_zero_buffer_.get() : nullptr,
            bias_data, out_data,
            alpha, m, n, k, block_size_, cuda_kernel_type, apply_alpha_in_advance);

        onnxruntime::llm::kernels::fpA_intB_gemv::kernel_launcher(sm_, params, stream);
      } else {
        const size_t workspace_size = weightOnlyGemmRunner_->getWorkspaceSize(m, n, k);
        auto workspace_buffer = GetScratchBuffer<void>(workspace_size, ctx->GetComputeStream());

        weightOnlyGemmRunner_->gemm(
            a_data,
            fpA_intB_weight_buffer_.get(),
            fpA_intB_scale_buffer_.get(),
            has_zero_points_ ? fpA_intB_zero_buffer_.get() : nullptr,
            bias_data,
            1.f,
            out_data,
            m, n, k,
            block_size_,
            *bestTactic,
            reinterpret_cast<char*>(workspace_buffer.get()),
            workspace_size,
            stream);
      }

      return Status::OK();
    }
  }

  if ((reorder_idx_data == nullptr) && (!zero_points || !zero_points->IsDataType<T>())) {
    bool done = (nbits_ == 8) ? TryMatMul8Bits(
                                    reinterpret_cast<CudaT*>(Y->MutableData<T>()),
                                    reinterpret_cast<const CudaT*>(a_data),
                                    blob_data,
                                    reinterpret_cast<const CudaT*>(scales_data),
                                    static_cast<const uint8_t*>(zero_points_data),
                                    m,
                                    n,
                                    k,
                                    SafeInt<int>(block_size_),
                                    GetDeviceProp().sharedMemPerBlock,
                                    stream)
                              : TryMatMul4Bits(
                                    reinterpret_cast<CudaT*>(Y->MutableData<T>()),
                                    reinterpret_cast<const CudaT*>(a_data),
                                    blob_data,
                                    reinterpret_cast<const CudaT*>(scales_data),
                                    static_cast<const uint8_t*>(zero_points_data),
                                    m,
                                    n,
                                    k,
                                    SafeInt<int>(block_size_),
                                    GetDeviceProp().sharedMemPerBlock,
                                    stream);
    if (done) {
      return Status::OK();
    }
  }

  int64_t K_padded = (K_ + block_size_ - 1) / block_size_ * block_size_;
  IAllocatorUniquePtr<T> b_data_ptr = GetScratchBuffer<T>(N_ * K_padded, ctx->GetComputeStream());
  auto* b_data = b_data_ptr.get();

  if (nbits_ == 8) {
    if (column_wise_quant_blk_) {
      if (reorder_idx) {
        ORT_ENFORCE(K_padded == reorder_idx->Shape()[0], "K_padded != g_idx->Shape()[0]");
      }
      if (zero_points && zero_points->IsDataType<T>()) {
        ORT_RETURN_IF_ERROR(Dequantize8Bits(
            reinterpret_cast<CudaT*>(b_data),
            blob_data,
            reinterpret_cast<const CudaT*>(scales_data),
            (const CudaT*)zero_points_data,
            reorder_idx_data,
            SafeInt<int>(K_padded),
            SafeInt<int>(N_),
            SafeInt<int>(block_size_),
            stream));
      } else {
        ORT_RETURN_IF_ERROR(Dequantize8Bits(
            reinterpret_cast<CudaT*>(b_data),
            blob_data,
            reinterpret_cast<const CudaT*>(scales_data),
            (const uint8_t*)zero_points_data,
            reorder_idx_data,
            SafeInt<int>(K_padded),
            SafeInt<int>(N_),
            SafeInt<int>(block_size_),
            stream));
      }
    } else {  // row-wise block
      ORT_RETURN_IF_ERROR(DequantizeBlockwise8b(
          reinterpret_cast<CudaT*>(b_data),
          blob_data,
          reinterpret_cast<const CudaT*>(scales_data),
          (const uint8_t*)zero_points_data,
          SafeInt<int>(block_size_),
          column_wise_quant_blk_,
          SafeInt<int>(K_),
          SafeInt<int>(N_),
          stream));
    }
  } else {  // 4 bits
    if (column_wise_quant_blk_) {
      if (reorder_idx) {
        ORT_ENFORCE(K_padded == reorder_idx->Shape()[0], "K_padded != g_idx->Shape()[0]");
      }
      // column-wise block
      if ((zero_points && zero_points->IsDataType<T>())) {
        ORT_RETURN_IF_ERROR(Dequantize4Bits(
            reinterpret_cast<CudaT*>(b_data),
            blob_data,
            reinterpret_cast<const CudaT*>(scales_data),
            (const CudaT*)zero_points_data,
            reorder_idx_data,
            SafeInt<int>(K_padded),
            SafeInt<int>(N_),
            SafeInt<int>(block_size_),
            stream));
      } else {
        ORT_RETURN_IF_ERROR(Dequantize4Bits(
            reinterpret_cast<CudaT*>(b_data),
            blob_data,
            reinterpret_cast<const CudaT*>(scales_data),
            (const uint8_t*)zero_points_data,
            reorder_idx_data,
            SafeInt<int>(K_padded),
            SafeInt<int>(N_),
            SafeInt<int>(block_size_),
            stream));
      }
    } else {
      // row-wise block
      K_padded = K_;

      ORT_RETURN_IF_ERROR(DequantizeBlockwise4b(
          reinterpret_cast<CudaT*>(b_data),
          blob_data,
          reinterpret_cast<const CudaT*>(scales_data),
          (const uint8_t*)zero_points_data,
          SafeInt<int>(block_size_),
          column_wise_quant_blk_,
          SafeInt<int>(K_),
          SafeInt<int>(N_),
          stream));
    }
  }

  DUMP_TENSOR_D("DeQuantized", b_data, N_, K_padded);

  const CudaT alpha = ToCudaType<T>::FromFloat(1.f);
  const CudaT zero = ToCudaType<T>::FromFloat(0.f);

  if (helper.OutputOffsets().size() == 1) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        GetCublasHandle(ctx),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        SafeInt<int>(helper.N()),
        SafeInt<int>(helper.M()),
        SafeInt<int>(helper.K()),
        &alpha,
        reinterpret_cast<const CudaT*>(b_data),
        SafeInt<int>(K_padded),
        reinterpret_cast<const CudaT*>(a_data),
        helper.Lda(transa),
        &zero,
        reinterpret_cast<CudaT*>(Y->MutableData<T>()),
        helper.Ldc(),
        GetDeviceProp(),
        UseTF32()));
  }

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<float>()}),
    MatMulNBits<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<MLFloat16>()}),
    MatMulNBits<MLFloat16>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
