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
#include "contrib_ops/cuda/llm/cutlass_preprocessors.h"
#include "contrib_ops/cpu/quantization/matmul_nbits_helper.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;
using onnxruntime::llm::kernels::weight_only::GemmPluginProfilerManager;
using onnxruntime::llm::kernels::weight_only::WeightOnlyGroupwiseQuantGemmPluginProfiler;
using onnxruntime::llm::kernels::weight_only::WeightTypeId;
static GemmPluginProfilerManager<WeightOnlyGroupwiseQuantGemmPluginProfiler> s_profilerManager;

template <typename T>
void MatMulNBits<T>::RunGemmProfile(bool hasWeightOnlyCudaKernel, int sm, int max_m) {
  gemmProfiler_ = s_profilerManager.createGemmPluginProfiler(/*inference*/ false);

  // Number of 16-bit elements after casting int8/int4 to fp16.
  int n_16b = N_ / (nbits_ == 8 ? 2 : 4);

  gemmId_ = GemmIdCore(n_16b, K_, onnxruntime::llm::nvinfer::DataType::kHALF);

  if constexpr (std::is_same_v<T, MLFloat16>) {
    if (nbits_ == 8) {
      weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<half, uint8_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
    } else if (nbits_ == 4) {
      weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
    }
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    if (nbits_ == 8) {
      weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<__nv_bfloat16, uint8_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
    } else if (nbits_ == 4) {
      weightOnlyGemmRunner_ = std::make_shared<CutlassFpAIntBGemmRunner<__nv_bfloat16, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
    }
  }

  using onnxruntime::llm::kernels::fpA_intB_gemv::KernelType;
  KernelType cuda_kernel_type = nbits_ == 8 ? KernelType::FP16Int8Groupwise : KernelType::FP16Int4Groupwise;
  gemmProfiler_->setCudaKernelType(cuda_kernel_type, sm);
  gemmProfiler_->setQuant(nbits_, has_bias_, has_zero_points_);
  gemmProfiler_->setGroupSize(block_size_);

  int minM = 1;
  int maxM = max_m;
  GemmDims dims = {minM, maxM, n_16b, K_};

  gemmProfiler_->profileTactics(weightOnlyGemmRunner_, gemmId_.dtype, dims, gemmId_, hasWeightOnlyCudaKernel);
}

template <typename T>
Status MatMulNBits<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b = ctx->Input<Tensor>(1);
  const Tensor* scales = ctx->Input<Tensor>(2);
  const Tensor* zero_points = ctx->Input<Tensor>(3);
  const Tensor* reorder_idx = ctx->Input<Tensor>(4);
  const Tensor* bias = ctx->Input<Tensor>(5);

  if (bias != nullptr) {
    ORT_THROW("MatMulNBits does not support bias in CUDA kernel");
  }

  ORT_RETURN_IF_ERROR(matmul_nbits_helper::CheckInputs<Tensor>(
      a, b, scales, zero_points, reorder_idx, bias, N_, K_, block_size_, nbits_));

  const auto* a_data = a->Data<T>();
  const uint8_t* blob_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<T>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->DataRaw();
  const auto* reorder_idx_data = reorder_idx == nullptr ? nullptr : reorder_idx->Data<int32_t>();
  const auto* bias_data = bias == nullptr ? nullptr : bias->Data<T>();

  constexpr bool transa = false;
  constexpr bool transb = true;
  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(
      helper.Compute(a->Shape(), b_shape, transa, transb));

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
      std::call_once(fpA_intB_init_once_flag_, [&]() {
        size_t k_blocks = (k + block_size_ - 1) / block_size_;
        size_t scale_bytes = n * k_blocks * sizeof(T);

        fpA_intB_scale_buffer_ = this->GetTransientScratchBuffer<void>(scale_bytes);
        CudaT* transposed_scales = reinterpret_cast<CudaT*>(fpA_intB_scale_buffer_.get());

        fpA_intB_zero_buffer_ = this->GetTransientScratchBuffer<void>(has_zero_points_ ? scale_bytes : 0);
        CudaT* scaled_zero_points = has_zero_points_ ? reinterpret_cast<CudaT*>(fpA_intB_zero_buffer_.get()) : nullptr;

        size_t weight_bytes = n * k;
        IAllocatorUniquePtr<void> transpose_weight_space = this->GetTransientScratchBuffer<void>(weight_bytes);
        int8_t* transposed_weight = reinterpret_cast<int8_t*>(transpose_weight_space.get());

        size_t packed_weight_bytes = n * k / (8 / nbits_);

        // uint8 does not need to be packed so we do not need to allocate extra space.
        IAllocatorUniquePtr<void> packed_transposed_weight_space = this->GetTransientScratchBuffer<void>(nbits_ == 4 ? packed_weight_bytes : 0);
        int8_t* packed_transposed_weight = reinterpret_cast<int8_t*>(packed_transposed_weight_space.get());

        fpA_intB_weight_buffer_ = this->GetTransientScratchBuffer<void>(packed_weight_bytes);
        int8_t* preprocessed_weight = reinterpret_cast<int8_t*>(fpA_intB_weight_buffer_.get());

        auto tranpose_weight_buffer = this->AllocateBufferOnCPUPinned<int8_t>(packed_weight_bytes);

        using onnxruntime::llm::kernels::cutlass_kernels::QuantType;
        QuantType quant_type = nbits_ == 4 ? QuantType::W4_A16 : QuantType::W8_A16;
        if (nbits_ == 4) {
          // transpose the weight and add default zp.
          onnxruntime::llm::kernels::fpA_intB_gemv::unpack_uint4_transposed_to_int8_cuda(
              stream,
              packed_transposed_weight,
              transposed_weight,
              blob_data,
              n,
              k);
          CUDA_CALL_THROW(cudaStreamSynchronize(stream));
          DUMP_TENSOR_D("transposed_weight in GPU", transposed_weight, k, n);
          DUMP_TENSOR_D("packed transposed_weight in GPU", packed_transposed_weight, k, n / 2);

          CUDA_CALL_THROW(cudaMemcpy(tranpose_weight_buffer.get(), packed_transposed_weight, packed_weight_bytes, cudaMemcpyDeviceToHost));
        } else {
          auto weight_buffer = this->AllocateBufferOnCPUPinned<uint8_t>(n * k);
          CUDA_CALL_THROW(cudaMemcpy(weight_buffer.get(), blob_data, n * k, cudaMemcpyDeviceToHost));

          // Transpose the weight from (n, k) to (k, n), and convert to int8_t by subtracting 128.
          for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j < k; j++) {
              tranpose_weight_buffer.get()[j * n + i] = int8_t(int(weight_buffer.get()[i * k + j]) - 128);
            }
          }
        }

        auto processed_weight_buffer = this->AllocateBufferOnCPUPinned<uint8_t>(n * k / (8 / nbits_));
        bool force_interleave = false;
        onnxruntime::llm::kernels::cutlass_kernels::preprocess_weights_for_mixed_gemm(
            reinterpret_cast<int8_t*>(processed_weight_buffer.get()),
            reinterpret_cast<const int8_t*>(tranpose_weight_buffer.get()),
            {static_cast<size_t>(k), static_cast<size_t>(n)},
            quant_type,
            force_interleave);

        CUDA_CALL_THROW(cudaMemcpy(preprocessed_weight, processed_weight_buffer.get(), n * k / (8 / nbits_), cudaMemcpyHostToDevice));
        CUDA_CALL_THROW(cudaDeviceSynchronize());

        DUMP_TENSOR_D("preprocessed_weight", reinterpret_cast<uint8_t*>(preprocessed_weight), k, n / 2);

        constexpr float kDefaultZeroPoint4Bit = 8.0f;
        constexpr float kDefaultZeroPoint8Bit = 128.0f;
        const float default_zero_point = nbits_ == 4 ? kDefaultZeroPoint4Bit : kDefaultZeroPoint8Bit;
        // The scaled zero point will be zero for the default zero point, so there is no need to scale when it is nullptr.
        if (zero_points != nullptr) {
          if (!zero_points->IsDataType<T>()) {  // zero point is uint8_t type
            if (nbits_ == 4) {
              onnxruntime::llm::kernels::fpA_intB_gemv::launch_scaled_zero_point_kernel<true, CudaT, uint8_t>(
                  stream, reinterpret_cast<const CudaT*>(scales_data), reinterpret_cast<const uint8_t*>(zero_points_data),
                  transposed_scales, scaled_zero_points, n, k_blocks, default_zero_point);
            } else {
              onnxruntime::llm::kernels::fpA_intB_gemv::launch_scaled_zero_point_kernel<false, CudaT, uint8_t>(
                  stream, reinterpret_cast<const CudaT*>(scales_data), reinterpret_cast<const uint8_t*>(zero_points_data),
                  transposed_scales, scaled_zero_points, n, k_blocks, default_zero_point);
            }
          } else {  // zero point is not uint8_t type
            onnxruntime::llm::kernels::fpA_intB_gemv::launch_scaled_zero_point_kernel<false, CudaT, CudaT>(
                stream, reinterpret_cast<const CudaT*>(scales_data), reinterpret_cast<const CudaT*>(zero_points_data),
                transposed_scales, scaled_zero_points, n, k_blocks, default_zero_point);
          }
        }

        DUMP_STRING("k_blocks=", k_blocks, " n=", n);
        DUMP_TENSOR_D("transposed_scales", transposed_scales, k_blocks, n);
        if (scaled_zero_points != nullptr) {
          DUMP_TENSOR_D("scaled_zero_points", scaled_zero_points, k_blocks, n);
        }
      });

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

      DUMP_TENSOR_D("out_data", out_data, m, n);
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
