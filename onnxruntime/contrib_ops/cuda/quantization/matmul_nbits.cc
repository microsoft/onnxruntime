// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_nbits.h"

#include <cstdint>

#include "core/common/status.h"
#include "core/framework/float16.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "matmul_nbits.cuh"
#include "dequantize_blockwise.cuh"
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;

// template <typename T>
// std::vector<ort_llm::cutlass_extensions::CutlassGemmConfig> get_configs(T& runner, int k)
// {
//     auto configs = runner.getConfigs();
//     std::vector<ort_llm::cutlass_extensions::CutlassGemmConfig> rets;
//     for (auto config : configs)
//     {
//         if (config.stages >= 5)
//         {
//             continue;
//         }
//         if (config.split_k_style != ort_llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K)
//         {
//             int k_size = (k + config.split_k_factor - 1) / config.split_k_factor;
//             if (k_size % 64)
//             {
//                 continue;
//             }
//         }
//         rets.push_back(config);
//     }
//     return rets;
// }


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
  
  auto& device_prop = this->GetDeviceProp();
  int sm = device_prop.major * 10 + device_prop.minor;

  int m =  SafeInt<int>(helper.M());
  int n =  SafeInt<int>(helper.N());
  int k =  SafeInt<int>(helper.K());

  if (use_fpA_intB_gemm_ && sm >= 75) {
    KernelType cuda_kernel_type = nbits_ == 8 ? KernelType::FP16Int8Groupwise : KernelType::FP16Int4Groupwise;
    gemmProfiler_->setCudaKernelType(cuda_kernel_type, sm);

    // get best tactic and check if CUDA kernel should be used
    bool use_cuda_kernel = false;
    auto const& bestTactic = gemmProfiler_->getBestConfig(m, gemmId_);
    TLLM_CHECK_WITH_INFO(bestTactic,
                         "No valid weight only per-channel GEMM tactic(It is usually caused by the failure to execute all candidate "
                         "configurations of the CUTLASS kernel, please pay attention to the warning information when building the "
                         "engine.)");

    use_cuda_kernel = bestTactic->enableCudaKernel;
    if (use_cuda_kernel) {
      ort_llm::kernels::weight_only::Params params(a_data, nullptr, blob_data,
                                                   scales_data, zero_points_data, bias_data, out_data, 1.f, m, n, k, block_size_, cuda_kernel_type);
      ort_llm::kernels::weight_only::kernel_launcher(sm, params, stream);
    } else {
      const size_t workspace_size = weightOnlyGemmRunner_->getWorkspaceSize(m, n, k);
      auto workspace_buffer = GetScratchBuffer<void>(workspace_size, ctx->GetComputeStream());
      weightOnlyGemmRunner_->gemm(
          a_data,
          blob_data,
          scales_data,
          zero_points_data,
          bias_data,
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

  DUMP_TENSOR_INIT();
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
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulNBits<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulNBits<MLFloat16>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
