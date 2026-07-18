// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_nbits.h"

#include <cstdint>

#include "core/common/status.h"
#include "core/common/float16.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cpu/utils/dump_tensor.h"
#include "contrib_ops/cuda/quantization/matmul_nbits.cuh"
#include "contrib_ops/cuda/quantization/dequantize_blockwise.cuh"
#if USE_FPA_INTB_GEMM
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_adaptor.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#endif
#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cpu/quantization/matmul_nbits_helper.h"

constexpr int MatMulNBits_Input_B = 1;
constexpr int MatMulNBits_Input_Scale = 2;
constexpr int MatMulNBits_Input_ZeroPoint = 3;

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;

#if USE_FPA_INTB_GEMM
using onnxruntime::llm::kernels::weight_only::GemmPluginProfilerManager;
using onnxruntime::llm::kernels::weight_only::WeightOnlyGroupwiseQuantGemmPluginProfiler;
using onnxruntime::llm::kernels::weight_only::WeightTypeId;
static GemmPluginProfilerManager<WeightOnlyGroupwiseQuantGemmPluginProfiler> s_profilerManager;

constexpr auto kScaleAndZeros = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS;
constexpr auto kScaleOnly = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;

template <typename T>
int MatMulNBits<T>::FpAIntBPackingSmForKernel() const {
  // Select the native SM90 (Hopper) mixed-weight layout only when the weights were prepacked for it
  // (weight_prepacked_ == 2) AND the device is SM90. Otherwise use the SM80 layout, which is also
  // used as the SM90 compatibility path for runtime-prepacked (or SM80-prepacked) weights.
  if (sm_ == 90 && weight_prepacked_ == kMatMulNBitsWeightPrepackedSm90) {
    return 90;
  }
  return 80;
}

template <typename T>
int64_t MatMulNBits<T>::RequiredWeightPrepackedFormat() const {
  return FpAIntBPackingSmForKernel() == 90 ? kMatMulNBitsWeightPrepackedSm90 : kMatMulNBitsWeightPrepackedSm80;
}

template <typename T>
void MatMulNBits<T>::InitGemmProfiler(int sm) {
  gemmProfiler_ = s_profilerManager.createGemmPluginProfiler(/*inference*/ false);

  using onnxruntime::llm::kernels::fpA_intB_gemv::KernelType;
  KernelType cuda_kernel_type;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    cuda_kernel_type = nbits_ == 8 ? KernelType::FP16Int8Groupwise : KernelType::FP16Int4Groupwise;
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
    cuda_kernel_type = nbits_ == 8 ? KernelType::BF16Int8Groupwise : KernelType::BF16Int4Groupwise;
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

  // On SM90 the half/bf16 weight-only path can run either the native Hopper (SM90 TMA/WGMMA) kernel
  // or the SM80 (Ampere) mixed-GEMM kernel (which also runs on Hopper via GemmFpAIntB::operator()).
  //   - Native SM90 (sm == 90): keep the runner targeting SM90 so getConfigs() enumerates Hopper
  //     tactics (tile_config_sm90) and getWorkspaceSize() reserves the stream-K workspace; opt in to
  //     the native kernel via setUseSm90Native(true).
  //   - SM80 compat (sm == 80 while the device is SM90): force the runner to SM80 so tactic
  //     enumeration and workspace sizing stay consistent with the dispatched SM80 kernel (the runner
  //     otherwise defaults to the detected device SM and would enumerate Hopper tactics the SM80
  //     dispatch cannot consume, leaving no CUTLASS GEMM tactic for M>=16).
  if (sm == 90) {
    weightOnlyGemmRunner_->setUseSm90Native(true);
  } else if (sm_ == 90) {
    weightOnlyGemmRunner_->setArch(sm);
  }

  gemmProfiler_->setCudaKernelType(cuda_kernel_type, sm);
  gemmProfiler_->setQuant(static_cast<int>(nbits_), has_bias_, has_zero_points_);
  gemmProfiler_->setGroupSize(static_cast<int>(block_size_));

  auto allocator = this->Info().GetAllocator(OrtMemType::OrtMemTypeDefault);
  gemmProfiler_->setAllocator(allocator);
}

template <typename T>
void MatMulNBits<T>::RunGemmProfile(bool hasWeightOnlyCudaKernel, int min_m, int max_m) {
  // Number of 16-bit elements after casting int8/int4 to fp16.
  int n_16b = static_cast<int>(N_ / (nbits_ == 8 ? 2 : 4));

  // Include the packing/kernel SM in the GEMM id so the SM80-compatibility and native SM90 kernels
  // (which need different tactics) do not share profiled configs for the same (N, K, dtype).
  const int kernel_sm = FpAIntBPackingSmForKernel();
  if constexpr (std::is_same_v<T, MLFloat16>) {
    gemmId_ = GemmIdCore(n_16b, static_cast<int>(K_), onnxruntime::llm::nvinfer::DataType::kHALF, kernel_sm);
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    gemmId_ = GemmIdCore(n_16b, static_cast<int>(K_), onnxruntime::llm::nvinfer::DataType::kBF16, kernel_sm);
  }

  GemmDims dims = {min_m, max_m, n_16b, K_};
  gemmProfiler_->profileTactics(weightOnlyGemmRunner_, gemmId_.dtype, dims, gemmId_, hasWeightOnlyCudaKernel);
}

template <typename T>
Status MatMulNBits<T>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                               /*out*/ bool& is_packed,
                               /*out*/ PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;
  if constexpr (std::is_same_v<T, MLFloat16> || std::is_same_v<T, BFloat16>) {
    if (has_fpA_intB_gemm_) {
      cudaStream_t stream = cudaStreamLegacy;  // Use default stream for prepacking.
      if (input_idx == MatMulNBits_Input_B) {
        ORT_RETURN_IF_ERROR(PrePack_B(tensor, alloc, stream, is_packed));
        is_prepacked_weight_ = is_packed;
      } else if (input_idx == MatMulNBits_Input_Scale) {
        ORT_RETURN_IF_ERROR(PrePack_Scale(tensor, alloc, stream));
        is_prepacked_scale_ = true;
        is_packed = true;
      } else if (input_idx == MatMulNBits_Input_ZeroPoint) {
        if (has_zero_points_) {
          ORT_RETURN_IF_ERROR(PrePack_ZeroPoint(tensor, alloc, stream));
          is_prepacked_zero_point_ = true;
          is_packed = true;
        }
      }
    }
  }

  return Status::OK();
}

template <typename T>
Status MatMulNBits<T>::PrePack_B([[maybe_unused]] const Tensor& tensor,
                                 [[maybe_unused]] AllocatorPtr alloc,
                                 [[maybe_unused]] cudaStream_t stream,
                                 [[maybe_unused]] bool& is_packed) {
  if constexpr (std::is_same_v<T, MLFloat16> || std::is_same_v<T, BFloat16>) {
    size_t n = static_cast<size_t>(N_);
    size_t k = static_cast<size_t>(K_);

    size_t packed_weight_bytes = n * k / (8 / nbits_);

    const uint8_t* blob_data = tensor.Data<uint8_t>();
    if (weight_prepacked_ != kMatMulNBitsWeightNotPrepacked) {
      ORT_ENFORCE(tensor.SizeInBytes() == packed_weight_bytes,
                  "Prepacked MatMulNBits weight size mismatch. Expected ", packed_weight_bytes,
                  " bytes, got ", tensor.SizeInBytes());

      // Keep device-resident prepacked weights as the original input to avoid a second GPU copy.
      if (tensor.Location().device.Type() == OrtDevice::GPU) {
        is_packed = false;
        return Status::OK();
      }

      fpA_intB_weight_buffer_ = IAllocator::MakeUniquePtr<void>(alloc, packed_weight_bytes, true);
      int8_t* preprocessed_weight = reinterpret_cast<int8_t*>(fpA_intB_weight_buffer_.get());
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(preprocessed_weight, blob_data, packed_weight_bytes, cudaMemcpyDefault, stream));
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
      DUMP_TENSOR_INIT();
      DUMP_TENSOR_D("preprocessed_weight", reinterpret_cast<uint8_t*>(preprocessed_weight), k, n * nbits_ / 8);
      is_packed = true;
      return Status::OK();
    }

    // uint8 does not need to be packed so we do not need to allocate extra space.
    IAllocatorUniquePtr<void> packed_transposed_weight_space = this->GetTransientScratchBuffer<void>(packed_weight_bytes);
    int8_t* packed_transposed_weight = reinterpret_cast<int8_t*>(packed_transposed_weight_space.get());

    fpA_intB_weight_buffer_ = IAllocator::MakeUniquePtr<void>(alloc, packed_weight_bytes, true);
    int8_t* preprocessed_weight = reinterpret_cast<int8_t*>(fpA_intB_weight_buffer_.get());

    if (nbits_ == 4) {
      // Transpose the weight and add default zero point.
      onnxruntime::llm::kernels::fpA_intB_gemv::unpack_uint4_transposed_to_int8_direct_cuda(
          stream, packed_transposed_weight, blob_data, static_cast<int>(n), static_cast<int>(k));
    } else {
      onnxruntime::llm::kernels::fpA_intB_gemv::transpose_uint8_matrix_and_convert_to_int8(
          stream, packed_transposed_weight, blob_data, static_cast<int>(n), static_cast<int>(k));
    }

    using onnxruntime::llm::kernels::weight_only::QuantType;
    QuantType quant_type = nbits_ == 4 ? QuantType::W4_A16 : QuantType::W8_A16;

    auto permutation_map_buffer = this->GetTransientScratchBuffer<int32_t>(32);
    onnxruntime::llm::kernels::weight_only::preprocess_weights_for_mixed_gemm_cuda(
        stream,
        FpAIntBPackingSmForKernel(),
        preprocessed_weight,
        packed_transposed_weight,
        permutation_map_buffer.get(),
        {static_cast<size_t>(k), static_cast<size_t>(n)},
        quant_type);

    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    DUMP_TENSOR_INIT();
    DUMP_TENSOR_D("packed transposed_weight in GPU", packed_transposed_weight, k, n * nbits_ / 8);
    DUMP_TENSOR_D("preprocessed_weight", reinterpret_cast<uint8_t*>(preprocessed_weight), k, n * nbits_ / 8);
    is_packed = true;
  }

  return Status::OK();
}

template <typename T>
Status MatMulNBits<T>::PrePack_Scale([[maybe_unused]] const Tensor& tensor,
                                     [[maybe_unused]] AllocatorPtr alloc,
                                     [[maybe_unused]] cudaStream_t stream) {
  if constexpr (std::is_same_v<T, MLFloat16> || std::is_same_v<T, BFloat16>) {
    size_t n = static_cast<size_t>(N_);
    size_t k = static_cast<size_t>(K_);

    size_t k_blocks = (k + block_size_ - 1) / block_size_;
    size_t scale_bytes = n * k_blocks * sizeof(T);

    fpA_intB_scale_buffer_ = IAllocator::MakeUniquePtr<void>(alloc, scale_bytes, true);  // Transient buffer.

    typedef typename onnxruntime::cuda::OrtToCudaType<T>::type CudaT;
    CudaT* transposed_scales = reinterpret_cast<CudaT*>(fpA_intB_scale_buffer_.get());

    onnxruntime::llm::kernels::fpA_intB_gemv::launch_transpose_scale_kernel<CudaT>(
        stream, reinterpret_cast<const CudaT*>(tensor.Data<T>()), transposed_scales, static_cast<int>(n), static_cast<int>(k_blocks));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

    DUMP_TENSOR_INIT();
    DUMP_TENSOR_D("transposed_scales", transposed_scales, static_cast<int>(k_blocks), static_cast<int>(n));
  }
  return Status::OK();
}

template <typename T>
Status MatMulNBits<T>::PrePack_ZeroPoint([[maybe_unused]] const Tensor& tensor,
                                         [[maybe_unused]] AllocatorPtr alloc,
                                         [[maybe_unused]] cudaStream_t stream) {
  if constexpr (std::is_same_v<T, MLFloat16> || std::is_same_v<T, BFloat16>) {
    size_t n = static_cast<size_t>(N_);
    size_t k = static_cast<size_t>(K_);

    size_t k_blocks = (k + block_size_ - 1) / block_size_;
    size_t scale_bytes = n * k_blocks * sizeof(T);

    typedef typename onnxruntime::cuda::OrtToCudaType<T>::type CudaT;
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
            transposed_scales, scaled_zero_points, static_cast<int>(n), static_cast<int>(k_blocks), default_zero_point);
      } else {
        onnxruntime::llm::kernels::fpA_intB_gemv::launch_scaled_zero_point_kernel<false, CudaT, uint8_t>(
            stream, reinterpret_cast<const uint8_t*>(zero_points_data),
            transposed_scales, scaled_zero_points, static_cast<int>(n), static_cast<int>(k_blocks), default_zero_point);
      }
    } else {  // zero point is not uint8_t type
      onnxruntime::llm::kernels::fpA_intB_gemv::launch_scaled_zero_point_kernel<false, CudaT, CudaT>(
          stream, reinterpret_cast<const CudaT*>(zero_points_data),
          transposed_scales, scaled_zero_points, static_cast<int>(n), static_cast<int>(k_blocks), default_zero_point);
    }
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

    DUMP_TENSOR_INIT();
    DUMP_TENSOR_D("scaled_zero_points", scaled_zero_points, static_cast<int>(k_blocks), static_cast<int>(n));
  }
  return Status::OK();
}
#endif

template <typename T>
Status MatMulNBits<T>::ComputeInternal(OpKernelContext* ctx) const {
  if constexpr (std::is_same_v<T, BFloat16>) {
    if (sm_ < 80) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "BFloat16 MatMulNBits is not supported on cuda device with compute capability < 8.0");
    }
  }

  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* reorder_idx = ctx->Input<Tensor>(4);
  const Tensor* bias = ctx->Input<Tensor>(5);

#if USE_FPA_INTB_GEMM
  const Tensor* b = is_prepacked_weight_ ? nullptr : ctx->Input<Tensor>(1);
  const Tensor* scales = is_prepacked_scale_ ? nullptr : ctx->Input<Tensor>(2);
  const Tensor* zero_points = is_prepacked_zero_point_ ? nullptr : ctx->Input<Tensor>(3);
  const uint8_t* blob_data = is_prepacked_weight_ ? nullptr : b->Data<uint8_t>();
  const auto* scales_data = is_prepacked_scale_ ? nullptr : scales->Data<T>();
  const auto* zero_points_data = (is_prepacked_zero_point_ || zero_points == nullptr) ? nullptr : zero_points->DataRaw();
#else
  const Tensor* b = ctx->Input<Tensor>(1);
  const Tensor* scales = ctx->Input<Tensor>(2);
  const Tensor* zero_points = ctx->Input<Tensor>(3);
  const uint8_t* blob_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<T>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->DataRaw();
#endif
  const auto* bias_data = bias == nullptr ? nullptr : bias->Data<T>();

  ORT_RETURN_IF_ERROR(matmul_nbits_helper::CheckInputs<Tensor>(
      a, b, scales, zero_points, reorder_idx, bias, N_, K_, block_size_, nbits_));

  const auto* a_data = a->Data<T>();
  const auto* reorder_idx_data = reorder_idx == nullptr ? nullptr : reorder_idx->Data<int32_t>();

  constexpr bool transa = false;
  constexpr bool transb = true;
  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, transa, transb));

  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

  cudaStream_t stream = this->Stream(ctx);

  typedef typename onnxruntime::cuda::OrtToCudaType<T>::type CudaT;

  int m = SafeInt<int>(helper.M());
  int n = SafeInt<int>(helper.N());
  int k = SafeInt<int>(helper.K());

  DUMP_TENSOR_INIT();

#if USE_FPA_INTB_GEMM
  CudaT* out_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());

  if constexpr (std::is_same<T, MLFloat16>::value || std::is_same<T, BFloat16>::value) {
    if (has_fpA_intB_gemm_) {
      // We expect weight/scale/zero_point(optional) inputs are initializers and have been prepacked.
      // A non-prepacked node can opt out by setting session config ep.cuda.fpa_intb_gemm=0 (or env
      // ORT_FPA_INTB_GEMM=0) if those tensors cannot be prepacked (it is rare).
      const bool has_fpA_intB_weight = is_prepacked_weight_ || weight_prepacked_ != kMatMulNBitsWeightNotPrepacked;
      ORT_ENFORCE(has_fpA_intB_weight && is_prepacked_scale_ && (is_prepacked_zero_point_ || !has_zero_points_),
                  "To use fpA_intB_gemm, prepacking must be done on weight, scale and zero point.");

      const void* fpA_intB_weight = is_prepacked_weight_ ? fpA_intB_weight_buffer_.get() : static_cast<const void*>(blob_data);

      // During CUDA graph capture we must not lazily profile: profiling launches kernels, records
      // and synchronizes events, and allocates/frees scratch, all of which are illegal while the
      // compute stream is being captured. Fall back to a lookup of an already-profiled bucket
      // (warmup runs before capture populate these); only outside capture do we allow lazy
      // single-bucket profiling. Note: a null cudaStream_t is the default stream (a valid capture
      // target under per-thread default streams), so query the capture status unconditionally.
      const bool stream_is_capturing = onnxruntime::llm::common::isCapturing(stream);
      auto const bestTactic = stream_is_capturing ? gemmProfiler_->getBestConfig(m, gemmId_)
                                                  : gemmProfiler_->getBestConfigOrProfile(m, gemmId_);
      if (!bestTactic.has_value()) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, FAIL,
            "No valid fpA_intB MatMulNBits tactic for M=", m, ", N=", n, ", K=", k,
            stream_is_capturing
                ? ". The M bucket was not profiled before CUDA graph capture; run a warmup inference outside capture first."
                : "");
      }

      // Env-gated diagnostics (ORT_FPA_INTB_DEBUG=1): dump the selected tactic, the kernel path
      // (GEMV CUDA kernel vs CUTLASS GEMM), the weight format, and the device/packing SM so that
      // SM90 correctness issues (e.g. running the SM80 kernel on Hopper) can be traced.
      static const bool fpA_intB_debug =
          ParseEnvironmentVariableWithDefault<int>("ORT_FPA_INTB_DEBUG", 0) != 0;
      if (fpA_intB_debug) {
        const char* weight_fmt = is_prepacked_weight_ ? "runtime-prepacked(SM80 layout)"
                                                      : (weight_prepacked_ == kMatMulNBitsWeightPrepackedSm80 ? "offline-prepacked-SM80"
                                                                                                              : (weight_prepacked_ == kMatMulNBitsWeightPrepackedSm90 ? "offline-prepacked-SM90"
                                                                                                                                                                      : "raw"));
        std::cout << "[fpA_intB_debug] M=" << m << " N=" << n << " K=" << k
                  << " nbits=" << nbits_ << " block_size=" << block_size_
                  << " device_sm=" << sm_
                  << " packing_sm=" << FpAIntBPackingSmForKernel()
                  << " has_bias=" << (bias_data != nullptr ? 1 : 0)
                  << " has_zero_points=" << (has_zero_points_ ? 1 : 0)
                  << " weight_format=" << weight_fmt
                  << " kernel=" << (bestTactic->enableCudaKernel ? "GEMV(cuda)" : (FpAIntBPackingSmForKernel() == 90 ? "CUTLASS(sm90 gemm)" : "CUTLASS(sm80 gemm)"))
                  << " tactic=" << bestTactic->toString()
                  << std::endl;
      }

#if ORT_LLM_VERBOSE > 1
      std::cout << "Best tactic for m=" << m << ", n=" << n << ", k=" << k << "group_size=" << block_size_
                << " is: " << bestTactic->toString() << std::endl;
#endif

      if (bestTactic->enableCudaKernel) {
        using onnxruntime::llm::kernels::fpA_intB_gemv::KernelType;
        KernelType cuda_kernel_type;
        if constexpr (std::is_same<T, MLFloat16>::value) {
          cuda_kernel_type = nbits_ == 8 ? KernelType::FP16Int8Groupwise : KernelType::FP16Int4Groupwise;
        } else if constexpr (std::is_same<T, BFloat16>::value) {
          cuda_kernel_type = nbits_ == 8 ? KernelType::BF16Int8Groupwise : KernelType::BF16Int4Groupwise;
        }

        void const* pre_quant_scale_ptr = nullptr;
        bool apply_alpha_in_advance = false;
        float alpha = 1.0f;
        onnxruntime::llm::kernels::fpA_intB_gemv::Params params(
            a_data, pre_quant_scale_ptr, fpA_intB_weight,
            fpA_intB_scale_buffer_.get(), has_zero_points_ ? fpA_intB_zero_buffer_.get() : nullptr,
            bias_data, out_data,
            alpha, m, n, k, static_cast<int>(block_size_), cuda_kernel_type, apply_alpha_in_advance);

        // Launch the GEMV with the arch the weights were PACKED for (FpAIntBPackingSmForKernel),
        // not the raw device SM. The GEMV interleave layout is arch-dependent: arch in [90,100)
        // uses ColumnMajorInterleavedForHopper while the SM80 packing uses ColumnMajorInterleaved.
        // PrePack_B packs the SM80 layout, and the tactic profiler also profiles with the packing
        // arch, so passing the device SM (e.g. 90) here would read the SM80-packed weights with the
        // Hopper interleave and produce wrong results.
        onnxruntime::llm::kernels::fpA_intB_gemv::kernel_launcher(FpAIntBPackingSmForKernel(), params, stream);
      } else {
        const size_t workspace_size = weightOnlyGemmRunner_->getWorkspaceSize(m, n, k);
        auto workspace_buffer = this->template GetScratchBuffer<void>(workspace_size, this->GetComputeStream(ctx));

        weightOnlyGemmRunner_->gemm(
            a_data,
            fpA_intB_weight,
            fpA_intB_scale_buffer_.get(),
            has_zero_points_ ? fpA_intB_zero_buffer_.get() : nullptr,
            bias_data,
            1.f,
            out_data,
            m, n, k,
            static_cast<int>(block_size_),
            *bestTactic,
            reinterpret_cast<char*>(workspace_buffer.get()),
            workspace_size,
            stream);
      }

      return Status::OK();
    }
  }
#endif

  if ((reorder_idx_data == nullptr) && (!zero_points || !zero_points->IsDataType<T>())) {
    // First, try the fused fast path. It handles bias only for the GPT-OSS router GEMV
    // specialization; for other shapes it fails when bias is present.
    if (TryMatMulNBits(
            static_cast<int>(nbits_),
            reinterpret_cast<CudaT*>(Y->MutableData<T>()),
            reinterpret_cast<const CudaT*>(a_data),
            blob_data,
            reinterpret_cast<const CudaT*>(scales_data),
            static_cast<const uint8_t*>(zero_points_data),
            reinterpret_cast<const CudaT*>(bias_data),
            m,
            n,
            k,
            SafeInt<int>(block_size_),
            GetDeviceProp().sharedMemPerBlock,
            stream)) {
      return Status::OK();
    }

    // If bias prevented the fused path from running, retry the fast path without bias and
    // add the bias with a separate kernel. This keeps the lightweight GEMV path for cases
    // (e.g. 8-bit, or 4-bit non-router shapes) where only the bias was unsupported.
    if (bias_data != nullptr &&
        TryMatMulNBits(
            static_cast<int>(nbits_),
            reinterpret_cast<CudaT*>(Y->MutableData<T>()),
            reinterpret_cast<const CudaT*>(a_data),
            blob_data,
            reinterpret_cast<const CudaT*>(scales_data),
            static_cast<const uint8_t*>(zero_points_data),
            /*bias_data*/ static_cast<const CudaT*>(nullptr),
            m,
            n,
            k,
            SafeInt<int>(block_size_),
            GetDeviceProp().sharedMemPerBlock,
            stream)) {
      LaunchMatMulNBitsBiasAdd<CudaT>(
          reinterpret_cast<CudaT*>(Y->MutableData<T>()),
          reinterpret_cast<const CudaT*>(bias_data),
          m,
          n,
          stream);
      return Status::OK();
    }
  }

  // When bias is present but the fused router GEMV specialization above did not apply,
  // fall back to the generic dequantize + GEMM path (which ignores bias) and add the
  // bias with a separate kernel after the GEMM completes.

  int64_t K_padded = (K_ + block_size_ - 1) / block_size_ * block_size_;

  // Chunked dequant+GEMM trades peak scratch memory for repeated kernel launches.
  // Thresholds:
  //   256 MB  – scratch budget; above this the full N*K_padded buffer would dominate
  //             device memory and risk OOM on consumer GPUs (8–12 GB).
  //   N > 2*chunk_target_rows (default 65536) – ensures at least two chunks so the
  //             overhead of per-chunk cuBLAS calls is amortised.
  //   chunk_target_rows (default 32768) – chosen so that each dequant+GEMM tile is
  //             large enough to saturate SMs while keeping scratch ≤ ~128 MB.
  // Only column-wise quantization without reorder_idx is supported; row-wise layouts
  // interleave K blocks across N and cannot be sliced along the N axis.
  const int64_t chunk_target_rows = chunk_target_rows_;
  const int64_t scratch_bytes = N_ * K_padded * static_cast<int64_t>(sizeof(T));
  const bool will_use_chunked = column_wise_quant_blk_ &&
                                (reorder_idx_data == nullptr) &&
                                (force_chunked_ ||
                                 ((scratch_bytes > 256 * 1024 * 1024) &&
                                  (N_ > chunk_target_rows * 2)));

  // Allocate scratch: full size normally, chunk size if chunked path will be used
  const int64_t scratch_n = will_use_chunked ? chunk_target_rows : N_;
  IAllocatorUniquePtr<T> b_data_ptr = this->template GetScratchBuffer<T>(scratch_n * K_padded, this->GetComputeStream(ctx));
  auto* b_data = b_data_ptr.get();

  // Column-wise dequant helper: dispatches 8-bit / 4-bit × typed / uint8 zero-points.
  // Used by both the full-N and chunked paths so the offset math stays in one place.
  const int64_t blocks_per_col = K_padded / block_size_;
  auto dequant_column_wise = [&](const uint8_t* chunk_blob,
                                 const CudaT* chunk_scales,
                                 const void* chunk_zp,
                                 const int32_t* chunk_reorder_idx,
                                 int n_rows) -> Status {
    if (nbits_ == 8) {
      if (zero_points && zero_points->IsDataType<T>()) {
        return Dequantize8Bits(
            reinterpret_cast<CudaT*>(b_data), chunk_blob, chunk_scales,
            static_cast<const CudaT*>(chunk_zp), chunk_reorder_idx,
            SafeInt<int>(K_padded), n_rows, SafeInt<int>(block_size_), stream);
      } else {
        return Dequantize8Bits(
            reinterpret_cast<CudaT*>(b_data), chunk_blob, chunk_scales,
            static_cast<const uint8_t*>(chunk_zp), chunk_reorder_idx,
            SafeInt<int>(K_padded), n_rows, SafeInt<int>(block_size_), stream);
      }
    } else {
      if (zero_points && zero_points->IsDataType<T>()) {
        return Dequantize4Bits(
            reinterpret_cast<CudaT*>(b_data), chunk_blob, chunk_scales,
            static_cast<const CudaT*>(chunk_zp), chunk_reorder_idx,
            SafeInt<int>(K_padded), n_rows, SafeInt<int>(block_size_), stream);
      } else {
        return Dequantize4Bits(
            reinterpret_cast<CudaT*>(b_data), chunk_blob, chunk_scales,
            static_cast<const uint8_t*>(chunk_zp), chunk_reorder_idx,
            SafeInt<int>(K_padded), n_rows, SafeInt<int>(block_size_), stream);
      }
    }
  };

  // Skip full dequant when chunked path will handle it in the GEMM loop
  if (!will_use_chunked) {
    if (column_wise_quant_blk_) {
      if (reorder_idx) {
        ORT_ENFORCE(K_padded == reorder_idx->Shape()[0], "K_padded != g_idx->Shape()[0]");
      }
      ORT_RETURN_IF_ERROR(dequant_column_wise(
          blob_data,
          reinterpret_cast<const CudaT*>(scales_data),
          zero_points_data,
          reorder_idx_data,
          SafeInt<int>(N_)));
    } else if (nbits_ == 8) {
      // row-wise block (8-bit)
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
    } else {
      // row-wise block (4-bit)
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

  }  // end if (!will_use_chunked)

  if (!will_use_chunked) {
    DUMP_TENSOR_D("DeQuantized", b_data, static_cast<int>(N_), static_cast<int>(K_padded));
  }

  const CudaT alpha = onnxruntime::cuda::OrtToCudaType<T>::FromFloat(1.f);
  const CudaT zero = onnxruntime::cuda::OrtToCudaType<T>::FromFloat(0.f);

  if (helper.OutputOffsets().size() == 1) {
    if (will_use_chunked) {
      // Chunked dequant+GEMM: scratch buffer is already sized for one chunk.
      const int64_t chunk_n = chunk_target_rows;

      auto* chunk_out_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());

      for (int64_t n_start = 0; n_start < N_; n_start += chunk_n) {
        const int64_t n_end = std::min(n_start + chunk_n, N_);
        const int this_chunk = static_cast<int>(n_end - n_start);

        // Dequantize this chunk of N rows
        // column_wise_quant_blk_ is guaranteed by will_use_chunked; assert to prevent
        // silent use of uninitialized b_data if the outer guard is ever relaxed.
        ORT_ENFORCE(column_wise_quant_blk_, "Chunked path requires column-wise quantization blocks");

        // Compute per-chunk pointers into the column-wise-packed weight, scale, and ZP arrays.
        const uint8_t* chunk_blob = blob_data + n_start * (nbits_ == 8 ? K_padded : K_padded / 2);
        const auto* chunk_scales = reinterpret_cast<const CudaT*>(scales_data) + n_start * blocks_per_col;
        const void* chunk_zp = nullptr;
        if (zero_points_data) {
          if (zero_points && zero_points->IsDataType<T>()) {
            chunk_zp = reinterpret_cast<const CudaT*>(zero_points_data) + n_start * blocks_per_col;
          } else if (nbits_ == 8) {
            chunk_zp = static_cast<const uint8_t*>(zero_points_data) + n_start * blocks_per_col;
          } else {
            // 4-bit ZP: packed, offset by n_start * ceil(blocks_per_col / 2)
            chunk_zp = static_cast<const uint8_t*>(zero_points_data) + n_start * ((blocks_per_col + 1) / 2);
          }
        }
        ORT_RETURN_IF_ERROR(dequant_column_wise(
            chunk_blob, chunk_scales, chunk_zp,
            nullptr,  // no reorder_idx for chunked path
            this_chunk));

        DUMP_TENSOR_D("DeQuantized_chunk", b_data, this_chunk, static_cast<int>(K_padded));

        // GEMM for this chunk: C[:, n_start:n_end] = A[M,K] @ B_chunk[chunk_n, K]^T
        CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
            GetCublasHandle(ctx),
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            this_chunk,                // n (output columns for this chunk)
            SafeInt<int>(helper.M()),  // m
            SafeInt<int>(helper.K()),  // k
            &alpha,
            reinterpret_cast<const CudaT*>(b_data),  // B_chunk [chunk_n, K_padded]
            SafeInt<int>(K_padded),
            reinterpret_cast<const CudaT*>(a_data),  // A [M, K]
            helper.Lda(transa),
            &zero,
            chunk_out_data + n_start,    // C[:, n_start] — strided output
            SafeInt<int>(helper.Ldc()),  // ldc = N (full output stride)
            GetDeviceProp(),
            UseTF32()));
      }
    } else {
      // Original non-chunked path for small N
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

    // Fallback bias handling: the generic dequant + GEMM path above ignores bias, so add it here.
    if (bias_data != nullptr) {
      LaunchMatMulNBitsBiasAdd<CudaT>(
          reinterpret_cast<CudaT*>(Y->MutableData<T>()),
          reinterpret_cast<const CudaT*>(bias_data),
          m,
          n,
          stream);
    }
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

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    BFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<BFloat16>()}),
    MatMulNBits<BFloat16>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
