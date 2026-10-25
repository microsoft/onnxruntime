// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <flashinfer/norm.cuh>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "cuda_check.h"
// Upstream's standalone host launcher exits the process on CUDA errors.
#undef CHECK_CUDA
#undef CHECK_CUDA_KERNEL_LAUNCH
#define CHECK_CUDA(call)                                                           \
  do {                                                                             \
    const cudaError_t error = (call);                                              \
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); \
  } while (0)
#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())
#include "flash_fwd_launch_template.h"

namespace onnxruntime::source_compat {

template <typename Callback>
OrtStatus* CatchErrors(Callback&& callback) noexcept {
  try {
    callback();
    return nullptr;
  } catch (const Ort::Exception& error) {
    return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(), error.what());
  } catch (const std::exception& error) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
  } catch (...) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, "Unexpected source-compatible CUDA error");
  }
}

template <typename Derived>
struct KernelBase {
  OrtStatus* ComputeV2(OrtKernelContext* context) noexcept {
    return CatchErrors([&] { static_cast<Derived*>(this)->Compute(context); });
  }
};

template <typename Kernel, size_t Inputs, size_t Outputs>
struct Operation : Ort::CustomOpBase<Operation<Kernel, Inputs, Outputs>, Kernel, true> {
  explicit Operation(const char* name) : name_(name) {}
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** out) const noexcept {
    return CatchErrors([&] { *out = new Kernel(api, info); });
  }
  const char* GetName() const { return name_; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return Inputs; }
  size_t GetOutputTypeCount() const { return Outputs; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return index == 4 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 : ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  }
  OrtMemType GetInputMemoryType(size_t index) const {
    return index == 4 ? OrtMemTypeCPUInput : OrtMemTypeDefault;
  }
  const char* name_;
};

struct NormKernel : KernelBase<NormKernel> {
  NormKernel(const OrtApi&, const OrtKernelInfo* info)
      : epsilon_(Ort::ConstKernelInfo(info).GetAttribute<float>("epsilon")) {
    if (!std::isfinite(epsilon_) || epsilon_ <= 0) {
      throw std::invalid_argument("RMSNorm epsilon must be finite and positive");
    }
  }

  void Compute(OrtKernelContext* raw_context) {
    Ort::KernelContext context(raw_context);
    auto input = context.GetInput(0);
    auto weight = context.GetInput(1);
    const auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
    Validate(input, weight);
    auto output = context.GetOutput(0, shape);
    const size_t count = input.GetTensorTypeAndShapeInfo().GetElementCount();
    if (!count) return;
    const auto width = static_cast<uint32_t>(shape.back());
    CHECK_CUDA(flashinfer::norm::RMSNorm(
        const_cast<__nv_bfloat16*>(reinterpret_cast<const __nv_bfloat16*>(input.GetTensorRawData())),
        const_cast<__nv_bfloat16*>(reinterpret_cast<const __nv_bfloat16*>(weight.GetTensorRawData())),
        reinterpret_cast<__nv_bfloat16*>(output.GetTensorMutableRawData()),
        static_cast<uint32_t>(count / width), width, width, width, epsilon_, false,
        static_cast<cudaStream_t>(context.GetGPUComputeStream())));
  }

  static void Validate(Ort::ConstValue input, Ort::ConstValue weight) {
    const auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
    const auto count = input.GetTensorTypeAndShapeInfo().GetElementCount();
    if (shape.empty() || shape.back() <= 0 ||
        weight.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{shape.back()}) {
      throw std::invalid_argument("RMSNorm requires matching one-dimensional scale");
    }
    const auto width = static_cast<uint64_t>(shape.back());
    if (width % 8 || width > std::numeric_limits<uint32_t>::max() ||
        count / width > std::numeric_limits<uint32_t>::max()) {
      throw std::invalid_argument("RMSNorm requires width divisible by 8 and uint32 dimensions");
    }
  }
  float epsilon_;
};

struct ResidualNormKernel : KernelBase<ResidualNormKernel> {
  ResidualNormKernel(const OrtApi& api, const OrtKernelInfo* info) : norm_(api, info) {}

  void Compute(OrtKernelContext* raw_context) {
    Ort::KernelContext context(raw_context);
    auto input = context.GetInput(0);
    auto residual = context.GetInput(1);
    auto weight = context.GetInput(2);
    NormKernel::Validate(input, weight);
    const auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
    if (shape != residual.GetTensorTypeAndShapeInfo().GetShape()) {
      throw std::invalid_argument("SkipRMSNorm input and residual shapes must match");
    }
    const size_t count = input.GetTensorTypeAndShapeInfo().GetElementCount();
    if (count > std::numeric_limits<size_t>::max() / sizeof(__nv_bfloat16)) {
      throw std::invalid_argument("SkipRMSNorm byte size overflow");
    }
    auto normalized = context.GetOutput(0, shape);
    auto stored_residual = context.GetOutput(1, shape);
    if (!count) return;
    auto stream = static_cast<cudaStream_t>(context.GetGPUComputeStream());
    auto* output = reinterpret_cast<__nv_bfloat16*>(normalized.GetTensorMutableRawData());
    auto* residual_output = reinterpret_cast<__nv_bfloat16*>(stored_residual.GetTensorMutableRawData());
    CHECK_CUDA(cudaMemcpyAsync(output, input.GetTensorRawData(), count * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(residual_output, residual.GetTensorRawData(), count * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToDevice, stream));
    const auto width = static_cast<uint32_t>(shape.back());
    CHECK_CUDA(flashinfer::norm::FusedAddRMSNorm(
        output, residual_output,
        const_cast<__nv_bfloat16*>(reinterpret_cast<const __nv_bfloat16*>(weight.GetTensorRawData())),
        static_cast<uint32_t>(count / width), width, width, width, norm_.epsilon_, false, stream));
  }
  NormKernel norm_;
};

__global__ void InitializeLengths(int* lengths, int queries, int keys) {
  if (threadIdx.x == 0) {
    lengths[0] = 0;
    lengths[1] = queries;
    lengths[2] = 0;
    lengths[3] = keys;
  }
}

struct Scratch {
  Scratch(size_t bytes, cudaStream_t stream) : stream_(stream) {
    CHECK_CUDA(cudaMallocAsync(&pointer_, bytes, stream_));
  }
  ~Scratch() { cudaFreeAsync(pointer_, stream_); }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  void* pointer_{};
  cudaStream_t stream_;
};

template <bool Local, bool OneWarpGroup>
void Launch(Flash_fwd_params& params, cudaStream_t stream) {
  using BF16 = cutlass::bfloat16_t;
  run_flash_fwd<90, 128, 128, 1, BF16, BF16, !Local, Local,
                false, true, false, false, false, false,
                true, false, false, OneWarpGroup>(params, stream);
}

struct AttentionKernel : KernelBase<AttentionKernel> {
  AttentionKernel(const OrtApi&, const OrtKernelInfo* info) {
    Ort::ConstKernelInfo attributes(info);
    heads_ = attributes.GetAttribute<int64_t>("q_num_heads");
    kv_heads_ = attributes.GetAttribute<int64_t>("kv_num_heads");
    window_ = attributes.GetAttribute<int64_t>("window_left");
    scale_ = attributes.GetAttribute<float>("scale");
    if (heads_ <= 0 || kv_heads_ <= 0 || heads_ % kv_heads_ ||
        heads_ <= kv_heads_ || heads_ > 1024 || window_ < -1 ||
        window_ > std::numeric_limits<int>::max() || !std::isfinite(scale_) || scale_ <= 0) {
      throw std::invalid_argument("SinkAttention requires valid grouped heads, window and scale");
    }
  }

  void Compute(OrtKernelContext* raw_context) {
    Ort::KernelContext context(raw_context);
    auto query = context.GetInput(0);
    auto key = context.GetInput(1);
    auto value = context.GetInput(2);
    auto sinks = context.GetInput(3);
    auto mask = context.GetInput(4);
    const auto qs = query.GetTensorTypeAndShapeInfo().GetShape();
    const auto ks = key.GetTensorTypeAndShapeInfo().GetShape();
    if (qs.size() != 3 || ks.size() != 3 || qs[0] != 1 || ks[0] != 1 ||
        qs[1] <= 0 || ks[1] < qs[1] ||
        qs[1] > std::numeric_limits<int>::max() - 127 ||
        ks[1] > std::numeric_limits<int>::max() - 127 ||
        qs[2] != heads_ * 128 || ks[2] != kv_heads_ * 128 ||
        ks != value.GetTensorTypeAndShapeInfo().GetShape() ||
        sinks.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{heads_}) {
      throw std::invalid_argument("SinkAttention supports one unpadded BF16 row, head_dim=128");
    }
    if (mask.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1, ks[1]}) {
      throw std::invalid_argument("SinkAttention mask must have shape [1, total_sequence_length]");
    }
    // Fail closed rather than silently applying unpadded attention to masked tokens.
    const int64_t* mask_data = mask.GetTensorData<int64_t>();
    for (int64_t index = 0; index < ks[1]; ++index) {
      if (mask_data[index] != 1) {
        throw std::invalid_argument("SinkAttention currently requires an all-one attention mask");
      }
    }
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 9) throw std::invalid_argument("SinkAttention requires Hopper");
    auto stream = static_cast<cudaStream_t>(context.GetGPUComputeStream());
    const auto q = static_cast<int>(qs[1]);
    const auto k = static_cast<int>(ks[1]);
    const size_t lse_bytes = static_cast<size_t>(heads_) * q * sizeof(float);
    const size_t aligned_lse_bytes = (lse_bytes + 255) / 256 * 256;
    Scratch scratch(aligned_lse_bytes + 256, stream);
    auto* lengths = reinterpret_cast<int*>(
        static_cast<char*>(scratch.pointer_) + aligned_lse_bytes);
    auto* scheduler = lengths + 4;
    InitializeLengths<<<1, 32, 0, stream>>>(lengths, q, k);
    CHECK_CUDA_KERNEL_LAUNCH();
    auto output = context.GetOutput(0, qs);
    Flash_fwd_params params{};
    params.q_ptr = const_cast<void*>(query.GetTensorRawData());
    params.k_ptr = const_cast<void*>(key.GetTensorRawData());
    params.v_ptr = const_cast<void*>(value.GetTensorRawData());
    params.o_ptr = output.GetTensorMutableRawData();
    params.sink_ptr = const_cast<void*>(sinks.GetTensorRawData());
    params.q_row_stride = params.o_row_stride = heads_ * 128;
    params.k_row_stride = params.v_row_stride = kv_heads_ * 128;
    params.q_head_stride = params.k_head_stride = params.v_head_stride =
        params.o_head_stride = 128;
    params.v_dim_stride = 1;
    params.b = params.b_k = 1;
    params.h = static_cast<int>(heads_);
    params.h_k = static_cast<int>(kv_heads_);
    params.d = params.dv = params.d_rounded = params.dv_rounded = 128;
    params.seqlen_q = params.total_q = q;
    params.seqlen_k = params.total_k = k;
    params.seqlen_q_rounded = (q + 127) / 128 * 128;
    params.seqlen_k_rounded = (k + 127) / 128 * 128;
    params.cu_seqlens_q = lengths;
    params.cu_seqlens_k = lengths + 2;
    params.softmax_lse_ptr = scratch.pointer_;
    params.scale_softmax = scale_;
    params.p_dropout = params.rp_dropout = 1.f;
    params.p_dropout_in_uint8_t = 255;
    params.is_bf16 = true;
    params.is_causal = window_ == -1;
    params.is_local = !params.is_causal;
    params.window_size_left = window_ < 0 ? k - 1 : static_cast<int>(window_);
    params.window_size_right = 0;
    params.num_splits = 1;
    params.pack_gqa = true;
    params.arch = 90;
    params.num_sm = properties.multiProcessorCount;
    params.varlen_sort_batches = !params.is_local;
    params.head_swizzle = true;
    params.prepare_varlen_pdl = true;
    params.num_splits_dynamic_ptr = scheduler;
    params.num_m_blocks_ptr = scheduler + 4;
    params.varlen_batch_idx_ptr = params.is_local ? nullptr : scheduler + 8;
    params.num_nheads_in_l2_ptr = scheduler + (params.is_local ? 8 : 12);
    params.tile_count_semaphore = scheduler + (params.is_local ? 12 : 16);
    const bool one_warp_group = static_cast<int64_t>(q) * (heads_ / kv_heads_) <= 64;
    if (params.is_local) {
      if (one_warp_group)
        Launch<true, true>(params, stream);
      else
        Launch<true, false>(params, stream);
    } else {
      if (one_warp_group)
        Launch<false, true>(params, stream);
      else
        Launch<false, false>(params, stream);
    }
  }
  int64_t heads_, kv_heads_, window_;
  float scale_;
};
}  // namespace onnxruntime::source_compat

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* base) {
  const auto* api = base->GetApi(ORT_API_VERSION);
  if (!api) {
    return base->GetApi(1)->CreateStatus(ORT_FAIL, "Source CUDA extension requires a matching ORT API version");
  }
  Ort::InitApi(api);
  return onnxruntime::source_compat::CatchErrors([&] {
    using namespace onnxruntime::source_compat;
    static Operation<NormKernel, 2, 1> norm("RMSNorm");
    static Operation<ResidualNormKernel, 3, 2> residual_norm("SkipRMSNorm");
    static Operation<AttentionKernel, 5, 1> attention("SinkAttention");
    static Ort::CustomOpDomain domain("com.microsoft.source_compat");
    static const bool initialized = [&] {
      domain.Add(&norm);
      domain.Add(&residual_norm);
      domain.Add(&attention);
      return true;
    }();
    (void)initialized;
    Ort::UnownedSessionOptions(options).Add(domain);
  });
}
