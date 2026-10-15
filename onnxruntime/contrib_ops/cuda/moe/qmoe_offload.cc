#include "contrib_ops/cuda/moe/qmoe_offload.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <map>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <nvtx3/nvToolsExt.h>
#endif

#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "contrib_ops/cuda/moe/qmoe_offload_plan.h"
#include "contrib_ops/cuda/moe/qmoe_spin_pool.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_util_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.h"

namespace onnxruntime::contrib::cuda {

using namespace onnxruntime::cuda;
namespace ck = onnxruntime::llm::kernels::cutlass_kernels;
namespace gemv = onnxruntime::llm::kernels::moe_gemv;

namespace {
std::atomic<int64_t> g_last_end_ns{0};
std::atomic<uint64_t> g_gap_in_ns{0}, g_gap_in_n{0}, g_gap_out_ns{0}, g_gap_out_n{0};
}  // namespace

#define REGISTER_QMOE_OFFLOAD_KERNEL_TYPED(T)                                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                      \
      QMoEOffload, kMSDomain, 1, T, kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                                      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())                               \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<MLFloat16>(),                            \
                                 DataTypeImpl::GetTensorType<BFloat16>(),                             \
                                 DataTypeImpl::GetTensorType<Float8E8M0>(),                           \
                                 DataTypeImpl::GetTensorType<Float8E4M3FN>()})                        \
          .TypeConstraint("T3", {DataTypeImpl::GetTensorType<MLFloat16>(),                            \
                                 DataTypeImpl::GetTensorType<BFloat16>(),                             \
                                 DataTypeImpl::GetTensorType<int8_t>()})                              \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<float>())                                 \
          .TypeConstraint("TI", DataTypeImpl::GetTensorType<int32_t>())                               \
          .TypeConstraint("TP", DataTypeImpl::GetTensorType<int8_t>())                                \
          .InputMemoryType(OrtMemTypeCPUInput,                                                        \
                           {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21})      \
          .OutputMemoryType(OrtMemTypeCPUOutput, {1, 2, 3}),                                          \
      QMoEOffload);

REGISTER_QMOE_OFFLOAD_KERNEL_TYPED(MLFloat16)
REGISTER_QMOE_OFFLOAD_KERNEL_TYPED(BFloat16)

namespace {

template <typename Function>
void OffloadParallelFor(OpKernelContext* context, ptrdiff_t count, Function& function) {
#ifdef ORT_USE_EP_API_ADAPTERS
  context->ParallelFor(
      [](void* data, size_t task) {
        (*static_cast<Function*>(data))(static_cast<ptrdiff_t>(task));
      },
      static_cast<size_t>(count), &function);
#else
  concurrency::ThreadPool::TrySimpleParallelFor(context->GetOperatorThreadPool(), count, function);
#endif
}

constexpr int kGroupedMoeMinRows =
    static_cast<int>(onnxruntime::llm::kernels::moe_gemv::kMaxProfiledExpandedRows) + 1;

size_t AddSlab(size_t bytes, size_t& total) {
  const size_t offset = (SafeInt<size_t>(total) + 255) / 256 * 256;
  total = SafeInt<size_t>(offset) + bytes;
  return offset;
}

// Unified bounded CPU expert service (v3 Section 8.3 / F9).
qmoe::SpinPool& SharedSpinPool(int threads) {
  static qmoe::SpinPool pool(threads, ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_POOL_SPIN_US", 2000));
  return pool;
}

class StreamDrain final {
 public:
  StreamDrain(cudaStream_t stream, bool& failed, std::shared_ptr<bool> completed = {})
      : stream_(stream), failed_(failed), completed_(std::move(completed)) {}
  ~StreamDrain() {
    if (pending_ && cudaStreamSynchronize(stream_) != cudaSuccess) {
      failed_ = true;
    }
    if (completed_ && !failed_) *completed_ = true;
  }
  cudaError_t Synchronize() {
    const auto error = cudaStreamSynchronize(stream_);
    failed_ = failed_ || error != cudaSuccess;
    return error;
  }
  cudaError_t Wait() {
    const auto error = Synchronize();
    pending_ = error != cudaSuccess;
    return error;
  }
  void Disarm() {
    pending_ = false;
    completed_.reset();
  }

 private:
  cudaStream_t stream_;
  bool& failed_;
  bool pending_ = true;
  std::shared_ptr<bool> completed_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(StreamDrain);
};

// Row conversions between FP16/BF16 and FP32 delegate to qmoe_shared_layout (AVX-512 when supported by compiler and CPU, scalar otherwise).
void ActToFloatRow(const MLFloat16* source, float* destination, int count) {
  qmoe_shared::HalfBitsToFloatRow(reinterpret_cast<const uint16_t*>(source), destination, count);
}

void FloatToActRow(const float* source, MLFloat16* destination, int count) {
  qmoe_shared::FloatToHalfBitsRow(source, reinterpret_cast<uint16_t*>(destination), count);
}

void ActToFloatRow(const BFloat16* source, float* destination, int count) {
  qmoe_shared::BFloat16BitsToFloatRow(reinterpret_cast<const uint16_t*>(source), destination, count);
}

void FloatToActRow(const float* source, BFloat16* destination, int count) {
  qmoe_shared::FloatToBFloat16BitsRow(source, reinterpret_cast<uint16_t*>(destination), count);
}

}  // namespace

struct QMoEOffload::CpuWork {
  struct ExpertCall {
    int expert;
    size_t first;
    int count;
  };
  InlinedVector<float> reordered;
  InlinedVector<float> intermediate;
  InlinedVector<float> result;
  InlinedVector<float> natural;
  InlinedVector<float> token_rows;
  InlinedVector<char> token_ready;
  InlinedVector<InlinedVector<int>> expert_routes;
  InlinedVector<ExpertCall> calls;

  CpuWork(int group_calls, int num_experts, int hidden_size, int inter_size)
      : reordered(static_cast<size_t>(SafeInt<size_t>(group_calls) * qmoe_shared::kMaxRowsPerCall * hidden_size)),
        intermediate(static_cast<size_t>(SafeInt<size_t>(group_calls) * qmoe_shared::kMaxRowsPerCall * inter_size)),
        result(static_cast<size_t>(SafeInt<size_t>(group_calls) * qmoe_shared::kMaxRowsPerCall * hidden_size)),
        natural(hidden_size),
        expert_routes(num_experts) {}
};

struct QMoEOffload::SharedLayerCache {
  int device_id = -1;
  int64_t layer_index = -1;
  int num_experts = 0;
  int hidden_size = 0;
  int inter_size = 0;
  int resident_slots = 0;
  int physical_slots = 0;
  std::array<size_t, 6> bank_bytes{};
  std::array<size_t, 6> bank_offsets{};
  uint8_t* slab_ptr = nullptr;
  const int* slot_table_dev_ptr = nullptr;
  const InlinedVector<int>* expert_slot_ptr = nullptr;

  static std::mutex& Mutex() {
    static std::mutex m;
    return m;
  }
  static std::map<std::pair<int, int64_t>, std::weak_ptr<SharedLayerCache>>& Registry() {
    static std::map<std::pair<int, int64_t>, std::weak_ptr<SharedLayerCache>> reg;
    return reg;
  }
};

bool TryGetQMoEResidentLayerSlabView(
    int device_id,
    int64_t layer_index,
    int num_experts,
    int hidden_size,
    int inter_size,
    QMoEResidentLayerSlabView* out_view) {
  if (!out_view || layer_index < 0) return false;
  std::lock_guard<std::mutex> lock(QMoEOffload::SharedLayerCache::Mutex());
  auto& reg = QMoEOffload::SharedLayerCache::Registry();
  auto it = reg.find({device_id, layer_index});
  if (it == reg.end()) return false;
  auto locked = it->second.lock();
  if (!locked || locked->num_experts != num_experts || locked->hidden_size != hidden_size ||
      locked->inter_size != inter_size || !locked->slab_ptr || !locked->slot_table_dev_ptr ||
      locked->resident_slots <= 0) {
    return false;
  }
  out_view->fc1_weights = locked->slab_ptr + locked->bank_offsets[0];
  out_view->fc2_weights = locked->slab_ptr + locked->bank_offsets[1];
  out_view->fc1_scales = locked->slab_ptr + locked->bank_offsets[2];
  out_view->fc2_scales = locked->slab_ptr + locked->bank_offsets[3];
  out_view->slot_table_dev = locked->slot_table_dev_ptr;
  return true;
}

struct QMoEOffload::AsyncTile {
  const QMoEOffload* node = nullptr;
  std::shared_ptr<bool> completed;
  std::shared_ptr<onnxruntime::cuda::QMoEInvocationRecord> invocation;
  int tile_rows = 0;
  const int* host_indices = nullptr;
  const int8_t* host_phases = nullptr;
  bool has_phases = false;
  const void* host_input = nullptr;
  void* host_result = nullptr;
  int32_t* route_rows = nullptr;
  int32_t* route_calls = nullptr;
  int32_t* counts = nullptr;
  IAllocatorUniquePtr<uint8_t> pinned;
  qmoe::CallbackTiming* timing = nullptr;
  qmoe::CallbackTiming::Call trace_call;
};

QMoEOffload::QMoEOffload(const OpKernelInfo& info)
    : CudaKernel(info), MoEBase(info, GetDeviceProp()) {
  staging_optimization_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_OFFLOAD_STAGING_OPT", 0);
  ORT_ENFORCE(staging_optimization_ >= 0 && staging_optimization_ <= 3,
              "ORT_QMOE_OFFLOAD_STAGING_OPT must be 0..3 (tiled scales=1, combined upload=2).");
  const auto& options = info.GetConfigOptions();
  for (const auto* key : {"ep.cuda.enable_cuda_graph", "ep.CUDAExecutionProvider.enable_cuda_graph", "enable_cuda_graph"}) {
    const auto graph = options.GetConfigEntry(key);
    ORT_ENFORCE(!graph.has_value() || *graph == "0" || *graph == "false",
                "QMoEOffload requires CUDA graphs off; enabled graph option ", key, ".");
  }
#ifndef BUILD_CUDA_EP_AS_PLUGIN
  ORT_ENFORCE(!info.GetExecutionProvider()->IsGraphCaptureEnabled(),
              "QMoEOffload requires CUDA graphs off; eager streaming cannot be captured.");
#endif
  ORT_ENFORCE(sm_ >= 80, "QMoEOffload requires SM80 or later.");

#ifdef BUILD_CUDA_EP_AS_PLUGIN
  auto input_elem_type = info.GetKernelInfo().GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
  is_bf16_ = (input_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
#else
  int32_t input_elem_type = info.node().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  is_bf16_ = (input_elem_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16);
#endif

  session_controller_ = GetQMoESessionController();
  if (session_controller_) {
    int cur_dev = -1;
    if (cudaGetDevice(&cur_dev) == cudaSuccess && cur_dev != session_controller_->DeviceId()) {
      CUDA_CALL_THROW(cudaSetDevice(session_controller_->DeviceId()));
    }
    session_controller_->RegisterOffloadNode();
  }

  const std::string quant_type = info.GetAttrOrDefault<std::string>("quant_type", "int");
  ORT_ENFORCE(quant_type == "int" || quant_type == "nvfp4" || quant_type == "mxfp4" || quant_type == "fp4",
              "QMoEOffload supports quant_type='int', 'nvfp4', or 'mxfp4' ('fp4'), got '", quant_type, "'.");
  mxfp4_mode_ = (quant_type == "mxfp4" || quant_type == "fp4");
  nvfp4_mode_ = (quant_type == "nvfp4") || mxfp4_mode_;

  ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("expert_weight_bits", 4) == 4,
              "QMoEOffload requires expert_weight_bits=4.");
  for (const auto* attribute : {"fc1_expert_weight_bits", "fc2_expert_weight_bits", "fc3_expert_weight_bits"}) {
    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>(attribute, 4) == 4,
                "QMoEOffload does not support mixed expert weight widths.");
  }

  ORT_ENFORCE(activation_type_ == ck::ActivationType::Swiglu && swiglu_fusion_ == 1 && !use_sparse_mixer_,
              "QMoEOffload requires fused interleaved SwiGLU and full softmax top-k routing.");
  const auto normalize = info.GetAttrOrDefault<int64_t>("normalize_routing_weights", 0);
  ORT_ENFORCE((normalize == 0 || normalize == 1) &&
                  info.GetAttrOrDefault<int64_t>("use_sparse_mixer", 0) == 0 &&
                  info.GetAttrOrDefault<int64_t>("accuracy_level", 0) == 0,
              "QMoEOffload does not support sparse mixer or CPU accuracy modes.");
  ORT_ENFORCE(std::isfinite(activation_alpha_) && std::isfinite(activation_beta_) &&
                  !std::isnan(swiglu_limit_) && swiglu_limit_ >= 0,
              "QMoEOffload has invalid SwiGLU parameters.");

  if (nvfp4_mode_) {
    InitializeNvfp4Offload(info);
    return;
  }

  ORT_ENFORCE(!is_bf16_, "QMoEOffload quant_type='int' requires FP16 activations.");
  ORT_ENFORCE(block_size_ == 32, "QMoEOffload symmetric INT4 requires block_size=32.");
  const auto prepacked = info.GetAttrOrDefault<int64_t>("weights_prepacked", -1);
  ORT_ENFORCE((prepacked == -1 || prepacked == 1) &&
                  info.GetAttrOrDefault<std::string>("packing_layout", "cutlass_sm80_int4_v1") ==
                      "cutlass_sm80_int4_v1",
              "QMoEOffload requires immutable cutlass_sm80_int4_v1 prepacked weights; raw layouts are unsupported.");

  const bool legacy_shared = InputExists(info, 21);
  const bool compact_shared = !legacy_shared && info.GetAttrOrDefault<int64_t>("layout_version", -1) == 1 &&
                              InputExists(info, 2) && !InputExists(info, 5) && !InputExists(info, 6);
  shared_mode_ = legacy_shared || compact_shared;
  v3_compact_shared_ = compact_shared;
  if (shared_mode_) {
    InitializeShared(info);
    return;
  }

  ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("layout_version", -1) == -1,
              "QMoEOffload layout_version requires shared expert_blocks.");
  for (int index = 22; index <= 25; ++index) {
    ORT_ENFORCE(!InputExists(info, index), "QMoEOffload shared inputs require expert_blocks.");
  }
  ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("expert_cache_bytes", 0) == 0,
              "QMoEOffload eager streaming supports expert_cache_bytes=0 only; persistent cache is not implemented.");
  for (int index : {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}) {
    ORT_ENFORCE(!InputExists(info, index),
                "QMoEOffload does not support FC3, zero points, separate router weights or FP quantization inputs; slot ",
                index, " is present.");
  }

  constexpr std::array<int, 4> bank_slots{2, 5, 3, 6};
  for (size_t bank = 0; bank < bank_slots.size(); ++bank) {
    ORT_ENFORCE(Info().TryGetConstantInput(bank_slots[bank], &banks_[bank]) && banks_[bank] != nullptr,
                "QMoEOffload bank and scale inputs must be immutable CPU initializers; missing slot ", bank_slots[bank], ".");
    const auto& location = banks_[bank]->Location().device;
    ORT_ENFORCE(location.Type() == OrtDevice::CPU && location.MemType() == OrtDevice::MemType::DEFAULT,
                "QMoEOffload bank and scale inputs must use pageable CPU memory.");
    ORT_ENFORCE(banks_[bank]->Shape().NumDimensions() == 3 && banks_[bank]->DataRaw() != nullptr,
                "QMoEOffload banks and scales must be nonempty contiguous rank-three tensors.");
    ORT_ENFORCE(bank < 2 ? banks_[bank]->IsDataType<uint8_t>() : banks_[bank]->IsDataType<MLFloat16>(),
                "QMoEOffload requires uint8 packed banks and FP16 scales.");
  }
  const auto& fc1 = banks_[0]->Shape();
  const auto& fc2 = banks_[1]->Shape();
  ORT_ENFORCE(fc1[0] >= 2 && fc1[0] <= std::numeric_limits<int>::max() &&
                  fc1[1] > 0 && fc1[1] <= std::numeric_limits<int>::max() &&
                  fc2[1] > 0 && fc2[1] <= std::numeric_limits<int>::max() / 2,
              "QMoEOffload expert/feature dimensions are invalid or exceed the runner's integer limits.");
  num_experts_ = static_cast<int>(fc1[0]);
  hidden_size_ = static_cast<int>(fc1[1]);
  inter_size_ = static_cast<int>(fc2[1]);
  ORT_ENFORCE(hidden_size_ % 64 == 0 && inter_size_ % 64 == 0 && fc1[2] == inter_size_ &&
                  fc2[0] == num_experts_ && fc2[2] == hidden_size_ / 2,
              "QMoEOffload requires SM80 packed FC1 [E,H,I] and FC2 [E,I,H/2], with H and I multiples of 64.");
  const auto& scales1 = banks_[2]->Shape();
  const auto& scales2 = banks_[3]->Shape();
  ORT_ENFORCE(scales1[0] == num_experts_ && scales1[1] == int64_t{2} * inter_size_ &&
                  scales1[2] == hidden_size_ / 32 && scales2[0] == num_experts_ &&
                  scales2[1] == hidden_size_ && scales2[2] == inter_size_ / 32,
              "QMoEOffload requires FP16 scales [E,2I,H/32] and [E,H,I/32].");
  for (size_t bias = 0; bias < 2; ++bias) {
    const int slot = bias == 0 ? 4 : 7;
    if (!InputExists(info, slot)) continue;
    const size_t bank = 4 + bias;
    ORT_ENFORCE(Info().TryGetConstantInput(slot, &banks_[bank]) && banks_[bank] != nullptr &&
                    banks_[bank]->IsDataType<MLFloat16>() && banks_[bank]->Shape().NumDimensions() == 2 &&
                    banks_[bank]->Shape()[0] == num_experts_ &&
                    banks_[bank]->Shape()[1] == (bias == 0 ? int64_t{2} * inter_size_ : hidden_size_) &&
                    banks_[bank]->Location().device.Type() == OrtDevice::CPU &&
                    banks_[bank]->Location().device.MemType() == OrtDevice::MemType::DEFAULT,
                "QMoEOffload bias inputs must be immutable pageable CPU FP16 [E,2I] or [E,H] tensors.");
  }
  for (size_t bank = 2; bank < banks_.size(); ++bank) {
    if (!banks_[bank]) continue;
    ORT_ENFORCE(banks_[bank]->DataRaw() != nullptr &&
                    reinterpret_cast<uintptr_t>(banks_[bank]->DataRaw()) % alignof(MLFloat16) == 0,
                "QMoEOffload FP16 CPU scales and biases require non-null naturally aligned storage.");
  }
  ORT_ENFORCE(k_ > 0 && k_ <= num_experts_, "QMoEOffload k must be in [1,E].");
  ORT_ENFORCE(!onnxruntime::llm::kernels::moe_gemv::is_moe_gemv_fused_finalize_supported(
                  sm_, 1, k_, k_, hidden_size_, inter_size_, 4, 32),
              "QMoEOffload requires the resident grouped FC2 FP16 contribution contract; "
              "decode shapes using fused FP32 GEMV finalization are unsupported. Use k >= ",
              kGroupedMoeMinRows, ".");

  const auto tile = info.GetAttrOrDefault<int64_t>("token_tile_size", 32);
  const auto wave = info.GetAttrOrDefault<int64_t>("expert_wave_size", 1);
  const auto gpu_limit = info.GetAttrOrDefault<int64_t>("gpu_working_bytes", int64_t{64} << 20);
  const auto pinned_limit = info.GetAttrOrDefault<int64_t>("pinned_staging_bytes", int64_t{8} << 20);
  ORT_ENFORCE(tile > 0 && tile <= std::numeric_limits<int>::max() && wave > 0 && wave < num_experts_ &&
                  gpu_limit > 0 && pinned_limit > 0,
              "QMoEOffload needs positive tile/byte limits and expert_wave_size in [1,E-1]; full-bank staging is forbidden.");
  token_tile_size_ = static_cast<int>(tile);
  expert_wave_size_ = static_cast<int>(wave);
  max_routes_ = SafeInt<int>(tile) * k_;
  max_wave_routes_ = std::max(kGroupedMoeMinRows, static_cast<int>(SafeInt<int>(tile) * std::min(k_, wave)));
  ORT_ENFORCE(SafeInt<int64_t>(tile) * num_experts_ <= std::numeric_limits<int>::max() &&
                  SafeInt<int64_t>(max_wave_routes_) * expert_wave_size_ <= std::numeric_limits<int>::max(),
              "QMoEOffload tile routing exceeds the runner's integer indexing limits.");

  expert_bytes_[0] = SafeInt<size_t>(hidden_size_) * inter_size_;
  expert_bytes_[1] = SafeInt<size_t>(inter_size_) * hidden_size_ / 2;
  expert_bytes_[2] = SafeInt<size_t>(2) * inter_size_ * (hidden_size_ / 32) * sizeof(MLFloat16);
  expert_bytes_[3] = SafeInt<size_t>(hidden_size_) * (inter_size_ / 32) * sizeof(MLFloat16);
  expert_bytes_[4] = banks_[4] ? static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * sizeof(MLFloat16)) : size_t{0};
  expert_bytes_[5] = banks_[5] ? static_cast<size_t>(SafeInt<size_t>(hidden_size_) * sizeof(MLFloat16)) : size_t{0};
  size_t staging_bytes = 0;
  for (size_t bank = 0; bank < banks_.size(); ++bank) {
    ORT_ENFORCE(!banks_[bank] || SafeInt<size_t>(expert_bytes_[bank]) * num_experts_ == banks_[bank]->SizeInBytes(),
                "QMoEOffload bank byte extent is inconsistent with its packed layout.");
    const size_t bytes = SafeInt<size_t>(expert_bytes_[bank]) * expert_wave_size_;
    staging_offsets_[bank] = AddSlab(bytes, staging_bytes);
    offsets_[bank] = AddSlab(bytes, scratch_bytes_);
  }
  expert_slab_bytes_ = scratch_bytes_;
  ORT_ENFORCE(staging_bytes == expert_slab_bytes_, "QMoEOffload CPU and CUDA bank slab layouts differ.");
  const int max_metadata_routes = std::max(max_routes_, kGroupedMoeMinRows);
  const size_t route_bytes = SafeInt<size_t>(max_metadata_routes) * sizeof(int);
  host_indices_offset_ = AddSlab(route_bytes, staging_bytes);
  host_routes_offset_ = AddSlab(route_bytes, staging_bytes);
  host_compact_offset_ = AddSlab(route_bytes, staging_bytes);
  ORT_ENFORCE(staging_bytes <= SafeInt<size_t>(pinned_limit),
              "QMoEOffload pinned ring requires ", staging_bytes, " bytes, exceeding pinned_staging_bytes=", pinned_limit, ".");
  ORT_ENFORCE(scratch_bytes_ <= SafeInt<size_t>(gpu_limit),
              "QMoEOffload expert wave alone requires ", scratch_bytes_, " bytes, exceeding gpu_working_bytes=", gpu_limit, ".");

  runner_ = std::make_unique<ck::CutlassMoeFCRunner<half, cutlass::uint4b_t>>(
      sm_, activation_type_, false, false);
  ORT_ENFORCE(!runner_->getTactics().empty(), "QMoEOffload has no compatible SM80 INT4 GEMM tactic.");
  const ck::MOEParallelismConfig parallelism;
  for (int experts = 1; experts <= expert_wave_size_; ++experts) {
    const int route_bound = std::max(kGroupedMoeMinRows,
                                     static_cast<int>(SafeInt<int>(token_tile_size_) * std::min<int64_t>(k_, experts)));
    for (int rows = kGroupedMoeMinRows; rows <= route_bound; ++rows) {
      workspace_bytes_ = std::max(workspace_bytes_, runner_->getWorkspaceSize(
                                                        rows, hidden_size_, inter_size_, experts, 1, activation_type_, parallelism, true, swiglu_fusion_));
    }
  }
  offsets_[RouterScales] = AddSlab(SafeInt<size_t>(max_routes_) * sizeof(float), scratch_bytes_);
  offsets_[RouterIndices] = AddSlab(route_bytes, scratch_bytes_);
  offsets_[UnitScales] = AddSlab(SafeInt<size_t>(max_metadata_routes) * sizeof(float), scratch_bytes_);
  offsets_[IdentityPermutation] = AddSlab(route_bytes, scratch_bytes_);
  offsets_[RouteIds] = AddSlab(route_bytes, scratch_bytes_);
  offsets_[CompactExperts] = AddSlab(route_bytes, scratch_bytes_);
  offsets_[WavePermutation] = AddSlab(route_bytes, scratch_bytes_);
  const size_t wave_activation_bytes = SafeInt<size_t>(max_wave_routes_) * hidden_size_ * sizeof(half);
  offsets_[GatheredInput] = AddSlab(wave_activation_bytes, scratch_bytes_);
  offsets_[WaveOutput] = AddSlab(wave_activation_bytes, scratch_bytes_);
  offsets_[Contributions] = AddSlab(SafeInt<size_t>(max_routes_) * hidden_size_ * sizeof(half), scratch_bytes_);
  offsets_[RouteBias] = AddSlab(banks_[5] ? static_cast<size_t>(SafeInt<size_t>(max_routes_) * hidden_size_ * sizeof(half)) : size_t{0},
                                scratch_bytes_);
  offsets_[RunnerWorkspace] = AddSlab(workspace_bytes_, scratch_bytes_);
  ORT_ENFORCE(scratch_bytes_ <= SafeInt<size_t>(gpu_limit),
              "QMoEOffload needs ", scratch_bytes_, " GPU scratch bytes (expert wave, route slabs, activations and ",
              workspace_bytes_, " runner workspace), exceeding gpu_working_bytes=", gpu_limit, ".");

  expert_to_slot_.reserve(num_experts_);
  for (int expert = 0; expert < num_experts_; ++expert) expert_to_slot_.emplace_back(-1);
  wave_experts_.reserve(expert_wave_size_);
  for (int expert = 0; expert < expert_wave_size_; ++expert) wave_experts_.emplace_back(0);
  pinned_ring_ = AllocateBufferOnCPUPinned<uint8_t>(staging_bytes);
  ORT_ENFORCE(pinned_ring_ != nullptr, "QMoEOffload failed to allocate its bounded pinned staging ring.");
  if (staging_optimization_ & 2) std::memset(pinned_ring_.get(), 0, expert_slab_bytes_);
}

QMoEOffload::~QMoEOffload() {
  // 4-step teardown order (v3 Section 7.7 / F10):
  // 1) Drain participating auxiliary and landing copy streams while CPU expert service remains active.
  //    Wait with failed-context guard so a dead CUDA context that skipped host callbacks never hangs.
  if (aux_stream_) {
    const cudaError_t sync_err = cudaStreamSynchronize(aux_stream_);
    if (sync_err != cudaSuccess) {
      async_in_flight_.store(false, std::memory_order_release);
      async_failed_.store(true, std::memory_order_release);
    }
  }
  if (promo_stream_) {
    static_cast<void>(cudaStreamSynchronize(promo_stream_));
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (async_in_flight_.load(std::memory_order_acquire)) {
    if (cudaPeekAtLastError() != cudaSuccess || std::chrono::steady_clock::now() > deadline) {
      break;
    }
    std::this_thread::yield();
  }
  if (stream_shared_) {
    std::lock_guard<std::mutex> lock(stream_shared_->mutex);
    if (stream_shared_->copy_stream) {
      static_cast<void>(cudaStreamSynchronize(stream_shared_->copy_stream));
    }
  }
  if (callback_timing_ && callback_timing_->SelectedCalls() > 0 && callback_timing_->Quiescent()) {
    try {
      callback_timing_->Persist();
    } catch (...) {
    }
  }
  if (join_event_) cudaEventDestroy(join_event_);
  if (fork_event_) cudaEventDestroy(fork_event_);
  if (aux_stream_) cudaStreamDestroy(aux_stream_);
  if (promo_ready_) cudaEventDestroy(promo_ready_);
  if (promo_stream_) cudaStreamDestroy(promo_stream_);
  // 4) Unregister pinned host ranges only after all DMA and CPU readers have finished.
  if (experts_registered_) {
    for (const void* ptr : registered_host_ptrs_) {
      if (ptr) static_cast<void>(cudaHostUnregister(const_cast<void*>(ptr)));
    }
    registered_host_ptrs_.clear();
  }
  if (cpu_timing_) {
    std::fprintf(stderr,
                 "QMoEOffload cpu layer=%lld calls=%llu routes=%llu cold_routes=%llu cold_experts=%llu cpu_us=%llu classify_us=%llu "
                 "gap_in_us=%llu gap_in_n=%llu gap_out_us=%llu gap_out_n=%llu prep_us=%llu fc1_us=%llu fc2_us=%llu write_us=%llu\n",
                 static_cast<long long>(layer_index_), static_cast<unsigned long long>(timing_calls_),
                 static_cast<unsigned long long>(timing_routes_), static_cast<unsigned long long>(timing_cold_routes_),
                 static_cast<unsigned long long>(timing_cold_experts_), static_cast<unsigned long long>(timing_cpu_ns_ / 1000),
                 static_cast<unsigned long long>(timing_classify_ns_ / 1000),
                 static_cast<unsigned long long>(g_gap_in_ns.load() / 1000), static_cast<unsigned long long>(g_gap_in_n.load()),
                 static_cast<unsigned long long>(g_gap_out_ns.load() / 1000), static_cast<unsigned long long>(g_gap_out_n.load()),
                 static_cast<unsigned long long>(timing_prep_ns_ / 1000), static_cast<unsigned long long>(timing_fc1_ns_ / 1000),
                 static_cast<unsigned long long>(timing_fc2_ns_ / 1000), static_cast<unsigned long long>(timing_write_ns_ / 1000));
  }
  if (resident_log_) {
    std::fprintf(stderr, "QMoEOffload resident layer=%lld slots=%d hit_routes=%llu routes=%llu promotions=%llu\n",
                 static_cast<long long>(layer_index_), resident_slots_, static_cast<unsigned long long>(resident_hit_routes_),
                 static_cast<unsigned long long>(resident_total_routes_), static_cast<unsigned long long>(resident_promoted_));
    for (size_t phase = 0; phase < phase_total_routes_.size(); ++phase) {
      std::fprintf(stderr, "QMoEOffload resident_phase layer=%lld phase=%zu hit_routes=%llu routes=%llu\n",
                   static_cast<long long>(layer_index_), phase,
                   static_cast<unsigned long long>(phase_hit_routes_[phase]),
                   static_cast<unsigned long long>(phase_total_routes_[phase]));
    }
  }
  if (stream_rows_ > 0 && resident_log_) {
    std::fprintf(stderr,
                 "QMoEOffload stream layer=%lld tiles=%llu experts=%llu routes=%llu cpu_routes=%llu waves=%llu reused_promotions=%llu\n",
                 static_cast<long long>(layer_index_), static_cast<unsigned long long>(stream_tiles_),
                 static_cast<unsigned long long>(stream_experts_), static_cast<unsigned long long>(stream_routes_),
                 static_cast<unsigned long long>(stream_cpu_routes_), static_cast<unsigned long long>(stream_waves_),
                 static_cast<unsigned long long>(stream_reused_promotions_));
  }
}

void QMoEOffload::StageExperts(const int* experts, int count) const {
  uint8_t* const pinned = pinned_ring_.get();
  for (int compact = 0; compact < count; ++compact) {
    const size_t expert = static_cast<size_t>(experts[compact]);
    for (size_t bank = 0; bank < 2; ++bank) {
      const size_t bytes = expert_bytes_[bank];
      const auto* source = static_cast<const uint8_t*>(banks_[bank]->DataRaw()) + expert * bytes;
      std::memcpy(pinned + staging_offsets_[bank] + static_cast<size_t>(compact) * bytes, source, bytes);
    }
    for (size_t bank = 2; bank < 4; ++bank) {
      const int columns = bank == 2 ? 2 * inter_size_ : hidden_size_;
      const int blocks = (bank == 2 ? hidden_size_ : inter_size_) / 32;
      const size_t elements = static_cast<size_t>(columns) * blocks;
      const auto* source = banks_[bank]->Data<MLFloat16>() + expert * elements;
      auto* destination = reinterpret_cast<MLFloat16*>(pinned + staging_offsets_[bank] +
                                                       static_cast<size_t>(compact) * expert_bytes_[bank]);
      if (staging_optimization_ & 1) {
        qmoe::TransposeOffloadScales(source, destination, columns, blocks);
      } else {
        for (int column = 0; column < columns; ++column) {
          for (int block = 0; block < blocks; ++block) {
            destination[static_cast<size_t>(block) * columns + column] =
                source[static_cast<size_t>(column) * blocks + block];
          }
        }
      }
    }
    for (size_t bank = 4; bank < banks_.size(); ++bank) {
      if (!banks_[bank]) continue;
      const size_t bytes = expert_bytes_[bank];
      const auto* source = static_cast<const uint8_t*>(banks_[bank]->DataRaw()) + expert * bytes;
      std::memcpy(pinned + staging_offsets_[bank] + static_cast<size_t>(compact) * bytes, source, bytes);
    }
  }
}

Status QMoEOffload::ComputeInternal(OpKernelContext* context) const {
  if (shared_mode_ || nvfp4_mode_) return ComputeShared(context);
  std::lock_guard<std::mutex> lock(mutex_);
  ORT_RETURN_IF(stream_failed_, "QMoEOffload stream synchronization failed on a previous call.");
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(input != nullptr && router_probs != nullptr, "QMoEOffload requires input and router_probs.");
  ORT_RETURN_IF_NOT(input->IsDataType<MLFloat16>() && router_probs->IsDataType<MLFloat16>(),
                    "QMoEOffload requires FP16 input and router_probs.");
  ORT_RETURN_IF_NOT(input->Location().device.Type() == OrtDevice::GPU &&
                        router_probs->Location().device.Type() == OrtDevice::GPU,
                    "QMoEOffload activation and router_probs must be on CUDA.");
  const auto& input_shape = input->Shape();
  ORT_RETURN_IF_NOT((input_shape.NumDimensions() == 2 || input_shape.NumDimensions() == 3) &&
                        input_shape[input_shape.NumDimensions() - 1] == hidden_size_,
                    "QMoEOffload input must have shape [tokens,H] or [batch,sequence,H].");
  const int64_t total_rows = input_shape.NumDimensions() == 2
                                 ? input_shape[0]
                                 : static_cast<int64_t>(SafeInt<int64_t>(input_shape[0]) * input_shape[1]);
  const auto& router_shape = router_probs->Shape();
  ORT_RETURN_IF_NOT(router_shape.NumDimensions() == 2 && router_shape[0] == total_rows && router_shape[1] == num_experts_,
                    "QMoEOffload router shape must be [flattened_tokens,E].");
  Tensor* output = context->Output(0, input_shape);
  if (total_rows == 0) return Status::OK();

  cudaStream_t stream = Stream(context);
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture_status));
  ORT_RETURN_IF(capture_status != cudaStreamCaptureStatusNone,
                "QMoEOffload cannot execute while the CUDA stream is capturing a graph.");

  auto scratch = GetScratchBuffer<uint8_t>(scratch_bytes_, GetComputeStream(context));
  ORT_RETURN_IF_NOT(scratch != nullptr, "QMoEOffload failed to allocate bounded GPU scratch.");
  StreamDrain drain(stream, stream_failed_);

  uint8_t* const base = scratch.get();
  uint8_t* const pinned = pinned_ring_.get();
  auto* router_scales = reinterpret_cast<float*>(base + offsets_[RouterScales]);
  auto* router_indices = reinterpret_cast<int*>(base + offsets_[RouterIndices]);
  auto* unit_scales = reinterpret_cast<float*>(base + offsets_[UnitScales]);
  auto* identity_permutation = reinterpret_cast<int*>(base + offsets_[IdentityPermutation]);
  auto* route_ids = reinterpret_cast<int*>(base + offsets_[RouteIds]);
  auto* compact_experts = reinterpret_cast<int*>(base + offsets_[CompactExperts]);
  auto* wave_permutation = reinterpret_cast<int*>(base + offsets_[WavePermutation]);
  auto* gathered_input = reinterpret_cast<half*>(base + offsets_[GatheredInput]);
  auto* wave_output = reinterpret_cast<half*>(base + offsets_[WaveOutput]);
  auto* contributions = reinterpret_cast<half*>(base + offsets_[Contributions]);
  auto* route_bias = banks_[5] ? reinterpret_cast<half*>(base + offsets_[RouteBias]) : nullptr;
  auto* workspace = reinterpret_cast<char*>(base + offsets_[RunnerWorkspace]);
  auto* host_indices = reinterpret_cast<int*>(pinned + host_indices_offset_);
  auto* host_routes = reinterpret_cast<int*>(pinned + host_routes_offset_);
  auto* host_compact = reinterpret_cast<int*>(pinned + host_compact_offset_);

  ck::ActivationParams activation_params(activation_type_);
  activation_params.alpha = activation_alpha_;
  activation_params.beta = activation_beta_;
  activation_params.limit = swiglu_limit_;
  activation_params.swiglu_fusion = swiglu_fusion_;
  const ck::MOEParallelismConfig parallelism;
  const auto quant_params = ck::QuantParams::GroupWise(32, base + offsets_[Fc1Scales], base + offsets_[Fc2Scales]);
  const void* fc1_bias = banks_[4] ? reinterpret_cast<const void*>(base + offsets_[Fc1Bias]) : nullptr;

  for (int64_t row_offset = 0; row_offset < total_rows;) {
    const int tile_rows = static_cast<int>(std::min<int64_t>(token_tile_size_, total_rows - row_offset));
    const int route_count = SafeInt<int>(tile_rows) * k_;
    const size_t activation_offset = SafeInt<size_t>(row_offset) * hidden_size_;
    const size_t router_offset = SafeInt<size_t>(row_offset) * num_experts_;
    const half* tile_input = reinterpret_cast<const half*>(input->DataRaw()) + activation_offset;
    half* tile_output = reinterpret_cast<half*>(output->MutableDataRaw()) + activation_offset;
    const half* tile_router = reinterpret_cast<const half*>(router_probs->DataRaw()) + router_offset;

    LaunchSoftmaxTopK(tile_router, router_scales, router_indices, tile_rows, num_experts_,
                      static_cast<int>(k_), normalize_routing_weights_, stream);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        host_indices, router_indices, SafeInt<size_t>(route_count) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_RETURN_IF_ERROR(drain.Synchronize());

    ORT_RETURN_IF_NOT(qmoe::InitializeOffloadRoutes(
                          host_indices, route_count, num_experts_, static_cast<int>(k_), expert_to_slot_.data()),
                      "QMoEOffload observed an out-of-range or duplicate top-k expert index.");
    CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMetadata(
        unit_scales, identity_permutation, std::max(route_count, max_wave_routes_), stream));

    for (int first_expert = 0; first_expert < num_experts_;) {
      const qmoe::OffloadWave wave = qmoe::MakeOffloadWave(
          host_indices, route_count, num_experts_, first_expert, expert_wave_size_,
          expert_to_slot_.data(), wave_experts_.data(), host_routes, host_compact);
      first_expert = wave.next_expert;
      if (wave.expert_count == 0) break;

      const int compute_rows = std::max(wave.route_count, kGroupedMoeMinRows);
      for (int row = wave.route_count; row < compute_rows; ++row) {
        host_routes[row] = host_routes[0];
        host_compact[row] = host_compact[0];
      }
      StageExperts(wave_experts_.data(), wave.expert_count);
      if (staging_optimization_ & 2) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(base, pinned, expert_slab_bytes_, cudaMemcpyHostToDevice, stream));
      } else {
        for (size_t bank = 0; bank < banks_.size(); ++bank) {
          if (expert_bytes_[bank] == 0) continue;
          const size_t bytes = SafeInt<size_t>(wave.expert_count) * expert_bytes_[bank];
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
              base + offsets_[bank], pinned + staging_offsets_[bank], bytes, cudaMemcpyHostToDevice, stream));
        }
      }
      const size_t metadata_bytes = SafeInt<size_t>(compute_rows) * sizeof(int);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(route_ids, host_routes, metadata_bytes, cudaMemcpyHostToDevice, stream));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
          compact_experts, host_compact, metadata_bytes, cudaMemcpyHostToDevice, stream));
      CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadGather(
          tile_input, route_ids, gathered_input, compute_rows, hidden_size_, static_cast<int>(k_), stream));

      runner_->runMoe(gathered_input, nullptr, compact_experts, unit_scales,
                      base + offsets_[Fc1Weights], fc1_bias, activation_type_,
                      base + offsets_[Fc2Weights], nullptr, quant_params,
                      compute_rows, hidden_size_, inter_size_, wave.expert_count, 1,
                      workspace, wave_output, wave_permutation, parallelism, activation_params, {}, stream);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
      CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
          wave_output, route_ids, contributions, wave.route_count, hidden_size_, tile_rows, static_cast<int>(k_), stream));
      if (route_bias != nullptr) {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
            reinterpret_cast<const half*>(base + offsets_[Fc2Bias]), compact_experts, route_ids,
            route_bias, wave.route_count, hidden_size_, stream));
      }
      CUDA_RETURN_IF_ERROR(drain.Synchronize());
    }

    ck::finalizeMoeRoutingKernelLauncher<half, half, half>(
        contributions, tile_output, route_bias, router_scales, identity_permutation, nullptr,
        identity_permutation, nullptr, tile_rows, hidden_size_, static_cast<int>(k_),
        route_count, parallelism, false, stream);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    row_offset += tile_rows;
  }
  CUDA_RETURN_IF_ERROR(drain.Wait());
  return Status::OK();
}

void QMoEOffload::InitializeNvfp4Offload(const OpKernelInfo& info) {
  const int64_t expected_block_size = mxfp4_mode_ ? 32 : 16;
  ORT_ENFORCE(block_size_ == -1 || block_size_ == expected_block_size,
              "QMoEOffload FP4 mode requires block_size=", expected_block_size, ", got ", block_size_, ".");
  block_size_ = expected_block_size;
  const auto prepacked = info.GetAttrOrDefault<int64_t>("weights_prepacked", -1);
  ORT_ENFORCE(prepacked == 1, "QMoEOffload FP4 mode requires weights_prepacked=1 (row-major FP4).");

  for (int index : {8, 9, 10, 11, 12, 13, 17, 18, 19, 20}) {
    ORT_ENFORCE(!InputExists(info, index), "QMoEOffload FP4 unsupported input slot ", index, " is present.");
  }

  layer_index_ = info.GetAttrOrDefault<int64_t>("layer_index", 0);
  if (layer_index_ < 0) layer_index_ = 0;

  // Check if using contiguous expert_blocks or 6-input NVFP4/MXFP4 CPU bank tensors.
  const bool has_blocks_21 = InputExists(info, 21);
  const bool has_blocks_2 = !has_blocks_21 && info.GetAttrOrDefault<int64_t>("layout_version", -1) == 1 &&
                            InputExists(info, 2) && !InputExists(info, 5);
  if (has_blocks_21 || has_blocks_2) {
    const int blk_slot = has_blocks_21 ? 21 : 2;
    v3_compact_shared_ = has_blocks_2;
    const auto hidden = info.GetAttrOrDefault<int64_t>("hidden_size", 0);
    const auto inter = info.GetAttrOrDefault<int64_t>("inter_size", 0);
    const bool layout_ok = mxfp4_mode_ ? qmoe_shared::MakeMxfp4ExpertBlockLayout(hidden, inter, &nvfp4_layout_)
                                       : qmoe_shared::MakeNvfp4ExpertBlockLayout(hidden, inter, &nvfp4_layout_);
    ORT_ENFORCE(layout_ok, "QMoEOffload FP4 shared mode has invalid dimensions.");
    ORT_ENFORCE(info.TryGetConstantInput(blk_slot, &expert_blocks_) && expert_blocks_ &&
                    expert_blocks_->IsDataType<uint8_t>() &&
                    expert_blocks_->Location().device.Type() == OrtDevice::CPU &&
                    expert_blocks_->Shape().NumDimensions() == 2 &&
                    expert_blocks_->Shape()[0] > 0 &&
                    expert_blocks_->Shape()[1] == static_cast<int64_t>(nvfp4_layout_.stride),
                "QMoEOffload FP4 expert_blocks must be CPU uint8 [E, stride].");
    num_experts_ = static_cast<int>(expert_blocks_->Shape()[0]);
    hidden_size_ = static_cast<int>(hidden);
    inter_size_ = static_cast<int>(inter);
  } else {
    ORT_ENFORCE(info.TryGetConstantInput(2, &nvfp4_fc1_w_) && nvfp4_fc1_w_ &&
                    info.TryGetConstantInput(3, &nvfp4_fc1_s_) && nvfp4_fc1_s_ &&
                    info.TryGetConstantInput(5, &nvfp4_fc2_w_) && nvfp4_fc2_w_ &&
                    info.TryGetConstantInput(6, &nvfp4_fc2_s_) && nvfp4_fc2_s_,
                "QMoEOffload FP4 bank mode requires constant CPU initializers for slots 2, 3, 5, 6.");
    if (mxfp4_mode_) {
      ORT_ENFORCE(nvfp4_fc1_s_->IsDataType<Float8E8M0>() && nvfp4_fc2_s_->IsDataType<Float8E8M0>(),
                  "QMoEOffload MXFP4 requires Float8E8M0 block scales.");
    } else {
      ORT_ENFORCE(nvfp4_fc1_s_->IsDataType<Float8E4M3FN>() && nvfp4_fc2_s_->IsDataType<Float8E4M3FN>(),
                  "QMoEOffload NVFP4 requires Float8E4M3FN block scales.");
    }
    for (const Tensor* t : {nvfp4_fc1_w_, nvfp4_fc1_s_, nvfp4_fc2_w_, nvfp4_fc2_s_}) {
      ORT_ENFORCE(t->Location().device.Type() == OrtDevice::CPU && t->DataRaw() != nullptr,
                  "QMoEOffload FP4 expert weights/scales must be on CPU.");
    }
    const auto& fc1_shape = nvfp4_fc1_w_->Shape();
    const auto& fc2_shape = nvfp4_fc2_w_->Shape();
    ORT_ENFORCE(fc1_shape.NumDimensions() == 3 && fc2_shape.NumDimensions() == 3 &&
                    fc1_shape[0] == fc2_shape[0] && fc1_shape[0] > 0 &&
                    fc1_shape[1] == fc2_shape[2] * 2 && fc1_shape[2] == fc2_shape[1],
                "QMoEOffload FP4 weight shapes are inconsistent.");
    num_experts_ = static_cast<int>(fc1_shape[0]);
    hidden_size_ = static_cast<int>(fc1_shape[1]);
    inter_size_ = static_cast<int>(fc2_shape[1]);
    const bool layout_ok = mxfp4_mode_
                               ? qmoe_shared::MakeMxfp4ExpertBlockLayout(hidden_size_, inter_size_, &nvfp4_layout_)
                               : qmoe_shared::MakeNvfp4ExpertBlockLayout(hidden_size_, inter_size_, &nvfp4_layout_);
    ORT_ENFORCE(layout_ok, "QMoEOffload FP4 dimensions are invalid.");
    if (InputExists(info, 15)) {
      ORT_ENFORCE(info.TryGetConstantInput(15, &nvfp4_fc1_g_) && nvfp4_fc1_g_ &&
                      nvfp4_fc1_g_->Location().device.Type() == OrtDevice::CPU,
                  "QMoEOffload FP4 fc1_global_scale must be a constant CPU initializer.");
    } else {
      ORT_ENFORCE(mxfp4_mode_, "QMoEOffload NVFP4 requires fc1_global_scale at slot 15.");
    }
    if (InputExists(info, 16)) {
      ORT_ENFORCE(info.TryGetConstantInput(16, &nvfp4_fc2_g_) && nvfp4_fc2_g_ &&
                      nvfp4_fc2_g_->Location().device.Type() == OrtDevice::CPU,
                  "QMoEOffload FP4 fc2_global_scale must be a constant CPU initializer.");
    } else {
      ORT_ENFORCE(mxfp4_mode_, "QMoEOffload NVFP4 requires fc2_global_scale at slot 16.");
    }
    if (!nvfp4_fc1_g_ || !nvfp4_fc2_g_) {
      default_global_scales_.assign(static_cast<size_t>(num_experts_), 1.0f);
    }
    if (InputExists(info, 4)) {
      ORT_ENFORCE(info.TryGetConstantInput(4, &nvfp4_fc1_b_) && nvfp4_fc1_b_ != nullptr,
                  "QMoEOffload NVFP4 fc1_bias must be constant.");
    }
    if (InputExists(info, 7)) {
      ORT_ENFORCE(info.TryGetConstantInput(7, &nvfp4_fc2_b_) && nvfp4_fc2_b_ != nullptr,
                  "QMoEOffload NVFP4 fc2_bias must be constant.");
      if (nvfp4_fc2_b_->Location().device.Type() == OrtDevice::CPU) {
        fc2_bias_gpu_bytes_ = nvfp4_fc2_b_->SizeInBytes();
        fc2_bias_gpu_ = GetTransientScratchBuffer<uint8_t>(fc2_bias_gpu_bytes_);
        CUDA_CALL_THROW(cudaMemcpy(fc2_bias_gpu_.get(), nvfp4_fc2_b_->DataRaw(), fc2_bias_gpu_bytes_,
                                   cudaMemcpyHostToDevice));
      }
    }
  }

  ORT_ENFORCE(k_ > 0 && k_ <= num_experts_, "QMoEOffload NVFP4 k must be in [1,E].");
  const auto tile = info.GetAttrOrDefault<int64_t>("token_tile_size", 32);
  const auto gpu_limit = info.GetAttrOrDefault<int64_t>("gpu_working_bytes", int64_t{256} << 20);
  const auto pinned_limit = info.GetAttrOrDefault<int64_t>("pinned_staging_bytes", int64_t{32} << 20);
  token_tile_size_ = static_cast<int>(tile);
  max_sync_tile_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("max_sync_tile_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_MAX_SYNC_TILE_TOKENS", token_tile_size_)));
  async_max_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("async_max_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_ASYNC_MAX_TOKENS", token_tile_size_)));
  stream_min_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("stream_min_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_STREAM_MIN_TOKENS", token_tile_size_ + 1)));
  max_routes_ = SafeInt<int>(std::max(max_sync_tile_tokens_, async_max_tokens_)) * k_;
  shared_gpu_limit_ = static_cast<size_t>(gpu_limit);
  shared_pinned_limit_ = static_cast<size_t>(pinned_limit);

  const int default_resident_slots = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_SLOTS", 0);
  resident_slots_ = std::clamp(
      static_cast<int>(info.GetAttrOrDefault<int64_t>("resident_slots", default_resident_slots)), 0, num_experts_);
  const int default_spare_slots = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_SPARE_SLOTS", 0);
  spare_slots_ = std::clamp(
      static_cast<int>(info.GetAttrOrDefault<int64_t>("spare_slots", default_spare_slots)), 0, 16);
  if (resident_slots_ == 0 || resident_slots_ == num_experts_) spare_slots_ = 0;
  dummy_slots_ = 0;  // MoeGemvFp4DecodeKernel natively skips slot < 0 without reading weights!
  physical_slots_ = resident_slots_ + spare_slots_ + dummy_slots_;
  dummy_slot_index_ = -1;

  group_calls_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_GROUPED_CALLS", 8), 1, 64);
  const int default_async = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_ASYNC", 0);
  async_ = (default_async == 1) && (resident_slots_ > 0 || info.GetAttrOrDefault<int64_t>("stream_rows", 0) == 0);
  pool_task_batch_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_POOL_TASK_BATCH", 1), 1, 16);
  cpu_threads_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_THREADS", 40), 1, 256);
  unified_cpu_service_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_UNIFIED_CPU_SERVICE", 1) == 1;
  continuous_decay_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_CONTINUOUS_DECAY", 1) == 1;
  cpu_fp16_boundaries_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_FP16_BOUNDARIES", 0) == 1;
  const int default_stream_rows = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_ROWS", 0);
  stream_rows_ = std::clamp(
      static_cast<int>(info.GetAttrOrDefault<int64_t>("stream_rows", default_stream_rows)), 0, 8192);
  prefill_chunk_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("prefill_chunk_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_CHUNK_TOKENS", stream_rows_)));
  stream_promotion_reuse_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_PROMOTION_REUSE", 0) == 1;
  if (stream_rows_ > 0) {
    const int default_min_rows = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_MIN_ROWS", 16);
    const int default_wave = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_WAVE", 32);
    stream_min_rows_ = std::clamp(
        static_cast<int>(info.GetAttrOrDefault<int64_t>("stream_min_rows", default_min_rows)), 1, 8192);
    stream_wave_ = std::clamp(
        static_cast<int>(info.GetAttrOrDefault<int64_t>("stream_wave", default_wave)), 1, num_experts_);
    stream_promotions_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_PROMOTIONS", 0), 0, 64);
    stream_gpu_limit_ = static_cast<size_t>(std::max(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_GPU_MIB", 1024), 1)) << 20;
  }
  if (async_) {
    async_aux_ = resident_slots_ > 0 && ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_AUX_STREAM", 1) == 1;
    async_work_ = std::make_unique<CpuWork>(group_calls_, num_experts_, hidden_size_, inter_size_);
  }

  const size_t fp4_scale_group = mxfp4_mode_ ? 32 : 16;
  resident_bank_bytes_ = {
      static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * (hidden_size_ / 2)),
      static_cast<size_t>(SafeInt<size_t>(hidden_size_) * (inter_size_ / 2)),
      static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * (hidden_size_ / fp4_scale_group)),
      static_cast<size_t>(SafeInt<size_t>(hidden_size_) * (inter_size_ / fp4_scale_group)),
      sizeof(float),
      sizeof(float)};

  if (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_PIN_EXPERTS", 0) == 1) {
    auto try_register = [this](const Tensor* t) {
      // Only register large (>= 1 MiB) tensors so small tensors sharing a 4 KiB OS page are never double-registered.
      if (!t || !t->DataRaw() || t->SizeInBytes() < (size_t{1} << 20)) return;
      for (const void* existing : registered_host_ptrs_) {
        if (existing == t->DataRaw()) return;
      }
      if (cudaHostRegister(const_cast<void*>(t->DataRaw()), t->SizeInBytes(),
                           cudaHostRegisterPortable | cudaHostRegisterReadOnly) == cudaSuccess) {
        registered_host_ptrs_.push_back(t->DataRaw());
        experts_registered_ = true;
      } else {
        static_cast<void>(cudaGetLastError());
      }
    };
    if (expert_blocks_) {
      try_register(expert_blocks_);
    } else {
      for (const Tensor* t : {nvfp4_fc1_w_, nvfp4_fc1_s_, nvfp4_fc2_w_, nvfp4_fc2_s_}) {
        try_register(t);
      }
    }
  }

  fc1_cols_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_FC1_COLS", 64), 32, 1024) / 32 * 32;
  fc2_cols_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_FC2_COLS", 128), 32, 1024) / 32 * 32;
  cpu_timing_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_TIMING", 0) == 1;

  if (resident_slots_ > 0) {
    resident_promotions_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_PROMOTIONS", 0), 0, 64);
    resident_decay_routes_ = std::max(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_DECAY_ROUTES", 4096), 1);
    resident_phase_scores_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_PHASE_SCORES", 0) == 1;
    resident_margin_percent_ = std::clamp(
        ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_MARGIN_PERCENT", 125), 100, 10000);
    resident_log_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_LOG", 0) == 1;

    size_t slab_bytes = 0;
    for (size_t bank = 0; bank < resident_bank_bytes_.size(); ++bank) {
      resident_bank_offsets_[bank] = AddSlab(SafeInt<size_t>(resident_bank_bytes_[bank]) * physical_slots_, slab_bytes);
    }
    resident_slab_ = GetTransientScratchBuffer<uint8_t>(slab_bytes);
    slot_table_dev_ = GetTransientScratchBuffer<uint8_t>(SafeInt<size_t>(num_experts_) * sizeof(int));
    ORT_ENFORCE(resident_slab_ != nullptr && slot_table_dev_ != nullptr,
                "QMoEOffload NVFP4 failed to allocate resident slab of ", slab_bytes, " bytes.");
    slot_expert_.assign(physical_slots_, -1);
    expert_slot_.assign(num_experts_, -1);
    expert_score_.assign(num_experts_, 0.0f);
    if (resident_phase_scores_) decode_score_.assign(num_experts_, 0.0f);

    // Pre-populate initial resident slots [0, resident_slots_) from profile-seeded initial_experts or 0..C-1.
    std::vector<int64_t> initial_experts = info.GetAttrsOrDefault<int64_t>("initial_experts");
    if (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_PREPOPULATE", 1) == 1) {
      uint8_t* slab = resident_slab_.get();
      for (int slot = 0; slot < resident_slots_; ++slot) {
        const int expert = (static_cast<size_t>(slot) < initial_experts.size())
                               ? static_cast<int>(initial_experts[slot])
                               : slot;
        ORT_ENFORCE(expert >= 0 && expert < num_experts_ && expert_slot_[expert] < 0,
                    "QMoEOffload invalid or duplicate initial_experts entry ", expert, " at slot ", slot, ".");
        const uint8_t* src_ptrs[6]{};
        if (expert_blocks_) {
          const size_t blk_off = SafeInt<size_t>(expert) * nvfp4_layout_.stride;
          const uint8_t* blk = expert_blocks_->Data<uint8_t>() + blk_off;
          src_ptrs[0] = blk + nvfp4_layout_.fc1_weights;
          src_ptrs[1] = blk + nvfp4_layout_.fc2_weights;
          src_ptrs[2] = blk + nvfp4_layout_.fc1_scales;
          src_ptrs[3] = blk + nvfp4_layout_.fc2_scales;
          src_ptrs[4] = blk + nvfp4_layout_.fc1_global_scale;
          src_ptrs[5] = blk + nvfp4_layout_.fc2_global_scale;
        } else {
          const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
          const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
          src_ptrs[0] = static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[0]);
          src_ptrs[1] = static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[1]);
          src_ptrs[2] = static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[2]);
          src_ptrs[3] = static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[3]);
          src_ptrs[4] = reinterpret_cast<const uint8_t*>(fc1_g_base + expert);
          src_ptrs[5] = reinterpret_cast<const uint8_t*>(fc2_g_base + expert);
        }
        for (size_t bank = 0; bank < 6; ++bank) {
          const size_t dst_off = resident_bank_offsets_[bank] + SafeInt<size_t>(slot) * resident_bank_bytes_[bank];
          CUDA_CALL_THROW(cudaMemcpy(slab + dst_off, src_ptrs[bank], resident_bank_bytes_[bank], cudaMemcpyHostToDevice));
        }
        slot_expert_[slot] = expert;
        expert_slot_[expert] = slot;
        const float prior = 4.0f + static_cast<float>(resident_slots_ - slot) * 0.25f;
        expert_score_[expert] = prior;
        if (resident_phase_scores_) decode_score_[expert] = prior;
      }
      CUDA_CALL_THROW(cudaMemcpy(slot_table_dev_.get(), expert_slot_.data(),
                                 SafeInt<size_t>(num_experts_) * sizeof(int), cudaMemcpyHostToDevice));
      slot_table_dirty_ = false;
    } else {
      CUDA_CALL_THROW(cudaMemcpy(slot_table_dev_.get(), expert_slot_.data(),
                                 SafeInt<size_t>(num_experts_) * sizeof(int), cudaMemcpyHostToDevice));
      slot_table_dirty_ = false;
    }

    int dev_id = 0;
    static_cast<void>(cudaGetDevice(&dev_id));
    owned_layer_cache_ = std::make_shared<SharedLayerCache>();
    owned_layer_cache_->device_id = dev_id;
    owned_layer_cache_->layer_index = layer_index_;
    owned_layer_cache_->num_experts = num_experts_;
    owned_layer_cache_->hidden_size = hidden_size_;
    owned_layer_cache_->inter_size = inter_size_;
    owned_layer_cache_->resident_slots = resident_slots_;
    owned_layer_cache_->physical_slots = physical_slots_;
    owned_layer_cache_->bank_bytes = resident_bank_bytes_;
    owned_layer_cache_->bank_offsets = resident_bank_offsets_;
    owned_layer_cache_->slab_ptr = resident_slab_.get();
    owned_layer_cache_->slot_table_dev_ptr = reinterpret_cast<const int*>(slot_table_dev_.get());
    owned_layer_cache_->expert_slot_ptr = &expert_slot_;
    {
      std::lock_guard<std::mutex> lock(SharedLayerCache::Mutex());
      SharedLayerCache::Registry()[{dev_id, layer_index_}] = owned_layer_cache_;
    }
  }

  if (stream_rows_ > 0) {
    for (auto& offsets : stream_bank_offsets_) {
      for (size_t bank = 0; bank < offsets.size(); ++bank) {
        offsets[bank] = AddSlab(SafeInt<size_t>(resident_bank_bytes_[bank]) * stream_wave_, stream_landing_bytes_);
      }
    }
    stream_shared_ = session_controller_ ? session_controller_->LandingRing()
                                         : std::make_shared<onnxruntime::cuda::QMoESessionLandingRing>();
  }
}

void QMoEOffload::InitializeShared(const OpKernelInfo& info) {
  ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("layout_version", -1) == 1 &&
                  info.GetAttrOrDefault<int64_t>("weights_prepacked", -1) == 1,
              "QMoEOffload shared mode requires layout_version=1 and weights_prepacked=1.");
  const auto layout_id = info.GetAttrOrDefault<std::string>("layout_id", qmoe_shared::kSharedLayoutId);
  ORT_ENFORCE(layout_id == qmoe_shared::kSharedLayoutId,
              "QMoEOffload INT4 shared mode requires layout_id='", qmoe_shared::kSharedLayoutId, "', got '", layout_id, "'.");

  int blk_slot = 21;
  if (v3_compact_shared_) {
    blk_slot = 2;
    for (int index = 5; index <= 25; ++index) {
      ORT_ENFORCE(!InputExists(info, index), "QMoEOffload v3 compact shared mode forbids input slot ", index, ".");
    }
  } else {
    for (int index = 2; index <= 20; ++index) {
      ORT_ENFORCE(!InputExists(info, index), "QMoEOffload shared and legacy expert inputs cannot be mixed.");
    }
    const bool optional_diag = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_OPTIONAL_DIAGNOSTICS", 0) == 1;
    ORT_ENFORCE(InputExists(info, 24) && (optional_diag || InputExists(info, 25)),
                "QMoEOffload shared mode requires fc2_bias and token_phase.");
    ORT_ENFORCE(!InputExists(info, 22) && !InputExists(info, 23),
                "QMoEOffload shared zero-cache checkpoint does not yet support resident slabs.");
  }

  const auto hidden = info.GetAttrOrDefault<int64_t>("hidden_size", 0);
  const auto inter = info.GetAttrOrDefault<int64_t>("inter_size", 0);
  layer_index_ = info.GetAttrOrDefault<int64_t>("layer_index", v3_compact_shared_ ? 0 : -1);
  ORT_ENFORCE(layer_index_ >= 0 && hidden <= std::numeric_limits<int>::max() &&
                  inter <= std::numeric_limits<int>::max() / 2 &&
                  qmoe_shared::MakeExpertBlockLayout(hidden, inter, &shared_layout_),
              "QMoEOffload shared mode has invalid dimensions or layer_index.");
  ORT_ENFORCE(info.TryGetConstantInput(blk_slot, &expert_blocks_) && expert_blocks_ &&
                  expert_blocks_->IsDataType<uint8_t>() &&
                  expert_blocks_->Location().device.Type() == OrtDevice::CPU &&
                  expert_blocks_->Location().device.MemType() == OrtDevice::MemType::DEFAULT &&
                  expert_blocks_->Shape().NumDimensions() == 2 &&
                  expert_blocks_->Shape()[0] > 0 &&
                  expert_blocks_->Shape()[0] <= std::numeric_limits<int>::max() &&
                  expert_blocks_->Shape()[1] == static_cast<int64_t>(shared_layout_.stride) &&
                  qmoe_shared::ValidateExpertStorageRange(expert_blocks_->Shape()[0], shared_layout_.stride, 0,
                                                          expert_blocks_->SizeInBytes()) &&
                  qmoe_shared::MakeExpertBlockLease(expert_blocks_->DataRaw(), expert_blocks_->SizeInBytes(),
                                                    alignof(uint16_t), &expert_lease_),
              "QMoEOffload expert_blocks must be immutable pageable CPU uint8 [E,stride].");
  num_experts_ = static_cast<int>(expert_blocks_->Shape()[0]);
  hidden_size_ = static_cast<int>(hidden);
  inter_size_ = static_cast<int>(inter);
  ORT_ENFORCE(k_ > 0 && k_ <= num_experts_, "QMoEOffload shared k must be in [1,E].");

  if (v3_compact_shared_ && InputExists(info, 3)) {
    const Tensor* cpu_fc2_bias = nullptr;
    if (info.TryGetConstantInput(3, &cpu_fc2_bias) && cpu_fc2_bias != nullptr &&
        cpu_fc2_bias->Location().device.Type() == OrtDevice::CPU) {
      fc2_bias_gpu_bytes_ = cpu_fc2_bias->SizeInBytes();
      fc2_bias_gpu_ = GetTransientScratchBuffer<uint8_t>(fc2_bias_gpu_bytes_);
      CUDA_CALL_THROW(cudaMemcpy(fc2_bias_gpu_.get(), cpu_fc2_bias->DataRaw(), fc2_bias_gpu_bytes_,
                                 cudaMemcpyHostToDevice));
    }
  }

  const auto tile = info.GetAttrOrDefault<int64_t>("token_tile_size", 32);
  const auto gpu_limit = info.GetAttrOrDefault<int64_t>("gpu_working_bytes", int64_t{64} << 20);
  const auto pinned_limit = info.GetAttrOrDefault<int64_t>("pinned_staging_bytes", int64_t{8} << 20);
  ORT_ENFORCE(tile > 0 && tile <= std::numeric_limits<int>::max() && gpu_limit > 0 && pinned_limit > 0,
              "QMoEOffload shared tile and memory limits must be positive.");
  token_tile_size_ = static_cast<int>(tile);
  max_sync_tile_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("max_sync_tile_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_MAX_SYNC_TILE_TOKENS", token_tile_size_)));
  async_max_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("async_max_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_ASYNC_MAX_TOKENS", token_tile_size_)));
  stream_min_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("stream_min_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_STREAM_MIN_TOKENS", token_tile_size_ + 1)));
  max_routes_ = SafeInt<int>(std::max(max_sync_tile_tokens_, async_max_tokens_)) * k_;
  ORT_ENFORCE(SafeInt<int64_t>(std::max(max_sync_tile_tokens_, async_max_tokens_)) * num_experts_ <= std::numeric_limits<int>::max(),
              "QMoEOffload shared tile routing exceeds integer limits.");
  shared_gpu_limit_ = static_cast<size_t>(gpu_limit);
  shared_pinned_limit_ = static_cast<size_t>(pinned_limit);

  resident_slots_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_SLOTS", 0), 0, num_experts_);
  spare_slots_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_SPARE_SLOTS", 0), 0, 16);
  if (resident_slots_ == 0 || resident_slots_ == num_experts_) spare_slots_ = 0;
  const bool use_dummy_slot = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_DUMMY_SLOT", 1) == 1;
  dummy_slots_ = (resident_slots_ > 0 && use_dummy_slot) ? 1 : 0;
  physical_slots_ = resident_slots_ + spare_slots_ + dummy_slots_;
  dummy_slot_index_ = dummy_slots_ > 0 ? (resident_slots_ + spare_slots_) : 0;

  group_calls_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_GROUPED_CALLS", 1), 1, 64);
  async_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_ASYNC", 0) == 1;
  pool_task_batch_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_POOL_TASK_BATCH", 1);
  ORT_ENFORCE(pool_task_batch_ >= 1 && pool_task_batch_ <= 16,
              "ORT_QMOE_SHARED_POOL_TASK_BATCH must be in [1,16].");
  ORT_ENFORCE(pool_task_batch_ == 1 || async_,
              "ORT_QMOE_SHARED_POOL_TASK_BATCH>1 requires shared async CPU experts.");
  cpu_fp16_boundaries_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_FP16_BOUNDARIES", 0) == 1;
  unified_cpu_service_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_UNIFIED_CPU_SERVICE", 0) == 1;
  continuous_decay_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_CONTINUOUS_DECAY", 0) == 1;
  stream_rows_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_ROWS", 0), 0, 8192);
  prefill_chunk_tokens_ = static_cast<int>(
      info.GetAttrOrDefault<int64_t>("prefill_chunk_tokens",
                                     ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_PREFILL_CHUNK_TOKENS", stream_rows_)));
  const int promotion_reuse = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_PROMOTION_REUSE", 0);
  ORT_ENFORCE(promotion_reuse == 0 || promotion_reuse == 1,
              "ORT_QMOE_SHARED_STREAM_PROMOTION_REUSE must be 0 or 1.");
  stream_promotion_reuse_ = promotion_reuse == 1;
  ORT_ENFORCE(!stream_promotion_reuse_ || stream_rows_ > 0,
              "ORT_QMOE_SHARED_STREAM_PROMOTION_REUSE=1 requires shared prefill streaming.");
  if (stream_rows_ > 0) {
    ORT_ENFORCE(stream_rows_ > token_tile_size_ && resident_slots_ > 0 &&
                    ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_PIN_EXPERTS", 0) == 1,
                "ORT_QMOE_SHARED_STREAM_ROWS needs a value above the token tile, resident slots and PIN_EXPERTS=1.");
    stream_min_rows_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_MIN_ROWS", 16), 1, 8192);
    stream_wave_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_WAVE", 32), 1, num_experts_);
    stream_promotions_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_PROMOTIONS", 16), 1, 64);
    stream_gpu_limit_ = static_cast<size_t>(std::max(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_STREAM_GPU_MIB", 1024), 1)) << 20;
  }
  if (async_) {
#ifdef ORT_USE_EP_API_ADAPTERS
    ORT_ENFORCE(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_PINNED_REUSE", 0) == 1,
                "ORT_QMOE_SHARED_ASYNC=1 requires ORT_QMOE_SHARED_PINNED_REUSE=1.");
#else
    ORT_THROW("ORT_QMOE_SHARED_ASYNC=1 requires the CUDA plugin EP.");
#endif
    cpu_threads_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_THREADS", 40), 1, 256);
    async_aux_ = resident_slots_ > 0 && ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_AUX_STREAM", 0) == 1;
    async_work_ = std::make_unique<CpuWork>(group_calls_, num_experts_, hidden_size_, inter_size_);
  }
  callback_timing_ = qmoe::CallbackTiming::FromEnvironment(layer_index_);
  ORT_ENFORCE(!callback_timing_ || (async_ && async_aux_),
              "QMoE callback trace requires shared async CPU + resident auxiliary stream.");
  if (resident_slots_ == 0) return;
  resident_promotions_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_PROMOTIONS", 2), 1, 64);
  resident_decay_routes_ = std::max(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_DECAY_ROUTES", 4096), 1);
  resident_phase_scores_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_PHASE_SCORES", 0) == 1;
  resident_margin_percent_ = std::clamp(
      ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_MARGIN_PERCENT", 100), 100, 10000);
  resident_log_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_LOG", 1) == 1;
  cpu_timing_ = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_CPU_TIMING", 0) == 1;
  fc1_cols_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_FC1_COLS", 64), 32, 1024) / 32 * 32;
  fc2_cols_ = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_FC2_COLS", 128), 32, 1024) / 32 * 32;
  resident_bank_bytes_ = {static_cast<size_t>(SafeInt<size_t>(hidden_size_) * inter_size_),
                          static_cast<size_t>(SafeInt<size_t>(inter_size_) * hidden_size_ / 2),
                          static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * (hidden_size_ / 32) * sizeof(MLFloat16)),
                          static_cast<size_t>(SafeInt<size_t>(hidden_size_) * (inter_size_ / 32) * sizeof(MLFloat16)),
                          static_cast<size_t>(SafeInt<size_t>(2) * inter_size_ * sizeof(MLFloat16)),
                          0};
  ORT_ENFORCE(resident_bank_bytes_[0] <= shared_layout_.fc1_scales - shared_layout_.fc1_weights &&
                  resident_bank_bytes_[2] <= shared_layout_.fc1_bias - shared_layout_.fc1_scales &&
                  resident_bank_bytes_[1] <= shared_layout_.fc2_scales - shared_layout_.fc2_weights &&
                  resident_bank_bytes_[3] <= shared_layout_.fc2_bias - shared_layout_.fc2_scales &&
                  resident_bank_bytes_[4] <= shared_layout_.fc2_weights - shared_layout_.fc1_bias,
              "QMoEOffload shared block banks are smaller than the resident runner layout.");
  size_t slab_bytes = 0;
  for (size_t bank = 0; bank < 5; ++bank) {
    resident_bank_offsets_[bank] = AddSlab(SafeInt<size_t>(resident_bank_bytes_[bank]) * physical_slots_, slab_bytes);
  }
  runner_ = std::make_unique<ck::CutlassMoeFCRunner<half, cutlass::uint4b_t>>(sm_, activation_type_, false, false);
  ORT_ENFORCE(!runner_->getTactics().empty(), "QMoEOffload has no compatible SM80 INT4 GEMM tactic.");
  const ck::MOEParallelismConfig parallelism;
  const int route_bound = std::max(kGroupedMoeMinRows, max_routes_);
  for (int route_rows = kGroupedMoeMinRows; route_rows <= route_bound; ++route_rows) {
    resident_workspace_bytes_ = std::max(
        resident_workspace_bytes_,
        runner_->getWorkspaceSize(route_rows, hidden_size_, inter_size_, physical_slots_, 1, activation_type_,
                                  parallelism, true, swiglu_fusion_));
  }
  if (stream_rows_ > 0) {
    const int stream_route_bound = SafeInt<int>(stream_rows_) * static_cast<int>(k_);
    const int workspace_experts = std::max(physical_slots_, stream_wave_);
    stream_workspace_bytes_ = resident_workspace_bytes_;
    for (int route_rows = kGroupedMoeMinRows; route_rows <= stream_route_bound; route_rows += route_rows < 1024 ? 1 : 16) {
      stream_workspace_bytes_ = std::max(
          stream_workspace_bytes_,
          runner_->getWorkspaceSize(route_rows, hidden_size_, inter_size_, workspace_experts, 1, activation_type_,
                                    parallelism, true, swiglu_fusion_));
    }
    stream_workspace_bytes_ = std::max(
        stream_workspace_bytes_,
        runner_->getWorkspaceSize(stream_route_bound, hidden_size_, inter_size_, workspace_experts, 1, activation_type_,
                                  parallelism, true, swiglu_fusion_));
  }
  resident_slab_ = GetTransientScratchBuffer<uint8_t>(slab_bytes);
  if (dummy_slots_ > 0 && resident_slab_) {
    // Initialize the dedicated immutable dummy slot (C + S) to zero once so padding/miss reads never read uninitialized memory (F3).
    uint8_t* slab = resident_slab_.get();
    for (size_t bank = 0; bank < 5; ++bank) {
      const size_t off = resident_bank_offsets_[bank] + SafeInt<size_t>(dummy_slot_index_) * resident_bank_bytes_[bank];
      CUDA_CALL_THROW(cudaMemset(slab + off, 0, resident_bank_bytes_[bank]));
    }
  }
  if (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_PIN_EXPERTS", 0) == 1) {
    const auto registered = cudaHostRegister(const_cast<void*>(expert_blocks_->DataRaw()), expert_blocks_->SizeInBytes(),
                                             cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    ORT_ENFORCE(registered == cudaSuccess, "QMoEOffload cannot pin its CPU expert blocks: ", cudaGetErrorString(registered));
    registered_host_ptrs_.push_back(expert_blocks_->DataRaw());
    experts_registered_ = true;
  }
  slot_table_dev_ = GetTransientScratchBuffer<uint8_t>(SafeInt<size_t>(num_experts_) * sizeof(int));
  ORT_ENFORCE(slot_table_dev_ != nullptr, "QMoEOffload cannot allocate its device slot table.");
  ORT_ENFORCE(resident_slab_ != nullptr, "QMoEOffload cannot allocate its resident expert slab of ", slab_bytes, " bytes.");
  slot_expert_.assign(physical_slots_, -1);
  expert_slot_.assign(num_experts_, -1);
  expert_score_.assign(num_experts_, 0.0f);
  if (resident_phase_scores_) decode_score_.assign(num_experts_, 0.0f);
  if (stream_rows_ > 0) {
    for (auto& offsets : stream_bank_offsets_) {
      for (size_t bank = 0; bank < 5; ++bank) {
        offsets[bank] = AddSlab(SafeInt<size_t>(resident_bank_bytes_[bank]) * stream_wave_, stream_landing_bytes_);
      }
    }
    stream_shared_ = session_controller_ ? session_controller_->LandingRing()
                                         : std::make_shared<onnxruntime::cuda::QMoESessionLandingRing>();
  }
}

void QMoEOffload::RecordResidentRoute(int phase, int expert, bool hit) const {
  ++resident_total_routes_;
  ++phase_total_routes_[phase];
  if (hit) {
    ++resident_hit_routes_;
    ++phase_hit_routes_[phase];
  }
  resident_decode_active_ = phase != 0;
  auto& scores = resident_phase_scores_ && resident_decode_active_ ? decode_score_ : expert_score_;
  scores[expert] += 1.0f;
  if (resident_phase_scores_ && !continuous_decay_) {
    auto& routes = phase_routes_since_decay_[resident_decode_active_ ? 1 : 0];
    if (++routes >= resident_decay_routes_) {
      for (auto& score : scores) score *= 0.5f;
      routes = 0;
    }
  }
}

void QMoEOffload::CommitInvocationObservation(const std::array<int, 5>& phase_rows,
                                              const std::array<std::vector<int>, 5>& phase_expert_rows) const {
  if (!continuous_decay_ || resident_slots_ == 0) return;
  // Continuous exponential decay per committed logical invocation (v3 Section 8.2 / F7):
  // d = exp2(-delta_routes / resident_decay_routes_).
  int total_delta_routes = 0;
  int decode_delta_routes = 0;
  for (int p = 0; p < 5; ++p) {
    const int r = phase_rows[p] * static_cast<int>(k_);
    total_delta_routes += r;
    if (p != 0) decode_delta_routes += r;
  }
  if (total_delta_routes > 0) {
    const float d = std::exp2(-static_cast<float>(total_delta_routes) / static_cast<float>(resident_decay_routes_));
    for (auto& s : expert_score_) s *= d;
  }
  if (resident_phase_scores_ && decode_delta_routes > 0) {
    const float d = std::exp2(-static_cast<float>(decode_delta_routes) / static_cast<float>(resident_decay_routes_));
    for (auto& s : decode_score_) s *= d;
  }
  (void)phase_expert_rows;
}

Status QMoEOffload::PromoteResident(cudaStream_t stream, const InlinedVector<InlinedVector<int>>& cold_routes,
                                    int max_promotions, const int* streamed_experts, int streamed_count) const {
  constexpr float kMinScore = 2.0f;
  constexpr float kMargin = 2.0f;
  const auto& scores = resident_phase_scores_ && resident_decode_active_ ? decode_score_ : expert_score_;
  const bool use_bg_promo = async_ && spare_slots_ > 0 && max_promotions == 0 && streamed_experts == nullptr &&
                            ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_BG_PROMO_STREAM", 1) == 1;

  // Step 1: Poll and publish any completed background promotions (v3 Section 7.5 / F3).
  if (!pending_promotions_.empty()) {
    if (use_bg_promo && promo_ready_ != nullptr) {
      const cudaError_t q = cudaEventQuery(promo_ready_);
      if (q == cudaErrorNotReady) {
        return Status::OK();
      }
      CUDA_RETURN_IF_ERROR(q);
    } else if (promo_stream_ != nullptr) {
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(promo_stream_));
    }
    for (const auto& p : pending_promotions_) {
      if (p.evicted_expert >= 0 && p.victim_slot >= 0) {
        expert_slot_[p.evicted_expert] = -1;
        slot_expert_[p.victim_slot] = -1;
      }
      slot_expert_[p.target_slot] = p.candidate;
      expert_slot_[p.candidate] = p.target_slot;
      ++placement_epoch_;
      ++resident_promoted_;
    }
    pending_promotions_.clear();
    slot_table_dirty_ = true;
    if (use_bg_promo) {
      // Allow at least one full invocation on the new slot_table_dev_ before reusing retired victim slots (v3 Section 7.5 / F3).
      return Status::OK();
    }
  }

  const int promotion_limit = max_promotions > 0 ? max_promotions : resident_promotions_;
  if (promotion_limit <= 0) return Status::OK();

  // Count currently active logical residents.
  int active_residents = 0;
  for (int idx = 0; idx < resident_slots_ + spare_slots_; ++idx) {
    if (slot_expert_[idx] >= 0) ++active_residents;
  }

  if (use_bg_promo) {
    for (int promoted = 0; promoted < std::min(promotion_limit, spare_slots_); ++promoted) {
      int candidate = -1;
      float best = kMinScore - 1.0f;
      for (int expert = 0; expert < num_experts_; ++expert) {
        if (cold_routes[expert].empty() || expert_slot_[expert] >= 0) continue;
        bool already_pending = false;
        for (const auto& p : pending_promotions_) {
          if (p.candidate == expert) {
            already_pending = true;
            break;
          }
        }
        if (!already_pending && scores[expert] > best) {
          candidate = expert;
          best = scores[expert];
        }
      }
      if (candidate < 0) break;

      int target_slot = -1;
      for (int idx = 0; idx < resident_slots_ + spare_slots_; ++idx) {
        if (slot_expert_[idx] < 0) {
          bool used = false;
          for (const auto& p : pending_promotions_) {
            if (p.target_slot == idx) {
              used = true;
              break;
            }
          }
          if (!used) {
            target_slot = idx;
            break;
          }
        }
      }
      if (target_slot < 0) break;

      int victim_slot = -1;
      int evicted = -1;
      if (active_residents + static_cast<int>(pending_promotions_.size()) >= resident_slots_) {
        float lowest = std::numeric_limits<float>::infinity();
        for (int idx = 0; idx < resident_slots_ + spare_slots_; ++idx) {
          if (slot_expert_[idx] >= 0) {
            bool used = false;
            for (const auto& p : pending_promotions_) {
              if (p.victim_slot == idx) {
                used = true;
                break;
              }
            }
            if (!used && scores[slot_expert_[idx]] < lowest) {
              lowest = scores[slot_expert_[idx]];
              victim_slot = idx;
            }
          }
        }
        if (victim_slot < 0) break;
        const float threshold = lowest * (static_cast<float>(resident_margin_percent_) / 100.0f) + kMargin;
        if (best < threshold) break;
        evicted = slot_expert_[victim_slot];
      }

      if (!promo_stream_) {
        CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&promo_stream_, cudaStreamNonBlocking));
      }
      if (!promo_ready_) {
        CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&promo_ready_, cudaEventDisableTiming));
      }

      uint8_t* slab = resident_slab_.get();
      if (nvfp4_mode_) {
        const uint8_t* host_sources[6]{};
        if (expert_blocks_) {
          const size_t blk_off = SafeInt<size_t>(candidate) * nvfp4_layout_.stride;
          const uint8_t* blk = expert_blocks_->Data<uint8_t>() + blk_off;
          host_sources[0] = blk + nvfp4_layout_.fc1_weights;
          host_sources[1] = blk + nvfp4_layout_.fc2_weights;
          host_sources[2] = blk + nvfp4_layout_.fc1_scales;
          host_sources[3] = blk + nvfp4_layout_.fc2_scales;
          host_sources[4] = blk + nvfp4_layout_.fc1_global_scale;
          host_sources[5] = blk + nvfp4_layout_.fc2_global_scale;
        } else {
          const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
          const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
          host_sources[0] = static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[0]);
          host_sources[1] = static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[1]);
          host_sources[2] = static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[2]);
          host_sources[3] = static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[3]);
          host_sources[4] = reinterpret_cast<const uint8_t*>(fc1_g_base + candidate);
          host_sources[5] = reinterpret_cast<const uint8_t*>(fc2_g_base + candidate);
        }
        for (size_t bank = 0; bank < 6; ++bank) {
          const size_t slot_offset = resident_bank_offsets_[bank] + SafeInt<size_t>(target_slot) * resident_bank_bytes_[bank];
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(slab + slot_offset, host_sources[bank], resident_bank_bytes_[bank],
                                               cudaMemcpyHostToDevice, promo_stream_));
        }
      } else {
        const size_t block_offset = SafeInt<size_t>(candidate) * shared_layout_.stride;
        const uint8_t* block = expert_blocks_->Data<uint8_t>() + block_offset;
        const std::array<size_t, 5> sources{shared_layout_.fc1_weights, shared_layout_.fc2_weights,
                                            shared_layout_.fc1_scales, shared_layout_.fc2_scales, shared_layout_.fc1_bias};
        for (size_t bank = 0; bank < sources.size(); ++bank) {
          const size_t slot_offset = resident_bank_offsets_[bank] + SafeInt<size_t>(target_slot) * resident_bank_bytes_[bank];
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(slab + slot_offset, block + sources[bank], resident_bank_bytes_[bank],
                                               cudaMemcpyHostToDevice, promo_stream_));
        }
      }
      pending_promotions_.push_back({candidate, target_slot, victim_slot, evicted});
    }
    if (!pending_promotions_.empty()) {
      CUDA_RETURN_IF_ERROR(cudaEventRecord(promo_ready_, promo_stream_));
    }
    return Status::OK();
  }

  for (int promoted = 0; promoted < promotion_limit; ++promoted) {
    int candidate = -1;
    float best = kMinScore - 1.0f;
    for (int expert = 0; expert < num_experts_; ++expert) {
      if (!cold_routes[expert].empty() && expert_slot_[expert] < 0 && scores[expert] > best) {
        candidate = expert;
        best = scores[expert];
      }
    }
    if (candidate < 0) break;

    int target_slot = -1;
    int evicted = -1;
    if (active_residents < resident_slots_) {
      for (int idx = 0; idx < resident_slots_ + spare_slots_; ++idx) {
        if (slot_expert_[idx] < 0) {
          target_slot = idx;
          break;
        }
      }
    } else {
      // Cache is at logical capacity C: find lowest-scoring resident victim, and if a spare physical slot
      // exists (P = C + S + D), write into the free spare physical slot first and unmap the victim!
      int victim_slot = -1;
      float lowest = std::numeric_limits<float>::infinity();
      for (int idx = 0; idx < resident_slots_ + spare_slots_; ++idx) {
        if (slot_expert_[idx] >= 0 && scores[slot_expert_[idx]] < lowest) {
          lowest = scores[slot_expert_[idx]];
          victim_slot = idx;
        }
      }
      if (victim_slot < 0) break;
      const float threshold = lowest * (static_cast<float>(resident_margin_percent_) / 100.0f) + kMargin;
      if (best < threshold) break;
      evicted = slot_expert_[victim_slot];
      for (int idx = 0; idx < resident_slots_ + spare_slots_; ++idx) {
        if (slot_expert_[idx] < 0) {
          target_slot = idx;
          break;
        }
      }
      if (target_slot < 0) {
        target_slot = victim_slot;
      } else {
        slot_expert_[victim_slot] = -1;
      }
    }

    const auto landing = stream_promotion_reuse_
                             ? qmoe::FindLiveStreamExpert(candidate, streamed_experts, streamed_count, stream_wave_)
                             : qmoe::StreamExpertLocation{};
    const bool reuse = landing.buffer >= 0 && stream_shared_ != nullptr && stream_shared_->landing != nullptr;
    if (reuse && stream_shared_->landing_ready[landing.buffer] != nullptr) {
      CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(stream, stream_shared_->landing_ready[landing.buffer], 0));
    }

    uint8_t* slab = resident_slab_.get();
    if (nvfp4_mode_) {
      const uint8_t* host_sources[6]{};
      if (expert_blocks_) {
        const size_t blk_off = SafeInt<size_t>(candidate) * nvfp4_layout_.stride;
        const uint8_t* blk = expert_blocks_->Data<uint8_t>() + blk_off;
        host_sources[0] = blk + nvfp4_layout_.fc1_weights;
        host_sources[1] = blk + nvfp4_layout_.fc2_weights;
        host_sources[2] = blk + nvfp4_layout_.fc1_scales;
        host_sources[3] = blk + nvfp4_layout_.fc2_scales;
        host_sources[4] = blk + nvfp4_layout_.fc1_global_scale;
        host_sources[5] = blk + nvfp4_layout_.fc2_global_scale;
      } else {
        const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
        const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
        host_sources[0] = static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[0]);
        host_sources[1] = static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[1]);
        host_sources[2] = static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[2]);
        host_sources[3] = static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(candidate) * resident_bank_bytes_[3]);
        host_sources[4] = reinterpret_cast<const uint8_t*>(fc1_g_base + candidate);
        host_sources[5] = reinterpret_cast<const uint8_t*>(fc2_g_base + candidate);
      }
      for (size_t bank = 0; bank < 6; ++bank) {
        const size_t slot_offset = resident_bank_offsets_[bank] + SafeInt<size_t>(target_slot) * resident_bank_bytes_[bank];
        const uint8_t* source = host_sources[bank];
        if (reuse) {
          const size_t landing_offset = SafeInt<size_t>(landing.slot) * resident_bank_bytes_[bank];
          source = stream_shared_->landing + stream_bank_offsets_[landing.buffer][bank] + landing_offset;
        }
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(slab + slot_offset, source, resident_bank_bytes_[bank],
                                             reuse ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, stream));
      }
    } else {
      const size_t block_offset = SafeInt<size_t>(candidate) * shared_layout_.stride;
      const uint8_t* block = expert_blocks_->Data<uint8_t>() + block_offset;
      const std::array<size_t, 5> sources{shared_layout_.fc1_weights, shared_layout_.fc2_weights,
                                          shared_layout_.fc1_scales, shared_layout_.fc2_scales, shared_layout_.fc1_bias};
      for (size_t bank = 0; bank < sources.size(); ++bank) {
        const size_t slot_offset = resident_bank_offsets_[bank] + SafeInt<size_t>(target_slot) * resident_bank_bytes_[bank];
        const uint8_t* source = block + sources[bank];
        if (reuse) {
          const size_t landing_offset = SafeInt<size_t>(landing.slot) * resident_bank_bytes_[bank];
          source = stream_shared_->landing + stream_bank_offsets_[landing.buffer][bank] + landing_offset;
        }
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(slab + slot_offset, source, resident_bank_bytes_[bank],
                                             reuse ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, stream));
      }
    }

    if (reuse && stream_shared_->promotion_read_done[landing.buffer] != nullptr) {
      CUDA_RETURN_IF_ERROR(cudaEventRecord(stream_shared_->promotion_read_done[landing.buffer], stream));
      stream_shared_->bank_has_promotion[landing.buffer] = true;
    }

    if (evicted >= 0) {
      expert_slot_[evicted] = -1;
    } else {
      ++active_residents;
    }
    slot_expert_[target_slot] = candidate;
    expert_slot_[candidate] = target_slot;
    slot_table_dirty_ = true;
    ++placement_epoch_;
    ++resident_promoted_;
    if (reuse) ++stream_reused_promotions_;
  }
  return Status::OK();
}

template <typename ActT, typename ParallelRunner>
int QMoEOffload::RunColdExperts(CpuWork& work, int group_count, const ActT* host_input, int tile_rows, ActT* host_result,
                                int* host_cpu_ids, bool compact, ParallelRunner&& run_parallel,
                                const char* skip_experts) const {
  using ExpertCall = CpuWork::ExpertCall;
  const qmoe_shared::SwiGluParams activation{activation_alpha_, activation_beta_, swiglu_limit_};
  const size_t call_rows = qmoe_shared::kMaxRowsPerCall;
  const size_t group_calls = static_cast<size_t>(group_count);
  auto& expert_routes = work.expert_routes;
  auto& calls = work.calls;
  int cpu_rows_written = 0;
  calls.clear();
  for (int expert = 0; expert < num_experts_; ++expert) {
    if (skip_experts && skip_experts[expert]) continue;
    const size_t assigned = expert_routes[expert].size();
    for (size_t first = 0; first < assigned; first += call_rows) {
      calls.push_back({expert, first, static_cast<int>(std::min(call_rows, assigned - first))});
    }
  }
  const int fc1_tasks = (2 * inter_size_ + fc1_cols_ - 1) / fc1_cols_;
  const int fc2_tasks = (hidden_size_ + fc2_cols_ - 1) / fc2_cols_;
  const bool timed = cpu_timing_ && tile_rows <= 4;
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto ns = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
  };
  const size_t token_Floats = SafeInt<size_t>(tile_rows) * hidden_size_;
  if (work.token_rows.size() < token_Floats) work.token_rows.resize(token_Floats);
  work.token_ready.assign(tile_rows, 0);

  auto make_nvfp4_view = [this](int expert) -> qmoe_shared::Nvfp4ExpertView {
    qmoe_shared::Nvfp4ExpertView v{};
    v.bias_is_bf16 = is_bf16_;
    v.is_mxfp4 = mxfp4_mode_;
    if (expert_blocks_) {
      const size_t blk_off = SafeInt<size_t>(expert) * nvfp4_layout_.stride;
      const uint8_t* blk = expert_blocks_->Data<uint8_t>() + blk_off;
      v.fc1_weights = blk + nvfp4_layout_.fc1_weights;
      v.fc1_scales = blk + nvfp4_layout_.fc1_scales;
      v.fc1_global_scale = *reinterpret_cast<const float*>(blk + nvfp4_layout_.fc1_global_scale);
      v.fc2_weights = blk + nvfp4_layout_.fc2_weights;
      v.fc2_scales = blk + nvfp4_layout_.fc2_scales;
      v.fc2_global_scale = *reinterpret_cast<const float*>(blk + nvfp4_layout_.fc2_global_scale);
    } else {
      const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
      const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
      v.fc1_weights = static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[0]);
      v.fc2_weights = static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[1]);
      v.fc1_scales = static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[2]);
      v.fc2_scales = static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()) + static_cast<size_t>(SafeInt<size_t>(expert) * resident_bank_bytes_[3]);
      v.fc1_global_scale = fc1_g_base[expert];
      v.fc2_global_scale = fc2_g_base[expert];
      if (nvfp4_fc1_b_) {
        const size_t bias_off = SafeInt<size_t>(expert) * 2 * inter_size_;
        v.fc1_bias = reinterpret_cast<const uint16_t*>(nvfp4_fc1_b_->DataRaw()) + bias_off;
      }
    }
    return v;
  };

  for (size_t first_call = 0; first_call < calls.size(); first_call += group_calls) {
    const auto t_prep = timed ? now() : std::chrono::steady_clock::time_point{};
    const ExpertCall* group = calls.data() + first_call;
    const int group_size = static_cast<int>(std::min(group_calls, calls.size() - first_call));
    for (int slot = 0; slot < group_size; ++slot) {
      const auto& assignments = expert_routes[group[slot].expert];
      for (int row = 0; row < group[slot].count; ++row) {
        const int token = assignments[group[slot].first + row] / static_cast<int>(k_);
        float* token_row = work.token_rows.data() + static_cast<size_t>(token) * hidden_size_;
        if (!work.token_ready[token]) {
          ActToFloatRow(host_input + static_cast<size_t>(token) * hidden_size_, work.natural.data(), hidden_size_);
          if (nvfp4_mode_) {
            qmoe_shared::ReorderNvfp4ActivationRow(work.natural.data(), token_row, hidden_size_);
          } else {
            qmoe_shared::ReorderActivationRow(work.natural.data(), token_row, hidden_size_);
          }
          work.token_ready[token] = 1;
        }
        if (group[slot].count > 1) {
          std::memcpy(work.reordered.data() + (slot * call_rows + row) * hidden_size_, token_row,
                      static_cast<size_t>(hidden_size_) * sizeof(float));
        }
      }
    }
    const auto t_fc1 = timed ? now() : std::chrono::steady_clock::time_point{};
    auto fc1_task = [&](ptrdiff_t task) {
      const int slot = static_cast<int>(task) / fc1_tasks;
      const int begin = (static_cast<int>(task) % fc1_tasks) * fc1_cols_;
      const int end = std::min(begin + fc1_cols_, 2 * inter_size_);
      const float* act_in = (group[slot].count == 1)
                                ? (work.token_rows.data() +
                                   static_cast<size_t>(expert_routes[group[slot].expert][group[slot].first] / static_cast<int>(k_)) * hidden_size_)
                                : (work.reordered.data() + slot * call_rows * hidden_size_);
      if (nvfp4_mode_) {
        const auto view = make_nvfp4_view(group[slot].expert);
        qmoe_shared::Nvfp4Fc1SwiGluRange(
            hidden_size_, inter_size_, view, act_in,
            group[slot].count, activation, work.intermediate.data() + slot * call_rows * inter_size_,
            begin, end, false, /*bf16_boundaries=*/is_bf16_, /*fp16_boundaries=*/!is_bf16_);
      } else {
        const size_t block_offset = SafeInt<size_t>(group[slot].expert) * shared_layout_.stride;
        const auto* block = expert_blocks_->Data<uint8_t>() + block_offset;
        qmoe_shared::Fc1SwiGluRange(shared_layout_, block, act_in,
                                    group[slot].count, activation,
                                    work.intermediate.data() + slot * call_rows * inter_size_, begin, end,
                                    false, cpu_fp16_boundaries_);
      }
    };
    run_parallel(static_cast<ptrdiff_t>(group_size) * fc1_tasks, fc1_task);
    const auto t_fc2 = timed ? now() : std::chrono::steady_clock::time_point{};
    auto fc2_task = [&](ptrdiff_t task) {
      const int slot = static_cast<int>(task) / fc2_tasks;
      const int begin = (static_cast<int>(task) % fc2_tasks) * fc2_cols_;
      const int end = std::min(begin + fc2_cols_, hidden_size_);
      if (nvfp4_mode_) {
        const auto view = make_nvfp4_view(group[slot].expert);
        qmoe_shared::Nvfp4Fc2Range(
            hidden_size_, inter_size_, view, work.intermediate.data() + slot * call_rows * inter_size_,
            group[slot].count, work.result.data() + slot * call_rows * hidden_size_,
            begin, end, false, /*apply_bias=*/false, /*round_scale_bf16=*/is_bf16_);
      } else {
        const size_t block_offset = SafeInt<size_t>(group[slot].expert) * shared_layout_.stride;
        const auto* block = expert_blocks_->Data<uint8_t>() + block_offset;
        qmoe_shared::Fc2Range(shared_layout_, block, work.intermediate.data() + slot * call_rows * inter_size_,
                              group[slot].count, work.result.data() + slot * call_rows * hidden_size_,
                              begin, end, false, false);
      }
    };
    run_parallel(static_cast<ptrdiff_t>(group_size) * fc2_tasks, fc2_task);
    const auto t_write = timed ? now() : std::chrono::steady_clock::time_point{};
    for (int slot = 0; slot < group_size; ++slot) {
      const auto& assignments = expert_routes[group[slot].expert];
      for (int row = 0; row < group[slot].count; ++row) {
        const int route = assignments[group[slot].first + row];
        const size_t expanded_row = SafeInt<size_t>(route % static_cast<int>(k_)) * tile_rows +
                                    route / static_cast<int>(k_);
        if (compact) host_cpu_ids[cpu_rows_written] = route;
        const size_t destination = SafeInt<size_t>(compact ? static_cast<size_t>(cpu_rows_written++) : expanded_row) * hidden_size_;
        const float* source = work.result.data() + (slot * call_rows + row) * hidden_size_;
        FloatToActRow(source, host_result + destination, hidden_size_);
      }
    }
    if (timed) {
      const auto t_end = now();
      timing_prep_ns_ += ns(t_prep, t_fc1);
      timing_fc1_ns_ += ns(t_fc1, t_fc2);
      timing_fc2_ns_ += ns(t_fc2, t_write);
      timing_write_ns_ += ns(t_write, t_end);
    }
  }
  return cpu_rows_written;
}

Status QMoEOffload::ComputeShared(OpKernelContext* context) const {
  const auto* input = context->Input<Tensor>(0);
  const auto* router = context->Input<Tensor>(1);
  const Tensor* bias = v3_compact_shared_ ? context->Input<Tensor>(3)
                       : nvfp4_mode_      ? (context->Input<Tensor>(7) ? context->Input<Tensor>(7) : context->Input<Tensor>(24))
                                          : context->Input<Tensor>(24);
  const Tensor* phase = v3_compact_shared_ ? context->Input<Tensor>(4) : context->Input<Tensor>(25);

  const bool legacy_strict = shared_mode_ && !v3_compact_shared_ && !nvfp4_mode_ &&
                             ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_OPTIONAL_DIAGNOSTICS", 0) == 0;
  if (legacy_strict) {
    ORT_RETURN_IF_NOT(input && router && bias && phase && input->IsDataType<MLFloat16>() &&
                          router->IsDataType<MLFloat16>() && bias->IsDataType<MLFloat16>() && phase->IsDataType<int8_t>(),
                      "QMoEOffload shared mode requires FP16 input/router/bias and int8 token_phase.");
    for (const auto* tensor : {input, router, bias, phase}) {
      ORT_RETURN_IF_NOT(tensor->Location().device.Type() == OrtDevice::GPU,
                        "QMoEOffload shared activation/router/bias/phase must be on CUDA.");
    }
  } else {
    ORT_RETURN_IF_NOT(input && router, "QMoEOffload requires input and router_probs.");
    ORT_RETURN_IF_NOT(input->Location().device.Type() == OrtDevice::GPU &&
                          router->Location().device.Type() == OrtDevice::GPU,
                      "QMoEOffload input and router_probs must be on CUDA.");
    if (is_bf16_) {
      ORT_RETURN_IF_NOT(input->IsDataType<BFloat16>() && router->IsDataType<BFloat16>(),
                        "QMoEOffload BFloat16 kernel requires BFloat16 input and router_probs.");
    } else {
      ORT_RETURN_IF_NOT(input->IsDataType<MLFloat16>() && router->IsDataType<MLFloat16>(),
                        "QMoEOffload MLFloat16 kernel requires MLFloat16 input and router_probs.");
    }
    if (phase) {
      ORT_RETURN_IF_NOT(phase->IsDataType<int8_t>(), "QMoEOffload token_phase must be int8.");
    }
  }

  const auto& shape = input->Shape();
  if (legacy_strict) {
    ORT_RETURN_IF_NOT(shape.NumDimensions() == 2 && shape[1] == hidden_size_,
                      "QMoEOffload shared input must be packed [T,H].");
  } else {
    ORT_RETURN_IF_NOT((shape.NumDimensions() == 2 || shape.NumDimensions() == 3) &&
                          shape[shape.NumDimensions() - 1] == hidden_size_,
                      "QMoEOffload input must be [T,H] or [B,S,H].");
  }
  const int64_t rows = shape.NumDimensions() == 2
                           ? shape[0]
                           : static_cast<int64_t>(SafeInt<int64_t>(shape[0]) * shape[1]);
  ORT_RETURN_IF_NOT(rows >= 0 && SafeInt<int64_t>(rows) * k_ <= std::numeric_limits<int>::max() &&
                        router->Shape() == TensorShape({rows, num_experts_}) &&
                        (!bias || bias->Shape() == TensorShape({num_experts_, hidden_size_})) &&
                        (!phase || phase->Shape() == TensorShape({rows})),
                    "QMoEOffload shared input shapes are inconsistent.");

  auto* output = context->Output(0, shape);
  auto* row_output = context->Output(1, TensorShape({5, num_experts_}));
  auto* call_output = context->Output(2, TensorShape({5, num_experts_}));
  auto* count_output = context->Output(3, TensorShape({6}));
  if (legacy_strict) {
    ORT_RETURN_IF_NOT(row_output && call_output && count_output, "QMoEOffload shared statistics outputs are required.");
  }
  const size_t statistics_size = SafeInt<size_t>(5) * num_experts_;
  int32_t* route_rows = row_output ? row_output->MutableData<int32_t>() : nullptr;
  int32_t* route_calls = call_output ? call_output->MutableData<int32_t>() : nullptr;
  int32_t* counts = count_output ? count_output->MutableData<int32_t>() : nullptr;
  if (route_rows) std::fill_n(route_rows, statistics_size, 0);
  if (route_calls) std::fill_n(route_calls, statistics_size, 0);
  if (counts) std::fill_n(counts, 6, 0);

  const auto stream = Stream(context);
  cudaStreamCaptureStatus capture;
  CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture));
  ORT_RETURN_IF(capture != cudaStreamCaptureStatusNone, "QMoEOffload shared CPU execution cannot be captured.");
  if (rows == 0) return Status::OK();
  if (async_ && rows <= async_max_tokens_) {
    return ComputeSharedAsync(context, input, router, bias, phase, output, route_rows, route_calls, counts,
                              static_cast<int>(rows));
  }

  std::shared_ptr<SharedLayerCache> borrowed_cache;
  if (nvfp4_mode_ && resident_slots_ == 0 &&
      ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARE_TARGET_SLAB", 1) == 1) {
    int dev_id = 0;
    static_cast<void>(cudaGetDevice(&dev_id));
    std::lock_guard<std::mutex> lock(SharedLayerCache::Mutex());
    auto& reg = SharedLayerCache::Registry();
    auto it = reg.find({dev_id, layer_index_});
    if (it != reg.end()) {
      auto locked = it->second.lock();
      if (locked && locked->num_experts == num_experts_ && locked->hidden_size == hidden_size_ &&
          locked->inter_size == inter_size_ && locked->slab_ptr != nullptr && locked->expert_slot_ptr != nullptr) {
        borrowed_cache = std::move(locked);
      }
    }
  }

  const bool resident = (resident_slots_ > 0) || (borrowed_cache != nullptr);
  const int eff_physical_slots = borrowed_cache ? borrowed_cache->physical_slots : physical_slots_;
  const auto& eff_bank_offsets = borrowed_cache ? borrowed_cache->bank_offsets : resident_bank_offsets_;
  const InlinedVector<int>& eff_expert_slot = borrowed_cache ? *borrowed_cache->expert_slot_ptr : expert_slot_;
  const bool streaming = stream_rows_ > 0 && rows >= stream_min_tokens_;
  const int tile_limit = streaming ? static_cast<int>(std::min<int64_t>(stream_rows_, rows)) : max_sync_tile_tokens_;
  const int route_cap = SafeInt<int>(tile_limit) * static_cast<int>(k_);
  std::unique_lock<std::mutex> resident_lock;
  if (resident) resident_lock = std::unique_lock<std::mutex>(mutex_);
  std::unique_lock<std::mutex> stream_lock;
  if (streaming) stream_lock = std::unique_lock<std::mutex>(stream_shared_->mutex);

  // Sync invocation also drains any prior async work on this node before touching shared score tables (F7).
  if (async_in_flight_.load(std::memory_order_acquire)) {
    if (aux_stream_) CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(aux_stream_));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
  }
  if (!pending_promotions_.empty()) {
    if (promo_stream_) CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(promo_stream_));
    for (const auto& p : pending_promotions_) {
      if (p.evicted_expert >= 0 && p.victim_slot >= 0) {
        expert_slot_[p.evicted_expert] = -1;
        slot_expert_[p.victim_slot] = -1;
      }
      slot_expert_[p.target_slot] = p.candidate;
      expert_slot_[p.candidate] = p.target_slot;
      ++placement_epoch_;
      ++resident_promoted_;
    }
    pending_promotions_.clear();
    slot_table_dirty_ = true;
  }
  ORT_RETURN_IF(async_failed_.load(std::memory_order_acquire),
                "QMoEOffload async CPU expert path failed earlier: ", async_error_message_);

  const size_t elem_size = sizeof(uint16_t);
  size_t gpu_bytes = 0;
  const int route_slots = std::max(route_cap, kGroupedMoeMinRows);
  const size_t route_bytes = SafeInt<size_t>(route_slots) * sizeof(int);
  const int cpu_row_cap = streaming ? std::min(route_cap, (stream_min_rows_ - 1) * num_experts_) : route_cap;
  const size_t contribution_bytes = SafeInt<size_t>(route_cap) * hidden_size_ * elem_size;
  const size_t cpu_row_bytes = SafeInt<size_t>(std::max(cpu_row_cap, 1)) * hidden_size_ * elem_size;
  const size_t resident_row_bytes = SafeInt<size_t>(route_slots) * hidden_size_ * elem_size;
  const size_t nvfp4_inter_bytes = nvfp4_mode_ ? static_cast<size_t>(SafeInt<size_t>(route_slots) * inter_size_ * elem_size) : size_t{0};
  const int stream_waves_max = streaming ? ((num_experts_ + stream_wave_ - 1) / stream_wave_) : 0;
  const size_t stream_meta_bytes = streaming
                                       ? static_cast<size_t>(SafeInt<size_t>(route_slots + stream_waves_max * kGroupedMoeMinRows) * sizeof(int))
                                       : size_t{0};
  const bool has_bias = (bias != nullptr) || (fc2_bias_gpu_ != nullptr);

  const size_t scale_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t index_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t unit_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t identity_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t contribution_offset = AddSlab(contribution_bytes, gpu_bytes);
  const size_t bias_offset = has_bias ? AddSlab(contribution_bytes, gpu_bytes) : 0;
  const size_t hit_ids_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t hit_compact_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t cpu_ids_offset = (resident || streaming) ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t permutation_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t gathered_offset = (resident || streaming) ? AddSlab(resident_row_bytes, gpu_bytes) : 0;
  const size_t wave_output_offset = (resident || streaming) ? AddSlab(resident_row_bytes, gpu_bytes) : 0;
  const size_t nvfp4_inter_offset = (nvfp4_mode_ && (resident || streaming)) ? AddSlab(nvfp4_inter_bytes, gpu_bytes) : 0;
  const size_t cpu_rows_offset = (resident || streaming) ? AddSlab(streaming ? cpu_row_bytes : contribution_bytes, gpu_bytes) : 0;
  const size_t workspace_offset = (!nvfp4_mode_ && resident) ? AddSlab(streaming ? stream_workspace_bytes_ : resident_workspace_bytes_, gpu_bytes) : 0;
  const size_t stream_ids_offset = streaming ? AddSlab(stream_meta_bytes, gpu_bytes) : 0;
  const size_t stream_compact_offset = streaming ? AddSlab(stream_meta_bytes, gpu_bytes) : 0;

  size_t host_bytes = 0;
  const size_t host_index_offset = AddSlab(route_bytes, host_bytes);
  const size_t host_phase_offset = AddSlab(tile_limit, host_bytes);
  const size_t host_input_offset = AddSlab(SafeInt<size_t>(tile_limit) * hidden_size_ * elem_size, host_bytes);
  const size_t host_result_offset = AddSlab(streaming ? cpu_row_bytes : contribution_bytes, host_bytes);
  const size_t host_hit_ids_offset = resident ? AddSlab(route_bytes, host_bytes) : 0;
  const size_t host_hit_compact_offset = resident ? AddSlab(route_bytes, host_bytes) : 0;
  const size_t host_cpu_ids_offset = (resident || streaming) ? AddSlab(route_bytes, host_bytes) : 0;
  const size_t host_stream_ids_offset = streaming ? AddSlab(stream_meta_bytes, host_bytes) : 0;
  const size_t host_stream_compact_offset = streaming ? AddSlab(stream_meta_bytes, host_bytes) : 0;

  ORT_RETURN_IF(gpu_bytes > (streaming ? std::max(shared_gpu_limit_, stream_gpu_limit_) : shared_gpu_limit_) ||
                    host_bytes > (streaming ? std::max(shared_pinned_limit_, stream_gpu_limit_) : shared_pinned_limit_),
                "QMoEOffload shared scratch exceeds configured GPU or pinned limits.");

  auto scratch = GetScratchBuffer<uint8_t>(gpu_bytes, GetComputeStream(context));
  auto completed = std::make_shared<bool>(false);
  IAllocatorUniquePtr<uint8_t> pinned;
  uint8_t* pinned_base = nullptr;
  if (streaming) {
    int dev_id = 0;
    static_cast<void>(cudaGetDevice(&dev_id));
    ORT_RETURN_IF_ERROR(stream_shared_->Ensure(dev_id, stream_landing_bytes_, host_bytes, static_cast<size_t>(stream_waves_max)));
    pinned_base = stream_shared_->pinned;
  } else {
#ifdef ORT_USE_EP_API_ADAPTERS
    const bool reuse_pinned = ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_PINNED_REUSE", 0) == 1;
    pinned = reuse_pinned ? AllocateReusableQMoEPinnedBuffer(host_bytes, shared_pinned_limit_, completed)
                          : AllocateBufferOnCPUPinned<uint8_t>(host_bytes);
#else
    pinned = AllocateBufferOnCPUPinned<uint8_t>(host_bytes);
#endif
    pinned_base = pinned.get();
  }
  ORT_RETURN_IF_NOT(scratch && pinned_base, "QMoEOffload cannot allocate bounded shared scratch.");

  bool failed = false;
  StreamDrain drain(stream, failed, completed);
  auto* base = scratch.get();
  auto* scales = reinterpret_cast<float*>(base + scale_offset);
  auto* indices = reinterpret_cast<int*>(base + index_offset);
  auto* unit_scales = reinterpret_cast<float*>(base + unit_offset);
  auto* identity = reinterpret_cast<int*>(base + identity_offset);
  auto* contributions = base + contribution_offset;
  auto* route_bias = has_bias ? (base + bias_offset) : nullptr;
  auto* host_indices = reinterpret_cast<int*>(pinned_base + host_index_offset);
  auto* host_phases = reinterpret_cast<int8_t*>(pinned_base + host_phase_offset);
  auto* host_input = pinned_base + host_input_offset;
  auto* host_result = pinned_base + host_result_offset;
  auto* host_hit_ids = reinterpret_cast<int*>(pinned_base + host_hit_ids_offset);
  auto* host_hit_compact = reinterpret_cast<int*>(pinned_base + host_hit_compact_offset);
  auto* host_cpu_ids = reinterpret_cast<int*>(pinned_base + host_cpu_ids_offset);
  auto* host_stream_ids = reinterpret_cast<int*>(pinned_base + host_stream_ids_offset);
  auto* host_stream_compact = reinterpret_cast<int*>(pinned_base + host_stream_compact_offset);
  auto* stream_ids = reinterpret_cast<int*>(base + stream_ids_offset);
  auto* stream_compact = reinterpret_cast<int*>(base + stream_compact_offset);
  auto* hit_ids = reinterpret_cast<int*>(base + hit_ids_offset);
  auto* hit_compact = reinterpret_cast<int*>(base + hit_compact_offset);
  auto* cpu_ids = reinterpret_cast<int*>(base + cpu_ids_offset);
  auto* permutation = reinterpret_cast<int*>(base + permutation_offset);
  auto* gathered = base + gathered_offset;
  auto* wave_output = base + wave_output_offset;
  auto* nvfp4_inter_buf = base + nvfp4_inter_offset;
  auto* cpu_rows = base + cpu_rows_offset;
  auto* workspace = reinterpret_cast<char*>(base + workspace_offset);

  ck::ActivationParams runner_activation(activation_type_);
  runner_activation.alpha = activation_alpha_;
  runner_activation.beta = activation_beta_;
  runner_activation.limit = swiglu_limit_;
  runner_activation.swiglu_fusion = swiglu_fusion_;
  uint8_t* slab = borrowed_cache ? borrowed_cache->slab_ptr : (resident ? resident_slab_.get() : nullptr);
  const auto quantization = (!nvfp4_mode_ && resident)
                                ? ck::QuantParams::GroupWise(32, slab + eff_bank_offsets[2], slab + eff_bank_offsets[3])
                                : ck::QuantParams{};
  struct HitRoute {
    int slot;
    int route;
  };
  InlinedVector<HitRoute> hits;
  const ck::MOEParallelismConfig parallelism;
  const int group_calls = std::clamp(ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_GROUPED_CALLS", group_calls_), 1, 64);
  CpuWork cpu_work(group_calls, num_experts_, hidden_size_, inter_size_);
  auto& expert_routes = cpu_work.expert_routes;
  InlinedVector<char> stream_skip;
  InlinedVector<int> stream_list;

  std::array<int, 5> obs_phase_rows{};
  std::array<std::vector<int>, 5> obs_phase_expert_rows{};

  const void* bias_gpu_ptr = fc2_bias_gpu_ ? fc2_bias_gpu_.get()
                                           : (bias && bias->Location().device.Type() == OrtDevice::GPU ? bias->DataRaw() : nullptr);
  IAllocatorUniquePtr<uint8_t> transient_bias_gpu;
  if (!bias_gpu_ptr && bias && bias->Location().device.Type() == OrtDevice::CPU) {
    transient_bias_gpu = GetScratchBuffer<uint8_t>(bias->SizeInBytes(), GetComputeStream(context));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(transient_bias_gpu.get(), bias->DataRaw(), bias->SizeInBytes(),
                                         cudaMemcpyHostToDevice, stream));
    bias_gpu_ptr = transient_bias_gpu.get();
  }

  auto launch_nvfp4_gemv_pair = [&](const void* act_in, const uint8_t* w_base, const std::array<size_t, 6>& bank_offs,
                                    const int* expert_map, int num_exp, int num_r, void* out_buf) -> Status {
    const uint8_t* fc1_w = w_base + bank_offs[0];
    const uint8_t* fc2_w = w_base + bank_offs[1];
    const uint8_t* fc1_s = w_base + bank_offs[2];
    const uint8_t* fc2_s = w_base + bank_offs[3];
    const float* fc1_g = reinterpret_cast<const float*>(w_base + bank_offs[4]);
    const float* fc2_g = reinterpret_cast<const float*>(w_base + bank_offs[5]);
    const int fp4_group = mxfp4_mode_ ? 32 : 16;
    if (is_bf16_) {
      gemv::launch_moe_gemv_fp4_symmetric_interleaved_swiglu<__nv_bfloat16>(
          static_cast<const __nv_bfloat16*>(act_in), fc1_w, nullptr, fc1_s, fc1_g, nullptr,
          reinterpret_cast<__nv_bfloat16*>(nvfp4_inter_buf), nullptr, expert_map, num_exp,
          num_r, inter_size_, hidden_size_, fp4_group, sm_, runner_activation, gemv::MoeGemvConfig::kDefault,
          false, true, nullptr, num_r, stream, true, false);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
      gemv::launch_moe_gemv_fp4_symmetric<__nv_bfloat16>(
          reinterpret_cast<const __nv_bfloat16*>(nvfp4_inter_buf), fc2_w, nullptr, fc2_s, fc2_g, nullptr,
          static_cast<__nv_bfloat16*>(out_buf), nullptr, expert_map, num_exp,
          num_r, hidden_size_, inter_size_, fp4_group, sm_, gemv::MoeGemvConfig::kDefault,
          false, true, stream, true, false);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
    } else {
      gemv::launch_moe_gemv_fp4_symmetric_interleaved_swiglu<half>(
          static_cast<const half*>(act_in), fc1_w, nullptr, fc1_s, fc1_g, nullptr,
          reinterpret_cast<half*>(nvfp4_inter_buf), nullptr, expert_map, num_exp,
          num_r, inter_size_, hidden_size_, fp4_group, sm_, runner_activation, gemv::MoeGemvConfig::kDefault,
          false, true, nullptr, num_r, stream, true, false);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
      gemv::launch_moe_gemv_fp4_symmetric<half>(
          reinterpret_cast<const half*>(nvfp4_inter_buf), fc2_w, nullptr, fc2_s, fc2_g, nullptr,
          static_cast<half*>(out_buf), nullptr, expert_map, num_exp,
          num_r, hidden_size_, inter_size_, fp4_group, sm_, gemv::MoeGemvConfig::kDefault,
          false, true, stream, true, false);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
    }
    return Status::OK();
  };

  for (int64_t start = 0; start < rows;) {
    const int tile_rows = static_cast<int>(std::min<int64_t>(tile_limit, rows - start));
    const int routes = SafeInt<int>(tile_rows) * k_;
    const size_t activation_start = SafeInt<size_t>(start) * hidden_size_;
    const size_t router_start = SafeInt<size_t>(start) * num_experts_;

    if (is_bf16_) {
      LaunchSoftmaxTopK(reinterpret_cast<const __nv_bfloat16*>(router->DataRaw()) + router_start,
                        scales, indices, tile_rows, num_experts_, static_cast<int>(k_), normalize_routing_weights_, stream);
    } else {
      LaunchSoftmaxTopK(reinterpret_cast<const half*>(router->DataRaw()) + router_start,
                        scales, indices, tile_rows, num_experts_, static_cast<int>(k_), normalize_routing_weights_, stream);
    }
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_indices, indices, SafeInt<size_t>(routes) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    if (phase) {
      if (phase->Location().device.Type() == OrtDevice::GPU) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_phases, phase->Data<int8_t>() + start, tile_rows, cudaMemcpyDeviceToHost, stream));
      } else {
        std::memcpy(host_phases, phase->Data<int8_t>() + start, tile_rows);
      }
    } else {
      const int8_t inferred_phase = (rows > async_max_tokens_) ? 0 : 1;
      std::fill_n(host_phases, tile_rows, inferred_phase);
    }
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_input, static_cast<const uint8_t*>(input->DataRaw()) + activation_start * elem_size,
                                         SafeInt<size_t>(tile_rows) * hidden_size_ * elem_size, cudaMemcpyDeviceToHost, stream));
    CUDA_RETURN_IF_ERROR(drain.Synchronize());

    for (auto& assignments : expert_routes) assignments.clear();
    hits.clear();
    for (int token = 0; token < tile_rows; ++token) {
      const int token_phase = host_phases[token];
      ORT_RETURN_IF(token_phase < 0 || token_phase >= 5, "QMoEOffload invalid token_phase.");
      if (counts) ++counts[token_phase];
      ++obs_phase_rows[token_phase];
      for (int rank = 0; rank < k_; ++rank) {
        const int route = token * static_cast<int>(k_) + rank;
        const int expert = host_indices[route];
        ORT_RETURN_IF(expert < 0 || expert >= num_experts_, "QMoEOffload invalid routed expert.");
        for (int previous = 0; previous < rank; ++previous) {
          ORT_RETURN_IF(host_indices[token * static_cast<int>(k_) + previous] == expert,
                        "QMoEOffload duplicate routed expert.");
        }
        const size_t statistic = SafeInt<size_t>(token_phase) * num_experts_ + expert;
        if (route_rows) ++route_rows[statistic];
        if (route_calls) route_calls[statistic] = 1;
        const int resident_slot = resident ? eff_expert_slot[expert] : -1;
        if (resident_slot >= 0) {
          hits.push_back({resident_slot, route});
        } else {
          expert_routes[expert].push_back(route);
        }
        if (resident_slots_ > 0) RecordResidentRoute(token_phase, expert, resident_slot >= 0);
      }
    }

    CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMetadata(unit_scales, identity, std::max(routes, kGroupedMoeMinRows), stream));
    if (!hits.empty()) {
      std::sort(hits.begin(), hits.end(), [](const HitRoute& left, const HitRoute& right) {
        return left.slot != right.slot ? left.slot < right.slot : left.route < right.route;
      });
      const int hit_count = static_cast<int>(hits.size());
      const int compute_rows = nvfp4_mode_ ? hit_count : std::max(hit_count, kGroupedMoeMinRows);
      for (int row = 0; row < compute_rows; ++row) {
        const auto& hit = hits[row < hit_count ? row : 0];
        host_hit_ids[row] = hit.route;
        host_hit_compact[row] = hit.slot;
      }
      const size_t metadata_bytes = SafeInt<size_t>(compute_rows) * sizeof(int);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(hit_ids, host_hit_ids, metadata_bytes, cudaMemcpyHostToDevice, stream));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(hit_compact, host_hit_compact, metadata_bytes, cudaMemcpyHostToDevice, stream));
      if (is_bf16_) {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadGather(
            reinterpret_cast<const __nv_bfloat16*>(input->DataRaw()) + activation_start,
            hit_ids, reinterpret_cast<__nv_bfloat16*>(gathered), compute_rows, hidden_size_, static_cast<int>(k_), stream));
      } else {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadGather(
            reinterpret_cast<const half*>(input->DataRaw()) + activation_start,
            hit_ids, reinterpret_cast<half*>(gathered), compute_rows, hidden_size_, static_cast<int>(k_), stream));
      }
      if (nvfp4_mode_) {
        ORT_RETURN_IF_ERROR(launch_nvfp4_gemv_pair(gathered, slab, eff_bank_offsets, hit_compact,
                                                   eff_physical_slots, compute_rows, wave_output));
      } else {
        runner_->runMoe(reinterpret_cast<half*>(gathered), nullptr, hit_compact, unit_scales,
                        slab + eff_bank_offsets[0], reinterpret_cast<const void*>(slab + eff_bank_offsets[4]), activation_type_,
                        slab + eff_bank_offsets[1], nullptr, quantization, compute_rows, hidden_size_, inter_size_,
                        eff_physical_slots, 1, workspace, reinterpret_cast<half*>(wave_output), permutation, parallelism, runner_activation, {}, stream);
        CUDA_RETURN_IF_ERROR(cudaGetLastError());
      }
      if (is_bf16_) {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
            reinterpret_cast<const __nv_bfloat16*>(wave_output), hit_ids,
            reinterpret_cast<__nv_bfloat16*>(contributions), hit_count, hidden_size_, tile_rows, static_cast<int>(k_), stream));
      } else {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
            reinterpret_cast<const half*>(wave_output), hit_ids,
            reinterpret_cast<half*>(contributions), hit_count, hidden_size_, tile_rows, static_cast<int>(k_), stream));
      }
    }

    const char* skip_experts = nullptr;
    uint64_t tile_stream_routes = 0;
    if (streaming) {
      stream_skip.assign(num_experts_, 0);
      stream_list.clear();
      for (int expert = 0; expert < num_experts_; ++expert) {
        if (static_cast<int>(expert_routes[expert].size()) >= stream_min_rows_) {
          stream_list.push_back(expert);
          stream_skip[expert] = 1;
        }
      }
      skip_experts = stream_skip.data();
      auto& shared = *stream_shared_;
      const size_t stream_count = stream_list.size();
      const int waves = static_cast<int>((stream_count + stream_wave_ - 1) / stream_wave_);
      size_t meta_offset = 0;
      for (int wave = 0; wave < waves; ++wave) {
        const int bank_idx = wave & 1;
        const size_t first = static_cast<size_t>(wave) * stream_wave_;
        const int count = static_cast<int>(std::min<size_t>(stream_wave_, stream_count - first));
        const auto& offsets = stream_bank_offsets_[bank_idx];

        // Event-ordered landing ring reuse across waves and across layers (v3 Section 7.6 / F5).
        if (wave >= 2) {
          CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(shared.copy_stream, shared.gemm_done[wave - 2], 0));
        } else {
          if (shared.bank_has_gemm[bank_idx]) {
            CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(shared.copy_stream, shared.last_gemm_done[bank_idx], 0));
          }
          if (shared.bank_has_promotion[bank_idx]) {
            CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(shared.copy_stream, shared.promotion_read_done[bank_idx], 0));
            shared.bank_has_promotion[bank_idx] = false;
          }
        }

        if (nvfp4_mode_) {
          for (int begin = 0; begin < count;) {
            int end = begin + 1;
            while (end < count && stream_list[first + end] == stream_list[first + end - 1] + 1) ++end;
            const int run_len = end - begin;
            const int first_exp = stream_list[first + begin];
            if (expert_blocks_) {
              const uint8_t* blocks = expert_blocks_->Data<uint8_t>();
              const std::array<size_t, 6> sources{nvfp4_layout_.fc1_weights, nvfp4_layout_.fc2_weights,
                                                  nvfp4_layout_.fc1_scales, nvfp4_layout_.fc2_scales,
                                                  nvfp4_layout_.fc1_global_scale, nvfp4_layout_.fc2_global_scale};
              for (size_t bank = 0; bank < 6; ++bank) {
                const size_t dst = offsets[bank] + SafeInt<size_t>(begin) * resident_bank_bytes_[bank];
                const size_t src = SafeInt<size_t>(first_exp) * nvfp4_layout_.stride + sources[bank];
                CUDA_RETURN_IF_ERROR(cudaMemcpy2DAsync(shared.landing + dst, resident_bank_bytes_[bank],
                                                       blocks + src, nvfp4_layout_.stride,
                                                       resident_bank_bytes_[bank], static_cast<size_t>(run_len),
                                                       cudaMemcpyHostToDevice, shared.copy_stream));
              }
            } else {
              const float* fc1_g_base = nvfp4_fc1_g_ ? nvfp4_fc1_g_->Data<float>() : default_global_scales_.data();
              const float* fc2_g_base = nvfp4_fc2_g_ ? nvfp4_fc2_g_->Data<float>() : default_global_scales_.data();
              const uint8_t* bank_bases[6] = {
                  static_cast<const uint8_t*>(nvfp4_fc1_w_->DataRaw()),
                  static_cast<const uint8_t*>(nvfp4_fc2_w_->DataRaw()),
                  static_cast<const uint8_t*>(nvfp4_fc1_s_->DataRaw()),
                  static_cast<const uint8_t*>(nvfp4_fc2_s_->DataRaw()),
                  reinterpret_cast<const uint8_t*>(fc1_g_base),
                  reinterpret_cast<const uint8_t*>(fc2_g_base)};
              for (size_t bank = 0; bank < 6; ++bank) {
                const size_t dst = offsets[bank] + SafeInt<size_t>(begin) * resident_bank_bytes_[bank];
                const size_t src = SafeInt<size_t>(first_exp) * resident_bank_bytes_[bank];
                const size_t copy_bytes = SafeInt<size_t>(run_len) * resident_bank_bytes_[bank];
                CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(shared.landing + dst, bank_bases[bank] + src, copy_bytes,
                                                     cudaMemcpyHostToDevice, shared.copy_stream));
              }
            }
            begin = end;
          }
        } else {
          const std::array<size_t, 5> sources{shared_layout_.fc1_weights, shared_layout_.fc2_weights,
                                              shared_layout_.fc1_scales, shared_layout_.fc2_scales, shared_layout_.fc1_bias};
          const uint8_t* const blocks = expert_blocks_->Data<uint8_t>();
          for (int begin = 0; begin < count;) {
            int end = begin + 1;
            while (end < count && stream_list[first + end] == stream_list[first + end - 1] + 1) ++end;
            for (size_t bank = 0; bank < sources.size(); ++bank) {
              const size_t destination = offsets[bank] + SafeInt<size_t>(begin) * resident_bank_bytes_[bank];
              const size_t source = SafeInt<size_t>(stream_list[first + begin]) * shared_layout_.stride + sources[bank];
              CUDA_RETURN_IF_ERROR(cudaMemcpy2DAsync(shared.landing + destination, resident_bank_bytes_[bank], blocks + source,
                                                     shared_layout_.stride, resident_bank_bytes_[bank],
                                                     static_cast<size_t>(end - begin), cudaMemcpyHostToDevice,
                                                     shared.copy_stream));
            }
            begin = end;
          }
        }

        CUDA_RETURN_IF_ERROR(cudaEventRecord(shared.copy_done[wave], shared.copy_stream));
        CUDA_RETURN_IF_ERROR(cudaEventRecord(shared.landing_ready[bank_idx], shared.copy_stream));
        ++shared.bank_generation[bank_idx];

        size_t wave_routes = 0;
        for (int local = 0; local < count; ++local) {
          for (const int route : expert_routes[stream_list[first + local]]) {
            host_stream_ids[meta_offset + wave_routes] = route;
            host_stream_compact[meta_offset + wave_routes] = local;
            ++wave_routes;
          }
        }
        const int compute_rows = nvfp4_mode_ ? static_cast<int>(wave_routes)
                                             : std::max(static_cast<int>(wave_routes), kGroupedMoeMinRows);
        for (int row = static_cast<int>(wave_routes); row < compute_rows; ++row) {
          host_stream_ids[meta_offset + row] = host_stream_ids[meta_offset];
          host_stream_compact[meta_offset + row] = host_stream_compact[meta_offset];
        }
        const size_t metadata_bytes = SafeInt<size_t>(compute_rows) * sizeof(int);
        CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(stream, shared.copy_done[wave], 0));
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(stream_ids + meta_offset, host_stream_ids + meta_offset, metadata_bytes,
                                             cudaMemcpyHostToDevice, stream));
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(stream_compact + meta_offset, host_stream_compact + meta_offset, metadata_bytes,
                                             cudaMemcpyHostToDevice, stream));
        if (is_bf16_) {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadGather(
              reinterpret_cast<const __nv_bfloat16*>(input->DataRaw()) + activation_start,
              stream_ids + meta_offset, reinterpret_cast<__nv_bfloat16*>(gathered), compute_rows, hidden_size_,
              static_cast<int>(k_), stream));
        } else {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadGather(
              reinterpret_cast<const half*>(input->DataRaw()) + activation_start,
              stream_ids + meta_offset, reinterpret_cast<half*>(gathered), compute_rows, hidden_size_,
              static_cast<int>(k_), stream));
        }
        uint8_t* landing = shared.landing;
        if (nvfp4_mode_) {
          ORT_RETURN_IF_ERROR(launch_nvfp4_gemv_pair(gathered, landing, offsets, stream_compact + meta_offset,
                                                     count, compute_rows, wave_output));
        } else {
          const auto wave_quantization = ck::QuantParams::GroupWise(32, landing + offsets[2], landing + offsets[3]);
          runner_->runMoe(reinterpret_cast<half*>(gathered), nullptr, stream_compact + meta_offset, unit_scales,
                          landing + offsets[0], reinterpret_cast<const void*>(landing + offsets[4]), activation_type_,
                          landing + offsets[1], nullptr, wave_quantization, compute_rows, hidden_size_, inter_size_,
                          count, 1, workspace, reinterpret_cast<half*>(wave_output), permutation, parallelism, runner_activation, {}, stream);
          CUDA_RETURN_IF_ERROR(cudaGetLastError());
        }
        if (is_bf16_) {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
              reinterpret_cast<const __nv_bfloat16*>(wave_output), stream_ids + meta_offset,
              reinterpret_cast<__nv_bfloat16*>(contributions), static_cast<int>(wave_routes), hidden_size_,
              tile_rows, static_cast<int>(k_), stream));
        } else {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
              reinterpret_cast<const half*>(wave_output), stream_ids + meta_offset,
              reinterpret_cast<half*>(contributions), static_cast<int>(wave_routes), hidden_size_,
              tile_rows, static_cast<int>(k_), stream));
        }
        CUDA_RETURN_IF_ERROR(cudaEventRecord(shared.gemm_done[wave], stream));
        CUDA_RETURN_IF_ERROR(cudaEventRecord(shared.last_gemm_done[bank_idx], stream));
        shared.bank_has_gemm[bank_idx] = true;
        meta_offset += compute_rows;
        tile_stream_routes += wave_routes;
      }
      ++stream_tiles_;
      stream_experts_ += stream_count;
      stream_routes_ += tile_stream_routes;
      stream_waves_ += static_cast<uint64_t>(waves);
      stream_cpu_routes_ += static_cast<uint64_t>(routes) - hits.size() - tile_stream_routes;
    }

    auto parallel_runner = [&](ptrdiff_t count, auto& function) {
      if (unified_cpu_service_ && cpu_threads_ > 0) {
        SharedSpinPool(cpu_threads_).ParallelFor(count, function, pool_task_batch_);
      } else {
        OffloadParallelFor(context, count, function);
      }
    };
    const int cpu_rows_written = is_bf16_
                                     ? RunColdExperts(cpu_work, group_calls, reinterpret_cast<const BFloat16*>(host_input),
                                                      tile_rows, reinterpret_cast<BFloat16*>(host_result), host_cpu_ids,
                                                      resident || streaming, parallel_runner, skip_experts)
                                     : RunColdExperts(cpu_work, group_calls, reinterpret_cast<const MLFloat16*>(host_input),
                                                      tile_rows, reinterpret_cast<MLFloat16*>(host_result), host_cpu_ids,
                                                      resident || streaming, parallel_runner, skip_experts);
    if (resident || streaming) {
      if (cpu_rows_written > 0) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cpu_ids, host_cpu_ids, SafeInt<size_t>(cpu_rows_written) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cpu_rows, host_result, SafeInt<size_t>(cpu_rows_written) * hidden_size_ * elem_size, cudaMemcpyHostToDevice, stream));
        if (is_bf16_) {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
              reinterpret_cast<const __nv_bfloat16*>(cpu_rows), cpu_ids,
              reinterpret_cast<__nv_bfloat16*>(contributions), cpu_rows_written, hidden_size_, tile_rows, static_cast<int>(k_), stream));
        } else {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatter(
              reinterpret_cast<const half*>(cpu_rows), cpu_ids,
              reinterpret_cast<half*>(contributions), cpu_rows_written, hidden_size_, tile_rows, static_cast<int>(k_), stream));
        }
      }
    } else {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(contributions, host_result, SafeInt<size_t>(routes) * hidden_size_ * elem_size, cudaMemcpyHostToDevice, stream));
    }

    if (bias_gpu_ptr) {
      if (is_bf16_) {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
            reinterpret_cast<const __nv_bfloat16*>(bias_gpu_ptr), indices, identity,
            reinterpret_cast<__nv_bfloat16*>(route_bias), routes, hidden_size_, stream));
      } else {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
            reinterpret_cast<const half*>(bias_gpu_ptr), indices, identity,
            reinterpret_cast<half*>(route_bias), routes, hidden_size_, stream));
      }
    }
    if (is_bf16_) {
      ck::finalizeMoeRoutingKernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
          reinterpret_cast<const __nv_bfloat16*>(contributions),
          reinterpret_cast<__nv_bfloat16*>(output->MutableDataRaw()) + activation_start,
          bias_gpu_ptr ? reinterpret_cast<const __nv_bfloat16*>(route_bias) : nullptr,
          scales, identity, nullptr, identity, nullptr, tile_rows, hidden_size_,
          static_cast<int>(k_), routes, parallelism, false, stream);
    } else {
      ck::finalizeMoeRoutingKernelLauncher<half, half, half>(
          reinterpret_cast<const half*>(contributions),
          reinterpret_cast<half*>(output->MutableDataRaw()) + activation_start,
          bias_gpu_ptr ? reinterpret_cast<const half*>(route_bias) : nullptr,
          scales, identity, nullptr, identity, nullptr, tile_rows, hidden_size_,
          static_cast<int>(k_), routes, parallelism, false, stream);
    }
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    CUDA_RETURN_IF_ERROR(drain.Synchronize());

    if (resident_slots_ > 0) {
      if (!resident_phase_scores_ && !continuous_decay_) {
        routes_since_decay_ += routes;
        if (routes_since_decay_ >= resident_decay_routes_) {
          for (auto& score : expert_score_) score *= 0.5f;
          routes_since_decay_ = 0;
        }
      }
      ORT_RETURN_IF_ERROR(PromoteResident(stream, expert_routes, streaming ? stream_promotions_ : 0,
                                          streaming ? stream_list.data() : nullptr,
                                          streaming ? static_cast<int>(stream_list.size()) : 0));
      if (slot_table_dirty_) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(slot_table_dev_.get(), expert_slot_.data(),
                                             SafeInt<size_t>(num_experts_) * sizeof(int), cudaMemcpyHostToDevice, stream));
        slot_table_dirty_ = false;
      }
      if (!streaming || !stream_promotion_reuse_) {
        CUDA_RETURN_IF_ERROR(drain.Synchronize());
      }
    }
    start += tile_rows;
  }

  CommitInvocationObservation(obs_phase_rows, obs_phase_expert_rows);
  CUDA_RETURN_IF_ERROR(drain.Wait());
  return Status::OK();
}

void QMoEOffload::RunAsyncCpu(AsyncTile& tile) const {
  const bool timed = cpu_timing_ && tile.tile_rows <= 4;
  const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  if (timed) {
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
    const int64_t previous = g_last_end_ns.load();
    if (previous > 0) {
      const int64_t gap = now_ns - previous;
      (gap < 5'000'000 ? g_gap_in_ns : g_gap_out_ns).fetch_add(static_cast<uint64_t>(gap));
      (gap < 5'000'000 ? g_gap_in_n : g_gap_out_n).fetch_add(1);
    }
  }
  auto& work = *async_work_;
  for (auto& assignments : work.expert_routes) assignments.clear();
  bool valid = true;
  std::string err_msg;
  std::array<int, 5> obs_phase_rows{};
  std::array<std::vector<int>, 5> obs_phase_expert_rows{};

  for (int token = 0; token < tile.tile_rows && valid; ++token) {
    const int token_phase = tile.has_phases ? static_cast<int>(tile.host_phases[token]) : 1;
    if (token_phase < 0 || token_phase >= 5) {
      valid = false;
      err_msg = "Invalid token_phase=" + std::to_string(token_phase) + " at token " + std::to_string(token);
      break;
    }
    if (tile.counts) ++tile.counts[token_phase];
    ++obs_phase_rows[token_phase];
    for (int rank = 0; rank < k_ && valid; ++rank) {
      const int route = token * static_cast<int>(k_) + rank;
      const int expert = tile.host_indices[route];
      valid = expert >= 0 && expert < num_experts_;
      if (!valid) {
        err_msg = "Out-of-range expert=" + std::to_string(expert) + " at route " + std::to_string(route);
        break;
      }
      for (int previous = 0; valid && previous < rank; ++previous) {
        valid = tile.host_indices[token * static_cast<int>(k_) + previous] != expert;
      }
      if (!valid) {
        err_msg = "Duplicate routed expert=" + std::to_string(expert) + " at token " + std::to_string(token);
        break;
      }
      const size_t statistic = static_cast<size_t>(token_phase) * num_experts_ + expert;
      if (tile.route_rows) ++tile.route_rows[statistic];
      if (tile.route_calls) tile.route_calls[statistic] = 1;
      if (resident_slots_ > 0) {
        RecordResidentRoute(token_phase, expert, expert_slot_[expert] >= 0);
        if (expert_slot_[expert] >= 0) {
          continue;
        }
      }
      work.expert_routes[expert].push_back(route);
    }
  }
  if (!valid) {
    async_error_message_ = err_msg;
    async_failed_.store(true, std::memory_order_release);
    if (tile.invocation) tile.invocation->MarkFailed(err_msg);
    return;
  }
  const int threads = cpu_threads_;
  std::chrono::steady_clock::time_point classified;
  if (timed) {
    classified = std::chrono::steady_clock::now();
    uint64_t cold_routes = 0;
    uint64_t cold_experts = 0;
    for (const auto& assignments : work.expert_routes) {
      cold_routes += assignments.size();
      cold_experts += assignments.empty() ? 0 : 1;
    }
    ++timing_calls_;
    timing_routes_ += static_cast<uint64_t>(tile.tile_rows) * k_;
    timing_cold_routes_ += cold_routes;
    timing_cold_experts_ += cold_experts;
  }
  if (tile.timing) tile.timing->Record(tile.trace_call, qmoe::CallbackEvent::ColdExpertsBegin);
  auto pool_runner = [this, threads](ptrdiff_t count, auto& function) {
    SharedSpinPool(threads).ParallelFor(count, function, pool_task_batch_);
  };
  if (is_bf16_) {
    RunColdExperts(work, group_calls_, static_cast<const BFloat16*>(tile.host_input), tile.tile_rows,
                   static_cast<BFloat16*>(tile.host_result), nullptr, false, pool_runner);
  } else {
    RunColdExperts(work, group_calls_, static_cast<const MLFloat16*>(tile.host_input), tile.tile_rows,
                   static_cast<MLFloat16*>(tile.host_result), nullptr, false, pool_runner);
  }
  if (tile.timing) tile.timing->Record(tile.trace_call, qmoe::CallbackEvent::ColdExpertsEnd);
  if (timed) {
    const auto end = std::chrono::steady_clock::now();
    timing_classify_ns_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(classified - start).count());
    timing_cpu_ns_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - classified).count());
    g_last_end_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count());
  }
  if (resident_slots_ > 0) {
    if (continuous_decay_) {
      CommitInvocationObservation(obs_phase_rows, obs_phase_expert_rows);
    } else if (!resident_phase_scores_) {
      routes_since_decay_ += tile.tile_rows * static_cast<int>(k_);
      if (routes_since_decay_ >= resident_decay_routes_) {
        for (auto& score : expert_score_) score *= 0.5f;
        routes_since_decay_ = 0;
      }
    }
  }
  if (tile.invocation) tile.invocation->MarkCpuDone();
}

Status QMoEOffload::ComputeSharedAsync(OpKernelContext* context, const Tensor* input, const Tensor* router,
                                       const Tensor* bias, const Tensor* phase, Tensor* output,
                                       int32_t* route_rows, int32_t* route_calls, int32_t* counts,
                                       int tile_rows) const {
#ifdef ORT_USE_EP_API_ADAPTERS
  ORT_RETURN_IF(async_failed_.load(std::memory_order_acquire),
                "QMoEOffload async CPU expert path failed earlier: ", async_error_message_);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (async_in_flight_.load(std::memory_order_acquire)) {
    ORT_RETURN_IF(std::chrono::steady_clock::now() > deadline, "QMoEOffload async CPU scratch was never released.");
    std::this_thread::yield();
  }
  if (callback_timing_) callback_timing_->HostCheckpoint();
  const auto trace_call = callback_timing_
                              ? callback_timing_->Begin(static_cast<uint32_t>(tile_rows), qmoe::CallbackTiming::CaptureArmed())
                              : qmoe::CallbackTiming::Call{};
  auto trace = [&](qmoe::CallbackEvent kind, uint64_t detail = 0) {
    return trace_call ? callback_timing_->Record(trace_call, kind, detail) : uint64_t{0};
  };
#if defined(__linux__)
  if (trace_call) {
    char message[128];
    std::snprintf(message, sizeof(message), "qmoe_callback_anchor:%llu:%llu:%llu",
                  static_cast<unsigned long long>(callback_timing_->Pid()),
                  static_cast<unsigned long long>(callback_timing_->Instance()),
                  static_cast<unsigned long long>(trace_call.id));
    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attributes.message.ascii = message;
    attributes.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    attributes.payload.ullValue = trace(qmoe::CallbackEvent::AnchorBefore);
    nvtxMarkEx(&attributes);
    trace(qmoe::CallbackEvent::AnchorAfter);
  }
#endif
  const auto stream = Stream(context);
  const bool resident = resident_slots_ > 0;
  const bool aux = async_aux_ && resident;
  if (resident) {
    ORT_RETURN_IF_ERROR(PromoteResident(stream, async_work_->expert_routes));
    if (slot_table_dirty_) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(slot_table_dev_.get(), expert_slot_.data(),
                                           SafeInt<size_t>(num_experts_) * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
      slot_table_dirty_ = false;
    }
  }

  const size_t elem_size = sizeof(uint16_t);
  const int routes = SafeInt<int>(tile_rows) * k_;
  const int route_slots = std::max(max_routes_, kGroupedMoeMinRows);
  const int compute_rows = nvfp4_mode_ ? routes : std::max(routes, kGroupedMoeMinRows);
  const size_t route_bytes = SafeInt<size_t>(route_slots) * sizeof(int);
  const size_t contribution_bytes = SafeInt<size_t>(max_routes_) * hidden_size_ * elem_size;
  const size_t resident_row_bytes = SafeInt<size_t>(route_slots) * hidden_size_ * elem_size;
  const size_t nvfp4_inter_bytes = nvfp4_mode_ ? static_cast<size_t>(SafeInt<size_t>(route_slots) * inter_size_ * elem_size) : size_t{0};
  const bool has_bias = (bias != nullptr) || (fc2_bias_gpu_ != nullptr);

  size_t gpu_bytes = 0;
  const size_t scale_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t index_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t unit_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t identity_offset = AddSlab(route_bytes, gpu_bytes);
  const size_t contribution_offset = AddSlab(contribution_bytes, gpu_bytes);
  const size_t bias_offset = has_bias ? AddSlab(contribution_bytes, gpu_bytes) : 0;
  const size_t hit_ids_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t hit_compact_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t hit_flag_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t permutation_offset = resident ? AddSlab(route_bytes, gpu_bytes) : 0;
  const size_t gathered_offset = (!nvfp4_mode_ && resident) ? AddSlab(resident_row_bytes, gpu_bytes) : 0;
  const size_t wave_output_offset = resident ? AddSlab(resident_row_bytes, gpu_bytes) : 0;
  const size_t nvfp4_inter_offset = (nvfp4_mode_ && resident) ? AddSlab(nvfp4_inter_bytes, gpu_bytes) : 0;
  const size_t cpu_rows_offset = resident ? AddSlab(contribution_bytes, gpu_bytes) : 0;
  const size_t workspace_offset = (!nvfp4_mode_ && resident) ? AddSlab(resident_workspace_bytes_, gpu_bytes) : 0;

  size_t host_bytes = 0;
  const size_t host_index_offset = AddSlab(SafeInt<size_t>(routes) * sizeof(int), host_bytes);
  const size_t host_phase_offset = AddSlab(tile_rows, host_bytes);
  const size_t host_input_offset = AddSlab(SafeInt<size_t>(tile_rows) * hidden_size_ * elem_size, host_bytes);
  const size_t host_result_offset = AddSlab(SafeInt<size_t>(routes) * hidden_size_ * elem_size, host_bytes);
  ORT_RETURN_IF(gpu_bytes > shared_gpu_limit_ || host_bytes > shared_pinned_limit_,
                "QMoEOffload shared scratch exceeds configured GPU or pinned limits.");

  auto scratch = GetScratchBuffer<uint8_t>(gpu_bytes, GetComputeStream(context));
  auto completed = std::make_shared<bool>(false);
  IAllocatorUniquePtr<uint8_t> pinned;
  try {
    pinned = AllocateReusableQMoEPinnedBuffer(host_bytes, shared_pinned_limit_, completed);
  } catch (const std::runtime_error&) {
    if (aux_stream_) CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(aux_stream_));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    pinned = AllocateReusableQMoEPinnedBuffer(host_bytes, shared_pinned_limit_, completed);
  }
  ORT_RETURN_IF_NOT(scratch && pinned, "QMoEOffload cannot allocate bounded shared scratch.");

  auto* base = scratch.get();
  auto* scales = reinterpret_cast<float*>(base + scale_offset);
  auto* indices = reinterpret_cast<int*>(base + index_offset);
  auto* unit_scales = reinterpret_cast<float*>(base + unit_offset);
  auto* identity = reinterpret_cast<int*>(base + identity_offset);
  auto* contributions = base + contribution_offset;
  auto* route_bias = has_bias ? (base + bias_offset) : nullptr;
  auto* host_indices = reinterpret_cast<int*>(pinned.get() + host_index_offset);
  auto* host_phases = reinterpret_cast<int8_t*>(pinned.get() + host_phase_offset);
  auto* host_input = pinned.get() + host_input_offset;
  auto* host_result = pinned.get() + host_result_offset;
  auto* hit_ids = reinterpret_cast<int*>(base + hit_ids_offset);
  auto* hit_compact = reinterpret_cast<int*>(base + hit_compact_offset);
  auto* hit_flag = reinterpret_cast<int*>(base + hit_flag_offset);
  auto* permutation = reinterpret_cast<int*>(base + permutation_offset);
  auto* gathered = base + gathered_offset;
  auto* wave_output = base + wave_output_offset;
  auto* nvfp4_inter_buf = base + nvfp4_inter_offset;
  auto* cpu_rows = base + cpu_rows_offset;
  auto* workspace = reinterpret_cast<char*>(base + workspace_offset);

  const void* bias_gpu_ptr = fc2_bias_gpu_ ? fc2_bias_gpu_.get()
                                           : (bias && bias->Location().device.Type() == OrtDevice::GPU ? bias->DataRaw() : nullptr);
  IAllocatorUniquePtr<uint8_t> transient_bias_gpu;
  if (!bias_gpu_ptr && bias && bias->Location().device.Type() == OrtDevice::CPU) {
    transient_bias_gpu = GetScratchBuffer<uint8_t>(bias->SizeInBytes(), GetComputeStream(context));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(transient_bias_gpu.get(), bias->DataRaw(), bias->SizeInBytes(),
                                         cudaMemcpyHostToDevice, stream));
    bias_gpu_ptr = transient_bias_gpu.get();
  }

  std::unique_ptr<AsyncTile> owned;
  bool failed = false;
  bool aux_used = false;
  StreamDrain drain(stream, failed, completed);
  const ck::MOEParallelismConfig parallelism;

  auto invocation_rec = std::make_shared<onnxruntime::cuda::QMoEInvocationRecord>();
  invocation_rec->layer_index = layer_index_;
  invocation_rec->node_name = Info().node().Name();

  auto enqueue_hits = [&]() -> Status {
    ck::ActivationParams runner_activation(activation_type_);
    runner_activation.alpha = activation_alpha_;
    runner_activation.beta = activation_beta_;
    runner_activation.limit = swiglu_limit_;
    runner_activation.swiglu_fusion = swiglu_fusion_;
    uint8_t* slab = resident_slab_.get();
    if (nvfp4_mode_) {
      CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadSourceRows(hit_ids, routes, static_cast<int>(k_), stream));
      const uint8_t* fc1_w = slab + resident_bank_offsets_[0];
      const uint8_t* fc2_w = slab + resident_bank_offsets_[1];
      const uint8_t* fc1_s = slab + resident_bank_offsets_[2];
      const uint8_t* fc2_s = slab + resident_bank_offsets_[3];
      const float* fc1_g = reinterpret_cast<const float*>(slab + resident_bank_offsets_[4]);
      const float* fc2_g = reinterpret_cast<const float*>(slab + resident_bank_offsets_[5]);
      const int fp4_group = mxfp4_mode_ ? 32 : 16;
      if (is_bf16_) {
        gemv::launch_moe_gemv_fp4_symmetric_interleaved_swiglu<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(input->DataRaw()), fc1_w, nullptr, fc1_s, fc1_g, nullptr,
            reinterpret_cast<__nv_bfloat16*>(nvfp4_inter_buf), nullptr, hit_compact, physical_slots_,
            routes, inter_size_, hidden_size_, fp4_group, sm_, runner_activation, gemv::MoeGemvConfig::kDefault,
            false, true, hit_ids, tile_rows, stream, true, false);
        CUDA_RETURN_IF_ERROR(cudaGetLastError());
        gemv::launch_moe_gemv_fp4_symmetric<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(nvfp4_inter_buf), fc2_w, nullptr, fc2_s, fc2_g, nullptr,
            reinterpret_cast<__nv_bfloat16*>(wave_output), nullptr, hit_compact, physical_slots_,
            routes, hidden_size_, inter_size_, fp4_group, sm_, gemv::MoeGemvConfig::kDefault,
            false, true, stream, true, false);
        CUDA_RETURN_IF_ERROR(cudaGetLastError());
      } else {
        gemv::launch_moe_gemv_fp4_symmetric_interleaved_swiglu<half>(
            reinterpret_cast<const half*>(input->DataRaw()), fc1_w, nullptr, fc1_s, fc1_g, nullptr,
            reinterpret_cast<half*>(nvfp4_inter_buf), nullptr, hit_compact, physical_slots_,
            routes, inter_size_, hidden_size_, fp4_group, sm_, runner_activation, gemv::MoeGemvConfig::kDefault,
            false, true, hit_ids, tile_rows, stream, true, false);
        CUDA_RETURN_IF_ERROR(cudaGetLastError());
        gemv::launch_moe_gemv_fp4_symmetric<half>(
            reinterpret_cast<const half*>(nvfp4_inter_buf), fc2_w, nullptr, fc2_s, fc2_g, nullptr,
            reinterpret_cast<half*>(wave_output), nullptr, hit_compact, physical_slots_,
            routes, hidden_size_, inter_size_, fp4_group, sm_, gemv::MoeGemvConfig::kDefault,
            false, true, stream, true, false);
        CUDA_RETURN_IF_ERROR(cudaGetLastError());
      }
      return Status::OK();
    }
    const auto quantization = ck::QuantParams::GroupWise(32, slab + resident_bank_offsets_[2], slab + resident_bank_offsets_[3]);
    CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadGather(reinterpret_cast<const half*>(input->DataRaw()), hit_ids,
                                                 reinterpret_cast<half*>(gathered), compute_rows, hidden_size_,
                                                 static_cast<int>(k_), stream));
    runner_->runMoe(reinterpret_cast<half*>(gathered), nullptr, hit_compact, unit_scales,
                    slab + resident_bank_offsets_[0], reinterpret_cast<const void*>(slab + resident_bank_offsets_[4]), activation_type_,
                    slab + resident_bank_offsets_[1], nullptr, quantization, compute_rows, hidden_size_, inter_size_,
                    physical_slots_, 1, workspace, reinterpret_cast<half*>(wave_output), permutation, parallelism, runner_activation, {}, stream);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return Status::OK();
  };

  auto enqueue = [&]() -> Status {
    if (is_bf16_) {
      LaunchSoftmaxTopK(reinterpret_cast<const __nv_bfloat16*>(router->DataRaw()), scales, indices, tile_rows, num_experts_,
                        static_cast<int>(k_), normalize_routing_weights_, stream);
    } else {
      LaunchSoftmaxTopK(reinterpret_cast<const half*>(router->DataRaw()), scales, indices, tile_rows, num_experts_,
                        static_cast<int>(k_), normalize_routing_weights_, stream);
    }
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    trace(qmoe::CallbackEvent::SourceD2HBegin);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_indices, indices, SafeInt<size_t>(routes) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    if (phase) {
      if (phase->Location().device.Type() == OrtDevice::GPU) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_phases, phase->Data<int8_t>(), tile_rows, cudaMemcpyDeviceToHost, stream));
      } else {
        std::memcpy(host_phases, phase->Data<int8_t>(), tile_rows);
      }
    }
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_input, input->DataRaw(),
                                         SafeInt<size_t>(tile_rows) * hidden_size_ * elem_size, cudaMemcpyDeviceToHost, stream));
    trace(qmoe::CallbackEvent::SourceD2HQueued, reinterpret_cast<uintptr_t>(stream));
    cudaStream_t callback_stream = stream;
    if (aux) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!aux_stream_) {
          CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
          CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&fork_event_, cudaEventDisableTiming));
          CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&join_event_, cudaEventDisableTiming));
        }
      }
      trace(qmoe::CallbackEvent::ForkEventRecordBegin);
      CUDA_RETURN_IF_ERROR(cudaEventRecord(fork_event_, stream));
      CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(aux_stream_, fork_event_, 0));
      trace(qmoe::CallbackEvent::ForkEventQueued, reinterpret_cast<uintptr_t>(fork_event_));
      aux_used = true;
      callback_stream = aux_stream_;
    }
    CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMetadata(unit_scales, identity, compute_rows, stream));
    if (resident) {
      CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadClassify(indices, reinterpret_cast<const int*>(slot_table_dev_.get()), routes,
                                                     compute_rows, hit_ids, hit_compact, hit_flag, stream, dummy_slot_index_));
      if (!aux) ORT_RETURN_IF_ERROR(enqueue_hits());
    }
    owned = std::make_unique<AsyncTile>();
    owned->node = this;
    owned->completed = completed;
    owned->invocation = invocation_rec;
    owned->tile_rows = tile_rows;
    owned->host_indices = host_indices;
    owned->host_phases = host_phases;
    owned->has_phases = (phase != nullptr);
    owned->host_input = host_input;
    owned->host_result = host_result;
    owned->route_rows = route_rows;
    owned->route_calls = route_calls;
    owned->counts = counts;
    owned->pinned = std::move(pinned);
    owned->timing = trace_call ? callback_timing_.get() : nullptr;
    owned->trace_call = trace_call;
    if (owned->timing) owned->timing->CallbacksQueued(trace_call, 1);
    async_in_flight_.store(true, std::memory_order_release);

    if (session_controller_) {
      cudaStream_t main_s = stream;
      cudaStream_t aux_s = aux_used ? aux_stream_ : nullptr;
      invocation_rec->drain_device = [this, main_s, aux_s]() -> common::Status {
        if (aux_s) {
          cudaError_t err = cudaStreamSynchronize(aux_s);
          if (err != cudaSuccess) {
            async_in_flight_.store(false, std::memory_order_release);
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload aux_stream sync failed: ", cudaGetErrorString(err));
          }
        }
        cudaError_t err = cudaStreamSynchronize(main_s);
        if (err != cudaSuccess) {
          async_in_flight_.store(false, std::memory_order_release);
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload compute stream sync failed: ", cudaGetErrorString(err));
        }
        return common::Status::OK();
      };
      session_controller_->RegisterInvocation(invocation_rec);
    }

    trace(qmoe::CallbackEvent::CallbackEnqueueBegin, reinterpret_cast<uintptr_t>(callback_stream));
    const auto cpu_status = cudaLaunchHostFunc(
        callback_stream,
        [](void* data) {
          auto* tile = static_cast<AsyncTile*>(data);
          const QMoEOffload* node = tile->node;
          if (tile->timing) {
            tile->timing->Record(tile->trace_call, qmoe::CallbackEvent::CallbackEnter);
            uint64_t phase_mask = 0;
            for (int row = 0; row < tile->tile_rows; ++row) {
              const int p = tile->has_phases ? static_cast<int>(tile->host_phases[row]) : 1;
              if (p >= 0 && p < 5) phase_mask |= uint64_t{1} << p;
            }
            tile->timing->Record(tile->trace_call, qmoe::CallbackEvent::InputReadyObserved, phase_mask);
          }
          try {
            node->RunAsyncCpu(*tile);
          } catch (const std::exception& ex) {
            node->async_error_message_ = ex.what();
            node->async_failed_.store(true, std::memory_order_release);
            if (tile->invocation) tile->invocation->MarkFailed(ex.what());
          } catch (...) {
            node->async_error_message_ = "Unknown exception in QMoEOffload CPU callback";
            node->async_failed_.store(true, std::memory_order_release);
            if (tile->invocation) tile->invocation->MarkFailed("Unknown exception in QMoEOffload CPU callback");
          }
          if (tile->timing) {
            tile->timing->Record(tile->trace_call, qmoe::CallbackEvent::CallbackExit);
            tile->timing->CallbackDone(tile->trace_call);
          }
          node->async_in_flight_.store(false, std::memory_order_release);
        },
        owned.get());
    trace(qmoe::CallbackEvent::CallbackEnqueueEnd);
    if (cpu_status != cudaSuccess) {
      if (owned->timing) owned->timing->CallbackDone(trace_call);
      async_in_flight_.store(false, std::memory_order_release);
    }
    CUDA_RETURN_IF_ERROR(cpu_status);
    trace(qmoe::CallbackEvent::ResultH2DBegin);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(resident ? static_cast<void*>(cpu_rows) : static_cast<void*>(contributions),
                                         host_result,
                                         SafeInt<size_t>(routes) * hidden_size_ * elem_size,
                                         cudaMemcpyHostToDevice, callback_stream));
    trace(qmoe::CallbackEvent::ResultH2DQueued);
    auto release = [](void* data) {
      std::unique_ptr<AsyncTile> tile(static_cast<AsyncTile*>(data));
      if (tile->timing) {
        tile->timing->Record(tile->trace_call, qmoe::CallbackEvent::ResultCopyReadyObserved);
        tile->timing->CallbackDone(tile->trace_call);
      }
      if (tile->invocation) tile->invocation->MarkCompleted();
      *tile->completed = true;
    };
    if (aux) {
      trace(qmoe::CallbackEvent::JoinEventRecordBegin);
      CUDA_RETURN_IF_ERROR(cudaEventRecord(join_event_, aux_stream_));
      trace(qmoe::CallbackEvent::JoinEventQueued, reinterpret_cast<uintptr_t>(join_event_));
      if (session_controller_ && !owned->timing) {
        std::shared_ptr<AsyncTile> tile_owner(std::move(owned));
        cudaStream_t aux_s = aux_stream_;
        invocation_rec->main_stream = stream;
        invocation_rec->has_main_stream = true;
        invocation_rec->drain_device = [this, aux_s, tile_owner]() mutable -> common::Status {
          cudaError_t err = cudaStreamSynchronize(aux_s);
          if (tile_owner) {
            if (tile_owner->invocation) tile_owner->invocation->MarkCompleted();
            *tile_owner->completed = true;
            tile_owner.reset();
          }
          if (err != cudaSuccess) {
            async_in_flight_.store(false, std::memory_order_release);
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload aux_stream sync failed: ", cudaGetErrorString(err));
          }
          return common::Status::OK();
        };
      } else {
        if (owned->timing) owned->timing->CallbacksQueued(trace_call, 1);
        const auto release_status = cudaLaunchHostFunc(aux_stream_, release, owned.get());
        if (release_status != cudaSuccess && owned->timing) owned->timing->CallbackDone(trace_call);
        CUDA_RETURN_IF_ERROR(release_status);
        owned.release();
      }
      ORT_RETURN_IF_ERROR(enqueue_hits());
      if (bias_gpu_ptr) {
        if (is_bf16_) {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
              reinterpret_cast<const __nv_bfloat16*>(bias_gpu_ptr), indices, identity,
              reinterpret_cast<__nv_bfloat16*>(route_bias), routes, hidden_size_, stream));
        } else {
          CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
              reinterpret_cast<const half*>(bias_gpu_ptr), indices, identity,
              reinterpret_cast<half*>(route_bias), routes, hidden_size_, stream));
        }
      }
      trace(qmoe::CallbackEvent::MergeWaitBegin);
      CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(stream, join_event_, 0));
      trace(qmoe::CallbackEvent::MergeWaitQueued);
    }
    if (resident) {
      trace(qmoe::CallbackEvent::MergeEnqueueBegin);
      if (is_bf16_) {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMerge(
            reinterpret_cast<const __nv_bfloat16*>(wave_output),
            reinterpret_cast<const __nv_bfloat16*>(cpu_rows), hit_flag,
            reinterpret_cast<__nv_bfloat16*>(contributions), routes, hidden_size_,
            tile_rows, static_cast<int>(k_), stream));
      } else {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadMerge(
            reinterpret_cast<const half*>(wave_output),
            reinterpret_cast<const half*>(cpu_rows), hit_flag,
            reinterpret_cast<half*>(contributions), routes, hidden_size_,
            tile_rows, static_cast<int>(k_), stream));
      }
      trace(qmoe::CallbackEvent::MergeEnqueueEnd);
    }
    if (bias_gpu_ptr && !aux) {
      if (is_bf16_) {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
            reinterpret_cast<const __nv_bfloat16*>(bias_gpu_ptr), indices, identity,
            reinterpret_cast<__nv_bfloat16*>(route_bias), routes, hidden_size_, stream));
      } else {
        CUDA_RETURN_IF_ERROR(LaunchQMoEOffloadScatterBias(
            reinterpret_cast<const half*>(bias_gpu_ptr), indices, identity,
            reinterpret_cast<half*>(route_bias), routes, hidden_size_, stream));
      }
    }
    if (is_bf16_) {
      ck::finalizeMoeRoutingKernelLauncher<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
          reinterpret_cast<const __nv_bfloat16*>(contributions),
          reinterpret_cast<__nv_bfloat16*>(output->MutableDataRaw()),
          bias_gpu_ptr ? reinterpret_cast<const __nv_bfloat16*>(route_bias) : nullptr,
          scales, identity, nullptr, identity, nullptr, tile_rows, hidden_size_,
          static_cast<int>(k_), routes, parallelism, false, stream);
    } else {
      ck::finalizeMoeRoutingKernelLauncher<half, half, half>(
          reinterpret_cast<const half*>(contributions),
          reinterpret_cast<half*>(output->MutableDataRaw()),
          bias_gpu_ptr ? reinterpret_cast<const half*>(route_bias) : nullptr,
          scales, identity, nullptr, identity, nullptr, tile_rows, hidden_size_,
          static_cast<int>(k_), routes, parallelism, false, stream);
    }
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    if (!aux) {
      CUDA_RETURN_IF_ERROR(cudaLaunchHostFunc(stream, release, owned.get()));
      owned.release();
    }
    trace(qmoe::CallbackEvent::CallEnqueueEnd);
    return Status::OK();
  };

  const Status status = enqueue();
  if (!status.IsOK()) {
    if (aux_used) cudaStreamSynchronize(aux_stream_);
    return status;
  }
  drain.Disarm();
  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(context);
  ORT_UNUSED_PARAMETER(input);
  ORT_UNUSED_PARAMETER(router);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(phase);
  ORT_UNUSED_PARAMETER(output);
  ORT_UNUSED_PARAMETER(route_rows);
  ORT_UNUSED_PARAMETER(route_calls);
  ORT_UNUSED_PARAMETER(counts);
  ORT_UNUSED_PARAMETER(tile_rows);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QMoEOffload async mode requires the CUDA plugin EP.");
#endif
}

}  // namespace onnxruntime::contrib::cuda
