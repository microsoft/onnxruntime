// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_packed.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <vector>

#include "contrib_ops/cuda/moe/moe_packed_cpu.h"

#if ORT_MOE_PACKED_CPU_X86_64
#include <immintrin.h>
#endif

#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"
#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "contrib_ops/cpu/quantization/gguf_vec_dot_cpu.h"
#include "contrib_ops/cuda/moe/qmoe_background_services.h"
#include "contrib_ops/cuda/moe/qmoe_spin_pool.h"
#include "core/common/parse_string.h"
#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
bool PackedTensorBytesFit(int64_t rows, int64_t columns, size_t element_bytes) {
  size_t row_bytes = 0;
  size_t bytes = 0;
  return rows >= 0 && columns >= 0 &&
         IAllocator::CalcMemSizeForArray(static_cast<size_t>(columns), element_bytes, &row_bytes) &&
         IAllocator::CalcMemSizeForArray(static_cast<size_t>(rows), row_bytes, &bytes) &&
         bytes <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
}

size_t AlignUp256(size_t v) {
  return (v + 255u) & ~size_t{255u};
}

std::mutex& NodeCpuQ8ModeMutex() {
  static std::mutex m;
  return m;
}

std::map<const void*, int>& NodeCpuQ8ModeMap() {
  static std::map<const void*, int> m;
  return m;
}

int ParseOptionalQ8Mode(const std::string& raw) {
  if (raw.empty()) return -1;
  if (raw == "0" || raw == "false" || raw == "FALSE" || raw == "False" ||
      raw == "off" || raw == "OFF" || raw == "Off") {
    return 0;
  }
  return 1;
}

void SetNodeCpuQ8Mode(const void* node, int mode) {
  std::lock_guard<std::mutex> lk(NodeCpuQ8ModeMutex());
  NodeCpuQ8ModeMap()[node] = mode;
}

int GetNodeCpuQ8Mode(const void* node) {
  std::lock_guard<std::mutex> lk(NodeCpuQ8ModeMutex());
  const auto& m = NodeCpuQ8ModeMap();
  const auto it = m.find(node);
  return (it == m.end()) ? -1 : it->second;
}

void EraseNodeCpuQ8Mode(const void* node) {
  std::lock_guard<std::mutex> lk(NodeCpuQ8ModeMutex());
  NodeCpuQ8ModeMap().erase(node);
}

// Round-to-nearest-even BF16 bits. NaN keeps its sign and is forced quiet so that truncation cannot turn it into
// infinity, and rounding cannot carry it into the sign bit.
inline uint16_t FloatToBf16Bits(float x) {
  uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
    return static_cast<uint16_t>((bits >> 16) | 0x0040u);
  }
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

inline float RoundBf16Scalar(float x) {
  const uint32_t bits = static_cast<uint32_t>(FloatToBf16Bits(x)) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// Rounds to the activation type with round-to-nearest-even at the points where the CUDA kernels store T: gate/up
// dot products, the SwiGLU output, and each route's FC2 output. Both BF16 and FP16 overflow to infinity as on the GPU.
inline float RoundToActivationType(float x, bool is_bf16) {
  return is_bf16 ? RoundBf16Scalar(x) : MLFloat16(x).ToFloat();
}

#if ORT_MOE_PACKED_CPU_X86_64
bool UseAvx2CpuKernels() {
  static const bool use_avx2 = qmoe_shared::CpuHasAvx2FmaF16c();
  return use_avx2;
}
#endif

// Dot product of one GGUF quantized row against R activation vectors x_rows[0..R-1].
void GgufDotRowCpu(
    const uint8_t* down_row,
    int down_type,
    int K,
    const float* const* inter_rows,
    int R,
    float* down_dots) {
#if ORT_MOE_PACKED_CPU_X86_64
  if (UseAvx2CpuKernels() && moe_packed_cpu::GgufDotRowAvx2(down_row, down_type, K, inter_rows, R, down_dots)) {
    return;
  }
#endif
  // FC2 passes K = inter_size, which kMoEPackedMaxHiddenSize does not bound.
  alignas(32) float stack_buf[kMoEPackedMaxHiddenSize];
  std::unique_ptr<float[]> heap_buf;
  float* buf_d = stack_buf;
  if (K > kMoEPackedMaxHiddenSize) {
    heap_buf = std::make_unique<float[]>(static_cast<size_t>(K));
    buf_d = heap_buf.get();
  }
  gguf_quant::DequantizeGgufRowCpu(down_row, down_type, K, buf_d);
  for (int r = 0; r < R; ++r) {
    float sd = 0.0f;
    const float* xr = inter_rows[r];
    for (int k = 0; k < K; ++k) {
      sd += buf_d[k] * xr[k];
    }
    down_dots[r] = sd;
  }
}

// Dot products of one (gate_row, up_row) pair against R token activation vectors x_rows[0..R-1].
void GgufDualDotRowCpu(
    const uint8_t* gate_row,
    const uint8_t* up_row,
    int gate_type,
    int up_type,
    int K,
    const float* const* x_rows,
    int R,
    float* gate_dots,
    float* up_dots) {
#if ORT_MOE_PACKED_CPU_X86_64
  if (gate_type == up_type && UseAvx2CpuKernels() &&
      moe_packed_cpu::GgufDualDotRowAvx2(gate_row, up_row, gate_type, K, x_rows, R, gate_dots, up_dots)) {
    return;
  }
#endif
  GgufDotRowCpu(gate_row, gate_type, K, x_rows, R, gate_dots);
  GgufDotRowCpu(up_row, up_type, K, x_rows, R, up_dots);
}

// Dot products of two adjacent (gate_row, up_row) channel pairs against R token activation vectors.
void GgufDualDot2ChCpu(
    const uint8_t* gate_row0,
    const uint8_t* up_row0,
    const uint8_t* gate_row1,
    const uint8_t* up_row1,
    int gate_type,
    int up_type,
    int K,
    const float* const* x_rows,
    int R,
    float* gate_dots0,
    float* up_dots0,
    float* gate_dots1,
    float* up_dots1) {
#if ORT_MOE_PACKED_CPU_X86_64
  if (gate_type == up_type && UseAvx2CpuKernels() &&
      moe_packed_cpu::GgufDualDot2ChAvx2(gate_row0, up_row0, gate_row1, up_row1, gate_type, K, x_rows, R,
                                         gate_dots0, up_dots0, gate_dots1, up_dots1)) {
    return;
  }
#endif
  GgufDualDotRowCpu(gate_row0, up_row0, gate_type, up_type, K, x_rows, R, gate_dots0, up_dots0);
  GgufDualDotRowCpu(gate_row1, up_row1, gate_type, up_type, K, x_rows, R, gate_dots1, up_dots1);
}

// Lock-free CCD/physical-core-pinned parallel pool backed by qmoe::SpinPoolRegistry.
class MoEPackedParallelPool {
 public:
  static MoEPackedParallelPool& Instance() {
    static MoEPackedParallelPool pool;
    return pool;
  }

  // Resolve once per node, like QMoE, rather than per ParallelFor: std::thread::hardware_concurrency() reads sysfs on
  // Linux, and the decode path that the GPU waits for runs two ParallelFor calls per layer.
  static int ResolveThreads(int32_t policy_threads = 0) {
    if (policy_threads > 0) {
      return static_cast<int>(policy_threads);
    }
    const unsigned int hw = std::thread::hardware_concurrency();
    const int def_t = static_cast<int>(std::clamp<unsigned int>(hw > 0 ? (hw * 3) / 4 : 16, 4u, 40u));
    const std::string env_s = GetEnvironmentVar("ORT_QMOE_SHARED_CPU_THREADS");
    int t = def_t;
    if (!env_s.empty()) {
      static_cast<void>(TryParseStringWithClassicLocale(env_s, t));
    }
    return std::clamp(t, 1, 256);
  }

  int CpuForWorker(int threads, int worker_idx) {
    return Pool(threads).CpuForWorker(worker_idx);
  }

  template <typename Fn>
  void ParallelFor(int threads, int num_tasks, Fn&& fn) {
    if (num_tasks <= 0) return;
    if (num_tasks == 1 || threads <= 1) {
      for (int i = 0; i < num_tasks; ++i) {
        fn(i);
      }
      return;
    }
    auto task_fn = [&](ptrdiff_t idx) { fn(static_cast<int>(idx)); };
    Pool(threads).ParallelFor(static_cast<ptrdiff_t>(num_tasks), task_fn, 1);
  }

  void Shutdown() noexcept {
    registry_.Shutdown();
  }

 private:
  MoEPackedParallelPool() = default;
  ~MoEPackedParallelPool() = default;

  qmoe::SpinPool& Pool(int threads) {
    return registry_.GetOrCreate(threads, [](int n) {
      return std::make_unique<qmoe::SpinPool>(n, 2000);
    });
  }

  qmoe::SpinPoolRegistry registry_;
};

}  // namespace

// Prefill double-buffered wave staging shared by the MoEPacked layers on one CUDA device. All members are guarded by
// `mu`. Device memory, streams and events belong to `device_id`.
struct MoEPackedSharedPrefillWaveStaging {
  static constexpr int kWaveExperts = 32;
  std::mutex mu;
  int device_id{-1};
  size_t bank_bytes{0};
  uint8_t* pinned_bank[2]{nullptr, nullptr};
  uint8_t* gpu_bank[2]{nullptr, nullptr};
  cudaStream_t copy_stream{nullptr};
  cudaEvent_t h2d_done[2]{nullptr, nullptr};
  cudaEvent_t compute_done[2]{nullptr, nullptr};
  bool bank_in_flight[2]{false, false};
  // Recorded on the caller stream after the last asynchronous read of the shared pinned route buffers.
  cudaEvent_t last_use_done{nullptr};
  bool last_use_pending{false};
  int32_t* pinned_topk_idx{nullptr};
  float* pinned_topk_scales{nullptr};
  int32_t* pinned_res_sorted_routes{nullptr};
  MoEPackedTile* pinned_res_tiles{nullptr};
  int32_t* pinned_res_num_tiles{nullptr};
  int32_t* pinned_sorted_routes[2]{nullptr, nullptr};
  MoEPackedTile* pinned_tiles[2]{nullptr, nullptr};
  int32_t* pinned_num_tiles[2]{nullptr, nullptr};
  int32_t* d_wave_sorted_routes[2]{nullptr, nullptr};
  MoEPackedTile* d_wave_tiles[2]{nullptr, nullptr};
  int32_t* d_wave_num_tiles[2]{nullptr, nullptr};
  size_t pinned_route_cap{0};
  size_t device_route_cap{0};
  size_t bank_accounted_pinned_bytes{0};
  std::shared_ptr<CudaDeviceMemoryBudget> bank_session_pinned_budget;
  size_t route_accounted_pinned_bytes{0};
  std::shared_ptr<CudaDeviceMemoryBudget> route_session_pinned_budget;

  ~MoEPackedSharedPrefillWaveStaging() = default;

  static size_t MaxTilesForRoutes(size_t max_routes) {
    return max_routes / 8 + 512;
  }

  // Pinned bytes of the routing buffers sized for max_routes, excluding the expert banks.
  static size_t RoutePinnedBytes(size_t max_routes) {
    const size_t tile_bytes = MaxTilesForRoutes(max_routes) * sizeof(MoEPackedTile);
    return max_routes * (sizeof(int32_t) + sizeof(float)) + 3 * (max_routes * sizeof(int32_t) + tile_bytes + sizeof(int32_t));
  }

  // Device bytes of the double-buffered per-wave routing metadata buffers sized for max_routes.
  static size_t WaveRouteDeviceBytes(size_t max_routes) {
    if (max_routes == 0) return 0;
    const size_t tile_bytes = MaxTilesForRoutes(max_routes) * sizeof(MoEPackedTile);
    return 2 * (max_routes * sizeof(int32_t) + tile_bytes + sizeof(int32_t));
  }

  size_t AccountedBytesFor(const std::shared_ptr<CudaDeviceMemoryBudget>& sess_pinned) const noexcept {
    if (!sess_pinned) return 0;
    size_t total = 0;
    if (bank_session_pinned_budget == sess_pinned) total += bank_accounted_pinned_bytes;
    if (route_session_pinned_budget == sess_pinned) total += route_accounted_pinned_bytes;
    return total;
  }

  template <typename P>
  static bool FreeHostChecked(P*& ptr) noexcept {
    bool ok = true;
    if (ptr != nullptr) {
      ok = (cudaFreeHost(ptr) == cudaSuccess);
      ptr = nullptr;
    }
    return ok;
  }

  void FreeBanks() {
    bool all_ok = true;
    for (int b = 0; b < 2; ++b) {
      if (!FreeHostChecked(pinned_bank[b])) all_ok = false;
      if (gpu_bank[b]) {
        cudaFree(gpu_bank[b]);
        gpu_bank[b] = nullptr;
      }
    }
    if (bank_accounted_pinned_bytes > 0) {
      auto host_ledger = GetCudaHostPinnedMemoryBudget();
      if (bank_session_pinned_budget) bank_session_pinned_budget->BeginFree(bank_accounted_pinned_bytes);
      if (host_ledger) host_ledger->BeginFreeAllocated(bank_accounted_pinned_bytes);
      if (host_ledger) host_ledger->FinishFreeAllocated(bank_accounted_pinned_bytes, all_ok);
      if (bank_session_pinned_budget) bank_session_pinned_budget->FinishFree(bank_accounted_pinned_bytes, all_ok);
      bank_accounted_pinned_bytes = 0;
    }
    bank_session_pinned_budget.reset();
    bank_bytes = 0;
  }

  void FreeWaveDeviceRouteBuffers() {
    for (int b = 0; b < 2; ++b) {
      if (d_wave_sorted_routes[b]) {
        cudaFree(d_wave_sorted_routes[b]);
        d_wave_sorted_routes[b] = nullptr;
      }
      if (d_wave_tiles[b]) {
        cudaFree(d_wave_tiles[b]);
        d_wave_tiles[b] = nullptr;
      }
      if (d_wave_num_tiles[b]) {
        cudaFree(d_wave_num_tiles[b]);
        d_wave_num_tiles[b] = nullptr;
      }
    }
    device_route_cap = 0;
  }

  void FreePinnedRouteBuffers() {
    bool all_ok = true;
    if (!FreeHostChecked(pinned_topk_idx)) all_ok = false;
    if (!FreeHostChecked(pinned_topk_scales)) all_ok = false;
    if (!FreeHostChecked(pinned_res_sorted_routes)) all_ok = false;
    if (!FreeHostChecked(pinned_res_tiles)) all_ok = false;
    if (!FreeHostChecked(pinned_res_num_tiles)) all_ok = false;
    for (int b = 0; b < 2; ++b) {
      if (!FreeHostChecked(pinned_sorted_routes[b])) all_ok = false;
      if (!FreeHostChecked(pinned_tiles[b])) all_ok = false;
      if (!FreeHostChecked(pinned_num_tiles[b])) all_ok = false;
    }
    if (route_accounted_pinned_bytes > 0) {
      auto host_ledger = GetCudaHostPinnedMemoryBudget();
      if (route_session_pinned_budget) route_session_pinned_budget->BeginFree(route_accounted_pinned_bytes);
      if (host_ledger) host_ledger->BeginFreeAllocated(route_accounted_pinned_bytes);
      if (host_ledger) host_ledger->FinishFreeAllocated(route_accounted_pinned_bytes, all_ok);
      if (route_session_pinned_budget) route_session_pinned_budget->FinishFree(route_accounted_pinned_bytes, all_ok);
      route_accounted_pinned_bytes = 0;
    }
    route_session_pinned_budget.reset();
    pinned_route_cap = 0;
  }

  void FreeRouteBuffers() {
    FreePinnedRouteBuffers();
    FreeWaveDeviceRouteBuffers();
  }

  // Waits until the previous user, possibly on another stream, no longer reads the shared pinned buffers.
  Status WaitForPreviousUse() {
    if (last_use_pending) {
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventSynchronize(last_use_done)));
      last_use_pending = false;
    }
    return Status::OK();
  }

  Status WaitForBanks() {
    for (int b = 0; b < 2; ++b) {
      if (bank_in_flight[b]) {
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventSynchronize(h2d_done[b])));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventSynchronize(compute_done[b])));
        bank_in_flight[b] = false;
      }
    }
    return Status::OK();
  }

  Status RecordUse(cudaStream_t stream) {
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventRecord(last_use_done, stream)));
    last_use_pending = true;
    return Status::OK();
  }

  // Requires the owning device to be current.
  void ReleaseAll() {
    std::lock_guard<std::mutex> lk(mu);
    static_cast<void>(WaitForPreviousUse());
    static_cast<void>(WaitForBanks());
    FreeBanks();
    FreeRouteBuffers();
    for (int b = 0; b < 2; ++b) {
      if (h2d_done[b]) {
        cudaEventDestroy(h2d_done[b]);
        h2d_done[b] = nullptr;
      }
      if (compute_done[b]) {
        cudaEventDestroy(compute_done[b]);
        compute_done[b] = nullptr;
      }
      bank_in_flight[b] = false;
    }
    if (last_use_done) {
      cudaEventDestroy(last_use_done);
      last_use_done = nullptr;
    }
    last_use_pending = false;
    if (copy_stream) {
      cudaStreamDestroy(copy_stream);
      copy_stream = nullptr;
    }
  }

  Status EnsureStreamAndEvents() {
    if (!copy_stream) {
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking)));
    }
    for (int b = 0; b < 2; ++b) {
      if (!h2d_done[b]) {
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventCreateWithFlags(&h2d_done[b], cudaEventDisableTiming)));
      }
      if (!compute_done[b]) {
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventCreateWithFlags(&compute_done[b], cudaEventDisableTiming)));
      }
    }
    if (!last_use_done) {
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventCreateWithFlags(&last_use_done, cudaEventDisableTiming)));
    }
    return Status::OK();
  }

  Status EnsureRouteCapacity(
      size_t max_routes,
      bool needs_streaming = true,
      std::shared_ptr<CudaDeviceMemoryBudget> sess_pinned = nullptr) {
    ORT_RETURN_IF_ERROR(EnsureStreamAndEvents());
    const bool need_pinned =
        (max_routes == 0 && pinned_route_cap > 0) ||
        (max_routes > pinned_route_cap) ||
        (max_routes > 0 && route_session_pinned_budget != sess_pinned);
    const bool need_device =
        needs_streaming ? (max_routes > device_route_cap) : (device_route_cap > 0);
    if (!need_pinned && !need_device) {
      return Status::OK();
    }
    ORT_RETURN_IF_ERROR(WaitForPreviousUse());
    ORT_RETURN_IF_ERROR(WaitForBanks());
    const size_t tile_bytes = MaxTilesForRoutes(max_routes) * sizeof(MoEPackedTile);
    if (need_pinned) {
      FreePinnedRouteBuffers();
      if (max_routes > 0) {
        const size_t route_pinned_bytes = RoutePinnedBytes(max_routes);
        auto host_ledger = GetCudaHostPinnedMemoryBudget();
        CudaDeviceMemoryBudget::Reservation sess_res;
        CudaDeviceMemoryBudget::Reservation host_res;
        try {
          if (sess_pinned) sess_res = sess_pinned->Reserve(route_pinned_bytes);
          if (host_ledger) host_res = host_ledger->ReserveAllocated(route_pinned_bytes);
        } catch (const std::exception& ex) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "MoEPacked prefill route pinned budget reservation failed (",
                                 route_pinned_bytes, " bytes) against session.moe.offload.pinned_bytes: ", ex.what());
        }
        auto cleanup_partial = gsl::finally([this]() {
          if (pinned_route_cap == 0) {
            FreePinnedRouteBuffers();
          }
        });
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_topk_idx), max_routes * sizeof(int32_t), cudaHostAllocDefault)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_topk_scales), max_routes * sizeof(float), cudaHostAllocDefault)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_res_sorted_routes), max_routes * sizeof(int32_t), cudaHostAllocDefault)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_res_tiles), tile_bytes, cudaHostAllocDefault)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_res_num_tiles), sizeof(int32_t), cudaHostAllocDefault)));
        for (int b = 0; b < 2; ++b) {
          ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_sorted_routes[b]), max_routes * sizeof(int32_t), cudaHostAllocDefault)));
          ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_tiles[b]), tile_bytes, cudaHostAllocDefault)));
          ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_num_tiles[b]), sizeof(int32_t), cudaHostAllocDefault)));
        }
        if (sess_res.Active()) sess_res.Commit();
        if (host_res.Active()) host_res.Commit();
        route_accounted_pinned_bytes = route_pinned_bytes;
        route_session_pinned_budget = std::move(sess_pinned);
        pinned_route_cap = max_routes;
      }
    }
    if (need_device) {
      FreeWaveDeviceRouteBuffers();
      if (needs_streaming && max_routes > 0) {
        auto cleanup_dev = gsl::finally([this]() {
          if (device_route_cap == 0) {
            FreeWaveDeviceRouteBuffers();
          }
        });
        for (int b = 0; b < 2; ++b) {
          ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_wave_sorted_routes[b]), max_routes * sizeof(int32_t))));
          ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_wave_tiles[b]), tile_bytes)));
          ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_wave_num_tiles[b]), sizeof(int32_t))));
        }
        device_route_cap = max_routes;
      }
    }
    return Status::OK();
  }

  Status EnsureBankCapacity(
      size_t required_bank_bytes,
      std::shared_ptr<CudaDeviceMemoryBudget> sess_pinned = nullptr) {
    if (required_bank_bytes == 0) {
      if (bank_bytes > 0) {
        ORT_RETURN_IF_ERROR(WaitForBanks());
        FreeBanks();
      }
      return Status::OK();
    }
    if (required_bank_bytes <= bank_bytes && bank_session_pinned_budget == sess_pinned) {
      return Status::OK();
    }
    ORT_RETURN_IF_ERROR(WaitForBanks());
    FreeBanks();
    const size_t total_pinned = 2 * required_bank_bytes;
    auto host_ledger = GetCudaHostPinnedMemoryBudget();
    CudaDeviceMemoryBudget::Reservation sess_res;
    CudaDeviceMemoryBudget::Reservation host_res;
    try {
      if (sess_pinned) sess_res = sess_pinned->Reserve(total_pinned);
      if (host_ledger) host_res = host_ledger->ReserveAllocated(total_pinned);
    } catch (const std::exception& ex) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "MoEPacked prefill bank pinned budget reservation failed (",
                             total_pinned, " bytes) against session.moe.offload.pinned_bytes: ", ex.what());
    }
    auto cleanup_partial = gsl::finally([this]() {
      if (bank_bytes == 0) {
        FreeBanks();
      }
    });
    for (int b = 0; b < 2; ++b) {
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_bank[b]), required_bank_bytes, cudaHostAllocDefault)));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&gpu_bank[b]), required_bank_bytes)));
    }
    if (sess_res.Active()) sess_res.Commit();
    if (host_res.Active()) host_res.Commit();
    bank_accounted_pinned_bytes = total_pinned;
    bank_session_pinned_budget = std::move(sess_pinned);
    bank_bytes = required_bank_bytes;
    return Status::OK();
  }

  static MoEPackedSharedPrefillWaveStaging& ForDevice(int device_id) {
    std::lock_guard<std::mutex> lk(RegistryMutex());
    auto& entry = Registry()[device_id];
    if (!entry) {
      entry = std::make_unique<MoEPackedSharedPrefillWaveStaging>();
      entry->device_id = device_id;
    }
    return *entry;
  }

  // Frees every device's staging under its owning device. Entries stay registered so references remain valid.
  static void ReleaseAllDevices() {
    std::lock_guard<std::mutex> lk(RegistryMutex());
    int previous_device = -1;
    const bool has_previous = cudaGetDevice(&previous_device) == cudaSuccess;
    for (auto& [device, entry] : Registry()) {
      if (!entry || cudaSetDevice(device) != cudaSuccess) continue;
      entry->ReleaseAll();
    }
    if (has_previous) {
      static_cast<void>(cudaSetDevice(previous_device));
    }
  }

 private:
  static std::mutex& RegistryMutex() {
    static std::mutex m;
    return m;
  }

  static std::map<int, std::unique_ptr<MoEPackedSharedPrefillWaveStaging>>& Registry() {
    static std::map<int, std::unique_ptr<MoEPackedSharedPrefillWaveStaging>> registry;
    return registry;
  }
};

struct MoEPackedOffloadState {
  int32_t layer_index{-1};
  int32_t num_experts{0};
  int32_t hidden_size{0};
  int32_t inter_size{0};
  int32_t k{0};
  bool is_bf16{true};
  bool normalize_routing_weights{true};
  int gate_type{0};
  int up_type{0};
  int down_type{0};
  size_t gate_row_bytes{0};
  size_t up_row_bytes{0};
  size_t down_row_bytes{0};
  size_t gate_exp_bytes{0};
  size_t up_exp_bytes{0};
  size_t down_exp_bytes{0};
  size_t slot_bytes{0};

  const uint8_t* host_gate_w{nullptr};
  const uint8_t* host_up_w{nullptr};
  const uint8_t* host_down_w{nullptr};

  int32_t resident_slots{0};
  int32_t host_cache_slots{0};
  bool is_adaptive{false};
  bool cpu_execution_enabled{true};
  int cpu_q8_dot_mode{-1};
  int cpu_threads{1};
  std::shared_ptr<onnxruntime::cuda::QMoESessionController> session_controller;
  std::shared_ptr<onnxruntime::cuda::QMoEInvocationRecord> invocation;
  std::vector<int32_t> host_expert_to_slot;
  std::vector<int32_t> slot_to_expert;
  onnxruntime::moe_offload::MoeExpertSourceDescriptor source_desc;
  std::vector<int32_t> evictable_host_experts;
  std::vector<uint8_t> is_evictable_host_expert;
  std::vector<int32_t> decode_touched_evictable_experts;
  std::vector<float> expert_ema;
  std::vector<float> x_f32_buf;
  std::vector<float> inter_buf;
  std::vector<gguf_quant::GgufQ8Block> x_q8_buf;
  std::vector<gguf_quant::GgufQ8Block> inter_q8_buf;
  IAllocatorUniquePtr<uint8_t> gpu_resident_buf;
  uint8_t* gpu_gate_bank{nullptr};
  uint8_t* gpu_up_bank{nullptr};
  uint8_t* gpu_down_bank{nullptr};
  IAllocatorUniquePtr<int32_t> d_expert_to_slot;
  IAllocatorUniquePtr<uint8_t> d_decode_ws;

  MoEPackedDecodePinned* h_decode_pinned{nullptr};
  MoEPackedDecodePinned* d_decode_pinned{nullptr};
  int32_t last_seen_seq{0};
  size_t decode_pinned_accounted_bytes{0};
  std::shared_ptr<CudaDeviceMemoryBudget> decode_session_pinned_budget;

  float tail_sub_threshold{0.0f};
  int32_t tail_sub_min_rank{6};
  float min_expert_weight{0.0f};
  bool skip_tail_cold_only{true};

  // Telemetry counters for ORT_QMOE_SHARED_CPU_TIMING / ORT_QMOE_SHARED_RESIDENT_LOG
  std::atomic<uint64_t> decode_calls{0};
  std::atomic<uint64_t> decode_routes{0};
  std::atomic<uint64_t> decode_cold_routes{0};
  std::atomic<uint64_t> decode_cold_experts{0};
  std::atomic<uint64_t> decode_cpu_us{0};
  std::atomic<uint64_t> decode_prep_us{0};
  std::atomic<uint64_t> decode_fc1_us{0};
  std::atomic<uint64_t> decode_fc2_us{0};
  std::atomic<uint64_t> prefill_routes{0};
  std::atomic<uint64_t> prefill_hit_routes{0};
  std::atomic<uint64_t> total_promotions{0};
  std::atomic<uint64_t> stream_tiles{0};
  std::atomic<uint64_t> stream_experts{0};
  std::atomic<uint64_t> stream_routes{0};
  std::atomic<uint64_t> stream_waves{0};
  std::atomic<bool> prefill_touched_evictable{false};

  bool InWarmup() const noexcept {
    return session_controller != nullptr && session_controller->RunControls().InWarmup();
  }

  void RegisterInvocation(cudaStream_t stream) {
    if (session_controller) {
      invocation->main_stream = stream;
      invocation->has_main_stream = true;
      session_controller->RegisterInvocation(invocation);
    }
  }

  void FreeDecodePinned() noexcept {
    bool ok = true;
    if (h_decode_pinned) {
      ok = (cudaFreeHost(h_decode_pinned) == cudaSuccess);
      h_decode_pinned = nullptr;
      d_decode_pinned = nullptr;
    }
    if (decode_pinned_accounted_bytes > 0) {
      auto host_ledger = GetCudaHostPinnedMemoryBudget();
      if (decode_session_pinned_budget) decode_session_pinned_budget->BeginFree(decode_pinned_accounted_bytes);
      if (host_ledger) host_ledger->BeginFreeAllocated(decode_pinned_accounted_bytes);
      if (host_ledger) host_ledger->FinishFreeAllocated(decode_pinned_accounted_bytes, ok);
      if (decode_session_pinned_budget) decode_session_pinned_budget->FinishFree(decode_pinned_accounted_bytes, ok);
      decode_pinned_accounted_bytes = 0;
    }
    decode_session_pinned_budget.reset();
  }

  Status TryAllocateDecodePinned(uint64_t policy_pinned_limit) {
    if (h_decode_pinned != nullptr) return Status::OK();
    constexpr size_t kDecodePinnedBytes = sizeof(MoEPackedDecodePinned);
    if (kDecodePinnedBytes > policy_pinned_limit) {
      return Status::OK();
    }
    auto sess_pinned = session_controller ? session_controller->PinnedBudget() : nullptr;
    auto host_ledger = GetCudaHostPinnedMemoryBudget();
    CudaDeviceMemoryBudget::Reservation sess_res;
    CudaDeviceMemoryBudget::Reservation host_res;
    try {
      if (sess_pinned) sess_res = sess_pinned->Reserve(kDecodePinnedBytes);
      if (host_ledger) host_res = host_ledger->ReserveAllocated(kDecodePinnedBytes);
    } catch (const std::exception&) {
      return Status::OK();
    }
    MoEPackedDecodePinned* h_ptr = nullptr;
    cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&h_ptr), kDecodePinnedBytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
      return CUDA_CALL(err);
    }
    new (h_ptr) MoEPackedDecodePinned{};
    MoEPackedDecodePinned* d_ptr = nullptr;
    err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_ptr), h_ptr, 0);
    if (err != cudaSuccess) {
      static_cast<void>(cudaFreeHost(h_ptr));
      return CUDA_CALL(err);
    }
    if (sess_res.Active()) sess_res.Commit();
    if (host_res.Active()) host_res.Commit();
    h_decode_pinned = h_ptr;
    d_decode_pinned = d_ptr;
    last_seen_seq = 0;
    decode_pinned_accounted_bytes = kDecodePinnedBytes;
    decode_session_pinned_budget = std::move(sess_pinned);
    return Status::OK();
  }

  ~MoEPackedOffloadState();
  void RunColdDecodeStep();
};

namespace {

class MoEPackedDoorbellService {
 public:
  static MoEPackedDoorbellService& Instance() {
    static MoEPackedDoorbellService svc;
    return svc;
  }

  // Register and Unregister hold lifecycle_mu_ for the whole transition, so a new worker is never started before a
  // retiring one has been joined and the shared staging and CPU pool are never torn down while any offload state
  // (including prefill-only states that do not run the doorbell worker) remains alive. The worker only takes mu_.
  void Register(MoEPackedOffloadState* state, bool enable_doorbell) {
    if (!state) return;
    std::lock_guard<std::mutex> lifecycle_lk(lifecycle_mu_);
    std::lock_guard<std::mutex> lk(mu_);
    if (std::find(live_states_.begin(), live_states_.end(), state) == live_states_.end()) {
      live_states_.push_back(state);
    }
    if (enable_doorbell) {
      if (std::find(states_.begin(), states_.end(), state) == states_.end()) {
        states_.push_back(state);
      }
      if (!worker_.joinable()) {
        stop_.store(false, std::memory_order_release);
        worker_ = std::thread([this]() { Loop(); });
      }
    }
  }

  void Unregister(MoEPackedOffloadState* state) {
    std::lock_guard<std::mutex> lifecycle_lk(lifecycle_mu_);
    std::thread to_join;
    bool release_shared = false;
    {
      std::lock_guard<std::mutex> lk(mu_);
      states_.erase(std::remove(states_.begin(), states_.end(), state), states_.end());
      const size_t prev_live = live_states_.size();
      live_states_.erase(std::remove(live_states_.begin(), live_states_.end(), state), live_states_.end());
      if (states_.empty() && worker_.joinable()) {
        stop_.store(true, std::memory_order_release);
        to_join = std::move(worker_);
      }
      release_shared = (live_states_.size() < prev_live && live_states_.empty());
    }
    if (to_join.joinable()) {
      to_join.join();
    }
    if (release_shared) {
      MoEPackedSharedPrefillWaveStaging::ReleaseAllDevices();
      MoEPackedParallelPool::Instance().Shutdown();
    }
  }

  void ShutdownIfIdle() noexcept {
    try {
      std::lock_guard<std::mutex> lifecycle_lk(lifecycle_mu_);
      std::thread to_join;
      {
        std::lock_guard<std::mutex> lk(mu_);
        if (!live_states_.empty()) {
          return;
        }
        if (worker_.joinable()) {
          stop_.store(true, std::memory_order_release);
          to_join = std::move(worker_);
        }
      }
      if (to_join.joinable()) {
        to_join.join();
      }
      MoEPackedSharedPrefillWaveStaging::ReleaseAllDevices();
      MoEPackedParallelPool::Instance().Shutdown();
    } catch (...) {
    }
  }

  Status EnsureDecodePinned(MoEPackedOffloadState* state, uint64_t policy_pinned_limit) {
    if (!state) return Status::OK();
    std::lock_guard<std::mutex> lk(mu_);
    if (state->h_decode_pinned != nullptr) return Status::OK();
    if (state->d_decode_ws) {
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemset(state->d_decode_ws.get(), 0, sizeof(int))));
    }
    return state->TryAllocateDecodePinned(policy_pinned_limit);
  }

  void ReclaimSessionDecodePinned(
      const onnxruntime::cuda::QMoESessionController* controller,
      cudaStream_t stream) {
    if (!controller) return;
    if (stream) {
      static_cast<void>(cudaStreamSynchronize(stream));
    }
    std::lock_guard<std::mutex> lk(mu_);
    bool any_freed = false;
    for (MoEPackedOffloadState* st : states_) {
      if (st && st->session_controller.get() == controller && st->h_decode_pinned != nullptr) {
        if (st->d_decode_ws) {
          static_cast<void>(cudaMemset(st->d_decode_ws.get(), 0, sizeof(int)));
        }
        st->FreeDecodePinned();
        any_freed = true;
      }
    }
    if (any_freed) {
      const_cast<onnxruntime::cuda::QMoESessionController*>(controller)->BumpGraphGeneration();
    }
  }

 private:
  MoEPackedDoorbellService() = default;
  ~MoEPackedDoorbellService() {
    stop_.store(true, std::memory_order_release);
    if (worker_.joinable()) worker_.detach();
  }

  void Loop() {
    qmoe::NameCurrentThread("moepacked-db");
#if defined(_WIN32)
    int init_threads = 0;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!states_.empty() && states_.front() != nullptr) {
        init_threads = states_.front()->cpu_threads;
      }
    }
    const int cpu0 = MoEPackedParallelPool::Instance().CpuForWorker(
        MoEPackedParallelPool::ResolveThreads(init_threads), 0);
    if (cpu0 >= 0 && cpu0 < static_cast<int>(sizeof(DWORD_PTR) * 8)) {
      static_cast<void>(SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu0));
    }
#endif
    auto last_active = std::chrono::steady_clock::now();
    for (;;) {
      if (stop_.load(std::memory_order_acquire)) return;
      bool did_work = false;
      {
        std::lock_guard<std::mutex> lk(mu_);
        for (MoEPackedOffloadState* st : states_) {
          if (!st || !st->h_decode_pinned) continue;
          // The sequence only advances, so inequality is a wrap-safe test for a new doorbell.
          const int32_t seq = st->h_decode_pinned->doorbell_seq;
          if (seq != st->last_seen_seq) {
            std::atomic_thread_fence(std::memory_order_acquire);
            st->last_seen_seq = seq;
            st->RunColdDecodeStep();
            std::atomic_thread_fence(std::memory_order_release);
            st->h_decode_pinned->doorbell_done = seq;
            if (st->prefill_touched_evictable.exchange(false, std::memory_order_relaxed) &&
                !st->decode_touched_evictable_experts.empty()) {
              onnxruntime::cuda::QMoESessionController::EvictUploadedHostPages(
                  st->source_desc, st->decode_touched_evictable_experts);
              st->decode_touched_evictable_experts.clear();
            }
            did_work = true;
          }
        }
      }
      if (did_work) {
        last_active = std::chrono::steady_clock::now();
      } else {
        if (std::chrono::steady_clock::now() - last_active > std::chrono::milliseconds(50)) {
          std::this_thread::sleep_for(std::chrono::microseconds(50));
        } else {
#if ORT_MOE_PACKED_CPU_X86_64
          _mm_pause();
#endif
        }
      }
    }
  }

  std::mutex lifecycle_mu_;
  std::mutex mu_;
  std::vector<MoEPackedOffloadState*> live_states_;
  std::vector<MoEPackedOffloadState*> states_;
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

}  // namespace

void ShutdownMoEPackedBackgroundServices() noexcept {
  MoEPackedDoorbellService::Instance().ShutdownIfIdle();
}

MoEPackedOffloadState::~MoEPackedOffloadState() {
  MoEPackedDoorbellService::Instance().Unregister(this);
  const std::string res_env = GetEnvironmentVar("ORT_QMOE_SHARED_RESIDENT_LOG");
  const std::string cpu_env = GetEnvironmentVar("ORT_QMOE_SHARED_CPU_TIMING");
  const bool log_res = !res_env.empty() && res_env != "0";
  const bool log_cpu = !cpu_env.empty() && cpu_env != "0";
  const uint64_t d_calls = decode_calls.load();
  const uint64_t d_routes = decode_routes.load();
  const uint64_t d_cold_routes = decode_cold_routes.load();
  const uint64_t d_hit_routes = d_routes >= d_cold_routes ? (d_routes - d_cold_routes) : 0;
  const uint64_t p_routes = prefill_routes.load();
  const uint64_t p_hits = prefill_hit_routes.load();
  const int eff_layer = layer_index >= 0 ? layer_index : 0;
  if (log_res) {
    const size_t logged_pinned_bytes =
        (session_controller && session_controller->PinnedBudget() &&
         session_controller->PinnedBudget()->GetStats().peak > 0)
            ? session_controller->PinnedBudget()->GetStats().peak
            : (MoEPackedSharedPrefillWaveStaging::kWaveExperts * slot_bytes);
    std::fprintf(stderr,
                 "QMoEOffload staging layer=%d weight_banks=2 pinned_bytes=%zu\n"
                 "QMoEOffload resident layer=%d slots=%d hit_routes=%llu routes=%llu promotions=%llu\n"
                 "QMoEOffload resident_phase layer=%d phase=0 hit_routes=%llu routes=%llu\n"
                 "QMoEOffload resident_phase layer=%d phase=1 hit_routes=%llu routes=%llu\n"
                 "QMoEOffload stream layer=%d tiles=%llu experts=%llu routes=%llu cpu_routes=0 waves=%llu reused_promotions=0\n",
                 eff_layer, logged_pinned_bytes,
                 eff_layer, resident_slots,
                 static_cast<unsigned long long>(p_hits + d_hit_routes),
                 static_cast<unsigned long long>(p_routes + d_routes),
                 static_cast<unsigned long long>(total_promotions.load()),
                 eff_layer, static_cast<unsigned long long>(p_hits), static_cast<unsigned long long>(p_routes),
                 eff_layer, static_cast<unsigned long long>(d_hit_routes), static_cast<unsigned long long>(d_routes),
                 eff_layer,
                 static_cast<unsigned long long>(stream_tiles.load()),
                 static_cast<unsigned long long>(stream_experts.load()),
                 static_cast<unsigned long long>(stream_routes.load()),
                 static_cast<unsigned long long>(stream_waves.load()));
  }
  if (log_cpu) {
    std::fprintf(stderr,
                 "QMoEOffload cpu layer=%d calls=%llu routes=%llu cold_routes=%llu cold_experts=%llu "
                 "cpu_us=%llu classify_us=0 gap_in_us=0 gap_in_n=0 gap_out_us=0 gap_out_n=0 "
                 "t1_us=%llu t2_us=%llu t3_us=0 inter_n=0 prep_us=%llu fc1_us=%llu fc2_us=%llu write_us=0 threads=%d\n",
                 eff_layer,
                 static_cast<unsigned long long>(d_calls),
                 static_cast<unsigned long long>(d_routes),
                 static_cast<unsigned long long>(d_cold_routes),
                 static_cast<unsigned long long>(decode_cold_experts.load()),
                 static_cast<unsigned long long>(decode_cpu_us.load()),
                 static_cast<unsigned long long>(decode_fc1_us.load()),
                 static_cast<unsigned long long>(decode_fc2_us.load()),
                 static_cast<unsigned long long>(decode_prep_us.load()),
                 static_cast<unsigned long long>(decode_fc1_us.load()),
                 static_cast<unsigned long long>(decode_fc2_us.load()),
                 cpu_threads);
  }
  FreeDecodePinned();
}

void MoEPackedOffloadState::RunColdDecodeStep() {
  const auto t0 = std::chrono::steady_clock::now();
  const int nrows = std::clamp<int>(h_decode_pinned->num_rows, 0, kMoEPackedMaxDecodeRows);
  const int ccount = std::clamp<int>(h_decode_pinned->cold_count, 0, kMoEPackedMaxDecodeRoutes);
  // Query the run controls here instead of latching state in ComputeInternal, which CUDA graph replay never calls.
  if (InWarmup()) {
    if (nrows > 0 && ccount > 0) {
      std::memset(h_decode_pinned->host_cold_output, 0, static_cast<size_t>(nrows * hidden_size) * sizeof(float));
    }
    return;
  }
  decode_calls.fetch_add(1, std::memory_order_relaxed);
  decode_routes.fetch_add(static_cast<uint64_t>(nrows * k), std::memory_order_relaxed);
  decode_cold_routes.fetch_add(static_cast<uint64_t>(ccount), std::memory_order_relaxed);
  if (ccount <= 0 || nrows <= 0) {
    return;
  }

  // 1. Group cold routes by unique expert
  struct ColdExpertGroup {
    int expert;
    int count;
    int token_rows[kMoEPackedMaxDecodeRows];
    float scales[kMoEPackedMaxDecodeRows];
  };
  ColdExpertGroup groups[kMoEPackedMaxDecodeRoutes];
  int num_groups = 0;
  for (int i = 0; i < ccount; ++i) {
    const int r = h_decode_pinned->cold_routes[i];
    const int exp = h_decode_pinned->cold_experts[i];
    const float sc = h_decode_pinned->cold_scales[i];
    const int trow = r / k;
    if (exp < 0 || exp >= num_experts || trow < 0 || trow >= nrows || sc == 0.0f) continue;
    int g_idx = -1;
    for (int g = 0; g < num_groups; ++g) {
      if (groups[g].expert == exp) {
        g_idx = g;
        break;
      }
    }
    if (g_idx < 0) {
      g_idx = num_groups++;
      groups[g_idx].expert = exp;
      groups[g_idx].count = 0;
    }
    if (groups[g_idx].count < kMoEPackedMaxDecodeRows) {
      const int idx = groups[g_idx].count++;
      groups[g_idx].token_rows[idx] = trow;
      groups[g_idx].scales[idx] = sc;
    }
  }
  decode_cold_experts.fetch_add(static_cast<uint64_t>(num_groups), std::memory_order_relaxed);
  if (num_groups <= 0) {
    std::memset(h_decode_pinned->host_cold_output, 0, static_cast<size_t>(nrows * hidden_size) * sizeof(float));
    return;
  }
  decode_touched_evictable_experts.clear();
  if (!is_evictable_host_expert.empty()) {
    for (int g = 0; g < num_groups; ++g) {
      const int exp = groups[g].expert;
      if (exp >= 0 && static_cast<size_t>(exp) < is_evictable_host_expert.size() &&
          is_evictable_host_expert[static_cast<size_t>(exp)] != 0) {
        decode_touched_evictable_experts.push_back(exp);
      }
    }
    if (!decode_touched_evictable_experts.empty()) {
      prefill_touched_evictable.store(true, std::memory_order_relaxed);
    }
  }

  const int env_q8_mode = ParseOptionalQ8Mode(GetEnvironmentVar("ORT_MOE_PACKED_CPU_Q8_DOT"));
  const int eff_q8_mode = (cpu_q8_dot_mode >= 0) ? cpu_q8_dot_mode : env_q8_mode;
  const bool all_quantized_weights =
      !gguf_quant::IsUnquantizedGgmlType(gate_type) &&
      !gguf_quant::IsUnquantizedGgmlType(up_type) &&
      !gguf_quant::IsUnquantizedGgmlType(down_type) &&
      gguf_quant::SupportsGgufQ8VecDotCpu(gate_type) &&
      gguf_quant::SupportsGgufQ8VecDotCpu(up_type) &&
      gguf_quant::SupportsGgufQ8VecDotCpu(down_type);
  const bool use_q8_dot =
      (eff_q8_mode > 0) &&
      all_quantized_weights &&
      ((hidden_size % 32) == 0) && ((inter_size % 32) == 0);

  const int q8_blocks_per_hidden = hidden_size / 32;
  const int q8_blocks_per_inter = inter_size / 32;
  if (use_q8_dot && x_q8_buf.size() < static_cast<size_t>(nrows * q8_blocks_per_hidden)) {
    x_q8_buf.resize(static_cast<size_t>(kMoEPackedMaxDecodeRows * (kMoEPackedMaxHiddenSize / 32)));
  }

  // 2. Convert input rows from BF16/FP16 to FP32 using pre-allocated x_f32_buf
  for (int r = 0; r < nrows; ++r) {
    const uint16_t* src = h_decode_pinned->host_input + static_cast<size_t>(r) * hidden_size;
    float* dst = x_f32_buf.data() + static_cast<size_t>(r) * kMoEPackedMaxHiddenSize;
    if (is_bf16) {
      for (int c = 0; c < hidden_size; ++c) {
        dst[c] = gguf_quant::Bf16BitsToFloatCpu(src[c]);
      }
    } else {
      for (int c = 0; c < hidden_size; ++c) {
        dst[c] = gguf_quant::Fp16BitsToFloatCpu(src[c]);
      }
    }
    if (use_q8_dot) {
      gguf_quant::QuantizeRowQ8Cpu(dst, hidden_size, x_q8_buf.data() + static_cast<size_t>(r) * q8_blocks_per_hidden);
    }
  }

  // Assign flat route offset for each group's routes
  int group_route_base[kMoEPackedMaxDecodeRoutes];
  int total_active_routes = 0;
  for (int g = 0; g < num_groups; ++g) {
    group_route_base[g] = total_active_routes;
    total_active_routes += groups[g].count;
  }
  if (inter_buf.size() < static_cast<size_t>(total_active_routes) * inter_size) {
    inter_buf.resize(static_cast<size_t>(total_active_routes) * inter_size);
  }
  if (use_q8_dot && inter_q8_buf.size() < static_cast<size_t>(total_active_routes) * q8_blocks_per_inter) {
    inter_q8_buf.resize(static_cast<size_t>(total_active_routes) * q8_blocks_per_inter);
  }
  const auto t_prep = std::chrono::steady_clock::now();

  // 3. Pass 1: Parallel FC1 (gate + up) + SwiGLU across (group, 32-channel tiles)
  constexpr int kFc1TileCh = 32;
  const int fc1_tiles_per_group = (inter_size + kFc1TileCh - 1) / kFc1TileCh;
  const int total_fc1_tasks = num_groups * fc1_tiles_per_group;
  MoEPackedParallelPool::Instance().ParallelFor(cpu_threads, total_fc1_tasks, [&](int task_idx) {
    const int g = task_idx / fc1_tiles_per_group;
    const int tile_idx = task_idx - g * fc1_tiles_per_group;
    const int ch_start = tile_idx * kFc1TileCh;
    const int ch_end = std::min(inter_size, ch_start + kFc1TileCh);
    const int exp = groups[g].expert;
    const int R = groups[g].count;
    float* group_inter = inter_buf.data() + static_cast<size_t>(group_route_base[g]) * inter_size;
    if (use_q8_dot) {
      const gguf_quant::GgufQ8Block* x_q8_ptrs[kMoEPackedMaxDecodeRows];
      for (int r = 0; r < R; ++r) {
        x_q8_ptrs[r] = x_q8_buf.data() + static_cast<size_t>(groups[g].token_rows[r]) * q8_blocks_per_hidden;
      }
      int j = ch_start;
      for (; j + 2 <= ch_end; j += 2) {
        const size_t row_idx0 = static_cast<size_t>(exp) * inter_size + j;
        const uint8_t* g_row0 = host_gate_w + row_idx0 * gate_row_bytes;
        const uint8_t* u_row0 = host_up_w + row_idx0 * up_row_bytes;
        const uint8_t* g_row1 = g_row0 + gate_row_bytes;
        const uint8_t* u_row1 = u_row0 + up_row_bytes;
        int r = 0;
        for (; r + 4 <= R; r += 4) {
          float g2_a[2], u2_a[2], g2_b[2], u2_b[2], g2_c[2], u2_c[2], g2_d[2], u2_d[2];
          gguf_quant::VecDotGgufGateUpRow2xQ8Batch4Cpu(
              gate_type, g_row0, g_row1, up_type, u_row0, u_row1,
              x_q8_ptrs[r + 0], x_q8_ptrs[r + 1], x_q8_ptrs[r + 2], x_q8_ptrs[r + 3],
              hidden_size, g2_a, u2_a, g2_b, u2_b, g2_c, u2_c, g2_d, u2_d);
          const float* g2_arr[4] = {g2_a, g2_b, g2_c, g2_d};
          const float* u2_arr[4] = {u2_a, u2_b, u2_c, u2_d};
          for (int sub_r = 0; sub_r < 4; ++sub_r) {
            for (int k_ch = 0; k_ch < 2; ++k_ch) {
              const float gv = RoundToActivationType(g2_arr[sub_r][k_ch], is_bf16);
              const float uv = RoundToActivationType(u2_arr[sub_r][k_ch], is_bf16);
              group_inter[static_cast<size_t>(r + sub_r) * inter_size + j + k_ch] =
                  RoundToActivationType(uv * (gv / (1.0f + std::exp(-gv))), is_bf16);
            }
          }
        }
        for (; r + 2 <= R; r += 2) {
          float g2_a[2], u2_a[2], g2_b[2], u2_b[2];
          gguf_quant::VecDotGgufGateUpRow2xQ8Batch2Cpu(
              gate_type, g_row0, g_row1, up_type, u_row0, u_row1,
              x_q8_ptrs[r + 0], x_q8_ptrs[r + 1], hidden_size, g2_a, u2_a, g2_b, u2_b);
          for (int k_ch = 0; k_ch < 2; ++k_ch) {
            const float gv_a = RoundToActivationType(g2_a[k_ch], is_bf16);
            const float uv_a = RoundToActivationType(u2_a[k_ch], is_bf16);
            group_inter[static_cast<size_t>(r + 0) * inter_size + j + k_ch] =
                RoundToActivationType(uv_a * (gv_a / (1.0f + std::exp(-gv_a))), is_bf16);
            const float gv_b = RoundToActivationType(g2_b[k_ch], is_bf16);
            const float uv_b = RoundToActivationType(u2_b[k_ch], is_bf16);
            group_inter[static_cast<size_t>(r + 1) * inter_size + j + k_ch] =
                RoundToActivationType(uv_b * (gv_b / (1.0f + std::exp(-gv_b))), is_bf16);
          }
        }
        for (; r < R; ++r) {
          float g2[2], u2[2];
          gguf_quant::VecDotGgufGateUpRow2xQ8Cpu(
              gate_type, g_row0, g_row1, up_type, u_row0, u_row1, x_q8_ptrs[r], hidden_size, g2, u2);
          for (int k_ch = 0; k_ch < 2; ++k_ch) {
            const float gv = RoundToActivationType(g2[k_ch], is_bf16);
            const float uv = RoundToActivationType(u2[k_ch], is_bf16);
            group_inter[static_cast<size_t>(r) * inter_size + j + k_ch] =
                RoundToActivationType(uv * (gv / (1.0f + std::exp(-gv))), is_bf16);
          }
        }
      }
      if (ch_end - ch_start == 32) {
        gguf_quant::GgufQ8Block* group_inter_q8 =
            inter_q8_buf.data() + static_cast<size_t>(group_route_base[g]) * q8_blocks_per_inter;
        for (int r = 0; r < R; ++r) {
          gguf_quant::QuantizeRowQ8Cpu(
              group_inter + static_cast<size_t>(r) * inter_size + ch_start, 32,
              group_inter_q8 + static_cast<size_t>(r) * q8_blocks_per_inter + tile_idx);
        }
      }
      return;
    }
    const float* x_ptrs[kMoEPackedMaxDecodeRows];
    for (int r = 0; r < R; ++r) {
      x_ptrs[r] = x_f32_buf.data() + static_cast<size_t>(groups[g].token_rows[r]) * kMoEPackedMaxHiddenSize;
    }
    int j = ch_start;
    for (; j + 2 <= ch_end; j += 2) {
      const size_t row_idx0 = static_cast<size_t>(exp) * inter_size + j;
      const uint8_t* g_row0 = host_gate_w + row_idx0 * gate_row_bytes;
      const uint8_t* u_row0 = host_up_w + row_idx0 * up_row_bytes;
      const uint8_t* g_row1 = g_row0 + gate_row_bytes;
      const uint8_t* u_row1 = u_row0 + up_row_bytes;
      float g_dots0[kMoEPackedMaxDecodeRows];
      float u_dots0[kMoEPackedMaxDecodeRows];
      float g_dots1[kMoEPackedMaxDecodeRows];
      float u_dots1[kMoEPackedMaxDecodeRows];
      GgufDualDot2ChCpu(g_row0, u_row0, g_row1, u_row1, gate_type, up_type, hidden_size, x_ptrs, R,
                        g_dots0, u_dots0, g_dots1, u_dots1);
      for (int r = 0; r < R; ++r) {
        const float gv0 = RoundToActivationType(g_dots0[r], is_bf16);
        const float uv0 = RoundToActivationType(u_dots0[r], is_bf16);
        const float sw0 = uv0 * (gv0 / (1.0f + std::exp(-gv0)));
        group_inter[static_cast<size_t>(r) * inter_size + j + 0] = RoundToActivationType(sw0, is_bf16);

        const float gv1 = RoundToActivationType(g_dots1[r], is_bf16);
        const float uv1 = RoundToActivationType(u_dots1[r], is_bf16);
        const float sw1 = uv1 * (gv1 / (1.0f + std::exp(-gv1)));
        group_inter[static_cast<size_t>(r) * inter_size + j + 1] = RoundToActivationType(sw1, is_bf16);
      }
    }
    for (; j < ch_end; ++j) {
      const size_t row_idx = static_cast<size_t>(exp) * inter_size + j;
      const uint8_t* g_row = host_gate_w + row_idx * gate_row_bytes;
      const uint8_t* u_row = host_up_w + row_idx * up_row_bytes;
      float g_dots[kMoEPackedMaxDecodeRows];
      float u_dots[kMoEPackedMaxDecodeRows];
      GgufDualDotRowCpu(g_row, u_row, gate_type, up_type, hidden_size, x_ptrs, R, g_dots, u_dots);
      for (int r = 0; r < R; ++r) {
        const float gv = RoundToActivationType(g_dots[r], is_bf16);
        const float uv = RoundToActivationType(u_dots[r], is_bf16);
        const float swiglu = uv * (gv / (1.0f + std::exp(-gv)));
        group_inter[static_cast<size_t>(r) * inter_size + j] = RoundToActivationType(swiglu, is_bf16);
      }
    }
  });
  const auto t_fc1 = std::chrono::steady_clock::now();

  // 4. Pass 2: Parallel FC2 partitioned by 64-column output slices across all cold groups (lock-free, L1-resident inter_buf)
  constexpr int kFc2TileCols = 64;
  const int total_fc2_tasks = (hidden_size + kFc2TileCols - 1) / kFc2TileCols;
#if ORT_MOE_PACKED_CPU_X86_64
  const bool use_avx2 = UseAvx2CpuKernels();
#endif
  MoEPackedParallelPool::Instance().ParallelFor(cpu_threads, total_fc2_tasks, [&](int task_idx) {
    const int col_start = task_idx * kFc2TileCols;
    const int col_end = std::min(hidden_size, col_start + kFc2TileCols);
    const int col_cnt = col_end - col_start;
    float acc_out[kMoEPackedMaxDecodeRows][kFc2TileCols];
#if ORT_MOE_PACKED_CPU_X86_64
    float avx2_dots[kMoEPackedMaxDecodeRows][kFc2TileCols];
#endif
    for (int r = 0; r < nrows; ++r) {
      for (int c = 0; c < col_cnt; ++c) acc_out[r][c] = 0.0f;
    }
    for (int g = 0; g < num_groups; ++g) {
      const int exp = groups[g].expert;
      const int R = groups[g].count;
      if (use_q8_dot && (inter_size % 32) == 0) {
        const gguf_quant::GgufQ8Block* group_inter_q8 =
            inter_q8_buf.data() + static_cast<size_t>(group_route_base[g]) * q8_blocks_per_inter;
        int c = 0;
        for (; c + 4 <= col_cnt; c += 4) {
          const size_t base_idx = static_cast<size_t>(exp) * hidden_size + col_start + c;
          const uint8_t* r0 = host_down_w + (base_idx + 0) * down_row_bytes;
          const uint8_t* r1 = host_down_w + (base_idx + 1) * down_row_bytes;
          const uint8_t* r2 = host_down_w + (base_idx + 2) * down_row_bytes;
          const uint8_t* r3 = host_down_w + (base_idx + 3) * down_row_bytes;
          int r = 0;
          for (; r + 4 <= R; r += 4) {
            float out4_a[4], out4_b[4], out4_c[4], out4_d[4];
            gguf_quant::VecDotGgufRow4xQ8Batch4Cpu(
                down_type, r0, r1, r2, r3,
                group_inter_q8 + static_cast<size_t>(r + 0) * q8_blocks_per_inter,
                group_inter_q8 + static_cast<size_t>(r + 1) * q8_blocks_per_inter,
                group_inter_q8 + static_cast<size_t>(r + 2) * q8_blocks_per_inter,
                group_inter_q8 + static_cast<size_t>(r + 3) * q8_blocks_per_inter,
                inter_size, out4_a, out4_b, out4_c, out4_d);
            const float* out4_arr[4] = {out4_a, out4_b, out4_c, out4_d};
            for (int sub_r = 0; sub_r < 4; ++sub_r) {
              const int trow = groups[g].token_rows[r + sub_r];
              const float sc = groups[g].scales[r + sub_r];
              for (int k_c = 0; k_c < 4; ++k_c) {
                acc_out[trow][c + k_c] += sc * RoundToActivationType(out4_arr[sub_r][k_c], is_bf16);
              }
            }
          }
          for (; r + 2 <= R; r += 2) {
            float out4_a[4], out4_b[4];
            gguf_quant::VecDotGgufRow4xQ8Batch2Cpu(
                down_type, r0, r1, r2, r3,
                group_inter_q8 + static_cast<size_t>(r + 0) * q8_blocks_per_inter,
                group_inter_q8 + static_cast<size_t>(r + 1) * q8_blocks_per_inter,
                inter_size, out4_a, out4_b);
            const int trow_a = groups[g].token_rows[r + 0];
            const int trow_b = groups[g].token_rows[r + 1];
            const float sc_a = groups[g].scales[r + 0];
            const float sc_b = groups[g].scales[r + 1];
            for (int k_c = 0; k_c < 4; ++k_c) {
              acc_out[trow_a][c + k_c] += sc_a * RoundToActivationType(out4_a[k_c], is_bf16);
              acc_out[trow_b][c + k_c] += sc_b * RoundToActivationType(out4_b[k_c], is_bf16);
            }
          }
          for (; r < R; ++r) {
            float out4[4];
            gguf_quant::VecDotGgufRow4xQ8Cpu(
                down_type, r0, r1, r2, r3,
                group_inter_q8 + static_cast<size_t>(r) * q8_blocks_per_inter,
                inter_size, out4);
            const int trow = groups[g].token_rows[r];
            const float sc = groups[g].scales[r];
            for (int k_c = 0; k_c < 4; ++k_c) {
              acc_out[trow][c + k_c] += sc * RoundToActivationType(out4[k_c], is_bf16);
            }
          }
        }
        for (; c < col_cnt; ++c) {
          const uint8_t* d_row = host_down_w + (static_cast<size_t>(exp) * hidden_size + col_start + c) * down_row_bytes;
          for (int r = 0; r < R; ++r) {
            const float dv = RoundToActivationType(
                gguf_quant::VecDotGgufRowQ8Cpu(
                    down_type, d_row, group_inter_q8 + static_cast<size_t>(r) * q8_blocks_per_inter, inter_size),
                is_bf16);
            acc_out[groups[g].token_rows[r]][c] += groups[g].scales[r] * dv;
          }
        }
        continue;
      }
      const float* inter_ptrs[kMoEPackedMaxDecodeRows];
      const float* group_inter = inter_buf.data() + static_cast<size_t>(group_route_base[g]) * inter_size;
      for (int r = 0; r < R; ++r) {
        inter_ptrs[r] = group_inter + static_cast<size_t>(r) * inter_size;
      }
      int c = 0;
#if ORT_MOE_PACKED_CPU_X86_64
      if (use_avx2) {
        const uint8_t* dr0 = host_down_w + (static_cast<size_t>(exp) * hidden_size + col_start) * down_row_bytes;
        c = moe_packed_cpu::GgufDotRowsBy4Avx2(dr0, down_row_bytes, col_cnt, down_type, inter_size, inter_ptrs, R,
                                               &avx2_dots[0][0], kFc2TileCols);
        for (int r = 0; r < R; ++r) {
          const int trow = groups[g].token_rows[r];
          const float sc = groups[g].scales[r];
          for (int j = 0; j < c; ++j) {
            acc_out[trow][j] += sc * RoundToActivationType(avx2_dots[r][j], is_bf16);
          }
        }
      }
#endif
      for (; c < col_cnt; ++c) {
        const int col = col_start + c;
        const uint8_t* d_row = host_down_w + (static_cast<size_t>(exp) * hidden_size + col) * down_row_bytes;
        float d_dots[kMoEPackedMaxDecodeRows];
        GgufDotRowCpu(d_row, down_type, inter_size, inter_ptrs, R, d_dots);
        for (int r = 0; r < R; ++r) {
          const float dv = RoundToActivationType(d_dots[r], is_bf16);
          acc_out[groups[g].token_rows[r]][c] += groups[g].scales[r] * dv;
        }
      }
    }
    for (int r = 0; r < nrows; ++r) {
      float* dst = h_decode_pinned->host_cold_output + static_cast<size_t>(r) * hidden_size + col_start;
      std::memcpy(dst, acc_out[r], static_cast<size_t>(col_cnt) * sizeof(float));
    }
  });
  const auto t_fc2 = std::chrono::steady_clock::now();

  decode_prep_us.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t_prep - t0).count()),
      std::memory_order_relaxed);
  decode_fc1_us.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t_fc1 - t_prep).count()),
      std::memory_order_relaxed);
  decode_fc2_us.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t_fc2 - t_fc1).count()),
      std::memory_order_relaxed);
  decode_cpu_us.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t_fc2 - t0).count()),
      std::memory_order_relaxed);
}

#define REGISTER_MOE_PACKED_CUDA(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      MoEPacked,                                                         \
      kMSDomain, 1,                                                      \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("TR", {DataTypeImpl::GetTensorType<T>(),       \
                                 DataTypeImpl::GetTensorType<float>()})  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()), \
      MoEPacked<T>);

REGISTER_MOE_PACKED_CUDA(MLFloat16)
REGISTER_MOE_PACKED_CUDA(BFloat16)

#undef REGISTER_MOE_PACKED_CUDA

template <typename T>
MoEPacked<T>::MoEPacked(const OpKernelInfo& info) : CudaKernel(info) {
  k_ = info.GetAttrOrDefault<int64_t>("k", 1);
  ORT_ENFORCE(k_ > 0 && k_ <= 32,
              "MoEPacked requires k in 1..32; softmax routing requires k <= 16");
  normalize_routing_weights_ = info.GetAttrOrDefault<int64_t>("normalize_routing_weights", 1);
  routed_scaling_factor_ = info.GetAttrOrDefault<float>("routed_scaling_factor", 1.0f);
  routing_epsilon_ = info.GetAttrOrDefault<float>("routing_epsilon", 0.0f);
  ORT_ENFORCE(std::isfinite(routed_scaling_factor_),
              "MoEPacked routed_scaling_factor must be finite");
  ORT_ENFORCE(std::isfinite(routing_epsilon_) && routing_epsilon_ >= 0.0f,
              "MoEPacked routing_epsilon must be finite and >= 0");
  activation_type_ = info.GetAttrOrDefault<std::string>("activation_type", "swiglu");
  if (activation_type_ == "relu_squared") activation_type_ = "relu2";
  ORT_ENFORCE(activation_type_ == "swiglu" || activation_type_ == "relu2",
              "MoEPacked supports activation_type='swiglu' or 'relu2' ('relu_squared' alias)");
  ORT_ENFORCE(info.GetAttr<int64_t>("gate_type", &gate_type_).IsOK(), "MoEPacked requires gate_type");
  up_type_ = info.GetAttrOrDefault<int64_t>("up_type", -1);
  ORT_ENFORCE(activation_type_ != "swiglu" || up_type_ >= 0, "MoEPacked swiglu requires up_type");
  ORT_ENFORCE(info.GetAttr<int64_t>("down_type", &down_type_).IsOK(), "MoEPacked requires down_type");
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(gate_type_), "MoEPacked: unsupported gate_type=", gate_type_);
  ORT_ENFORCE(activation_type_ != "swiglu" || gguf_quant::IsGgmlTypeSupported(up_type_),
              "MoEPacked: unsupported up_type=", up_type_);
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(down_type_), "MoEPacked: unsupported down_type=", down_type_);
  ORT_ENFORCE(activation_type_ != "relu2" ||
                  (gate_type_ == gguf_quant::kGgmlTypeIQ2_XXS && down_type_ == gguf_quant::kGgmlTypeIQ2_XXS),
              "MoEPacked relu2 currently requires IQ2_XXS gate and down weights");
  if constexpr (std::is_same_v<T, BFloat16>) {
    ORT_ENFORCE(GetDeviceProp().major >= 8, "MoEPacked BF16 requires compute capability 8.0 or higher");
    ORT_THROW_IF_ERROR(GetMoEPackedBf16KernelSupport(GetDeviceId(), bf16_tc_supported_));
  }

  // Check if inputs 2..4 were placed on CPU by PlaceMoeOffloadInitializersOnCpu for hybrid offload.
  const Tensor* gate_w = nullptr;
  const Tensor* up_w = nullptr;
  const Tensor* down_w = nullptr;
  if (info.TryGetConstantInput(2, &gate_w) && gate_w &&
      info.TryGetConstantInput(3, &up_w) && up_w &&
      info.TryGetConstantInput(4, &down_w) && down_w &&
      gate_w->Location().device.Type() == OrtDevice::CPU &&
      up_w->Location().device.Type() == OrtDevice::CPU &&
      down_w->Location().device.Type() == OrtDevice::CPU &&
      gate_w->Shape().NumDimensions() == 3 &&
      up_w->Shape().NumDimensions() == 3 &&
      down_w->Shape().NumDimensions() == 3) {
    using onnxruntime::moe_offload::MoeOffloadSessionPolicy;
    auto controller = GetQMoESessionController();
    std::filesystem::path model_path = controller ? controller->GetModelPath() : std::filesystem::path{};
    ORT_THROW_IF_ERROR(MoeOffloadSessionPolicy::ParseFromConfigOptions(
        info.GetConfigOptions(), model_path, offload_policy_));
    if (controller != nullptr && offload_policy_.mode != onnxruntime::moe_offload::OffloadMode::kDisabled) {
      auto get_input_name = [&](size_t slot) -> std::string {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
        if (slot < info.GetKernelInfo().GetInputCount()) {
          return info.GetKernelInfo().GetInputName(slot);
        }
#else
        const auto& in_defs = info.node().InputDefs();
        if (slot < in_defs.size() && in_defs[slot] != nullptr && in_defs[slot]->Exists()) {
          return in_defs[slot]->Name();
        }
#endif
        return {};
      };
      int32_t parsed_layer = -1;
      if (MoeOffloadSessionPolicy::TryExtractLayerIndexFromName(info.node().Name(), parsed_layer)) {
        offload_layer_index_ = parsed_layer;
      } else {
        const std::string in2_name = get_input_name(2);
        if (!in2_name.empty() && MoeOffloadSessionPolicy::TryExtractLayerIndexFromName(in2_name, parsed_layer)) {
          offload_layer_index_ = parsed_layer;
        }
      }

      const int32_t num_e = static_cast<int32_t>(gate_w->Shape()[0]);
      const int32_t inter = static_cast<int32_t>(gate_w->Shape()[1]);
      const int32_t hid = static_cast<int32_t>(down_w->Shape()[1]);
      // Pre-populate the OpKernelInfo adapter's allocator cache while OrtKernelInfo is still live during CreateKernel.
      static_cast<void>(info.GetAllocator(OrtMemTypeDefault));
      const std::string q8_cfg = info.GetConfigOptions().GetConfigOrDefault("ep.cuda.moe_packed_cpu_q8_dot", "");
      const int parsed_q8_mode = ParseOptionalQ8Mode(q8_cfg);
      if (parsed_q8_mode >= 0) {
        SetNodeCpuQ8Mode(this, parsed_q8_mode);
      }
      offload_source_desc_.op_type = "MoEPacked";
      offload_source_desc_.quant_type = "gguf";
      offload_source_desc_.stable_node_id = info.node().Name();
      offload_source_desc_.layer_index = offload_layer_index_;
      offload_source_desc_.num_experts = num_e;
      offload_source_desc_.top_k = static_cast<int32_t>(k_);
      offload_source_desc_.hidden_size = hid;
      offload_source_desc_.inter_size = inter;
      offload_source_desc_.block_size = 32;
      offload_source_desc_.weight_bits = 2;
      offload_source_desc_.weights_prepacked = true;
      offload_source_desc_.placement.device_id = controller->DeviceId();
      offload_source_desc_.shard_spec.global_num_experts = num_e;
      offload_source_desc_.shard_spec.local_num_experts = num_e;
      offload_source_desc_.shard_spec.global_hidden_size = hid;
      offload_source_desc_.shard_spec.global_inter_size = inter;
      offload_source_desc_.shard_spec.local_hidden_size = hid;
      offload_source_desc_.shard_spec.local_inter_size = inter;
      offload_source_desc_.shard_spec.quant_block_size = 32;

      static_cast<void>(
          onnxruntime::moe_offload::PromoteFileMappingToSharedReadOnly(gate_w->DataRaw(), gate_w->SizeInBytes()));
      static_cast<void>(
          onnxruntime::moe_offload::PromoteFileMappingToSharedReadOnly(up_w->DataRaw(), up_w->SizeInBytes()));
      static_cast<void>(
          onnxruntime::moe_offload::PromoteFileMappingToSharedReadOnly(down_w->DataRaw(), down_w->SizeInBytes()));

      auto record_slice = [&](onnxruntime::moe_offload::ExpertBankKind kind, int slot_idx, const Tensor* t) {
        onnxruntime::moe_offload::MoeExpertBankSlice slice;
        slice.kind = kind;
        slice.input_index = slot_idx;
        slice.initializer_name = get_input_name(static_cast<size_t>(slot_idx));
        slice.elem_type = static_cast<ONNXTensorElementDataType>(t->GetElementType());
        slice.full_shape = t->Shape();
        slice.total_bytes = t->SizeInBytes();
        slice.bytes_per_expert = t->SizeInBytes() / static_cast<size_t>(num_e);
        slice.host_data = t->DataRaw();
        slice.is_file_mapped = onnxruntime::moe_offload::IsReloadableFileMapping(slice.host_data, slice.total_bytes);
        offload_source_desc_.banks[static_cast<size_t>(kind)] = std::move(slice);
      };
      record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc1Weights, 2, gate_w);
      record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc1Scales, 3, up_w);
      record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc2Weights, 4, down_w);

      const size_t host_page_size = onnxruntime::cuda::QMoESessionController::GetHostPageSize();
      const auto& gate_bank =
          *offload_source_desc_.banks[static_cast<size_t>(onnxruntime::moe_offload::ExpertBankKind::kFc1Weights)];
      const auto& up_bank =
          *offload_source_desc_.banks[static_cast<size_t>(onnxruntime::moe_offload::ExpertBankKind::kFc1Scales)];
      const auto& down_bank =
          *offload_source_desc_.banks[static_cast<size_t>(onnxruntime::moe_offload::ExpertBankKind::kFc2Weights)];
      const bool all_file_mapped =
          gate_bank.is_file_mapped && gate_bank.bytes_per_expert >= host_page_size &&
          up_bank.is_file_mapped && up_bank.bytes_per_expert >= host_page_size &&
          down_bank.is_file_mapped && down_bank.bytes_per_expert >= host_page_size;
      const size_t slot_bytes =
          (gate_w->SizeInBytes() + up_w->SizeInBytes() + down_w->SizeInBytes()) / static_cast<size_t>(num_e);
      offload_capabilities_.supports_full_gpu_resident = true;
      offload_capabilities_.supports_hybrid_doorbell_decode = true;
      offload_capabilities_.supports_gpu_streaming_prefill = true;
      // When all expert banks are reloadable file mappings (e.g., GGUF), bounded host_expert_bytes caps are honored
      // via madvise(MADV_DONTNEED) on tail cold experts; in-memory initializers still reject caps below the cold complement.
      offload_capabilities_.supports_file_backed_bounded_host_cache = all_file_mapped;
      offload_capabilities_.supports_cuda_graph_capture = true;
      offload_capabilities_.decode_kernel_skips_negative_slots = true;
      offload_capabilities_.source_bytes_per_expert = slot_bytes;
      offload_capabilities_.gpu_slot_bytes = slot_bytes;
      offload_capabilities_.gpu_fixed_table_bytes = static_cast<size_t>(2) * num_e * sizeof(int32_t);
      offload_capabilities_.cpu_slot_bytes = slot_bytes;
      offload_capabilities_.min_host_cache_slots = 1;

      controller->RegisterMoeOffloadNode(this, offload_policy_, offload_layer_index_);
      offload_node_registered_ = true;
    }
  }
}

template <typename T>
MoEPacked<T>::~MoEPacked() {
  EraseNodeCpuQ8Mode(this);
  if (offload_node_registered_) {
    if (auto controller = GetQMoESessionController()) {
      if (offload_state_ != nullptr) {
        controller->UnregisterOffloadNode(this);
      }
      controller->UnregisterMoeOffloadNode(this);
    }
  }
}

template <typename T>
Status MoEPacked<T>::InitializeAdmittedPlan(const onnxruntime::moe_offload::MoeOffloadSessionPolicy& policy,
                                            const onnxruntime::moe_offload::MoeNodeExecutionPlan& plan) {
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaSetDevice(GetDeviceId())));
  offload_policy_ = policy;
  offload_plan_ = plan;
  offload_layer_index_ = plan.layer_index;

  const Tensor* gate_w = nullptr;
  const Tensor* up_w = nullptr;
  const Tensor* down_w = nullptr;
  ORT_RETURN_IF_NOT(Info().TryGetConstantInput(2, &gate_w) && gate_w &&
                        Info().TryGetConstantInput(3, &up_w) && up_w &&
                        Info().TryGetConstantInput(4, &down_w) && down_w,
                    "MoEPacked InitializeAdmittedPlan requires constant expert weight inputs 2..4.");

  auto st = std::make_shared<MoEPackedOffloadState>();
  st->layer_index = plan.layer_index;
  st->num_experts = static_cast<int32_t>(gate_w->Shape()[0]);
  st->inter_size = static_cast<int32_t>(gate_w->Shape()[1]);
  st->hidden_size = static_cast<int32_t>(down_w->Shape()[1]);
  st->k = static_cast<int32_t>(k_);
  st->is_bf16 = std::is_same_v<T, BFloat16>;
  st->normalize_routing_weights = (normalize_routing_weights_ != 0);
  st->gate_type = static_cast<int>(gate_type_);
  st->up_type = static_cast<int>(up_type_);
  st->down_type = static_cast<int>(down_type_);
  st->gate_row_bytes = static_cast<size_t>(gate_w->Shape()[2]);
  st->up_row_bytes = static_cast<size_t>(up_w->Shape()[2]);
  st->down_row_bytes = static_cast<size_t>(down_w->Shape()[2]);
  st->gate_exp_bytes = static_cast<size_t>(st->inter_size) * st->gate_row_bytes;
  st->up_exp_bytes = static_cast<size_t>(st->inter_size) * st->up_row_bytes;
  st->down_exp_bytes = static_cast<size_t>(st->hidden_size) * st->down_row_bytes;
  st->slot_bytes = st->gate_exp_bytes + st->up_exp_bytes + st->down_exp_bytes;

  st->host_gate_w = static_cast<const uint8_t*>(gate_w->DataRaw());
  st->host_up_w = static_cast<const uint8_t*>(up_w->DataRaw());
  st->host_down_w = static_cast<const uint8_t*>(down_w->DataRaw());

  st->tail_sub_threshold = policy.tail_substitution_threshold;
  st->tail_sub_min_rank = policy.tail_substitution_min_rank;
  st->min_expert_weight = policy.min_expert_weight;
  st->skip_tail_cold_only = policy.skip_tail_cold_only;
  const std::string det_env = GetEnvironmentVar("ORT_QMOE_DETERMINISTIC_REPLAY");
  const bool det_replay = !det_env.empty() && det_env != "0";
  st->is_adaptive = !det_replay && (policy.placement == onnxruntime::moe_offload::PlacementPolicy::kAdaptive);
  st->cpu_execution_enabled = (policy.cpu_execution != onnxruntime::moe_offload::CpuExecutionMode::kDisabled);
  st->cpu_q8_dot_mode = GetNodeCpuQ8Mode(this);
  st->cpu_threads = MoEPackedParallelPool::ResolveThreads(policy.cpu_threads);
  st->session_controller = GetQMoESessionController();
  st->x_f32_buf.resize(static_cast<size_t>(kMoEPackedMaxDecodeRows) * kMoEPackedMaxHiddenSize, 0.0f);
  st->inter_buf.resize(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * st->inter_size, 0.0f);

  st->resident_slots = std::clamp<int32_t>(plan.resident_slots, 0, st->num_experts);
  st->host_expert_to_slot.assign(static_cast<size_t>(st->num_experts), -1);
  st->expert_ema.assign(static_cast<size_t>(st->num_experts), 0.0f);

  st->slot_to_expert.reserve(static_cast<size_t>(st->resident_slots));
  if (st->resident_slots == st->num_experts) {
    for (int32_t exp = 0; exp < st->num_experts; ++exp) {
      st->host_expert_to_slot[static_cast<size_t>(exp)] = exp;
      st->slot_to_expert.push_back(exp);
      st->expert_ema[static_cast<size_t>(exp)] = 4.0f;
    }
  } else {
    for (int32_t exp : plan.initial_resident_experts) {
      if (static_cast<int32_t>(st->slot_to_expert.size()) >= st->resident_slots) break;
      if (exp >= 0 && exp < st->num_experts && st->host_expert_to_slot[static_cast<size_t>(exp)] < 0) {
        const int32_t slot = static_cast<int32_t>(st->slot_to_expert.size());
        st->host_expert_to_slot[static_cast<size_t>(exp)] = slot;
        st->slot_to_expert.push_back(exp);
        st->expert_ema[static_cast<size_t>(exp)] =
            10.0f - 6.0f * (static_cast<float>(slot) / static_cast<float>(std::max(1, st->resident_slots)));
      }
    }
    for (int32_t exp = 0; exp < st->num_experts && static_cast<int32_t>(st->slot_to_expert.size()) < st->resident_slots; ++exp) {
      if (st->host_expert_to_slot[static_cast<size_t>(exp)] < 0) {
        const int32_t slot = static_cast<int32_t>(st->slot_to_expert.size());
        st->host_expert_to_slot[static_cast<size_t>(exp)] = slot;
        st->slot_to_expert.push_back(exp);
        st->expert_ema[static_cast<size_t>(exp)] = 2.0f;
      }
    }
  }
  st->resident_slots = static_cast<int32_t>(st->slot_to_expert.size());

  AllocatorPtr alloc = Info().GetAllocator(OrtMemTypeDefault);
  ORT_RETURN_IF_NOT(alloc != nullptr, "MoEPacked InitializeAdmittedPlan requires a valid CUDA device allocator.");

  if (st->resident_slots > 0) {
    const size_t total_gate = static_cast<size_t>(st->resident_slots) * st->gate_exp_bytes;
    const size_t total_up = static_cast<size_t>(st->resident_slots) * st->up_exp_bytes;
    const size_t total_down = static_cast<size_t>(st->resident_slots) * st->down_exp_bytes;
    st->gpu_resident_buf = IAllocator::MakeUniquePtr<uint8_t>(alloc, total_gate + total_up + total_down, false, nullptr);
    st->gpu_gate_bank = st->gpu_resident_buf.get();
    st->gpu_up_bank = st->gpu_gate_bank + total_gate;
    st->gpu_down_bank = st->gpu_up_bank + total_up;

    for (int32_t s = 0; s < st->resident_slots; ++s) {
      const int32_t exp = st->slot_to_expert[static_cast<size_t>(s)];
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
          st->gpu_gate_bank + static_cast<size_t>(s) * st->gate_exp_bytes,
          st->host_gate_w + static_cast<size_t>(exp) * st->gate_exp_bytes,
          st->gate_exp_bytes, cudaMemcpyHostToDevice)));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
          st->gpu_up_bank + static_cast<size_t>(s) * st->up_exp_bytes,
          st->host_up_w + static_cast<size_t>(exp) * st->up_exp_bytes,
          st->up_exp_bytes, cudaMemcpyHostToDevice)));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
          st->gpu_down_bank + static_cast<size_t>(s) * st->down_exp_bytes,
          st->host_down_w + static_cast<size_t>(exp) * st->down_exp_bytes,
          st->down_exp_bytes, cudaMemcpyHostToDevice)));
    }
    onnxruntime::cuda::QMoESessionController::EvictUploadedHostPages(offload_source_desc_, st->slot_to_expert);
  }

  st->source_desc = offload_source_desc_;
  st->host_cache_slots = plan.host_cache_slots;
  if (plan.use_file_backing && plan.host_cache_slots < st->num_experts - st->resident_slots) {
    // Disable adaptive GPU slot promotion when bounded file-backed host eviction is active so an initial GPU-resident
    // expert whose host pages were evicted after upload is never demoted to cold without host eviction tracking.
    st->is_adaptive = false;
    // Select cold experts from the tail of plan.ranked_experts (falling back to descending expert index) so popular
    // cold experts remain cached in RAM while the coldest tail experts are evicted.
    st->is_evictable_host_expert.assign(static_cast<size_t>(st->num_experts), 0);
    const int32_t max_cold_to_evict = std::max<int32_t>(0, (st->num_experts - st->resident_slots) - plan.host_cache_slots);
    int32_t evicted_cnt = 0;
    for (auto it = plan.ranked_experts.rbegin();
         it != plan.ranked_experts.rend() && evicted_cnt < max_cold_to_evict; ++it) {
      const int32_t exp = *it;
      if (exp < 0 || exp >= st->num_experts) continue;
      if (st->host_expert_to_slot[static_cast<size_t>(exp)] >= 0 ||
          st->is_evictable_host_expert[static_cast<size_t>(exp)] != 0) {
        continue;
      }
      st->is_evictable_host_expert[static_cast<size_t>(exp)] = 1;
      st->evictable_host_experts.push_back(exp);
      ++evicted_cnt;
    }
    for (int32_t exp = st->num_experts - 1; exp >= 0 && evicted_cnt < max_cold_to_evict; --exp) {
      if (st->host_expert_to_slot[static_cast<size_t>(exp)] >= 0 ||
          st->is_evictable_host_expert[static_cast<size_t>(exp)] != 0) {
        continue;
      }
      st->is_evictable_host_expert[static_cast<size_t>(exp)] = 1;
      st->evictable_host_experts.push_back(exp);
      ++evicted_cnt;
    }
    if (!st->evictable_host_experts.empty()) {
      onnxruntime::cuda::QMoESessionController::EvictUploadedHostPages(offload_source_desc_, st->evictable_host_experts);
    }
  }

  st->d_expert_to_slot = IAllocator::MakeUniquePtr<int32_t>(alloc, static_cast<size_t>(st->num_experts), false, nullptr);
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
      st->d_expert_to_slot.get(), st->host_expert_to_slot.data(),
      static_cast<size_t>(st->num_experts) * sizeof(int32_t), cudaMemcpyHostToDevice)));

  // Release any shared prefill staging buffers retained by an earlier session on this device before reserving pinned
  // memory for the new session.
  {
    auto& wave_staging = MoEPackedSharedPrefillWaveStaging::ForDevice(GetDeviceId());
    std::lock_guard<std::mutex> wave_lk(wave_staging.mu);
    auto sess_pinned = st->session_controller ? st->session_controller->PinnedBudget() : nullptr;
    if (wave_staging.bank_bytes > 0 && wave_staging.bank_session_pinned_budget != sess_pinned) {
      ORT_RETURN_IF_ERROR(wave_staging.WaitForBanks());
      wave_staging.FreeBanks();
    }
    if (wave_staging.pinned_route_cap > 0 && wave_staging.route_session_pinned_budget != sess_pinned) {
      ORT_RETURN_IF_ERROR(wave_staging.WaitForPreviousUse());
      ORT_RETURN_IF_ERROR(wave_staging.WaitForBanks());
      wave_staging.FreePinnedRouteBuffers();
    }
  }

  // Only allocate persistent decode workspace and mapped pinned doorbell state when this node can execute hybrid CPU
  // decode. Fully GPU-resident and CPU-disabled plans need no decode doorbell buffers.
  const bool want_hybrid_decode =
      plan.enable_hybrid_cpu_decode &&
      st->resident_slots < st->num_experts &&
      st->cpu_execution_enabled &&
      policy.async_max_tokens > 0 &&
      st->hidden_size <= kMoEPackedMaxHiddenSize;
  if (want_hybrid_decode) {
    size_t decode_ws_bytes = 0;
    decode_ws_bytes += AlignUp256(sizeof(int));
    decode_ws_bytes += AlignUp256(sizeof(int));
    decode_ws_bytes += AlignUp256(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * sizeof(float));
    decode_ws_bytes += AlignUp256(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * sizeof(int));
    decode_ws_bytes += AlignUp256(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * sizeof(int));
    decode_ws_bytes += AlignUp256(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * sizeof(MoEPackedTile));
    decode_ws_bytes += AlignUp256(sizeof(int));
    decode_ws_bytes += AlignUp256(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * st->inter_size * sizeof(T));
    decode_ws_bytes += AlignUp256(static_cast<size_t>(kMoEPackedMaxDecodeRoutes) * st->hidden_size * sizeof(T));
    st->d_decode_ws = IAllocator::MakeUniquePtr<uint8_t>(alloc, decode_ws_bytes, false, nullptr);
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemset(st->d_decode_ws.get(), 0, decode_ws_bytes)));

    ORT_RETURN_IF_ERROR(st->TryAllocateDecodePinned(policy.pinned_bytes));
  }

  if (st->session_controller) {
    st->invocation = std::make_shared<onnxruntime::cuda::QMoEInvocationRecord>();
    st->invocation->layer_index = plan.layer_index;
    st->invocation->node_name = offload_source_desc_.stable_node_id;
    // Keep run controls and the session's overlap guard alive until the doorbell and GPU work finish. Graph replay
    // skips ComputeInternal, so register its stream through the controller hook as well.
    st->session_controller->RegisterGraphReplayHook(
        this, [weak_state = std::weak_ptr<MoEPackedOffloadState>(st)](cudaStream_t stream) {
          if (auto state = weak_state.lock()) state->RegisterInvocation(stream);
          return Status::OK();
        });
    st->session_controller->RegisterOffloadNode();
  }
  MoEPackedDoorbellService::Instance().Register(st.get(), want_hybrid_decode);
  offload_state_ = std::move(st);
  return Status::OK();
}

template <typename T>
Status MoEPacked<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* gate_w = context->Input<Tensor>(2);
  const Tensor* up_w = context->Input<Tensor>(3);
  const Tensor* down_w = context->Input<Tensor>(4);
  const Tensor* router_weights = context->InputCount() > 5 ? context->Input<Tensor>(5) : nullptr;
  const bool relu_squared = activation_type_ == "relu2";
  ORT_RETURN_IF_NOT(input && router_probs && gate_w && down_w,
                    "MoEPacked requires input, router_probs, gate and down weights");
  ORT_RETURN_IF_NOT((router_probs->IsDataType<float>() || router_probs->DataType() == input->DataType()) &&
                        (!router_weights || router_weights->DataType() == router_probs->DataType()),
                    "MoEPacked routing tensors must share FP32 or the activation type");
  ORT_RETURN_IF_NOT(router_weights || k_ <= 16, "MoEPacked softmax routing requires k <= 16");
  ORT_RETURN_IF_NOT(router_weights || (routed_scaling_factor_ == 1.0f && routing_epsilon_ == 0.0f),
                    "MoEPacked routed_scaling_factor and routing_epsilon require router_weights");
  ORT_RETURN_IF_NOT(relu_squared ? !up_w : up_w != nullptr,
                    "MoEPacked swiglu requires up weights; relu2 requires omitted up weights");
  ORT_RETURN_IF((relu_squared || router_weights || router_probs->IsDataType<float>()) &&
                    (gate_w->Location().device.Type() == OrtDevice::CPU ||
                     down_w->Location().device.Type() == OrtDevice::CPU),
                "MoEPacked hybrid offload supports only SwiGLU with activation-type softmax routing");

  const auto& in_shape = input->Shape();
  const auto& gw_shape = gate_w->Shape();
  const auto& dw_shape = down_w->Shape();

  ORT_RETURN_IF_NOT(in_shape.NumDimensions() >= 2, "MoEPacked input must have rank >= 2");
  ORT_RETURN_IF_NOT(gw_shape.NumDimensions() == 3 && dw_shape.NumDimensions() == 3 &&
                        (!up_w || up_w->Shape().NumDimensions() == 3),
                    "MoEPacked expert weights must be 3D [num_experts, out_dim, row_bytes]");

  const size_t last_axis = in_shape.NumDimensions() - 1;
  const int64_t hidden_size = in_shape[last_axis];
  const int64_t num_rows = in_shape.SizeToDimension(last_axis);
  const int64_t num_experts = gw_shape[0];
  const int64_t inter_size = gw_shape[1];

  ORT_RETURN_IF_NOT(hidden_size > 0 && hidden_size <= kMoEPackedMaxDimension &&
                        inter_size > 0 && inter_size <= kMoEPackedMaxDimension,
                    "MoEPacked requires hidden_size and inter_size in 1..", kMoEPackedMaxDimension);
  ORT_RETURN_IF_NOT(num_rows >= 0, "MoEPacked requires a nonnegative row count");
  ORT_RETURN_IF_NOT(num_experts >= k_ && num_experts <= 512, "MoEPacked requires k <= num_experts <= 512");
  ORT_RETURN_IF_NOT(router_probs->Shape() == TensorShape({num_rows, num_experts}),
                    "MoEPacked router_probs must be [num_rows, num_experts]");
  ORT_RETURN_IF_NOT(!router_weights || router_weights->Shape() == router_probs->Shape(),
                    "MoEPacked router_weights must be [num_rows, num_experts]");
  ORT_RETURN_IF_NOT((!up_w || up_w->Shape()[0] == num_experts) && dw_shape[0] == num_experts,
                    "MoEPacked expert count mismatch");
  ORT_RETURN_IF_NOT((!up_w || up_w->Shape()[1] == inter_size) && dw_shape[1] == hidden_size,
                    "MoEPacked expert inner/outer dimension mismatch");
  ORT_RETURN_IF_NOT(hidden_size > 0 && inter_size > 0 &&
                        (hidden_size % 32) == 0 && (inter_size % 32) == 0,
                    "MoEPacked requires hidden_size and inter_size to be multiples of 32");

  const int64_t exp_gate_rb = gguf_quant::GgmlRowBytes(static_cast<int>(gate_type_), hidden_size);
  const int64_t exp_up_rb = relu_squared ? exp_gate_rb : gguf_quant::GgmlRowBytes(static_cast<int>(up_type_), hidden_size);
  const int64_t exp_down_rb = gguf_quant::GgmlRowBytes(static_cast<int>(down_type_), inter_size);
  ORT_RETURN_IF_NOT(exp_gate_rb > 0 && exp_up_rb > 0 && exp_down_rb > 0 &&
                        gw_shape[2] == exp_gate_rb && (!up_w || up_w->Shape()[2] == exp_up_rb) &&
                        dw_shape[2] == exp_down_rb,
                    "MoEPacked expert weight row_bytes mismatch");
  constexpr bool is_bf16 = std::is_same_v<T, BFloat16>;
  ORT_RETURN_IF_NOT(PackedTensorBytesFit(num_rows, hidden_size, sizeof(T)) &&
                        PackedTensorBytesFit(num_rows, num_experts, router_probs->DataType()->Size()) &&
                        PackedTensorBytesFit(num_experts * inter_size, exp_gate_rb, 1) &&
                        PackedTensorBytesFit(num_experts * inter_size, exp_up_rb, 1) &&
                        PackedTensorBytesFit(num_experts * hidden_size, exp_down_rb, 1),
                    "MoEPacked tensor byte size exceeds the supported address range");

  // Hybrid CPU/GPU Offload path when weights were admitted on CPU by QMoESessionController
  if (offload_state_ != nullptr) {
    Tensor* output = context->Output(0, in_shape);
    if (num_rows == 0) return Status::OK();
    MoEPackedOffloadState& st = *offload_state_;
    st.RegisterInvocation(Stream(context));
    // The session policy already resolved the ORT_QMOE_* environment fallbacks with config precedence and validation.
    const float min_expert_weight = st.min_expert_weight;

    // Fast pure-GPU decode path when all experts are resident on the GPU (no pinned host buffers required).
    if (st.resident_slots >= st.num_experts &&
        num_rows <= offload_policy_.async_max_tokens &&
        num_rows <= kMoEPackedMaxDecodeRows &&
        min_expert_weight <= 0.0f) {
      MoEPackedParams p{};
      p.num_rows = static_cast<int>(num_rows);
      p.num_experts = static_cast<int>(num_experts);
      p.hidden_size = static_cast<int>(hidden_size);
      p.inter_size = static_cast<int>(inter_size);
      p.k = static_cast<int>(k_);
      p.normalize_routing_weights = (normalize_routing_weights_ != 0);
      p.bf16_tc_supported = bf16_tc_supported_;
      p.gate_type = static_cast<int>(gate_type_);
      p.up_type = static_cast<int>(up_type_);
      p.down_type = static_cast<int>(down_type_);
      p.gate_row_bytes = exp_gate_rb;
      p.up_row_bytes = exp_up_rb;
      p.down_row_bytes = exp_down_rb;

      size_t ws_bytes = 0;
      ORT_RETURN_IF_ERROR(GetMoEPackedWorkspaceBytes(p, sizeof(T), ws_bytes));
      auto workspace = GetScratchBuffer<uint8_t>(ws_bytes, GetComputeStream(context));
      return LaunchMoEPackedCuda(
          Stream(context),
          is_bf16,
          input->DataRaw(),
          router_probs->DataRaw(),
          st.gpu_gate_bank,
          st.gpu_up_bank,
          st.gpu_down_bank,
          p,
          workspace.get(),
          output->MutableDataRaw());
    }

    // 1. Decode path (num_rows <= min(async_max_tokens, 16), CUDA-graph-capturable)
    if (st.d_decode_ws != nullptr &&
        st.cpu_execution_enabled &&
        st.resident_slots < st.num_experts &&
        num_rows <= offload_policy_.async_max_tokens &&
        num_rows <= kMoEPackedMaxDecodeRows && num_rows * k_ <= kMoEPackedMaxDecodeRoutes &&
        hidden_size <= kMoEPackedMaxHiddenSize) {
      if (st.h_decode_pinned == nullptr) {
        {
          auto& wave_staging = MoEPackedSharedPrefillWaveStaging::ForDevice(GetDeviceId());
          std::lock_guard<std::mutex> wave_lk(wave_staging.mu);
          ORT_RETURN_IF_ERROR(wave_staging.WaitForPreviousUse());
          ORT_RETURN_IF_ERROR(wave_staging.WaitForBanks());
          wave_staging.FreeBanks();
          wave_staging.FreePinnedRouteBuffers();
        }
        ORT_RETURN_IF_ERROR(
            MoEPackedDoorbellService::Instance().EnsureDecodePinned(&st, offload_policy_.pinned_bytes));
        ORT_RETURN_IF(st.h_decode_pinned == nullptr,
                      "MoEPacked persistent decode pinned buffer (", sizeof(MoEPackedDecodePinned),
                      " bytes) exceeds remaining session.moe.offload.pinned_bytes budget (",
                      offload_policy_.pinned_bytes, ").");
      }
      MoEPackedParams p{};
      p.num_rows = static_cast<int>(num_rows);
      p.num_experts = static_cast<int>(num_experts);
      p.hidden_size = static_cast<int>(hidden_size);
      p.inter_size = static_cast<int>(inter_size);
      p.k = static_cast<int>(k_);
      p.normalize_routing_weights = (normalize_routing_weights_ != 0);
      p.bf16_tc_supported = bf16_tc_supported_;
      p.gate_type = static_cast<int>(gate_type_);
      p.up_type = static_cast<int>(up_type_);
      p.down_type = static_cast<int>(down_type_);
      p.gate_row_bytes = exp_gate_rb;
      p.up_row_bytes = exp_up_rb;
      p.down_row_bytes = exp_down_rb;

      return LaunchMoEPackedOffloadDecodeCuda(
          Stream(context),
          is_bf16,
          input->DataRaw(),
          router_probs->DataRaw(),
          st.gpu_gate_bank,
          st.gpu_up_bank,
          st.gpu_down_bank,
          st.d_expert_to_slot.get(),
          st.resident_slots,
          p,
          st.tail_sub_threshold,
          st.tail_sub_min_rank,
          min_expert_weight,
          st.skip_tail_cold_only,
          st.d_decode_ws.get(),
          st.d_decode_pinned,
          output->MutableDataRaw());
    }

    // 2. Prefill path: resident Tensor-Core launch + double-buffered wave streaming of cold experts.
    // Single 8192-row chunks stream each cold expert once per 8K prefill. Resolved working and pinned budgets shrink
    // the chunk and wave sizes; session.moe.offload.prefill_wave_experts caps the experts per wave.
    const auto& policy = offload_policy_;
    const bool needs_streaming = st.resident_slots < st.num_experts;
    int wave_experts = std::clamp<int>(policy.prefill_wave_experts, 1, MoEPackedSharedPrefillWaveStaging::kWaveExperts);
    if (!st.evictable_host_experts.empty() && st.host_cache_slots > 0) {
      wave_experts = std::min(wave_experts, static_cast<int>(st.host_cache_slots));
    }
    int64_t max_chunk = std::min<int64_t>(num_rows, kMoEPackedMaxOffloadChunkRows);
    auto sess_pinned = st.session_controller ? st.session_controller->PinnedBudget() : nullptr;

    MoEPackedParams max_params{};
    max_params.num_experts = static_cast<int>(num_experts);
    max_params.hidden_size = static_cast<int>(hidden_size);
    max_params.inter_size = static_cast<int>(inter_size);
    max_params.k = static_cast<int>(k_);
    max_params.normalize_routing_weights = (normalize_routing_weights_ != 0);
    max_params.bf16_tc_supported = bf16_tc_supported_;
    max_params.gate_type = static_cast<int>(gate_type_);
    max_params.up_type = static_cast<int>(up_type_);
    max_params.down_type = static_cast<int>(down_type_);
    max_params.gate_row_bytes = exp_gate_rb;
    max_params.up_row_bytes = exp_up_rb;
    max_params.down_row_bytes = exp_down_rb;

    auto& wave_staging = MoEPackedSharedPrefillWaveStaging::ForDevice(GetDeviceId());
    std::lock_guard<std::mutex> wave_lk(wave_staging.mu);
    ORT_RETURN_IF_ERROR(wave_staging.WaitForPreviousUse());
    if ((!needs_streaming && (wave_staging.bank_bytes > 0 || wave_staging.device_route_cap > 0)) ||
        (wave_staging.bank_bytes > 0 && wave_staging.bank_session_pinned_budget != sess_pinned)) {
      ORT_RETURN_IF_ERROR(wave_staging.WaitForBanks());
      wave_staging.FreeBanks();
      if (!needs_streaming) {
        wave_staging.FreeWaveDeviceRouteBuffers();
      }
    }
    if (wave_staging.pinned_route_cap > 0 && wave_staging.route_session_pinned_budget != sess_pinned) {
      ORT_RETURN_IF_ERROR(wave_staging.WaitForBanks());
      wave_staging.FreePinnedRouteBuffers();
    }

    auto compute_avail_pinned = [&]() -> size_t {
      size_t limit = static_cast<size_t>(policy.pinned_bytes);
      if (sess_pinned) {
        const auto stats = sess_pinned->GetStats();
        const size_t wave_held = wave_staging.AccountedBytesFor(sess_pinned);
        const size_t other_used = (stats.committed + stats.pending > wave_held)
                                      ? (stats.committed + stats.pending - wave_held)
                                      : 0;
        const size_t sess_rem = (stats.limit > other_used) ? (stats.limit - other_used) : 0;
        limit = std::min(limit, sess_rem);
      }
      if (auto host_ledger = GetCudaHostPinnedMemoryBudget()) {
        const auto hstats = host_ledger->GetStats();
        if (hstats.limit < std::numeric_limits<size_t>::max()) {
          const size_t wave_held =
              wave_staging.bank_accounted_pinned_bytes + wave_staging.route_accounted_pinned_bytes;
          const size_t other_used = (hstats.committed + hstats.pending > wave_held)
                                        ? (hstats.committed + hstats.pending - wave_held)
                                        : 0;
          const size_t host_rem = (hstats.limit > other_used) ? (hstats.limit - other_used) : 0;
          limit = std::min(limit, host_rem);
        }
      }
      return limit;
    };
    size_t avail_pinned = compute_avail_pinned();
    if (needs_streaming) {
      const size_t per_wave_expert = 2 * st.slot_bytes;
      const size_t min_route_bytes =
          MoEPackedSharedPrefillWaveStaging::RoutePinnedBytes(static_cast<size_t>(k_));
      if (avail_pinned < min_route_bytes + per_wave_expert && st.session_controller) {
        MoEPackedDoorbellService::Instance().ReclaimSessionDecodePinned(
            st.session_controller.get(), Stream(context));
        avail_pinned = compute_avail_pinned();
      }
      while (max_chunk > 1) {
        const size_t route_bytes =
            MoEPackedSharedPrefillWaveStaging::RoutePinnedBytes(static_cast<size_t>(max_chunk * k_));
        if (avail_pinned >= route_bytes + per_wave_expert) break;
        max_chunk = (max_chunk + 1) / 2;
      }
      const size_t route_bytes =
          MoEPackedSharedPrefillWaveStaging::RoutePinnedBytes(static_cast<size_t>(max_chunk * k_));
      const size_t fit = avail_pinned > route_bytes ? (avail_pinned - route_bytes) / per_wave_expert : 0;
      ORT_RETURN_IF(fit == 0, "MoEPacked prefill streaming needs ", route_bytes + per_wave_expert,
                    " pinned bytes (two single-expert staging banks and routing buffers), which exceeds "
                    "session.moe.offload.pinned_bytes (",
                    static_cast<size_t>(policy.pinned_bytes), ").");
      wave_experts = static_cast<int>(std::min<size_t>(static_cast<size_t>(wave_experts), fit));
    }

    size_t ws_bytes = 0;
    for (;;) {
      max_params.num_rows = static_cast<int>(max_chunk);
      ORT_RETURN_IF_ERROR(GetMoEPackedWorkspaceBytes(max_params, sizeof(T), ws_bytes));
      const size_t chunk_routes = static_cast<size_t>(max_chunk * k_);
      const size_t gpu_bank_bytes = needs_streaming ? 2 * static_cast<size_t>(wave_experts) * st.slot_bytes : 0;
      const size_t wave_meta_gpu_bytes =
          needs_streaming ? MoEPackedSharedPrefillWaveStaging::WaveRouteDeviceBytes(chunk_routes) : 0;
      const size_t total_working_bytes = ws_bytes + gpu_bank_bytes + wave_meta_gpu_bytes;
      if (policy.device_working_bytes == 0 || total_working_bytes <= policy.device_working_bytes) {
        break;
      }
      if (needs_streaming && wave_experts > 1 &&
          (max_chunk == 1 || gpu_bank_bytes > policy.device_working_bytes / 2)) {
        wave_experts = (wave_experts + 1) / 2;
        continue;
      }
      if (max_chunk > 1) {
        max_chunk = (max_chunk + 1) / 2;
        continue;
      }
      ORT_RETURN_IF(true, "MoEPacked prefill working set (", total_working_bytes,
                    " bytes: one-row scratch and GPU staging banks) exceeds session.moe.offload.device_working_bytes (",
                    policy.device_working_bytes, ").");
    }
    const size_t req_bank_bytes = needs_streaming ? static_cast<size_t>(wave_experts) * st.slot_bytes : 0;
    const size_t chunk_route_cap = static_cast<size_t>(max_chunk * k_);
    const size_t req_route_pinned = MoEPackedSharedPrefillWaveStaging::RoutePinnedBytes(chunk_route_cap);
    const size_t kept_route_pinned =
        MoEPackedSharedPrefillWaveStaging::RoutePinnedBytes(std::max(chunk_route_cap, wave_staging.pinned_route_cap));
    const size_t kept_bank_bytes = std::max(req_bank_bytes, wave_staging.bank_bytes);
    const size_t kept_wave_meta_gpu =
        needs_streaming
            ? MoEPackedSharedPrefillWaveStaging::WaveRouteDeviceBytes(
                  std::max(chunk_route_cap, wave_staging.device_route_cap))
            : 0;
    if (needs_streaming && wave_staging.bank_bytes > req_bank_bytes &&
        ((policy.device_working_bytes > 0 &&
          ws_bytes + 2 * kept_bank_bytes + kept_wave_meta_gpu > policy.device_working_bytes) ||
         (kept_route_pinned + 2 * kept_bank_bytes > avail_pinned))) {
      ORT_RETURN_IF_ERROR(wave_staging.WaitForBanks());
      wave_staging.FreeBanks();
    }
    if (wave_staging.pinned_route_cap > chunk_route_cap &&
        kept_route_pinned + 2 * std::max(req_bank_bytes, wave_staging.bank_bytes) > avail_pinned) {
      ORT_RETURN_IF_ERROR(wave_staging.WaitForPreviousUse());
      ORT_RETURN_IF_ERROR(wave_staging.WaitForBanks());
      wave_staging.FreePinnedRouteBuffers();
    }
    if (needs_streaming && wave_staging.device_route_cap > chunk_route_cap &&
        policy.device_working_bytes > 0 &&
        ws_bytes + 2 * std::max(req_bank_bytes, wave_staging.bank_bytes) + kept_wave_meta_gpu >
            policy.device_working_bytes) {
      ORT_RETURN_IF_ERROR(wave_staging.WaitForBanks());
      wave_staging.FreeWaveDeviceRouteBuffers();
    }
    auto workspace = GetScratchBuffer<uint8_t>(ws_bytes, GetComputeStream(context));
    if (!needs_streaming && avail_pinned < req_route_pinned && st.session_controller) {
      MoEPackedDoorbellService::Instance().ReclaimSessionDecodePinned(
          st.session_controller.get(), Stream(context));
      avail_pinned = compute_avail_pinned();
    }
    const bool use_pinned_routes = needs_streaming || (avail_pinned >= req_route_pinned);
    std::vector<int32_t> pageable_topk_idx;
    std::vector<float> pageable_topk_scales;
    std::vector<int32_t> pageable_res_sorted_routes;
    std::vector<MoEPackedTile> pageable_res_tiles;
    int32_t pageable_res_num_tiles = 0;
    if (use_pinned_routes) {
      ORT_RETURN_IF_ERROR(wave_staging.EnsureRouteCapacity(chunk_route_cap, needs_streaming, sess_pinned));
    } else {
      ORT_RETURN_IF_ERROR(wave_staging.EnsureRouteCapacity(0, false, nullptr));
      pageable_topk_idx.resize(chunk_route_cap);
      pageable_topk_scales.resize(chunk_route_cap);
      pageable_res_sorted_routes.resize(chunk_route_cap);
      pageable_res_tiles.resize(MoEPackedSharedPrefillWaveStaging::MaxTilesForRoutes(chunk_route_cap));
    }
    int32_t* h_topk_idx = use_pinned_routes ? wave_staging.pinned_topk_idx : pageable_topk_idx.data();
    float* h_topk_scales = use_pinned_routes ? wave_staging.pinned_topk_scales : pageable_topk_scales.data();
    int32_t* h_res_sorted_routes =
        use_pinned_routes ? wave_staging.pinned_res_sorted_routes : pageable_res_sorted_routes.data();
    MoEPackedTile* h_res_tiles = use_pinned_routes ? wave_staging.pinned_res_tiles : pageable_res_tiles.data();
    int32_t* h_res_num_tiles = use_pinned_routes ? wave_staging.pinned_res_num_tiles : &pageable_res_num_tiles;
    cudaStream_t stream = Stream(context);
    // Copies enqueued below read the shared pinned buffers after this call returns, including on error paths.
    auto record_staging_use = gsl::finally([&wave_staging, stream, use_pinned_routes]() {
      if (use_pinned_routes) {
        static_cast<void>(wave_staging.RecordUse(stream));
      }
    });

    const uint8_t* in_bytes = static_cast<const uint8_t*>(input->DataRaw());
    const uint8_t* rp_bytes = static_cast<const uint8_t*>(router_probs->DataRaw());
    uint8_t* out_bytes = static_cast<uint8_t*>(output->MutableDataRaw());
    const bool in_warmup = st.InWarmup();

    for (int64_t r0 = 0; r0 < num_rows; r0 += max_chunk) {
      const int chunk_rows = static_cast<int>(std::min<int64_t>(max_chunk, num_rows - r0));
      const int total_routes = chunk_rows * static_cast<int>(k_);
      const bool use_prefill_tiles = is_bf16 && bf16_tc_supported_ && (total_routes >= 2048);
      const int tile_cap = use_prefill_tiles ? 32 : 8;
      // Same bound as GetMoEPackedWorkspaceBytes; every tile holds at least one route.
      const int max_tiles_alloc =
          (total_routes <= 128) ? total_routes : ((total_routes + 7) / 8 + static_cast<int>(num_experts));

      MoEPackedParams chunk_params = max_params;
      chunk_params.num_rows = chunk_rows;

      uint8_t* ws_ptr = workspace.get();
      float* d_topk_scales = reinterpret_cast<float*>(ws_ptr);
      ws_ptr += AlignUp256(static_cast<size_t>(total_routes) * sizeof(float));
      int* d_topk_indices = reinterpret_cast<int*>(ws_ptr);
      ws_ptr += AlignUp256(static_cast<size_t>(total_routes) * sizeof(int));
      int* d_sorted_routes = reinterpret_cast<int*>(ws_ptr);
      ws_ptr += AlignUp256(static_cast<size_t>(total_routes) * sizeof(int));
      MoEPackedTile* d_tiles = reinterpret_cast<MoEPackedTile*>(ws_ptr);
      ws_ptr += AlignUp256(static_cast<size_t>(max_tiles_alloc) * sizeof(MoEPackedTile));
      int* d_num_tiles = reinterpret_cast<int*>(ws_ptr);
      ws_ptr += AlignUp256(sizeof(int));
      void* d_fc1_out = ws_ptr;
      ws_ptr += AlignUp256(static_cast<size_t>(total_routes) * inter_size * sizeof(T));
      void* d_fc2_out = ws_ptr;

      const void* chunk_in = in_bytes + static_cast<size_t>(r0 * hidden_size) * sizeof(T);
      const void* chunk_rp = rp_bytes + static_cast<size_t>(r0 * num_experts) * sizeof(T);
      void* chunk_out = out_bytes + static_cast<size_t>(r0 * hidden_size) * sizeof(T);

      // 2a. Softmax + TopK on GPU and copy indices/scales to host staging
      ORT_RETURN_IF_ERROR(LaunchMoEPackedSoftmaxTopKCuda(
          stream, is_bf16, chunk_rp, d_topk_scales, d_topk_indices, chunk_rows,
          static_cast<int>(num_experts), static_cast<int>(k_), chunk_params.normalize_routing_weights));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
          h_topk_idx, d_topk_indices, static_cast<size_t>(total_routes) * sizeof(int32_t),
          cudaMemcpyDeviceToHost, stream)));
      const bool host_scales = min_expert_weight > 0.0f || (in_warmup && needs_streaming);
      if (host_scales) {
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            h_topk_scales, d_topk_scales, static_cast<size_t>(total_routes) * sizeof(float),
            cudaMemcpyDeviceToHost, stream)));
      }
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaStreamSynchronize(stream)));

      // 2b. Bucket routes by expert on CPU
      std::vector<std::vector<int>> exp_routes(static_cast<size_t>(num_experts));
      bool any_pruned = false;
      for (int r = 0; r < total_routes; ++r) {
        const int e = h_topk_idx[r];
        if (min_expert_weight > 0.0f && (r % static_cast<int>(k_)) > 0 &&
            h_topk_scales[r] < min_expert_weight) {
          h_topk_scales[r] = 0.0f;
          any_pruned = true;
          continue;
        }
        if (e >= 0 && e < num_experts) {
          exp_routes[static_cast<size_t>(e)].push_back(r);
        }
      }

      if (!in_warmup) {
        st.prefill_routes.fetch_add(static_cast<uint64_t>(total_routes), std::memory_order_relaxed);
        st.stream_tiles.fetch_add(1, std::memory_order_relaxed);
        for (int e = 0; e < num_experts; ++e) {
          st.expert_ema[static_cast<size_t>(e)] =
              0.85f * st.expert_ema[static_cast<size_t>(e)] +
              static_cast<float>(exp_routes[static_cast<size_t>(e)].size());
        }
      }

      // 2c. Build & launch GPU-resident expert tiles first
      int res_route_cnt = 0;
      int res_tile_cnt = 0;
      std::vector<int> cold_experts;
      cold_experts.reserve(static_cast<size_t>(num_experts));
      for (int e = 0; e < num_experts; ++e) {
        const auto& rvec = exp_routes[static_cast<size_t>(e)];
        if (rvec.empty()) continue;
        const int slot = st.host_expert_to_slot[static_cast<size_t>(e)];
        if (slot >= 0 && slot < st.resident_slots) {
          const int start_r = res_route_cnt;
          for (int r : rvec) {
            h_res_sorted_routes[res_route_cnt++] = r;
          }
          const int c = static_cast<int>(rvec.size());
          const int n_t = (c + tile_cap - 1) / tile_cap;
          for (int t = 0; t < n_t; ++t) {
            MoEPackedTile tile{};
            tile.expert = slot;
            tile.start_route = start_r + t * tile_cap;
            tile.count = std::min(tile_cap, c - t * tile_cap);
            tile.pad = 0;
            h_res_tiles[res_tile_cnt++] = tile;
          }
        } else {
          cold_experts.push_back(e);
        }
      }
      if (!in_warmup) {
        st.prefill_hit_routes.fetch_add(static_cast<uint64_t>(res_route_cnt), std::memory_order_relaxed);
        // Sort cold experts by descending EMA score so Wave 0 holds the hottest cold experts for promotion
        std::sort(cold_experts.begin(), cold_experts.end(), [&](int a, int b) {
          return st.expert_ema[static_cast<size_t>(a)] > st.expert_ema[static_cast<size_t>(b)];
        });
#if defined(_WIN32)
        if (r0 == 0 && !cold_experts.empty()) {
          MEMORYSTATUSEX mem_stat{};
          mem_stat.dwLength = sizeof(mem_stat);
          if (GlobalMemoryStatusEx(&mem_stat) && mem_stat.ullAvailPhys >= (2ULL << 30)) {
            std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
            ranges.reserve(cold_experts.size() * 3);
            for (int ce : cold_experts) {
              ranges.push_back({const_cast<uint8_t*>(st.host_gate_w + static_cast<size_t>(ce) * st.gate_exp_bytes), st.gate_exp_bytes});
              ranges.push_back({const_cast<uint8_t*>(st.host_up_w + static_cast<size_t>(ce) * st.up_exp_bytes), st.up_exp_bytes});
              ranges.push_back({const_cast<uint8_t*>(st.host_down_w + static_cast<size_t>(ce) * st.down_exp_bytes), st.down_exp_bytes});
            }
            static_cast<void>(PrefetchVirtualMemory(GetCurrentProcess(), ranges.size(), ranges.data(), 0));
          }
        }
#endif
      }

      if (res_tile_cnt > 0) {
        *h_res_num_tiles = res_tile_cnt;
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            d_sorted_routes, h_res_sorted_routes, static_cast<size_t>(res_route_cnt) * sizeof(int),
            cudaMemcpyHostToDevice, stream)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            d_tiles, h_res_tiles, static_cast<size_t>(res_tile_cnt) * sizeof(MoEPackedTile),
            cudaMemcpyHostToDevice, stream)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            d_num_tiles, h_res_num_tiles, sizeof(int), cudaMemcpyHostToDevice, stream)));
        ORT_RETURN_IF_ERROR(LaunchMoEPackedGroupTilesCuda(
            stream, is_bf16, use_prefill_tiles, chunk_in,
            st.gpu_gate_bank, st.gpu_up_bank, st.gpu_down_bank,
            d_sorted_routes, d_tiles, d_num_tiles, res_tile_cnt, chunk_params, d_fc1_out, d_fc2_out));
      }

      // 2d. Double-buffered streaming waves for cold experts (CPU parallel staging + H2D copy_stream + GPU Tensor-Core compute)
      // During dummy-token Warmup(), run at most 1 wave to warm wave buffers and Tensor-Core kernels without faulting 40 GiB.
      const int num_cold = in_warmup ? std::min<int>(static_cast<int>(cold_experts.size()), wave_experts)
                                     : static_cast<int>(cold_experts.size());
      for (size_t i = static_cast<size_t>(num_cold); i < cold_experts.size(); ++i) {
        for (int r : exp_routes[static_cast<size_t>(cold_experts[i])]) {
          h_topk_scales[r] = 0.0f;
          any_pruned = true;
        }
      }
      if (any_pruned) {
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            d_topk_scales, h_topk_scales, static_cast<size_t>(total_routes) * sizeof(float),
            cudaMemcpyHostToDevice, stream)));
      }
      const int num_waves = (num_cold + wave_experts - 1) / wave_experts;
      if (num_waves > 0) {
        ORT_RETURN_IF_ERROR(
            wave_staging.EnsureBankCapacity(static_cast<size_t>(wave_experts) * st.slot_bytes, sess_pinned));
      }
      if (!in_warmup) {
        st.stream_experts.fetch_add(static_cast<uint64_t>(num_cold), std::memory_order_relaxed);
        st.stream_waves.fetch_add(static_cast<uint64_t>(num_waves), std::memory_order_relaxed);
      }

      auto stage_and_enqueue_wave = [&](int w_idx) -> Status {
        const int b = w_idx & 1;
        const int e_begin = w_idx * wave_experts;
        const int w_cnt = std::min(wave_experts, num_cold - e_begin);
        const size_t w_gate_bytes = static_cast<size_t>(w_cnt) * st.gate_exp_bytes;
        const size_t w_up_bytes = static_cast<size_t>(w_cnt) * st.up_exp_bytes;
        const size_t w_down_bytes = static_cast<size_t>(w_cnt) * st.down_exp_bytes;

        if (wave_staging.bank_in_flight[b]) {
          // Only wait for copy_stream's previous H2D read of pinned_bank[b] before staging on the CPU;
          // copy_stream waits on compute_done[b] on the GPU before overwriting gpu_bank[b] and d_wave_*[b].
          ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventSynchronize(wave_staging.h2d_done[b])));
        }
        uint8_t* p_gate = wave_staging.pinned_bank[b];
        uint8_t* p_up = p_gate + w_gate_bytes;
        uint8_t* p_down = p_up + w_up_bytes;

        MoEPackedParallelPool::Instance().ParallelFor(st.cpu_threads, w_cnt * 3, [&](int task_idx) {
          const int local_s = task_idx / 3;
          const int bank_id = task_idx - local_s * 3;
          const int exp = cold_experts[static_cast<size_t>(e_begin + local_s)];
          if (bank_id == 0) {
            std::memcpy(p_gate + static_cast<size_t>(local_s) * st.gate_exp_bytes,
                        st.host_gate_w + static_cast<size_t>(exp) * st.gate_exp_bytes, st.gate_exp_bytes);
          } else if (bank_id == 1) {
            std::memcpy(p_up + static_cast<size_t>(local_s) * st.up_exp_bytes,
                        st.host_up_w + static_cast<size_t>(exp) * st.up_exp_bytes, st.up_exp_bytes);
          } else {
            std::memcpy(p_down + static_cast<size_t>(local_s) * st.down_exp_bytes,
                        st.host_down_w + static_cast<size_t>(exp) * st.down_exp_bytes, st.down_exp_bytes);
          }
        });
        if (!st.is_evictable_host_expert.empty()) {
          std::vector<int32_t> wave_evictable_experts;
          for (int local_s = 0; local_s < w_cnt; ++local_s) {
            const int exp = cold_experts[static_cast<size_t>(e_begin + local_s)];
            if (exp >= 0 && static_cast<size_t>(exp) < st.is_evictable_host_expert.size() &&
                st.is_evictable_host_expert[static_cast<size_t>(exp)] != 0) {
              wave_evictable_experts.push_back(exp);
            }
          }
          if (!wave_evictable_experts.empty()) {
            onnxruntime::cuda::QMoESessionController::EvictUploadedHostPages(
                st.source_desc, wave_evictable_experts);
          }
        }

        int w_route_cnt = 0;
        int w_tile_cnt = 0;
        for (int local_s = 0; local_s < w_cnt; ++local_s) {
          const int exp = cold_experts[static_cast<size_t>(e_begin + local_s)];
          const auto& rvec = exp_routes[static_cast<size_t>(exp)];
          const int start_r = w_route_cnt;
          for (int r : rvec) {
            wave_staging.pinned_sorted_routes[b][w_route_cnt++] = r;
          }
          const int c = static_cast<int>(rvec.size());
          const int n_t = (c + tile_cap - 1) / tile_cap;
          for (int t = 0; t < n_t; ++t) {
            MoEPackedTile tile{};
            tile.expert = local_s;
            tile.start_route = start_r + t * tile_cap;
            tile.count = std::min(tile_cap, c - t * tile_cap);
            tile.pad = 0;
            wave_staging.pinned_tiles[b][w_tile_cnt++] = tile;
          }
        }
        *wave_staging.pinned_num_tiles[b] = w_tile_cnt;

        if (wave_staging.bank_in_flight[b]) {
          ORT_RETURN_IF_ERROR(CUDA_CALL(cudaStreamWaitEvent(wave_staging.copy_stream, wave_staging.compute_done[b], 0)));
        }
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            wave_staging.gpu_bank[b], wave_staging.pinned_bank[b],
            w_gate_bytes + w_up_bytes + w_down_bytes,
            cudaMemcpyHostToDevice, wave_staging.copy_stream)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            wave_staging.d_wave_sorted_routes[b], wave_staging.pinned_sorted_routes[b],
            static_cast<size_t>(w_route_cnt) * sizeof(int),
            cudaMemcpyHostToDevice, wave_staging.copy_stream)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            wave_staging.d_wave_tiles[b], wave_staging.pinned_tiles[b],
            static_cast<size_t>(w_tile_cnt) * sizeof(MoEPackedTile),
            cudaMemcpyHostToDevice, wave_staging.copy_stream)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
            wave_staging.d_wave_num_tiles[b], wave_staging.pinned_num_tiles[b],
            sizeof(int), cudaMemcpyHostToDevice, wave_staging.copy_stream)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventRecord(wave_staging.h2d_done[b], wave_staging.copy_stream)));
        wave_staging.bank_in_flight[b] = true;
        return Status::OK();
      };

      if (num_waves > 0) {
        ORT_RETURN_IF_ERROR(stage_and_enqueue_wave(0));
      }
      for (int w_idx = 0; w_idx < num_waves; ++w_idx) {
        const int b = w_idx & 1;
        const int e_begin = w_idx * wave_experts;
        const int w_cnt = std::min(wave_experts, num_cold - e_begin);
        const size_t w_gate_bytes = static_cast<size_t>(w_cnt) * st.gate_exp_bytes;
        const size_t w_up_bytes = static_cast<size_t>(w_cnt) * st.up_exp_bytes;

        int w_route_cnt = 0;
        for (int local_s = 0; local_s < w_cnt; ++local_s) {
          const int exp = cold_experts[static_cast<size_t>(e_begin + local_s)];
          w_route_cnt += static_cast<int>(exp_routes[static_cast<size_t>(exp)].size());
        }
        const int w_tile_cnt = *wave_staging.pinned_num_tiles[b];
        if (!in_warmup) {
          st.stream_routes.fetch_add(static_cast<uint64_t>(w_route_cnt), std::memory_order_relaxed);
        }

        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaStreamWaitEvent(stream, wave_staging.h2d_done[b], 0)));
        const uint8_t* g_bank = wave_staging.gpu_bank[b];
        const uint8_t* u_bank = g_bank + w_gate_bytes;
        const uint8_t* d_bank = u_bank + w_up_bytes;
        ORT_RETURN_IF_ERROR(LaunchMoEPackedGroupTilesCuda(
            stream, is_bf16, use_prefill_tiles, chunk_in,
            g_bank, u_bank, d_bank,
            wave_staging.d_wave_sorted_routes[b], wave_staging.d_wave_tiles[b], wave_staging.d_wave_num_tiles[b],
            w_tile_cnt, chunk_params, d_fc1_out, d_fc2_out));

        // Adaptive D2D promotion of hottest cold experts already in gpu_bank[b] into coldest unpinned resident slots
        if (!in_warmup && st.is_adaptive && st.resident_slots > 32 && w_idx == 0) {
          const int min_evict_slot = std::max(16, st.resident_slots - 32);
          int promoted_in_wave = 0;
          for (int local_s = 0; local_s < w_cnt && promoted_in_wave < 8; ++local_s) {
            const int exp = cold_experts[static_cast<size_t>(e_begin + local_s)];
            const float cand_score = st.expert_ema[static_cast<size_t>(exp)];
            if (cand_score < 2.0f) continue;
            int best_victim_slot = -1;
            float lowest_score = cand_score - 0.5f;
            for (int vs = min_evict_slot; vs < st.resident_slots; ++vs) {
              const int v_exp = st.slot_to_expert[static_cast<size_t>(vs)];
              if (v_exp >= 0 && exp_routes[static_cast<size_t>(v_exp)].empty() &&
                  st.expert_ema[static_cast<size_t>(v_exp)] < lowest_score) {
                lowest_score = st.expert_ema[static_cast<size_t>(v_exp)];
                best_victim_slot = vs;
              }
            }
            if (best_victim_slot >= 0) {
              const int v_exp = st.slot_to_expert[static_cast<size_t>(best_victim_slot)];
              st.host_expert_to_slot[static_cast<size_t>(v_exp)] = -1;
              st.host_expert_to_slot[static_cast<size_t>(exp)] = best_victim_slot;
              st.slot_to_expert[static_cast<size_t>(best_victim_slot)] = exp;
              ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
                  st.gpu_gate_bank + static_cast<size_t>(best_victim_slot) * st.gate_exp_bytes,
                  g_bank + static_cast<size_t>(local_s) * st.gate_exp_bytes,
                  st.gate_exp_bytes, cudaMemcpyDeviceToDevice, stream)));
              ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
                  st.gpu_up_bank + static_cast<size_t>(best_victim_slot) * st.up_exp_bytes,
                  u_bank + static_cast<size_t>(local_s) * st.up_exp_bytes,
                  st.up_exp_bytes, cudaMemcpyDeviceToDevice, stream)));
              ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
                  st.gpu_down_bank + static_cast<size_t>(best_victim_slot) * st.down_exp_bytes,
                  d_bank + static_cast<size_t>(local_s) * st.down_exp_bytes,
                  st.down_exp_bytes, cudaMemcpyDeviceToDevice, stream)));
              ++promoted_in_wave;
              st.total_promotions.fetch_add(1, std::memory_order_relaxed);
            }
          }
          if (promoted_in_wave > 0) {
            ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
                st.d_expert_to_slot.get(), st.host_expert_to_slot.data(),
                static_cast<size_t>(st.num_experts) * sizeof(int32_t), cudaMemcpyHostToDevice, stream)));
          }
        }

        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventRecord(wave_staging.compute_done[b], stream)));
        // Stage and enqueue Wave w_idx + 1 on CPU + copy_stream IN PARALLEL while GPU executes Wave w_idx on stream!
        if (w_idx + 1 < num_waves) {
          ORT_RETURN_IF_ERROR(stage_and_enqueue_wave(w_idx + 1));
        }
      }

      // 2e. Finalize combine across all routes (resident + streamed waves)
      ORT_RETURN_IF_ERROR(LaunchMoEPackedFinalizeOnlyCuda(
          stream, is_bf16, d_fc2_out, d_topk_scales, chunk_rows,
          static_cast<int>(hidden_size), static_cast<int>(k_), chunk_out));
      if (!use_pinned_routes) {
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaStreamSynchronize(stream)));
      }
    }
    return Status::OK();
  }

  // Process in chunks of at most 2048 rows so workspace stays compact during large prefills.
  const int64_t max_chunk = std::min<int64_t>(num_rows, kMoEPackedMaxChunkRows);
  const int64_t max_routes = max_chunk * k_;
  const int64_t max_tiles = max_routes <= 128 ? max_routes : (max_routes + 7) / 8 + num_experts;
  ORT_RETURN_IF_NOT((hidden_size + 7) / 8 <= GetDeviceProp().maxGridSize[0] &&
                        (inter_size + 7) / 8 <= GetDeviceProp().maxGridSize[0] &&
                        max_tiles <= GetDeviceProp().maxGridSize[1] &&
                        max_chunk <= GetDeviceProp().maxGridSize[1],
                    "MoEPacked dimensions exceed the CUDA grid limits");

  MoEPackedParams max_params{};
  max_params.num_rows = static_cast<int>(max_chunk);
  max_params.num_experts = static_cast<int>(num_experts);
  max_params.hidden_size = static_cast<int>(hidden_size);
  max_params.inter_size = static_cast<int>(inter_size);
  max_params.k = static_cast<int>(k_);
  max_params.normalize_routing_weights = (normalize_routing_weights_ != 0);
  max_params.routed_scaling_factor = routed_scaling_factor_;
  max_params.routing_epsilon = routing_epsilon_;
  max_params.bf16_tc_supported = bf16_tc_supported_;
  max_params.relu_squared = relu_squared;
  max_params.router_is_float = router_probs->IsDataType<float>();
  max_params.gate_type = static_cast<int>(gate_type_);
  max_params.up_type = static_cast<int>(relu_squared ? gate_type_ : up_type_);
  max_params.down_type = static_cast<int>(down_type_);
  max_params.gate_row_bytes = exp_gate_rb;
  max_params.up_row_bytes = exp_up_rb;
  max_params.down_row_bytes = exp_down_rb;

  size_t ws_bytes = 0;
  ORT_RETURN_IF_ERROR(GetMoEPackedWorkspaceBytes(max_params, sizeof(T), ws_bytes));
  Tensor* output = context->Output(0, in_shape);
  if (num_rows == 0) return Status::OK();
  auto workspace = GetScratchBuffer<uint8_t>(ws_bytes, GetComputeStream(context));

  const uint8_t* in_bytes = static_cast<const uint8_t*>(input->DataRaw());
  const uint8_t* rp_bytes = static_cast<const uint8_t*>(router_probs->DataRaw());
  uint8_t* out_bytes = static_cast<uint8_t*>(output->MutableDataRaw());

  for (int64_t r0 = 0; r0 < num_rows;) {
    const int64_t chunk_rows = std::min<int64_t>(kMoEPackedMaxChunkRows, num_rows - r0);
    MoEPackedParams chunk_params = max_params;
    chunk_params.num_rows = static_cast<int>(chunk_rows);
    const size_t input_offset = SafeInt<size_t>(r0) * hidden_size * sizeof(T);
    const size_t router_offset = SafeInt<size_t>(r0) * num_experts * router_probs->DataType()->Size();
    chunk_params.router_weights =
        router_weights ? static_cast<const uint8_t*>(router_weights->DataRaw()) + router_offset : nullptr;
    ORT_RETURN_IF_ERROR(LaunchMoEPackedCuda(
        Stream(context),
        is_bf16,
        in_bytes + input_offset,
        rp_bytes + router_offset,
        gate_w->Data<uint8_t>(),
        up_w ? up_w->Data<uint8_t>() : gate_w->Data<uint8_t>(),
        down_w->Data<uint8_t>(),
        chunk_params,
        workspace.get(),
        out_bytes + input_offset));
    r0 += chunk_rows;
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
