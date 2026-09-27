// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2026 CERN
// Licensed under the MIT License.

// Thrust code needs to be compiled with nvcc
#include <algorithm>
#include <atomic>
#include <bit>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <vector>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _Fill(
    T* output_data,
    T val,
    CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * blockDim.x * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = val;
      id += blockDim.x;
    }
  }
}

template <typename T>
void Fill(cudaStream_t stream, T* output, T value, int64_t count) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _Fill<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(output, value, N);
}

template <typename T>
class ConstantBufferImpl : public IConstantBuffer<T> {
 public:
  ConstantBufferImpl(T val) : val_(val), buffer_(nullptr), count_(0) {
  }
  ~ConstantBufferImpl() {
    if (buffer_)
      cudaFree(buffer_);
  }

  virtual const T* GetBuffer(cudaStream_t stream, size_t count) {
    if (count > count_) {
      if (buffer_) {
        cudaFree(buffer_);
        buffer_ = nullptr;
      }
      CUDA_CALL_THROW(cudaMalloc(&buffer_, count * sizeof(T)));
      count_ = count;

      Fill(stream, buffer_, val_, count);
    }
    return buffer_;
  }

 private:
  T* buffer_;
  size_t count_;
  T val_;
};

template <typename T>
std::unique_ptr<IConstantBuffer<T>> CreateConstantOnes() {
  return std::make_unique<ConstantBufferImpl<T>>(Consts<T>::One);
}

template std::unique_ptr<IConstantBuffer<float>> CreateConstantOnes<float>();
template std::unique_ptr<IConstantBuffer<double>> CreateConstantOnes<double>();
template std::unique_ptr<IConstantBuffer<half>> CreateConstantOnes<half>();
template std::unique_ptr<IConstantBuffer<BFloat16>> CreateConstantOnes<BFloat16>();
#if !defined(DISABLE_FLOAT8_TYPES)
template std::unique_ptr<IConstantBuffer<Float8E4M3FN>> CreateConstantOnes<Float8E4M3FN>();
template std::unique_ptr<IConstantBuffer<Float8E5M2>> CreateConstantOnes<Float8E5M2>();
#endif

namespace {

// Switches the calling thread to the relaxed stream capture mode, and restores its previous mode on destruction.
class ScopedRelaxedStreamCaptureMode {
 public:
  ScopedRelaxedStreamCaptureMode() {
    CUDA_CALL_THROW(cudaThreadExchangeStreamCaptureMode(&mode_));
  }
  ~ScopedRelaxedStreamCaptureMode() {
    cudaThreadExchangeStreamCaptureMode(&mode_);
  }
  ScopedRelaxedStreamCaptureMode(const ScopedRelaxedStreamCaptureMode&) = delete;
  ScopedRelaxedStreamCaptureMode& operator=(const ScopedRelaxedStreamCaptureMode&) = delete;

 private:
  cudaStreamCaptureMode mode_ = cudaStreamCaptureModeRelaxed;
};

// Makes `device_id` the current device of the calling thread, and restores its previous device on destruction.
class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int device_id) {
    CUDA_CALL_THROW(cudaGetDevice(&previous_device_id_));
    if (device_id != previous_device_id_) {
      CUDA_CALL_THROW(cudaSetDevice(device_id));
      changed_ = true;
    }
  }
  ~ScopedCudaDevice() {
    if (changed_) {
      cudaSetDevice(previous_device_id_);
    }
  }
  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

 private:
  int previous_device_id_ = -1;
  bool changed_ = false;
};

}  // namespace

// Constant buffer shared by all the sessions, streams and threads that use the same device, as in the CUDA plugin
// execution provider (see GetConstOnesBufferForDevice() in core/providers/cuda/plugin/cuda_kernel_adapter.h):
//   - a buffer is allocated and filled on an internal stream, and that stream is synchronised before the buffer is
//     published: a published buffer is always complete, so any stream can use it without further synchronisation,
//     and GetBuffer() does not make any CUDA call when the current buffer is large enough;
//   - the buffer and its size are published together, as an immutable snapshot, through an atomic pointer: GetBuffer()
//     is lock free unless the buffer needs to grow, and growing is serialised by a mutex;
//   - a buffer is never freed while the object is alive, because kernels queued in other streams, or CUDA graphs
//     captured by other sessions, may still be using it; to bound the number of buffers, and the memory they use to
//     less than twice the size of the largest one, the size is rounded up to a power of two;
//   - the buffer can grow while this or another thread is capturing a CUDA graph; allocating memory and synchronising
//     a stream are not allowed during a capture in the global or thread-local capture modes, so the calling thread is
//     switched to the relaxed mode while the buffer grows; the internal stream is not part of any capture, so the
//     allocation and the fill are executed immediately and are not recorded in the graph.
template <typename T>
class SharedConstantBufferImpl : public IConstantBuffer<T> {
 public:
  SharedConstantBufferImpl(int device_id, T value) : device_id_(device_id), value_(value) {
  }

  ~SharedConstantBufferImpl() override {
    // This is only destroyed at process exit, or when the plugin library is unloaded, when the CUDA runtime may already
    // have been shut down: errors are ignored.
    for (const auto& buffer : buffers_) {
      cudaFree(buffer->data());
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  const T* GetBuffer(cudaStream_t /* stream */, size_t count) override {
    const std::span<T>* buffer = current_.load(std::memory_order_acquire);
    if (buffer != nullptr && buffer->size() >= count) {
      return buffer->data();
    }
    return Grow(count);
  }

 private:
  const T* Grow(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);

    // another thread may have grown the buffer while this one was waiting for the lock
    const std::span<T>* current = current_.load(std::memory_order_acquire);
    if (current != nullptr && current->size() >= count) {
      return current->data();
    }

    // _Fill() indexes the buffer with a CUDA_LONG; round the size up to a power of two, or to kMaxCount above the
    // largest power of two that fits, so that the result of std::bit_ceil() is never larger than kMaxCount
    constexpr size_t kMaxCount = static_cast<size_t>(std::numeric_limits<CUDA_LONG>::max());
    constexpr size_t kMaxPowerOfTwo = std::bit_floor(kMaxCount);
    static_assert(kMaxCount <= std::numeric_limits<size_t>::max() / sizeof(T),
                  "the size in bytes of the largest constant buffer does not fit in a size_t");
    ORT_ENFORCE(count <= kMaxCount, "Too many elements requested from a constant buffer: ", count);
    const size_t new_count = count > kMaxPowerOfTwo ? kMaxCount : std::bit_ceil(std::max<size_t>(count, 1));
    const size_t bytes = new_count * sizeof(T);  // cannot overflow, see the static_assert above

    ScopedRelaxedStreamCaptureMode capture_mode;
    ScopedCudaDevice device(device_id_);

    if (stream_ == nullptr) {
      int memory_pools_supported = 0;
      CUDA_CALL_THROW(cudaDeviceGetAttribute(&memory_pools_supported, cudaDevAttrMemoryPoolsSupported, device_id_));
      if (memory_pools_supported) {
        CUDA_CALL_THROW(cudaDeviceGetDefaultMemPool(&pool_, device_id_));
      }
      CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    // Add an empty entry to buffers_ before allocating the memory, so that nothing can throw between the allocation and
    // the point where the destructor takes ownership of it.
    buffers_.push_back(std::make_unique<std::span<T>>());
    std::span<T>* buffer = buffers_.back().get();

    // A stream-ordered allocation from the default memory pool of the device does not synchronise with the work
    // queued in other streams; use cudaMalloc if the device does not support memory pools.
    void* data = nullptr;
    if (pool_ != nullptr) {
      CUDA_CALL_THROW(cudaMallocFromPoolAsync(&data, bytes, pool_, stream_));
    } else {
      CUDA_CALL_THROW(cudaMalloc(&data, bytes));
    }
    *buffer = std::span<T>(static_cast<T*>(data), new_count);

    Fill(stream_, buffer->data(), value_, static_cast<int64_t>(buffer->size()));
    CUDA_CALL_THROW(cudaGetLastError());
    // wait for the allocation and the fill to complete: after this, any stream can use the buffer
    CUDA_CALL_THROW(cudaStreamSynchronize(stream_));

    current_.store(buffer, std::memory_order_release);
    return buffer->data();
  }

  const int device_id_;
  const T value_;
  std::atomic<const std::span<T>*> current_{nullptr};   // the largest buffer, fully initialised
  std::mutex mutex_;                                    // serialises Grow()
  std::vector<std::unique_ptr<std::span<T>>> buffers_;  // all the buffers, freed in the destructor
  cudaStream_t stream_ = nullptr;                       // internal stream used to allocate and fill the buffers
  cudaMemPool_t pool_ = nullptr;                        // default memory pool of the device, if supported
};

template <typename T>
std::unique_ptr<IConstantBuffer<T>> CreateSharedConstantOnes(int device_id) {
  return std::make_unique<SharedConstantBufferImpl<T>>(device_id, Consts<T>::One);
}

template std::unique_ptr<IConstantBuffer<float>> CreateSharedConstantOnes<float>(int);
template std::unique_ptr<IConstantBuffer<double>> CreateSharedConstantOnes<double>(int);
template std::unique_ptr<IConstantBuffer<half>> CreateSharedConstantOnes<half>(int);
template std::unique_ptr<IConstantBuffer<BFloat16>> CreateSharedConstantOnes<BFloat16>(int);
#if !defined(DISABLE_FLOAT8_TYPES)
template std::unique_ptr<IConstantBuffer<Float8E4M3FN>> CreateSharedConstantOnes<Float8E4M3FN>(int);
template std::unique_ptr<IConstantBuffer<Float8E5M2>> CreateSharedConstantOnes<Float8E5M2>(int);
#endif

#define SPECIALIZED_FILL(T) \
  template void Fill<T>(cudaStream_t stream, T * output, T value, int64_t count);

SPECIALIZED_FILL(int8_t)
SPECIALIZED_FILL(uint8_t)
SPECIALIZED_FILL(bool)
SPECIALIZED_FILL(int16_t)
SPECIALIZED_FILL(int32_t)
SPECIALIZED_FILL(int64_t)
SPECIALIZED_FILL(float)
SPECIALIZED_FILL(double)
SPECIALIZED_FILL(__half)
SPECIALIZED_FILL(BFloat16)
#if !defined(DISABLE_FLOAT8_TYPES)
SPECIALIZED_FILL(Float8E4M3FN)
SPECIALIZED_FILL(Float8E5M2)
#endif

}  // namespace cuda
}  // namespace onnxruntime
