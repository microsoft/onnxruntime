#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/platform/env_var_utils.h"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct SoftmaxEnvVarSingleton {
  static SoftmaxEnvVarSingleton& Instance() {
    static SoftmaxEnvVarSingleton instance;
    return instance;
  }

  SoftmaxEnvVarSingleton(const SoftmaxEnvVarSingleton&) = delete;

  SoftmaxEnvVarSingleton& operator=(const SoftmaxEnvVarSingleton&) = delete;

  bool IsOnlineSoftmaxEnabled() {
    return online_soft_max_enabled;
  }

 private:
  bool online_soft_max_enabled;

  SoftmaxEnvVarSingleton() {
    online_soft_max_enabled = ParseEnvironmentVariableWithDefault<bool>(attention::kEnableOnlineSoftmax, false);
  }

  ~SoftmaxEnvVarSingleton() {}
};

/////////////////////////////////////////////////////////////////////////////
// Shared section for Online softmax reduce, the MD struct are integrated from
// https://github.com/NVIDIA/online-softmax/blob/master/online_softmax_benchmark.cu
// Please see paper "Online normalizer calculation for softmax (https://arxiv.org/abs/1805.02867)" .
/////////////////////////////////////////////////////////////////////////////
struct __align__(8) MD {
  float m;
  float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b) {
  bool a_bigger = (a.m > b.m);
  MD bigger_m = a_bigger ? a : b;
  MD smaller_m = a_bigger ? b : a;
  MD res;
  res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
  res.m = bigger_m.m;
  return res;
}

/////////////////////////////////////////////////////////////////////////////
// ComputeSoftmax() and related kernels.  Some device functions are shared
// with ComputeSoftmaxWithMask1D().
/////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ __forceinline__ float CalcNoMaskSoftmaxInput(const int base_index,
                                                        const int seq_idx,
                                                        const int all_sequence_length,
                                                        const int sequence_length,
                                                        const int valid_end,
                                                        const int valid_start,
                                                        const T* add_before_softmax,
                                                        const T* input,
                                                        bool is_unidirectional,
                                                        bool& is_valid,
                                                        int& end) {
  is_valid = false;

  // Update end position for unidirectional.
  end = valid_end;
  if (is_unidirectional) {
    int end_unid = all_sequence_length - sequence_length + (blockIdx.x % sequence_length) + 1;
    if (end_unid <= valid_start) {
      // In this situation, mask of [0, end_unid) and [valid_start, valid_end) has -10000,
      //              and [end_unid, valid_start) and [valid_end, all_seq_len) has -20000.
      // So [0, end_unid) will also have value after softmax.
      // KEEP SMALL KERNEL CODE LOGIC HERE as COMMENT
      is_valid = seq_idx < end_unid;
    } else {
      end = min(valid_end, end_unid);
    }
  }

  const int index = base_index + seq_idx;
  is_valid = is_valid || (seq_idx >= valid_start && seq_idx < end);

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  const bool no_add = (add_before_softmax == nullptr);
  return is_valid ? (no_add ? float(input[index]) : float(input[index] + add_before_softmax[index]))
                  : float(-CUDART_INF_F);
}

template <typename T, unsigned TPB>
__device__ inline void SoftmaxSmall(const int all_sequence_length,
                                    const int sequence_length,
                                    const int valid_end,
                                    const int valid_start,
                                    const T* add_before_softmax,
                                    const T* input,
                                    T* output,
                                    bool is_unidirectional) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;
  bool is_valid;
  int end;
  const float input_data = CalcNoMaskSoftmaxInput(
      offset, threadIdx.x, all_sequence_length, sequence_length, valid_end, valid_start,
      add_before_softmax, input, is_unidirectional, is_valid, end);
  const auto max = BlockReduce(tmp_storage).Reduce(input_data, cub::Max(), end);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_exp(0.f);
  if (is_valid) {
    thread_data_exp = expf(input_data - max_block);
  }

  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, cub::Sum(), end);

  // Store value of 1.0/sum.
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  // threadIdx.x might be larger than all_sequence_length due to alignment to 32x.
  if (threadIdx.x < all_sequence_length) {
    output[offset + threadIdx.x] = T(thread_data_exp * sum_reverse_block);
  }
}

template <typename T, unsigned TPB>
__device__ __forceinline__ void OnlineSoftmaxDeviceFunction(const int all_sequence_length,
                                                            const int sequence_length,
                                                            const int valid_end,
                                                            const int valid_start,
                                                            const T* add_before_softmax,
                                                            const T* input,
                                                            T* output,
                                                            bool is_unidirectional) {
  using BlockReduce = cub::BlockReduce<MD, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  __shared__ MD md_total;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  const int base_index = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;

  bool is_valid;
  int end;
  MD md_partial;
  md_partial.m = -FLT_MAX;  //-CUDART_INF_F;
  md_partial.d = 1.0F;
  for (int seq_idx = threadIdx.x; seq_idx < all_sequence_length; seq_idx += TPB) {
    MD new_elem;
    new_elem.d = 1.0F;
    new_elem.m = CalcNoMaskSoftmaxInput(
        base_index, seq_idx, all_sequence_length, sequence_length, valid_end, valid_start,
        add_before_softmax, input, is_unidirectional, is_valid, end);
    md_partial = reduce_md_op(md_partial, new_elem);
  }

  const auto md = BlockReduce(tmp_storage).Reduce(md_partial, reduce_md_op);

  if (threadIdx.x == 0) {
    md_total = md;
  }
  __syncthreads();

  float d_total_inverse = __fdividef(1.0F, md_total.d);
  for (int seq_idx = threadIdx.x; seq_idx < all_sequence_length; seq_idx += TPB) {
    auto input_value = CalcNoMaskSoftmaxInput(
        base_index, seq_idx, all_sequence_length, sequence_length, valid_end, valid_start,
        add_before_softmax, input, is_unidirectional, is_valid, end);
    output[base_index + seq_idx] = T(__expf(input_value - md_total.m) * d_total_inverse);
  }
}

template <typename T, unsigned TPB>
__global__ void OnlineSoftmaxKernel(const int all_sequence_length,
                                    const int sequence_length,
                                    const int valid_end,
                                    const int valid_start,
                                    const T* add_before_softmax,
                                    const T* input,
                                    T* output,
                                    bool is_unidirectional) {
  OnlineSoftmaxDeviceFunction<T, TPB>(all_sequence_length, sequence_length, valid_end, valid_start, add_before_softmax,
                                      input, output, is_unidirectional);
}

template <typename T, unsigned TPB>
__global__ void SoftmaxKernelSmall(const int all_sequence_length,
                                   const int sequence_length,
                                   const T* add_before_softmax,
                                   const T* input,
                                   T* output,
                                   bool is_unidirectional) {
  SoftmaxSmall<T, TPB>(all_sequence_length, sequence_length, all_sequence_length, 0,
                       add_before_softmax, input, output, is_unidirectional);
}

template <typename T, unsigned TPB>
__device__ inline void Softmax(const int all_sequence_length,
                               const int sequence_length,
                               const int valid_end,
                               const int valid_start,
                               const T* add_before_softmax,
                               const T* input,
                               T* output) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  float thread_data_max(-CUDART_INF_F);

  const bool no_add = (add_before_softmax == nullptr);

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;
  for (int i = threadIdx.x; i < valid_end; i += TPB) {
    if (i >= valid_start) {
      const int index = offset + i;
      float input_at_idx = no_add ? float(input[index]) : float(input[index] + add_before_softmax[index]);
      if (thread_data_max < input_at_idx) {
        thread_data_max = input_at_idx;
      }
    }
  }

  const auto max = BlockReduce(tmp_storage).Reduce(thread_data_max, cub::Max());

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_sum(0.f);
  for (int i = threadIdx.x; i < valid_end; i += TPB) {
    if (i >= valid_start) {
      const int index = offset + i;
      float val = no_add ? input[index] : input[index] + add_before_softmax[index];
      thread_data_sum += expf(val - max_block);
    }
  }

  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_sum, cub::Sum());
  if (threadIdx.x == 0) {
    sum_reverse_block = 1.f / sum;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < all_sequence_length; i += TPB) {
    const int index = offset + i;
    float input_at_idx = no_add ? float(input[index]) : float(input[index] + add_before_softmax[index]);
    const float val = (i >= valid_start && i < valid_end) ? expf(input_at_idx - max_block) * sum_reverse_block : 0.f;
    output[index] = T(val);
  }
}

template <typename T, unsigned TPB>
__global__ void SoftmaxKernel(const int all_sequence_length,
                              const int sequence_length,
                              const T* add_before_softmax,
                              const T* input,
                              T* output) {
  Softmax<T, TPB>(all_sequence_length, sequence_length, all_sequence_length, 0,
                  add_before_softmax, input, output);
}

template <typename T>
Status ComputeSoftmax(cudaStream_t stream,
                      const int all_sequence_length,
                      const int sequence_length,
                      const int batch_size,
                      const int num_heads,
                      const T* add_before_softmax,
                      const T* input,
                      T* output,
                      bool is_unidirectional) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);

  bool need_call_online_softmax = SoftmaxEnvVarSingleton::Instance().IsOnlineSoftmaxEnabled();
  if (!need_call_online_softmax) {
    if (all_sequence_length <= 32) {
      const int blockSize = 32;
      SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
          all_sequence_length, sequence_length, add_before_softmax, input, output, is_unidirectional);
    } else if (all_sequence_length <= 64) {
      const int blockSize = 64;
      SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
          all_sequence_length, sequence_length, add_before_softmax, input, output, is_unidirectional);
    } else if (all_sequence_length <= 128) {
      const int blockSize = 128;
      SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
          all_sequence_length, sequence_length, add_before_softmax, input, output, is_unidirectional);
    } else if (all_sequence_length <= 256) {
      const int blockSize = 256;
      SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
          all_sequence_length, sequence_length, add_before_softmax, input, output, is_unidirectional);
    } else if (all_sequence_length <= 512) {
      const int blockSize = 512;
      SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
          all_sequence_length, sequence_length, add_before_softmax, input, output, is_unidirectional);
    } else if (all_sequence_length <= 1024) {
      const int blockSize = 1024;
      SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
          all_sequence_length, sequence_length, add_before_softmax, input, output, is_unidirectional);
    } else if (!is_unidirectional) {
      const int blockSize = 1024;
      SoftmaxKernel<T, blockSize><<<grid, blockSize, 0, stream>>>(
          all_sequence_length, sequence_length, add_before_softmax, input, output);
    } else {
      need_call_online_softmax = true;
    }
  }

  if (need_call_online_softmax) {
    const int blockSize = 256;
    OnlineSoftmaxKernel<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, sequence_length, all_sequence_length, 0, add_before_softmax, input, output, is_unidirectional);
  }

  return CUDA_CALL(cudaGetLastError());
}

#define InstantiateComputeSoftmax(T) \
  template Status ComputeSoftmax(    \
      cudaStream_t stream,           \
      const int all_sequence_length, \
      const int sequence_length,     \
      const int batch_size,          \
      const int num_heads,           \
      const T* add_before_softmax,   \
      const T* input,                \
      T* output,                     \
      bool is_unidirectional)

InstantiateComputeSoftmax(float);
InstantiateComputeSoftmax(__half);

/////////////////////////////////////////////////////////////////////////////
// ComputeSoftmaxWithMask1D() related
/////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ void CalcStartEndPosition(int& start_position,
                                                     int& end_position,
                                                     const int all_sequence_length,
                                                     const int sequence_length,
                                                     const int* mask_end,
                                                     const int* mask_start) {
  const int batch = blockIdx.y;
  start_position = mask_start != nullptr ? max(0, mask_start[batch]) : 0;
  end_position = min(all_sequence_length, mask_end[batch]);

  // Attend to no word has same effect as attend to all words. This is added to get parity with CPU result.
  if (start_position >= end_position) {
    start_position = 0;
    end_position = all_sequence_length;
  }
}

template <typename T, unsigned TPB>
__global__ void MaskedSoftmaxKernelSmall(const int all_sequence_length,
                                         const int sequence_length,
                                         const int* mask_end,
                                         const int* mask_start,
                                         const T* add_before_softmax,
                                         const T* input,
                                         T* output,
                                         bool is_unidirectional) {
  __shared__ int start_position;
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    CalcStartEndPosition(start_position, end_position, all_sequence_length, sequence_length, mask_end, mask_start);
  }
  __syncthreads();

  SoftmaxSmall<T, TPB>(all_sequence_length, sequence_length, end_position, start_position,
                       add_before_softmax, input, output, is_unidirectional);
}

template <typename T, unsigned TPB>
__global__ void MaskedSoftmaxKernel(const int all_sequence_length,
                                    const int sequence_length,
                                    const int* mask_end,
                                    const int* mask_start,
                                    const T* add_before_softmax,
                                    const T* input,
                                    T* output) {
  __shared__ int start_position;
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    CalcStartEndPosition(start_position, end_position, all_sequence_length, sequence_length, mask_end, mask_start);
  }
  __syncthreads();

  Softmax<T, TPB>(all_sequence_length, sequence_length, end_position, start_position,
                  add_before_softmax, input, output);
}

template <typename T, unsigned TPB>
__global__ void OnlineMaskedSoftmaxKernel(const int all_sequence_length,
                                          const int sequence_length,
                                          const int* mask_end,
                                          const int* mask_start,
                                          const T* add_before_softmax,
                                          const T* input,
                                          T* output,
                                          bool is_unidirectional) {
  __shared__ int start_position;
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    CalcStartEndPosition(start_position, end_position, all_sequence_length, sequence_length, mask_end, mask_start);
  }
  __syncthreads();

  OnlineSoftmaxDeviceFunction<T, TPB>(all_sequence_length, sequence_length, end_position, start_position,
                                      add_before_softmax, input, output, is_unidirectional);
}

template <typename T>
Status ComputeSoftmaxWithMask1D(cudaStream_t stream,
                                const int all_sequence_length,
                                const int sequence_length,
                                const int batch_size,
                                const int num_heads,
                                const int* mask_index,
                                const int* mask_start,
                                const T* add_before_softmax,
                                const T* input,
                                T* output,
                                const bool is_unidirectional) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);

  bool need_call_online_softmax = SoftmaxEnvVarSingleton::Instance().IsOnlineSoftmaxEnabled();
  if (!need_call_online_softmax) {
    if (all_sequence_length <= 32) {
      const int blockSize = 32;
      MaskedSoftmaxKernelSmall<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                           add_before_softmax, input, output, is_unidirectional);
    } else if (all_sequence_length <= 64) {
      const int blockSize = 64;
      MaskedSoftmaxKernelSmall<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                           add_before_softmax, input, output, is_unidirectional);
    } else if (all_sequence_length <= 128) {
      const int blockSize = 128;
      MaskedSoftmaxKernelSmall<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                           add_before_softmax, input, output, is_unidirectional);
    } else if (all_sequence_length <= 256) {
      const int blockSize = 256;
      MaskedSoftmaxKernelSmall<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                           add_before_softmax, input, output, is_unidirectional);
    } else if (all_sequence_length <= 512) {
      const int blockSize = 512;
      MaskedSoftmaxKernelSmall<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                           add_before_softmax, input, output, is_unidirectional);
    } else if (all_sequence_length <= 1024) {
      const int blockSize = 1024;
      MaskedSoftmaxKernelSmall<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                           add_before_softmax, input, output, is_unidirectional);
    } else if (!is_unidirectional) {
      const int blockSize = 1024;
      MaskedSoftmaxKernel<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                           add_before_softmax, input, output);
    } else {
      need_call_online_softmax = true;
    }
  }

  if (need_call_online_softmax) {
    const int blockSize = 256;
    OnlineMaskedSoftmaxKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length, mask_index, mask_start,
                                         add_before_softmax, input, output, is_unidirectional);
  }

  return CUDA_CALL(cudaGetLastError());
}

#define InstantiateComputeSoftmaxWithMask1D(T) \
  template Status ComputeSoftmaxWithMask1D(    \
      cudaStream_t stream,                     \
      const int all_sequence_length,           \
      const int sequence_length,               \
      const int batch_size,                    \
      const int num_heads,                     \
      const int* mask_index,                   \
      const int* mask_start,                   \
      const T* add_before_softmax,             \
      const T* input,                          \
      T* output,                               \
      const bool is_unidirectional)

InstantiateComputeSoftmaxWithMask1D(float);
InstantiateComputeSoftmaxWithMask1D(__half);

/////////////////////////////////////////////////////////////////////////////
// ComputeSoftmaxWithRawMask() and its related kernels
/////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ __forceinline__ float CalcRawMaskedSoftmaxInput(const int base_index,
                                                           const int seq_idx,
                                                           const int all_sequence_length,
                                                           const int sequence_length,
                                                           const int* attention_mask,
                                                           const bool* key_padding_mask,
                                                           const T* add_before_softmax,
                                                           const T* input,
                                                           const bool is_unidirectional,
                                                           const float rsqrt_head_size,
                                                           const int mask_dimension,
                                                           const int max_sequence_length,
                                                           const float mask_filter_value) {
  const int index = base_index + seq_idx;
  float updated_input_value = float(input[index]) * rsqrt_head_size;

  const int sequence_index = blockIdx.x % sequence_length;
  if (is_unidirectional) {
    int from_index = all_sequence_length - sequence_length + sequence_index;  // offset in all sequence length.
    if (seq_idx > from_index) {
      updated_input_value = mask_filter_value;
    }
  }

  int mask_offset = 0;
  const int batch_index = blockIdx.y;
  if (mask_dimension == 2) {
    mask_offset = batch_index * all_sequence_length + seq_idx;
  } else if (mask_dimension == 3) {
    mask_offset = (batch_index * sequence_length + sequence_index) * all_sequence_length + seq_idx;
  } else if (mask_dimension == 4) {
    int from_index = all_sequence_length - sequence_length + sequence_index;
    mask_offset = (batch_index * max_sequence_length + from_index) * max_sequence_length + seq_idx;
  }

  if (nullptr == key_padding_mask) {
    const int& mask = attention_mask[mask_offset];
    if (mask == 0)
      updated_input_value += mask_filter_value;
  } else {
    const bool mask = key_padding_mask[mask_offset];
    if (mask) {
      updated_input_value = -CUDART_INF_F;
    }
  }

  if (add_before_softmax != nullptr) {
    updated_input_value += float(add_before_softmax[index]);
  }

  return updated_input_value;
}

// using shared mem store masked value, all_sequence_length will be limited
template <typename T, int TPB>
__global__ void OnlineSoftmaxWithRawMaskKernel(const int all_sequence_length,
                                               const int sequence_length,
                                               const int* attention_mask,  // 2D, 3D or 4D attention mask
                                               const bool* key_padding_mask,
                                               const T* add_before_softmax,
                                               const T* input,
                                               T* output,
                                               const bool is_unidirectional,
                                               const float rsqrt_head_size,
                                               const int mask_dimension,
                                               const int max_sequence_length,
                                               const bool skip_softmax,
                                               const float mask_filter_value) {
  extern __shared__ float cached_data[];  // float[all_sequence_length]
  using BlockReduce = cub::BlockReduce<MD, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  __shared__ MD md_total;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  int base_index = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;

  MD md_partial;
  md_partial.m = -FLT_MAX;  //-CUDART_INF_F;
  md_partial.d = 1.0F;
  for (int seq_idx = threadIdx.x; seq_idx < all_sequence_length; seq_idx += TPB) {
    const float input_value = CalcRawMaskedSoftmaxInput(
        base_index, seq_idx, all_sequence_length, sequence_length, attention_mask, key_padding_mask,
        add_before_softmax, input, is_unidirectional, rsqrt_head_size, mask_dimension,
        max_sequence_length, mask_filter_value);

    if (skip_softmax) {
      output[base_index + seq_idx] = T(input_value);
    } else {
      cached_data[seq_idx] = input_value;
      MD new_elem;
      new_elem.m = input_value;
      new_elem.d = 1.0F;
      md_partial = reduce_md_op(md_partial, new_elem);
    }
  }

  if (skip_softmax) {
    return;
  }

  const auto md = BlockReduce(tmp_storage).Reduce(md_partial, reduce_md_op);

  if (threadIdx.x == 0) {
    md_total = md;
  }
  __syncthreads();

  float d_total_inverse = __fdividef(1.0F, md_total.d);
  for (int seq_idx = threadIdx.x; seq_idx < all_sequence_length; seq_idx += TPB) {
    output[base_index + seq_idx] = T(__expf(cached_data[seq_idx] - md_total.m) * d_total_inverse);
  }
}

// no shared mem store masked value, long_sequence_length supportted
template <typename T, int TPB>
__global__ void OnlineSoftmaxWithRawMaskLongSequenceKernel(const int all_sequence_length,
                                                           const int sequence_length,
                                                           const int* attention_mask,
                                                           const bool* key_padding_mask,
                                                           const T* add_before_softmax,
                                                           const T* input,
                                                           T* output,
                                                           const bool is_unidirectional,
                                                           const float rsqrt_head_size,
                                                           const int mask_dimension,
                                                           const int max_sequence_length,
                                                           const float mask_filter_value) {
  using BlockReduce = cub::BlockReduce<MD, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  __shared__ MD md_total;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  int base_index = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;

  MD md_partial;
  md_partial.m = -CUDART_INF_F;
  md_partial.d = 1.0F;
  for (int seq_idx = threadIdx.x; seq_idx < all_sequence_length; seq_idx += TPB) {
    MD new_elem;
    new_elem.d = 1.0F;
    new_elem.m = CalcRawMaskedSoftmaxInput(
        base_index, seq_idx, all_sequence_length, sequence_length, attention_mask, key_padding_mask,
        add_before_softmax, input, is_unidirectional, rsqrt_head_size, mask_dimension,
        max_sequence_length, mask_filter_value);
    md_partial = reduce_md_op(md_partial, new_elem);
  }

  const auto md = BlockReduce(tmp_storage).Reduce(md_partial, reduce_md_op);

  if (threadIdx.x == 0) {
    md_total = md;
  }
  __syncthreads();

  float d_total_inverse = __fdividef(1.0F, md_total.d);
  for (int seq_idx = threadIdx.x; seq_idx < all_sequence_length; seq_idx += TPB) {
    const float input_value = CalcRawMaskedSoftmaxInput(
        base_index, seq_idx, all_sequence_length, sequence_length, attention_mask, key_padding_mask,
        add_before_softmax, input, is_unidirectional, rsqrt_head_size, mask_dimension,
        max_sequence_length, mask_filter_value);
    output[base_index + seq_idx] = T(__expf(input_value - md_total.m) * d_total_inverse);
  }
}

template <typename T, unsigned TPB>
__device__ inline void SoftmaxWithRawMaskSmall(const int all_sequence_length,
                                               const int sequence_length,
                                               const int* attention_mask,  // 2D, 3D or 4D attention mask
                                               const bool* key_padding_mask,
                                               const T* add_before_softmax,
                                               const T* input,
                                               T* output,
                                               const bool is_unidirectional,
                                               const float rsqrt_head_size,
                                               const int mask_dimension,
                                               const int max_sequence_length,
                                               const bool skip_softmax,
                                               const float mask_filter_value) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  int base_index = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;

  float thread_data = -CUDART_INF_F;
  if (threadIdx.x < all_sequence_length) {
    thread_data = CalcRawMaskedSoftmaxInput(
        base_index, threadIdx.x, all_sequence_length, sequence_length, attention_mask, key_padding_mask,
        add_before_softmax, input, is_unidirectional, rsqrt_head_size, mask_dimension,
        max_sequence_length, mask_filter_value);
  }

  if (skip_softmax) {
    if (threadIdx.x < all_sequence_length) {
      output[base_index + threadIdx.x] = T(thread_data);
    }
    return;
  }

  const float max = BlockReduce(tmp_storage).Reduce(thread_data, cub::Max(), all_sequence_length);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_exp = threadIdx.x < all_sequence_length ? expf(thread_data - max_block) : 0.0f;
  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, cub::Sum(), all_sequence_length);

  // Store value of 1.0/sum
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  if (threadIdx.x < all_sequence_length) {
    output[base_index + threadIdx.x] = T(thread_data_exp * sum_reverse_block);
  }
}

template <typename T, unsigned TPB>
__global__ void SoftmaxWithRawMaskSmallKernel(const int all_sequence_length,
                                              const int sequence_length,
                                              const int* attention_mask,
                                              const bool* key_padding_mask,
                                              const T* add_before_softmax,
                                              const T* input,
                                              T* output,
                                              const bool is_unidirectional,
                                              const float rsqrt_head_size,
                                              const int mask_dimension,
                                              const int max_sequence_length,
                                              const bool skip_softmax,
                                              const float mask_filter_value) {
  SoftmaxWithRawMaskSmall<T, TPB>(
      all_sequence_length, sequence_length,
      attention_mask, key_padding_mask, add_before_softmax, input, output,
      is_unidirectional, rsqrt_head_size, mask_dimension, max_sequence_length,
      skip_softmax, mask_filter_value);
}

template <typename T>
Status ComputeSoftmaxWithRawMask(cudaStream_t stream,
                                 const int all_sequence_length,
                                 const int sequence_length,
                                 const int batch_size,
                                 const int num_heads,
                                 const int* attention_mask,
                                 const bool* key_padding_mask,
                                 const T* add_before_softmax,
                                 const T* input,
                                 T* output,
                                 const bool is_unidirectional,
                                 const float rsqrt_head_size,
                                 const int mask_dimension,
                                 const int max_sequence_length,
                                 const bool use_persistent_softmax,
                                 T* persistent_softmax_workspace,
                                 const float mask_filter_value) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);
  T* out = use_persistent_softmax ? persistent_softmax_workspace : output;

  bool need_call_online_softmax = SoftmaxEnvVarSingleton::Instance().IsOnlineSoftmaxEnabled();
  if (!need_call_online_softmax) {
    if (all_sequence_length <= 32) {
      const int blockSize = 32;
      SoftmaxWithRawMaskSmallKernel<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length,
                                           attention_mask, key_padding_mask, add_before_softmax, input, out,
                                           is_unidirectional, rsqrt_head_size, mask_dimension, max_sequence_length,
                                           use_persistent_softmax, mask_filter_value);
    } else if (all_sequence_length <= 64) {
      const int blockSize = 64;
      SoftmaxWithRawMaskSmallKernel<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length,
                                           attention_mask, key_padding_mask, add_before_softmax, input, out,
                                           is_unidirectional, rsqrt_head_size, mask_dimension, max_sequence_length,
                                           use_persistent_softmax, mask_filter_value);
    } else if (all_sequence_length <= 128) {
      const int blockSize = 128;
      SoftmaxWithRawMaskSmallKernel<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length,
                                           attention_mask, key_padding_mask, add_before_softmax, input, out,
                                           is_unidirectional, rsqrt_head_size, mask_dimension, max_sequence_length,
                                           use_persistent_softmax, mask_filter_value);
    } else if (all_sequence_length <= 256) {
      const int blockSize = 256;
      SoftmaxWithRawMaskSmallKernel<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length,
                                           attention_mask, key_padding_mask, add_before_softmax, input, out,
                                           is_unidirectional, rsqrt_head_size, mask_dimension, max_sequence_length,
                                           use_persistent_softmax, mask_filter_value);
    } else if (all_sequence_length <= 512) {
      const int blockSize = 512;
      SoftmaxWithRawMaskSmallKernel<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length,
                                           attention_mask, key_padding_mask, add_before_softmax, input, out,
                                           is_unidirectional, rsqrt_head_size, mask_dimension, max_sequence_length,
                                           use_persistent_softmax, mask_filter_value);
    } else if (all_sequence_length <= 1024) {
      const int blockSize = 1024;
      SoftmaxWithRawMaskSmallKernel<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length,
                                           attention_mask, key_padding_mask, add_before_softmax, input, out,
                                           is_unidirectional, rsqrt_head_size, mask_dimension, max_sequence_length,
                                           use_persistent_softmax, mask_filter_value);
    } else {
      need_call_online_softmax = true;
    }
  }

  if (need_call_online_softmax) {
    const int blockSize = 256;
    if (use_persistent_softmax || all_sequence_length <= 2048) {
      const int sh_bytes = sizeof(float) * (use_persistent_softmax ? 0 : all_sequence_length);
      OnlineSoftmaxWithRawMaskKernel<T, blockSize>
          <<<grid, blockSize, sh_bytes, stream>>>(all_sequence_length, sequence_length,
                                                  attention_mask, key_padding_mask, add_before_softmax, input, out,
                                                  is_unidirectional, rsqrt_head_size, mask_dimension, max_sequence_length,
                                                  use_persistent_softmax, mask_filter_value);
    } else {
      OnlineSoftmaxWithRawMaskLongSequenceKernel<T, blockSize>
          <<<grid, blockSize, 0, stream>>>(all_sequence_length, sequence_length,
                                           attention_mask, key_padding_mask, add_before_softmax, input, out,
                                           is_unidirectional, rsqrt_head_size, mask_dimension, max_sequence_length,
                                           mask_filter_value);
    }
  }

  if (use_persistent_softmax) {
    return dispatch_warpwise_softmax_forward<T, T, float, false>(stream,
                                                                 output,
                                                                 persistent_softmax_workspace,
                                                                 all_sequence_length,
                                                                 all_sequence_length,
                                                                 batch_size * num_heads * sequence_length);
  }

  return CUDA_CALL(cudaGetLastError());
}

#define InstantiateComputeSoftmaxWithRawMask(T) \
  template Status ComputeSoftmaxWithRawMask(    \
      cudaStream_t stream,                      \
      const int all_sequence_length,            \
      const int sequence_length,                \
      const int batch_size,                     \
      const int num_heads,                      \
      const int* attention_mask,                \
      const bool* key_padding_mask,             \
      const T* add_before_softmax,              \
      const T* input,                           \
      T* output,                                \
      const bool is_unidirectional,             \
      const float rsqrt_head_size,              \
      const int mask_dimension,                 \
      const int max_sequence_length,            \
      const bool use_persistent_softmax,        \
      T* persistent_softmax_workspace,          \
      const float mask_filter_value)

InstantiateComputeSoftmaxWithRawMask(float);
InstantiateComputeSoftmaxWithRawMask(__half);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
