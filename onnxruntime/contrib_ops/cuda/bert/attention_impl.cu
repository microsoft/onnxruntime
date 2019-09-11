
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "normalize_impl.h"
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include <cub/cub.cuh>

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {
const size_t kAlignment = 256;

template <typename IntType>
static IntType alignTo(IntType a, IntType b) {
  return CeilDiv(a, b) * b;
}

size_t scratchSize(size_t wordSize, int batchsize, int numHeads, int sequenceLength) {
  const size_t len = batchsize * numHeads * sequenceLength * sequenceLength;
  const size_t bytes = len * wordSize;

  const size_t bytesAligned = alignTo<size_t>(bytes, kAlignment);
  return bytesAligned;
}

size_t getAttentionWorkspaceSize(size_t wordSize, int batchsize, int numHeads, int headSize, int sequenceLength) {
  return 2 * scratchSize(wordSize, batchsize, numHeads, sequenceLength) \
      + 3 * batchsize * sequenceLength * numHeads * headSize * wordSize;
}


struct CublasConfigHelper {
  cublasPointerMode_t pm;
  cublasMath_t mm;
  cublasHandle_t cublas;
  CublasConfigHelper(cublasHandle_t cublas_)
      : cublas(cublas_) {
    cublasGetPointerMode(cublas, &pm);
    cublasGetMathMode(cublas, &mm);
    cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST);
    cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);
  }
  ~CublasConfigHelper() {
    cublasSetMathMode(cublas, mm);
    cublasSetPointerMode(cublas, pm);
  }
};


template <typename T, unsigned TPB>
__device__ inline void scaledSoftmax(
    const int ld, const int lastValid, const float rsqrtHeadSize, const T* input, T* output)
{

    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float rZ;

    const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;

    const float w(rsqrtHeadSize);
    cub::Sum sum;
    float threadData(0);

    for (int i = threadIdx.x; i < lastValid; i += TPB)
    {
        const int idx = offset + i;
        const float val = input[idx];
        threadData += exp(val * w);
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        rZ = 1.f / Z;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        const float val = (i < lastValid) ? exp(float(input[idx]) * w) * rZ : 0.f;
        output[idx] = T(val);
    }
}

template <typename T, unsigned TPB>
__device__ inline void scaledSoftmaxSmall(
    const int ld, const int lastValid, const float rsqrtHeadSize, const T* input, T* output) {
  using BlockReduce = cub::BlockReduce<float, TPB>;

  __shared__ typename BlockReduce::TempStorage tmpStorage;

  __shared__ float rZ;

  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;

  const float w(rsqrtHeadSize);
  cub::Sum sum;
  float threadData(0);

  const int idx = offset + threadIdx.x;
  if (threadIdx.x < lastValid) {
    const float val = input[idx];
    threadData = exp(val * w);
  }

  const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

  if (threadIdx.x == 0) {
    rZ = (1.f) / Z;
  }
  __syncthreads();

  if (threadIdx.x < ld) {
    // this will be 0 for threadIdx.x >= lastValid
    output[idx] = T(threadData * rZ);
  }
}

template <typename T, unsigned TPB>
__global__ void maskedScaledSoftmaxKernelSmall(
    const int ld, const float rsqrtHeadSize, const int* maskIdx, const T* input, T* output) {
  __shared__ int lastValid;

  if (threadIdx.x == 0) {
    lastValid = min(ld, maskIdx[blockIdx.y]);
  }
  __syncthreads();

  scaledSoftmaxSmall<T, TPB>(ld, lastValid, rsqrtHeadSize, input, output);
}

template <typename T, unsigned TPB>
__global__ void maskedScaledSoftmaxKernel(
    const int ld, const float rsqrtHeadSize, const int* maskIdx, const T* input, T* output) {
  __shared__ int lastValid;

  if (threadIdx.x == 0) {
    lastValid = min(ld, maskIdx[blockIdx.y]);
  }
  __syncthreads();
  scaledSoftmax<T, TPB>(ld, lastValid, rsqrtHeadSize, input, output);
}

template <typename T>
int computeMaskedScaledSoftmax(cudaStream_t stream, const int ld, const int B, const int N, const float rsqrtHeadSize,
                               const int* maskIdx, const T* input, T* output) {
  // Mask idx is of length B and assumes the valid region is contiguous starting
  // from the beginning of the sequence

  const dim3 grid(ld * N, B, 1);

  if (ld <= 32) {
    const int blockSize = 32;
    maskedScaledSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, maskIdx, input, output);
  } else if (ld <= 128) {
    const int blockSize = 128;
    maskedScaledSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, maskIdx, input, output);
  } else if (ld == 384) {
    const int blockSize = 384;
    maskedScaledSoftmaxKernelSmall<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, maskIdx, input, output);
  } else {
    const int blockSize = 256;
    maskedScaledSoftmaxKernel<T, blockSize>
        <<<grid, blockSize, 0, stream>>>(ld, rsqrtHeadSize, maskIdx, input, output);
  }

  CUDA_CALL(cudaPeekAtLastError());
  return 0;
}


template <typename T>
cublasStatus_t inline cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                 int n, int k, const T alpha, const T* A, int lda, const T* B, int ldb, const T beta, T* C, int ldc);

template <>
cublasStatus_t inline cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                 int n, int k, const float alpha, const float* A, int lda, const float* B, int ldb, const float beta, float* C,
                                 int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <>
cublasStatus_t inline cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
                                 int n, int k, const half alpha, const half* A, int lda, const half* B, int ldb, const half beta, half* C, int ldc) {
  return cublasHgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <typename T>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                               cublasOperation_t transb, int m, int n, int k, const T alpha, const T* A, int lda, long long int strideA,
                                               const T* B, int ldb, long long int strideB, const T beta, T* C, int ldc, long long int strideC, int batchCount);

template <>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                               cublasOperation_t transb, int m, int n, int k, const float alpha, const float* A, int lda, long long int strideA,
                                               const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC,
                                               int batchCount) {
  return cublasSgemmStridedBatched(
      handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}

template <>
cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                               cublasOperation_t transb, int m, int n, int k, const half alpha, const half* A, int lda, long long int strideA,
                                               const half* B, int ldb, long long int strideB, const half beta, half* C, int ldc, long long int strideC,
                                               int batchCount) {
  return cublasHgemmStridedBatched(
      handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
}


template <typename T>
__global__ void transposeCtx(const int H, const T* input, T* output) {
  // Input:  BxNxSxH
  // Output: BxSxNxH

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;

  int N = blockDim.y;
  int S = gridDim.x;
 
  const int NH = N * H;
  const int NHS = NH * S;
  const int in_offset = s * H + n * S * H + b * NHS;
  const int out_offset = n * H + s * NH + b * NHS;

  const int i = threadIdx.x;
  if (i < H) {
    output[out_offset + i] = input[in_offset + i];
  }
}

void launchTransCtx(cudaStream_t stream, const int S, const int B, const int headSize, const int numHeads,
                    const float* input, float* output) {
  const dim3 grid(S, B, 1);
  if (0 == (headSize & 1)) {
    const int H = headSize / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    const dim3 block(H, numHeads, 1);
    transposeCtx<float2><<<grid, block, 0, stream>>>(H, input2, output2);
    CUDA_CALL(cudaPeekAtLastError());
  } else {
    const dim3 block(headSize, numHeads, 1);
    transposeCtx<float><<<grid, block, 0, stream>>>(headSize, input, output);
    CUDA_CALL(cudaPeekAtLastError());
  }
}

void launchTransCtx(cudaStream_t stream, const int S, const int B, const int headSize, const int numHeads,
                    const half* input, half* output) {
  const dim3 grid(S, B, 1);
  if (0 == (headSize % 4)) {
    const int H = headSize / 4;
    const dim3 block(H, numHeads, 1);
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    transposeCtx<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else if (0 == (headSize & 1)) {
    const int H = headSize / 2;
    const dim3 block(H, numHeads, 1);
    const half2* input2 = reinterpret_cast<const half2*>(input);
    half2* output2 = reinterpret_cast<half2*>(output);
    transposeCtx<half2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel.
    const dim3 block(headSize, numHeads, 1);
    transposeCtx<half><<<grid, block, 0, stream>>>(headSize, input, output);
  }
  CUDA_CALL(cudaPeekAtLastError());
}

template <typename T>
__global__ void transposeQKV(const int H, const T* input, T* output) {
  // Input:  BxSx3xNxH
  // Output: 3xBxNxSxH

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int N = blockDim.y;

  const int S = gridDim.x;
  const int B = gridDim.y;
  const int NH = N * H;
  const int NHS = NH * S;
  const int in_offset = n * H + m * NH + s * 3 * NH + b * NHS * 3;
  const int out_offset = s * H + n * S * H + b * NHS + m * NHS * B;

  const int i = threadIdx.x;
  if (i < H) {
    output[out_offset + i] = input[in_offset + i];
  }
}

void launchTransQkv(cudaStream_t stream, const int S, const int B, const int headSize, const int numHeads,
                    const float* input, float* output) {
  const dim3 grid(S, B, 3);
  if (0 == (headSize & 1)) {
    const int H = headSize / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    const dim3 block(H, numHeads, 1);
    transposeQKV<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else {
    const dim3 block(headSize, numHeads, 1);
    transposeQKV<float><<<grid, block, 0, stream>>>(headSize, input, output);
  }
  CUDA_CALL(cudaPeekAtLastError());
}

void launchTransQkv(cudaStream_t stream, const int S, const int B, const int headSize, const int numHeads,
                    const half* input, half* output) {
  const dim3 grid(S, B, 3);
  if (0 == (headSize % 4)) {
    const int H = headSize / 4;
    const dim3 block(H, numHeads, 1);
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    transposeQKV<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else if (0 == (headSize & 1)) {
    const int H = headSize / 2;
    const dim3 block(H, numHeads, 1);
    const half2* input2 = reinterpret_cast<const half2*>(input);
    half2* output2 = reinterpret_cast<half2*>(output);
    transposeQKV<half2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel..
    const dim3 block(headSize, numHeads, 1);
    transposeQKV<half><<<grid, block, 0, stream>>>(headSize, input, output);
  }
  CUDA_CALL(cudaPeekAtLastError());
}

template <typename T>
int qkvToCtx(cublasHandle_t& cublas, cudaStream_t stream,
             const int batchSize, const int sequenceLength, const int numHeads, const int headSize,
             const T* input, T* output, T* scratch1, T* scratch2, T* scratch3,
             const int* maskIdx = nullptr) {
  // input should be BxSx3xNxH => scratch3: 3xBxNxSxH
  launchTransQkv(stream, sequenceLength, batchSize, headSize, numHeads, input, scratch3);

  // now scratch3 has Q, K, V: each has size BxNxSxH
  const int tsize = batchSize * numHeads * sequenceLength * headSize;
  const int imatSize = sequenceLength * headSize;
  const int omatSize = sequenceLength * sequenceLength;
  const int numMats = batchSize * numHeads;
  const T* q = scratch3;
  const T* k = q + tsize;
  const T* v = k + tsize;

  cublasSetStream(cublas, stream);
  CublasConfigHelper helper(cublas);

  // compute Q*K' (as K'*Q) and store in scratch1: BxNxSxS
  CUBLAS_CALL(cublasGemmStridedBatched<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, sequenceLength, sequenceLength, headSize, 1.f, k, headSize, imatSize,
                                          q, headSize, imatSize, 0.f, scratch1, sequenceLength, omatSize, numMats));

  // apply softmax and store result P to scratch2: BxNxSxS
  const float rsqrtHeadSize = 1.f / sqrt(float(headSize));
  computeMaskedScaledSoftmax<T>(stream, sequenceLength, batchSize, numHeads, rsqrtHeadSize, maskIdx, scratch1, scratch2);

  // compute P*V (as V*P), and store in scratch3: BxNxSxH
  CUBLAS_CALL(cublasGemmStridedBatched<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, headSize, sequenceLength, sequenceLength, 1.f, v, headSize, imatSize,
                                          scratch2, sequenceLength, omatSize, 0.f, scratch3, headSize, imatSize, numMats));

  // scratch3 is BxNxSxH, transpose to output BxSxNxH
  launchTransCtx(stream, sequenceLength, batchSize, headSize, numHeads, scratch3, output);
  return 0;
}

void launchAttentionKernel(
    const float* input,
    const int* mask,
    float* output,
    const int batchSize,
    const int sequenceLength,
    const int numHeads,
    const int headSize,
    void* workspace,
    cublasHandle_t& cublas) {
  //TODO: derive word size from input tensor type
  const size_t wordSize = 4;
  const size_t bytes = scratchSize(wordSize, batchSize, numHeads, sequenceLength);

  float* scratch1 = reinterpret_cast<float*>(workspace);
  float* scratch2 = scratch1 + (bytes / wordSize);
  float* scratch3 = scratch2 + (bytes / wordSize);

  // use default stream
  cudaStream_t stream = nullptr;

  qkvToCtx(cublas, stream,
           batchSize, sequenceLength, numHeads, headSize, 
           input, output, scratch1, scratch2, scratch3,
           mask);
}

}  // namespace cuda

}  // namespace contrib
}  // namespace onnxruntime
