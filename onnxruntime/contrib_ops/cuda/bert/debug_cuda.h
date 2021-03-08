#pragma once

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class PinnedHostBuffer {
 public:
  typedef std::shared_ptr<PinnedHostBuffer<T>> ptr;

  PinnedHostBuffer(size_t elementCount)
      : mBuffer(nullptr) {
    cudaHostAlloc(&mBuffer, elementCount * sizeof(T), cudaHostAllocDefault);
  }

  virtual ~PinnedHostBuffer() {
    if (mBuffer) {
      cudaFreeHost(mBuffer);
    }
  }

  operator T*() {
    return mBuffer;
  }

  operator const T*() const {
    return mBuffer;
  }

 protected:
  T* mBuffer;
};


template <typename T>
__global__ void Print2DTensor(const T* tensor, int dim0, int dim1, char title, char subtitle = ' ') {
  if (threadIdx.x == 0) {
    for (int i = 0; i < dim0; i++) {
      printf("%c%c[%d]:", title, subtitle, i);
      for (int j = 0; j < dim1; j++) {
        T value = tensor[i * dim1 + j];
        if (std::is_same<T, half>::value) {
          printf("\t%f", __half2float(value));
        } else if (std::is_integral<T>::value) {
          printf("\t%d", (int)value);
        } else {
          printf("\t%f", (float)value);
        }
      }
      printf("\n");
    }
  }
}

template <typename T>
__global__ void Print3DTensor(const T* tensor, int dim0, int dim1, int dim2, char title, char subtitle = ' ') {
  if (threadIdx.x == 0) {
    for (int i = 0; i < dim0; i++) {
      for (int j = 0; j < dim1; j++) {
        printf("%c%c[%d][%d]:", title, subtitle, i, j);
        for (int k = 0; k < dim2; k++) {
          T value = tensor[i * dim1 * dim2 + j * dim2 + k];
          if (std::is_same<T, half>::value) {
            printf("\t%f", __half2float(value));
          } else if (std::is_integral<T>::value) {
            printf("\t%d", (int)value);
          } else {
            printf("\t%f", (float)value);
          }
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}

template <typename T>
__global__ void Print4DTensor(const T* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle = ' ') {
  if (threadIdx.x == 0) {
    for (int i = 0; i < dim0; i++) {
      for (int j = 0; j < dim1; j++) {
        for (int k = 0; k < dim2; k++) {
          printf("%c%c[%d][%d][%d]:", title, subtitle, i, j, k);
          for (int x = 0; x < dim3; x++) {
            T value = tensor[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + x];
            if (std::is_same<T, half>::value) {
              printf("\t%f", __half2float(value));
            } else if (std::is_integral<T>::value) {
              printf("\t%d", (int)value);
            } else {
              printf("\t%f", (float)value);
            }
          }
          printf("\n");
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}

template <typename T>
__global__ void Print4DTensorSub2(const T* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle = ' ') {
  if (threadIdx.x == 0) {
    int i = dim0;
    int j = dim1;
    for (int k = 0; k < dim2; k++) {
      printf("%c%c[%d][%d][%d]:", title, subtitle, i, j, k);
      for (int x = 0; x < dim3; x++) {
        T value = tensor[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + x];
        if (std::is_same<T, half>::value) {
          printf("\t%f", __half2float(value));
        } else if (std::is_integral<T>::value) {
          printf("\t%d", (int)value);
        } else {
          printf("\t%f", (float)value);
        }
      }
      printf("\n");
    }
  }
}

template <typename T>
__global__ void Print4DTensorSub3(const T* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle = ' ') {
  if (threadIdx.x == 0) {
    int i = dim0;
    for (int j = 0; j < dim1; j++) {
      for (int k = 0; k < dim2; k++) {
        printf("%c%c[%d][%d][%d]:", title, subtitle, i, j, k);
        for (int x = 0; x < dim3; x++) {
          T value = tensor[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + x];
          if (std::is_same<T, half>::value) {
            printf("\t%f", __half2float(value));
          } else if (std::is_integral<T>::value) {
            printf("\t%d", (int)value);
          } else {
            printf("\t%f", (float)value);
          }
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
}

template <typename T>
void DumpTensor4D(cudaStream_t stream, const T* tensor, int dim0, int dim1, int dim2, int dim3, const std::string& title) {
  int elements = dim0 * dim1 * dim2 * dim3;

  auto data = std::make_shared<PinnedHostBuffer<T>>(elements);
  cudaMemcpyAsync(*data, tensor, elements * sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  const T* pinned_data = *data;
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      for (int k = 0; k < dim2; k++) {
        printf("%s[%d][%d][%d]:", title.c_str(), i, j, k);
        for (int x = 0; x < dim3; x++) {
          T value = pinned_data[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + x];
          if (std::is_same<T, half>::value) {
            printf("\t%f", __half2float(value));
          } else if (std::is_integral<T>::value) {
            printf("\t%d", (int)value);
          } else {
            printf("\t%f", (float)value);
          }
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
}

template <typename T>
void DumpTensor2D(cudaStream_t stream, const T* tensor, int dim0, int dim1, const std::string& title) {
  int elements = dim0 * dim1;

  auto data = std::make_shared<PinnedHostBuffer<T>>(elements);
  cudaMemcpyAsync(*data, tensor, elements * sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  const T* pinned_data = *data;

  for (int i = 0; i < dim0; i++) {
    printf("%s[%d]:", title.c_str(), i);
    for (int j = 0; j < dim1; j++) {
      T value = pinned_data[i * dim1 + j];
      if (std::is_same<T, half>::value) {
        printf("\t%f", __half2float(value));
      } else if (std::is_integral<T>::value) {
        printf("\t%d", (int)value);
      } else {
        printf("\t%f", (float)value);
      }
    }
    printf("\n");
  }
}

  template <typename T>
void Dump2DTensor(cudaStream_t stream, const T* tensor, int dim0, int dim1, char title, char subtitle = ' ') {
  cudaDeviceSynchronize();
  Print2DTensor<<<1, 1, 0, stream>>>(tensor, dim0, dim1, title, subtitle);
  cudaDeviceSynchronize();
  }

template <typename T>
void Dump3DTensor(cudaStream_t stream, const T* tensor, int dim0, int dim1, int dim2, char title, char subtitle = ' ') {
  cudaDeviceSynchronize();
  Print3DTensor<<<1, 1, 0, stream>>>(tensor, dim0, dim1, dim2, title, subtitle);
  cudaDeviceSynchronize();
}

template <typename T>
void Dump4DTensor(cudaStream_t stream, const T* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle = ' ') {
  cudaDeviceSynchronize();
  Print4DTensor<<<1, 1, 0, stream>>>(tensor, dim0, dim1, dim2, dim3, title, subtitle);
  cudaDeviceSynchronize();
}

inline void Dump4DTensor2(bool is_float16, cudaStream_t stream, const void* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle = ' ') {
  if (is_float16) {
    Print4DTensorSub2<<<1, 1, 0, stream>>>(reinterpret_cast<const float*>(tensor), dim0, dim1, dim2, dim3, title, subtitle);
  } else {
    Print4DTensorSub2<<<1, 1, 0, stream>>>(reinterpret_cast<const half*>(tensor), dim0, dim1, dim2, dim3, title, subtitle);
  }
}

inline void Dump4DTensor3(bool is_float16, cudaStream_t stream, const void* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle = ' ') {
  if (is_float16) {
    Print4DTensorSub3<<<1, 1, 0, stream>>>(reinterpret_cast<const float*>(tensor), dim0, dim1, dim2, dim3, title, subtitle);
  } else {
    Print4DTensorSub3<<<1, 1, 0, stream>>>(reinterpret_cast<const half*>(tensor), dim0, dim1, dim2, dim3, title, subtitle);
  }
}

inline void Dump4DTensor4(bool is_float16, cudaStream_t stream, const void* tensor, int dim0, int dim1, int dim2, int dim3, char title, char subtitle = ' ') {
  if (is_float16) {
    Dump4DTensor(stream, reinterpret_cast<const float*>(tensor), dim0, dim1, dim2, dim3, title, subtitle);
  } else {
    Dump4DTensor(stream, reinterpret_cast<const half*>(tensor), dim0, dim1, dim2, dim3, title, subtitle);
  }
}
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
