#include <vector>
#include <iostream>
#include <string>

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace ort_fastertransformer {

template <typename T>
static void print_cuda_buffer(const std::string& name, const T* cuda_buffer, const int size) {
  cudaDeviceSynchronize();
  std::vector<T> buffer(size);
  cudaMemcpy(buffer.data(), cuda_buffer, size * sizeof(T), cudaMemcpyDeviceToHost);
  std::cout << name << ": " << std::endl;
  for (int i = 0; i < size; ++i) {
    std::cout << (float)buffer[i] << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
static void print_cuda_buffer(const std::string& name, const T* cuda_buffer, const int size1, const int size2) {
  cudaDeviceSynchronize();
  std::vector<T> buffer(size1 * size2);
  cudaMemcpy(buffer.data(), cuda_buffer, size1 * size2 * sizeof(T), cudaMemcpyDeviceToHost);
  std::cout << name << ": " << std::endl;
  for (int i = 0; i < size1; ++i) {
    for (int j = 0; j < size2; ++j) {
      std::cout << (float)buffer[i * size2 + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

}  // namespace ort_fastertransformer