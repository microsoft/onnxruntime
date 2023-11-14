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
  std::cout << name << ": ";
  for (int i = 0; i < size; ++i) {
    std::cout << buffer[i] << ", ";
  }
  std::cout << std::endl;
}

}  // namespace ort_fastertransformer