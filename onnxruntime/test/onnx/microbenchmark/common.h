#pragma once

#include <algorithm>
#include <cstdlib>
#include <malloc.h>
#include <new>
#include <random>

// aligned memory allocate and free functions
inline void* aligned_alloc(size_t size, size_t align) {
  void* ptr;
#if _MSC_VER
  ptr = _aligned_malloc(size, align);
  if (ptr == nullptr) throw std::bad_alloc();
#else
  int ret = posix_memalign(&ptr, align, size);
  if (ret != 0) throw std::bad_alloc();
#endif
  return ptr;
}

inline void aligned_free(void* p) {
#ifdef _WIN32
  _aligned_free((void*)p);
#else
  ::free((void*)p);
#endif /* _WIN32 */
}

template <typename T>
T Clamp(T n, T lower, T upper) {
  return std::max(lower, std::min(n, upper));
}

template <typename T>
T* GenerateArrayWithRandomValue(size_t batch_size, T low, T high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(low, high);
  T* data = (T*)aligned_alloc(sizeof(T) * batch_size, 64);
  for (size_t i = 0; i != batch_size; ++i) {
    data[i] = static_cast<T>(dist(gen));
  }
  return data;
}