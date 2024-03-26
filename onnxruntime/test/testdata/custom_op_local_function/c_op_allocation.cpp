#include "c_op_allocation.h"
#include <cstdlib>

#if (!(defined(PYTHON_MANYLINUX) && PYTHON_MANYLINUX))
#include <new>
#endif

namespace onnx_c_ops {

#if (defined(PYTHON_MANYLINUX) && PYTHON_MANYLINUX)

void *AllocatorDefaultAlloc(std::size_t size) { return malloc(size); }

void AllocatorDefaultFree(void *p) { free(p); }

#else

void *AllocatorDefaultAlloc(std::size_t size) {
  const std::size_t alignment = 64;
  void *p;
#if _MSC_VER
  p = _aligned_malloc(size, alignment);
  if (p == nullptr)
#if __cplusplus >= 202002L
    throw std::bad_alloc();
#else
    abort();
#endif
#elif defined(_LIBCPP_SGX_CONFIG)
  p = memalign(alignment, size);
  if (p == nullptr)
#if __cplusplus >= 202002L
    throw std::bad_alloc();
#else
    abort();
#endif
#else
  int ret = posix_memalign(&p, alignment, size);
  if (ret != 0)
#if __cplusplus >= 202002L
    throw std::bad_alloc();
#else
    abort();
#endif
#endif
  return p;
}

void AllocatorDefaultFree(void *p) {
#if _MSC_VER
  _aligned_free(p);
#else
  free(p);
#endif
}

#endif

} // namespace onnx_c_ops
