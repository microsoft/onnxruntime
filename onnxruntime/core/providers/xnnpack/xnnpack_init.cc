
#include "xnnpack_init.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>

#include "core/common/safeint.h"
#include "core/graph/constants.h"
#include "core/mlas/inc/mlas.h"

#include "xnnpack.h"

namespace onnxruntime {
namespace xnnpack {
namespace {
// Leave ample room for padding added by the underlying ORT allocator.
constexpr size_t kMaxAllocationSize = std::numeric_limits<size_t>::max() / 2;
#if defined(ORT_XNNPACK_TEST_ALLOCATOR)
thread_local int reallocation_failure = 0;
#endif

struct AllocationHeader {
  void* base;
  size_t size;
};

AllocationHeader* GetAllocationHeader(void* pointer) {
  return reinterpret_cast<AllocationHeader*>(static_cast<char*>(pointer) - sizeof(AllocationHeader));
}

void* xnn_allocate(void* context, size_t size) {
  if (size == 0 || size > kMaxAllocationSize) {
    return nullptr;
  }
  auto* allocator = static_cast<IAllocator*>(context);
  const size_t alignment = std::max({MlasGetPreferredBufferAlignment(), allocator->Info().device.GetAlignment(),
                                     size_t{XNN_ALLOCATION_ALIGNMENT}, alignof(AllocationHeader)});
  size_t total_size;
  if (!SafeAdd(size, sizeof(AllocationHeader), total_size) ||
      !SafeAdd(total_size, alignment - 1, total_size)) {
    return nullptr;
  }
  void* base = allocator->Alloc(total_size);
  if (base == nullptr) {
    return nullptr;
  }
  void* pointer = static_cast<char*>(base) + sizeof(AllocationHeader);
  size_t space = total_size - sizeof(AllocationHeader);
  std::align(alignment, size, pointer, space);
  // Keep metadata before the aligned payload so realloc needs no shared size map.
  new (GetAllocationHeader(pointer)) AllocationHeader{base, size};
  return pointer;
}

void xnn_deallocate(void* context, void* pointer) {
  if (pointer != nullptr) {
    (static_cast<IAllocator*>(context))->Free(GetAllocationHeader(pointer)->base);
  }
}

void* xnn_reallocate(void* context, void* pointer, size_t size) {
  if (size == 0) {
    xnn_deallocate(context, pointer);
    return nullptr;
  }
  if (pointer == nullptr) {
    return xnn_allocate(context, size);
  }
#if defined(ORT_XNNPACK_TEST_ALLOCATOR)
  const int failure = SetReallocationFailureForTesting(0);
  if (failure == 1) {
    return nullptr;
  }
#if !defined(ORT_NO_EXCEPTIONS)
  if (failure == 2) {
    throw std::bad_alloc();
  }
#endif
#endif
  void* replacement = xnn_allocate(context, size);
  if (replacement == nullptr) {
    return nullptr;
  }
  std::memcpy(replacement, pointer, std::min(GetAllocationHeader(pointer)->size, size));
  xnn_deallocate(context, pointer);
  return replacement;
}

void* xnn_aligned_allocate(void* context, size_t alignment, size_t size) {
  if (size == 0 || size > kMaxAllocationSize)
    return nullptr;

#if defined(__wasm__) && !defined(__wasm_relaxed_simd__) && !defined(__wasm_simd128__)
  ORT_ENFORCE(alignment <= 2 * sizeof(void*));
  return (static_cast<IAllocator*>(context))->Alloc(size);
#else
  void* ptr = (static_cast<IAllocator*>(context))->Alloc(size);
  ORT_ENFORCE((int64_t(ptr) & (alignment - 1)) == 0, "xnnpack allocation was not aligned to ", alignment, " bytes.");
  // if ptr is not aligned, we have to find a way to return a aligned ptr and store the original ptr
  return ptr;
#endif
}

void xnn_aligned_deallocate(void* context, void* pointer) {
  if (pointer != nullptr) {
    (static_cast<IAllocator*>(context))->Free(pointer);
  }
}
}  // namespace

#if defined(ORT_XNNPACK_TEST_ALLOCATOR)
int SetReallocationFailureForTesting(int failure) {
  return std::exchange(reallocation_failure, failure);
}
#endif

std::pair<AllocatorPtr&, xnn_allocator*> GetStoredAllocator() {
  static AllocatorPtr ort_allocator;
  static xnn_allocator xnn_allocator_wrapper_ = {ort_allocator.get(),
                                                 xnn_allocate,
                                                 xnn_reallocate,
                                                 xnn_deallocate,
                                                 xnn_aligned_allocate,
                                                 xnn_aligned_deallocate};
  return {ort_allocator, &xnn_allocator_wrapper_};
}

}  // namespace xnnpack
}  // namespace onnxruntime
