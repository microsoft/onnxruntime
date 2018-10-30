#include "core/framework/func_kernel.h"
#include "core/framework/allocator.h"
namespace onnxruntime {
void* allocate_helper_func(void* allocator, size_t size) {
  auto* alloc = static_cast<IAllocator*>(allocator);
  return alloc->Alloc(size);
}

void release_helper_func(void* allocator, void* p) {
  auto* alloc = static_cast<IAllocator*>(allocator);
  return alloc->Free(p);
}
}  // namespace onnxruntime
