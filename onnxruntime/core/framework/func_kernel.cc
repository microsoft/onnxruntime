#include "core/framework/func_kernel.h"
#include "core/framework/allocator.h"
namespace onnxruntime {
void* allocate_helper_func(void* allocator, size_t alignment, size_t size) {
  // Here we only align the size, we expect the underline device allocator will
  // guarantee the address alignment.
  // We may update lotus' IAllocator interface to support alignment.
  size_t rounded_bytes = (alignment * ((size + alignment - 1) / alignment));
  auto* alloc = static_cast<IAllocator*>(allocator);
  return alloc->Alloc(rounded_bytes);
}

void release_helper_func(void* allocator, void* p) {
  auto* alloc = static_cast<IAllocator*>(allocator);
  return alloc->Free(p);
}

}  // namespace onnxruntime
