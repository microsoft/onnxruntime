
#include "xnnpack_init.h"
#include <mutex>
#include "core/framework/allocatormgr.h"
#include "core/graph/constants.h"

namespace onnxruntime {
namespace xnnpack {
namespace {
void* xnn_allocate(void* context, size_t size) {
  IAllocator* allocator = (*reinterpret_cast<AllocatorPtr*>(context)).get();
  return allocator->Alloc(size);
}

void* xnn_reallocate(void* context, void* pointer, size_t size) {
  if (pointer == nullptr) {
    return xnn_allocate(context, size);
  }
  ORT_NOT_IMPLEMENTED("xnn_reallocate is not implemented");
}

void xnn_deallocate(void* context, void* pointer) {
  if (pointer != nullptr) {
    IAllocator* allocator = (*reinterpret_cast<AllocatorPtr*>(context)).get();
    allocator->Free(pointer);
  }
}

void* xnn_aligned_allocate(void* context, size_t alignment, size_t size) {
#if defined(__wasm__) && !defined(__wasm_relaxed_simd__) && !defined(__wasm_simd128__)
  ORT_ENFORCE(alignment <= 2 * sizeof(void*));
  return xnn_allocate(context, size);
#else
  void* ptr = xnn_allocate(context, size);
  ORT_ENFORCE((int64_t(ptr) & (alignment - 1)) == 0,
              " xnnpack wants to allocate a space with ", alignment, "bytes aligned. But it's not satisfied");
  // if ptr is not aligned, we have to find a way to return a aligned ptr and store the original ptr
  return ptr;
#endif
}

void xnn_aligned_deallocate(void* context, void* pointer) {
  return xnn_deallocate(context, pointer);
}
}  // namespace

AllocatorPtr XnnpackInitWrapper::GetOrCreateAllocator() {
  std::lock_guard<OrtMutex> lock(mutex);

  if (!ort_allocator_) {
    // create our allocator
    AllocatorCreationInfo allocator_info(
        [](int) {
          return std::make_unique<CPUAllocator>(OrtMemoryInfo(kXnnpackExecutionProvider,
                                                              OrtAllocatorType::OrtDeviceAllocator));
        });
    ort_allocator_ = CreateAllocator(allocator_info);
  }
  return ort_allocator_;
}

void XnnpackInitWrapper::InitXnnpackWithAllocatorAndAddRef(AllocatorPtr allocator) {
  increase_ref();

  static std::once_flag once;
  std::call_once(once, [allocator, this]() {
    ort_allocator_ = allocator;
    xnn_allocator_wrapper_ = {&ort_allocator_,
                              xnn_allocate,
                              xnn_reallocate,
                              xnn_deallocate,
                              xnn_aligned_allocate,
                              xnn_aligned_deallocate};
    const xnn_status st = xnn_initialize(&xnn_allocator_wrapper_);
    if (st != xnn_status_success) {
      ORT_THROW("XNNPACK initialization failed with status ", st);
    }
  });
}

}  // namespace xnnpack
}  // namespace onnxruntime
