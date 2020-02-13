// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {

// An STL wrapper for ORT allocators. This enables overriding the 
// std::allocator used in STL containers for better memory performance.
template <class T>
class OrtStlAllocator {
  AllocatorPtr allocator;

public:
  typedef T value_type;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;
  using is_always_equal = std::true_type;

  OrtStlAllocator(const AllocatorPtr& a) noexcept {
    allocator = a;
  }
  OrtStlAllocator(const OrtStlAllocator& other) noexcept {
    allocator = other.allocator;
  }
  template <class U>
  OrtStlAllocator(const OrtStlAllocator<U>& other) noexcept {
    allocator = other.allocator;
  }

  T* allocate(size_t n, const void* hint = 0) {
    ORT_UNUSED_PARAMETER(hint);
    return reinterpret_cast<T*>(allocator->Alloc(n * sizeof(T)));
  }

  void deallocate(T* p, size_t n) {
    ORT_UNUSED_PARAMETER(n);
    allocator->Free(p);
  }
};

template <class T1, class T2>
bool operator==(const OrtStlAllocator<T1>& lhs, const OrtStlAllocator<T2>& rhs) noexcept { return lhs.allocator == rhs.allocator; }
template <class T1, class T2>
bool operator!=(const OrtStlAllocator<T1>& lhs, const OrtStlAllocator<T2>& rhs) noexcept { return lhs.allocator != rhs.allocator; }

} // namespace onnxruntime