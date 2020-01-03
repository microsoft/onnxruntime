#pragma once
#include <mimalloc.h>

#pragma warning(disable: 4100)

template <class T>
struct allocator_mimalloc {
  typedef T value_type;
  
  using propagate_on_container_copy_assignment = std::true_type;
	using propagate_on_container_move_assignment = std::true_type;
	using propagate_on_container_swap = std::true_type;
	using is_always_equal = std::true_type;

  allocator_mimalloc() noexcept {}
  allocator_mimalloc(const allocator_mimalloc& other) noexcept {}
  template <class U>
  allocator_mimalloc(const allocator_mimalloc<U>& other) noexcept {}


  T* allocate(std::size_t n, const void* hint = 0) {
    return (T*)mi_mallocn(n, sizeof(T));
  }

  void deallocate(T* p, std::size_t n) {
    mi_free(p);
  }
};

template <class T1, class T2>
bool operator==(const allocator_mimalloc<T1>& lhs, const allocator_mimalloc<T2>& rhs) noexcept { return true; }
template <class T1, class T2>
bool operator!=(const allocator_mimalloc<T1>& lhs, const allocator_mimalloc<T2>& rhs) noexcept { return false; }
