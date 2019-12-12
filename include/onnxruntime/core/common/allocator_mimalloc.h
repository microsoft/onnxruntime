#pragma once
#ifdef USE_MIMALLOC
#include <stdio.h>
#include <mimalloc.h>
#include <cassert> // for assert
#include <limits>  // for max_size

#pragma warning(disable: 4100)

template <class T>
struct allocator_mimalloc {
  typedef T value_type;
  
  using propagate_on_container_copy_assignment = std::true_type; // for consistency
	using propagate_on_container_move_assignment = std::true_type; // to avoid the pessimization
	using propagate_on_container_swap = std::true_type; // to avoid the undefined behavior

	// to get the C++17 optimization: add this line for non-empty allocators which are always equal
	using is_always_equal = std::true_type;

  allocator_mimalloc() noexcept {}
  allocator_mimalloc(const allocator_mimalloc& other) noexcept {
  }

  template <class U>
  allocator_mimalloc(const allocator_mimalloc<U>& other) noexcept {
  }


  T* allocate(std::size_t n, const void* hint = 0);
  void deallocate(T* p, std::size_t n);
};

template <class T>
T* allocator_mimalloc<T>::allocate(std::size_t n, const void* hint) {
  return (T*)mi_mallocn(n, sizeof(T));
}

template <class T>
void allocator_mimalloc<T>::deallocate(T* p, std::size_t n) {
  mi_free(p);
}

template <class T1, class T2>
bool operator==(const allocator_mimalloc<T1>& lhs, const allocator_mimalloc<T2>& rhs) noexcept { 
  return true; }
template <class T1, class T2>
bool operator!=(const allocator_mimalloc<T1>& lhs, const allocator_mimalloc<T2>& rhs) noexcept { return false; }

template <typename T>
using Ty_Alloc = allocator_mimalloc<T>;

#else

template <typename T>
using Ty_Alloc = std::allocator<T>;

#endif

// std::vector<int, Tensor_Alloc<int> > x;