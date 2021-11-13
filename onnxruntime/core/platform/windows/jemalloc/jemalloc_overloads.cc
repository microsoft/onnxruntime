// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef USE_JEMALLOC
#error Must be compiled only when jemalloc is in use
#endif

#ifndef _WIN32
#define _In_
#define _In_z_
#define _In_opt_
#define _In_opt_z_
#define _Out_
#define _Outptr_
#define _Out_opt_
#define _Inout_
#define _Inout_opt_
#define _Frees_ptr_opt_
#define _Ret_maybenull_
#define _Ret_notnull_
#define _Check_return_
#define _Outptr_result_maybenull_
#define _In_reads_(X)
#define _Inout_updates_all_(X)
#define _Out_writes_bytes_all_(X)
#define _Out_writes_all_(X)
#define _Success_(X)
#define _Outptr_result_buffer_maybenull_(X)
#define ORT_ALL_ARGS_NONNULL __attribute__((nonnull))
#else
#include <specstrings.h>
#endif

#include "jemalloc/jemalloc.h"

#include <new>
#include <stdexcept>

namespace onnxruntime {
namespace port {
void* ort_jemalloc_aligned_alloc(size_t size, size_t alignment) noexcept {
  // zero size requests are legal but je_malloc
  // deals with them as if one byte requests
  // Query arena

  if (size == 0) {
    size = 1;
  }

  int flags = 0;
  if (alignment > 0) {
    flags |= MALLOCX_ALIGN(alignment);
  }
  return je_mallocx(size, flags);
}

void ort_jemalloc_dellocate(void* p) noexcept {
  je_dallocx(p, 0);
}
}  // namespace port
}  // namespace onnxruntime

using namespace onnxruntime::port;

_Ret_notnull_ _Post_writable_byte_size_(size) void* operator new(size_t size) {
  void* result = ort_jemalloc_aligned_alloc(size, 0);
  if (result == nullptr) {
    throw std::bad_alloc();
  }
  return result;
}

_Ret_notnull_ _Post_writable_byte_size_(size) void* operator new[](size_t size) {
  void* result = ort_jemalloc_aligned_alloc(size, 0);
  if (result == nullptr) {
    throw std::bad_alloc();
  }
  return result;
}

_Ret_maybenull_ _Success_(return != NULL) _Post_writable_byte_size_(size) void* operator new(size_t size, const std::nothrow_t&) noexcept {
  return ort_jemalloc_aligned_alloc(size, 0);
}

_Ret_maybenull_ _Success_(return != NULL) _Post_writable_byte_size_(size) void* operator new[](size_t size, const std::nothrow_t&) noexcept {
  return ort_jemalloc_aligned_alloc(size, 0);
}

_Ret_notnull_ _Post_writable_byte_size_(size) void* operator new(size_t size, std::align_val_t al) {
  void* result = ort_jemalloc_aligned_alloc(size, static_cast<size_t>(al));
  if (result == nullptr) {
    throw std::bad_alloc();
  }
  return result;
}

_Ret_maybenull_ _Success_(return != NULL) _Post_writable_byte_size_(size) void* operator new(size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
  return ort_jemalloc_aligned_alloc(size, static_cast<size_t>(al));
}

void* operator new[](size_t size, std::align_val_t al) {
  void* result = ort_jemalloc_aligned_alloc(size, static_cast<size_t>(al));
  if (result == nullptr) {
    throw std::bad_alloc();
  }
  return result;
}

_Ret_maybenull_ _Success_(return != NULL) _Post_writable_byte_size_(size) void* operator new[](size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
  return ort_jemalloc_aligned_alloc(size, static_cast<size_t>(al));
}

// We do not provide aligned overloads since all deallocate can handle aligned.
void operator delete(void* p) noexcept {
  if (p) {
    ort_jemalloc_dellocate(p);
  }
}

void operator delete[](void* p) noexcept {
  if (p) {
    ort_jemalloc_dellocate(p);
  }
}

// Hard to override malloc/free. Replace base functions
//#if defined(_MSC_VER) && !defined(_DEBUG)
//
//extern "C" {
// _Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(size) 
//_CRTRESTRICT
//void*  __cdecl _malloc_base(_In_ _CRT_GUARDOVERFLOW size_t size) {
//  return je_malloc(size);
//}
//
//void __cdecl _free_base(_Pre_maybenull_ _Post_invalid_ void* p) {
//  if (p) {
//    je_free(p);
//  }
//}
//
//_Success_(return != 0) _Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(size)
//_CRTRESTRICT
//void* __cdecl _realloc_base(_Pre_maybenull_ _Post_invalid_ void* p, _In_ size_t size) {
//  return je_realloc(p, size);
//}
//
//_Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(count* size)
//_CRTRESTRICT
//void* __cdecl _calloc_base(
//    _In_ size_t count,
//    _In_ size_t size) {
//  return je_calloc(count, size);
//}
//
//_Check_return_
//size_t __cdecl _msize_base(_Pre_notnull_ void* p) {
//  return je_malloc_usable_size(p);
//}
//}
//
//#endif
