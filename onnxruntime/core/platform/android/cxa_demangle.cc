// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_DUMMY_EXA_DEMANGLE

#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define DEMANGLE_INVALID_ARGUMENTS -3
#define DEMANGLE_INVALID_MANGLED_NAME -2
#define DEMANGLE_MEMORY_ALLOCATION_FAILURE -1
#define DEMANGLE_SUCCESS 0

// Reduced from https://github.com/llvm/llvm-project/blob/dbd80d7d27/libcxxabi/src/demangle/Utility.h
static inline bool initialize_output_buffer(char*& buf, size_t* n, size_t& buf_size, size_t init_size) {
  // only handle buf == null case, is buf is provided, in original impl, it realloc to expand buf space.
  // we don't handle the expansion, instead, the name copied to buf will be truncated.
  if (buf == nullptr) {
    buf = static_cast<char*>(malloc(init_size));
    if (buf == nullptr)
      return false;
    buf_size = init_size;
  } else {
    buf_size = *n;
  }
  return true;
}

/**
 * An dummy implementation of __cxa_demangle https://itanium-cxx-abi.github.io/cxx-abi/abi.html#demangler
 * It is referred in https://github.com/llvm/llvm-project/blob/d09d297c5d/libcxxabi/src/cxa_default_handlers.cpp#L53
 * However it contribute a large percentage of binary size in minimal build. To avoid it, the LLVM user can compile
 * libc++abi.a with LIBCXXABI_NON_DEMANGLING_TERMINATE defined, see https://reviews.llvm.org/D88189
 * But this make the compilation process extremely complex.
 *
 * Here we provide a dummy __cxa_demangle. It always copy the mangled name to output buffer.
 *
 * Reduced from https://github.com/llvm/llvm-project/blob/dbd80d7d27/libcxxabi/src/cxa_demangle.cpp
 */
extern "C" {
char* __cxa_demangle(const char* mangled_name, char* buf, size_t* n, int* status) {
  if (mangled_name == nullptr || (buf != nullptr && n == nullptr)) {
    if (status)
      *status = DEMANGLE_INVALID_ARGUMENTS;
    return nullptr;
  }

  int internal_status = DEMANGLE_SUCCESS;
  size_t buf_size;

  if (!initialize_output_buffer(buf, n, buf_size, 1024)) {
    internal_status = DEMANGLE_MEMORY_ALLOCATION_FAILURE;
  } else {
    // This might cause a truncated mangled name being returned without error,
    // but should be fine for debugging purpose.
    strncpy(buf, mangled_name, buf_size);
    buf[buf_size - 1] = '\0';
  }

  if (status)
    *status = internal_status;
  return internal_status == DEMANGLE_SUCCESS ? buf : nullptr;
}
}

#undef DEMANGLE_SUCCESS
#undef DEMANGLE_MEMORY_ALLOCATION_FAILURE
#undef DEMANGLE_INVALID_MANGLED_NAME
#undef DEMANGLE_INVALID_ARGUMENTS

#endif
