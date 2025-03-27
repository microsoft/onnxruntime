// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <cstdint>
#include <memory>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime::qnn {

struct DynamicLibraryHandleDeleter {
  void operator()(void* library_handle) noexcept;
};

using UniqueDynamicLibraryHandle = std::unique_ptr<void, DynamicLibraryHandleDeleter>;

// This namespace contains constants and typedefs corresponding to functions from rpcmem.h.
// https://github.com/quic/fastrpc/blob/v0.1.1/inc/rpcmem.h
namespace rpcmem {

constexpr uint32_t RPCMEM_DEFAULT_FLAGS = 1;

constexpr int RPCMEM_HEAP_ID_SYSTEM = 25;

// External flag passed to fastrpc_register_buf_attr that specifies that a buffer should be copied to RPCMEM space
constexpr int FASTRPC_ATTR_IMPORT_BUFFER = 256;

// FastRPC invalid client handle
constexpr int INVALID_CLIENT_HANDLE = -1;

  // RPCMEM alignment is in 4KB blocks
constexpr const size_t BUFFER_ALIGNMENT_BLOCK_SIZE = 4096;
 
 /**
 * Allocate a zero-copy buffer for size upto 2 GB with the FastRPC framework.
 * Buffers larger than 2 GB must be allocated with rpcmem_alloc2
 * @param[in] heapid  Heap ID to use for memory allocation.
 * @param[in] flags   ION flags to use for memory allocation.
 * @param[in] size    Buffer size to allocate.
 * @return            Pointer to the buffer on success; NULL on failure.
 */
using AllocFnPtr = void* (*)(int heapid, uint32_t flags, int size);

/**
 * Free a buffer and ignore invalid buffers.
 */
using FreeFnPtr = void (*)(void* po);

/**
 * Return an associated file descriptor.
 * @param[in] po  Data pointer for an RPCMEM-allocated buffer.
 * @return        Buffer file descriptor.
 */
using ToFdFnPtr = int (*)(void* po);

using RemoteRegisterBufAttr = void (*)(void *po, int size, int fd, int attr);
}  // namespace rpcmem

// RPCMEM API function pointers.
struct RpcMemApi {
  rpcmem::AllocFnPtr alloc;
  rpcmem::FreeFnPtr free;
  rpcmem::ToFdFnPtr to_fd;
  rpcmem::RemoteRegisterBufAttr register_buff_attr;
};

// Loads and provides access to the RPCMEM API functions from a dynamically loaded library.
class RpcMemLibrary {
 public:
  RpcMemLibrary();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RpcMemLibrary);

  const RpcMemApi& Api() const { return api_; }

 private:
  UniqueDynamicLibraryHandle library_handle_;
  RpcMemApi api_;
};

}  // namespace onnxruntime::qnn
