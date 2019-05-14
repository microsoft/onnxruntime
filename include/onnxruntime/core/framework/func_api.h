#pragma once
#include "core/common/common.h"
namespace onnxruntime {
#if 0
//TODO: Should use the lotus cpi element type definition.
enum DType {
  TFloat32 = 0,
  TInt32 = 1,
  TDouble = 2,
  TInt64 = 3,
  TBool = 4,
  TUint8 = 5,
  TInt8 = 6,
  TUint16 = 7,
  TInt16 = 8,
  TUint32 = 9,
  TUint64 = 10
  //TODO: more types
};
#endif

// AllocateFunc(void* handle, size_t alignment, size_t size)
using AllocateFunc = void* (*)(void*, size_t, size_t);
using DestroyFunc = void (*)(void*, void*);
using AllocatorHandle = void*;

typedef struct {
  //right now we only include allocation for host memory
  AllocateFunc allocate_func;
  DestroyFunc release_func;
  AllocatorHandle allocator_handle;
  const char* node_name;
} ComputeContext;

using FunctionState = void*;
// take the ComputeContext, and create a function state.
using CreateFunctionStateC = int (*)(ComputeContext*, FunctionState*);
// pass in the function state and input/output tensors, perform compute and return status code, 0 - succeed.
using ComputeFuncC = int (*)(FunctionState, const OrtCustomOpApi*, OrtKernelContext*);
// release the function state.
using DestroyFunctionStateC = void (*)(FunctionState);
}  // namespace onnxruntime
