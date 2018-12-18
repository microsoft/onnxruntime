#pragma once
#include "core/common/common.h"
namespace onnxruntime {
//TODO: Should use the lotus cpi element type definition.
enum DType {
  Float32Type = 0,
  Int32Type = 1,
  DoubleType = 2
  //TODO: more types
};

typedef struct {
  void* data;
  /*! \brief Number of dimensions */
  size_t ndim;
  /*! \brief The data type of the pointer*/
  DType dtype;
  /*! \brief The shape of the tensor */
  int64_t* shape;
} ONNXRunTimeTensor;

// AllocateFunc(void* handle, size_t alignment, size_t size)
typedef void* (*AllocateFunc)(void*, size_t, size_t);
typedef void (*ReleaseFunc)(void*, void*);
typedef void* AllocatorHandle;

typedef struct {
  //right now we only include allocation for host memory
  AllocateFunc allocate_func;
  ReleaseFunc release_func;
  AllocatorHandle allocator_handle;
  const char* node_name;
} ComputeContext;

typedef void* FunctionState;
// take the ComputeContext, and create a function state.
using CreateFunctionStateC = int (*)(ComputeContext*, FunctionState*);
// pass in the function state and input/output tensors, perform compute and return status code, 0 - succeed.
using ComputeFuncC = int (*)(FunctionState, ONNXRunTimeTensor*, size_t, ONNXRunTimeTensor*, size_t);
// release the function state.
using ReleaseFunctionStateC = void (*)(FunctionState);
}  // namespace onnxruntime
