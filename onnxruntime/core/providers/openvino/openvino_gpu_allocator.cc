// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocatormgr.h"
#include "openvino_gpu_allocator.h"
//#include <CL/cl2.hpp>
//#include <cldnn/gpu_context_api_ocl.hpp>
//#include <cldnn/cldnn_config.hpp>

namespace onnxruntime {

void* OpenVINOAllocator::Alloc(size_t size) {
  void* p = nullptr;
  p = malloc(size);
 /* cl_int err;
  if (size > 0) {
    cl_context context = 
    cl::Buffer shared_buffer(openvino_ep::BackendManager::GetGlobalContext().context, CL_MEM_READ_WRITE, size, NULL, &err);
    p = (void *)shared_buffer;
  }*/
  return p;
}

void OpenVINOAllocator::Free(void* p) {
  free(p);
}

}  // namespace onnxruntime
