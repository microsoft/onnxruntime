// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/triton_kernel.h"
#include "core/framework/tunable.h"
#include <fstream>
#include <thread>

#ifdef USE_TRITON_KERNEL
#include <dlfcn.h>
#include "triton_kernel_infos.h"
#endif

#define ORT_TRITON_CHECK(status, msg)                \
  if ((status) != CUDA_SUCCESS) {                      \
      ORT_RETURN_IF(true, msg);                      \
  }

#define ORT_TRITON_THROW(status, msg)          \
  if ((status) != CUDA_SUCCESS) {                \
      ORT_THROW(msg);                          \
  }

namespace onnxruntime {
namespace cuda {
namespace {

// a vector of kernel metadata
static std::vector<TritonKernelMetaData> ort_triton_kernel_metadata;

// store group_name -> [kernel metadata id vector]
static std::unordered_map<std::string, std::vector<int>> ort_triton_kernel_group_map;

#ifdef USE_TRITON_KERNEL

// store func_name -> kernel metadata id
static std::unordered_map<std::string, int> ort_triton_kernel_map;

const int GPU_WARP_SIZE = 32;

Status GetSymbolFromLibrary(const std::string& symbol_name, void** symbol) {
  dlerror();  // clear any old error str

  // USe RTLD_DEFAULT for search current lib.so
  // value of RTLD_DEFAULT differs across posix platforms (-2 on macos, 0 on linux).
  void* handle = RTLD_DEFAULT;
  *symbol = dlsym(handle, symbol_name.c_str());

  char* error_str = dlerror();
  if (error_str) {
    Status status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
		    "Failed to get symbol " + symbol_name + " with error: " + error_str);
    return status;
  }
  // it's possible to get a NULL symbol in our case when Schemas are not custom.
  return Status::OK();
}
#endif


/*
 *  Try to load HIP kernels that compiled by triton.
 *  They are in hsaco/cubin format, and should use cuModuleLoad to load these kernels.
 */
void TryToLoadKernel() {
  auto status = Status::OK();

#ifdef USE_TRITON_KERNEL
  // get all kernel symbols from curret lib.so
  size_t size = sizeof(kernel_infos) / sizeof(kernel_infos[0]);

  for (int i = 0; i < size; ++i) {
    auto k_i = kernel_infos[i];

    void *buff;
    ORT_THROW_IF_ERROR(GetSymbolFromLibrary(k_i.name_start, &buff));

    // try to load module and get function
    CUmodule module;
    ORT_TRITON_THROW(cuModuleLoadData(&module, buff), "load module data failed.");

    CUfunction function;
    ORT_TRITON_THROW(cuModuleGetFunction(&function, module, k_i.func_name), "get funcion from module failed.");

    // setup kernel metadata
    TritonKernelMetaData metadata;
    metadata.num_warps = k_i.num_warps;
    metadata.shared_mem_size = k_i.shared;
    metadata.func = function;
    std::string fname = k_i.name;  // name is not same as func_name
    metadata.name = fname;
    std::string group_name = k_i.group_name;

    // pass constants
    for (auto &kv : k_i.constants) {
      metadata.constants[kv.first] = kv.second;
    }

    auto idx = ort_triton_kernel_metadata.size();
    ort_triton_kernel_metadata.push_back(metadata);
    ort_triton_kernel_map[fname] = idx;
    ort_triton_kernel_group_map[group_name].push_back(idx);
    LOGS_DEFAULT(VERBOSE) << "loaded ort triton kernel: " << fname << " idx: " << idx;
  }
#endif

  ORT_THROW_IF_ERROR(status);
}

static std::once_flag load_ort_triton_kernel_flag;

}  // end of namespace

void LoadOrtTritonKernel() {
  // load kernel should be called only once
  std::call_once(load_ort_triton_kernel_flag, TryToLoadKernel);
}

Status LaunchTritonKernel(cudaStream_t stream, std::string fname, int grid0, int grid1, int grid2, void* args, size_t args_size) {
#ifdef USE_TRITON_KERNEL
  if (ort_triton_kernel_map.count(fname) == 0) {
    // return unsupported status when not found function name in registry
    // this error status will be used by tunableOp
    std::ostringstream message_stream;
    message_stream << "can't find ort triton kernel name: " << fname;
    std::string message = message_stream.str();
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(true, message);
  }
  auto idx = ort_triton_kernel_map[fname];
  auto metadata = ort_triton_kernel_metadata[idx];

  void* config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, args, CU_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                    CU_LAUNCH_PARAM_END};

  ORT_TRITON_CHECK(cuLaunchKernel(metadata.func,
                                  grid0, grid1, grid2,
                                  GPU_WARP_SIZE * metadata.num_warps, 1, 1,
                                  metadata.shared_mem_size,
                                  stream,
                                  nullptr,
                                  (void**)&config), "launch kernel failed.");
#endif

  return Status::OK();
}

Status LaunchTritonKernel(cudaStream_t stream, size_t idx, int grid0, int grid1, int grid2, void* args, size_t args_size) {
#ifdef USE_TRITON_KERNEL
  if (idx >= ort_triton_kernel_metadata.size()) {
    // return unsupported status when not found function name in registry
    // this error status will be used by tunableOp
    std::ostringstream message_stream;
    message_stream << "can't find ort triton kernel idx: " << idx;
    std::string message = message_stream.str();
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(true, message);
  }
  auto metadata = ort_triton_kernel_metadata[idx];

  void* config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, args, CU_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                    CU_LAUNCH_PARAM_END};

  ORT_TRITON_CHECK(cuLaunchKernel(metadata.func,
                                  grid0, grid1, grid2,
                                  GPU_WARP_SIZE * metadata.num_warps, 1, 1,
                                  metadata.shared_mem_size,
                                  stream,
                                  nullptr,
                                  (void**)&config), "launch kernel failed.");
#endif

  return Status::OK();
}

const TritonKernelMetaData* GetOrtTritonKernelMetadata(size_t idx) {
  if (idx >= ort_triton_kernel_metadata.size()) {
    return nullptr;
  }
  return &ort_triton_kernel_metadata[idx];
}

const std::vector<int>* GetOrtTritonKernelByGroup(std::string group_name) {
  if (ort_triton_kernel_group_map.count(group_name) == 0) {
    return nullptr;
  }
  return &ort_triton_kernel_group_map.at(group_name);
}

}  // namespace cuda
}  // namespace onnxruntime
