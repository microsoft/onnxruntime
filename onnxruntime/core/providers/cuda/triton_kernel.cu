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

#define ORT_TRITON_CHECK(expr, msg)                                   \
  do {                                                                \
    auto status = expr;                                               \
    const char* error_str;                                            \
    if (status != CUDA_SUCCESS) {                                     \
      auto get_status_err_str = cuGetErrorString(status, &error_str); \
      ORT_UNUSED_PARAMETER(get_status_err_str);                       \
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, msg, " ", error_str); \
    }                                                                 \
  } while (0)

#define ORT_TRITON_THROW(expr, msg)                                   \
  do {                                                                \
    auto status = expr;                                               \
    const char* error_str;                                            \
    if (status != CUDA_SUCCESS) {                                     \
      auto get_status_err_str = cuGetErrorString(status, &error_str); \
      ORT_UNUSED_PARAMETER(get_status_err_str);                       \
      ORT_THROW(msg, error_str);                                      \
    }                                                                 \
  } while (0)

namespace onnxruntime {
namespace cuda {
namespace {

// A vector of kernel metadata
static std::vector<TritonKernelMetaData> ort_triton_kernel_metadata;

// Store group_name -> [kernel metadata id vector]
static std::unordered_map<std::string, std::vector<int>> ort_triton_kernel_group_map;

#ifdef USE_TRITON_KERNEL

// Store func_name -> kernel metadata id
static std::unordered_map<std::string, int> ort_triton_kernel_map;

const int GPU_WARP_SIZE = 32;
constexpr int kMaxThreadsPerBlock = 1024;
// Currently the max shared memory per block is hardcoded to 64KB.
constexpr int kMaxSharedMemoryPerBlock = 64 * 1024;

Status GetSymbolFromLibrary(const std::string& symbol_name, void** symbol) {
  dlerror();  // Clear any old error str

  // Use RTLD_DEFAULT for search current lib.so.
  // Value of RTLD_DEFAULT differs across posix platforms (-2 on macos, 0 on linux).
  void* handle = RTLD_DEFAULT;
  *symbol = dlsym(handle, symbol_name.c_str());

  char* error_str = dlerror();
  if (error_str) {
    Status status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                    "Failed to get symbol " + symbol_name + " with error: " + error_str);
    return status;
  }
  // It's possible to get a NULL symbol in our case when Schemas are not custom.
  return Status::OK();
}
#endif

/*
 *  Try to load CUDA kernels that are compiled by Triton.
 *  They are in hsaco/cubin format, and should use cuModuleLoad to load these kernels.
 */
void TryToLoadKernel() {
  auto status = Status::OK();

#ifdef USE_TRITON_KERNEL
  // get all kernel symbols from curret lib.so
  size_t size = sizeof(kernel_infos) / sizeof(kernel_infos[0]);

  for (int i = 0; i < size; ++i) {
    auto k_i = kernel_infos[i];

    void* buff;
    ORT_THROW_IF_ERROR(GetSymbolFromLibrary(k_i.name_start, &buff));

    // try to load module and get function
    CUmodule module;
    ORT_TRITON_THROW(cuModuleLoadData(&module, buff), "Loading module data failed.");

    CUfunction function;
    ORT_TRITON_THROW(cuModuleGetFunction(&function, module, k_i.func_name), "Getting function from module failed.");

    // setup kernel metadata
    TritonKernelMetaData metadata;
    metadata.num_warps = k_i.num_warps;
    metadata.shared_mem_size = k_i.shared;
    metadata.func = function;
    std::string fname = k_i.name;  // name is not same as func_name
    metadata.name = fname;
    std::string group_name = k_i.group_name;

    // pass constants
    for (auto& kv : k_i.constants) {
      metadata.constants[kv.first] = kv.second;
    }

    auto idx = ort_triton_kernel_metadata.size();
    ort_triton_kernel_metadata.push_back(metadata);
    ort_triton_kernel_map[fname] = idx;
    ort_triton_kernel_group_map[group_name].push_back(idx);
    LOGS_DEFAULT(VERBOSE) << "Loaded ort triton kernel: " << fname << " idx: " << idx;
  }
#endif

  ORT_THROW_IF_ERROR(status);
}

static std::once_flag load_ort_triton_kernel_flag;

}  // namespace

void LoadOrtTritonKernel() {
  // load kernel should be called only once
  std::call_once(load_ort_triton_kernel_flag, TryToLoadKernel);
}



#ifdef USE_TRITON_KERNEL
Status LaunchTritonKernel(cudaStream_t stream, size_t idx, int grid0, int grid1, int grid2,
                          void* args, size_t args_size) {
  if (idx >= ort_triton_kernel_metadata.size()) {
    // Return unsupported status when idx exceeds the size of ort_triton_kernel_metadata.
    // This error status will be used by TunableOp
    std::ostringstream message_stream;
    message_stream << "Can't find ort triton kernel idx: " << idx;
    std::string message = message_stream.str();
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(true, message);
  }
  auto metadata = ort_triton_kernel_metadata[idx];

  int threads_per_block = GPU_WARP_SIZE * metadata.num_warps;
  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
      threads_per_block > kMaxThreadsPerBlock,
      "The threads_per_block (", threads_per_block, ") exceeds the max allowed value (", kMaxThreadsPerBlock, ").");
  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
      metadata.shared_mem_size > kMaxSharedMemoryPerBlock,
      "The shared_mem_size (", metadata.shared_mem_size, ") exceeds the max allowed value (",
      kMaxSharedMemoryPerBlock, " bytes).");

  void* config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, args, CU_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                    CU_LAUNCH_PARAM_END};

  ORT_TRITON_CHECK(cuLaunchKernel(metadata.func,
                                  grid0, grid1, grid2,
                                  threads_per_block, 1, 1,
                                  metadata.shared_mem_size,
                                  stream,
                                  nullptr,
                                  (void**)&config),
                   "Launching kernel failed.");

  return Status::OK();
}

Status LaunchTritonKernel(cudaStream_t stream, std::string fname, int grid0, int grid1, int grid2,
                          void* args, size_t args_size) {
  if (ort_triton_kernel_map.count(fname) == 0) {
    // Return unsupported status if function name not found in registry.
    // This error status will be used by TunableOp
    std::ostringstream message_stream;
    message_stream << "Can't find ort triton kernel name: " << fname;
    std::string message = message_stream.str();
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(true, message);
  }
  auto idx = ort_triton_kernel_map[fname];
  return LaunchTritonKernel(stream, idx, grid0, grid1, grid2, args, args_size);
}

#else
Status LaunchTritonKernel(cudaStream_t /*stream*/, std::string /*fname*/, int /*grid0*/, int /*grid1*/, int /*grid2*/,
                          void* /*args*/, size_t /*args_size*/) {
  return Status::OK();
}

Status LaunchTritonKernel(cudaStream_t /*stream*/, size_t /*idx*/, int /*grid0*/, int /*grid1*/, int /*grid2*/,
                          void* /*args*/, size_t /*args_size*/) {
  return Status::OK();
}
#endif


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
