// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/triton_kernel.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/framework/tunable.h"
#include "hip/hip_runtime_api.h"
#include <fstream>
#include <thread>
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace onnxruntime {
namespace rocm {
namespace {

#define HIP_CHECK(status)                                       \
	if (status != hipSuccess) {                             \
		ORT_RETURN_IF(true, hipGetErrorName(status));  \
	}

const int HIP_WARP_SIZE=64;
const std::string FILE_SP = "/";
const std::string META_FILENAME = "meta.json";
static std::unordered_map<std::string, TritonKernelMetaData> rocm_triton_kernel_map;

const std::string GetRocmTritonKernelPath() {
  std::string path = onnxruntime::ParseEnvironmentVariableWithDefault<std::string>("ORT_TRITON_LIB_PATH", "");
  return path;
}

/*
 *  Try to load HIP kernels that compiled by triton.
 *  They are in hsaco format, and should use hipModuleLoad to load these kernels.
 */
void TryToLoadKernel() {
  auto status = Status::OK();
  
  ORT_TRY {
    // get kernel lib metadata
    const auto path = GetRocmTritonKernelPath();
    if (path == "") {
      // just return, not throw error
      // in case there no meta.json or user don't want to use triton.
      return;
    }
    std::string metadata_file = path + FILE_SP + META_FILENAME;

    std::ifstream meta_fd(metadata_file);
    if (!meta_fd.is_open()) {
      // if open file failed, just return
      // in case there no meta.json or user don't want to use triton.
      return;
    }

    json j = json::parse(meta_fd);
    auto num_meta = j.size();
    for (int i = 0; i < num_meta; ++i) {
      auto j_m = j[i];
      std::string lib_path = path + FILE_SP + std::string(j_m["lib_file"]);
      // try to load hasco module
      hipModule_t module;
      hipFunction_t function;
      HIP_CALL_THROW(hipModuleLoad(&module, lib_path.c_str()));
      std::string fname = j_m["func_name"];
      HIP_CALL_THROW(hipModuleGetFunction(&function, module, fname.c_str()));

      // setup kernel metadata
      TritonKernelMetaData metadata;
      metadata.num_warps = j_m["num_warps"];
      metadata.shared_mem_size = j_m["shared"];
      metadata.func = function;
      fname = j_m["name"];  // name is not same as func_name

      // parse constants
      if (j_m.contains("constants")) {
        auto constants = j_m["constants"];
	for (auto &el : constants.items()) {
          auto k = el.key();
	  auto v = el.value();
	  if (!v.is_number_integer()) {
            // only support k is str and v is integer
	    continue;
	  }
	  metadata.constants[k] = v;
	}
      }
      rocm_triton_kernel_map[fname] = metadata;
      LOGS_DEFAULT(VERBOSE) << "loaded rocm triton kernel: " << fname;
    }
  }
  ORT_CATCH (const std::exception &e) {
    ORT_HANDLE_EXCEPTION([&]() {
          std::ostringstream message_stream;
          message_stream << "rocm triton load metadata failed. Error message: " << e.what();

          std::string message = message_stream.str();

          LOGS_DEFAULT(ERROR) << message;
          status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, message);
        });
  }
  ORT_THROW_IF_ERROR(status);
}

static std::once_flag load_triton_kernel_flag;

}  // end of namespace

void LoadRocmTritonKernel() {
  // load kernel should be called only once
  std::call_once(load_triton_kernel_flag, TryToLoadKernel);
}

Status LaunchTritonKernel(hipStream_t stream, std::string fname, int grid0, int grid1, int grid2, void *args, size_t args_size) {
  if (rocm_triton_kernel_map.count(fname) == 0) {
    // return unsupported status when not found function name in registry
    // this error status will be used by tunableOp
    std::ostringstream message_stream;
    message_stream << "can't find triton kernel name: " << fname;
    std::string message = message_stream.str();
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(true, message);
  }
  auto metadata = rocm_triton_kernel_map[fname];
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                      HIP_LAUNCH_PARAM_END};

  HIP_CHECK(hipModuleLaunchKernel(metadata.func,
                                  grid0, grid1, grid2,
                                  HIP_WARP_SIZE * metadata.num_warps, 1, 1,
                                  metadata.shared_mem_size,
                                  stream,
                                  nullptr,
                                  (void**)&config));
  return Status::OK();
}

}  // end of rocm
}  // end of onnxruntime
