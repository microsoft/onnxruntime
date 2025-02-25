// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_utils.h"
#include "opencl_execution_provider.h"
#include "opencl_program_manager.h"
#include <functional>
#include <memory>

namespace onnxruntime {
namespace opencl {

const char* GetErrorString(cl_int error_code) {
  switch (error_code) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case CL_COMPILE_PROGRAM_FAILURE:
      return "CL_COMPILE_PROGRAM_FAILURE";
    case CL_LINKER_NOT_AVAILABLE:
      return "CL_LINKER_NOT_AVAILABLE";
    case CL_LINK_PROGRAM_FAILURE:
      return "CL_LINK_PROGRAM_FAILURE";
    case CL_DEVICE_PARTITION_FAILED:
      return "CL_DEVICE_PARTITION_FAILED";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:
      return "CL_INVALID_PROPERTY";
    case CL_INVALID_IMAGE_DESCRIPTOR:
      return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS:
      return "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS:
      return "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
      return "CL_INVALID_DEVICE_PARTITION_COUNT";
    // case CL_INVALID_PIPE_SIZE:
    //   return "CL_INVALID_PIPE_SIZE";
    // case CL_INVALID_DEVICE_QUEUE:
    //   return "CL_INVALID_DEVICE_QUEUE";
    // case CL_INVALID_SPEC_ID:
    //   return "CL_INVALID_SPEC_ID";
    // case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
    //   return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
    default:
      return "Unknown";
  }
}
#define ORT_TUNE_RETURN_IF_CL_ERROR(error_code, ...) \
  if ((error_code) != CL_SUCCESS) {                  \
    return double(ULONG_MAX);                        \
  }

Status KernelLauncher::Launch(const OpenCLExecutionProvider& exec, const NDRange& global, const NDRange& local) {
  ORT_RETURN_IF_CL_ERROR(err_, " on setting argument ", static_cast<int>(err_index_));
  LOGS_DEFAULT(VERBOSE) << "Launching " << GetKernelFunctionName()
                          << " with global work size: " << global.ToString()
                          << " local work size: " << local.ToString();
  NDRange local_ (local);
  NDRange global_(global);
  opencl::TuneKernelWithTimeFunc func = [&exec, this](const NDRange& l, const NDRange& g) -> double {
    std::string build_errmsg = "[error occored in tuning local-size,local--gobal:" + l.ToString() + g.ToString();
    //try run, sometimes we got weird error;
    if (clEnqueueNDRangeKernel(exec.GetCommandQueueForTune(), kernel_, g.Size(), nullptr,
                               g.Data(), l.Data(), 0, nullptr, nullptr) != 0) {
      return double(ULONG_MAX);
    }
    double avg_tc = 0;
    constexpr int REPEAT_TUNING_RUNs = 5;
    std::vector<cl_event> events(REPEAT_TUNING_RUNs);
    for (size_t repeat = 0; repeat < REPEAT_TUNING_RUNs; ++repeat) {
      ORT_TUNE_RETURN_IF_CL_ERROR(clEnqueueNDRangeKernel(exec.GetCommandQueueForTune(), kernel_, g.Size(), nullptr,
                                                         g.Data(), l.Data(), 0, nullptr, &events[repeat]));
    }
    cl_int res = clWaitForEvents(REPEAT_TUNING_RUNs, events.data());
    ORT_TUNE_RETURN_IF_CL_ERROR(res, "event");
    for (size_t repeat = 0; repeat < REPEAT_TUNING_RUNs; ++repeat) {
      cl_ulong starttime, endtime;
      res = clGetEventProfilingInfo(events[repeat], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL);
      ORT_TUNE_RETURN_IF_CL_ERROR(res);
      res = clGetEventProfilingInfo(events[repeat], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL);
      ORT_TUNE_RETURN_IF_CL_ERROR(res);
      avg_tc += ((endtime - starttime) / 1000.0);
    }

    return avg_tc / REPEAT_TUNING_RUNs;
  };
  if (local.Size() == 0) {
    if (global.Size() == 2) {
      auto v_cache = const_cast<OpenCLExecutionProvider&>(exec).GetProgramManager().GetLocalSizeFromCache(kernel_, global);
      if (v_cache.has_value()) {
        local_ = v_cache.value();
      } else {
        local_ = exec.DefaultLocalWG2DOrTune(global, func);
        const_cast<OpenCLExecutionProvider&>(exec).GetProgramManager().SetLocalSizeToCache(kernel_, global, local_);
      }
      if (local_.Size() == global_.Size()) {
        // in some mobile GPUs, global size must be divisible by local-size
        absl::InlinedVector<size_t, 2> internalGlobalWS{global[0], global[1]};
        for (size_t i = 0; i < global_.Size(); ++i) {
          internalGlobalWS[i] = ROUND_UP(global_[i], std::max<size_t>(1, local_[i]));
        }
        global_ = NDRange(internalGlobalWS[0], internalGlobalWS[1]);
      }
    } else {
        //TODO 3D kernel
    }
  }
#ifdef TRACY_ENABLE
  cl_event kernel_launch_event;
  {
    TracyCLZoneTransient(const_cast<TracyCLCtx>(exec.GetTracyCLContext()), _tracy_cl_launch,
                         GetKernelFunctionName().c_str(), /*active=*/true);
    ORT_RETURN_IF_CL_ERROR(clEnqueueNDRangeKernel(exec.GetCommandQueue(), kernel_, global_.Size(), nullptr,
                                                  global_.Data(), local_.Data(), 0, nullptr, &kernel_launch_event));
    TracyCLNamedZoneSetEvent(_tracy_cl_launch, kernel_launch_event);
  }
#else
  ORT_RETURN_IF_CL_ERROR(clEnqueueNDRangeKernel(exec.GetCommandQueue(), kernel_, global_.Size(), nullptr,
                                                global_.Data(), local_.Data(), 0, nullptr, nullptr));
#endif
  return exec.AfterCLLaunch();
}

}  // namespace opencl
}  // namespace onnxruntime
