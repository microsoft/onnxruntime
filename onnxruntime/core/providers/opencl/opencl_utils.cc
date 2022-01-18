#include "opencl_utils.h"
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

namespace {
#define CONTENT_NAME prelude_f16_src
#include "opencl_generated/kernels/prelude_f16.cl.inc"
#define CONTENT_NAME prelude_f32_src
#include "opencl_generated/kernels/prelude_f32.cl.inc"
}  // namespace

cl_program LoadProgram(cl_context ctx, cl_device_id dev, const std::string& src, bool use_fp16) {
  return LoadProgram(ctx, dev, src.data(), src.size(), use_fp16);
}

cl_program LoadProgram(cl_context ctx, cl_device_id dev, const char* src, size_t src_len, bool use_fp16) {
  std::ostringstream oss;
  if (use_fp16) {
    oss << std::string(prelude_f16_src, prelude_f16_src_len) << "\n";
  } else {
    oss << std::string(prelude_f32_src, prelude_f32_src_len) << "\n";
  }
  oss << std::string(src, src_len);
  auto full_src = oss.str();
  const auto* full_src_c = full_src.c_str();
  auto full_src_size = full_src.size();

  cl_int err{};
  auto* program = clCreateProgramWithSource(ctx, 1, &full_src_c, &full_src_size, &err);
  ORT_THROW_IF_CL_ERROR(err);

  // Specially handle this error, we need compiler error message here.
  err = clBuildProgram(program, 1, &dev, "", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t ret_size;
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ret_size);
    std::string log(ret_size + 1, '\0');
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
    // LOGS_DEFAULT(ERROR) << "\nKernel Source:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
    std::cout << "\nKernel Source:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
                        << full_src
                        << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
                        << "\nBuild Log:\n"
                        << log
                        << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
    ORT_THROW("\nOpenCL Error Code  : ", static_cast<int>(err), "\n       Error String: ", onnxruntime::opencl::GetErrorString(err));
  }
  return program;
}

cl_kernel LoadKernel(cl_program program, const char* name) {
  LOGS_DEFAULT(INFO) << "[CL] Loading kernel " << name;
  cl_int err{};
  auto* kernel = clCreateKernel(program, name, &err);
  ORT_THROW_IF_CL_ERROR(err);
  return kernel;
}

Status KernelLauncher::Launch(const OpenCLExecutionProvider& exec, const NDRange& global, const NDRange& local) {
    ORT_RETURN_IF_CL_ERROR(err_, " on setting argument ", static_cast<int>(err_index_));
    VLOGS_DEFAULT(1) << "[CL] Launching " << GetKernelFunctionName()
                     << " with global work size: " << global.ToString()
                     << " local work size: " << local.ToString();

#ifdef TRACY_ENABLE
    cl_event kernel_launch_event;
    {
        ZoneScopedN("clEnqueueNDRangeKernel");
        TracyCLZone(const_cast<TracyCLCtx>(exec.GetTracyCLContext()), "clEnqueueNDRangeKernel");
        ORT_RETURN_IF_CL_ERROR(clEnqueueNDRangeKernel(exec.GetCommandQueue(), kernel_, global.Size(), nullptr, global.Data(), local.Data(), 0, nullptr, &kernel_launch_event));
        TracyCLZoneSetEvent(kernel_launch_event);
    }
#else
    ORT_RETURN_IF_CL_ERROR(clEnqueueNDRangeKernel(exec.GetCommandQueue(), kernel_, global.Size(), nullptr, global.Data(), local.Data(), 0, nullptr, nullptr));
#endif
    return exec.AfterCLLaunch();
}

std::unique_ptr<float, std::function<void(float*)>> mapImage2dToHost(const OpenCLExecutionProvider& exec, const Tensor& tensor, int width, int height,bool write) {
  cl_mem image = CL_IMAGE2D_FROM_TENSOR(tensor);
  return mapImage2dToHost(exec, image, width, height, write);
}

std::unique_ptr<float, std::function<void(float*)>> mapImage2dToHost(const OpenCLExecutionProvider& exec, cl_mem image, int width, int height, bool write ) {
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {width, height, 1};
  size_t image_row_pitch = width;
  size_t image_slice_pitch = width * height;
  cl_int err;
  bool flag = CL_MAP_READ;
  if (write) {
    flag = CL_MAP_WRITE;
  }
  float* hostptr = static_cast<float*>(clEnqueueMapImage(exec.GetCommandQueue(), image, CL_TRUE, flag, origin, region,
                                   &image_row_pitch, &image_slice_pitch,
                                   0, NULL, NULL, &err));
  if (err != CL_SUCCESS) {
    printf("map error\n ");
  }
  auto climage_deleter = [](cl_command_queue q, cl_mem image, float* p) {
    cl_int ret = clEnqueueUnmapMemObject(q, image, p, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      printf(" ");
    }
  };
  std::function<void(float*)> delete_binded = std::bind(climage_deleter, exec.GetCommandQueue(), image, std::placeholders::_1);
  return std::unique_ptr<float, std::function<void(float*)>>(hostptr, delete_binded);
}

}  // namespace opencl
}  // namespace onnxruntime
