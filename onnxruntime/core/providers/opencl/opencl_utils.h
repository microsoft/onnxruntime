#pragma once

#include <CL/cl.hpp>

#include <cstdio>

namespace onnxruntime {
namespace opencl {
const char* GetErrorString(cl_int error_code);
}
}  // namespace onnxruntime

#define TO_STRING_(T) #T
#define TO_STRING(T) TO_STRING_(T)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define OPENCL_CHECK_ERROR(error_code)                                                             \
  if ((error_code) != CL_SUCCESS) {                                                                \
    fprintf(stderr, __FILE__ ":" TO_STRING(__LINE__) "\n");                                        \
    fprintf(stderr, "OpenCL Error Code  : %d\n", (int)(error_code));                               \
    fprintf(stderr, "       Error String: %s\n", onnxruntime::opencl::GetErrorString(error_code)); \
    exit(-1);                                                                                      \
  }
