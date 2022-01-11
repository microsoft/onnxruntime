#pragma once

#include <CL/opencl.h>
#include <Tracy.hpp>
#include <TracyOpenCL.hpp>

#include <cstdio>
#include <iomanip>
#include <sstream>
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "opencl_forward_decl.h"
#include "opencl_execution_provider.h"

#ifdef NDEBUG
#define USE_CL_CHECKED_CAST 0
#else
#define USE_CL_CHECKED_CAST 1
#endif

#define ONNX_OPENCL_OPERATOR_KERNEL(name, ver, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(name, kOnnxDomain, ver, kOpenCLExecutionProvider, builder, __VA_ARGS__)

#define OPENCL_EXEC_PROVIDER_FROM_INFO(info) \
  const_cast<OpenCLExecutionProvider*>(static_cast<const OpenCLExecutionProvider*>((info).GetExecutionProvider()))

#define TO_STRING_(T) #T
#define TO_STRING(T) TO_STRING_(T)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ORT_RETURN_IF_CL_ERROR(error_code, ...)                                          \
  if ((error_code) != CL_SUCCESS) {                                                      \
    std::ostringstream oss;                                                              \
    oss << __FILE__ ":" TO_STRING(__LINE__)                                              \
        << "\nOpenCL Error Code  : " << (int)(error_code)                                \
        << "\n       Error String: " << onnxruntime::opencl::GetErrorString(error_code); \
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, oss.str(),                              \
                           ::onnxruntime::MakeString(__VA_ARGS__));                      \
  }

#define ORT_THROW_IF_CL_ERROR(error_code, ...)                                           \
  if ((error_code) != CL_SUCCESS) {                                                      \
    std::ostringstream oss;                                                              \
    oss << __FILE__ ":" TO_STRING(__LINE__)                                              \
        << "\nOpenCL Error Code  : " << (int)(error_code)                                \
        << "\n       Error String: " << onnxruntime::opencl::GetErrorString(error_code); \
    ORT_THROW(oss.str(), ::onnxruntime::MakeString(__VA_ARGS__));                        \
  }

#if USE_CL_CHECKED_CAST
#define CL_BUFFER_FROM_TENSOR(TENSOR) [&]() {                                                             \
  cl_mem ptr = const_cast<cl_mem>(static_cast<const std::remove_pointer_t<cl_mem>*>((TENSOR).DataRaw())); \
  cl_mem_object_type ret;                                                                                 \
  ORT_THROW_IF_CL_ERROR(clGetMemObjectInfo(ptr, CL_MEM_TYPE, sizeof(cl_mem_object_type), &ret, nullptr)); \
  ORT_ENFORCE(ret == CL_MEM_OBJECT_BUFFER, ptr, " is not Buffer");                                        \
  return ptr;                                                                                             \
}()

#define CL_IMAGE2D_FROM_TENSOR(TENSOR) [&]() {                                                            \
  cl_mem ptr = const_cast<cl_mem>(static_cast<const std::remove_pointer_t<cl_mem>*>((TENSOR).DataRaw())); \
  cl_mem_object_type ret;                                                                                 \
  ORT_THROW_IF_CL_ERROR(clGetMemObjectInfo(ptr, CL_MEM_TYPE, sizeof(cl_mem_object_type), &ret, nullptr)); \
  ORT_ENFORCE(ret == CL_MEM_OBJECT_IMAGE2D, ptr, " is not Image2D");                                      \
  return ptr;                                                                                             \
}()
#else
#define CL_BUFFER_FROM_TENSOR(TENSOR) const_cast<cl_mem>(static_cast<const std::remove_pointer_t<cl_mem>*>((TENSOR).DataRaw()))
#define CL_IMAGE2D_FROM_TENSOR(TENSOR) const_cast<cl_mem>(static_cast<const std::remove_pointer_t<cl_mem>*>((TENSOR).DataRaw()))
#endif

#define VLOG_CL_NODE()                                          \
  VLOGS_DEFAULT(0) << "[CL] Node: " << context->GetNodeName()   \
                   << ", num inputs: " << context->InputCount() \
                   << ", num outputs: " << context->OutputCount()
#define VLOG_CL_BUFFER(desc, tensor_ptr)                                      \
  VLOGS_DEFAULT(0) << "[CL]  " << std::setfill(' ') << std::setw(9) << (desc) \
                   << " shape " << (tensor_ptr)->Shape()                      \
                   << "Buffer(" << CL_BUFFER_FROM_TENSOR(*(tensor_ptr)) << ")"
#define VLOG_CL_IMAGE2D(desc, tensor_ptr)                                     \
  VLOGS_DEFAULT(0) << "[CL]  " << std::setfill(' ') << std::setw(9) << (desc) \
                   << " shape " << (tensor_ptr)->Shape()                      \
                   << "Image2D(" << CL_IMAGE2D_FROM_TENSOR(*(tensor_ptr)) << ")"

namespace onnxruntime {
namespace opencl {

class NDRange {
  uint8_t size;
  size_t values[3];

 public:
  NDRange() : size(0), values{0, 0, 0} {}

  template<typename T>
  explicit NDRange(T x) : size(1), values{static_cast<size_t>(x), 0, 0} {}

  template<typename T1, typename T2>
  NDRange(T1 x, T2 y) : size(2), values{static_cast<size_t>(x), static_cast<size_t>(y), 0} {}

  template<typename T1, typename T2, typename T3>
  NDRange(T1 x, T2 y, T3 z) : size(3), values{static_cast<size_t>(x), static_cast<size_t>(y), static_cast<size_t>(z)} {}

  uint8_t Size() const { return size; }

  const size_t* Data() const {
    if (size != 0) {
      return values;
    }
    return nullptr;
  }

  inline std::string ToString() const {
    if (size == 0) {
      return "[<unspecified>]";
    }
    if (size == 1) {
      return onnxruntime::MakeString("[", values[0], "]");
    }
    if (size == 2) {
      return onnxruntime::MakeString("[", values[0], ",", values[1], "]");
    }
    return onnxruntime::MakeString("[", values[0], ",", values[1], ",", values[2], "]");
  }
};

const char* GetErrorString(cl_int error_code);
cl_program LoadProgram(cl_context ctx, cl_device_id dev, const std::string& src, bool use_fp16);
cl_program LoadProgram(cl_context ctx, cl_device_id dev, const char* src, size_t src_len, bool use_fp16);
cl_kernel LoadKernel(cl_program program, const char* name);

// NOTE: for OrtDevice ctor
struct CLMemType {
  static constexpr OrtDevice::MemoryType OPENCL_BUFFER = OrtDevice::MemType::DEFAULT;
  static constexpr OrtDevice::MemoryType OPENCL_IMAGE_2D = 5;
};

// NOTE: for opencl internal definition.
enum MemoryKind : uint8_t {
  Buffer = CLMemType::OPENCL_BUFFER,
  Image2D = CLMemType::OPENCL_IMAGE_2D,
};

template <typename T, typename E = std::enable_if_t<std::is_integral_v<T>>>
T CeilDiv(T a, T b) {
  return (a - 1) / b + 1;
}

template <typename T1, typename T2, typename E = std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>>>
T1 CeilDiv(T1 a, T2 b) {
  return (a - 1) / b + 1;
}

template <typename T1, typename T2, typename E = std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>>>
T1 RoundToMultiple(T1 a, T2 m) {
  return CeilDiv(a, m) * m;
}

class Image2DDesc : private std::pair<int64_t, int64_t> {
 public:
  using pair::pair;

  static Image2DDesc PackFromTensor(const TensorShape& shape) {
    switch (shape.NumDimensions()) {
      case 1:
        return PackFromTensor1D(shape);
      case 2:
        return PackFromTensor2D(shape);
      case 4:
        return PackFromTensorNCHW(shape);
      case 5:
        return PackFromTensorNCHWc(shape);
      default:
        return {0, 0};
    }
  }

  static Image2DDesc PackFromTensor1D(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 1);
    //
    return {1024, CeilDiv(shape[0], 4 * 1024)};
  }

  static Image2DDesc PackFromTensor2D(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 2);
    return {CeilDiv(shape[0], 4), shape[1]};
  }

  static Image2DDesc PackFromTensorNCHW(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 4);
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t c = 4;
    int64_t Cc = CeilDiv(C, c);
    return {Cc * W, N * H};
  }

  // NCHWc is actually Tensor of shape N[C/c]HWc then packed as NH C/cWc
  static Image2DDesc PackFromTensorNCHWc(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 5);
    int64_t N = shape[0];
    int64_t Cc = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t c = shape[4];
    ORT_ENFORCE(c == 4);
    return {Cc * W, N * H};
  }

  static Image2DDesc PackFromConv2DWeight(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 4);
    int64_t C_o = shape[0];
    int64_t C_i = shape[1];
    int64_t K_h = shape[2];
    int64_t K_w = shape[3];
    return {C_i, CeilDiv(C_o, 4) * K_h * K_w};
  }

  static Image2DDesc PackFromDepthwiseConv2DWeight(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 4);
    int64_t C_o = shape[0];
    int64_t C_i = shape[1];
    int64_t K_h = shape[2];
    int64_t K_w = shape[3];
    return {K_h * K_w * C_i, CeilDiv(C_o, 4)};
  }

  auto Height() const {
    return second;
  }

  auto Width() const {
    return first;
  }

  size_t UHeight() const {
    return static_cast<size_t>(second);
  }

  size_t UWidth() const {
    return static_cast<size_t>(first);
  }

  TensorShape AsTensorShape() const {
    return {Width(), Height()};
  }

  NDRange AsNDRange() const {
    return {UWidth(), UHeight()};
  }
};

class KernelLauncher {
 public:
  explicit KernelLauncher(cl_kernel kernel) : kernel_{kernel}, index_{0}, err_{CL_SUCCESS} {}
  cl_kernel Kernel() const { return kernel_; }

#define SKIP_IF_ERRORED(expr) \
  if (err_ == CL_SUCCESS) {   \
    err_ = (expr);            \
    err_index_ = index_;      \
  }

  template <typename T, typename E = std::is_convertible<T, cl_int>>
  KernelLauncher& setInt2(T v1, T v2) {
    cl_int tmp[2] = {static_cast<cl_int>(v1), static_cast<cl_int>(v2)};
    SKIP_IF_ERRORED(clSetKernelArg(kernel_, index_, sizeof(tmp), tmp));
    index_ += 1;
    return *this;
  }

  template <typename T, typename E = std::is_convertible<T, cl_int>>
  KernelLauncher& setInt3(T v1, T v2, T v3) {
    cl_int3 tmp{{static_cast<cl_int>(v1), static_cast<cl_int>(v2), static_cast<cl_int>(v3)}};
    SKIP_IF_ERRORED(clSetKernelArg(kernel_, index_, sizeof(tmp), &tmp));
    index_ += 1;
    return *this;
  }

  template <typename T, typename E = std::is_convertible<T, cl_int>>
  KernelLauncher& setInt4(T v1, T v2, T v3, T v4) {
    cl_int tmp[4] = {static_cast<cl_int>(v1), static_cast<cl_int>(v2), static_cast<cl_int>(v3), static_cast<cl_int>(v4)};
    SKIP_IF_ERRORED(clSetKernelArg(kernel_, index_, sizeof(tmp), tmp));
    index_ += 1;
    return *this;
  }

  template <typename T>
  KernelLauncher& setArg(const T& arg) {
    SKIP_IF_ERRORED(clSetKernelArg(kernel_, index_, sizeof(T), &arg));
    index_ += 1;
    return *this;
  }

  KernelLauncher& setBuffer(cl_mem arg) {
    SKIP_IF_ERRORED(clSetKernelArg(kernel_, index_, sizeof(cl_mem), &arg));
    index_ += 1;
    return *this;
  }

  KernelLauncher& setBuffer(const Tensor& arg) {
    return setBuffer(CL_BUFFER_FROM_TENSOR(arg));
  }

  template <typename T1, typename T2>
  KernelLauncher& setBuffers(T1&& arg, T2&& other) {
    setBuffer(std::forward<T1>(arg));
    return setBuffer(std::forward<T2>(other));
  }

  template <typename T, typename... Ts>
  KernelLauncher& setBuffers(T&& arg, Ts&&... args) {
    setBuffer(std::forward<T>(arg));
    return setBuffers(std::forward<Ts>(args)...);
  }

  KernelLauncher& setImage2D(cl_mem arg) {
    SKIP_IF_ERRORED(clSetKernelArg(kernel_, index_, sizeof(cl_mem), &arg));
    index_ += 1;
    return *this;
  }

  KernelLauncher& setImage2D(const Tensor& arg) {
    return setImage2D(CL_IMAGE2D_FROM_TENSOR(arg));
  }

  template <typename T1, typename T2>
  KernelLauncher& setImage2Ds(T1&& arg, T2&& other) {
    setImage2D(std::forward<T1>(arg));
    return setImage2D(std::forward<T2>(other));
  }

  template <typename T, typename... Ts>
  KernelLauncher& setImage2Ds(T&& arg, Ts&&... args) {
    setImage2D(std::forward<T>(arg));
    return setImage2Ds(std::forward<Ts>(args)...);
  }

  Status Launch(const OpenCLExecutionProvider& exec, const NDRange& global, const NDRange& local = {});

 private:
  inline std::string GetKernelFunctionName() {
    size_t ret_size;
    clGetKernelInfo(kernel_, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &ret_size);
    std::string ret(ret_size, '\0');
    clGetKernelInfo(kernel_, CL_KERNEL_FUNCTION_NAME, ret.size(), ret.data(), nullptr);
    return ret;
  }

#undef SKIP_IF_ERRORED
  cl_kernel kernel_;
  cl_uint index_;
  cl_int err_;
  cl_uint err_index_;
};

}  // namespace opencl
}  // namespace onnxruntime
