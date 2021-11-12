#pragma once

#include <CL/cl.hpp>

#include <cstdio>
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "opencl_execution_provider.h"

#define ONNX_OPENCL_OPERATOR_KERNEL(name, ver, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(name, kOnnxDomain, ver, kOpenCLExecutionProvider, builder, __VA_ARGS__)

#define OPENCL_EXEC_PROVIDER_FROM_INFO(info) \
  const_cast<OpenCLExecutionProvider*>(static_cast<const OpenCLExecutionProvider*>((info).GetExecutionProvider()))

#define CL_BUFFER_FROM_TENSOR(TENSOR) (*const_cast<cl::Buffer*>(static_cast<const cl::Buffer*>((TENSOR).DataRaw())))
#define CL_IMAGE2D_FROM_TENSOR(TENSOR) (*const_cast<cl::Image2D*>(static_cast<const cl::Image2D*>((TENSOR).DataRaw())))

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

namespace onnxruntime {
namespace opencl {

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

const char* GetErrorString(cl_int error_code);

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
    return {512, CeilDiv(shape[0], 4 * 512)};
  }

  static Image2DDesc PackFromTensor2D(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 2);
    return {CeilDiv(shape[0], 4), shape[1]};
  }

  static Image2DDesc PackFromTensorNCHW(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 4);
    int64_t W = shape[0];
    int64_t H = shape[1];
    int64_t C = shape[2];
    int64_t N = shape[3];
    int64_t c = 4;
    int64_t Cc = CeilDiv(C, c);
    return {Cc * W, N * H};
  }

  // NCHWc is actually Tensor of shape N[C/c]HWc then packed as NH C/cWc
  static Image2DDesc PackFromTensorNCHWc(const TensorShape& shape) {
    ORT_ENFORCE(shape.NumDimensions() == 5);
    int64_t W = shape[0];
    int64_t H = shape[1];
    int64_t Cc = shape[2];
    int64_t N = shape[3];
    int64_t c = shape[4];
    ORT_ENFORCE(c == 4);
    return {Cc * W, N * H};
  }

  int64_t Height() const {
    return second;
  }

  int64_t Width() const {
    return first;
  }

  TensorShape AsTensorShape() const {
    return {Height(), Width()};
  }
};

class KernelLauncher {
  cl::Kernel kernel_;
  cl_uint index_;

 public:
  explicit KernelLauncher(const cl::Kernel& kernel) : kernel_{kernel}, index_{0} {}
  const cl::Kernel& Kernel() const { return kernel_; }

  template <typename T>
  KernelLauncher& setArg(T&& arg) {
    OPENCL_CHECK_ERROR(kernel_.setArg(index_++, std::forward<T>(arg)));
    return *this;
  }

  KernelLauncher& setBuffer(const Tensor& arg) {
    OPENCL_CHECK_ERROR(kernel_.setArg<cl::Buffer>(index_++, CL_BUFFER_FROM_TENSOR(arg)));
    return *this;
  }

  KernelLauncher& setImage2D(const Tensor& arg) {
    OPENCL_CHECK_ERROR(kernel_.setArg<cl::Image2D>(index_++, CL_IMAGE2D_FROM_TENSOR(arg)));
    return *this;
  }

  void Launch(const cl::CommandQueue& queue, const cl::NDRange& global, const cl::NDRange& local = cl::NullRange) {
    OPENCL_CHECK_ERROR(queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local));
  }
};

}  // namespace opencl
}  // namespace onnxruntime
