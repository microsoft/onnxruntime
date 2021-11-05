#include "elementwise.h"

#include <sstream>

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"
#include "core/providers/opencl/opencl_onnxruntime_utils.h"

namespace {

#define CONTENT_NAME elementwise_kernel_src
#include "opencl_generated/math/kernels/elementwise.cl.inc"
#undef CONTENT_NAME

std::string GetKernelSrc(const std::string& name_define, const std::string& type_define, const std::string& op_define) {
  std::stringstream oss;
  oss << "#define NAME " << name_define << "\n"
      << "#define T " << type_define << "\n"
      << "#define OP(X,Y) " << op_define << "\n"
      << elementwise_kernel_src;
  return oss.str();
}

}  // namespace

namespace onnxruntime {
namespace opencl {

#define ELEMENT_WISE_OP_IMPL(CLASS_NAME, OP_DEFINE)                                 \
  class CLASS_NAME : public OpenCLKernel {                                          \
   public:                                                                          \
    explicit CLASS_NAME(const OpKernelInfo& info) : OpenCLKernel(info) {            \
      std::cout << "Init " #CLASS_NAME " (OpenCLKernel)" << std::endl;              \
      LoadProgram(GetKernelSrc((#CLASS_NAME), "float", (OP_DEFINE)));               \
      LoadKernel(#CLASS_NAME);                                                      \
    };                                                                              \
                                                                                    \
    Status Compute(OpKernelContext* context) const override {                       \
      std::cout << "Node: " << context->GetNodeName()                               \
                << ", num inputs: " << context->InputCount()                        \
                << ", num outputs: " << context->OutputCount() << std::endl;        \
      const auto* a = context->Input<Tensor>(0);                                    \
      const auto* b = context->Input<Tensor>(1);                                    \
      const auto* c = context->Output(0, a->Shape());                               \
                                                                                    \
      size_t n = a->Shape().Size();                                                 \
                                                                                    \
      auto kernel = GetKernel(#CLASS_NAME);                                         \
      OPENCL_CHECK_ERROR(kernel.setArg(0, CL_BUFFER_FROM_TENSOR(*a)));              \
      OPENCL_CHECK_ERROR(kernel.setArg(1, CL_BUFFER_FROM_TENSOR(*b)));              \
      OPENCL_CHECK_ERROR(kernel.setArg(2, CL_BUFFER_FROM_TENSOR(*c)));              \
      OPENCL_CHECK_ERROR(kernel.setArg<cl_int>(3, n));                              \
      OPENCL_CHECK_ERROR(GetCommandQueue().enqueueNDRangeKernel(kernel, {0}, {n})); \
      return Status::OK();                                                          \
    }                                                                               \
  };                                                                                \
                                                                                    \
  ONNX_OPENCL_OPERATOR_KERNEL(                                                      \
      CLASS_NAME,                                                                   \
      7,                                                                            \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      CLASS_NAME)

ELEMENT_WISE_OP_IMPL(Add, "(X)+(Y)");
ELEMENT_WISE_OP_IMPL(Sub, "(X)-(Y)");
ELEMENT_WISE_OP_IMPL(Mul, "(X)*(Y)");
ELEMENT_WISE_OP_IMPL(Div, "(X)/(Y)");

}  // namespace opencl
}  // namespace onnxruntime
