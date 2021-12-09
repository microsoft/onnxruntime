#include "elementwise.h"

#include <sstream>

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace {

#define CONTENT_NAME elementwise_kernel_src
#include "opencl_generated/math/kernels/elementwise_image2d.cl.inc"

inline std::string GetKernelSrc(const std::string& name_define, const std::string& op_define) {
  std::stringstream oss;
  oss << "#define NAME " << name_define << "\n"
      << "#define OP(X,Y,OUT) " << op_define << "\n"
      << elementwise_kernel_src;
  return oss.str();
}

}  // namespace

namespace onnxruntime {
namespace opencl {

#define ELEMENT_WISE_OP_IMPL(CLASS_NAME, OP_DEFINE)                         \
  class CLASS_NAME : public OpenCLKernel {                                  \
   public:                                                                  \
    explicit CLASS_NAME(const OpKernelInfo& info) : OpenCLKernel(info) {    \
      VLOGS_DEFAULT(0) << "[CL] Init " #CLASS_NAME "  (OpenCLKernel)";      \
      LoadProgram(GetKernelSrc((#CLASS_NAME), (OP_DEFINE)));                \
      LoadKernel(#CLASS_NAME);                                              \
    };                                                                      \
                                                                            \
    Status Compute(OpKernelContext* context) const override {               \
      VLOG_CL_NODE();                                                       \
      const auto* a = context->Input<Tensor>(0);                            \
      const auto* b = context->Input<Tensor>(1);                            \
      const auto* c = context->Output(0, a->Shape());                       \
      VLOG_CL_IMAGE2D("Input[0]", a);                                       \
      VLOG_CL_IMAGE2D("Input[1]", b);                                       \
      VLOG_CL_IMAGE2D("Output[0]", c);                                      \
                                                                            \
      auto desc = Image2DDesc::PackFromTensor(a->Shape());                  \
                                                                            \
      ORT_RETURN_IF_ERROR(                                                  \
          KernelLauncher{GetKernel(#CLASS_NAME)}                            \
              .setImage2D(*a)                                               \
              .setImage2D(*b)                                               \
              .setImage2D(*c)                                               \
              .Launch(GetCommandQueue(), {desc.UWidth(), desc.UHeight()})); \
                                                                            \
      return Status::OK();                                                  \
    }                                                                       \
  };                                                                        \
                                                                            \
  ONNX_OPENCL_OPERATOR_KERNEL(                                              \
      CLASS_NAME,                                                           \
      7,                                                                    \
      KernelDefBuilder()                                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())        \
          .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0)       \
          .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 1)       \
          .OutputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0),     \
      CLASS_NAME)

ELEMENT_WISE_OP_IMPL(Add, "(OUT)=(X)+(Y)");
ELEMENT_WISE_OP_IMPL(Sub, "(OUT)=(X)-(Y)");
ELEMENT_WISE_OP_IMPL(Mul, "(OUT)=(X)*(Y)");
ELEMENT_WISE_OP_IMPL(Div, "(OUT)=(X)/(Y)");

}  // namespace opencl
}  // namespace onnxruntime
