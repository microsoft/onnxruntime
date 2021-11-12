#include "elementwise.h"

#include <sstream>

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

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

class Add : public OpenCLKernel {
 public:
  explicit Add(const OpKernelInfo& info) : OpenCLKernel(info) {
    std::cout << "Init Add (OpenCLKernel)" << std::endl;
    LoadProgram(GetKernelSrc(("Add"), "float", ("(X)+(Y)")));
    LoadKernel("Add");
  };

  Status Compute(OpKernelContext* context) const override {
    std::cerr << "Node: " << context->GetNodeName()
              << ", num inputs: " << context->InputCount()
              << ", num outputs: " << context->OutputCount() << std::endl;
    const auto* a = context->Input<Tensor>(0);
    const auto* b = context->Input<Tensor>(1);
    const auto* c = context->Output(0, a->Shape());
    std::cerr << " Input[0] shape " << a->Shape() << " " << a << "--> cl::Image(" << CL_IMAGE2D_FROM_TENSOR(*a)() << ")\n";
    std::cerr << " Input[1] shape " << b->Shape() << " " << b << "--> cl::Image(" << CL_IMAGE2D_FROM_TENSOR(*b)() << ")\n";
    std::cerr << " Output[0] shape " << c->Shape() << " " << c << "--> cl::Image(" << CL_IMAGE2D_FROM_TENSOR(*c)() << ")\n";

    size_t n = a->Shape().Size();

    KernelLauncher{GetKernel("Add")}
        .setImage2D(*a)
        .setImage2D(*b)
        .setImage2D(*c)
        .setArg<cl_int>(n)
        .Launch(GetCommandQueue(), {n});

    return Status::OK();
  }
};

ONNX_OPENCL_OPERATOR_KERNEL(
    Add,
    7,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 1),
    Add)

}  // namespace opencl
}  // namespace onnxruntime
