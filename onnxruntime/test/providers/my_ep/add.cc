#include "add.h"
#include "my_kernel.h"

namespace onnxruntime {
Status Add::Compute(OpKernelContext* context) const {
  const Tensor* a = context->Input<Tensor>(0);
  const Tensor* b = context->Input<Tensor>(1);
  ORT_ENFORCE(a->Shape() == b->Shape(), "a and b must have the same shape.");

  // calcuate output shape
  std::vector<int64_t> output_shape;
  my_kernel_lib::DataType a_type = my_kernel_lib::DataType::kFloat;
  my_kernel_lib::DataType b_type = my_kernel_lib::DataType::kFloat;
  my_kernel_lib::DataType output_type;
  std::vector<int64_t> a_shape;
  a_shape.assign(a->Shape().GetDims().begin(), a->Shape().GetDims().end());
  std::vector<int64_t> b_shape;
  b_shape.assign(b->Shape().GetDims().begin(), b->Shape().GetDims().end());
  auto status = AddKernelTypeShapeInference(a_type, a_shape,
                                            b_type, b_shape,
                                            &output_type, &output_shape);
  if (status != my_kernel_lib::Status::kOK) {
    throw std::runtime_error("add: shape inference failed");
  }

  TensorShape ort_output_shape(output_shape);
  Tensor* c = context->Output(0, ort_output_shape);
  const auto* a_data = a->Data<float>();
  const auto* b_data = b->Data<float>();
  auto* c_data = c->MutableData<float>();
  // invoke add kernel
  status = my_kernel_lib::AddKernel(a_data, a_shape,
                                    b_data, b_shape,
                                    c_data, output_shape);
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    Add,
    kOnnxDomain,
    13,
    kMyProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<float>()),
    Add);

}  // namespace onnxruntime
