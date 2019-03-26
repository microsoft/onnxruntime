#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/training/gradient_schema_defs.h"
#include "core/training/gradient_op_schema.h"
#include "core/graph/op.h"

namespace onnxruntime {
namespace training {

using namespace ONNX_NAMESPACE;
void RegisterGradientSchemas() {
  ONNX_GRADIENT_OPERATOR_SCHEMA(SinGrad)
      .NumInputs(2)
      .NumOutputs(1)
      .Reference("Sin");

  ONNX_GRADIENT_OPERATOR_SCHEMA(MulGrad)
      .NumInputs(2, 3)
      .NumOutputs(1, 2)
      .Reference("Mul");

  ONNX_GRADIENT_OPERATOR_SCHEMA(FlattenGrad)
      .NumInputs(1)
      .NumOutputs(1)
      .Reference("Flatten");

  ONNX_GRADIENT_OPERATOR_SCHEMA(UnsqueezeGrad)
      .NumInputs(1)
      .NumOutputs(1)
      .Reference("Unsqueeze");

  ONNX_GRADIENT_OPERATOR_SCHEMA(ReluGrad)
      .NumInputs(2)
      .NumOutputs(1)
      .Reference("Relu");

  ONNX_GRADIENT_OPERATOR_SCHEMA(AddGrad)
      .NumInputs(1)
      .NumOutputs(1, 2)
      .Reference("Add");

  ONNX_GRADIENT_OPERATOR_SCHEMA(MatMulGrad)
      .NumInputs(2, 3)
      .NumOutputs(1, 2)
      .Reference("MatMul");

  ONNX_GRADIENT_OPERATOR_SCHEMA(SubGrad)
      .NumInputs(1)
      .NumOutputs(1, 2)
      .Reference("Sub");

  ONNX_GRADIENT_OPERATOR_SCHEMA(PowGrad)
      .NumInputs(3)
      .NumOutputs(1, 2)
      .Reference("Pow");

  ONNX_GRADIENT_OPERATOR_SCHEMA(ReduceMeanGrad)
      .NumInputs(1)
      .NumOutputs(1)
      .Reference("ReduceMean");

  ONNX_GRADIENT_OPERATOR_SCHEMA(SigmoidGrad)
      .NumInputs(2)
      .NumOutputs(1)
      .Reference("Sigmoid");

  ONNX_GRADIENT_OPERATOR_SCHEMA(SoftmaxGrad)
      .NumInputs(2)
      .NumOutputs(1)
      .Reference("Softmax");

  ONNX_GRADIENT_OPERATOR_SCHEMA(AveragePoolGrad)
      .NumInputs(3)
      .NumOutputs(1)
      .Reference("AveragePool");

  ONNX_GRADIENT_OPERATOR_SCHEMA(MaxPoolGrad)
      .Input(0, "dY", "Gradient of output, Y", "T")
      .Input(1, "Indices", "Indices tensor from max pooling across the input tensor.", "I")
      .Output(0, "dX", "Gradient of input, X", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "I",
          {"tensor(int64)"},
          "Constrain index tensor to int64")
      .ReferenceAttributes("MaxPool");

  ONNX_GRADIENT_OPERATOR_SCHEMA(ConvGrad)
      .NumInputs(2, 3)
      .NumOutputs(1, 3)
      .Reference("Conv");

  ONNX_GRADIENT_OPERATOR_SCHEMA(LRNGrad)
      .NumInputs(3)
      .NumOutputs(1)
      .Reference("LRN");

  ONNX_GRADIENT_OPERATOR_SCHEMA(DropoutGrad)
      .NumInputs(1, 2)
      .NumOutputs(1)
      .Reference("Dropout");
}
}  // namespace training
}  // namespace onnxruntime
