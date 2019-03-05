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
      .Reference("Sin", 8);

  ONNX_GRADIENT_OPERATOR_SCHEMA(MulGrad)
      .NumInputs(2, 3)
      .NumOutputs(1, 2)
      .Reference("Mul", 8);

  ONNX_GRADIENT_OPERATOR_SCHEMA(FlattenGrad)
      .NumInputs(1)
      .NumOutputs(1)
      .Reference("Flatten", 8);

  ONNX_GRADIENT_OPERATOR_SCHEMA(UnsqueezeGrad)
      .NumInputs(1)
      .NumOutputs(1)
      .Reference("Unsqueeze", 8);

  ONNX_GRADIENT_OPERATOR_SCHEMA(ReluGrad)
      .NumInputs(2)
      .NumOutputs(1)
      .Reference("Relu", 8);

  ONNX_GRADIENT_OPERATOR_SCHEMA(AddGrad)
      .NumInputs(1)
      .NumOutputs(1, 2)
      .Reference("Add", 8);

  ONNX_GRADIENT_OPERATOR_SCHEMA(MatMulGrad)
      .NumInputs(2, 3)
      .NumOutputs(1, 2)
      .Reference("MatMul", 8);

  ONNX_GRADIENT_OPERATOR_SCHEMA(SubGrad)
      .NumInputs(1)
      .NumOutputs(1, 2)
      .Reference("Sub", 8);

  ONNX_GRADIENT_OPERATOR_SCHEMA(PowGrad)
      .NumInputs(3)
      .NumOutputs(1, 2)
      .Reference("Pow", 8);

  ONNX_GRADIENT_OPERATOR_SCHEMA(ReduceMeanGrad)
      .NumInputs(1)
      .NumOutputs(1)
      .Reference("ReduceMean", 8);

  ONNX_GRADIENT_OPERATOR_SCHEMA(SigmoidGrad)
      .NumInputs(2)
      .NumOutputs(1)
      .Reference("Sigmoid", 9);

  ONNX_GRADIENT_OPERATOR_SCHEMA(SoftmaxGrad)
      .NumInputs(2)
      .NumOutputs(1)
      .Reference("Softmax", 9);
}
}  // namespace training
}  // namespace onnxruntime
