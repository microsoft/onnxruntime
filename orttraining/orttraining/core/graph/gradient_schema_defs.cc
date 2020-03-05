// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/op.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "gradient_schema_defs.h"
#include "onnx/defs/function.h"
#include <math.h>

namespace onnxruntime {
namespace training {

using namespace ONNX_NAMESPACE;

void AddRepeatedInputs(
    OpSchema& op_schema,
    const int start,
    const int count,
    const std::vector<std::string>& names,
    const std::vector<std::string>& descriptions,
    const std::vector<std::string>& type_strs,
    const OpSchema::FormalParameterOption param_option) {
  ORT_ENFORCE(names.size() == descriptions.size(),
              "Names and descriptions must be equal-length.");
  ORT_ENFORCE(names.size() == type_strs.size(),
              "Names and type_strs must be equal-length.");
  ORT_ENFORCE(param_option != OpSchema::Variadic,
              "param_option cannot be variadic.");
  ORT_ENFORCE(count > 0, "Count must be positive.");

  for (int i = 0; i < count; ++i) {
    const int input_index_start = start + i * static_cast<int>(names.size());
    // Repeat one group of names once.
    for (size_t j = 0; j < names.size(); ++j) {
      const int input_index = input_index_start + static_cast<int>(j);
      std::string modified_input_name = "__group_" + std::to_string(i) + "__" + names[j];
      ORT_ENFORCE(input_index >= static_cast<int>(op_schema.inputs().size()),
                  "Invalid redefinition of input ", input_index, " for OpSchema ", op_schema.Name());
      op_schema.Input(input_index, modified_input_name, descriptions[j], type_strs[j], param_option, false);
    }
  }
}

void AddRepeatedOutputs(
    OpSchema& op_schema,
    const int start,
    const int count,
    const std::vector<std::string>& names,
    const std::vector<std::string>& descriptions,
    const std::vector<std::string>& type_strs,
    const OpSchema::FormalParameterOption param_option) {
  ORT_ENFORCE(names.size() == descriptions.size(),
              "Names and descriptions must be equal-length.");
  ORT_ENFORCE(names.size() == type_strs.size(),
              "Names and type_strs must be equal-length.");
  ORT_ENFORCE(param_option != OpSchema::Variadic,
              "param_option cannot be variadic.");
  ORT_ENFORCE(count > 0, "Count must be positive.");

  for (int i = 0; i < count; ++i) {
    const int output_index_start = start + i * static_cast<int>(names.size());
    // Repeat one group of names once.
    for (int j = 0; j < static_cast<int>(names.size()); ++j) {
      const int output_index = output_index_start + j;
      std::string modified_output_name = "__group_" + std::to_string(i) + "__" + names[j];
      ORT_ENFORCE(output_index >= static_cast<int>(op_schema.outputs().size()),
                  "Invalid redefinition of output ", output_index, " for OpSchema ", op_schema.Name());
      op_schema.Output(output_index, modified_output_name, descriptions[j], type_strs[j], param_option, false);
    }
  }
}

// TODO: This is copied from onnx schemas. When the change is in and we update this can be removed.
// For Brevity documentation was not copied
OpSchema& RegisterLambOpSchema(OpSchema&& op_schema) {
  op_schema
      .SinceVersion(9)
      .Attr(
          "alpha",
          "Coefficient of previous gradient in running average.",
          AttributeProto::FLOATS,
          std::vector<float>(1024, 0.9f))
      .Attr(
          "beta",
          "Coefficient of previous squared gradient in running average."
          "The effective learning rate is computed by r = R / (1 + T * decay_factor). "
          "Default to 0 so that increasing update counts doesn't reduce the learning rate.",
          AttributeProto::FLOATS,
          std::vector<float>(1024, 0.999f))
      .Attr(
          "lambda",
          "Regularization coefficient of 0.5 * lambda * ||X||_2^2. Default to 0, "
          "which means no regularization.",
          AttributeProto::FLOATS,
          std::vector<float>(1024, 0.0f))
      .Attr(
          "epsilon",
          "Small scalar to avoid dividing by zero.",
          AttributeProto::FLOATS,
          std::vector<float>(1024, 1e-6f))
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float scalars.")
      .TypeConstraint(
          "T2",
          {"tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T3",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T4",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T_FP16",
          {"tensor(float16)"},
          "Constrain input types to float16 tensors.")
      .TypeConstraint(
          "T_GRAD_NORM",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T_BOOL",
          {"tensor(bool)"},
          "Constrain types to boolean tensors.")
      .TypeConstraint(
          "TInt64",
          {"tensor(int64)"},
          "Constrain update count to 64-bit integer");

  op_schema
      .Input(
          0,
          "update_signal",
          "This signal indicates if weight tensors should be updated.",
          "T_BOOL",
          OpSchema::Optional)
      .Input(
          1,
          "loss_scale",
          "Loss scale for mixed precision training.",
          "T2",
          OpSchema::Optional)
      .Input(
          2,
          "gradient_norm",
          "Norm of global gradient.",
          "T_GRAD_NORM",
          OpSchema::Optional)
      .Input(
          3,
          "R",
          "The initial learning rate.",
          "T1",
          OpSchema::Optional)
      .Input(
          4,
          "step",
          "One-based index of the current training iteration.",
          "TInt64",
          OpSchema::Optional);

  AddRepeatedInputs(
      op_schema,
      5,
      1024,
      {"weights",
       "gradients",
       "moment1",
       "moment2",
       "fp16_weights"},
      {"weights to optimize.",
       "gradients computed in this iteration.",
       "exponentially averaged historical gradients.",
       "exponentially averaged historical squared gradients.",
       "FP16 weights to optimize."},
      {"T2",
       "T3",
       "T4",
       "T4",
       "T_FP16"},
      OpSchema::Optional);

  op_schema
      .Output(
          0,
          "new_step",
          "One-based index of the next training iteration.",
          "TInt64",
          OpSchema::Optional);

  AddRepeatedOutputs(
      op_schema,
      1,
      1024,
      {"new_weights",
       "new_gradients",
       "new_moment_1",
       "new_moment_2",
       "new_fp16_weights"},
      {"New weights",
       "New gradients",
       "New averaged gradients",
       "New averaged squared gradients",
       "New FP16 weights"},
      {"T2",
       "T3",
       "T4",
       "T4",
       "T_FP16"},
      OpSchema::Optional);

  return op_schema;
}

void RegisterGradientSchemas() {
  ONNX_CONTRIB_OPERATOR_SCHEMA(ReluGrad)
      .SinceVersion(9)
      .Input(0, "dY", "Gradient of output Y", "T")
      .Input(1, "X", "Input tensor", "T")
      .Output(0, "dX", "Gradient of input X", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(SoftmaxGrad)
      .SinceVersion(9)
      .Input(0, "dY", "Gradient of output Y", "T")
      .Input(1, "X", "Input tensor", "T")
      .Output(0, "dX", "Gradient of input X", "T")
      .Attr(
          "axis",
          "Describes the axis of the inputs when coerced "
          "to 2D; defaults to one because the 0th axis most likely describes "
          "the batch_size",
          AttributeProto::INT,
          static_cast<int64_t>(1))
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(AveragePoolGrad)
      .SinceVersion(9)
      .Input(0, "dY", "Gradient of output Y", "T")
      .Output(0, "dX", "Gradient of input X", "T")
      .Attr(
          "kernel_shape",
          "The size of the kernel along each axis.",
          AttributeProto::INTS)
      .Attr(
          "strides", "Stride along each axis.", AttributeProto::INTS, OPTIONAL)
      .Attr(
          "auto_pad",
          "auto_pad doc",
          AttributeProto::STRING,
          std::string("NOTSET"))
      .Attr("pads", "pads_doc", AttributeProto::INTS, OPTIONAL)
      .Attr(
          "count_include_pad",
          "",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .AllowUncheckedAttributes()
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(MaxPoolGrad)
      .SinceVersion(9)
      .Input(0, "dY", "Gradient of output, Y", "T")
      .Input(1, "Indices", "Indices tensor from max pooling across the input tensor.", "I")
      .Output(0, "dX", "Gradient of input, X", "T")
      .AllowUncheckedAttributes()
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "I",
          {"tensor(int64)"},
          "Constrain index tensor to int64");

  ONNX_CONTRIB_OPERATOR_SCHEMA(ConvGrad)
      .SinceVersion(9)
      .Input(0, "dY", "Gradient of output Y", "T")
      .Input(1, "X", "Input tensor", "T")
      .Input(2, "W", "Weight tensor", "T")
      .Output(0, "dX", "Gradient of input X", "T")
      .Output(1, "dW", "Gradient of W", "T")
      .Output(2, "dB", "Gradient of B", "T")
      .AllowUncheckedAttributes()
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(GatherGrad)
      .SinceVersion(9)
      .Input(0, "shape", "Shape of the Gather input X.", "I")
      .Input(1, "indices", "Tensor of int32/int64 indices, of any rank q.", "Tind")
      .Input(2, "dY", "Gradient of output", "T")
      .Output(0, "dX", "Gradient of input", "T")
      .Attr(
          "axis",
          "Which axis to gather on. Negative value means "
          "counting dimensions from the back. Accepted range in [-r, r-1]",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .TypeConstraint(
          "I",
          {"tensor(int64)"},
          "Constrain input shape to integer tensors.")
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types(),
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "Tind",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain indices to integer types");

  ONNX_CONTRIB_OPERATOR_SCHEMA(DivGrad)
      .SinceVersion(9)
      .Input(0, "dY", "Gradient of output", "T")
      .Input(1, "A", "dividend", "T")
      .Input(2, "B", "divisor", "T")
      .Output(0, "dA", "Gradient of dividend", "T", OpSchema::Optional)
      .Output(1, "dB", "Gradient of divisor", "T", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to numeric tensors.");

  //TODO: Move this to the right location. Its only here for quick experimentation.
  //TODO: Use the mutli weight / grad version.
  ONNX_CONTRIB_OPERATOR_SCHEMA(SGDOptimizer)
      .SinceVersion(9)
      .Input(0, "ETA", "Learning Rate", "L")
      .Input(1, "W", "Original weight(s)", "T")
      .Input(2, "G", "Gradient of Weight(s)", "T")
      .Output(0, "NW", "Updated weight(s)", "T", OpSchema::Optional)
      .Output(1, "NG", "Updated gradients(s)", "T", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "L",
          {"float"},
          "Constrain learning rate to float");

  // TODO: This is copied from onnx schemas. When the change is in and we update this can be removed.
  // For Brevity documentation was not copied
  ONNX_CONTRIB_OPERATOR_SCHEMA(AdamOptimizer)
      .SinceVersion(9)
      .Input(0, "R", "The initial learning rate.", "T1")
      .Input(1, "T", "The update count of \"X\". It should be a scalar.", "T2")
      .Input(
          2,
          "weights",
          "weights to optimize.",
          "T3")
      .Input(
          3,
          "gradients",
          "gradients computed in this iteration.",
          "T_GRAD")
      .Input(
          4,
          "moment_1",
          "exponentially averaged historical gradients.",
          "T4")
      .Input(
          5,
          "moment_2",
          "exponentially averaged historical squared gradients.",
          "T4")
      .Input(
          6,
          "fp16_weights",
          "FP16 weights to optimize.",
          "T_FP16",
          OpSchema::Optional)
      .Input(
          7,
          "loss_scale",
          "loss scale for mixed precision training",
          "T3",
          OpSchema::Optional)
      .Input(
          8,
          "global_gradient_norm",
          "Global gradient norm.",
          "T_GRAD_NORM",
          OpSchema::Optional)
      .Input(
          9,
          "update_signal",
          "This signal indicates if weight tensors should be updated.",
          "T_BOOL",
          OpSchema::Optional)
      .Output(
          0,
          "new_T",
          "New update count.",
          "T2")
      .Output(
          1,
          "new_moment_1",
          "New averaged gradients.",
          "T4")
      .Output(
          2,
          "new_moment_2",
          "New averaged squared gradients.",
          "T4")
      .Output(
          3,
          "new_weights",
          "New weights.",
          "T3",
          OpSchema::Optional)
      .Output(
          4,
          "new_gradients",
          "New gradients.",
          "T_GRAD",
          OpSchema::Optional)
      .Output(
          5,
          "new_fp16_weights",
          "New FP16 weights",
          "T_FP16",
          OpSchema::Optional)
      .Attr(
          "alpha",
          "Coefficient of previous gradient in running average.",
          AttributeProto::FLOAT,
          0.9f)
      .Attr(
          "beta",
          "Coefficient of previous squared gradient in running average."
          "The effective learning rate is computed by r = R / (1 + T * decay_factor). "
          "Default to 0 so that increasing update counts doesn't reduce the learning rate.",
          AttributeProto::FLOAT,
          0.999f)
      .Attr(
          "lambda",
          "Regularization coefficient of 0.5 * lambda * ||X||_2^2. Default to 0, "
          "which means no regularization.",
          AttributeProto::FLOAT,
          0.0f)
      .Attr(
          "epsilon",
          "Small scalar to avoid dividing by zero.",
          AttributeProto::FLOAT,
          1e-8f)
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain learning rate to float")
      .TypeConstraint(
          "T2",
          {"int64"},
          "Constrain step count to 64-bit integer")
      .TypeConstraint(
          "T3",
          {"tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T4",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T_GRAD",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T_FP16",
          {"tensor(float16)"},
          "Constrain input types to float16 tensors.")
      .TypeConstraint(
          "T_GRAD_NORM",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "T_BOOL",
          {"tensor(bool)"},
          "Constrain types to boolean tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA_ELSEWHERE(LambOptimizer, RegisterLambOpSchema);

  ONNX_CONTRIB_OPERATOR_SCHEMA(InPlaceAccumulator)
      .SinceVersion(9)
      .SetDoc("in-place accumulator for tensors")
      .Input(0, "old_sum", "historical result of accumulator", "T")
      .Input(1, "value", "the value that will be added to the accumulator", "T_GRAD")
      .Input(2, "update_signal", "This signal indicates if tensor should be updated", "T_BOOL", OpSchema::Optional)
      .Output(0, "new_sum", "updated result of accumulator", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T_GRAD",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T_BOOL",
          {"tensor(bool)"},
          "Constrain types to boolean tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(ZeroGradient)
      .SinceVersion(9)
      .SetDoc("reset the accumulator for gradient")
      .Input(0, "old_gradient", "historical result of accumulated gradient", "T1")
      .Input(1, "reset_signal", "if this input is available, it is ready to reset the accumulator", "T2")
      .Output(0, "zero_gradient", "reset the gradient", "T1")
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output gradient types to float tensors.")
      .TypeConstraint(
          "T2",
          OpSchema::all_tensor_types(),
          "reset_signal can be of any tensor type.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GatherNDGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(1)
      .Attr(
          "axis",
          "The number of batch dims. The gather of indexing starts from dimension of data[axis+1:]",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Input(0, "shape", "The shape of source data input of GatherND.", "T1")
      .Input(1, "indices", "Tensor of rank q >= 1.", "Tind")
      .Input(2, "update", "The gradient of the output.", "T")
      .Output(0, "output", "Tensor graident of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to any tensor type.")
      .TypeConstraint(
          "Tind",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain indice type to int32 or int64")
      .TypeConstraint(
          "T1",
          {"tensor(int64)"},
          "Constrain shape type to int64");

  // TODO: push this to ONNX
  static const char* reduction_doc =
      "Type of reduction to apply to loss: none, sum, mean(default). "
      "'none': the output is the loss for each sample in the batch."
      "'sum': the output will be summed. "
      "'mean': the sum of the output will be divided by the batch_size.";

  ONNX_CONTRIB_OPERATOR_SCHEMA(SoftmaxCrossEntropy)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .Input(0, "logits", "Unscaled log probabilities, N-D input of shape (-1, num_classes).", "T")
      .Input(1, "label", "The onehot label is N-D input with the same shape as logits.", "T")
      .Output(0, "Y", "loss.", "T")
      .Output(1, "log_prob", "logsoftmax(logits)", "T", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .SetDoc(R"DOC(SoftmaxCrossEntropy)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(SoftmaxCrossEntropyGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .Input(0, "dY", "gradient of Y", "T")
      .Input(1, "log_prob", "logsoftmax(logits), N-D input of shape (-1, num_classes).", "T")
      .Input(2, "label", "The onehot label is N-D input with the same shape as logits.", "T")
      .Output(0, "d_logits", "gradient of logits", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .SetDoc(R"DOC(SoftmaxCrossEntropyGrad)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(HorovodAllReduce)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Attr("reduce_op", "Reduce operation supported by Horovod. Valid values are: AVERAGE(0), SUM(1) or ADASUM(2)", AttributeProto::INT, int64_t(1))
      .Input(0, "input", "tensor to be reduced", "T")
      .Output(0, "output", "reduced tensor", "T")
      .Output(1, "ready", "true when reduced tensor is ready", "B")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeConstraint("B", {"tensor(bool)"}, "Constrain to bool tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::BOOL);
        updateOutputShape(ctx, 1, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(HorovodBarrier)
      .SetDomain(kOnnxDomain)
      .SetDoc("Waits for one or more async Horovod operators to complete")
      .SinceVersion(9)
      .Input(0, "input", "input tensor", "T")
      .Input(1, "input_ready", "one or more bool tensors to wait on", "B", OpSchema::Variadic)
      .Output(0, "output", "output tensor", "T")
      .Output(1, "output_ready", "output tensor is ready", "B")
      .TypeConstraint("B", {"tensor(bool)"}, "Only bool")
      .TypeConstraint("T", OpSchema::all_tensor_types(), "All Tensor types")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::BOOL);
        updateOutputShape(ctx, 1, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(NcclAllReduce)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Attr("group_type", "0 - data parallel group, 1 - horizontal parallel group",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0, "input", "tensors to be reduced", "T", OpSchema::Variadic)
      .Output(0, "output", "reduced tensors", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(NcclAllGather)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Attr("group_type", "0 - data parallel group, 1 - horizontal parallel group",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0, "input", "tensors to be sent", "T", OpSchema::Variadic)
      .Output(0, "output", "gathered tensors", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(NcclReduceScatter)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Attr("group_type", "0 - data parallel group, 1 - horizontal parallel group",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0, "input", "tensors to be reduced and scattered", "T", OpSchema::Variadic)
      .Output(0, "output", "reduced tensors", "T", OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(SparseSoftmaxCrossEntropy)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .Input(0, "logits", "Unscaled log probabilities, (N+1)-D input of shape (-1, num_classes).", "T")
      .Input(1, "label",
             "label is N-D input whose shape should match that of logits. "
             "It is a tensor of nonnegative integers, "
             "where each element is the nonnegative integer label for the element of the batch.",
             "Tind")
      .Input(2, "weight", "weight for each sample. The shape is the same as label's", "T", OpSchema::Optional)
      .Output(0, "Y", "loss.", "T")
      .Output(1, "log_prob", "logsoftmax(logits)", "T", OpSchema::Optional)
      .TypeConstraint("T",
                      {"tensor(float16)", "tensor(float)", "tensor(double)"},
                      "Constrain to float, float16 and double tensors.")
      .TypeConstraint("Tind",
                      {"tensor(int32)", "tensor(int64)"},
                      "Constrain indices to integer types")
      .SetDoc(R"DOC(SparseSoftmaxCrossEntropy)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(SparseSoftmaxCrossEntropyGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Attr("reduction",
            reduction_doc,
            AttributeProto::STRING,
            std::string("mean"))
      .Input(0, "dY", "gradient of Y", "T")
      .Input(1, "log_prob", "logsoftmax(logits), (N+1)-D input of shape (batch_size).", "T")
      .Input(2, "label",
             "label is N-D input whose shape should match that of logits. "
             "It is a tensor of nonnegative integers, "
             "where each element is the nonnegative integer label for the element of the batch.",
             "Tind")
      .Input(3, "weight", "weight for each sample. The shape is the same as label's", "T", OpSchema::Optional)
      .Output(0, "d_logits", "gradient of logits", "T")
      .TypeConstraint("T",
                      {"tensor(float16)", "tensor(float)", "tensor(double)"},
                      "Constrain to float, float16 and double tensors.")
      .TypeConstraint("Tind",
                      {"tensor(int32)", "tensor(int64)"},
                      "Constrain indices to integer types")
      .SetDoc(R"DOC(SparseSoftmaxCrossEntropyGrad)DOC");

  ONNX_CONTRIB_OPERATOR_SCHEMA(TrainableDropout)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("TrainableDropout")
      .Attr("seed", "(Optional) Seed to the random generator, if not specified we will auto generate one.", AttributeProto::INT, OPTIONAL)
      .AllowUncheckedAttributes()
      .Input(0, "data", "The input data as Tensor.", "T")
      .Input(1, "ratio",
             "The ratio of random dropout, with value in [0, 1). If this input was not set, "
             "or if it was set to 0, the output would be a simple copy of the input. "
             "If it's non-zero, output will be a random dropout of input, which is typically "
             "the case during training.",
             "T1",
             OpSchema::Optional)
      .Output(0, "output", "The output.", "T")
      .Output(1, "mask", "The output mask.", "T2", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input 'ratio' types to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(bool)"},
          "Constrain output 'mask' types to boolean tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        if (ctx.getNumOutputs() == 2) {
          updateOutputElemType(ctx, 1, ONNX_NAMESPACE::TensorProto::BOOL);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 1);
          }
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(TrainableDropoutGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("TrainableDropoutGrad")
      .AllowUncheckedAttributes()
      .Input(0, "dy", "The gradient tensor from output.", "T")
      .Input(1, "mask",
             "The mask tensor of the dropout. ", "T2")
      .Input(2, "ratio",
             "The ratio of random dropout, with value in [0, 1). If this input was not set, "
             "or if it was set to 0, the output would be a simple copy of the input. "
             "If it's non-zero, output will be a random dropout of input, which is typically "
             "the case during training.",
             "T1",
             OpSchema::Optional)
      .Output(0, "dx", "Gradient of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(float)"},
          "Constrain input 'ratio' types to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(bool)"},
          "Constrain 'mask' types to boolean tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(DropoutGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("DropoutGrad")
      .AllowUncheckedAttributes()
      .Input(0, "dy", "The gradient tensor from output.", "T")
      .Input(1, "mask",
             "The mask tensor of the dropout. ", "T2")
      .Input(2, "ratio",
             "The ratio of random dropout, with value in [0, 1). If this input was not set, "
             "or if it was set to 0, the output would be a simple copy of the input. "
             "If it's non-zero, output will be a random dropout of input, which is typically "
             "the case during training.",
             "T1",
             OpSchema::Optional)
      .Output(0, "dx", "Gradient of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input 'ratio' types to float tensors.")
      .TypeConstraint(
          "T2",
          {"tensor(bool)"},
          "Constrain 'mask' types to boolean tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistBinarizeEncoder)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X", "uncompressed input", "T")
      .Output(0, "Y", "uncompressed output", "T")
      .Output(1, "Y1", "compressed output", "T1")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(bool)"},
          "Binarize tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(GistBinarizeDecoder)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "X1", "dummy input for late decoding", "T")
      .Input(1, "X", "compresssed input", "T1")
      .Output(0, "Y", "uncompressed output", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to all numeric tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(bool)"},
          "Binarize tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(SinGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Sin")
      .AllowUncheckedAttributes()
      .Input(0, "dY", "Sin output's grad", "T")
      .Input(1, "X", "Input tensor", "T")
      .Output(0, "dX", "Sin input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           {{"X_1"}, "Cos", {"X"}},
           {{"dX"}, "Mul", {"X_1", "dY"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(TanhGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Tanh")
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "dY", "Tanh output's grad", "T")
      .Output(0, "dX", "Tanh input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("One", 1.0f),
           {{"Squared_output"}, "Mul", {"X", "X"}},
           {{"Tanh_Grad"}, "Sub", {"One", "Squared_output"}},
           {{"dX"}, "Mul", {"dY", "Tanh_Grad"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(SqrtGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Sqrt")
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "dY", "Sqrt output's grad", "T")
      .Output(0, "dX", "Sqrt input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("One_half", 0.5f),
           {{"Sqrt_Grad"}, "Div", {"One_half", "X"}},
           {{"dX"}, "Mul", {"dY", "Sqrt_Grad"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(ErfGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Erf")
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "dY", "Erf output's grad", "T")
      .Output(0, "dX", "Erf input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("Two_sqrt_pi", static_cast<float>(M_2_SQRTPI)),
           {{"Square_x"}, "Mul", {"X", "X"}},
           {{"Neg_Square_x"}, "Neg", {"Square_x"}},
           {{"Exp_Neg_Square_x"}, "Exp", {"Neg_Square_x"}},
           {{"Erf_Grad"}, "Mul", {"Two_sqrt_pi", "Exp_Neg_Square_x"}},
           {{"dX"}, "Mul", {"dY", "Erf_Grad"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReshapeGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Reshape")
      .AllowUncheckedAttributes()
      .Input(0, "X", "Input tensor", "T")
      .Input(1, "dY", "Reshape output's grad", "T")
      .Output(0, "dX", "REshape input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           {{"x_shape"}, "Shape", {"X"}},
           {{"dX"}, "Reshape", {"dY", "x_shape"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(PowGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Gradient function for Pow")
      .AllowUncheckedAttributes()
      .Input(0, "dY", "Reshape output's grad", "T")
      .Input(1, "X", "Input tensor", "T")
      .Input(2, "Exponent", "Input tensor", "T")
      .Output(0, "dX", "Pow input's grad", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to all numeric tensors.")
      .FunctionBody(ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
           ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("One", 1.0f),
           {{"p_minus_one"}, "Sub", {"Exponent", "One"}},
           {{"X_Pow_p_minus_one"}, "Pow", {"X", "p_minus_one"}},
           {{"a_X_Pow_p_minus_one"}, "Mul", {"X_Pow_p_minus_one", "Exponent"}},
           {{"dX"}, "Mul", {"a_X_Pow_p_minus_one", "dY"}}}));

  ONNX_CONTRIB_OPERATOR_SCHEMA(SummaryScalar)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SummaryScalar")
      .Attr("tags", "The tags corresponding to each input scalar.", AttributeProto::STRINGS)
      .Input(0, "input", "The scalar tensor to summarize as simple values.", "T")
      .Output(0, "summary", "The serialized Tensorboard Summary.", "S")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bool)"},
          "Constrain input type to float and bool tensors.")
      .TypeConstraint(
          "S",
          {"tensor(string)"},
          "Constrain output type to string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::STRING);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SummaryHistogram)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SummaryHistogram")
      .Attr("tag", "The tag corresponding to the histogram data.", AttributeProto::STRING)
      .Input(0, "input", "The scalar tensor to produce a histogram over.", "T")
      .Output(0, "summary", "The serialized Tensorboard Summary.", "S")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input type to float tensors.")
      .TypeConstraint(
          "S",
          {"tensor(string)"},
          "Constrain output type to string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::STRING);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SummaryMerge)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SummaryMerge")
      .Input(0, "input", "One or more serialized Tensorboard Summary tensors to merge into a single Summary.", "S", OpSchema::Variadic)
      .Output(0, "summary", "The serialized Tensorboard Summary.", "S")
      .TypeConstraint(
          "S",
          {"tensor(string)"},
          "Constrain input and output types to string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::STRING);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SummaryText)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("SummaryText")
      .Attr("tag", "The tag corresponding to the text data.", AttributeProto::STRING)
      .Input(0, "input", "The string tensor to render in the Tensorboard Text dashboard.", "S")
      .Output(0, "summary", "The serialized Tensorboard Summary.", "S")
      .TypeConstraint(
          "S",
          {"tensor(string)"},
          "Constrain input and output types to string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::STRING);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(GeluGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("GeluGrad")
      .AllowUncheckedAttributes()
      .Input(0, "dY", "The gradient tensor from output.", "T")
      .Input(1, "X", "The input tensor. ", "T")
      .Output(0, "dX", "Gradient of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  ONNX_CONTRIB_OPERATOR_SCHEMA(LayerNormalizationGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("LayerNormalizationGrad")
      .Attr("axis",
            "The first normalization dimension: normalization will be performed along dimensions axis : rank(inputs).",
            AttributeProto::INT, static_cast<int64_t>(-1))
      .AllowUncheckedAttributes()
      .Input(0, "Y_grad", "The gradient tensor from output.", "T")
      .Input(1, "X", "Input data tensor from the forward path", "T")
      .Input(2, "scale", "Scale tensor.", "T")
      .Input(3, "mean", "mean of X.", "U")
      .Input(4, "inv_std_var", "inverse std variance of X.", "U")
      .Output(0, "X_grad", "Gradient of the input.", "T")
      .Output(1, "scale_grad", "Gradient of the scale.", "T")
      .Output(2, "bias_grad", "Gradient of the bias.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types (except mean and inv_std_var) to float tensors.")
      .TypeConstraint(
          "U",
          {"tensor(float)"},
          "Constrain mean and inv_std_var to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(BatchNormalizationGrad)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetDoc("BatchNormalization")
      .Attr("epsilon",
            "epsilon value",
            AttributeProto::FLOAT)
      .Input(0, "dY", "Gradient output from previous node", "T")
      .Input(1, "X", "Input", "T")
      .Input(2, "scale", "Scale tensor", "T")
      .Input(3, "mean", "Mean of X", "T")
      .Input(4, "variance", "Variance of X", "T")
      .Output(0, "X_grad", "Gradient of the input", "T")
      .Output(1, "scale_grad", "Gradient of the scale", "T")
      .Output(2, "bias_grad", "Gradient of the bias", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(Group)
      .SetDomain(kOnnxDomain)
      .SetDoc("if all the inputs are available, the output will be true")
      .SinceVersion(9)
      .Input(0, "input_tensors", "list of dependency tensors", "T", OpSchema::Variadic, false)
      .Output(0, "done", "all the dependency tensors are ready", "B")
      .TypeConstraint("T", OpSchema::all_tensor_types(), "All Tensor types")
      .TypeConstraint("B", {"tensor(bool)"}, "Only bool")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::BOOL);
        updateOutputShape(ctx, 0, {});
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(IsFinite)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("IsFinite")
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(bool)"},
          "Constrain the output to a boolean tensor.")
      .Input(
          0,
          "X",
          "The input tensor.",
          "T")
      .Output(
          0,
          "Y",
          "The output tensor. Its shape is the same as the input.",
          "T1");

  ONNX_CONTRIB_OPERATOR_SCHEMA(IsAllFinite)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("IsAllFinite")
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .TypeConstraint(
          "V",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T",
          {"tensor(bool)"},
          "Constrain the output to a boolean tensor.")
      .Input(0, "input", "Input tensors to check.", "V",
             OpSchema::Variadic)
      .Output(
          0,
          "output",
          "The output scalar. Its value is true if all input "
          "tensors are finite. Otherwise, the output value would "
          "be false.",
          "T");

  static const char* All_doc = R"DOC(
Return true if all elements are true and false otherwise.
)DOC";

  ONNX_CONTRIB_OPERATOR_SCHEMA(All)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .Input(0, "X", "input", "T")
      .Output(0, "Y", "output.", "T")
      .TypeConstraint(
          "T",
          {"tensor(bool)"},
          "Constrain input and output types to boolean tensors.")
      .SetDoc(All_doc)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MixedPrecisionScale)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("MixedPrecisionScale")
      .Input(0, "S", "scale", "ScaleT")
      .Input(1, "X", "inputs", "SrcT", OpSchema::Variadic)
      .Output(0, "Y", "output", "DstT", OpSchema::Variadic)
      .Attr("to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
      .Attr("fuse_outputs",
            "If true, fuse all outputs into one continous buffer.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .TypeConstraint(
          "SrcT",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "ScaleT",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain scale types to float tensors.")
      .TypeConstraint(
          "DstT",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain output types to float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        bool fuse_outputs = static_cast<bool>(getAttribute(ctx, "fuse_outputs", int64_t(0)));
        if (fuse_outputs) {
          int64_t total_num_elements = 0;
          for (size_t i = 1; i < ctx.getNumInputs(); ++i) {
            if (!hasInputShape(ctx, i))
              return;
            auto& input_shape = getInputShape(ctx, i);
            int rank = static_cast<int>(input_shape.dim_size());
            int64_t num_elements = multiplyDims(input_shape, 0, rank).dim_value();
            total_num_elements += num_elements;
          }

          ONNX_NAMESPACE::TensorShapeProto output_shape;
          output_shape.add_dim()->set_dim_value(total_num_elements);
          updateOutputShape(ctx, 0, output_shape);
          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
        } else {
          for (size_t i = 1; i < ctx.getNumInputs(); ++i) {
            propagateElemTypeFromAttributeToOutput(ctx, "to", i - 1);
            propagateShapeFromInputToOutput(ctx, i, i - 1);
          }
        }
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(View)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("View. The output tensors are views of the input, according to the shapes provided.")
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .Input(0, "input", "Input tensor.", "T")
      .Input(1, "shapes", "Shapes of each view output. The shapes must adds up to the input buffer size.",
             "tensor(int64)",
             OpSchema::Variadic)
      .Output(0, "outputs", "Output tensors viewed according the shapes input. It has a one to one mapping to the shapes input",
              "T",
              OpSchema::Variadic)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(ReduceAllL2)
      .SetDomain(kOnnxDomain)
      .SinceVersion(9)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Multi-tensor version of ReduceL2.")
      .Input(0, "X", "inputs", "TIn", OpSchema::Variadic)
      .Output(0, "Y", "output", "TOut")
      .TypeConstraint(
          "TIn",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "TOut",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain scale types to float tensors.");

  ONNX_CONTRIB_OPERATOR_SCHEMA(Send)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Send data tensor to the specified destination.")
      .Input(0, "InputSignal", "Input control signal.", "TBool")
      .Input(1, "Data", "Tensor to send.", "T")
      .Output(0, "OutputSignal", "Output control signal.", "TBool")
      .Attr("src",
            "Abstractive memory ID of Data's source.",
            AttributeProto::INT)
      .Attr("dst",
            "Abstractive memory ID of Data's destination.",
            AttributeProto::INT)
      .Attr("tag", "The tag of the message carrying Data.",
            AttributeProto::INT)
      .Attr("element_type", "Element type of the sent tensor.",
            AttributeProto::INT)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "TBool",
          {"tensor(bool)"},
          "Constrain types to boolean tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        if (ctx.getNumInputs() != 2)
          fail_shape_inference("Send must have two inputs.");
        if (ctx.getNumOutputs() != 1)
          fail_shape_inference("Send must have one output.");

        auto output_element_type = ctx.getOutputType(0)->mutable_tensor_type();
        output_element_type->set_elem_type(TensorProto::BOOL);
        ONNX_NAMESPACE::TensorShapeProto output_shape;
        updateOutputShape(ctx, 0, {});
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::BOOL);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(Recv)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Receive a tensor from the the specified source.")
      .Input(0, "InputSignal", "Input control signal.", "TBool")
      .Output(0, "OutputSignal", "Output control signal.", "TBool")
      .Output(1, "Data", "The Received tensor.", "T")
      .Attr("src",
            "Abstractive memory ID of Data's source.",
            AttributeProto::INT)
      .Attr("dst",
            "Abstractive memory ID of Data's destination.",
            AttributeProto::INT)
      .Attr("tag", "The tag of the message carrying Data.",
            AttributeProto::INT)
      .Attr("element_type", "Element type of the received tensor.",
            AttributeProto::INT)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input types to float tensors.")
      .TypeConstraint(
          "TBool",
          {"tensor(bool)"},
          "Constrain types to boolean tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        if (ctx.getNumInputs() != 1)
          fail_shape_inference("Recv must have one inputs.");
        if (ctx.getNumOutputs() != 2)
          fail_shape_inference("Recv must have two output.");

        updateOutputShape(ctx, 0, {});
        updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::BOOL);
        propagateElemTypeFromAttributeToOutput(ctx, "element_type", 1);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MegatronF)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "input", "The input data as Tensor.", "T")
      .Output(0, "output", "The output.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(MegatronG)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Attr("group_type", "0 - data parallel group, 1 - horizontal parallel group",
            AttributeProto::INT,
            static_cast<int64_t>(0))
      .Input(0, "input", "The input data as Tensor.", "T")
      .Output(0, "output", "The output.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain to float, float16 and double tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
      });

  ONNX_CONTRIB_OPERATOR_SCHEMA(SliceGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Input(0, "dY", "Gradient of output", "T")
      .Input(1, "shape", "Shape of the Slice input X.", "I")
      .Input(2, "starts", "Tensor of starting indices of corresponding axis in axes", "Tind")
      .Input(3, "ends", "Tensor of starting indices of corresponding axis in 'axes'", "Tind")
      .Input(4, "axes", "Tensor of axes that `starts` and `ends` apply to", "Tind", OpSchema::Optional)
      .Input(5, "steps", "Tensor of slice step of corresponding axis in `axes`", "Tind", OpSchema::Optional)
      .Output(0, "dX", "Gradient of input", "T")
      .TypeConstraint(
          "I",
          {"tensor(int64)"},
          "Constrain input shape to integer tensors.")
      .TypeConstraint(
          "T",
          OpSchema::all_tensor_types(),
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "Tind",
          {"tensor(int32)", "tensor(int64)"},
          "Constrain indices to integer types");

  ONNX_CONTRIB_OPERATOR_SCHEMA(FastGeluGrad)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("FastGeluGrad")
      .AllowUncheckedAttributes()
      .Input(0, "dY", "The gradient tensor from output.", "T")
      .Input(1, "X", "The input tensor. ", "T")
      .Output(0, "dX", "Gradient of the input.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);
}
}  // namespace training
}  // namespace onnxruntime
