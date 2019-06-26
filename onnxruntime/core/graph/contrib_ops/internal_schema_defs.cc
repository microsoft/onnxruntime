#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/internal_schema_defs.h"
#include "core/graph/op.h"

namespace onnxruntime {
namespace contrib {
void RegisterInternalSchemas() {
#ifdef USE_BRAINSLICE
  using namespace ONNX_NAMESPACE;
  ONNX_CONTRIB_OPERATOR_SCHEMA(BrainSlice)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("Generic BrainSlice FPGA node")
      .Attr("firmware_data", "FPGA firmware bytes from data.bin file", AttributeProto::STRING)
      .Attr("firmware_instructions", "FPGA firmware bytes from instructions.bin file", AttributeProto::STRING)
      .Attr("firmware_schema", "FPGA firmware bytes from schema.bin file", AttributeProto::STRING)
      .Attr("input_addresses", "FPGA input addresses for matrix/vector initializers", AttributeProto::INTS)
      .Attr("input_memtypes", "FPGA input memory types for initializers (matrix/vector, dram/on-chip registers)", AttributeProto::INTS)
      .Attr("output_interleaved", "FPGA output tensor is written interleaved due to the last layer being a BrainSlice convolution", AttributeProto::INT, false)
      .AllowUncheckedAttributes()
      .Input(0, "input", "Tensor input", "T", OpSchema::Variadic, false)
      .Output(0, "output", "Tensor output", "T", OpSchema::Variadic, false)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input/output types to float16 and float32 tensors.");
#endif
#ifndef ENABLE_TRAINING
  using namespace ONNX_NAMESPACE;
  ONNX_CONTRIB_OPERATOR_SCHEMA(TrainableDropout)
      .SetDomain(kOnnxDomain)
      .SinceVersion(10)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("TrainableDropout")
      .Attr("seed", "(Optional) Seed to the random generator, if not specified we will auto generate one.", AttributeProto::FLOAT, OPTIONAL)
      .AllowUncheckedAttributes()
      .Input(0, "data", "The input data as Tensor.", "T")
      .Input(1, "ratio",
             "The ratio of random dropout, with value in [0, 1]. If this input was not set, "
             "or if it was set to 0, the output would be a simple copy of the input. "
             "If it's non-zero, output will be a random dropout of input, which is typically "
             "the case during training.",
             "T",
             OpSchema::Optional)
      .Output(0, "output", "The output.", "T")
      .Output(1, "mask", "The output mask.", "T1", OpSchema::Optional)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)", "tensor(double)"},
          "Constrain input and output types to float tensors.")
      .TypeConstraint(
          "T1",
          {"tensor(bool)"},
          "Constrain output 'mask' types to boolean tensors.")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateShapeAndTypeFromFirstInput(ctx);
        if (ctx.getNumOutputs() == 2) {
          updateOutputElemType(ctx, 1, TensorProto::BOOL);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 1);
          }
        }
      });
#endif
}
}  // namespace contrib
}  // namespace onnxruntime
