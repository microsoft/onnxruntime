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
}
}  // namespace contrib
}  // namespace onnxruntime
