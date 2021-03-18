#include "core/graph/contrib_ops/onnx_function_util.h"
#include "core/util/math.h"

namespace ONNX_NAMESPACE {

TensorProto ToTensor(double value, TensorProto_DataType elem_type) {
  TensorProto t;
  t.set_data_type(elem_type);
  switch (elem_type) {
    case TensorProto_DataType::TensorProto_DataType_FLOAT:
      t.add_float_data((float)value);
      break;
    case TensorProto_DataType::TensorProto_DataType_DOUBLE:
      t.add_double_data(value);
      break;
    case TensorProto_DataType::TensorProto_DataType_FLOAT16:
      t.add_int32_data(onnxruntime::math::floatToHalf((float)value));
      break;
    default:
      assert(false);
  }

  return t;
}

void BuildNodes(FunctionProto& functionProto, const std::vector<FunctionBodyHelper::NodeDef>& node_defs) {
  for (size_t i = 0; i < node_defs.size(); i++) {
    const FunctionBodyHelper::NodeDef& node = node_defs[i];
    auto* np = functionProto.add_node();

    np->set_op_type(node.op_type);
    for (const auto& inp : node.inputs) {
      np->add_input(inp);
    }
    for (const auto& o : node.outputs) {
      np->add_output(o);
    }
    for (const auto& attr : node.attributes) {
      *(np->add_attribute()) = attr.proto;
    }
  }
}

bool BuildFunctionProto(FunctionProto& functionProto, const OpSchema& schema,
                        const std::vector<FunctionBodyHelper::NodeDef>& node_defs,
                        const std::vector<OperatorSetIdProto>& relied_opsets) {
  BuildNodes(functionProto, node_defs);
  schema.BuildFunction(functionProto, relied_opsets);
  return true;
}

}  // namespace ONNX_NAMESPACE