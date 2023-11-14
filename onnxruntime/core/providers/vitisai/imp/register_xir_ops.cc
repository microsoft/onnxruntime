// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/framework/customregistry.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/custom_ops.h"
#include "core/session/inference_session.h"

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

#include "vaip/vai_assert.h"
using namespace onnxruntime;
namespace onnxruntime {
ONNX_NAMESPACE::OpSchema CreateSchema(const std::string& domain, const OrtCustomOp* op);
}
namespace vaip {

static void xir_shape_infer(ONNX_NAMESPACE::InferenceContext& ctx) {
  auto* shape = ctx.getAttribute("shape");
  auto* data_type = ctx.getAttribute("data_type");
  updateOutputElemType(ctx, 0, int(data_type->i()));
  if (shape != nullptr) {
    for (auto i = 0; i < shape->ints_size(); ++i) {
      ONNX_NAMESPACE::appendDim(ONNX_NAMESPACE::getOutputShape(ctx, 0), shape->ints(i));
    }
  } else {
    // set scalar type.
    auto* output_shape = ONNX_NAMESPACE::getOutputShape(ctx, 0);
    output_shape->clear_dim();
  }
  return;
}

static void xir_fixneuron_shape_inference(ONNX_NAMESPACE::InferenceContext& ctx) {
  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
  ONNX_NAMESPACE::propagateShapeFromInputToOutput(ctx, 0, 0);
}

static void xir_subgraph_shape_inference(ONNX_NAMESPACE::InferenceContext& ctx) {
  auto num_inputs = ctx.getNumInputs();

  // Run inferencing on the subgraph
  ONNX_NAMESPACE::GraphInferencer* graphInferencer = ctx.getGraphAttributeInferencer("body");
  if (!graphInferencer) {
    fail_type_inference("body is missing.");
  }

  std::vector<const ONNX_NAMESPACE::TensorProto*> input_data;
  std::vector<const ONNX_NAMESPACE::TypeProto*> subgraph_input_types;
  for (size_t i = 0; i < num_inputs; ++i) {
    input_data.push_back(ctx.getInputData(i));
    subgraph_input_types.push_back(ctx.getInputType(i));
  }
  std::vector<const ONNX_NAMESPACE::TypeProto*> output_types;
  output_types = graphInferencer->doInferencing(subgraph_input_types, input_data);

  auto num_outputs = ctx.getNumOutputs();
  auto num_of_the_subgraph_outputs = output_types.size();
  if (num_outputs != num_of_the_subgraph_outputs) {
    fail_type_inference("super layer has ", num_outputs, " but subgraphs produce ", num_of_the_subgraph_outputs);
  }
  for (size_t i = 0, end = output_types.size(); i < end; ++i) {
    auto subgraph_output = output_types[i];
    auto* super_layer_output = ctx.getOutputType(i);
    *super_layer_output = *subgraph_output;
  }
}

void register_xir_ops(const std::vector<OrtCustomOpDomain*>& domains) {
  auto& domain_to_version_range_instance = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
  const auto* schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
  const auto& domain_to_version_map = domain_to_version_range_instance.Map();

  for (auto domain : domains) {
    if (domain_to_version_map.find(domain->domain_) == domain_to_version_map.end()) {
      domain_to_version_range_instance.AddDomainToVersion(domain->domain_, 1, 1000);
    }
    for (auto op : domain->custom_ops_) {
      auto name = op->GetName(op);
      auto schema = CreateSchema(domain->domain_, op);
      if ((std::string)name == "super_layer") {
        schema.TypeAndShapeInferenceFunction(xir_subgraph_shape_inference);
      } else if ((std::string)name == "FixNeuron") {
        schema.TypeAndShapeInferenceFunction(xir_fixneuron_shape_inference);
      } else {
        schema.TypeAndShapeInferenceFunction(xir_shape_infer);
      }
      if (schema_registry->GetSchema(name, ORT_API_VERSION, domain->domain_) == nullptr) {
        ONNX_NAMESPACE::RegisterSchema(schema, ORT_API_VERSION);
      }
    }
  }
}

}  // namespace vaip
