

// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "./register_xir_ops.h"
#include "./vai_assert.h"

#include "core/common/logging/logging.h"
#include "core/common/status.h"

#include "core/framework/customregistry.h"

#include "core/session/onnxruntime_c_api.h"
#include "core/session/custom_ops.h"
#include "core/session/inference_session.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

using namespace onnxruntime;
namespace vaip {

static void xir_shape_infer(ONNX_NAMESPACE::InferenceContext& ctx) {
  auto* shape = ctx.getAttribute("shape");
  auto* data_type = ctx.getAttribute("data_type");
  if (data_type->s() == "float32") {
    updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::FLOAT);
  } else if (data_type->s() == "int8") {
    updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::INT8);
  } else if (data_type->s() == "uint8") {
    updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::UINT8);
  } else if (data_type->s() == "int32") {
    updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::INT32);
  } else if (data_type->s() == "int64") {
    updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::INT64);
  } else if (data_type->s() == "int1") {
    updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::BOOL);
  } else if (data_type->s() == "bfloat16") {
    updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::BFLOAT16);
  } else if (data_type->s() == "float16") {
    updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto::FLOAT16);
  } else {
    vai_assert(false, ", not supported data_type: " + data_type->s());
  }
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
  output_types =
      graphInferencer->doInferencing(subgraph_input_types, input_data);

  auto num_outputs = ctx.getNumOutputs();
  auto num_of_the_subgraph_outputs = output_types.size();
  if (num_outputs != num_of_the_subgraph_outputs) {
    fail_type_inference("super layer has ", num_outputs,
                        " but subgraphs produce ", num_of_the_subgraph_outputs);
  }
  for (size_t i = 0, end = output_types.size(); i < end; ++i) {
    auto subgraph_output = output_types[i];
    auto* super_layer_output = ctx.getOutputType(i);
    *super_layer_output = *subgraph_output;
  }
}

void register_xir_ops(const std::vector<OrtCustomOpDomain*>& domains) {
  std::shared_ptr<CustomRegistry> custom_registry;
  auto status = CreateCustomRegistry(gsl::span(domains), custom_registry);
  vai_assert(status.IsOK(), status.ErrorMessage());
  for (auto domain : domains) {
    for (auto op : domain->custom_ops_) {
      auto name = op->GetName(op);
      auto schema1 = custom_registry->GetOpschemaRegistry()->GetSchema(name, ORT_API_VERSION, domain->domain_);
      auto schema2 = ::ONNX_NAMESPACE::OpSchema();
      schema2.SetName(schema1->Name());
      schema2.SetDomain(schema1->domain());
      auto n = 0;
      for (auto input : schema1->inputs()) {
        schema2.Input(n, input.GetName(), input.GetDescription(), std::string("T") + std::to_string(n), input.GetOption(), false, input.GetMinArity(), input.GetDifferentiationCategory());
        schema2.TypeConstraint(std::string("T") + std::to_string(n), DataTypeImpl::ToString(DataTypeImpl::AllTensorTypes()), "all types");
        n = n + 1;
      }
      auto m = n;
      n = 0;
      for (auto output : schema1->outputs()) {
        auto type_str = std::string("T") + std::to_string(n + m);
        schema2.Output(n, output.GetName(), output.GetDescription(), type_str, output.GetOption(), false, output.GetMinArity(), output.GetDifferentiationCategory());
        schema2.TypeConstraint(type_str, DataTypeImpl::ToString(DataTypeImpl::AllTensorTypes()), "all types");
        n = n + 1;
      }
      schema2.SinceVersion(1);
      schema2.AllowUncheckedAttributes();
      if ((std::string)name == "super_layer") {
        schema2.TypeAndShapeInferenceFunction(xir_subgraph_shape_inference);
      } else if ((std::string)name == "FixNeuron") {
        schema2.TypeAndShapeInferenceFunction(xir_fixneuron_shape_inference);
      } else {
        schema2.TypeAndShapeInferenceFunction(xir_shape_infer);
      }
      ONNX_NAMESPACE::RegisterSchema(schema2, ORT_API_VERSION);
    }
  }
}

}  // namespace vaip
