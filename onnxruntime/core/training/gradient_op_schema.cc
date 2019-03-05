// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gradient_op_schema.h"
#include <string>

namespace onnxruntime {
namespace training {

GradOpSchema& GradOpSchema::NumInputs(const int min, const int max) {
  min_input_ = min;
  max_input_ = max;
  return *this;
}

GradOpSchema& GradOpSchema::NumInputs(const int n) {
  return NumInputs(n, n);
}

GradOpSchema& GradOpSchema::NumInputs(const std::set<int>& num_inputs_allowed) {
  op_schema_->NumInputs(num_inputs_allowed);
  return *this;
}

GradOpSchema& GradOpSchema::NumOutputs(const int min, const int max) {
  min_output_ = min;
  max_output_ = max;
  return *this;
}

GradOpSchema& GradOpSchema::NumOutputs(const int n) {
  return NumOutputs(n, n);
}

GradOpSchema& GradOpSchema::NumOutputs(const std::set<int>& num_outputs_allowed) {
  op_schema_->NumInputs(num_outputs_allowed);
  return *this;
}

GradOpSchema& GradOpSchema::VariadicInput() {
  variadic_input_ = true;
  return *this;
};

GradOpSchema& GradOpSchema::VariadicOutput() {
  variadic_output_ = true;
  return *this;
};

using namespace ONNX_NAMESPACE;

GradOpSchema& GradOpSchema::Reference(const std::string& fw_op_schema_name, const int sinceVersion) {
  op_schema_->FillUsing(GenGradientSchema(schema_registry_->GetSchema(fw_op_schema_name, sinceVersion)));
  return *this;
}

GradOpSchema::ParameterOption GradOpSchema::GetParameterType(const int arg_index, const int max, const bool variadic = false) {
  if (arg_index == max - 1 && variadic) {
    return ParameterOption::Variadic;
  } else {
    return ParameterOption::Optional;
  }
}

GradOpSchema::ParameterOption GradOpSchema::GetInputParameterType(const int arg_index) {
  return GetParameterType(arg_index, max_input_, variadic_input_);
}

GradOpSchema::ParameterOption GradOpSchema::GetOutputParameterType(const int arg_index) {
  return GetParameterType(arg_index, max_output_, variadic_output_);
}

std::function<void(ONNX_NAMESPACE::OpSchema&)> GradOpSchema::GenGradientSchema(const OpSchema* base_op) {
  {
    return [=](OpSchema& grad_op_schema) {
      // TODO: The version here has to match the kernel version and cannot just depend on fw_op version
      // Figure out a good way to derive matching versions for schema and kernel
      grad_op_schema.SinceVersion(8);
      grad_op_schema.SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL);
      grad_op_schema.TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types");

      if (base_op != nullptr && base_op->typeConstraintParams().size() == 1) {
        auto type_constraint = base_op->typeConstraintParams()[0];
        grad_op_schema.TypeConstraint(
            type_constraint.type_param_str,
            type_constraint.allowed_type_strs,
            type_constraint.description);
      } else {
        grad_op_schema.TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types");
      }

      // add inputs
      // TODO: Should have a way to express heterogenous variadic param for both inputs and outputs
      std::string type_str = grad_op_schema.typeConstraintParams()[0].type_param_str;
      for (int i = 0; i < max_input_; i++) {
        grad_op_schema.Input(
            i,
            "grad_input_arg" + std::to_string(i),  // name
            "",                                    //domain
            type_str,
            GetInputParameterType(i),
            true);
      }

      // add outputs
      for (int i = 0; i < max_output_; i++) {
        grad_op_schema.Output(
            i,
            "grad_output_arg" + std::to_string(i),  //name
            "",                                     //domain
            type_str,
            GetOutputParameterType(i),
            true);
      }

      // copy over all the attributes schema
      if (base_op != nullptr) {
        auto attributes = base_op->attributes();
        for (auto pair : attributes) {
          grad_op_schema.Attr(pair.second);
        }
      }
    };
  }
}

}  // namespace training
}  // namespace onnxruntime
