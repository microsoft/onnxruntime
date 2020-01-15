// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <set>
#include "core/common/make_unique.h"
#include "core/graph/contrib_ops/contrib_defs.h"

namespace onnxruntime {
namespace training {

#define GRADIENT_OP_VERSION 9

/**
 * @brief An adapter for OpSchema in onnx.
 *
 * GradOpSchema provides an easy way to create a relaxed schema for gradient ops
 **/

class GradOpSchema {
  typedef ONNX_NAMESPACE::OpSchema::FormalParameterOption ParameterOption;

 public:
  GradOpSchema(const std::string& name,
               const std::string& file,
               const int line) : op_schema_(onnxruntime::make_unique<ONNX_NAMESPACE::OpSchema>(name, file, line)),
                                 schema_registry_(ONNX_NAMESPACE::OpSchemaRegistry::Instance()),
                                 variadic_input_(false),
                                 variadic_output_(false) {}

  ~GradOpSchema() = default;

  GradOpSchema& SinceVersion(ONNX_NAMESPACE::OperatorSetVersion v);

  GradOpSchema& SetSupportLevel(ONNX_NAMESPACE::OpSchema::SupportType supportType);

  /**
   * @brief A single input.
   */
  GradOpSchema& NumInputs(const int n);

  /**
   * @brief Input could be in range [min, max], inclusive.
   */
  GradOpSchema& NumInputs(const int min, const int max);

  /**
   * @brief Input could be one of the values specified in allowed_input_nums.
   */
  GradOpSchema& NumInputs(const std::set<int>& allowed_input_nums);

  // Sets the number of outputs, either a fixed number, a min and a max,
  // or a function that takes in the input number and produces an output
  // number. Use only one function in the set below.

  /**
   * @brief A single output.
   */
  GradOpSchema& NumOutputs(const int n);

  /**
   * @brief Output could be in range [min, max], inclusive.
   */
  GradOpSchema& NumOutputs(const int min, const int max);

  /**
   * @brief Output could be one of the values specified in allowed_output_nums.
   */
  GradOpSchema& NumOutputs(const std::set<int>& allowed_output_nums);

  GradOpSchema& Input(
      const int n,
      const std::string& name,
      const std::string& description,
      const std::string& type_str,
      const ParameterOption& param_option = ParameterOption::Single,
      bool is_homogeneous = true);

  GradOpSchema& Output(
      const int n,
      const std::string& name,
      const std::string& description,
      const std::string& type_str,
      const ParameterOption& param_option = ParameterOption::Single,
      bool is_homogeneous = true);

  GradOpSchema& TypeConstraint(
      const std::string& type_str,
      const std::vector<std::string>& constraints,
      const std::string& description);

  /**
   * @brief Last Input is variadic
   */
  GradOpSchema& VariadicInput();

  /**
   * @brief Last Output is variadic
   */
  GradOpSchema& VariadicOutput();

  // Fills the gradient schema op using the parameter schema name provided
  GradOpSchema& Reference(const std::string& fw_op_schema_name, const int sinceVersion = GRADIENT_OP_VERSION);

  // Fills the gradient schema op using the parameter schema name provided
  GradOpSchema& ReferenceAttributes(const std::string& fw_op_schema_name, const int sinceVersion = GRADIENT_OP_VERSION);

  ONNX_NAMESPACE::OpSchema& GetOpSchema() { return *op_schema_; }

 private:
  int min_input_ = 0;
  int max_input_ = 0;
  int min_output_ = 0;
  int max_output_ = 0;

  std::unique_ptr<ONNX_NAMESPACE::OpSchema> op_schema_;
  const ONNX_NAMESPACE::ISchemaRegistry* schema_registry_;

  bool variadic_input_;
  bool variadic_output_;

  std::function<void(ONNX_NAMESPACE::OpSchema&)> GenGradientSchema(const ONNX_NAMESPACE::OpSchema* base_op);
  std::function<void(ONNX_NAMESPACE::OpSchema&)> CopyAttributes(const ONNX_NAMESPACE::OpSchema* base_op);
  ParameterOption GetInputParameterType(const int arg_index);
  ParameterOption GetOutputParameterType(const int arg_index);
  ParameterOption GetParameterType(const int arg_index, const int max, const bool variadic);
};

class GradOpSchemaRegisterOnce final {
 public:
  GradOpSchemaRegisterOnce(GradOpSchema& grad_op_schema) {
    ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(grad_op_schema.GetOpSchema());
  }
};

#define ONNX_GRADIENT_OPERATOR_SCHEMA(name) \
  ONNX_GRADIENT_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define ONNX_GRADIENT_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name) \
  ONNX_GRADIENT_OPERATOR_SCHEMA_UNIQ(Counter, name)
#define ONNX_GRADIENT_OPERATOR_SCHEMA_UNIQ(Counter, name)                               \
  static GradOpSchemaRegisterOnce(op_schema_register_once##name##Counter) ONNX_UNUSED = \
      GradOpSchema(#name, __FILE__, __LINE__)                                           \
          .SinceVersion(GRADIENT_OP_VERSION)                                            \
          .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)

}  // namespace training
}  // namespace onnxruntime
