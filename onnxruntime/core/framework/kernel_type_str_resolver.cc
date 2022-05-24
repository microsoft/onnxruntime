// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_type_str_resolver.h"

#include "core/graph/op_identifier_utils.h"

namespace onnxruntime {

gsl::span<const ArgTypeAndIndex> KernelTypeStrResolver::ResolveKernelTypeStr(
    const OpIdentifier& op_id, const std::string& kernel_type_str) const {
  if (auto op_it = op_type_str_map_.find(op_id); op_it != op_type_str_map_.end()) {
    const auto& type_str_map = op_it->second;
    if (const auto type_str_it = type_str_map.find(kernel_type_str); type_str_it != type_str_map.end()) {
      return type_str_it->second;
    }
  }
  ORT_THROW("Failed to resolve type string '", kernel_type_str, "' for op ", op_id);
}

bool KernelTypeStrResolver::RegisterKernelTypeStrToArgsMap(OpIdentifier op_id,
                                                           KernelTypeStrToArgsMap kernel_type_str_to_args) {
  return op_type_str_map_.try_emplace(std::move(op_id), std::move(kernel_type_str_to_args)).second;
}

#if !defined(ORT_MINIMAL_BUILD)
bool KernelTypeStrResolver::RegisterOpSchema(const ONNX_NAMESPACE::OpSchema& op_schema) {
  auto op_id = MakeOpId(op_schema);
  if (Contains(op_type_str_map_, op_id)) {
    return false;
  }

  const auto type_constraint_names = [&]() {
    const auto& type_constraints = op_schema.typeConstraintParams();
    InlinedHashSet<std::string_view> names{};
    names.reserve(type_constraints.size());
    for (const auto& type_constraint : type_constraints) {
      names.emplace(type_constraint.type_param_str);
    }
    return names;
  }();

  InlinedHashMap<std::string, InlinedVector<ArgTypeAndIndex>> kernel_type_str_map{};
  // one entry for each type constraint, input, and output name
  kernel_type_str_map.reserve(type_constraint_names.size() +
                              op_schema.inputs().size() + op_schema.outputs().size());

  auto process_formal_params = [&](ArgType arg_type,
                                   gsl::span<const ONNX_NAMESPACE::OpSchema::FormalParameter> params) {
    for (size_t i = 0; i < params.size(); ++i) {
      const auto& param = params[i];
      auto curr_arg_type_and_idx = ArgTypeAndIndex{arg_type, i};

      // handle type constraint
      if (const auto& type_str = param.GetTypeStr();
          Contains(type_constraint_names, type_str)) {
        kernel_type_str_map[type_str].push_back(curr_arg_type_and_idx);
      }

      // handle input/output name
      auto& args_for_io_name = kernel_type_str_map[param.GetName()];
      if (!args_for_io_name.empty()) {
        // It's possible that an input and output have the same name (e.g, BatchNormalization-9 has both an input and
        // an output named 'mean').
        // If so, their formal parameters also need to have the same type string. Otherwise, it would be ambiguous to
        // use that name as a kernel type string.
        auto formal_param_type_str = [&](const ArgTypeAndIndex& arg_type_and_idx) {
          const auto [arg_type, idx] = arg_type_and_idx;
          const auto& formal_params = arg_type == ArgType::kInput ? op_schema.inputs() : op_schema.outputs();
          return formal_params[idx].GetTypeStr();
        };

        ORT_ENFORCE(formal_param_type_str(curr_arg_type_and_idx) == formal_param_type_str(args_for_io_name.front()),
                    "Kernel type string already exists for formal parameter name '", param.GetName(),
                    "', but the existing argument's formal parameter type string is different.");
      }
      args_for_io_name.push_back(std::move(curr_arg_type_and_idx));
    }
  };

  process_formal_params(ArgType::kInput, op_schema.inputs());
  process_formal_params(ArgType::kOutput, op_schema.outputs());

  return RegisterKernelTypeStrToArgsMap(std::move(op_id), std::move(kernel_type_str_map));
}

bool KernelTypeStrResolver::RegisterNodeOpSchema(const Node& node) {
  ORT_ENFORCE(node.Op() != nullptr, "Op schema must be available.");
  return RegisterOpSchema(*node.Op());
}
#endif  // !defined(ORT_MINIMAL_BUILD)

}
