// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4702)
#endif

#include "core/framework/kernel_registry.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <unordered_map>

#include "core/framework/kernel_type_str_resolver.h"
#include "core/framework/session_state.h"

namespace onnxruntime {

namespace {
bool IsTypeProtoCompatible(gsl::span<const MLDataType> enabled_types, const ONNX_NAMESPACE::TypeProto& actual_type,
                           std::string& mismatch_reason) {
  const bool is_type_compatible = std::any_of(
      enabled_types.begin(), enabled_types.end(),
      [&actual_type](const DataTypeImpl* expected_type) {
        bool rc = expected_type->IsCompatible(actual_type);  // for easier debugging
        return rc;
      });

  if (!is_type_compatible) {
    std::ostringstream ostr;
    ostr << "This op has been implemented only for the following types (";
    for (const auto& enabled_type : enabled_types) {
      ostr << DataTypeImpl::ToString(enabled_type) << ",";
    }
    ostr << "),";
    const char* actual_type_str = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(actual_type));
    ostr << " but the node in the model has the following type (" << actual_type_str << ")";
    mismatch_reason = ostr.str();
    return false;
  }

  return true;
}

// TODO: should this function return a enum with 3 states: match/mismatch/arg not exists?
bool IsTypeCompatibleOnIthArg(const ConstPointerContainer<std::vector<NodeArg*>>& actual_inputs, const ConstPointerContainer<std::vector<NodeArg*>>& actual_outputs,
                              const std::vector<int>& actual_input_arg_counts, const InlinedVector<int> actual_input_arg_offsets,
                              const std::vector<MLDataType>& enabled_types, const ArgType& arg_type, const size_t& formal_arg_idx, std::string& mismatch_reason) {
  const NodeArg* arg;
  if (arg_type == ArgType::kInput) {
    if (formal_arg_idx >= actual_input_arg_counts.size() ||
        actual_input_arg_counts[formal_arg_idx] == 0) {
      arg = nullptr;
    } else {
      const auto first_arg_idx = actual_input_arg_offsets[formal_arg_idx];
      ORT_ENFORCE(static_cast<size_t>(first_arg_idx) < actual_inputs.size());
      arg = actual_inputs[first_arg_idx];
    }
  } else {
    arg = formal_arg_idx < actual_outputs.size() ? actual_outputs[formal_arg_idx] : nullptr;
  }

  if (arg && arg->Exists()) {
    const ONNX_NAMESPACE::TypeProto* type_proto = arg->TypeAsProto();
    ORT_ENFORCE(type_proto != nullptr);

    if (!IsTypeProtoCompatible(enabled_types, *type_proto, mismatch_reason)) {
      return false;
    }
  }
  return true;
}

// match the kernel using type info from the Node's args
bool MatchKernelDefTypes(const Node& node,
                         const std::unordered_map<std::string, std::vector<MLDataType>>& kernel_type_constraints,
                         const IKernelTypeStrResolver& kernel_type_str_resolver,
                         std::string& mismatch_reason) {
  const auto actual_inputs = node.InputDefs();
  const auto actual_outputs = node.OutputDefs();
  const auto& actual_input_arg_counts = node.InputArgCount();
  const auto actual_input_arg_offsets = [&actual_input_arg_counts]() {
    InlinedVector<int> offsets{};
    offsets.reserve(actual_input_arg_counts.size());
    // std::exclusive_scan() is not supported until GCC 9.3
    // std::exclusive_scan(actual_input_arg_counts.begin(), actual_input_arg_counts.end(),
    //                     std::back_inserter(offsets), 0);
    int current_offset = 0;
    for (size_t i = 0; i < actual_input_arg_counts.size(); ++i) {
      offsets.push_back(current_offset);
      current_offset += actual_input_arg_counts[i];
    }
    return offsets;
  }();

  // custom kernel's type constraints look like {"Input0"->[float, double], "Input1"->[..,..], "Output0"->[..,..]}
  // compare the ith arg with the node's corresponding arg directly
  if (kernel_type_constraints.find("Input0") != kernel_type_constraints.end() ||
      kernel_type_constraints.find("Output0") != kernel_type_constraints.end()) {
    for (const auto& [kernel_type_str, enabled_types] : kernel_type_constraints) {
      ArgType arg_type = ArgType::kInput;
      size_t arg_idx = 0;
      if (kernel_type_str.substr(0, 5) == "Input") {
        arg_idx = std::stoi(kernel_type_str.substr(5));
      } else if (kernel_type_str.substr(0, 6) == "Output") {
        arg_type = ArgType::kOutput;
        arg_idx = std::stoi(kernel_type_str.substr(6));
      }
      else {
        mismatch_reason = "Custom Op but key does not start with 'Input' or 'Output' in type constraints";
        return false;
      }

      if (!IsTypeCompatibleOnIthArg(actual_inputs, actual_outputs, actual_input_arg_counts, actual_input_arg_offsets,
                                    enabled_types, arg_type, arg_idx, mismatch_reason)) return false;
    }
    return true;
  }

  // for each type constraint
  //   map type constraint to arg
  //   check arg type against type constraint enabled types
  for (const auto& [kernel_type_str, enabled_types] : kernel_type_constraints) {
    gsl::span<const ArgTypeAndIndex> constraint_args{};
    ORT_THROW_IF_ERROR(kernel_type_str_resolver.ResolveKernelTypeStr(node, kernel_type_str, constraint_args));

    for (const auto& [arg_type, formal_arg_idx] : constraint_args) {
        if (!IsTypeCompatibleOnIthArg(actual_inputs, actual_outputs, actual_input_arg_counts, actual_input_arg_offsets,
                                    enabled_types, arg_type, formal_arg_idx, mismatch_reason)) return false;

        // found a match, don't need to check other args with this constraint
        break;
    }
  }

  return true;
}

bool MatchKernelDefTypes(const std::unordered_map<std::string, std::vector<MLDataType>>& kernel_type_constraints,
                         const KernelRegistry::TypeConstraintMap& type_constraints) {
  bool match = true;
  for (auto& constraint : type_constraints) {
    auto iter = kernel_type_constraints.find(constraint.first);
    if (iter == kernel_type_constraints.end() ||
        find(iter->second.begin(), iter->second.end(), constraint.second) == iter->second.end()) {
      match = false;
      break;
    }
  }

  return match;
}
}  // namespace

static bool VerifyVersion(int since_ver, const KernelDef& kernel_def, std::string& error_str) {
  // check if version matches
  int kernel_start_version;
  int kernel_end_version;
  kernel_def.SinceVersion(&kernel_start_version, &kernel_end_version);

  bool valid_version =
      // exact match. typical usage.
      kernel_start_version == since_ver ||
      // allow match if the kernel def has an end version. if it does not, all we know is that the kernel supported
      // the start version when it was created, and not whether a new version of the operator was added since then
      // that the kernel doesn't support.
      (kernel_end_version != INT_MAX &&
       kernel_start_version <= since_ver && kernel_end_version >= since_ver);

  if (!valid_version) {
    std::ostringstream ostr;
    ostr << " Version mismatch."
         << " node_version: " << since_ver
         << " kernel start version: " << kernel_start_version
         << " kernel_end_version: " << kernel_end_version;
    error_str = ostr.str();
  }
  return valid_version;
}

bool KernelRegistry::VerifyKernelDef(const Node& node,
                                     const KernelDef& kernel_def,
                                     const IKernelTypeStrResolver* kernel_type_str_resolver,
                                     const TypeConstraintMap* type_constraint_values,
                                     std::string& error_str) {
  // check if version matches
  bool valid_version = VerifyVersion(node.SinceVersion(), kernel_def, error_str);

  if (!valid_version) {
    return false;
  }

  std::string mismatch_reason;
  const auto& kernel_type_constraints = kernel_def.TypeConstraints();

  bool matched = type_constraint_values ? MatchKernelDefTypes(kernel_type_constraints, *type_constraint_values)
                                        : MatchKernelDefTypes(node, kernel_type_constraints, *kernel_type_str_resolver,
                                                              mismatch_reason);

  if (!matched) {
    std::ostringstream ostr;
    ostr << "Kernel found kernel"
         << " in the supported version range"
         << " (node_version: " << node.SinceVersion() << ")."
         << " However the types are incompatible. " << mismatch_reason;
    error_str = ostr.str();
  }

  return matched;
}

// It's often this function returns a failed status, but it is totally expected.
// It just means this registry doesn't have such a kernel, please search it elsewhere.
// if this function is called before graph partition, then node.provider is not set.
// In this case, the kernel's provider must equal to exec_provider
// otherwise, kernel_def.provider must equal to node.provider. exec_provider is ignored.
Status KernelRegistry::TryFindKernelImpl(const Node& node,
                                         ProviderType exec_provider,
                                         const IKernelTypeStrResolver* kernel_type_str_resolver,
                                         const TypeConstraintMap* type_constraints,
                                         const KernelCreateInfo** out) const {
  const auto& node_provider = node.GetExecutionProviderType();
  const auto& expected_provider = (node_provider.empty() ? exec_provider : node_provider);

  auto range = kernel_creator_fn_map_.equal_range(GetMapKey(node.OpType(), node.Domain(), expected_provider));
  if (out) *out = nullptr;

  std::vector<std::string> verify_kernel_def_error_strs;

  for (auto i = range.first; i != range.second; ++i) {
    std::string error_str;
    if (VerifyKernelDef(node, *i->second.kernel_def, kernel_type_str_resolver, type_constraints, error_str)) {
      if (out) {
        *out = &i->second;
      }
      return Status::OK();
    }

    verify_kernel_def_error_strs.push_back(error_str);
  }

  if (!verify_kernel_def_error_strs.empty()) {
    std::ostringstream oss;
    oss << "Op with name (" << node.Name() << ")"
        << " domain (" << node.Domain() << ")"
        << " and type (" << node.OpType() << ")"
        << " kernel is not supported in " << expected_provider << "."
        << " Encountered following errors: (";
    std::copy(verify_kernel_def_error_strs.begin(), verify_kernel_def_error_strs.end(),
              std::ostream_iterator<std::string>(oss, "\n"));
    oss << ")";

    VLOGS_DEFAULT(2) << "TryFindKernel failed, Reason: " << oss.str();
    return Status(common::ONNXRUNTIME, common::FAIL, oss.str());
  }

  return Status(common::ONNXRUNTIME, common::FAIL, "Kernel not found");
}

Status KernelRegistry::TryFindKernel(const Node& node, ProviderType exec_provider,
                                     const IKernelTypeStrResolver& kernel_type_str_resolver,
                                     const KernelCreateInfo** out) const {
  return TryFindKernelImpl(node, exec_provider, &kernel_type_str_resolver, nullptr, out);
}

Status KernelRegistry::TryFindKernel(const Node& node, ProviderType exec_provider,
                                     const TypeConstraintMap& type_constraints,
                                     const KernelCreateInfo** out) const {
  return TryFindKernelImpl(node, exec_provider, nullptr, &type_constraints, out);
}

static bool KernelDefCompatible(int version, const KernelDef& kernel_def,
                                const KernelRegistry::TypeConstraintMap& type_constraint_values,
                                std::string& error_str) {
  if (!VerifyVersion(version, kernel_def, error_str)) {
    return false;
  }

  const auto& kernel_type_constraints = kernel_def.TypeConstraints();
  bool matched = MatchKernelDefTypes(kernel_type_constraints, type_constraint_values);

  if (!matched) {
    std::ostringstream ostr;
    ostr << "Kernel found kernel"
         << " in the supported version range"
         << " (node_version: " << version << ")."
         << " However the types are incompatible.";
    error_str = ostr.str();
  }

  return matched;
}

Status KernelRegistry::TryFindKernel(ProviderType exec_provider,
                                     std::string_view op_type,
                                     std::string_view domain,
                                     int version,
                                     const KernelRegistry::TypeConstraintMap& type_constraints,
                                     const KernelCreateInfo** out) const {
  auto range = kernel_creator_fn_map_.equal_range(GetMapKey(op_type, domain, exec_provider));
  if (out) *out = nullptr;

  std::vector<std::string> verify_kernel_def_error_strs;

  for (auto i = range.first; i != range.second; ++i) {
    std::string error_str;
    if (KernelDefCompatible(version, *i->second.kernel_def, type_constraints, error_str)) {
      if (out) {
        *out = &i->second;
      }
      return Status::OK();
    }

    verify_kernel_def_error_strs.push_back(error_str);
  }

  if (!verify_kernel_def_error_strs.empty()) {
    std::ostringstream oss;
    oss << "Op type (" << op_type << ")"
        << " domain (" << domain << ")"
        << " kernel is not supported in " << exec_provider << "."
        << " Encountered following errors: (";
    std::copy(verify_kernel_def_error_strs.begin(), verify_kernel_def_error_strs.end(),
              std::ostream_iterator<std::string>(oss, "\n"));
    oss << ")";

    VLOGS_DEFAULT(2) << "TryFindKernel failed, Reason: " << oss.str();
    return Status(common::ONNXRUNTIME, common::FAIL, oss.str());
  }

  return Status(common::ONNXRUNTIME, common::FAIL, "Kernel not found");
}

Status KernelRegistry::Register(KernelDefBuilder& kernel_builder,
                                const KernelCreateFn& kernel_creator) {
  return Register(KernelCreateInfo(kernel_builder.Build(), kernel_creator));
}

Status KernelRegistry::Register(KernelCreateInfo&& create_info) {
  if (!create_info.kernel_def) {
    return Status(common::ONNXRUNTIME, common::FAIL, "kernel def can't be NULL");
  }
  const std::string key = GetMapKey(*create_info.kernel_def);
  // Check op version conflicts.
  const auto range = kernel_creator_fn_map_.equal_range(key);
  for (auto i = range.first; i != range.second; ++i) {
    if (i->second.kernel_def &&
        i->second.kernel_def->IsConflict(*create_info.kernel_def)) {
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "Failed to add kernel for " + key +
                        ": Conflicting with a registered kernel with op versions.");
    }
  }

  // Register the kernel.
  // Ownership of the KernelDef is transferred to kernel_creator_fn_map_.
  kernel_creator_fn_map_.emplace(key, std::move(create_info));
  return Status::OK();
}

}  // namespace onnxruntime
