// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

bool MatchKernelDefTypes(const Node& node,
                         const KernelDef& kernel_def,
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

  // for each type constraint
  //   map type constraint to arg
  //   check arg type against type constraint enabled types
  const auto& kernel_type_constraints = kernel_def.TypeConstraints();
  for (const auto& [kernel_type_str, enabled_types] : kernel_type_constraints) {
    gsl::span<const ArgTypeAndIndex> constraint_args{};
    ORT_THROW_IF_ERROR(kernel_type_str_resolver.ResolveKernelTypeStr(node, kernel_type_str,
                                                                     constraint_args));

    for (const auto& [arg_type, formal_arg_idx] : constraint_args) {
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

        // found a match, don't need to check other args with this constraint
        break;
      }
    }
  }

  return true;
}
}  // namespace

bool KernelRegistry::VerifyKernelDef(const Node& node,
                                     const KernelDef& kernel_def,
                                     const IKernelTypeStrResolver& kernel_type_str_resolver,
                                     std::string& error_str) {
  // check if version matches
  int kernel_start_version;
  int kernel_end_version;
  kernel_def.SinceVersion(&kernel_start_version, &kernel_end_version);

  int node_since_version = node.SinceVersion();
  // Ideal case is, if schema is Since(5), current opset version is opset 7,
  // kernel_def Since(8)     Invalid
  // kernel_def Since(6)     Valid
  // kernel_def Since(5)     Valid
  // kernel_def Since(4)     Invalid
  // kernel_def Since(4, 6)  Valid

  // Right now there is no "until version" on schema, it is difficult to get opset version here.(require a lot of interface change.)
  // As a trade off, we will temporary require kernel definition to have the same since version as schema definition.
  // so kernel_def Since(6) will become invalid now.
  // After ONNX add "until version" on the schema object, we will update this place
  bool valid_version = kernel_start_version == node_since_version  // the idea case this branch should be kernel_start_version >= node_version && kernel_start_version <= until_version
                       || (kernel_start_version < node_since_version && kernel_end_version != INT_MAX && kernel_end_version >= node_since_version);
  if (!valid_version) {
    std::ostringstream ostr;
    ostr << "Op with name (" << node.Name() << ")"
         << " and type (" << node.OpType() << ")"
         << " Version mismatch."
         << " node_version: " << node_since_version
         << " kernel start version: " << kernel_start_version
         << " kernel_end_version: " << kernel_end_version;
    error_str = ostr.str();
    return false;
  }

  if (std::string mismatch_reason;
      !MatchKernelDefTypes(node, kernel_def, kernel_type_str_resolver, mismatch_reason)) {
    std::ostringstream ostr;
    ostr << "Found kernel for Op with name (" << node.Name() << ")"
         << " and type (" << node.OpType() << ")"
         << " in the supported version range"
         << " (node_version: " << node_since_version
         << " kernel start version: " << kernel_start_version
         << " kernel_end_version: " << kernel_end_version << ")."
         << " However the types are incompatible. " << mismatch_reason;
    error_str = ostr.str();
    return false;
  }

  return true;
}

// It's often this function returns a failed status, but it is totally expected.
// It just means this registry doesn't have such a kernel, please search it elsewhere.
// if this function is called before graph partition, then node.provider is not set.
// In this case, the kernel's provider must equal to exec_provider
// otherwise, kernel_def.provider must equal to node.provider. exec_provider is ignored.
Status KernelRegistry::TryFindKernel(const Node& node,
                                     ProviderType exec_provider,
                                     const IKernelTypeStrResolver& kernel_type_str_resolver,
                                     const KernelCreateInfo** out) const {
  const auto& node_provider = node.GetExecutionProviderType();
  const auto& expected_provider = (node_provider.empty() ? exec_provider : node_provider);

  auto range = kernel_creator_fn_map_.equal_range(GetMapKey(node.OpType(), node.Domain(), expected_provider));
  if (out) *out = nullptr;

  std::vector<std::string> verify_kernel_def_error_strs;

  for (auto i = range.first; i != range.second; ++i) {
    std::string error_str;
    if (VerifyKernelDef(node, *i->second.kernel_def, kernel_type_str_resolver, error_str)) {
      if (out) *out = &i->second;
      return Status::OK();
    }
    verify_kernel_def_error_strs.push_back(error_str);
  }

  if (!verify_kernel_def_error_strs.empty()) {
    std::ostringstream oss;
    oss << "Op with name (" << node.Name() << ")"
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

#if !defined(ORT_MINIMAL_BUILD)
Status KernelRegistry::TryFindKernel(const std::string& op_name, const std::string& domain, const int& version,
                                     const std::unordered_map<std::string, MLDataType>& type_constraints,
                                     ProviderType exec_provider, const KernelCreateInfo** kernel_out) const {
  const KernelCreateInfo* kernel = nullptr;
  auto range = kernel_creator_fn_map_.equal_range(GetMapKey(op_name, domain, exec_provider));
  for (auto i = range.first; i != range.second; ++i) {  // loop through all kernels
    const KernelCreateInfo& kci = i->second;
    int start_ver{};
    int end_ver{};
    kci.kernel_def->SinceVersion(&start_ver, &end_ver);
    if (start_ver <= version && end_ver >= version) {  // try match the version
      auto& kci_constraints = kci.kernel_def->TypeConstraints();
      bool match = true;
      for (auto& constraint : type_constraints) {  // try match type constraints
        auto iter = kci_constraints.find(constraint.first);
        if (iter == kci_constraints.end() || find(iter->second.begin(), iter->second.end(), constraint.second) == iter->second.end()) {
          match = false;
          break;
        }
      }  // for
      if (match) {
        kernel = &kci;  // found match, exit loop
        break;
      }
    }  // if
  }    // for
  if (kernel_out) *kernel_out = kernel;
  return kernel == nullptr ? Status(common::ONNXRUNTIME, common::FAIL, "Kernel not found") : Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

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
