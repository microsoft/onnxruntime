// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_registry.h"

#include <algorithm>
#include <memory>
#include <unordered_map>

#include "core/common/container_utils.h"
#include "core/framework/session_state.h"

namespace onnxruntime {

#if !defined(ORT_MINIMAL_BUILD)
bool KernelTypeStrResolver::RegisterOpSchema(const ONNX_NAMESPACE::OpSchema& op_schema) {
  const auto type_constraint_names = [&]() {
    const auto& type_constraints = op_schema.typeConstraintParams();
    InlinedHashSet<std::string_view> names{};
    names.reserve(type_constraints.size());
    for (const auto& type_constraint : type_constraints) {
      names.emplace(type_constraint.type_param_str);
    }
    return names;
  }();

  InlinedHashMap<std::string, InlinedVector<ArgTypeAndIndex>> type_str_map{};
  // one entry for each type constraint, input, and output name
  type_str_map.reserve(type_constraint_names.size() +
                       op_schema.inputs().size() + op_schema.outputs().size());

  auto process_formal_params = [&](ArgType arg_type,
                                   gsl::span<const ONNX_NAMESPACE::OpSchema::FormalParameter> params) {
    for (size_t i = 0; i < params.size(); ++i) {
      const auto& param = params[i];
      const auto& type_str = param.GetTypeStr();
      if (Contains(type_constraint_names, type_str)) {
        type_str_map[type_str].push_back(ArgTypeAndIndex{arg_type, i});
      }
      const bool added = type_str_map.try_emplace(param.GetName(),
                                                  InlinedVector<ArgTypeAndIndex>{{arg_type, i}})
                             .second;
      ORT_ENFORCE(added, "Type string already exists for formal parameter name: ", param.GetName());
    }
  };

  process_formal_params(ArgType::kInput, op_schema.inputs());
  process_formal_params(ArgType::kOutput, op_schema.outputs());

  return op_type_str_map_.try_emplace(OpIdentifier{op_schema.Name(), op_schema.domain(), op_schema.SinceVersion()},
                                      std::move(type_str_map))
      .second;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

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
                         const KernelTypeStrResolver& kernel_type_str_resolver,
                         std::string& mismatch_reason) {
  // for each type constraint
  //   map type constraint to arg
  //   check arg type against type constraint enabled types
  const auto& kernel_type_constraints = kernel_def.EnabledTypeConstraints();
  for (const auto& [kernel_type_str, enabled_types] : kernel_type_constraints) {
    const auto op_id = OpIdFromNode(node);
    const auto constraint_args = kernel_type_str_resolver.ResolveKernelTypeStr(op_id, kernel_type_str);

    for (const auto [arg_type, formal_arg_idx] : constraint_args) {
      const NodeArg* arg;
      if (arg_type == ArgType::kInput) {
        const auto& input_arg_counts = node.InputArgCount();
        const size_t first_arg_idx = static_cast<size_t>(std::accumulate(input_arg_counts.begin(),
                                                                         input_arg_counts.begin() + formal_arg_idx,
                                                                         int{0}));
        arg = node.InputDefs()[first_arg_idx];
      } else {
        arg = node.OutputDefs()[formal_arg_idx];
      }

      if (arg->Exists()) {
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
                                     const KernelTypeStrResolver& kernel_type_str_resolver,
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
                                     const KernelTypeStrResolver& kernel_type_str_resolver,
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
Status KernelRegistry::TryCreateKernel(const Node& node,
                                       const IExecutionProvider& execution_provider,
                                       const std::unordered_map<int, OrtValue>& constant_initialized_tensors,
                                       const OrtValueNameIdxMap& ort_value_name_idx_map,
                                       FuncManager& funcs_mgr,
                                       const DataTransferManager& data_transfer_mgr,
                                       /*out*/ std::unique_ptr<OpKernel>& op_kernel) const {
  const KernelCreateInfo* kernel_create_info = nullptr;
  ORT_RETURN_IF_ERROR(TryFindKernel(node, execution_provider.Type(), &kernel_create_info));
  OpKernelInfo kernel_info(node,
                           *kernel_create_info->kernel_def,
                           execution_provider,
                           constant_initialized_tensors,
                           ort_value_name_idx_map,
                           data_transfer_mgr);
  return kernel_create_info->kernel_create_func(funcs_mgr, kernel_info, op_kernel);
}

Status KernelRegistry::TryFindKernel(const Node& node,
                                     ProviderType exec_provider,
                                     const KernelCreateInfo** out) const {
  KernelTypeStrResolver kernel_type_str_resolver{};
  ORT_RETURN_IF(node.Op() == nullptr, "Op schema must be available.");
  kernel_type_str_resolver.RegisterOpSchema(*node.Op());
  return TryFindKernel(node, exec_provider, kernel_type_str_resolver, out);
}

Status KernelRegistry::TryFindKernel(const std::string& op_name, const std::string& domain, const int& version,
                                     const std::unordered_map<std::string, MLDataType>& type_constraints,
                                     ProviderType exec_provider, const KernelCreateInfo** out) const {
  *out = nullptr;
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
        *out = &kci;  // found match, exit loop
        break;
      }
    }  // if
  }    // for
  return *out == nullptr ? Status(common::ONNXRUNTIME, common::FAIL, "Kernel not found") : Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

bool KernelRegistry::TryFindKernelByHash(HashValue kernel_def_hash, const KernelCreateInfo** out) const {
  const auto hash_lookup_it = kernel_def_hash_lookup_.find(kernel_def_hash);
  if (hash_lookup_it == kernel_def_hash_lookup_.end()) {
    if (out) *out = nullptr;
    return false;
  }

  if (out) *out = &hash_lookup_it->second->second;
  return true;
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

  // check for existing hash conflict
  const auto kernel_def_hash = create_info.kernel_def->GetHash();
  ORT_RETURN_IF(kernel_def_hash_lookup_.find(kernel_def_hash) != kernel_def_hash_lookup_.end(),
                "Failed to add kernel for " + key + ": Conflict with existing kernel def hash.");

  // Register the kernel.
  // Ownership of the KernelDef is transferred to kernel_creator_fn_map_.
  auto it = kernel_creator_fn_map_.emplace(key, std::move(create_info));
  kernel_def_hash_lookup_.emplace(kernel_def_hash, it);
  return Status::OK();
}

KernelDefHashes KernelRegistry::ExportKernelDefHashes() const {
  KernelDefHashes result{};
  result.reserve(kernel_creator_fn_map_.size());
  std::transform(
      kernel_creator_fn_map_.begin(), kernel_creator_fn_map_.end(),
      std::back_inserter(result),
      [](const KernelCreateMap::value_type& kvp) {
        return std::make_pair(kvp.first, kvp.second.kernel_def->GetHash());
      });
  std::sort(result.begin(), result.end());
  return result;
}

}  // namespace onnxruntime
