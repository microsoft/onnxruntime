// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_registry.h"

#include <algorithm>
#include <memory>
#include <unordered_map>

#include "core/framework/session_state.h"

namespace onnxruntime {

#if !defined(ORT_MINIMAL_BUILD)
namespace {
// Traverses the node's formal parameters and calls TraverseFn with the formal
// parameter and its associated TypeProto.
//   node - the node to traverse
//   param_filter_fn - called to determine whether to consider a given formal parameter:
//     bool ParamFilterFn(const ONNX_NAMESPACE::OpSchema::FormalParameter& param)
//       param - the formal parameter
//       returns true if the formal parameter should be considered, false otherwise
//   traverse_fn - called to process the formal parameter and its associated TypeProto:
//     bool TraverseFn(const ONNX_NAMESPACE::OpSchema::FormalParameter& param,
//                     const ONNX_NAMESPACE::TypeProto* type)
//       param - the formal parameter
//       type - the associated TypeProto
//       returns true if traversal should continue, false otherwise
template <typename ParamFilterFn, typename TraverseFn>
bool TraverseFormalParametersWithTypeProto(const Node& node,
                                           ParamFilterFn param_filter_fn,
                                           TraverseFn traverse_fn) {
  const ONNX_NAMESPACE::OpSchema& op_schema = *node.Op();

  // was the param name matched in either inputs, outputs or type constraints. 
  // this validates the name was valid and that the type involved will be returned if available.
  // if the name is invalid we do not return a type, and any applicable type constraint can not be applied 
  // in VerifyKernelDef.
  bool matched = false;

  // process inputs:
  const size_t len = node.InputArgCount().size();
  ORT_ENFORCE(len <= op_schema.inputs().size());
  int actual_index = 0;
  for (size_t formal_index = 0; formal_index != len; ++formal_index) {
    const auto& param = op_schema.inputs()[formal_index];
    if (param_filter_fn(param)) {
      matched = true;
      // get type of any corresponding actual parameter, if present
      for (int i = 0, end = node.InputArgCount()[formal_index]; i < end; ++i) {
        const NodeArg* arg = node.InputDefs()[static_cast<size_t>(actual_index) + i];
        if (!arg->Exists()) continue;  // a missing optional argument
        if (!traverse_fn(param, arg->TypeAsProto())) return matched;
      }
    }
    actual_index += node.InputArgCount()[formal_index];
  }

  // process outputs:
  auto actual_outputs = node.OutputDefs();
  const auto num_actual_outputs = actual_outputs.size();
  const auto& schema_outputs = op_schema.outputs();
  const auto last_formal = schema_outputs.size() - 1;
  size_t i = 0;
  for (; i != num_actual_outputs; ++i) {
    const auto& formal = schema_outputs[std::min(i, last_formal)];
    if (!param_filter_fn(formal)) continue;
    matched = true;
    const NodeArg* arg = actual_outputs[i];
    if (!arg->Exists()) continue;
    if (!traverse_fn(formal, arg->TypeAsProto())) return matched;
  }

  // missing optional outputs. check if type constraint name was valid if we haven't matched anything yet.
  if (!matched) {
    while (i <= last_formal) {
      if (param_filter_fn(schema_outputs[i])) {
        matched = true;
        break;
      }

      ++i;
    }
  }

  return matched;
}

class TypeBindingResolver {
 public:
  TypeBindingResolver(const Node& node, bool use_lookup_map)
      : node_(node),
        type_binding_map_() {
    if (use_lookup_map) {
      type_binding_map_ = std::make_unique<TypeBindingMap>();
      TraverseFormalParametersWithTypeProto(
          node_,
          [](const ONNX_NAMESPACE::OpSchema::FormalParameter&) -> bool { return true; },
          [this](const ONNX_NAMESPACE::OpSchema::FormalParameter& param,
                 const ONNX_NAMESPACE::TypeProto* type) -> bool {
            type_binding_map_->emplace(param.GetName(), type);
            type_binding_map_->emplace(param.GetTypeStr(), type);
            return true;
          });
    }
  }

  // Resolves a type constraint name to a TypeProto* for a given node. ONNX code checks that all usages of the type
  // constraint name by the node are consistent, so we just need to match the first usage to see the actual type
  // being used by the node. e.g. if type constraint 'T' allows float and double, any input or output for that node
  // that has constraint 'T' must use the same type, be that float or double.
  //
  // Also can resolve an input/output name to a contraint when a type constraint name is not used.
  // e.g. the 'shape' input of Reshape has a directly specified constraint of 'tensor(int64)'.
  //
  // Returns the resolved TypeProto* or nullptr if unable to resolve due to the
  // constraint being for a missing optional output.
  const ONNX_NAMESPACE::TypeProto* Resolve(const std::string& name_or_type_str) const {
    const ONNX_NAMESPACE::TypeProto* result{};
    bool matched = false;

    // lookup if available
    if (type_binding_map_) {
      auto found_it = type_binding_map_->find(name_or_type_str);
      matched = found_it != type_binding_map_->end();
      if (matched) {
        result = found_it->second;
      }
    }

    if (!matched) {
      // fall back to node parameter traversal
      matched = TraverseFormalParametersWithTypeProto(
          node_,
          [&name_or_type_str](const ONNX_NAMESPACE::OpSchema::FormalParameter& param) -> bool {
            return param.GetTypeStr() == name_or_type_str || param.GetName() == name_or_type_str;
          },
          [&result](const ONNX_NAMESPACE::OpSchema::FormalParameter&,
                    const ONNX_NAMESPACE::TypeProto* type) -> bool {
            result = type;
            return false;
          });
    }

// invalid kernel def with type constraints that don't match the schema. this means the type constraints are not
// actually applied, making the kernel def misleading and potentially matching an unexpected/incorrect kernel.
// warn in a release build as we do not have coverage of every single opset for every single operator
// in the unit tests, so issues may be missed and the model may still work (e.g. matches the correct kernel by chance).
// throw in a debug build so the issue is obvious and force it to be fixed.
#ifdef NDEBUG
    if (!matched) {
      LOGS_DEFAULT(WARNING) << name_or_type_str << " constraint was not found for " << node_.OpType();
    }
#else
    ORT_ENFORCE(matched, name_or_type_str, " constraint was not found for ", node_.OpType());
#endif
    return result;
  }

 private:
  // map from input/output name or type string to TypeProto pointer
  using TypeBindingMap = std::unordered_map<std::string, const ONNX_NAMESPACE::TypeProto*>;

  const Node& node_;
  std::unique_ptr<TypeBindingMap> type_binding_map_;
};
};  // namespace

bool KernelRegistry::VerifyKernelDef(const Node& node,
                                     const KernelDef& kernel_def,
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

  // check if type matches
  auto& kernel_type_constraints = kernel_def.EnabledTypeConstraints();

  // Note: The number of formal input/output parameters is N and the number of
  // type constraints is M. We select between an O(N*M) and an O(N+M) approach.
  // The O(N*M) approach has lower initial overhead.
  // kTypeBindingResolverComplexityThreshold is the value of N*M above which we
  // will use the O(N+M) approach.
  constexpr int kTypeBindingResolverComplexityThreshold = 50 * 50;
  const bool use_lookup_map = (kernel_type_constraints.size() * (node.Op()->inputs().size() + node.Op()->outputs().size()) >
                               kTypeBindingResolverComplexityThreshold);
  TypeBindingResolver type_binding_resolver{node, use_lookup_map};

  for (auto& constraint : kernel_type_constraints) {
    const std::string& name = constraint.first;
    const std::vector<MLDataType>& allowed_types = constraint.second;
    const ONNX_NAMESPACE::TypeProto* actual_type = type_binding_resolver.Resolve(name);

    // If actual_type is null, this represents a type-constraint on a
    // missing optional parameter, which can be skipped.
    // TODO: We should check that names specified in kernel_type_constraints are
    // valid names (of types or parameters) at the time that kernels are registered.
    if (nullptr != actual_type) {
      bool is_type_compatible = std::any_of(allowed_types.begin(), allowed_types.end(),
                                            [actual_type](const DataTypeImpl* expected_type) {
                                              bool rc = expected_type->IsCompatible(*actual_type);  // for easier debugging
                                              return rc;
                                            });
      if (!is_type_compatible) {
        std::ostringstream ostr;
        ostr << "Found kernel for Op with name (" << node.Name() << ")"
             << " and type (" << node.OpType() << ")"
             << " in the supported version range"
             << " (node_version: " << node_since_version
             << " kernel start version: " << kernel_start_version
             << " kernel_end_version: " << kernel_end_version << ")."
             << " However the types are incompatible."
             << " This op has been implemented only for the following types (";
        for (const auto& allowed_type : allowed_types) {
          ostr << DataTypeImpl::ToString(allowed_type) << ",";
        }
        ostr << "),";
        const char* actual_type_str = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*actual_type));
        ostr << " but the node in the model has the following type (" << actual_type_str << ")";
        error_str = ostr.str();
        return false;
      }
    }
  }
  return true;
}

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

static std::string ToString(const std::vector<std::string>& error_strs) {
  std::ostringstream ostr;
  std::for_each(std::begin(error_strs), std::end(error_strs),
                [&ostr](const std::string& str) { ostr << str << "\n"; });
  return ostr.str();
}

// It's often this function returns a failed status, but it is totally expected.
// It just means this registry doesn't have such a kernel, please search it elsewhere.
// if this function is called before graph partition, then node.provider is not set.
// In this case, the kernel's provider must equal to exec_provider
// otherwise, kernel_def.provider must equal to node.provider. exec_provider is ignored.
Status KernelRegistry::TryFindKernel(const Node& node,
                                     ProviderType exec_provider,
                                     const KernelCreateInfo** out) const {
  const auto& node_provider = node.GetExecutionProviderType();
  const auto& expected_provider = (node_provider.empty() ? exec_provider : node_provider);

  auto range = kernel_creator_fn_map_.equal_range(GetMapKey(node.OpType(), node.Domain(), expected_provider));
  if (out) *out = nullptr;

  std::vector<std::string> verify_kernel_def_error_strs;

  for (auto i = range.first; i != range.second; ++i) {
    std::string error_str;
    if (VerifyKernelDef(node, *i->second.kernel_def, error_str)) {
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
        << " Encountered following errors: (" << ToString(verify_kernel_def_error_strs) << ")";

    VLOGS_DEFAULT(2) << "TryFindKernel failed, Reason: " << oss.str();
    return Status(common::ONNXRUNTIME, common::FAIL, oss.str());
  }

  return Status(common::ONNXRUNTIME, common::FAIL, "Kernel not found");
}

Status KernelRegistry::TryFindKernel(const std::string& op_name, const std::string& domain, const int& version,
                                     const std::unordered_map<std::string, MLDataType>& type_constraints,
                                     ProviderType exec_provider, const KernelCreateInfo** out) const {
  *out = nullptr;
  auto range = kernel_creator_fn_map_.equal_range(GetMapKey(op_name, domain, exec_provider));
  for (auto i = range.first; i != range.second; ++i) {  //loop through all kernels
    const KernelCreateInfo& kci = i->second;
    int start_ver{};
    int end_ver{};
    kci.kernel_def->SinceVersion(&start_ver, &end_ver);
    if (start_ver <= version && end_ver >= version) {  //try match the version
      auto& kci_constraints = kci.kernel_def->TypeConstraints();
      bool match = true;
      for (auto& constraint : type_constraints) {  //try match type constraints
        auto iter = kci_constraints.find(constraint.first);
        if (iter == kci_constraints.end() || find(iter->second.begin(), iter->second.end(), constraint.second) == iter->second.end()) {
          match = false;
          break;
        }
      }  //for
      if (match) {
        *out = &kci;  //found match, exit loop
        break;
      }
    }  //if
  }    //for
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
