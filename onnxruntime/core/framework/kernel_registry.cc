// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_registry.h"

#include <algorithm>
#include <memory>
#include <unordered_map>

#include "core/framework/session_state.h"

using namespace ::onnxruntime::common;
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
//       param - the formal paremeter
//       type - the associated TypeProto
//       returns true if traversal should continue, false otherwise
template <typename ParamFilterFn, typename TraverseFn>
void TraverseFormalParametersWithTypeProto(const Node& node,
                                           ParamFilterFn param_filter_fn,
                                           TraverseFn traverse_fn) {
  const ONNX_NAMESPACE::OpSchema& op_schema = *node.Op();

  // process inputs:
  const size_t len = node.InputArgCount().size();
  ORT_ENFORCE(len <= op_schema.inputs().size());
  int actual_index = 0;
  for (size_t formal_index = 0; formal_index != len; ++formal_index) {
    const auto& param = op_schema.inputs()[formal_index];
    if (param_filter_fn(param)) {
      // get type of any corresponding actual parameter, if present
      for (int i = 0, end = node.InputArgCount()[formal_index]; i < end; ++i) {
        const NodeArg* arg = node.InputDefs()[actual_index + i];
        if (!arg->Exists()) continue;  // a missing optional argument
        if (!traverse_fn(param, arg->TypeAsProto())) return;
      }
    }
    actual_index += node.InputArgCount()[formal_index];
  }

  // process outputs:
  auto actual_outputs = node.OutputDefs();
  const auto num_actual_outputs = actual_outputs.size();
  const auto last_formal = op_schema.outputs().size() - 1;
  for (size_t i = 0; i != num_actual_outputs; ++i) {
    const auto& formal = op_schema.outputs()[std::min(i, last_formal)];
    if (!param_filter_fn(formal)) continue;
    const NodeArg* arg = actual_outputs[i];
    if (!arg->Exists()) continue;
    if (!traverse_fn(formal, arg->TypeAsProto())) return;
  }
}

class TypeBindingResolver {
 public:
  TypeBindingResolver(const Node& node, bool use_lookup_map)
      : node_(node),
        type_binding_map_() {
    if (use_lookup_map) {
      type_binding_map_ = onnxruntime::make_unique<TypeBindingMap>();
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

  // Resolves a name to a TypeProto* for a given node.
  // The name can represent either a type parameter or an input/output parameter.
  // Returns the resolved TypeProto* or nullptr if unable to resolve.
  const ONNX_NAMESPACE::TypeProto* Resolve(const std::string& name_or_type_str) const {
    // lookup if available
    if (type_binding_map_) {
      auto found_it = type_binding_map_->find(name_or_type_str);
      if (found_it == type_binding_map_->end()) return nullptr;
      return found_it->second;
    }

    // fall back to node parameter traversal
    const ONNX_NAMESPACE::TypeProto* result{};
    TraverseFormalParametersWithTypeProto(
        node_,
        [&name_or_type_str](const ONNX_NAMESPACE::OpSchema::FormalParameter& param) -> bool {
          return param.GetName() == name_or_type_str || param.GetTypeStr() == name_or_type_str;
        },
        [&result](const ONNX_NAMESPACE::OpSchema::FormalParameter&,
                  const ONNX_NAMESPACE::TypeProto* type) -> bool {
          result = type;
          return false;
        });
    return result;
  }

 private:
  // map from input/output name or type string to TypeProto pointer
  using TypeBindingMap = std::unordered_map<std::string, const ONNX_NAMESPACE::TypeProto*>;

  const Node& node_;
  std::unique_ptr<TypeBindingMap> type_binding_map_;
};
};  // namespace

bool KernelRegistry::VerifyKernelDef(const onnxruntime::Node& node,
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

Status KernelRegistry::TryCreateKernel(const onnxruntime::Node& node,
                                       const IExecutionProvider& execution_provider,
                                       const std::unordered_map<int, OrtValue>& constant_initialized_tensors,
                                       const OrtValueNameIdxMap& ort_value_name_idx_map,
                                       const FuncManager& funcs_mgr,
                                       const DataTransferManager& data_transfer_mgr,
                                       /*out*/ std::unique_ptr<OpKernel>& op_kernel) const {
  const KernelCreateInfo* kernel_create_info = nullptr;
  ORT_RETURN_IF_ERROR(TryFindKernel(node, execution_provider.Type(), &kernel_create_info));
  OpKernelInfo kernel_info(node,
                           *kernel_create_info->kernel_def,
                           execution_provider,
                           constant_initialized_tensors,
                           ort_value_name_idx_map,
                           funcs_mgr,
                           data_transfer_mgr);
  op_kernel.reset(kernel_create_info->kernel_create_func(kernel_info));
  return Status::OK();
}

static std::string ToString(const std::vector<std::string>& error_strs) {
  std::ostringstream ostr;
  std::for_each(std::begin(error_strs), std::end(error_strs),
                [&ostr](const std::string& str) { ostr << str << "\n"; });
  return ostr.str();
}

Status KernelRegistry::TryFindKernel(const onnxruntime::Node& node,
                                     onnxruntime::ProviderType exec_provider,
                                     const KernelCreateInfo** out) const {
  return TryFindKernel(node, exec_provider, uint64_t(0), out);
}
#endif  // !defined(ORT_MINIMAL_BUILD)

// It's often this function returns a failed status, but it is totally expected.
// It just means this registry doesn't have such a kernel, please search it elsewhere.
// if this function is called before graph partition, then node.provider is not set.
// In this case, the kernel's provider must equal to exec_provider
// otherwise, kernel_def.provider must equal to node.provider. exec_provider is ignored.
Status KernelRegistry::TryFindKernel(const onnxruntime::Node& node,
                                     onnxruntime::ProviderType exec_provider,
                                     uint64_t kernel_def_hash,
                                     const KernelCreateInfo** out) const {
  const auto& node_provider = node.GetExecutionProviderType();
  const auto& expected_provider = (node_provider.empty() ? exec_provider : node_provider);

  auto range = kernel_creator_fn_map_.equal_range(GetMapKey(node.OpType(), node.Domain(), expected_provider));
  *out = nullptr;

  // if we have a hash (ORT format model) use only that.
  if (kernel_def_hash != 0) {
    for (auto i = range.first; i != range.second; ++i) {
      if (i->second.kernel_def->GetHash() == kernel_def_hash) {
        *out = &i->second;
        return Status::OK();
      }
    }

    std::ostringstream oss;
    oss << "Op with name (" << node.Name() << ")"
        << " and type (" << node.OpType() << ")"
        << " kernel not found in " << expected_provider << "."
        << " No matching hash for " << kernel_def_hash;

    return Status(ONNXRUNTIME, FAIL, oss.str());
  }
#if !defined(ORT_MINIMAL_BUILD)
  else {
    std::vector<std::string> verify_kernel_def_error_strs;

    for (auto i = range.first; i != range.second; ++i) {
      std::string error_str;
      if (VerifyKernelDef(node, *i->second.kernel_def, error_str)) {
        *out = &i->second;
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

      return Status(ONNXRUNTIME, FAIL, oss.str());
    }
  }

  return Status(ONNXRUNTIME, FAIL, "Kernel not found");
#else
  ORT_THROW("Kernel hash must be provided in minimal build.");
#endif
}

Status KernelRegistry::Register(KernelDefBuilder& kernel_builder,
                                const KernelCreateFn& kernel_creator) {
  return Register(KernelCreateInfo(kernel_builder.Build(), kernel_creator));
}

Status KernelRegistry::Register(KernelCreateInfo&& create_info) {
  if (!create_info.kernel_def) {
    return Status(ONNXRUNTIME, FAIL, "kernel def can't be NULL");
  }
  std::string key = GetMapKey(*create_info.kernel_def);
  // Check op version conflicts.
  auto range = kernel_creator_fn_map_.equal_range(key);
  for (auto i = range.first; i != range.second; ++i) {
    if (i->second.kernel_def &&
        i->second.kernel_def->IsConflict(*create_info.kernel_def)) {
      return Status(ONNXRUNTIME, FAIL,
                    "Failed to add kernel for " + key +
                        ": Conflicting with a registered kernel with op versions.");
    }
  }

  // Register the kernel.
  // Ownership of the KernelDef is transferred to the map.
  kernel_creator_fn_map_.emplace(key, std::move(create_info));
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
