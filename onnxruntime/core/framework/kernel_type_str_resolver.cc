// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_type_str_resolver.h"

#include <mutex>  // for std::lock_guard

#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/graph/op_identifier_utils.h"

namespace fb = flatbuffers;

namespace onnxruntime {

static OpKernelTypeStrMap::const_iterator LookUpOpId(const OpIdentifier& op_id,
                                                     const OpKernelTypeStrMap& map) {
  auto op_it = map.find(op_id);

  if (op_it == map.end() && op_id.domain == kMSInternalNHWCDomain) {
    // Special case for kMSInternalNHWCDomain.
    // kMSInternalNHWCDomain is set (replacing the original domain) by ORT during layout transformation.
    // However, ORT format models contain kernel type string information with the original domain.
    // kMSInternalNHWCDomain maps to one of these alternate domains, so fall back to them.
    constexpr std::array alternate_domains{
        std::string_view{kOnnxDomain},
        std::string_view{kMSDomain},
    };

    for (auto alternate_domain : alternate_domains) {
      const auto alternate_op_id = OpIdentifier{std::string{alternate_domain}, op_id.op_type, op_id.since_version};
      op_it = map.find(alternate_op_id);
      if (op_it != map.end()) {
        break;
      }
    }
  }

  return op_it;
}

Status KernelTypeStrResolver::ResolveKernelTypeStr(const Node& node, std::string_view kernel_type_str,
                                                   gsl::span<const ArgTypeAndIndex>& resolved_args) const {
  const auto op_id = utils::MakeOpId(node);
  const auto op_it = LookUpOpId(op_id, op_kernel_type_str_map_);
  ORT_RETURN_IF(op_it == op_kernel_type_str_map_.end(), "Failed to find op_id: ", op_id);
  const auto& type_str_map = op_it->second;

  // TODO(edgchen1) maybe we can use transparent hash/eq to enable lookup with string_view
  const auto type_str_it = type_str_map.find(std::string(kernel_type_str));

  ORT_RETURN_IF(type_str_it == type_str_map.end(),
                "Failed to find args for kernel type string '", kernel_type_str,
                "' for node type '", node.OpType(), "'. ",
                "If type constraint names are available, ",
                "ensure that they are used in the kernel def type "
                "constraints instead of op input or output names. Not doing so will result in this error.");
  resolved_args = type_str_it->second;
  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD)
Status KernelTypeStrResolver::RegisterOpSchema(const ONNX_NAMESPACE::OpSchema& op_schema, bool* registered_out) {
  auto op_id = utils::MakeOpId(op_schema);
  if (Contains(op_kernel_type_str_map_, op_id)) {
    if (registered_out) {
      *registered_out = false;
    }
    return Status::OK();
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
  // at most one entry for each input/output
  kernel_type_str_map.reserve(op_schema.inputs().size() + op_schema.outputs().size());

  auto process_formal_params = [&](ArgType arg_type) -> Status {
    const auto& formal_params = arg_type == ArgType::kInput ? op_schema.inputs() : op_schema.outputs();
    for (size_t i = 0; i < formal_params.size(); ++i) {
      const auto& formal_param = formal_params[i];
      auto curr_arg_type_and_idx = ArgTypeAndIndex{arg_type, i};

      // first, try to use type constraint name as kernel type string
      if (const auto& type_str = formal_param.GetTypeStr();
          Contains(type_constraint_names, type_str)) {
        kernel_type_str_map[type_str].push_back(curr_arg_type_and_idx);
        continue;
      }

      // otherwise, use input/output name as kernel type string
      auto& args_for_io_name = kernel_type_str_map[formal_param.GetName()];
      if (!args_for_io_name.empty()) {
        // It's possible that an input and output have the same name (e.g, BatchNormalization-9 has both an input and
        // an output named 'mean').
        // If so, their formal parameters also need to have the same type string. Otherwise, it would be ambiguous to
        // use that name as a kernel type string.
        auto formal_param_type_str = [&op_schema](const ArgTypeAndIndex& arg_type_and_idx) {
          const auto& [arg_type, idx] = arg_type_and_idx;
          const auto& formal_params = arg_type == ArgType::kInput ? op_schema.inputs() : op_schema.outputs();
          return formal_params[idx].GetTypeStr();
        };

        ORT_RETURN_IF_NOT(
            formal_param_type_str(curr_arg_type_and_idx) == formal_param_type_str(args_for_io_name.front()),
            "Kernel type string already exists for formal parameter name '", formal_param.GetName(),
            "', but the existing argument with that formal parameter name has a different formal parameter "
            "type string.");
      }
      args_for_io_name.push_back(std::move(curr_arg_type_and_idx));
    }
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(process_formal_params(ArgType::kInput));
  ORT_RETURN_IF_ERROR(process_formal_params(ArgType::kOutput));

  op_kernel_type_str_map_.emplace(std::move(op_id), std::move(kernel_type_str_map));
  if (registered_out) {
    *registered_out = true;
  }
  return Status::OK();
}

Status KernelTypeStrResolver::RegisterNodeOpSchema(const Node& node) {
  ORT_RETURN_IF(node.Op() == nullptr, "Op schema must be available.");
  return RegisterOpSchema(*node.Op());
}

Status KernelTypeStrResolver::RegisterGraphNodeOpSchemas(const Graph& graph) {
  for (const Node& node : graph.Nodes()) {
    ORT_RETURN_IF_ERROR(RegisterNodeOpSchema(node));

    if (node.ContainsSubgraph()) {
      const auto subgraphs = node.GetSubgraphs();
      for (const auto& subgraph : subgraphs) {
        ORT_RETURN_IF_ERROR(RegisterGraphNodeOpSchemas(*subgraph));
      }
    }
  }
  return Status::OK();
}

Status KernelTypeStrResolver::SaveToOrtFormat(
    fb::FlatBufferBuilder& builder,
    fb::Offset<fbs::KernelTypeStrResolver>& fbs_kernel_type_str_resolver) const {
  std::vector<fb::Offset<fbs::OpIdKernelTypeStrArgsEntry>> fbs_op_kernel_type_str_args{};
  fbs_op_kernel_type_str_args.reserve(op_kernel_type_str_map_.size());

  for (const auto& [op_id, kernel_type_str_map] : op_kernel_type_str_map_) {
    std::vector<fb::Offset<fbs::KernelTypeStrArgsEntry>> fbs_kernel_type_str_args{};
    fbs_kernel_type_str_args.reserve(kernel_type_str_map.size());

    for (const auto& [kernel_type_str, args] : kernel_type_str_map) {
      std::vector<fb::Offset<fbs::ArgTypeAndIndex>> fbs_args{};
      fbs_args.reserve(args.size());

      for (const auto& arg : args) {
        auto fbs_arg = fbs::CreateArgTypeAndIndex(
            builder,
            arg.first == ArgType::kInput ? fbs::ArgType::INPUT : fbs::ArgType::OUTPUT,
            gsl::narrow<uint32_t>(arg.second));
        fbs_args.push_back(fbs_arg);
      }

      auto fbs_kernel_type_str_args_entry = fbs::CreateKernelTypeStrArgsEntry(
          builder,
          builder.CreateSharedString(kernel_type_str),
          builder.CreateVector(fbs_args));
      fbs_kernel_type_str_args.push_back(fbs_kernel_type_str_args_entry);
    }

    fb::Offset<flatbuffers::String> fbs_op_id{};
    ORT_RETURN_IF_ERROR(fbs::utils::SaveOpIdentifierOrtFormat(builder, op_id, fbs_op_id));

    auto fbs_op_kernel_type_str_args_entry = fbs::CreateOpIdKernelTypeStrArgsEntry(
        builder,
        fbs_op_id,
        builder.CreateVectorOfSortedTables(&fbs_kernel_type_str_args));
    fbs_op_kernel_type_str_args.push_back(fbs_op_kernel_type_str_args_entry);
  }

  fbs_kernel_type_str_resolver = fbs::CreateKernelTypeStrResolver(
      builder,
      builder.CreateVectorOfSortedTables(&fbs_op_kernel_type_str_args));
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

// returns an error message string which is empty if successful
static std::string LoadFromOrtFormatImpl(const fbs::KernelTypeStrResolver& fbs_kernel_type_str_resolver,
                                         OpKernelTypeStrMap& op_kernel_type_str_map_out) {
  const auto* fbs_op_kernel_type_str_args = fbs_kernel_type_str_resolver.op_kernel_type_str_args();
  if (!fbs_op_kernel_type_str_args) {
    return "op_kernel_type_str_args is null.";
  }

  OpKernelTypeStrMap op_kernel_type_str_map{};
  op_kernel_type_str_map.reserve(fbs_op_kernel_type_str_args->size());
  for (const auto* fbs_op_kernel_type_str_args_entry : *fbs_op_kernel_type_str_args) {
    if (!fbs_op_kernel_type_str_args_entry) {
      return "op_kernel_type_str_args entry is null.";
    }

    const auto* fbs_op_id = fbs_op_kernel_type_str_args_entry->op_id();
    if (!fbs_op_id) {
      return "op_id is null.";
    }

    const auto* fbs_kernel_type_str_args = fbs_op_kernel_type_str_args_entry->kernel_type_str_args();
    if (!fbs_kernel_type_str_args) {
      return "kernel_type_str_args is null.";
    }

    KernelTypeStrToArgsMap kernel_type_str_map{};
    kernel_type_str_map.reserve(fbs_kernel_type_str_args->size());
    for (const auto* fbs_kernel_type_str_args_entry : *fbs_kernel_type_str_args) {
      if (!fbs_kernel_type_str_args_entry) {
        return "kernel_type_str_args entry is null.";
      }

      const auto* fbs_kernel_type_str = fbs_kernel_type_str_args_entry->kernel_type_str();
      if (!fbs_kernel_type_str) {
        return "kernel_type_str is null.";
      }

      const auto* fbs_args = fbs_kernel_type_str_args_entry->args();
      if (!fbs_args) {
        return "args is null.";
      }

      InlinedVector<ArgTypeAndIndex> args{};
      args.reserve(fbs_args->size());
      for (const auto* fbs_arg : *fbs_args) {
        if (!fbs_arg) {
          return "args entry is null.";
        }
        args.push_back(ArgTypeAndIndex{
            fbs_arg->arg_type() == fbs::ArgType::INPUT ? ArgType::kInput : ArgType::kOutput,
            fbs_arg->index()});
      }

      const auto [it, inserted] = kernel_type_str_map.try_emplace(fbs_kernel_type_str->str(), std::move(args));
      if (!inserted) {
        return MakeString("Duplicate entry for kernel type str: ", it->first, ".");
      }
    }

    OpIdentifier op_id;
    const auto load_op_id_status = fbs::utils::LoadOpIdentifierOrtFormat(*fbs_op_id, op_id);
    if (!load_op_id_status.IsOK()) {
      return load_op_id_status.ErrorMessage();
    }

    const auto [it, inserted] = op_kernel_type_str_map.try_emplace(std::move(op_id),
                                                                   std::move(kernel_type_str_map));
    if (!inserted) {
      return MakeString("Duplicate entry for op id: ", it->first, ".");
    }
  }

  op_kernel_type_str_map_out = std::move(op_kernel_type_str_map);
  return "";
}

Status KernelTypeStrResolver::LoadFromOrtFormat(const fbs::KernelTypeStrResolver& fbs_kernel_type_str_resolver) {
  const auto error_message = LoadFromOrtFormatImpl(fbs_kernel_type_str_resolver, op_kernel_type_str_map_);
  return error_message.empty() ? Status::OK()
                               : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, error_message, " ",
                                                 fbs::utils::kInvalidOrtFormatModelMessage);
}

void KernelTypeStrResolver::Merge(KernelTypeStrResolver src) {
  op_kernel_type_str_map_.merge(src.op_kernel_type_str_map_);
}

#if !defined(ORT_MINIMAL_BUILD)
Status OpSchemaKernelTypeStrResolver::ResolveKernelTypeStr(
    const Node& node, std::string_view kernel_type_str,
    gsl::span<const ArgTypeAndIndex>& resolved_args) const {
  std::lock_guard lock{resolver_mutex_};
  ORT_RETURN_IF_ERROR(resolver_.RegisterNodeOpSchema(node));
  ORT_RETURN_IF_ERROR(resolver_.ResolveKernelTypeStr(node, kernel_type_str, resolved_args));
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace onnxruntime
