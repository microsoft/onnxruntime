// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <string_view>
#include <utility>

#include "gsl/gsl"

#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/schema.h"
#endif  // !defined(ORT_MINIMAL_BUILD)

#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"
#include "core/graph/graph.h"

namespace flatbuffers {
class FlatBufferBuilder;
template <typename T>
struct Offset;
}  // namespace flatbuffers

namespace onnxruntime {

namespace fbs {
struct KernelTypeStrResolver;
}  // namespace fbs

using common::Status;

using ArgTypeAndIndex = std::pair<ArgType, size_t>;

class IKernelTypeStrResolver {
public:
 /**
  * Resolves an op's kernel type string to its associated arguments.
  * @param node The op's node.
  * @param kernel_type_str The op kernel type string.
  * @param[out] resolved_args The op arguments associated with kernel_type_str.
  */
 virtual Status ResolveKernelTypeStr(const Node& node, std::string_view kernel_type_str,
                                     gsl::span<const ArgTypeAndIndex>& resolved_args) const = 0;
};

class KernelTypeStrResolver : public IKernelTypeStrResolver {
 public:
  Status ResolveKernelTypeStr(const Node& node, std::string_view kernel_type_str,
                              gsl::span<const ArgTypeAndIndex>& resolved_args) const override;

#if !defined(ORT_MINIMAL_BUILD)
  Status RegisterOpSchema(const ONNX_NAMESPACE::OpSchema& op_schema, bool* registered = nullptr);

  Status RegisterNodeOpSchema(const Node& node);

  static KernelTypeStrResolver CreateFromNodeOpSchema(const Node& node) {
    KernelTypeStrResolver k{};
    ORT_THROW_IF_ERROR(k.RegisterNodeOpSchema(node));
    return k;
  }

  Status RegisterGraphNodeOpSchemas(const Graph& graph);

  static KernelTypeStrResolver CreateFromGraphNodeOpSchemas(const Graph& graph) {
    KernelTypeStrResolver k{};
    ORT_THROW_IF_ERROR(k.RegisterGraphNodeOpSchemas(graph));
    return k;
  }

  Status SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                         flatbuffers::Offset<fbs::KernelTypeStrResolver>& fbs_kernel_type_str_resolver) const;
#endif  // !defined(ORT_MINIMAL_BUILD)

  Status LoadFromOrtFormat(const fbs::KernelTypeStrResolver& fbs_kernel_type_str_resolver);

 private:
  using KernelTypeStrToArgsMap = InlinedHashMap<std::string, InlinedVector<ArgTypeAndIndex>>;
  using OpKernelTypeStrMap = InlinedHashMap<OpIdentifier, KernelTypeStrToArgsMap>;

  OpKernelTypeStrMap op_kernel_type_str_map_;
};

#if !defined(ORT_MINIMAL_BUILD)
class AutoRegisteringKernelTypeStrResolver : public IKernelTypeStrResolver {
 public:
  Status ResolveKernelTypeStr(const Node& node, std::string_view kernel_type_str,
                              gsl::span<const ArgTypeAndIndex>& resolved_args) const override;

 private:
  // used as a cache when resolving
  // TODO thread-safety? it may change even through const functions
  mutable KernelTypeStrResolver resolver_;
};
#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace onnxruntime
