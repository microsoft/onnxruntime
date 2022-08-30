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
#include "core/graph/op_identifier.h"
#include "core/graph/graph.h"
#include "core/platform/ort_mutex.h"

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
using KernelTypeStrToArgsMap = InlinedHashMap<std::string, InlinedVector<ArgTypeAndIndex>>;
using OpKernelTypeStrMap = InlinedHashMap<OpIdentifier, KernelTypeStrToArgsMap>;

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

  Status RegisterGraphNodeOpSchemas(const Graph& graph);

  Status SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                         flatbuffers::Offset<fbs::KernelTypeStrResolver>& fbs_kernel_type_str_resolver) const;
#endif  // !defined(ORT_MINIMAL_BUILD)

  Status LoadFromOrtFormat(const fbs::KernelTypeStrResolver& fbs_kernel_type_str_resolver);

  void Merge(KernelTypeStrResolver& src);

  const OpKernelTypeStrMap& GetOpKernelTypeStrMap() const { return op_kernel_type_str_map_; }

 private:
  OpKernelTypeStrMap op_kernel_type_str_map_;
};

#if !defined(ORT_MINIMAL_BUILD)
class AutoRegisteringKernelTypeStrResolver : public IKernelTypeStrResolver {
 public:
  Status ResolveKernelTypeStr(const Node& node, std::string_view kernel_type_str,
                              gsl::span<const ArgTypeAndIndex>& resolved_args) const override;

 private:
  // used as a cache when resolving
  // since the cache may be modified with a const instance, ensure that access to the cache is thread-safe
  mutable KernelTypeStrResolver resolver_;
  mutable OrtMutex resolver_mutex_;
};
#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace onnxruntime
