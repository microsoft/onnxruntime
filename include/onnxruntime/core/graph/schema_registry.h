// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/graph/constants.h"
#include "core/common/common.h"
#include "core/common/status.h"
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "onnx/defs/schema.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include <mutex>
#include <deque>
#include "sstream"

namespace onnxruntime {
using OpName_Domain_Version_Schema_Map = std::unordered_map<
    std::string,
    std::unordered_map<std::string, std::map<ONNX_NAMESPACE::OperatorSetVersion, ONNX_NAMESPACE::OpSchema>>>;

// onnxruntime schema registry is a supplement to built-in schema,
// Every schema registry represent a collection of schema deltas from baseline_opset_version to opset_version
struct SchemaRegistryVersion {
  int baseline_opset_version;
  int opset_version;
};

using Domain_To_Version_Map = std::unordered_map<std::string, int>;
using Domain_To_Version_Range_Map = std::unordered_map<std::string, SchemaRegistryVersion>;

class IOnnxRuntimeOpSchemaCollection : public ONNX_NAMESPACE::ISchemaRegistry {
 public:
  virtual Domain_To_Version_Map GetLatestOpsetVersions(bool is_onnx_only) const = 0;

  using ISchemaRegistry::GetSchema;

  virtual const ONNX_NAMESPACE::OpSchema* GetSchema(
      const std::string& key,
      const int maxInclusiveVersion,
      const std::string& domain) const final {
    const ONNX_NAMESPACE::OpSchema* latest_schema = nullptr;
    int earliest_opset_where_unchanged = std::numeric_limits<int>::max();
    GetSchemaAndHistory(key, maxInclusiveVersion, domain, &latest_schema, &earliest_opset_where_unchanged);

    assert(latest_schema == nullptr || (latest_schema->SinceVersion() <= maxInclusiveVersion &&
                                        earliest_opset_where_unchanged == latest_schema->SinceVersion()));

    return latest_schema;
  }

  virtual void GetSchemaAndHistory(
      const std::string& key,
      int maxInclusiveVersion,
      const std::string& domain,
      const ONNX_NAMESPACE::OpSchema** latest_schema,
      int* earliest_opset_where_unchanged) const = 0;
};

// OnnxRuntimeOpSchemaRegistry is used to provide supplement for built-in ONNX schemas.
// Each OnnxRuntimeOpSchemaRegistry must register complete opsets delta from a baseline version to max opset version.
// (Please notice that baseline opsets are not include in the delta)
// For example, ONNXRuntime is build with ONNX 1.2 which is at opset7, to use onnx opset8 and opset9,
// user could create a OnnxRuntimeOpSchemaRegistry and config it as {baseline_opset_version = 7, opset_version = 9}
// it means this OnnxRuntimeOpSchemaRegistry contains the complete delta from opset7 to opset9.
class OnnxRuntimeOpSchemaRegistry : public IOnnxRuntimeOpSchemaCollection {
 public:
  OnnxRuntimeOpSchemaRegistry() = default;

  ::onnxruntime::common::Status SetBaselineAndOpsetVersionForDomain(
      const std::string& domain,
      int baseline_opset_version,
      int opset_version);

  Domain_To_Version_Map GetLatestOpsetVersions(bool is_onnx_only) const override;

  // OnnxRuntimeOpSchemaRegistry must register complete delta for a opset.
  ::onnxruntime::common::Status RegisterOpSet(
      std::vector<ONNX_NAMESPACE::OpSchema>& schemas,
      const std::string& domain,
      int baseline_opset_version,
      int opset_version);

// conversion of kOnnxDomain to std::string creates unnamed temporary.  Suppress C26444 (es.84) the hard way.
// GSL_SUPPRESS(es.84) doesn't work as the default arg temporary isn't in a scope the suppress attribute handles.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26444)
#endif

  using IOnnxRuntimeOpSchemaCollection::GetSchema;

  void GetSchemaAndHistory(
      const std::string& key,
      const int maxInclusiveVersion,
      const std::string& domain,
      const ONNX_NAMESPACE::OpSchema** latest_schema,
      int* earliest_opset_where_unchanged) const override;

#ifdef _MSC_VER
#pragma warning(pop)  // C26444
#endif

  bool empty() const {
    return map_.empty();
  }

 private:
  ::onnxruntime::common::Status RegisterOpSchema(ONNX_NAMESPACE::OpSchema&& op_schema);

  ::onnxruntime::common::Status RegisterOpSchemaInternal(ONNX_NAMESPACE::OpSchema&& op_schema);

  std::mutex mutex_;

  OpName_Domain_Version_Schema_Map map_;
  Domain_To_Version_Range_Map domain_version_range_map_;
};

// SchemaRegistryManager provides a view based on built-in ONNX schema and a list of OnnxRuntimeOpSchemaRegistry as supplement.
// User need to make sure the customized schema registry is valid, otherwise the behavior is undefined.
// We may add more consistent check later.
class SchemaRegistryManager : public onnxruntime::IOnnxRuntimeOpSchemaCollection {
 public:
  // The schema registry priority is the reverse of register order.
  void RegisterRegistry(std::shared_ptr<IOnnxRuntimeOpSchemaCollection> registry);

  Domain_To_Version_Map GetLatestOpsetVersions(bool is_onnx_only) const override;

  void GetSchemaAndHistory(
      const std::string& key,
      const int maxInclusiveVersion,
      const std::string& domain,
      const ONNX_NAMESPACE::OpSchema** latest_schema,
      int* earliest_opset_where_unchanged) const override;

 private:
  std::deque<std::shared_ptr<IOnnxRuntimeOpSchemaCollection>> registries;
};

}  // namespace onnxruntime
