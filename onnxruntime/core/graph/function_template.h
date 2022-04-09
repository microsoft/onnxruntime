#pragma once
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "onnx/onnx_pb.h"
#include "core/common/logging/logging.h"
#include "core/graph/function.h"

namespace onnxruntime {

class FunctionTemplate {
public:
  FunctionTemplate(const std::string& function_domain, 
      const std::string& function_name,
      const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions,
      const std::unordered_map<std::string, int>& domain_version_map);

  const ONNX_NAMESPACE::OpSchema& OpSchema() const { return *op_schema_; }

  const std::string& Name() const;

  const std::string& Domain() const;

  std::unique_ptr<Function> Instantiate(const onnxruntime::Graph& graph,
                                        const onnxruntime::NodeIndex& node_index,
                                        const logging::Logger& logger) const;

  static std::unique_ptr<Function> Instantiate(const onnxruntime::Graph& graph,
                                               const onnxruntime::NodeIndex& node_index,
                                               const ONNX_NAMESPACE::FunctionProto& onnx_func_proto,
                                               const logging::Logger& logger);

private:
  // The generated schema for the function proto
  // ORT rely on it to run type/shape inference
  std::unique_ptr<ONNX_NAMESPACE::OpSchema> op_schema_;
  // reference to the function proto in local function
  const ONNX_NAMESPACE::FunctionProto* onnx_func_proto_;
};

}  // namespace onnxruntime
