#pragma once

#include <cstddef>
#include <memory>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"

namespace onnxruntime {

class Graph;
class Model;

#if !defined(ORT_MINIMAL_BUILD)

struct FunctionExtractorOptions {
  size_t max_pattern_nodes{1024};
  size_t max_target_nodes{1'000'000};
  size_t max_output_root_tuples{100'000};
  size_t max_worklist_bindings{1'000'000};
  size_t max_literal_bytes{64U * 1024U * 1024U};
};

struct FunctionExtractionResult {
  common::Status status{common::Status::OK()};
  size_t replacements_applied{0};
};

class FunctionExtractor final {
 public:
  explicit FunctionExtractor(
      const ONNX_NAMESPACE::FunctionProto& function_proto,
      FunctionExtractorOptions options = {});
  ~FunctionExtractor();

  FunctionExtractionResult Extract(Model& model);
  FunctionExtractionResult Extract(Graph& graph);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FunctionExtractor);
};

#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace onnxruntime
