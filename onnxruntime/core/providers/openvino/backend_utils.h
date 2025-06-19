// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#define ORT_API_MANUAL_INIT
#include <iomanip>
#include <unordered_map>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <string_view>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/ov_interface.h"
#ifdef _WIN32
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#include <sys/stat.h>

namespace onnxruntime {
namespace openvino_ep {
constexpr std::string log_tag = "[OpenVINO-EP] ";

struct ParameterShape {
  using ort_shape_t = std::vector<int64_t>;

  static ov::PartialShape ToOvPartialShape(const ort_shape_t& ort_shape) {
    std::vector<ov::Dimension> ov_shape(ort_shape.size());
    std::transform(ort_shape.begin(), ort_shape.end(), ov_shape.begin(), [](int64_t dim) {
      return dim == -1 ? ov::Dimension::dynamic() : ov::Dimension(dim);
    });
    return ov::PartialShape(ov_shape);
  }

  static ort_shape_t ToOrtShape(const ov::PartialShape& ov_shape) {
    ort_shape_t ort_shape(ov_shape.size());
    std::transform(ov_shape.begin(), ov_shape.end(), ort_shape.begin(), [](const auto& dim) {
      return dim.is_dynamic() ? -1 : dim.get_length();
    });
    return ort_shape;
  }

  static ort_shape_t ToOrtShape(const ov::Shape& ov_shape) {
    ort_shape_t ort_shape(ov_shape.size());
    std::transform(ov_shape.begin(), ov_shape.end(), ort_shape.begin(), [](const auto& dim) {
      return narrow<int64_t>(dim);
    });
    return ort_shape;
  }

  operator ov::Shape() const { return ov_.get_shape(); }
  operator const ov::PartialShape&() const { return ov_; }
  operator const ort_shape_t&() const { return ort_; }

  explicit ParameterShape(const ort_shape_t& ort_shape) : ort_(ort_shape), ov_(ToOvPartialShape(ort_shape)) {}
  explicit ParameterShape(const ov::PartialShape& ov_partial_shape) : ov_(ov_partial_shape), ort_(ToOrtShape(ov_partial_shape)) {}

 private:
  ort_shape_t ort_;
  ov::PartialShape ov_;
};

namespace backend_utils {

bool IsDebugEnabled();

// Internal diagnostic function.
bool IsCILogEnabled();

int GetFirstAvailableDevice(SessionContext& session_context);

void FillOutputsWithConstantData(std::shared_ptr<ov::Node> node, Ort::UnownedValue& out_tensor);

template <typename T>
void FillOutputHelper(Ort::UnownedValue& out_tensor, std::shared_ptr<ov::Node> node);

Ort::UnownedValue
GetOutputTensor(Ort::KernelContext& context,
                std::string output_name,
                const SubGraphContext::string_index_map_t& output_names,
                std::shared_ptr<ov::Node> node);

void FillInputBlob(OVTensorPtr inputBlob, size_t batch_slice_idx,
                   std::string input_name, Ort::KernelContext& context,
                   const SubGraphContext& subgraph_context);

std::shared_ptr<const OVNetwork>
CreateOVModel(std::string&& model,
              const SessionContext& session_context,
              std::map<std::string, std::shared_ptr<ov::Node>>& const_outputs_map);

void CreateOVTensors(const std::string& device_name,
                     SharedContext::SharedWeights::Metadata::Map& metadata_map,
                     SharedContext::SharedWeights::WeightsFile& weights);
void DestroyOVTensors(SharedContext::SharedWeights::Metadata::Map& metadata_map);

void printPerformanceCounts(const std::vector<OVProfilingInfo>& performanceMap,
                            std::ostream& stream, std::string deviceName);

void printPerformanceCounts(OVInferRequestPtr request, std::ostream& stream, std::string deviceName);

}  // namespace backend_utils
}  // namespace openvino_ep
}  // namespace onnxruntime
