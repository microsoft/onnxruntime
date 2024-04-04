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

#include "core/session/onnxruntime_cxx_api.h"
#include "contexts.h"
#include "ov_interface.h"
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
namespace backend_utils {
const std::string log_tag = "[OpenVINO-EP] ";

#ifndef NDEBUG
bool IsDebugEnabled();
#endif

// Internal diagnostic function.
bool IsCILogEnabled();

int GetFirstAvailableDevice(GlobalContext& global_context);

void FillOutputsWithConstantData(std::shared_ptr<ov::Node> node, Ort::UnownedValue& out_tensor);

template <typename T>
void FillOutputHelper(Ort::UnownedValue& out_tensor, std::shared_ptr<ov::Node> node);

Ort::UnownedValue
GetOutputTensor(Ort::KernelContext& context,
                std::string output_name,
                std::unordered_map<std::string, int> output_names,
                std::shared_ptr<ov::Node> node);

Ort::UnownedValue
GetOutputTensor(Ort::KernelContext& context, size_t batch_size,
                OVInferRequestPtr infer_request,
                std::string output_name,
                std::unordered_map<std::string, int> output_names);

void FillInputBlob(OVTensorPtr inputBlob, size_t batch_slice_idx,
                   std::string input_name, Ort::KernelContext& context,
                   const SubGraphContext& subgraph_context);

void FillOutputBlob(OVTensorPtr outputBlob, Ort::UnownedValue& output_tensor,
                    size_t batch_slice_idx);

std::shared_ptr<OVNetwork>
CreateOVModel(const ONNX_NAMESPACE::ModelProto& model_proto,
              const GlobalContext& global_context,
              std::map<std::string, std::shared_ptr<ov::Node>>& const_outputs_map);

void printPerformanceCounts(const std::vector<OVProfilingInfo>& performanceMap,
                            std::ostream& stream, std::string deviceName);

void printPerformanceCounts(OVInferRequestPtr request, std::ostream& stream, std::string deviceName);

}  // namespace backend_utils
}  // namespace openvino_ep
}  // namespace onnxruntime
