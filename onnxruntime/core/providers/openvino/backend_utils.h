// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#pragma once

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "contexts.h"
#include <iomanip>
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

//Internal diagnostic function. 
bool IsCILogEnabled();

bool UseCompiledNetwork();

std::string GetCurrentWorkingDir();

bool IsDirExists(const std::string& pathname);

void CreateDirectory(const std::string& ov_compiled_blobs_dir);

void SetIODefs(const ONNX_NAMESPACE::ModelProto& model_proto,
               std::shared_ptr<InferenceEngine::CNNNetwork> network,
               std::unordered_map<std::string, int> output_names,
               std::map<std::string, std::shared_ptr<ngraph::Node>>& const_outputs_map,
               std::string device);

std::shared_ptr<InferenceEngine::CNNNetwork>
CreateCNNNetwork(const ONNX_NAMESPACE::ModelProto& model_proto, const GlobalContext& global_context, const SubGraphContext& subgraph_context, std::map<std::string, std::shared_ptr<ngraph::Node>>& const_outputs_map);

int GetFirstAvailableDevice(GlobalContext& global_context);

void FillOutputsWithConstantData(Ort::CustomOpApi& ort, std::shared_ptr<ngraph::Node> node, OrtValue* out_tensor);

template <typename T>
void FillOutputHelper(Ort::CustomOpApi& ort, OrtValue* out_tensor, std::shared_ptr<ngraph::Node> node);

OrtValue*
GetOutputTensor(Ort::CustomOpApi& ort, OrtKernelContext* context,
                std::string output_name,
                std::unordered_map<std::string, int> output_names,
                std::shared_ptr<ngraph::Node> node);

InferenceEngine::Precision
ConvertPrecisionONNXToOpenVINO(const ONNX_NAMESPACE::TypeProto& onnx_type, std::string device);

OrtValue*
GetOutputTensor(Ort::CustomOpApi& ort, OrtKernelContext* context, size_t batch_size,
                OVInferRequestPtr infer_request,
                std::string output_name,
                std::unordered_map<std::string, int> output_names);

#if defined (OV_API_20)
void FillInputBlob(OVTensorPtr inputBlob, size_t batch_slice_idx,
                   std::string input_name, Ort::CustomOpApi& ort, OrtKernelContext* context,
                   const SubGraphContext& subgraph_context);

void FillOutputBlob(OVTensorPtr outputBlob, OrtValue* output_tensor,
                    Ort::CustomOpApi& ort, size_t batch_slice_idx);

std::shared_ptr<OVNetwork>
CreateOVModel(const ONNX_NAMESPACE::ModelProto& model_proto, const GlobalContext& global_context, const SubGraphContext& subgraph_context, std::map<std::string, std::shared_ptr<ngraph::Node>>& const_outputs_map);

void printPerformanceCounts(const std::vector<OVProfilingInfo>& performanceMap,
                            std::ostream& stream, std::string deviceName);
#endif

void printPerformanceCounts(const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& performanceMap,
                            std::ostream& stream, std::string deviceName);

std::vector<std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>>
perfCountersSorted(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap);

void FillInputBlob(InferenceEngine::Blob::Ptr& inputBlob, size_t batch_slice_idx,
                   std::string input_name, Ort::CustomOpApi& ort, OrtKernelContext* context,
                   InferenceEngine::Precision precision, const SubGraphContext& subgraph_context);

void FillOutputBlob(InferenceEngine::Blob::Ptr& outputBlob, OrtValue* output_tensor,
                    Ort::CustomOpApi& ort, InferenceEngine::Precision precision, size_t batch_slice_idx);

void printPerformanceCounts(OVInferRequestPtr request, std::ostream& stream, std::string deviceName);

}  // namespace backend_utils
}  // namespace openvino_ep
}  // namespace onnxruntime
