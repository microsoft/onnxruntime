// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <inference_engine.hpp>

#include "core/session/onnxruntime_cxx_api.h"
#include "contexts.h"

namespace onnxruntime {
namespace openvino_ep {
namespace backend_utils {
const std::string log_tag = "[OpenVINO-EP] ";

#ifndef NDEBUG
bool IsDebugEnabled();
#endif

void SetIODefs(const ONNX_NAMESPACE::ModelProto& model_proto,
               std::shared_ptr<InferenceEngine::CNNNetwork> network,
               std::unordered_map<std::string, int> output_names,
               std::map<std::string, std::shared_ptr<ngraph::Node>>& const_outputs_map,
               std::string device);

std::shared_ptr<InferenceEngine::CNNNetwork>
CreateCNNNetwork(const ONNX_NAMESPACE::ModelProto& model_proto, const GlobalContext& global_context, const SubGraphContext& subgraph_context, std::map<std::string,
                   std::shared_ptr<ngraph::Node>>& const_outputs_map);

int GetFirstAvailableDevice(GlobalContext& global_context);

#if defined(OPENVINO_2020_4) || defined (OPENVINO_2021_1)
void FillOutputsWithConstantData(Ort::CustomOpApi& ort, std::shared_ptr<ngraph::Node> node, OrtValue* out_tensor);

template <typename T>
void FillOutputHelper(Ort::CustomOpApi& ort, OrtValue* out_tensor, std::shared_ptr<ngraph::Node> node);

OrtValue*
GetOutputTensor(Ort::CustomOpApi& ort, OrtKernelContext* context,
                 std::string output_name,
                 std::unordered_map<std::string, int> output_names,
                 std::shared_ptr<ngraph::Node> node);
#endif

InferenceEngine::Precision
ConvertPrecisionONNXToOpenVINO(const ONNX_NAMESPACE::TypeProto& onnx_type, std::string device);

OrtValue*
GetOutputTensor(Ort::CustomOpApi& ort, OrtKernelContext* context, size_t batch_size,
                 InferenceEngine::InferRequest::Ptr infer_request,
                 std::string output_name,
                 std::unordered_map<std::string, int> output_names);


void FillInputBlob(InferenceEngine::Blob::Ptr& inputBlob, size_t request_id, size_t batch_slice_idx,
                   std::string input_name, Ort::CustomOpApi& ort, OrtKernelContext* context,
                   InferenceEngine::Precision precision, const SubGraphContext& subgraph_context);

void FillOutputBlob(InferenceEngine::Blob::Ptr& outputBlob, OrtValue* output_tensor,
                    Ort::CustomOpApi& ort, InferenceEngine::Precision precision, size_t batch_slice_idx);

}  // namespace backend_utils
}  // namespace openvino_ep
}  // namespace onnxruntime