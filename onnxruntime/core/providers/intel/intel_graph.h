// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once 
#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <ngraph/ngraph.hpp>
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

#include "core/session/onnxruntime_c_api.h"
#include "core/framework/func_api.h"
#include "core/graph/onnx_protobuf.h"

#include <map>
#include <memory>

#include <inference_engine.hpp>
#include <ie_builders.hpp>
#include <cpp/ie_infer_request.hpp>
//#include "intel_custom_op.h"

#include "core/framework/func_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace intel_ep {

class IntelGraph {
 public:

  IntelGraph(const onnxruntime::Node* fused_node);

  void Infer(const ONNX_NAMESPACE::ModelProto& model_proto, Ort::CustomOpApi ort, OrtKernelContext* context);

  static void ConvertONNXModelToIntelIR(const std::string& onnx_model, std::string& intel_xml, std::string& intel_bin, bool precision_fp32);

  static const std::string log_tag;
  void CreateNGraphFunc(const ONNX_NAMESPACE::ModelProto& model_proto);//, Ort::CustomOpApi api, OrtKernelContext* context) const;

  //static std::shared_ptr<InferenceEngine::CNNNetwork> cnetwork;
  static InferenceEngine::CNNNetwork cnetwork;


 private:
  mutable std::shared_ptr<ngraph::runtime::Executable> ng_curr_exe_ = nullptr;

  //AllocateFunc allocate_func_ = nullptr;

  //DestroyFunc release_func_ = nullptr;

  //AllocatorHandle allocator_ = nullptr;

  std::string name_="test";
  mutable std::unordered_map<std::string, std::shared_ptr<ngraph::runtime::Executable>> ng_exe_map_;
  mutable std::list<std::string> keyCache;

  mutable std::mutex compute_lock_;

  mutable ONNX_NAMESPACE::ModelProto model_proto_;

  std::shared_ptr<InferenceEngine::CNNNetwork> BuildIntelNetworkWithMO();

  InferenceEngine::Precision ConvertPrecisionONNXToIntel(ONNX_NAMESPACE::DataType onnx_type);

  void GetExecutableHandle(
      std::shared_ptr<InferenceEngine::CNNNetwork> network);

  size_t DeduceBatchSize(Ort::CustomOpApi ort, const OrtValue* input_tensor,
                         InferenceEngine::SizeVector graph_dims);

  void GetInputTensors(Ort::CustomOpApi ort, OrtKernelContext* context, const OrtValue* input_tensors[]);

  void GetOutputTensors(Ort::CustomOpApi ort, OrtKernelContext* context, OrtValue* output_tensors[], size_t batch_size);

  void StartAsyncInference(Ort::CustomOpApi ort, const OrtValue* input_tensors[], size_t batch_slice_idx, size_t infer_req_idx);

  void CompleteAsyncInference(Ort::CustomOpApi ort, OrtValue* output_tensors[], size_t batch_slice_idx, size_t infer_req_idx);

  std::vector<std::string> GetEnvLdLibraryPath() const;

  const onnxruntime::Node* fused_node_;
  std::shared_ptr<InferenceEngine::CNNNetwork> intel_network_;
  size_t num_inf_reqs_;
  InferenceEngine::InferencePlugin plugin_;
  std::vector<InferenceEngine::InferRequest::Ptr> infer_requests_;
  std::string device_id_;
//  mutable std::mutex compute_lock_;
  std::vector<int> input_indexes_;
  InferenceEngine::Precision precision_;
  const onnxruntime::Graph* onnx_graph_;
};
}  // namespace openvino_ep
}  // namespace onnxruntime
