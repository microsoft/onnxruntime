#include <map>
#include <memory>

#include <inference_engine.hpp>
#include <ie_utils.hpp>
#include <ie_builders.hpp>
#include <cpp/ie_infer_request.hpp>

#include "core/framework/func_api.h"

#include "core/graph/graph.h"
#include "openvino_node.h"

#ifndef OPENVINO_EP__OPENVINO_GRAPH_H
#define OPENVINO_EP__OPENVINO_GRAPH_H

namespace openvino_ep {

class OpenVINOGraph {
public:
  InferenceEngine::Precision precision_;
  const onnxruntime::Graph* onnx_graph_;

  OpenVINOGraph(onnxruntime::Node* fused_node, std::string device_info);

  std::shared_ptr<InferenceEngine::CNNNetwork> GetCNNNetwork();


  void Infer(onnxruntime::ONNXRunTimeTensor* input_tensors,
      size_t num_inputs, onnxruntime::ONNXRunTimeTensor* output_tensors,
      size_t num_outputs, onnxruntime::AllocateFunc& output_allocator_func,
      onnxruntime::AllocatorHandle& output_allocator_handle);


  std::shared_ptr<InferenceEngine::Builder::Network> GetBuilder(){
    return builder_;
  }

  bool IsInitializer(std::string name) {
	  return (initializers_->find(name) == initializers_->end()) ?
			  false : true;
  }
  std::shared_ptr<OpenVINONode> GetTensorProducer(std::string name) {
	  return tensor_producers_.at(name);
  }

private:
  void TranslateToOpenVINOOperator(
      std::vector<const onnxruntime::Node*>& onnx_nodes,
      std::vector<std::shared_ptr<OpenVINONode>>& openvino_nodes, int i);

  std::shared_ptr<InferenceEngine::CNNNetwork> BuildCNNNetwork();

  std::vector<InferenceEngine::InferRequest::Ptr> GetExecutableHandle(
      std::shared_ptr<InferenceEngine::CNNNetwork> network,
      const std::string& device, InferenceEngine::Precision precision);

  std::vector<std::string> GetEnvLdLibraryPath();

  onnxruntime::Node* fused_node_;
  std::shared_ptr<InferenceEngine::CNNNetwork> cnn_network_;
  size_t num_inf_reqs_;
  std::vector<InferenceEngine::InferRequest::Ptr> infer_requests_;
  std::string device_id_;
  std::shared_ptr<InferenceEngine::Builder::Network> builder_;

  std::vector<std::shared_ptr<OpenVINONode>> openvino_nodes_;
  std::map<std::string, std::shared_ptr<OpenVINONode>> tensor_producers_;
  onnxruntime::InitializedTensorSet* initializers_;
  std::map<std::string, InferenceEngine::Blob::Ptr> const_blob_map_;

};
}

#endif
