#include <map>
#include <memory>

#include <inference_engine.hpp>
#include <ie_utils.hpp>
#include <ie_builders.hpp>
#include <cpp/ie_infer_request.hpp>

#include "core/framework/func_api.h"

#include "core/graph/graph.h"

#ifndef OPENVINO_EP__OPENVINO_GRAPH_H
#define OPENVINO_EP__OPENVINO_GRAPH_H

namespace openvino_ep {

class OpenVINONode {
public:
  std::shared_ptr<InferenceEngine::Builder::LayerFragment> layer_;
  std::size_t layerID_;

  const onnxruntime::Node* onnx_node_;
  const onnxruntime::Graph* onnx_graph_;

  std::vector<const onnxruntime::NodeArg*> input_defs_;
  std::vector<const onnxruntime::NodeArg*> output_defs_;
  std::vector<onnxruntime::Node::EdgeEnd> input_edges_;
  std::vector<onnxruntime::Node::EdgeEnd> output_edges_;
  std::vector<const onnxruntime::NodeArg*> graph_input_defs_;
  std::vector<const onnxruntime::NodeArg*> graph_output_defs_;
  bool is_input_node_;
  bool is_output_node_;
  bool node_connects_to_graph_inputs_;
  bool node_connects_to_graph_outputs_;
  std::vector<std::pair<std::shared_ptr<OpenVINONode>, InferenceEngine::idx_t>> input_connections_;
  std::vector<std::pair<std::shared_ptr<OpenVINONode>, InferenceEngine::idx_t>> output_connections_;

  OpenVINONode();

  OpenVINONode(const onnxruntime::Node* node, const onnxruntime::Graph* graph);

  static std::shared_ptr<OpenVINONode> MakeInputLayer(std::string name,
      const InferenceEngine::SizeVector& shape,
      std::shared_ptr<InferenceEngine::Builder::Network>& builder);

  static std::shared_ptr<OpenVINONode> MakeOutputLayer(std::string name,
      std::shared_ptr<InferenceEngine::Builder::Network>& builder);

  void GenerateGraphConnections(
      std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>> onnx_openvino_map,
      std::map<std::string, std::shared_ptr<OpenVINONode>> openvino_io_map,
      std::shared_ptr<InferenceEngine::Builder::Network>& builder);

  void CreateOpenVINOLayer(
      std::shared_ptr<InferenceEngine::Builder::Network>& builder,
      InferenceEngine::Precision precision,
      std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
      std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map_);

  void ConnectToNeighbors(
      std::shared_ptr<InferenceEngine::Builder::Network>& builder);

private:
  InferenceEngine::SizeVector GetDimsVector(const std::string& tensor_name);
  void* GetTensorData(const std::string& tensor_name, InferenceEngine::Precision precision);
  size_t GetTensorElemCount(const std::string& tensor_name);
  void IdentifyIONodes();

  void CreateConvLayer(
      std::shared_ptr<InferenceEngine::Builder::Network>& builder,
      InferenceEngine::Precision precision,
      std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
      std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);
  void CreateTransposeLayer(
          std::shared_ptr<InferenceEngine::Builder::Network>& builder,
          std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
          std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);
  void CreateReLULayer(
        std::shared_ptr<InferenceEngine::Builder::Network>& builder,
        std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
        std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);
  /*void CreateConcatLayer(
        std::shared_ptr<InferenceEngine::Builder::Network>& builder,
        std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
        std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);
  void CreateNormLayer(
        std::shared_ptr<InferenceEngine::Builder::Network>& builder,
        std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
        std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);
  void CreateEltwiseLayer(
       std::shared_ptr<InferenceEngine::Builder::Network>& builder,
	   int EltwiseType,
       std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
       std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);
	   */
  void CreateReshapeLayer(
      std::shared_ptr<InferenceEngine::Builder::Network>& builder,
      InferenceEngine::Precision precision,
      std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
      std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);

  void CreatePoolingLayer(
      std::shared_ptr<InferenceEngine::Builder::Network>& builder,
      int poolingType,
      std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
      std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);

  void CreateSoftMaxLayer(
      std::shared_ptr<InferenceEngine::Builder::Network>& builder,
      std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
      std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);

  void CreateFCMatMulLayer(
    std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);

  void CreateFCGemmLayer(
    std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);

  void CreateUnsqueezeLayer(
    std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);


  void CreateScaleShiftLayer(
    std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);

  void CreateScaleShiftImgLayer(
    std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);

};

class OpenVINOGraph {
public:
  OpenVINOGraph(onnxruntime::Node* fused_node, std::string device_info);

  std::shared_ptr<InferenceEngine::CNNNetwork> GetCNNNetwork();


  void Infer(onnxruntime::ONNXRunTimeTensor* input_tensors,
      size_t num_inputs, onnxruntime::ONNXRunTimeTensor* output_tensors,
      size_t num_outputs, onnxruntime::AllocateFunc& output_allocator_func,
      onnxruntime::AllocatorHandle& output_allocator_handle);

private:
  void TranslateToOpenVINOOperator(
      std::vector<const onnxruntime::Node*>& onnx_nodes,
      std::vector<std::shared_ptr<OpenVINONode>>& openvino_nodes, int i);

  std::shared_ptr<InferenceEngine::CNNNetwork> BuildCNNNetwork();

  std::vector<InferenceEngine::InferRequest::Ptr> GetExecutableHandle(
      std::shared_ptr<InferenceEngine::CNNNetwork> network,
      const std::string& device, InferenceEngine::Precision precision);

  std::vector<std::string> GetEnvLdLibraryPath();

  void SetDevIDAndPrecision(std::string info, std::string& dev_id, InferenceEngine::Precision& prec);

  onnxruntime::Node* fused_node_;
  const onnxruntime::Graph* onnx_graph_;
  std::shared_ptr<InferenceEngine::CNNNetwork> cnn_network_;
  size_t num_inf_reqs_;
  std::vector<InferenceEngine::InferRequest::Ptr> infer_requests_;
  std::string device_id_;
  InferenceEngine::Precision precision_;

  std::vector<std::shared_ptr<OpenVINONode>> openvino_nodes_;
  std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>> onnx_openvino_map_;
  std::map<std::string, std::shared_ptr<OpenVINONode>> openvino_io_map_;

};
}

#endif
