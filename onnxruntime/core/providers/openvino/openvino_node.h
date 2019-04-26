#include <map>
#include <memory>

#include <inference_engine.hpp>
#include <ie_utils.hpp>
#include <ie_builders.hpp>
#include <cpp/ie_infer_request.hpp>

#include "core/framework/func_api.h"

#include "core/graph/graph.h"

#ifndef OPENVINO_EP__OPENVINO_NODE_H
#define OPENVINO_EP__OPENVINO_NODE_H

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

  void CreateOpenVINOLayer(
      std::shared_ptr<InferenceEngine::Builder::Network>& builder,
      InferenceEngine::Precision precision,
      std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
      std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map_,
      std::map<std::string, InferenceEngine::Blob::Ptr>& blob_map);


  void ConnectToNeighbors(
      std::shared_ptr<InferenceEngine::Builder::Network>& builder);

private:
  InferenceEngine::SizeVector GetDimsVector(const std::string& tensor_name);
  void* GetTensorData(const std::string& tensor_name, InferenceEngine::Precision precision);
  size_t GetTensorElemCount(const std::string& tensor_name);

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
  void CreateConcatLayer(
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

  void CreateUnsqueezeLayer(
    // std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    // std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    // std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map);
    std::map<std::string, InferenceEngine::Blob::Ptr>& blob_map);

  void CreateScaleMulAddLayer(
    std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    int type,
    std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map,
    std::map<std::string,InferenceEngine::Blob::Ptr>& blob_map);

};

}

#endif
