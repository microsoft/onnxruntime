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

//Forward Declaration
class OpenVINOGraph;

class OpenVINONode {
public:
  std::shared_ptr<InferenceEngine::Builder::LayerFragment> layer_;
  std::size_t layerID_;

  const onnxruntime::Node* onnx_node_;
  const onnxruntime::NodeArg* onnx_nodearg_; // used by I/O nodes
  OpenVINOGraph* openvino_graph_;

  bool is_input_node_;
  bool is_output_node_;
  std::map<std::string, InferenceEngine::idx_t> input_connections_info_;
  std::map<std::string, InferenceEngine::idx_t> output_connections_info_;


  OpenVINONode(const onnxruntime::NodeArg* nodearg, OpenVINOGraph* ov_graph);
  OpenVINONode(const onnxruntime::Node* node, OpenVINOGraph* ov_graph);

  void InitializeOp(
            std::map<std::string, InferenceEngine::Blob::Ptr>& blob_map);


  void ConnectToInputs();

  InferenceEngine::idx_t GetOutputPort(std::string name) {
	  return output_connections_info_.at(name);
  }

private:
  InferenceEngine::SizeVector GetDimsVector(const std::string& tensor_name);
  void* GetTensorData(const std::string& tensor_name, InferenceEngine::Precision precision);
  size_t GetTensorElemCount(const std::string& tensor_name);

  bool AttributeExists(std::string name) {
	  auto attributes = onnx_node_->GetAttributes();
	  return ( attributes.find(name) == attributes.end() ) ?
			  false : true;
  }

  // OpenVINO Operator implementations
  // ??? Use Factory pattern instead???
  void CreateInputLayer();
  void CreateOutputLayer();
  void CreateConvLayer();
  void CreateTransposeLayer();
  void CreateReLULayer();
  void CreateConcatLayer();
  void CreateNormLayer();
  void CreateEltwiseLayer(int EltwiseType);
  void CreateReshapeLayer();
  void CreatePoolingLayer(int poolingType);
  void CreateSoftMaxLayer();
  void CreateFCMatMulLayer();
  void CreateFCGemmLayer();
  void CreateScaleShiftLayer();
  void CreateScaleShiftImgLayer();
  void CreateUnsqueezeLayer(
    std::map<std::string, InferenceEngine::Blob::Ptr>& blob_map);
  void CreateScaleMulAddLayer( int type,
    std::map<std::string,InferenceEngine::Blob::Ptr>& blob_map);

};

}

#endif
