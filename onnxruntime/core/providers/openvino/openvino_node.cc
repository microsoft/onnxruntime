#include <iostream>
#include <cstdlib>
#include <map>
#include <string>
#include <memory>
#include <cstdlib>

#include <inference_engine.hpp>
#include <ie_builders.hpp>

#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"

#include "openvino_node.h"
#include "openvino_graph.h"

namespace openvino_ep {

OpenVINONode::OpenVINONode() {
  onnx_node_ = nullptr;
  onnx_graph_ = nullptr;
  layerID_ = -1;
  node_connects_to_graph_inputs_ = false;
  node_connects_to_graph_outputs_ = false;
  is_input_node_ = false;
  is_output_node_ = false;
}

OpenVINONode::OpenVINONode(const onnxruntime::Node* node,
    const onnxruntime::Graph* graph) :
    onnx_node_ { node }, onnx_graph_ { graph } {
  is_input_node_ = false;
  is_output_node_ = false;
  node_connects_to_graph_inputs_ = false;
  node_connects_to_graph_outputs_ = false;
  layerID_ = -1;

  for (auto iter = onnx_node_->InputDefs().begin();
      iter != onnx_node_->InputDefs().end(); ++iter) {
    input_defs_.push_back(*iter);
  }

  for (auto iter = onnx_node_->OutputDefs().begin();
      iter != onnx_node_->OutputDefs().end(); ++iter) {
    output_defs_.push_back(*iter);
  }

  for (auto iter = onnx_node_->InputEdgesBegin();
        iter != onnx_node_->InputEdgesEnd(); ++iter) {
      input_edges_.push_back(*iter);
    }

    for (auto iter = onnx_node_->OutputEdgesBegin();
        iter != onnx_node_->OutputEdgesEnd(); ++iter) {
      output_edges_.push_back(*iter);
    }

  for (auto graph_input : onnx_graph_->GetInputs()) {
    for (auto node_input : onnx_node_->InputDefs()) {
      if (node_input == graph_input) {
        graph_input_defs_.push_back(node_input);
        node_connects_to_graph_inputs_= true;
      }
    }
  }

  for (auto graph_output : onnx_graph_->GetOutputs()) {
    for (auto node_output : onnx_node_->OutputDefs()) {
      if (node_output == graph_output) {
        graph_output_defs_.push_back(node_output);
        node_connects_to_graph_outputs_= true;
      }
    }
  }

}


std::shared_ptr<OpenVINONode> OpenVINONode::MakeInputLayer(
    std::string name, const InferenceEngine::SizeVector& shape,
    std::shared_ptr<InferenceEngine::Builder::Network>& builder) {

  auto input_layer = std::make_shared<InferenceEngine::Builder::InputLayer>(
      name);
  input_layer->setPort(InferenceEngine::Port(shape));
  auto ov_layer = std::make_shared<OpenVINONode>();
  ov_layer->layer_ = input_layer;
  ov_layer->layerID_ = builder->addLayer(*ov_layer->layer_);
  ov_layer->is_input_node_ = true;
  return ov_layer;
}

std::shared_ptr<OpenVINONode> OpenVINONode::MakeOutputLayer(
    std::string name,
    std::shared_ptr<InferenceEngine::Builder::Network>& builder) {
  auto output_layer = std::make_shared<InferenceEngine::Builder::OutputLayer>(
      name);
  auto ov_layer = std::make_shared<OpenVINONode>();
  ov_layer->layer_ = output_layer;
  ov_layer->layerID_ = builder->addLayer(*ov_layer->layer_);
  ov_layer->is_output_node_ = true;
  return ov_layer;
}


InferenceEngine::SizeVector OpenVINONode::GetDimsVector(
    const std::string& tensor_name) {

  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  onnx_graph_->GetInitializedTensor(tensor_name, tensor_proto);
  InferenceEngine::SizeVector dims;
  for (int i = 0; i < tensor_proto->dims_size(); i++) {
    dims.push_back(size_t(tensor_proto->dims(i)));
  }
  return dims;
}

float asfloat(uint32_t v) {
    union {
        float f;
        std::uint32_t u;
    } converter = {0};
    converter.u = v;
    return converter.f;
}

short f32tof16(float x) {
    static float min16 = asfloat((127 - 14) << 23);

    static float max16 = asfloat(((127 + 15) << 23) | 0x007FE000);
    static uint32_t max16f16 = ((15 + 15) << 10) | 0x3FF;

    static constexpr std::uint32_t EXP_MASK_F32 = 0x7F800000U;

    union {
        float f;
        uint32_t u;
    } v = {0};
    v.f = x;

    uint32_t s = (v.u >> 16) & 0x8000;
    v.u &= 0x7FFFFFFF;

    if ((v.u & EXP_MASK_F32) == EXP_MASK_F32) {
        if (v.u & 0x007FFFFF) {
            return static_cast<short>(s | (v.u >> (23 - 10)) | 0x0200);
        } else {
            return static_cast<short>(s | (v.u >> (23 - 10)));
        }
    }

    float halfULP = asfloat(v.u & EXP_MASK_F32) * asfloat((127 - 11) << 23);
    v.f += halfULP;

    if (v.f < min16 * 0.5f) {
        return static_cast<short>(s);
    }
    if (v.f < min16) {
        return static_cast<short>(s | (1 << 10));
    }
    if (v.f >= max16) {
        return static_cast<short>(max16f16 | s);
    }
    v.u -= ((127 - 15) << 23);
    v.u >>= (23 - 10);
    return static_cast<short>(v.u | s);
}

short* f32tof16Arrays(short* dst, const float *src, size_t nelem, float scale = 1.0, float bias = 0) {

    for (size_t i = 0; i < nelem; i++){
        dst[i] = f32tof16(src[i] * scale + bias);
        // if(i%1000 == 0){
            // std::cout << "Src is " << src[i] << "  Dst is  " << dst[i] << std::endl;
        // }
    }
    return dst;
}

void* OpenVINONode::GetTensorData(const std::string& tensor_name, InferenceEngine::Precision precision) {

  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  onnx_graph_->GetInitializedTensor(tensor_name, tensor_proto);
  float* fp32_data = (float*) tensor_proto->raw_data().c_str();
  void* return_ptr = nullptr;

  if(precision == InferenceEngine::Precision::FP32) {
        return_ptr =  (void*) fp32_data;
  } else if ( precision == InferenceEngine::Precision::FP16) {
        auto element_count = GetTensorElemCount(tensor_name);
        // TODO: Memory Leak!!!!
        // fix before shipping.
        short* fp16_data = new short[element_count];
        f32tof16Arrays(fp16_data, fp32_data, element_count);
        return_ptr =  (void*) fp16_data;
  }
  return return_ptr;
}

size_t OpenVINONode::GetTensorElemCount(
    const std::string& tensor_name) {

  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  onnx_graph_->GetInitializedTensor(tensor_name, tensor_proto);
  size_t size = 1;
  for (int i = 0; i < tensor_proto->dims_size(); i++) {
    size *= tensor_proto->dims(i);
  }
  return size;
}


void OpenVINONode::CreateOpenVINOLayer(
        std::shared_ptr<InferenceEngine::Builder::Network>& builder, InferenceEngine::Precision precision,
	  std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
		std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map,
		std::map<std::string, InferenceEngine::Blob::Ptr>& blob_map) {
    // TODO - ??? Surya will update the function to reflect the accurate EtlwiseType.
    //int EltwiseType = 1;

	if (onnx_node_->OpType() == "Conv") {
		CreateConvLayer(builder, precision, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "Relu") {
		CreateReLULayer(builder, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "Transpose") {
		CreateTransposeLayer(builder, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "Concat") {
		CreateConcatLayer(builder, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "LRN") {
		CreateNormLayer(builder, onnx_openvino_map, openvino_io_map);
	// } else if (onnx_node_->OpType() == "Eltwise") {
		// CreateEltwiseLayer(builder, EltwiseType, onnx_openvino_map, openvino_io_map);
	// } else if (onnx_node_->OpType() == "ReLU") {
		// CreateReLULayer(builder, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "AveragePool"){
		CreatePoolingLayer(builder,2,onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "GlobalAveragePool"){
		CreatePoolingLayer(builder,3,onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "MaxPool"){
		CreatePoolingLayer(builder,1,onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "Mul"){
		CreateScaleMulAddLayer(builder, precision, 1, onnx_openvino_map, openvino_io_map,blob_map);
	} else if (onnx_node_->OpType() == "Add"){
		CreateScaleMulAddLayer(builder, precision, 2,onnx_openvino_map, openvino_io_map,blob_map);
	} else if (onnx_node_->OpType() == "Sum"){
		CreateEltwiseLayer(builder, 1,onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "SoftMax") {
		CreateSoftMaxLayer(builder, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "MatMul") {
		CreateFCMatMulLayer(builder, precision, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "Gemm") {
		CreateFCGemmLayer(builder, precision, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "Unsqueeze") {
		CreateUnsqueezeLayer(precision,blob_map);
	} else if (onnx_node_->OpType() == "BatchNormalization") {
		CreateScaleShiftLayer(builder,precision,onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "ImageScaler") {
		CreateScaleShiftImgLayer(builder,precision,onnx_openvino_map, openvino_io_map);
	} else {
		CreateReshapeLayer(builder, precision, onnx_openvino_map, openvino_io_map);
    }
}


void OpenVINONode::ConnectToNeighbors(std::shared_ptr<InferenceEngine::Builder::Network>& builder) {

	// Connect to this nodes inputs
	for(auto entry : input_connections_) {

		auto dest_layerID = layerID_;
		auto dest_port_idx = entry.second;
		auto src_node = entry.first;
		auto src_layerID = src_node->layerID_;
		InferenceEngine::idx_t src_port_idx = 0;

		// if input is an internal node, find appropriate port
		if(!src_node->is_input_node_) {
			for(auto out_conn : src_node->output_connections_) {
				if( out_conn.first->layerID_ == dest_layerID) {
					src_port_idx = out_conn.second;
				}
				break;
			}
		}

		InferenceEngine::PortInfo src_port_info(src_layerID, src_port_idx);
		InferenceEngine::PortInfo dest_port_info(dest_layerID, dest_port_idx);
		builder->connect(src_port_info, dest_port_info);
	}

	// Connect to Graph's output nodes, if required
	if(node_connects_to_graph_outputs_) {
		for(auto entry : output_connections_) {
			auto dest_node = entry.first;
			if(dest_node->is_output_node_){
				auto dest_port_idx = 0;
				auto dest_layerID = dest_node->layerID_;
				auto src_layerID = layerID_;
				auto src_port_idx = entry.second;
				InferenceEngine::PortInfo src_port_info(src_layerID, src_port_idx);
				InferenceEngine::PortInfo dest_port_info(dest_layerID, dest_port_idx);
				builder->connect(src_port_info, dest_port_info);
			}
		}
	}

}

} // namespace openvino_ep
