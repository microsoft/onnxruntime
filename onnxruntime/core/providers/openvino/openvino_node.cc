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

OpenVINONode::OpenVINONode(const onnxruntime::NodeArg* nodearg, OpenVINOGraph* graph) {
  onnx_node_ = nullptr;
  onnx_nodearg_ = nodearg;
  openvino_graph_ = graph;
  layerID_ = -1;
  is_input_node_ = false;
  is_output_node_ = false;
}

OpenVINONode::OpenVINONode(const onnxruntime::Node* node,
    OpenVINOGraph* graph) {
  onnx_node_ = node;
  openvino_graph_ = graph;
  onnx_nodearg_ = nullptr;
  is_input_node_ = false;
  is_output_node_ = false;
  layerID_ = -1;
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
  openvino_graph_->onnx_graph_->GetInitializedTensor(tensor_name, tensor_proto);
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

InferenceEngine::SizeVector OpenVINONode::GetDimsVector(
    const std::string& tensor_name) {

  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  openvino_graph_->onnx_graph_->GetInitializedTensor(tensor_name, tensor_proto);
  InferenceEngine::SizeVector dims;
  for (int i = 0; i < tensor_proto->dims_size(); i++) {
    dims.push_back(size_t(tensor_proto->dims(i)));
  }
  return dims;
}


size_t OpenVINONode::GetTensorElemCount(
    const std::string& tensor_name) {

  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  openvino_graph_->onnx_graph_->GetInitializedTensor(tensor_name, tensor_proto);
  size_t size = 1;
  for (int i = 0; i < tensor_proto->dims_size(); i++) {
    size *= tensor_proto->dims(i);
  }
  return size;
}


void OpenVINONode::InitializeOp(
		std::map<std::string, InferenceEngine::Blob::Ptr>& blob_map) {
    // TODO - ??? Surya will update the function to reflect the accurate EtlwiseType.
    //int EltwiseType = 1;

		 if (onnx_node_->OpType() == "Conv") { CreateConvLayer(); }
	else if (onnx_node_->OpType() == "Relu") { CreateReLULayer(); }
	else if (onnx_node_->OpType() == "Transpose") { CreateTransposeLayer(); }
	else if (onnx_node_->OpType() == "Concat") { CreateConcatLayer(); }
	else if (onnx_node_->OpType() == "LRN") { CreateNormLayer(); }
  //else if (onnx_node_->OpType() == "Eltwise") { CreateEltwiseLayer(EltwiseType); }
  //else if (onnx_node_->OpType() == "ReLU") { CreateReLULayer(); }
	else if (onnx_node_->OpType() == "AveragePool"){ CreatePoolingLayer(2); }
	else if (onnx_node_->OpType() == "GlobalAveragePool"){ CreatePoolingLayer(3); }
	else if (onnx_node_->OpType() == "MaxPool"){ CreatePoolingLayer(1); }
	else if (onnx_node_->OpType() == "Mul"){ CreateScaleMulAddLayer(1 ,blob_map); }
	else if (onnx_node_->OpType() == "Add"){ CreateScaleMulAddLayer(2, blob_map); }
	else if (onnx_node_->OpType() == "Sum"){ CreateEltwiseLayer(1); }
	else if (onnx_node_->OpType() == "SoftMax") { CreateSoftMaxLayer(); }
	else if (onnx_node_->OpType() == "MatMul") { CreateFCMatMulLayer(); }
	else if (onnx_node_->OpType() == "Gemm") { CreateFCGemmLayer(); }
	else if (onnx_node_->OpType() == "Unsqueeze") { CreateUnsqueezeLayer(blob_map); }
	else if (onnx_node_->OpType() == "BatchNormalization") { CreateScaleShiftLayer(); }
	else if (onnx_node_->OpType() == "ImageScaler") { CreateScaleShiftImgLayer(); }
	else if (onnx_node_->OpType() == "Reshape") { CreateReshapeLayer(); }
}

void OpenVINONode::ConnectToInputs() {

	for(auto entry : input_connections_info_) {
		auto dest_layerID = layerID_;
		auto dest_port = entry.second;

		auto tensor_name = entry.first;
		auto source_node = openvino_graph_->GetTensorProducer(tensor_name);
		auto source_layerID = source_node->layerID_;
		auto source_port = source_node->GetOutputPort(tensor_name);

		InferenceEngine::PortInfo src_port_info(source_layerID, source_port);
		InferenceEngine::PortInfo dest_port_info(dest_layerID, dest_port);
		openvino_graph_->GetBuilder()->connect(src_port_info, dest_port_info);

	}

}

} // namespace openvino_ep
