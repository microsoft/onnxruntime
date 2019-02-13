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
		std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map) {
    // TODO - ??? Surya will update the function to reflect the accurate EtlwiseType.
    //int EltwiseType = 1;

	if (onnx_node_->OpType() == "Conv") {
		CreateConvLayer(builder, precision, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "Relu") {
		CreateReLULayer(builder, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "Transpose") {
		CreateTransposeLayer(builder, onnx_openvino_map, openvino_io_map);
	/*
	} else if (onnx_node_->OpType() == "Concat") {
		CreateConcatLayer(builder, onnx_openvino_map, openvino_io_map);\
	} else if (onnx_node_->OpType() == "Norm") {
		CreateNormLayer(builder, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "Eltwise") {
		CreateEltwiseLayer(builder, EltwiseType, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "ReLU") {
		CreateReLULayer(builder, onnx_openvino_map, openvino_io_map);
	}
	*/
	} else if (onnx_node_->OpType() == "AveragePool"){
		CreatePoolingLayer(builder,2,onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "SoftMax") {
		CreateSoftMaxLayer(builder, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "MatMul") {
		CreateFCMatMulLayer(builder, precision, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "Gemm") {
		CreateFCGemmLayer(builder, precision, onnx_openvino_map, openvino_io_map);
	} else if (onnx_node_->OpType() == "Unsqueeze") {
		CreateUnsqueezeLayer(builder,precision,onnx_openvino_map, openvino_io_map);
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

void OpenVINOGraph::SetDevIDAndPrecision(std::string info, std::string& dev_id,
    InferenceEngine::Precision& prec) {
  std::istringstream tokenStream(info);
  char delimiter = '_';
  std::vector<std::string> values;
  std::string token;

  while (std::getline(tokenStream, token, delimiter)) {
    values.push_back(token);
  }

  dev_id = values[0];
  std::string prec_str = values[1];
  if(prec_str == "FP32") {
    prec = InferenceEngine::Precision::FP32;
  } else if (prec_str == "FP16") {
    prec = InferenceEngine::Precision::FP16;
  }


  std::cout<< "OpenVINO EP device:" << dev_id << std::endl;
  std::cout<< "OpenVINO EP precision:" << prec_str << std::endl;

}

OpenVINOGraph::OpenVINOGraph(onnxruntime::Node* fused_node, std::string device_info) {
	//TODO: parse device info to obtain the following values

	SetDevIDAndPrecision(device_info, device_id_, precision_);

	num_inf_reqs_ = (device_id_ == "HDDL") ? 8 : 1;

	fused_node_ = fused_node;
	onnx_graph_ = &(fused_node_->GetFunctionBody()->Body());
  cnn_network_ = BuildCNNNetwork();
  std::string file_name = "./conv_" + fused_node->Name();
  cnn_network_->serialize( file_name+".xml", file_name+".bin");
  infer_requests_ = GetExecutableHandle(cnn_network_, device_id_, precision_);
}

std::vector<std::string> OpenVINOGraph::GetEnvLdLibraryPath() {
    std::string plugin_path = std::getenv("LD_LIBRARY_PATH");
    std::vector<std::string> paths;
    std::string token;
    std::istringstream tokenStream(plugin_path);
    char delimiter = ':';

    while (std::getline(tokenStream , token, delimiter)) {
      paths.push_back(token);
    }
    return paths;
}

std::shared_ptr<InferenceEngine::CNNNetwork> OpenVINOGraph::BuildCNNNetwork() {

  // OpenVINO graph info
  auto builder = std::make_shared<InferenceEngine::Builder::Network>(
      fused_node_->Name());

  // Generate Input nodes
  for (auto input_arg : onnx_graph_->GetInputs()) {
    auto shape_vector = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(
        *(input_arg->Shape()));
    InferenceEngine::SizeVector size_vector(shape_vector.begin(),
        shape_vector.end());
    auto name = input_arg->Name();
    auto ov_layer = OpenVINONode::MakeInputLayer(name, size_vector, builder);
    openvino_io_map_.insert( { name, ov_layer });
  }

  // Generate Output nodes
  for (auto output_arg : onnx_graph_->GetOutputs()) {
    auto name = output_arg->Name();
    auto ov_layer = OpenVINONode::MakeOutputLayer(name, builder);
    openvino_io_map_.insert( { name, ov_layer });
  }



  // Generate op independent info for intermediate nodes (non graph I/O nodes)
  for (int i = 0; i < onnx_graph_->NumberOfNodes(); i++) {
    auto* onnx_node = onnx_graph_->GetNode(i);
    auto openvino_node = std::make_shared<OpenVINONode>(onnx_node, onnx_graph_);
    openvino_nodes_.push_back(openvino_node);
    onnx_openvino_map_.insert( { onnx_node, openvino_node });
  }

  // Create OpenVINO ops for intermediate node (non graph I/O nodes)
  for (auto openvino_node : openvino_nodes_) {
    openvino_node->CreateOpenVINOLayer(builder, precision_, onnx_openvino_map_, openvino_io_map_);
  }


  // Connect the OpenVINO Graph
  for(auto openvino_node : openvino_nodes_) {
	  openvino_node->ConnectToNeighbors(builder);
  }


  std::cout << "builder ready\n";

  auto inetworkptr = builder->build();

  std::cout << " builder built\n";

  return std::make_shared<InferenceEngine::CNNNetwork>(
      InferenceEngine::Builder::convertToICNNNetwork(inetworkptr));
}

std::vector<InferenceEngine::InferRequest::Ptr> OpenVINOGraph::GetExecutableHandle(
    std::shared_ptr<InferenceEngine::CNNNetwork> network,
    const std::string& device, InferenceEngine::Precision precision) {


  // TODO: make this better

  precision = InferenceEngine::Precision::FP32;


  // Load Plugin for inference engine
  std::cout << "[OpenVINO-EP]Loading plugin" << std::endl;

  std::vector<std::string> plugin_path = GetEnvLdLibraryPath();
  plugin_path.push_back("");
  InferenceEngine::InferencePlugin plugin = InferenceEngine::PluginDispatcher(
      plugin_path).getPluginByDevice(device);
  //InferenceEngine::printPluginVersion(plugin, std::cout);

  // Configure input & output
  // Prepare input blobs
  std::cout << "[OpenVINO-EP]Preparing input blobs" << std::endl;

  auto inputInfo = network->getInputsInfo();
  for(auto iter = inputInfo.begin(); iter != inputInfo.end(); ++iter) {
    iter->second->setPrecision(precision);
    switch (iter->second->getTensorDesc().getDims().size()) {
      case 1:
        iter->second->setLayout(InferenceEngine::Layout::C);
        break;
      case 2:
        iter->second->setLayout(InferenceEngine::Layout::NC);
        break;
      case 3:
        iter->second->setLayout(InferenceEngine::Layout::CHW);
        break;
      case 4:
        iter->second->setLayout(InferenceEngine::Layout::NCHW);
        break;
      case 5:
        iter->second->setLayout(InferenceEngine::Layout::NCDHW);
        break;
      default:
        throw "Invalid Dims type for input data map for: " + iter->first;
    }
  }

  network->setBatchSize(1);

  // Prepare output blobs
  auto outputInfo = network->getOutputsInfo();
  for(auto iter = outputInfo.begin(); iter != outputInfo.end(); ++iter) {
    iter->second->setPrecision(precision);
    switch (iter->second->getTensorDesc().getDims().size()) {
      case 1:
        iter->second->setLayout(InferenceEngine::Layout::C);
        break;
      case 2:
        iter->second->setLayout(InferenceEngine::Layout::NC);
        break;
      case 3:
        iter->second->setLayout(InferenceEngine::Layout::CHW);
        break;
      case 4:
        iter->second->setLayout(InferenceEngine::Layout::NCHW);
        break;
      case 5:
        iter->second->setLayout(InferenceEngine::Layout::NCDHW);
        break;
      default:
        throw "Invalid Dims type for output data map for: " + iter->first;
    }
  }

  // Loading model to the plugin
  std::cout << "[OpenVINO-EP]Loading model to the plugin" << std::endl;
  InferenceEngine::ExecutableNetwork exeNetwork = plugin.LoadNetwork(*network,
      { });

  // Create infer request
  std::cout << "[OpenVINO-EP]Creating Infer requests : " << num_inf_reqs_ << std::endl;
  std::vector<InferenceEngine::InferRequest::Ptr> infer_requests;
  for(int i = 0; i < num_inf_reqs_; i++) {
      infer_requests.push_back(exeNetwork.CreateInferRequestPtr());
  }

  return infer_requests;
}

std::shared_ptr<InferenceEngine::CNNNetwork> OpenVINOGraph::GetCNNNetwork() {
  return cnn_network_;
}


void OpenVINOGraph::Infer(onnxruntime::ONNXRunTimeTensor* input_tensors,
		size_t num_inputs, onnxruntime::ONNXRunTimeTensor* output_tensors,
		size_t num_outputs, onnxruntime::AllocateFunc& output_allocator_func,
		onnxruntime::AllocatorHandle& output_allocator_handle) {

  num_inputs = 1;
  //num_outputs = 1;

	// Check I/O sizes
	auto graph_input_info = cnn_network_->getInputsInfo();
	if (num_inputs != graph_input_info.size()) {
		throw "OpenVINO Inference: Inputs count mismatch!";
	}

	auto graph_output_info = cnn_network_->getOutputsInfo();
	if (num_outputs != graph_output_info.size()) {
		throw "OpenVINO Inference: Outputs count mismatch!";
	}

	//
	// Copies the same input to all infer request blobs and
	// starts an async inference on each of them.
	// Output from only the first infer_request is returned.
	//



	// Prepare input
	for(auto infer_request : infer_requests_) {

    size_t i = 0;
    for (auto input_info_iter = graph_input_info.begin();
        input_info_iter != graph_input_info.end(); ++input_info_iter, ++i) {

      // Get OpenVINO's input buffer
      auto graph_input_blob = infer_request->GetBlob(input_info_iter->first);
      auto graph_input_buffer =
          graph_input_blob->buffer().as<
              InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

      // Get data size
      size_t num_input_elements = 1;
      for (auto dim : input_info_iter->second->getTensorDesc().getDims()) {
        num_input_elements *= dim;
      }

      size_t input_data_size = num_input_elements * sizeof(float);

      // Copy input data into OpenVINO's input buffer
      std::memcpy(graph_input_buffer, input_tensors[i].data, input_data_size);
    }
  }


	// Start Async inferences
	for(auto infer_request : infer_requests_) {
	  infer_request->StartAsync();
	}

	// Wait for results
	for(auto infer_request : infer_requests_) {
	  infer_request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
	}

	// Process output

	auto infer_request = infer_requests_[0];

	size_t i = 0;
	for (auto output_info_iter = graph_output_info.begin();
			output_info_iter != graph_output_info.end();
			++output_info_iter, ++i) {

		// Get OpenVINO's output buffer
		auto graph_output_blob = infer_request->GetBlob(
				output_info_iter->first);
		auto graph_output_buffer =
				graph_output_blob->buffer().as<
						InferenceEngine::PrecisionTrait<
								InferenceEngine::Precision::FP32>::value_type*>();


		// Get data size & initialize output tensor info
		auto graph_output_dims = graph_output_blob->getTensorDesc().getDims();
		auto num_dims = graph_output_dims.size();
		size_t output_data_size = graph_output_blob->byteSize();

    // TODO: Memory Leak!!!!
    // fix before shipping.
		output_tensors[i].shape = new int64_t[num_dims];
		for (int j = 0; j < num_dims; j++) {
			output_tensors[i].shape[j] = (int64_t)graph_output_dims[j];
		}


		output_tensors[i].ndim = num_dims;
		output_tensors[i].dtype = onnxruntime::DType::TFloat32;
		output_tensors[i].data = (*output_allocator_func)(output_allocator_handle, 64, output_data_size);
		std::memcpy(output_tensors[i].data, graph_output_buffer, output_data_size);

	}
}
} // namespace openvino_ep
