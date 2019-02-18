// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "trt_kernel.h"
#include "core/platform/env.h"
#include "core/graph/model.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime{
TRTKernel::TRTKernel(const OpKernelInfo& info) : OpKernel(info){
    std::unordered_map<std::string, int> domain_to_version;
    domain_to_version[onnxruntime::kOnnxDomain] = 8;
    onnxruntime::Model model("", true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version);
    onnxruntime::Graph& graph = model.MainGraph();

    auto& node = info.node();
    if (node.NodeType() == onnxruntime::Node::Type::Primitive){
        //This is a primitive node. Refer to an op directly
        std::vector<onnxruntime::NodeArg*> inputs, outputs;
        for (const auto& input : node.InputDefs()){
            auto& n_input = graph.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
            inputs.push_back(&n_input);
        }

        for (const auto& output : node.OutputDefs()){
            auto& n_output = graph.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
            outputs.push_back(&n_output);
        }

        graph.AddNode(node.Name(), node.OpType(), node.Description(), inputs, outputs, &node.GetAttributes(), node.Domain());

        ORT_ENFORCE(graph.Resolve().IsOK());
    }
    else if (node.NodeType() == onnxruntime::Node::Type::Fused){
        //This is a fused node. Refer to a function (sub-graph)
        const Graph& graph_body = node.GetFunctionBody()->Body();
        for (auto& graph_body_node : graph_body.Nodes()){
            graph.AddNode(graph_body_node);
        }
        ORT_ENFORCE(graph.Resolve().IsOK());

        //Add inputs to graph
        for (int i = 0, end = node.InputDefs().size(); i < end; ++i){
            const onnxruntime::Tensor* temp = nullptr;
            const onnxruntime::Tensor** constant_input_value = &temp;
            auto stat = info.TryGetConstantInput(i, constant_input_value);
            if (stat){ // Initialized inputs
                auto& input_tensor = *constant_input_value;
                const auto& tensor_type = input_tensor->DataType();
                TensorProto::DataType dtype = TensorProto_DataType_UNDEFINED;
                TensorProto tensor_proto;

                if (tensor_type == DataTypeImpl::GetType<float>()){
                    dtype = TensorProto_DataType_FLOAT;
                    const auto& input_data = input_tensor->Data<float>();
                    for (int j = 0, end = input_tensor->Shape().Size(); j < end; ++j){
                        tensor_proto.add_float_data(input_data[j]);
                    }
                }
                else if (tensor_type == DataTypeImpl::GetType<int64_t>()){
                    dtype = TensorProto_DataType_INT64;
                    const auto& input_data = input_tensor->Data<int64_t>();
                    for (int j = 0, end = input_tensor->Shape().Size(); j < end; ++j){
                        tensor_proto.add_int64_data(input_data[j]);
                    }
                }
                else{ // TODO: add other tensor types
                    dtype = TensorProto_DataType_UNDEFINED;
                }

                tensor_proto.set_name(node.InputDefs()[i]->Name());

                auto& shape = input_tensor->Shape().GetDims();
                for (auto& dim : shape){
                    tensor_proto.add_dims(dim);
                }

                tensor_proto.set_data_type(dtype);
                graph.AddInitializedTensor(tensor_proto);
            }
            else{ //Graph inputs
                graph_input_indexes_.push_back(i);
                graph_input_names_.push_back(node.InputDefs()[i]->Name());
            }
        }
    }

    ONNX_NAMESPACE::ModelProto model_p = model.ToProto();

    //Add node's outputs to graphproto's outputs if the node's EdgeEnd nodes are not in the graph
    std::set<string> output_set;
    for (int i = 0, end = model_p.graph().output().size(); i < end; ++i){
        output_set.insert(model_p.graph().output()[i].name());
    }

    std::vector<int> output_to_add;
    for (int i = 0, end = node.OutputDefs().size(); i < end; ++i){
        const std::string& output_name = node.OutputDefs()[i]->Name();
        if (output_set.find(output_name) == output_set.end()){
            for (int j = 0, end = model_p.graph().value_info().size(); j < end; ++j){
                if (output_name == model_p.graph().value_info()[j].name()){
                    output_to_add.push_back(j);
                }
            }
        }
    }

    for (auto& i: output_to_add){
        *(model_p.mutable_graph()->mutable_output()->Add()) = model_p.graph().value_info()[i];
    }

    // Create TensorRT engine
    string string_buf;
    model_p.SerializeToString(&string_buf);
    std::vector<char> onnx_buf(string_buf.begin(), string_buf.end());
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    TRTLogger trt_logger(static_cast<nvinfer1::ILogger::Severity>(verbosity));
    const auto& trt_builder = InferObject(nvinfer1::createInferBuilder(trt_logger));
    const auto& trt_network = InferObject(trt_builder->createNetwork());
    const auto& trt_parser  = InferObject(nvonnxparser::createParser(trt_network.get(), trt_logger));
    trt_parser->parse(onnx_buf.data(), onnx_buf.size());
    trt_builder->setMaxBatchSize(max_batch_size);
    trt_builder->setMaxWorkspaceSize(max_workspace_size);
    engine_ = InferObject(trt_builder->buildCudaEngine(*trt_network.get()));
    ORT_ENFORCE(engine_ != nullptr);

    // Build trt context
    tensorrt_context_ = engine_->createExecutionContext();
    ORT_ENFORCE(tensorrt_context_ != nullptr);

    // Get input shape and binding index
    input_shapes_.resize(trt_network->getNbInputs());
    for (int i = 0, end = trt_network->getNbInputs(); i < end; ++i){
        const std::string& name = trt_network->getInput(i)->getName();
        size_t bindingIndex = engine_->getBindingIndex(name.c_str());
        nvinfer1::Dims4 dimensions = static_cast<nvinfer1::Dims4&&>(engine_->getBindingDimensions(static_cast<int>(bindingIndex)));
        size_t dim_size = 1;
        input_shapes_[i].push_back(dim_size);

        for (int j = 0, end = dimensions.nbDims; j < end; ++j){
            input_shapes_[i].push_back(dimensions.d[j]);
            dim_size *= dimensions.d[j];
        }

        input_binding_indexes_.push_back(bindingIndex);
        input_dim_sizes_.push_back(dim_size);
    }

    // Get output shape and binding index
    output_shapes_.resize(trt_network->getNbOutputs());
    for (int i = 0, end = trt_network->getNbOutputs(); i < end; ++i){
        const std::string& name = trt_network->getOutput(i)->getName();
        size_t bindingIndex = engine_->getBindingIndex(name.c_str());
        nvinfer1::Dims4 dimensions = static_cast<nvinfer1::Dims4&&>(engine_->getBindingDimensions(static_cast<int>(bindingIndex)));
        size_t dim_size = 1;
        output_shapes_[i].push_back(dim_size);

        for (int j = 0, end = dimensions.nbDims; j < end; ++j){
            output_shapes_[i].push_back(dimensions.d[j]);
            dim_size *= dimensions.d[j];
        }

        output_binding_indexes_.push_back(bindingIndex);
        output_dim_sizes_.push_back(dim_size);
    }

    num_inputs_ = graph_input_names_.size();
    num_outputs_ = output_binding_indexes_.size();
    ORT_ENFORCE(engine_->getNbBindings() == (num_inputs_ + num_outputs_));
}

Status TRTKernel::Compute(OpKernelContext* context) const{
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    void* buffers[num_inputs_ + num_outputs_];
    std::vector<int> batch_size(num_inputs_, 1);

    // Get batch size and allocate cuda memory for inputs
    for (int i = 0, end = num_inputs_; i < end; ++i){
        const auto& tensor_input = context->Input<Tensor>(graph_input_indexes_[i]);
        auto& tensor_shape = tensor_input->Shape();
        auto& dim = tensor_shape.GetDims();
        batch_size.push_back(dim[0]);

        const auto& tensor_type = tensor_input->DataType();
        if (tensor_type == DataTypeImpl::GetType<float>()){
            const float* input = tensor_input->template Data<float>();
            CHECK(cudaMalloc(&buffers[input_binding_indexes_[i]], batch_size[i] * input_dim_sizes_[i] * sizeof(float)));
            CHECK(cudaMemcpy(buffers[input_binding_indexes_[i]], input, batch_size[i] * input_dim_sizes_[i] * sizeof(float), cudaMemcpyHostToDevice));
        }
        else{
            ORT_THROW("Invalid data type for TensorRT of ", tensor_type);
        }
    }

    // Allocate cuda memory for outputs
    for (int i = 0, end = num_outputs_; i < end; ++i){
        CHECK(cudaMalloc(&buffers[output_binding_indexes_[i]], batch_size[0] * output_dim_sizes_[i] * sizeof(float)));
    }

    // Run trt inference
    tensorrt_context_->enqueue(batch_size[0], &buffers[0], stream, nullptr);

    // Copy trt output
    std::vector<float*> output;
    for (int i = 0, end = num_outputs_; i < end; ++i){
        onnxruntime::TensorShape shape(vector<int64_t>(output_shapes_[i].begin(), output_shapes_[i].end()));
        const auto& tensor_output = context->Output(i, shape);
        output.push_back(tensor_output->template MutableData<float>());
    }

    for (int i = 0, end = num_outputs_; i < end; ++i){
        CHECK(cudaMemcpy(output[i], buffers[output_binding_indexes_[i]], batch_size[0] * output_dim_sizes_[i] * sizeof(float), cudaMemcpyDeviceToHost));
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    for (int i = 0, end = num_inputs_; i < end; ++i){
        CHECK(cudaFree(buffers[input_binding_indexes_[i]]));
    }

    for (int i = 0, end = num_outputs_; i < end; ++i){
        CHECK(cudaFree(buffers[output_binding_indexes_[i]]));
    }

    return Status::OK();
}
}


