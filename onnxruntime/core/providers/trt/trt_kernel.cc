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

namespace onnxruntime
{
void printErrorNodes(std::shared_ptr<nvonnxparser::IParser> trt_parser, ::ONNX_NAMESPACE::ModelProto onnx_model)
{
    int nerror = trt_parser->getNbErrors();
    for( int i=0; i<nerror; ++i )
    {
        nvonnxparser::IParserError const* error = trt_parser->getError(i);
        if( error->node() != -1 )
        {
            ::ONNX_NAMESPACE::NodeProto const& node = onnx_model.graph().node(error->node());
            std::cout << "While parsing node number " << error->node() << " [" << node.op_type();
            if( node.output().size() )
            {
                std::cout << " -> \"" << node.output(0) << "\"";
            }
            std::cout << "]:" << std::endl;
            std::cout << node.op_type() << std::endl;
        }
        std::cout << "ERROR: " << error->file() << ":" << error->line()
             << " In function " << error->func() << ":\n"
             << "[" << static_cast<int>(error->code()) << "] " << error->desc()
             << std::endl;
    }
}

TRTKernel::TRTKernel(const OpKernelInfo& info) : OpKernel(info)
{
    onnxruntime::Model model("");
    auto& graph = model.MainGraph();

    auto& node = info.node();
    if (node.NodeType() == onnxruntime::Node::Type::Primitive)
    {
        //This is a primitive node. Refer to an op directly
        std::vector<onnxruntime::NodeArg*> inputs, outputs;
        for (auto input : node.InputDefs())
        {
            auto& n_input = graph.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
            inputs.push_back(&n_input);
        }
        for (auto output : node.OutputDefs())
        {
            auto& n_output = graph.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
            outputs.push_back(&n_output);
        }
        graph.AddNode(node.Name(), node.OpType(), node.Description(), inputs, outputs, &node.GetAttributes(), node.Domain());

        ORT_ENFORCE(graph.Resolve().IsOK());
    }
    else if (node.NodeType() == onnxruntime::Node::Type::Fused)
    {
        //This is a fused node. Refer to a function (sub-graph)
        const Graph& graph_body = node.GetFunctionBody()->Body();
        for (auto& graph_body_node : graph_body.Nodes())
        {
            graph.AddNode(graph_body_node);
        }
        ORT_ENFORCE(graph.Resolve().IsOK());

        //Add inputs to graph
        for (int i = 0; i < node.InputDefs().size(); ++i)
        {
            const onnxruntime::Tensor* temp = nullptr;
            const onnxruntime::Tensor** constant_input_value = &temp;
            auto stat = info.TryGetConstantInput(i, constant_input_value);
            if (stat) //initialized inputs
            {
                auto& input_tensor = *constant_input_value;
                auto tensor_type = input_tensor->DataType();
                TensorProto::DataType dtype = TensorProto_DataType_UNDEFINED;
                TensorProto tensor_proto;
                if (tensor_type == DataTypeImpl::GetType<float>())
                {
                    dtype = TensorProto_DataType_FLOAT;
                    auto input_data = input_tensor->Data<float>();
                    for (int j = 0; j < input_tensor->Shape().Size(); ++j)
                    {
                        tensor_proto.add_float_data(input_data[j]);
                    }
                }
                else if (tensor_type == DataTypeImpl::GetType<int64_t>())
                {
                    dtype = TensorProto_DataType_INT64;
                    auto input_data = input_tensor->Data<int64_t>();
                    for (int j = 0; j < input_tensor->Shape().Size(); ++j)
                    {
                        tensor_proto.add_int64_data(input_data[j]);
                    }
                }
                else  //TODO: add other tensor types
                {
                    dtype = TensorProto_DataType_UNDEFINED;
                }
                tensor_proto.set_name(node.InputDefs()[i]->Name());
                auto& shape = input_tensor->Shape().GetDims();
                for (auto& dim : shape)
                {
                    tensor_proto.add_dims(dim);
                }
                tensor_proto.set_data_type(dtype);
                graph.AddInitializedTensor(tensor_proto);
            }
            else  //graph inputs
            {
                graph_input_index_.push_back(i);
                graph_input_name_.push_back(node.InputDefs()[i]->Name());
            }
        }
    }

    ONNX_NAMESPACE::ModelProto model_p = model.ToProto();

    //Add node's outputs to graphproto's outputs if the node's EdgeEnd nodes are not in the graph
    std::set<string> output_set;
    for (int i = 0; i < model_p.graph().output().size(); ++i)
    {
        output_set.insert(model_p.graph().output()[i].name());
    }

    std::vector<int> output_to_add;
    for (int i = 0; i < node.OutputDefs().size(); ++i)
    {
        string output_name = node.OutputDefs()[i]->Name();
        if (output_set.find(output_name) == output_set.end())
        {
            for (int j = 0; j < model_p.graph().value_info().size(); ++j)
            {
                if (output_name == model_p.graph().value_info()[j].name())
                {
                    output_to_add.push_back(j);
                }
            }
        }
    }
    for (auto& i: output_to_add)
    {
        *(model_p.mutable_graph()->mutable_output()->Add()) = model_p.graph().value_info()[i];
    }

    //Create trt engine
    string string_buf;
    model_p.SerializeToString(&string_buf);
    std::vector<char> onnx_buf(string_buf.begin(), string_buf.end());

    int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
    TRT_Logger trt_logger((nvinfer1::ILogger::Severity)verbosity);
    auto trt_builder = infer_object(nvinfer1::createInferBuilder(trt_logger));
    auto trt_network = infer_object(trt_builder->createNetwork());
    auto trt_parser  = infer_object(nvonnxparser::createParser(trt_network.get(), trt_logger));
    if( !trt_parser->parse(onnx_buf.data(), onnx_buf.size()))
    {
        printErrorNodes(trt_parser, model_p);
    }
    trt_builder->setMaxBatchSize(max_batch_size);
    trt_builder->setMaxWorkspaceSize(max_workspace_size);
    engine_ = infer_object(trt_builder->buildCudaEngine(*trt_network.get()));
    assert(engine_ != nullptr);

    //Build trt context
    tensorrt_context_ = engine_->createExecutionContext();
    assert(tensorrt_context_ != nullptr);

    //Get input shape and binding index
    input_dimension_.resize(trt_network->getNbInputs());
    for (int i = 0; i < trt_network->getNbInputs(); i++)
    {
        std::string name = trt_network->getInput(i)->getName();
        size_t bindingIndex = engine_->getBindingIndex(name.c_str());

        nvinfer1::Dims4 dimensions = static_cast<nvinfer1::Dims4&&>(engine_->getBindingDimensions((int)bindingIndex));
        size_t eltCount = kBatchSize;
        input_dimension_[i].push_back(kBatchSize);

        for (int j = 0; j < dimensions.nbDims; ++j)
        {
            input_dimension_[i].push_back(dimensions.d[j]);
            eltCount *= dimensions.d[j];
        }

        input_binding_index_.push_back(bindingIndex);
        input_dim_size_.push_back(eltCount);
    }

    //Get output shape and binding index
    output_dimension_.resize(trt_network->getNbOutputs());
    for (int i = 0; i < trt_network->getNbOutputs(); i++)
    {
        std::string name = trt_network->getOutput(i)->getName();
        size_t bindingIndex = engine_->getBindingIndex(name.c_str());

        nvinfer1::Dims4 dimensions = static_cast<nvinfer1::Dims4&&>(engine_->getBindingDimensions((int)bindingIndex));
        size_t eltCount = kBatchSize;
        output_dimension_[i].push_back(kBatchSize);

        for (int j = 0; j < dimensions.nbDims; ++j)
        {
            output_dimension_[i].push_back(dimensions.d[j]);
            eltCount *= dimensions.d[j];
        }

        output_binding_index_.push_back(bindingIndex);
        output_dim_size_.push_back(eltCount);
    }

    int input_number = graph_input_name_.size();
    int output_number = output_binding_index_.size();
    assert(engine_->getNbBindings() == (input_number + output_number));
}

Status TRTKernel::Compute(OpKernelContext* context) const
{
    int input_number = graph_input_name_.size();
    int output_number = output_binding_index_.size();
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    void* buffers[input_number + output_number];
    std::vector<int> batch_size(input_number, 1);

    //Get batch size and allocate cuda memory for inputs
    for (auto i = 0; i < input_number; ++i)
    {
        auto tensor_input = context->Input<Tensor>(graph_input_index_[i]);
        auto& tensor_shape = tensor_input->Shape();
        auto& dim = tensor_shape.GetDims();
        batch_size.push_back(dim[0]);

        auto tensor_type = tensor_input->DataType();
        if (tensor_type == DataTypeImpl::GetType<float>())
        {
            const float* input = tensor_input->template Data<float>();
            CHECK(cudaMalloc(&buffers[input_binding_index_[i]], batch_size[i] * input_dim_size_[i] * sizeof(float)));
            CHECK(cudaMemcpy(buffers[input_binding_index_[i]], input, batch_size[i] * input_dim_size_[i] * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    //Allocate cuda memory for outputs
    for (int i = 0; i < output_number; ++i)
    {
        CHECK(cudaMalloc(&buffers[output_binding_index_[i]], batch_size[0] * output_dim_size_[i] * sizeof(float)));
    }

    //Run trt inference
    tensorrt_context_->enqueue(batch_size[0], &buffers[0], stream, nullptr);

    //Copy trt output
    std::vector<float*> output;
    for (int i = 0; i < output_number; ++i)
    {
        onnxruntime::TensorShape shape(vector<int64_t>(output_dimension_[i].begin(), output_dimension_[i].end()));
        auto tensor_output = context->Output(i, shape);
        output.push_back(tensor_output->template MutableData<float>());
    }
    for (int i = 0; i < output_number; ++i)
    {
        CHECK(cudaMemcpy(output[i], buffers[output_binding_index_[i]], batch_size[0] * output_dim_size_[i] * sizeof(float), cudaMemcpyDeviceToHost));
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    for (int i = 0; i < input_number; ++i)
    {
        CHECK(cudaFree(buffers[input_binding_index_[i]]));
    }
    for (int i = 0; i < output_number; ++i)
    {
        CHECK(cudaFree(buffers[output_binding_index_[i]]));
    }

    return Status::OK();
}
}

