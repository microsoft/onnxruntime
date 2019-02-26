// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "trt_execution_provider.h"
#include "trt_allocator.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/platform/env.h"
#include "onnx/shape_inference/implementation.h"
#include "cuda_runtime_api.h"
#include "gsl/pointers"
#include "core/graph/model.h"
#include "cuda_runtime_api.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime{

GraphProto ToGraphProto(const GraphViewer& graph){

    GraphProto graph_proto;
    graph_proto.clear_node();

    // Nodes is sorted in Topological Order in the GraphProto
    for (auto& node_idx : graph.GetNodesInTopologicalOrder()){
        const Node* p_node = graph.GetNode(node_idx);
        NodeProto* node_proto{graph_proto.add_node()};
        p_node->ToProto(*node_proto);
    }

    graph_proto.clear_input();
    graph_proto.clear_output();
    graph_proto.clear_value_info();

    for (const auto* input_arg : graph.GetInputsIncludingInitializers()){
        *(graph_proto.mutable_input()->Add()) = input_arg->ToProto();
    }

    for (const auto* output_arg : graph.GetOutputs()){
        *(graph_proto.mutable_output()->Add()) = output_arg->ToProto();
    }

    for (const auto* value_info : graph.GetValueInfo()){
        *(graph_proto.mutable_value_info()->Add()) = value_info->ToProto();
    }

    return graph_proto;
}

TRTExecutionProvider::TRTExecutionProvider()
    : IExecutionProvider{onnxruntime::kTRTExecutionProvider}{
    DeviceAllocatorRegistrationInfo trt_device_info({OrtMemTypeCPU, [](int){
        return std::make_unique<TRTPinnedAllocator>();
    }, std::numeric_limits<size_t>::max()});
    InsertAllocator(CreateAllocator(trt_device_info));
    DeviceAllocatorRegistrationInfo default_device_info({OrtMemTypeDefault, [](int){
        return std::make_unique<TRTAllocator>();
    }, std::numeric_limits<size_t>::max()});
    InsertAllocator(CreateAllocator(default_device_info));
}

TRTExecutionProvider::~TRTExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
        TRTExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const{
    // Construct modelproto from graph
    onnxruntime::Model model(graph.Name(), true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), graph.DomainToVersionMap());
    onnxruntime::Graph& graph_build = model.MainGraph();
    const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
    for (auto& index: node_index){
        const Node* node = graph.GetNode(index);
        graph_build.AddNode(*node);
    }
    ORT_ENFORCE(graph_build.Resolve().IsOK());
    ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
    model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

    // Serialize modelproto to string
    string string_buf;
    model_proto.SerializeToString(&string_buf);
    std::vector<char> onnx_buf(string_buf.begin(), string_buf.end());

    // Get supported node list
    SubGraphCollection_t supported_nodes_vector;
    TRTLogger trt_logger(static_cast<nvinfer1::ILogger::Severity>(nvinfer1::ILogger::Severity::kWARNING));
    const auto& trt_builder = InferObject(nvinfer1::createInferBuilder(trt_logger));
    const auto& trt_network = InferObject(trt_builder->createNetwork());
    const auto& trt_parser  = InferObject(nvonnxparser::createParser(trt_network.get(), trt_logger));
    trt_parser->supportsModel(onnx_buf.data(), onnx_buf.size(), supported_nodes_vector);
    model_proto.release_graph();

    // Find inputs, initializers and outputs for each supported subgraph
    std::vector<std::unique_ptr<ComputeCapability>> result;
    for (const auto& group : supported_nodes_vector){
        std::set<size_t> node_set(group.begin(), group.end());
        std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();

        if (!group.empty()){
            // Find inputs and outputs of the subgraph
            std::map<const NodeArg*, int> fused_inputs, fused_outputs, fused_outputs_to_add;
            std::set<const NodeArg*> erased;
            int input_order = 0;
            int output_order = 0;

            for (const auto& index : group){
                sub_graph->nodes.push_back(index);
                const auto& node = graph.GetNode(index);

                for (const auto& input : node->InputDefs()){
                    const auto& it = fused_outputs.find(input);

                    if (it != fused_outputs.end()){
                        fused_outputs.erase(it);
                        erased.insert(input);
                    }
                    //only when input is neither in output list nor erased list, add the input to input list
                    else if (erased.find(input) == erased.end()){
                        fused_inputs[input] = input_order++;
                    }
                }

                // For output searching, there is a special case:
                // If node's OutputEdges are more than its outputs, meaning certain output is used more than once,
                // if the output is connected to nodes that don't belong to the subgraph, the output need to be added
                // to the output list
                if (node->GetOutputEdgesCount() > node->OutputDefs().size()){
                    for (auto it = node->OutputEdgesBegin(); it != node->OutputEdgesEnd(); ++it){
                        const auto& node_index = it->GetNode().Index();
                        const auto& output = (it->GetNode()).InputDefs()[it->GetDstArgIndex()];

                        if (node_set.find(node_index) != node_set.end()){
                            const auto& iter = fused_inputs.find(output);

                            if (iter != fused_inputs.end()){
                                fused_inputs.erase(iter);
                                erased.insert(output);
                            }
                            else if (erased.find(output) == erased.end()){
                                fused_outputs[output] = output_order++;
                            }
                        }
                        else{
                            fused_outputs_to_add[output] = output_order++;
                        }

                    }
                }
                else{
                    for (const auto& output : node->OutputDefs()){
                        const auto& it = fused_inputs.find(output);

                        if (it != fused_inputs.end()){
                            fused_inputs.erase(it);
                            erased.insert(output);
                        }
                        // only when output is neither in input list nor erased list, add the output to output list
                        else if (erased.find(output) == erased.end()){
                            fused_outputs[output] = output_order++;
                        }
                    }
                }
            }

            fused_outputs.insert(fused_outputs_to_add.begin(), fused_outputs_to_add.end());

            // Sort inputs and outputs by the order they were added
            std::multimap<int, const NodeArg*> inputs, outputs;

            for (auto it = fused_inputs.begin(); it != fused_inputs.end(); ++it){
                inputs.insert(std::pair<int, const NodeArg*>(it->second,it->first));
            }

            for (auto it = fused_outputs.begin(); it != fused_outputs.end(); ++it){
                outputs.insert(std::pair<int, const NodeArg*>(it->second,it->first));
            }

            // Assign inputs and outputs to subgraph's meta_def
            auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
            meta_def->name = "TRTKernel";
            meta_def->domain = kMSDomain;

            for (auto& input : inputs){
                meta_def->inputs.push_back(input.second->Name());
            }

            for (auto& output : outputs){
                meta_def->outputs.push_back(output.second->Name());
            }

            meta_def->since_version = 1;
            sub_graph->SetMetaDef(meta_def);
        }

        result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }

    return result;
}

std::shared_ptr<KernelRegistry> TRTExecutionProvider::GetKernelRegistry() const{
    return nullptr;
}

common::Status TRTExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const{
    ORT_UNUSED_PARAMETER(src);
    ORT_UNUSED_PARAMETER(dst);
    return Status::OK();
}

common::Status TRTExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
        std::vector<NodeComputeInfo>& node_compute_funcs){

    for (auto* fused_node : fused_nodes){
        std::unordered_map<std::string, int> input_map;
        std::vector<int> input_indexes;
        std::vector<int> input_binding_indexes;
        std::vector<int> input_dim_sizes;
        std::vector<int> output_binding_indexes;
        std::vector<int> output_dim_sizes;
        std::vector<std::vector<int64_t>> output_shapes;
        std::vector<std::string> graph_input_name;
        std::vector<std::vector<int>> input_shape;

        // Build map from input name to its index in InputDefs
        for (int i = 0, end = fused_node->InputDefs().size(); i < end; ++i){
            input_map[fused_node->InputDefs()[i]->Name()] = i;
        }

        auto func_body = fused_node->GetFunctionBody();
        if (!func_body){
            return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Function body is empty");
        }

        // Reconstruct graph from fused node's function body
        const Graph& graph_body = fused_node->GetFunctionBody()->Body();
        onnxruntime::Model model(graph_body.Name(), true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), graph_body.DomainToVersionMap());
        onnxruntime::Graph& graph = model.MainGraph();
        
        for (auto& graph_body_node : graph_body.Nodes()){
            graph.AddNode(graph_body_node);
        }

        ORT_ENFORCE(graph.Resolve().IsOK());

        // Add initializer to graph
        auto& init_tensors = graph_body.GetAllInitializedTensors();
        for (auto& tensor : init_tensors){
            graph.AddInitializedTensor(*(tensor.second));
        }

        // Find graph's inputs in input map
        auto& inputs_tensors = graph_body.GetInputs();
        for (int i = 0, end = inputs_tensors.size(); i < end; ++i){
            auto iter = input_map.find(inputs_tensors[i]->Name());
            if (iter != input_map.end()){
                input_indexes.push_back(iter->second);
                graph_input_name.push_back(iter->first);
            }
        }

        // Add fused node's outputs to graph's outputs if the outputs are not included yet
        // for the case that node's output is connected to more than one EdgeEnd nodes and some of them don't belong to the graph
        ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
        std::set<string> output_set;
        for (int i = 0, end = model_proto.graph().output().size(); i < end; ++i){
            output_set.insert(model_proto.graph().output()[i].name());
        }

        std::vector<int> output_to_add;
        for (int i = 0, end = fused_node->OutputDefs().size(); i < end; ++i){
            const std::string& output_name = fused_node->OutputDefs()[i]->Name();
            if (output_set.find(output_name) == output_set.end()){
                for (int j = 0, end = model_proto.graph().value_info().size(); j < end; ++j){
                    if (output_name == model_proto.graph().value_info()[j].name()){
                        output_to_add.push_back(j);
                    }
                }
            }
        }

        for (auto& i: output_to_add){
            *(model_proto.mutable_graph()->mutable_output()->Add()) = model_proto.graph().value_info()[i];
        }

        // Set version
        model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

        // Create TensorRT engine
        string string_buf;
        model_proto.SerializeToString(&string_buf);
        std::vector<char> onnx_buf(string_buf.begin(), string_buf.end());

        TRTLogger trt_logger(static_cast<nvinfer1::ILogger::Severity>(nvinfer1::ILogger::Severity::kWARNING));
        const auto& trt_builder = InferObject(nvinfer1::createInferBuilder(trt_logger));
        const auto& trt_network = InferObject(trt_builder->createNetwork());
        const auto& trt_parser  = InferObject(nvonnxparser::createParser(trt_network.get(), trt_logger));
        trt_parser->parse(onnx_buf.data(), onnx_buf.size());
        trt_builder->setMaxBatchSize(max_batch_size);
        trt_builder->setMaxWorkspaceSize(max_workspace_size);
        const auto& trt_engine = InferObject(trt_builder->buildCudaEngine(*trt_network.get()));
        ORT_ENFORCE(trt_engine != nullptr);

        // Build TensorRT context
        const auto& trt_context = InferObject(trt_engine->createExecutionContext());
        ORT_ENFORCE(trt_context != nullptr);

        // Get input shape and binding index
        input_shape.resize(trt_network->getNbInputs());
        for (int i = 0, end = trt_network->getNbInputs(); i < end; ++i){
            const std::string& name = trt_network->getInput(i)->getName();
            size_t bindingIndex = trt_engine->getBindingIndex(name.c_str());

            nvinfer1::Dims4 dimensions = static_cast<nvinfer1::Dims4&&>(trt_engine->getBindingDimensions(static_cast<int>(bindingIndex)));
            size_t dim_size = 1;
            for (int j = 0, end = dimensions.nbDims; j < end; ++j){
                input_shape[i].push_back(dimensions.d[j]);
                dim_size *= dimensions.d[j];
            }

            input_binding_indexes.push_back(bindingIndex);
            input_dim_sizes.push_back(dim_size);
        }

        // Get output shape and binding index
        output_shapes.resize(trt_network->getNbOutputs());
        for (int i = 0, end = trt_network->getNbOutputs(); i < end; ++i){
            const std::string& name = trt_network->getOutput(i)->getName();
            size_t bindingIndex = trt_engine->getBindingIndex(name.c_str());

            nvinfer1::Dims4 dimensions = static_cast<nvinfer1::Dims4&&>(trt_engine->getBindingDimensions(static_cast<int>(bindingIndex)));
            size_t dim_size = 1;
            for (int j = 0, end = dimensions.nbDims; j < end; ++j){
                output_shapes[i].push_back(dimensions.d[j]);
                dim_size *= dimensions.d[j];
            }

            output_binding_indexes.push_back(bindingIndex);
            output_dim_sizes.push_back(dim_size);
        }

        int num_inputs = input_binding_indexes.size();
        int num_outputs = output_binding_indexes.size();
        ORT_ENFORCE(trt_engine->getNbBindings() == (num_inputs + num_outputs));

        // Save engine, context and input/output info to map
        parsers_[fused_node->Name()] = trt_parser;
        engines_[fused_node->Name()] = trt_engine;
        contexts_[fused_node->Name()] = trt_context;
        input_info_[fused_node->Name()].push_back(input_indexes);
        input_info_[fused_node->Name()].push_back(input_binding_indexes);
        input_info_[fused_node->Name()].push_back(input_dim_sizes);
        output_info_[fused_node->Name()].push_back(output_binding_indexes);
        output_info_[fused_node->Name()].push_back(output_dim_sizes);
        output_shapes_[fused_node->Name()] = output_shapes;

        // Create function state
        NodeComputeInfo compute_info;
        compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state){
            auto* p = new TRTFuncState();
            *p = {context->allocate_func, context->release_func, context->allocator_handle, parsers_[context->node_name].get(), engines_[context->node_name].get(), contexts_[context->node_name].get(),
                  input_info_[context->node_name], output_info_[context->node_name], output_shapes_[context->node_name]};
            *state = p;
            return 0;
        };

        // Release function state
        compute_info.release_state_func = [](FunctionState state){
            if (state)
                delete static_cast<TRTFuncState*>(state);
        };

        // Create compute function
        compute_info.compute_func = [](FunctionState state, ONNXRunTimeTensor* input_tensors, size_t num_inputs, ONNXRunTimeTensor* output_tensors, size_t num_outputs){
            ORT_UNUSED_PARAMETER(num_inputs);
            ORT_UNUSED_PARAMETER(num_outputs);
            TRTFuncState* trt_state = reinterpret_cast<TRTFuncState*>(state);
            std::vector<int> input_indexes = (trt_state->input_info)[0];
            std::vector<int> input_binding_indexes = (trt_state->input_info)[1];
            std::vector<int> input_dim_sizes = (trt_state-> input_info)[2];
            std::vector<int> output_binding_indexes = (trt_state->output_info)[0];
            std::vector<int> output_dim_sizes = (trt_state->output_info)[1];
            std::vector<std::vector<int64_t>> output_shapes = trt_state->output_shapes;
            int num_binding_inputs = input_binding_indexes.size();
            int num_binding_outputs =  output_binding_indexes.size();
            cudaStream_t stream;
            CHECK(cudaStreamCreate(&stream));
            std::vector<void*> buffers(num_binding_inputs + num_binding_outputs);
            std::vector<int> batch_size;

            // Get batch size and allocate cuda memory for inputs
            for (int i = 0, end = num_binding_inputs; i < end; ++i){
                const auto& tensor_input = input_tensors[input_indexes[i]];
                auto& tensor_shape = tensor_input.shape;
                batch_size.push_back(tensor_shape[0]);

                const float* input = static_cast<float*>(tensor_input.data);
                CHECK(cudaMalloc(&buffers[input_binding_indexes[i]], batch_size[i] * input_dim_sizes[i] * sizeof(float)));
                CHECK(cudaMemcpy(buffers[input_binding_indexes[i]], input, batch_size[i] * input_dim_sizes[i] * sizeof(float), cudaMemcpyHostToDevice));
            }

            // Allocate cuda memory for outputs
            for (int i = 0, end = num_binding_outputs; i < end; ++i){
                CHECK(cudaMalloc(&buffers[output_binding_indexes[i]], batch_size[0] * output_dim_sizes[i] * sizeof(float)));
            }

            // Run TRT inference
            trt_state->context->enqueue(batch_size[0], &buffers[0], stream, nullptr);

            // Copy TRT outputs to output tensors
            for (int i = 0, end = num_binding_outputs; i < end; ++i){
                // Setup output tensor property
                output_shapes[i].insert(output_shapes[i].begin(), batch_size[0]);
                output_tensors[i].dtype = input_tensors[0].dtype;
                // TODO: shape inference
                output_tensors[i].ndim = output_shapes[i].size();
                output_tensors[i].shape = new int64_t[output_tensors[i].ndim];
                memcpy(output_tensors[i].shape, &output_shapes[i][0], sizeof(int64_t) * output_tensors[i].ndim);
                output_tensors[i].data = (*(trt_state->test_allocate_func))(trt_state->allocator, sizeof(double) * batch_size[0] * output_dim_sizes[i], 64);

                CHECK(cudaMemcpy(output_tensors[i].data, buffers[output_binding_indexes[i]], batch_size[0] * output_dim_sizes[i] * sizeof(float), cudaMemcpyDeviceToHost));
            }

            // Sync stream
            cudaStreamSynchronize(stream);

            // Free CUDA memory
            cudaStreamDestroy(stream);

            for (int i = 0, end = num_binding_inputs; i < end; ++i){
                CHECK(cudaFree(buffers[input_binding_indexes[i]]));
            }

            for (int i = 0, end = num_binding_outputs; i < end; ++i){
                CHECK(cudaFree(buffers[output_binding_indexes[i]]));
            }

            return 0;
        };

        node_compute_funcs.push_back(compute_info);
    }

    return Status::OK();
}
}  // namespace onnxruntime


