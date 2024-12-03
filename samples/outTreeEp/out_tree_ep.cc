#include "out_tree_ep.h"
#include <memory>
#include <vector>
#include <iostream>
#include "core/session/onnxruntime_cxx_api_ep.h"
namespace onnxruntime {

OutTreeEp::OutTreeEp(const char* ep_type, const OutTreeEpInfo& ep_info) : OrtExecutionProvider(), info(ep_info) {
    type = ep_type;
    OrtExecutionProvider::GetCapability = [](const OrtExecutionProvider* this_, const OrtGraphViewer* graph, size_t* cnt, OrtIndexedSubGraph*** indexed_sub_graph) {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        const OrtGraphApi* ort_graph_api = api->GetGraphApi(ORT_API_VERSION);

        // Test Graph C++ API
        Ort::PluginEP::Graph graph_cxx(graph);
        std::cout<<"Test Graph C++ API Graph::GetName:"<<graph_cxx.GetName()<<"\n";
        std::vector<OrtIndexedSubGraph*> cache;
        const size_t* nodes_index = nullptr;
        size_t nodes_count = 0;
        ort_graph_api->OrtGraph_GetNodesIndexInTopologicalOrder(graph, 0, &nodes_index, &nodes_count);
        for (size_t i = 0; i < nodes_count; i++) {
            const OrtNode* node = nullptr;
            ort_graph_api->OrtGraph_GetOrtNode(graph, nodes_index[i], &node);
            Ort::PluginEP::Node node_cxx(node);
            std::cout<<"Test Graph C++ API Node::GetName:"<<node_cxx.GetName()<<"\n";
            const char* node_op_type = nullptr;
            ort_graph_api->OrtNode_GetOpType(node, &node_op_type);
            if (!strcmp(node_op_type, "Relu")) {
                OrtIndexedSubGraph* subgraph = new OrtIndexedSubGraph();
                subgraph->node_index_len = 1;
                subgraph->node_index = new size_t [subgraph->node_index_len];
                subgraph->node_index[0] = nodes_index[0];

                subgraph->meta_def = new OrtMetaDef();
                subgraph->meta_def->name = "Relu_subgraph";
                subgraph->meta_def->input_len = 0;
                ort_graph_api->OrtNode_GetNumInputs(node, &subgraph->meta_def->input_len);
                subgraph->meta_def->inputs = new char* [subgraph->meta_def->input_len];
                for (size_t j = 0; j < subgraph->meta_def->input_len; j++) {
                    const char* input_j = nullptr;
                    ort_graph_api->OrtNode_GetIthInputName(node, j, &input_j);
                    subgraph->meta_def->inputs[j] = const_cast<char*>(input_j);
                }

                ort_graph_api->OrtNode_GetNumOutputs(node, &subgraph->meta_def->output_len);
                subgraph->meta_def->outputs = new char* [subgraph->meta_def->output_len];
                for (size_t j = 0; j < subgraph->meta_def->output_len; j++) {
                    const char* output_j = nullptr;
                    ort_graph_api->OrtNode_GetIthOutputName(node, j, &output_j);
                    subgraph->meta_def->outputs[j] = const_cast<char*>(output_j);
                }

                cache.push_back(subgraph);
            }
        }

        *cnt = cache.size();
        *indexed_sub_graph = new OrtIndexedSubGraph* [*cnt];
        for (size_t i = 0; i < *cnt; i++) {
            (*indexed_sub_graph)[i] = cache[i];
        }
    };

    OrtExecutionProvider::Compile = [](OrtExecutionProvider* this_, const OrtGraphViewer** graph, const OrtNode** node, size_t cnt, OrtNodeComputeInfo* node_compute_info) -> OrtStatusPtr {
        OutTreeEp* p = static_cast<OutTreeEp*>(this_);
        this_->extra_param_for_compute_func = p;
        for (size_t i = 0; i < cnt; i++) {
            node_compute_info[i].ComputeFunc = [](void* state, void* extra_param, const OrtApi* api, OrtKernelContext* context) -> OrtStatusPtr {
                const OrtValue* input = nullptr;
                api->KernelContext_GetInput(context, 0, &input);
                std::vector<int64_t> dim(1,4);
                OrtValue* output = nullptr;
                api->KernelContext_GetOutput(context, 0, dim.data(), dim.size(), &output);

                float* input_raw = nullptr, *output_raw = nullptr;
                api->GetTensorMutableData(const_cast<OrtValue*>(input), reinterpret_cast<void**>(&input_raw));
                api->GetTensorMutableData(output, reinterpret_cast<void**>(&output_raw));

                for (int i = 0; i < 4; i++) {
                    output_raw[i] = input_raw[i];
                    if (input_raw[i] < 0) output_raw[i] = 0;

                    output_raw[i] = 1.0;
                }

                OutTreeEp* this_ = reinterpret_cast<OutTreeEp*>(extra_param);
                std::cout<<"int_property: "<<this_->info.int_property<<"\nstr_property: "<<this_->info.str_property<<"\n";
                return nullptr;
            };
        }
        return nullptr;
    };

    OrtExecutionProvider::ReleaseIndexedSubGraphs = [](OrtIndexedSubGraph** indexed_sub_graphs, size_t num_sub_graph) {
      if (indexed_sub_graphs == nullptr) return;
      for (size_t i = 0; i < num_sub_graph; i++) {
        OrtIndexedSubGraph* sub_graph = indexed_sub_graphs[i];
        delete[] sub_graph->node_index;
        delete sub_graph->meta_def;
        delete sub_graph;
      }
      delete[] indexed_sub_graphs;
    };
}

OutTreeEpFactory::OutTreeEpFactory() {
    OrtExecutionProviderFactory::CreateExecutionProvider = [](OrtExecutionProviderFactory* this_, const char* const* ep_option_keys, const char* const* ep_option_values, size_t option_size) -> OrtExecutionProvider* {
        OutTreeEpInfo info;
        for (size_t i = 0; i < option_size; i++) {
            if (!strcmp(ep_option_keys[i], "int_property")) info.int_property = std::atoi(ep_option_values[i]);
            else if (!strcmp(ep_option_keys[i], "str_property")) info.str_property = ep_option_values[i];
            // TODO(leca): else throw
        }
        std::unique_ptr<OutTreeEp> ret = std::make_unique<OutTreeEp>("outTreeEp", std::move(info));
        return ret.release();
    };
}

}

#ifdef __cplusplus
extern "C" {
#endif
OrtExecutionProviderFactory* RegisterCustomEp() {
    std::unique_ptr<onnxruntime::OutTreeEpFactory> ret = std::make_unique<onnxruntime::OutTreeEpFactory>();
    return ret.release();
}
#ifdef __cplusplus
}
#endif
