#include "out_tree_ep.h"
#include <memory>
#include <vector>
namespace onnxruntime {

OutTreeEp::OutTreeEp(const char* ep_type, const OutTreeEpInfo& ep_info) : info(ep_info) {
    type = ep_type;
    OrtExecutionProvider::GetCapability = [](const OrtExecutionProvider* this_, const OrtGraphViewer* graph, size_t* cnt, OrtIndexedSubGraph*** indexed_sub_graph) {
        std::vector<OrtIndexedSubGraph*> cache;
        size_t nodes_count = 0;
        const size_t* nodes_index = OrtGraph_GetNodesIndexInTopologicalOrder(graph, &nodes_count);
        for (size_t i = 0; i < nodes_count; i++) {
            const OrtNode* node = OrtGraph_GetOrtNode(graph, nodes_index[i]);
            if (OrtNode_GetOpType(node) == "Relu") {
                OrtIndexedSubGraph* subgraph = new OrtIndexedSubGraph();
                subgraph->node_index_len = 1;
                subgraph->node_index = new size_t [subgraph->node_index_len];
                subgraph->node_index[0] = nodes_index[0];

                subgraph->meta_def = new OrtMetaDef();
                subgraph->meta_def->name = "Relu_subgraph";
                subgraph->meta_def->input_len = OrtNode_GetInputSize(node);
                subgraph->meta_def->inputs = new const char* [subgraph->meta_def->input_len];
                for (int j = 0; j < subgraph->meta_def->input_len; j++) subgraph->meta_def->inputs[j] = OrtNode_GetIthInputName(node, j);

                subgraph->meta_def->output_len = OrtNode_GetOutputSize(node);
                subgraph->meta_def->outputs = new const char* [subgraph->meta_def->output_len];
                for (int j = 0; j < subgraph->meta_def->output_len; j++) subgraph->meta_def->outputs[j] = OrtNode_GetIthOutputName(node, j);

                cache.push_back(subgraph);
            }
        }

        *cnt = cache.size();
        *indexed_sub_graph = new OrtIndexedSubGraph* [*cnt];
        for (size_t i = 0; i < *cnt; i++) {
            (*indexed_sub_graph)[i] = cache[i];
        }
    };

    OrtExecutionProvider::Compile = [](OrtExecutionProvider* this_, const OrtGraphViewer** graph, const OrtNode** node, size_t cnt, OrtNodeComputeInfo*** node_compute_info) {
        for (size_t i = 0; i < cnt; i++) {
            (*node_compute_info)[i]->ComputeFunc = [](void* state, const OrtApi* api, OrtKernelContext* context) ->OrtStatusPtr {
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
                }

                return nullptr;
            };
        }
    };
}

OutTreeEpFactory::OutTreeEpFactory() {
    OrtExecutionProviderFactory::CreateExecutionProvider = [](OrtExecutionProviderFactory* this_, const char* const* ep_option_keys, const char* const* ep_option_values, size_t option_size) -> void* {
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
