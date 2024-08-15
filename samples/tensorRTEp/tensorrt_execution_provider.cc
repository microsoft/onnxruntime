#include "tensorrt_execution_provider.h"
#include <memory>
#include <cuda_runtime.h>
namespace onnxruntime {

TensorrtExecutionProvider::TensorrtExecutionProvider(const char* ep_type, const ProviderOptions& ep_info) : OrtExecutionProvider() {
    type = ep_type;
    create_stream = new OrtCreateStream();
    create_stream->CreateStreamFunc = [](const OrtDevice* device) -> void* {
        cudaStream_t stream = nullptr;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        return stream;
    };
    OrtExecutionProvider::GetCapability = [](const OrtExecutionProvider* this_, const OrtGraphViewer* graph, size_t* cnt, OrtIndexedSubGraph*** indexed_sub_graph) {
    };

    OrtExecutionProvider::Compile = [](OrtExecutionProvider* this_, const OrtGraphViewer** graph, const OrtNode** node, size_t cnt, OrtNodeComputeInfo** node_compute_info) {
    };
}

TensorrtExecutionProviderFactory::TensorrtExecutionProviderFactory() {
    OrtExecutionProviderFactory::CreateExecutionProvider = [](OrtExecutionProviderFactory* this_, const char* const* ep_option_keys, const char* const* ep_option_values, size_t option_size) -> OrtExecutionProvider* {
        ProviderOptions options;
        for (size_t i = 0; i < option_size; i++) options[ep_option_keys[i]] = ep_option_values[i];
        std::unique_ptr<TensorrtExecutionProvider> ret = std::make_unique<TensorrtExecutionProvider>("TensorrtExecutionProvider", std::move(options));
        return ret.release();
    };
}

}   // namespace onnxruntime

#ifdef __cplusplus
extern "C" {
#endif
OrtExecutionProviderFactory* RegisterCustomEp() {
    std::unique_ptr<onnxruntime::TensorrtExecutionProviderFactory> ret = std::make_unique<onnxruntime::TensorrtExecutionProviderFactory>();
    return ret.release();
}
#ifdef __cplusplus
}
#endif
