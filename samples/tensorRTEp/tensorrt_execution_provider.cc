#include "tensorrt_execution_provider.h"
#include <memory>
#include <cuda_runtime.h>
namespace onnxruntime {

TensorrtExecutionProvider::TensorrtExecutionProvider(const char* ep_type, const ProviderOptions& ep_info) : OrtExecutionProvider() {
    OrtExecutionProvider::GetCapability = [](const OrtExecutionProvider* this_, const OrtGraphViewer* graph, size_t* cnt, OrtIndexedSubGraph*** indexed_sub_graph) {
    };

    OrtExecutionProvider::Compile = [](OrtExecutionProvider* this_, const OrtGraphViewer** graph, const OrtNode** node, size_t cnt, OrtNodeComputeInfo** node_compute_info) {
    };

    OrtExecutionProvider::CanCopy = [](const OrtDevice* source, const OrtDevice* target) {
      const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
      OrtMemoryInfoDeviceType source_device_type, target_device_type;
      api->DeviceGetDeviceType(source, &source_device_type);
      api->DeviceGetDeviceType(target, &target_device_type);
      OrtMemoryType source_mem_type, target_mem_type;
      api->DeviceGetMemoryType(source, &source_mem_type);
      api->DeviceGetMemoryType(target, &target_mem_type);

      return source_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU ||
             source_mem_type == OrtMemoryType::OrtMemoryType_CUDA_PINNED ||
             target_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU ||
             target_mem_type == OrtMemoryType::OrtMemoryType_CUDA_PINNED;
    };

    OrtExecutionProvider::CopyTensor = [](const void* src, OrtMemoryInfoDeviceType source_device_type, OrtMemoryType source_mem_type, void* dst, OrtMemoryInfoDeviceType target_device_type, size_t count, void* stream) -> OrtStatusPtr {
        // TODO(leca): convert cudaError_t to OrtStatusPtr
        if (source_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU && target_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU) {
            if (src != dst) {
                stream ? cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)) : cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
            }
            return nullptr;
        }
        if (source_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU && target_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU) {
            if (stream) cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
            else {
                cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
                cudaStreamSynchronize(nullptr);
            }
            return nullptr;
        }
        if (source_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU && target_device_type == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU) {
            if (stream) cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
            else {
                cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
                cudaStreamSynchronize(nullptr);
            }
            return nullptr;
        }
        if (stream && source_mem_type == OrtMemoryType::OrtMemoryType_CUDA_PINNED) {
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        }
        memcpy(dst, src, count);
        return nullptr;
    };

    type = ep_type;
    create_stream = new OrtCreateStream();
    create_stream->CreateStreamFunc = [](const OrtDevice* device) -> void* {
        cudaStream_t stream = nullptr;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        return stream;
    };

    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    api->CreateDevice(OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU, OrtMemoryType::OrtMemoryType_Default, 0, &default_device);
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
