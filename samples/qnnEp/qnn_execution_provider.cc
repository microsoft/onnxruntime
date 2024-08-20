#include "qnn_execution_provider.h"
#include "builder/qnn_def.h"
#include <memory>
#include <vector>
namespace onnxruntime {

static void ParseProfilingLevel(std::string profiling_level_string,
                                qnn::ProfilingLevel& profiling_level) {
  std::transform(profiling_level_string.begin(),
                 profiling_level_string.end(),
                 profiling_level_string.begin(),
                 [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });
//  LOGS_DEFAULT(INFO) << "profiling_level: " << profiling_level_string;
  if (profiling_level_string == "off") {
    profiling_level = qnn::ProfilingLevel::OFF;
  } else if (profiling_level_string == "basic") {
    profiling_level = qnn::ProfilingLevel::BASIC;
  } else if (profiling_level_string == "detailed") {
    profiling_level = qnn::ProfilingLevel::DETAILED;
  } else {
//    LOGS_DEFAULT(WARNING) << "Profiling level not valid.";
  }
}

QNNExecutionProvider::QNNExecutionProvider(const char* ep_type, const ProviderOptions& ep_info) : OrtExecutionProvider() {
    type = ep_type;
    OrtExecutionProvider::GetCapability = [](const OrtExecutionProvider* this_, const OrtGraphViewer* graph, size_t* cnt, OrtIndexedSubGraph*** indexed_sub_graph) {
    };

    OrtExecutionProvider::Compile = [](OrtExecutionProvider* this_, const OrtGraphViewer** graph, const OrtNode** node, size_t cnt, OrtNodeComputeInfo** node_compute_info) -> OrtStatusPtr {
      return nullptr;
    };
}

QNNExecutionProviderFactory::QNNExecutionProviderFactory() {
    OrtExecutionProviderFactory::CreateExecutionProvider = [](OrtExecutionProviderFactory* this_, const char* const* ep_option_keys, const char* const* ep_option_values, size_t option_size) -> OrtExecutionProvider* {
        ProviderOptions options;
        for (size_t i = 0; i < option_size; i++) options[ep_option_keys[i]] = ep_option_values[i];
        std::unique_ptr<QNNExecutionProvider> ret = std::make_unique<QNNExecutionProvider>("QNNExecutionProvider", std::move(options));
        return ret.release();
    };
}

}   // namespace onnxruntime

#ifdef __cplusplus
extern "C" {
#endif
OrtExecutionProviderFactory* RegisterCustomEp() {
    std::unique_ptr<onnxruntime::QNNExecutionProviderFactory> ret = std::make_unique<onnxruntime::QNNExecutionProviderFactory>();
    return ret.release();
}
#ifdef __cplusplus
}
#endif
