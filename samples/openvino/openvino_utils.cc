#if defined(_WIN32)
#include <windows.h>
#endif
#include "openvino_utils.h"

namespace onnxruntime {
    std::string GetEnvironmentVar(const std::string& var_name) {
#if defined(_WIN32)
        constexpr DWORD kBufferSize = 32767;

        // Create buffer to hold the result
        std::string buffer(kBufferSize, '\0');

        // The last argument is the size of the buffer pointed to by the lpBuffer parameter, including the null-terminating character, in characters.
        // If the function succeeds, the return value is the number of characters stored in the buffer pointed to by lpBuffer, not including the terminating null character.
        // Therefore, If the function succeeds, kBufferSize should be larger than char_count.
        auto char_count = GetEnvironmentVariableA(var_name.c_str(), buffer.data(), kBufferSize);

        if (kBufferSize > char_count) {
            buffer.resize(char_count);
            return buffer;
        }
#else
        char* val = getenv(var_name.c_str());
        return val == nullptr ? std::string() : std::string(val);
#endif
        return std::string();
    }

    OrtStatus* ForEachNodeDef(const OrtGraphApi* graph_api, const OrtGraphViewer* graph, const OrtNode* node,
                              std::function<void(const char*, const OrtValueInfoRef*, bool/*is_input*/)> func) {
        size_t input_count = 0;
        graph_api->OrtNode_GetNumInputs(node, &input_count);
        for (int i = 0; i < input_count; i++) {
            const char* input_name = nullptr;
            graph_api->OrtNode_GetIthInputName(node, i, &input_name);
            OrtValueInfoRef* value_info = nullptr;
            graph_api->OrtGraph_GetValueInfo(graph, input_name, &value_info);
            func(input_name, value_info, true);
            graph_api->OrtGraph_ReleaseValueInfo(value_info);
        }

        size_t implicit_input_count = 0;
        graph_api->OrtNode_GetImplicitInputSize(node, &implicit_input_count);
        for (int i = 0; i < implicit_input_count; i++) {
            const char* input_name = nullptr;
            graph_api->OrtNode_GetIthImplicitInputName(node, i, &input_name);
            OrtValueInfoRef* value_info = nullptr;
            graph_api->OrtGraph_GetValueInfo(graph, input_name, &value_info);
            func(input_name, value_info, true);
            graph_api->OrtGraph_ReleaseValueInfo(value_info);
        }

        size_t output_count = 0;
        graph_api->OrtNode_GetNumOutputs(node, &output_count);
        for (int i = 0; i < output_count; i++) {
            const char* output_name = nullptr;
            graph_api->OrtNode_GetIthOutputName(node, i, &output_name);
            OrtValueInfoRef* value_info = nullptr;
            graph_api->OrtGraph_GetValueInfo(graph, output_name, &value_info);
            func(output_name, value_info, false);
            graph_api->OrtGraph_ReleaseValueInfo(value_info);
        }
        return nullptr;
    }
}
