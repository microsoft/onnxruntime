#include "core/session/onnxruntime_c_api.h"
#include <vector>
#include <iostream>

inline void THROW_ON_ERROR(OrtStatus* status) {
    if (status != nullptr) abort();
}

void TestCompileBasedEp(const OrtApi* g_ort, OrtEnv* env, OrtSessionOptions* so) {
    THROW_ON_ERROR(g_ort->RegisterOrtExecutionProviderLibrary("/home/leca/code/onnxruntime/samples/outTreeEp/build/liboutTreeEp.so", env, "outTreeEp"));
    std::vector<const char*> keys{"int_property", "str_property"}, values{"3", "strvalue"};
    THROW_ON_ERROR(g_ort->SessionOptionsAppendOrtExecutionProvider(so, "outTreeEp", env, keys.data(), values.data(), keys.size()));
}

void TestKernelBasedEp(const OrtApi* g_ort, OrtEnv* env, OrtSessionOptions* so) {
    THROW_ON_ERROR(g_ort->RegisterOrtExecutionProviderLibrary("/home/leca/code/onnxruntime/samples/outTreeEp_kernel/build/libkernelEp.so", env, "kernelEp"));
    std::vector<const char*> keys{"int_property", "str_property"}, values{"3", "strvalue"};
    THROW_ON_ERROR(g_ort->SessionOptionsAppendOrtExecutionProvider(so, "kernelEp", env, keys.data(), values.data(), keys.size()));
}

void TestTensorRTEp(const OrtApi* g_ort, OrtEnv* env, OrtSessionOptions* so) {
    THROW_ON_ERROR(g_ort->RegisterOrtExecutionProviderLibrary("/home/leca/code/onnxruntime/samples/tensorRTEp/build/libTensorRTEp.so", env, "tensorrtEp"));
    std::vector<const char*> keys{"int_property", "str_property"}, values{"3", "strvalue"};
    THROW_ON_ERROR(g_ort->SessionOptionsAppendOrtExecutionProvider(so, "tensorrtEp", env, keys.data(), values.data(), keys.size()));

//    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
//    THROW_ON_ERROR(g_ort->CreateCUDAProviderOptions(&cuda_options));
//    THROW_ON_ERROR(g_ort->SessionOptionsAppendExecutionProvider_CUDA_V2(so, cuda_options));
//    g_ort->ReleaseCUDAProviderOptions(cuda_options);
}

void TestOriginalTensorRTEp(const OrtApi* g_ort, OrtSessionOptions* so) {
    OrtTensorRTProviderOptionsV2* tensorrt_options = nullptr;
    THROW_ON_ERROR(g_ort->CreateTensorRTProviderOptions(&tensorrt_options));
    THROW_ON_ERROR(g_ort->SessionOptionsAppendExecutionProvider_TensorRT_V2(so, tensorrt_options));

    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    THROW_ON_ERROR(g_ort->CreateCUDAProviderOptions(&cuda_options));
    THROW_ON_ERROR(g_ort->SessionOptionsAppendExecutionProvider_CUDA_V2(so, cuda_options));

    g_ort->ReleaseCUDAProviderOptions(cuda_options);
    g_ort->ReleaseTensorRTProviderOptions(tensorrt_options);
}

int main() {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* p_env = nullptr;
    OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;//OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
    THROW_ON_ERROR(g_ort->CreateEnv(log_level, "", &p_env));
    OrtSessionOptions* so = nullptr;
    THROW_ON_ERROR(g_ort->CreateSessionOptions(&so));

    //TestCompileBasedEp(g_ort, p_env, so);
    //TestKernelBasedEp(g_ort, p_env, so);
    TestTensorRTEp(g_ort, p_env, so);
    //TestOriginalTensorRTEp(g_ort, so);

    OrtSession* session = nullptr;
    THROW_ON_ERROR(g_ort->CreateSession(p_env, "/home/leca/code/onnxruntime/samples/c_test/Relu.onnx", so, &session));

    OrtMemoryInfo* memory_info = nullptr;
    THROW_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    float input_data[] = {-3.0f, 5.0f, -2.0f, 4.0f};
    const size_t input_len = 4 * sizeof(float);
    const int64_t input_shape[] = {4};
    const size_t shape_len = sizeof(input_shape)/sizeof(input_shape[0]);

    OrtValue* input_tensor = nullptr;
    THROW_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_data, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    const char* input_names[] = {"x"};
    const char* output_names[] = {"graphOut"};
    OrtValue* output_tensor = nullptr;
    THROW_ON_ERROR(g_ort->Run(session, nullptr, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor));

    float* output_tensor_data = nullptr;
    THROW_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));
    std::cout<<"Result:\n";
    for (size_t i = 0; i < 4; i++) std::cout<<output_tensor_data[i]<<" \n";

    g_ort->ReleaseEnv(p_env);
    return 0;
}
