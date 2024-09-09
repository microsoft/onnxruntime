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
    std::vector<const char*> keys{"device_id", "str_property"}, values{"0", "strvalue"};
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

void RunResnet18v1_7(const OrtApi* g_ort, OrtEnv* p_env, OrtSessionOptions* so) {
    // download resnet18-v1-7 model at:
    // https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet18-v1-7.tar.gz
    OrtSession* session = nullptr;
    THROW_ON_ERROR(g_ort->CreateSession(p_env, "/home/leca/models/resnet18-v1-7/resnet18-v1-7.onnx", so, &session));

    const int input_data_cnt = 3 * 224 * 224;
    float input_data[input_data_cnt];
    for (int i = 0; i < input_data_cnt; i++) {
        input_data[i] = -1 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/(2)));   // [-1, 1) uniform distribution
    }
    const size_t input_len = input_data_cnt * sizeof(float);
    const int64_t input_shape[] = {1, 3, 224, 224};
    const size_t shape_len = sizeof(input_shape)/sizeof(input_shape[0]);

    OrtMemoryInfo* memory_info = nullptr;
    THROW_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    OrtValue* input_tensor = nullptr;
    THROW_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_data, input_len, input_shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    const char* input_names[] = {"data"};
    const char* output_names[] = {"resnetv15_dense0_fwd"};
    OrtValue* output_tensor = nullptr;
    THROW_ON_ERROR(g_ort->Run(session, nullptr, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor));

    float* output_tensor_data = nullptr;
    THROW_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));
    std::cout<<"Result:\n";
    for (size_t i = 0; i < 4; i++) std::cout<<output_tensor_data[i]<<" \n";
}

void RunRelu(const OrtApi* g_ort, OrtEnv* p_env, OrtSessionOptions* so) {
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
}

void RunDecoder(const OrtApi* g_ort, OrtEnv* p_env, OrtSessionOptions* so) {
    OrtSession* session = nullptr;
    THROW_ON_ERROR(g_ort->CreateSession(p_env, "/home/leca/models/decoder/decoder.onnx", so, &session));

    OrtMemoryInfo* memory_info = nullptr;
    THROW_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    std::vector<OrtValue*> input_tensors(28, nullptr);

    const int input_0_cnt = 16;
    int64_t input_0_data[input_0_cnt];
    for (int i = 0; i < input_0_cnt; i++) input_0_data[i] = static_cast<int64_t>(rand());
    const size_t input_0_len = input_0_cnt * sizeof(int64_t);
    const int64_t input_0_shape[] = {16, 1};
    THROW_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_0_data, input_0_len, input_0_shape, sizeof(input_0_shape)/sizeof(input_0_shape[0]), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensors[0]));

    const int input_1_cnt = 16;
    bool input_1_data[input_1_cnt];
    for (int i = 0; i < input_1_cnt; i++) input_1_data[i] = false;
    const size_t input_1_len = input_1_cnt * sizeof(bool);
    const int64_t input_1_shape[] = {16, 1};
    THROW_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_1_data, input_1_len, input_1_shape, sizeof(input_1_shape)/sizeof(input_1_shape[0]), ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, &input_tensors[1]));

    const int input_3_cnt = 16*256;
    bool input_3_data[input_3_cnt];
    for (int i = 0; i < input_3_cnt; i++) input_3_data[i] = false;
    const size_t input_3_len = input_3_cnt * sizeof(bool);
    const int64_t input_3_shape[] = {16, 256};
    THROW_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_3_data, input_3_len, input_3_shape, sizeof(input_3_shape)/sizeof(input_3_shape[0]), ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, &input_tensors[3]));

    for (int j = 2; j < 28; j++) {
        if (j == 3) continue;
        const int input_cnt = 16 * 256 * 1024;
        float input_data[input_cnt];
        for (int i = 0; i < input_cnt; i++) input_data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);  // [0, 1)
        const size_t input_len = input_cnt * sizeof(float);
        const int64_t input_shape[] = {16, 256, 1024};
        THROW_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_data, input_len, input_shape, sizeof(input_shape)/sizeof(input_shape[0]), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors[j]));
    }

    const char* input_names[] = {"input_ids", "input_mask", "encoder_states", "encoder_input_mask", "history_states_0",
    "history_states_1", "history_states_2", "history_states_3", "history_states_4", "history_states_5", "history_states_6",
    "history_states_7", "history_states_8", "history_states_9", "history_states_10", "history_states_11", "history_states_12",
    "history_states_13", "history_states_14", "history_states_15", "history_states_16", "history_states_17", "history_states_18",
    "history_states_19", "history_states_20", "history_states_21", "history_states_22", "history_states_23"};
    const char* output_names[] = {"lm_logits", "log_lm_logits", "hidden_states_0", "hidden_states_1", "hidden_states_2",
    "hidden_states_3", "hidden_states_4", "hidden_states_5", "hidden_states_6", "hidden_states_7", "hidden_states_8",
    "hidden_states_9", "hidden_states_10", "hidden_states_11", "hidden_states_12", "hidden_states_13", "hidden_states_14",
    "hidden_states_15", "hidden_states_16", "hidden_states_17", "hidden_states_18", "hidden_states_19", "hidden_states_20",
    "hidden_states_21", "hidden_states_22", "hidden_states_23", "hidden_states_24"};
    OrtValue* output_tensor = nullptr;
    THROW_ON_ERROR(g_ort->Run(session, nullptr, input_names, (const OrtValue* const*)input_tensors.data(), sizeof(input_names)/sizeof(input_names[0]), output_names, sizeof(output_names)/sizeof(output_names[0]), &output_tensor));

    float* output_tensor_data = nullptr;
    THROW_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));
    std::cout<<"Result:\n";
    for (size_t i = 0; i < 4; i++) std::cout<<output_tensor_data[i]<<" \n";
}

void RunFastRcnn(const OrtApi* g_ort, OrtEnv* p_env, OrtSessionOptions* so) {
    OrtSession* session = nullptr;
    THROW_ON_ERROR(g_ort->CreateSession(p_env, "/home/leca/models/faster_rcnn/faster_rcnn_R_50_FPN_1x.onnx", so, &session));

    OrtMemoryInfo* memory_info = nullptr;
    THROW_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    const int input_cnt = 3 * 800 * 1088;
    int64_t input_data[input_cnt];
    for (int i = 0; i < input_cnt; i++) input_data[i] = static_cast<float>(rand());
    const size_t input_len = input_cnt * sizeof(float);
    const int64_t input_shape[] = {3, 800, 1088};
    OrtValue* input_tensor = nullptr;
    THROW_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_data, input_len, input_shape, sizeof(input_shape)/sizeof(input_shape[0]), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    const char* input_names[] = {"image"};
    const char* output_names[] = {"6379", "6381", "6383"};

    OrtValue* output_tensor = nullptr;
    THROW_ON_ERROR(g_ort->Run(session, nullptr, input_names, (const OrtValue* const*)&input_tensor, sizeof(input_names)/sizeof(input_names[0]), output_names, sizeof(output_names)/sizeof(output_names[0]), &output_tensor));

    float* output_tensor_data = nullptr;
    THROW_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));
    std::cout<<"Result:\n";
    for (size_t i = 0; i < 4; i++) std::cout<<output_tensor_data[i]<<" \n";
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

    //RunRelu(g_ort, p_env, so);
    RunResnet18v1_7(g_ort, p_env, so);

    g_ort->ReleaseEnv(p_env);
    return 0;
}
