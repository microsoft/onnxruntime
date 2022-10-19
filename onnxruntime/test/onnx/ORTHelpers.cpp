#include "pch.h"
#include "ORTHelpers.h"

#define THROW_IF_FAILED(hr) {HRESULT localHr = (hr); if (FAILED(hr)) throw hr;}
#define RETURN_IF_FAILED(hr) {HRESULT localHr = (hr); if (FAILED(hr)) return hr;}
#define THROW_IF_NOT_OK(status) {auto localStatus = (status); if (localStatus) throw E_FAIL;}
#define RETURN_HR_IF_NOT_OK(status) {auto localStatus = (status); if (localStatus) return E_FAIL;}

template <typename T>
using BaseType =
std::remove_cv_t<
    std::remove_reference_t<
    std::remove_pointer_t<
    std::remove_all_extents_t<T>
    >
    >
>;

template<typename T>
using deleting_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

template <typename C, typename T = BaseType<decltype(*std::declval<C>().data())>>
T GetElementCount(C const& range)
{
    return std::accumulate(range.begin(), range.end(), static_cast<T>(1), std::multiplies<T>());
};

// Create an ORT Session from a given model file path
Ort::Session CreateSession(const wchar_t* model_file_path)
{
    OrtApi const& ortApi = Ort::GetApi();
    const OrtDmlApi* ortDmlApi;
    ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi));
    Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "DirectML_Direct3D_TensorAllocation_Test");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.DisableMemPattern();
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    ortApi.AddFreeDimensionOverrideByName(sessionOptions, "batch_size", 1);
    OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0);

    return Ort::Session(ortEnvironment, model_file_path, sessionOptions);
}


// Run the buffer through a preprocessing model that will shrink the
// image from 512 x 512 x 4 to 224 x 224 x 3
Ort::Value Preprocess(Ort::Session& session,
    ComPtr<ID3D12Resource> currentBuffer)
{
    // Init OrtAPI
    OrtApi const& ortApi = Ort::GetApi(); // Uses ORT_API_VERSION
    const OrtDmlApi* ortDmlApi;
    THROW_IF_NOT_OK(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));

    // Create ORT Value from buffer currently being drawn to screen
    const char* memoryInformationName = "DML";
    Ort::MemoryInfo memoryInformation(memoryInformationName, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    ComPtr<IUnknown> inputTensorEpWrapper;
    
    // Calculate input shape
    auto width = 800;
    auto height = 600;
    auto rowPitchInBytes = (width * 4 + 255) & ~255;
    auto rowPitchInPixels = rowPitchInBytes / 4;
    auto bufferInBytes = rowPitchInBytes * height;
    const std::array<int64_t, 2> inputShape = { 1, bufferInBytes };

    Ort::Value inputTensor = CreateTensorValueFromD3DResource(
        *ortDmlApi,
        memoryInformation,
        currentBuffer.Get(),
        inputShape,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        /*out*/ IID_PPV_ARGS_Helper(inputTensorEpWrapper.GetAddressOf())
    );

    // Create input and output node names
    const char* inputTensorName = "Input";
    const char* outputTensorName = "Output";
    std::vector<const char*> input_node_names;
    input_node_names.push_back(inputTensorName);
    std::vector<const char*> output_node_names;
    output_node_names.push_back(outputTensorName);
    
    // Evaluate input (resize from 512 x 512 x 4 to 224 x 224 x 3)
    Ort::Value outputTensor(nullptr);
    session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(),
        &inputTensor, 1, output_node_names.data(), &outputTensor, 1);

    return outputTensor;
}

// Classify the image using EfficientNet and return the results
winrt::com_array<float> Eval(Ort::Session& session,
    const Ort::Value& prev_input)
{
    // Create input and output node names
    const char* inputTensorName = "images:0";
    const char* outputTensorName = "Softmax:0";
    std::vector<const char*> input_node_names;
    input_node_names.push_back(inputTensorName);
    std::vector<const char*> output_node_names;
    output_node_names.push_back(outputTensorName);

    // Evaluate
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &prev_input, 1, output_node_names.data(), 1);
    
    // Get the 1000 EfficientNet classifications as a com_array and return
    // the results
    float* floatArray = output_tensors.front().GetTensorMutableData<float>();
    winrt::com_array<float> final_results(1000);
    for (int i = 0; i < 1000; i++) {
        final_results[i] = floatArray[i];
    }

    return final_results;
}

// Create ORT Value from the D3D buffer currently being drawn to the screen
Ort::Value CreateTensorValueFromD3DResource(
    OrtDmlApi const& ortDmlApi,
    Ort::MemoryInfo const& memoryInformation,
    ID3D12Resource* d3dResource,
    std::span<const int64_t> tensorDimensions,
    ONNXTensorElementDataType elementDataType,
    /*out*/ void** dmlEpResourceWrapper // Must stay alive with Ort::Value.
)
{
    *dmlEpResourceWrapper = nullptr;

    void* dmlAllocatorResource;
    THROW_IF_NOT_OK(ortDmlApi.CreateGPUAllocationFromD3DResource(d3dResource, &dmlAllocatorResource));
    auto deleter = [&](void*) {ortDmlApi.FreeGPUAllocation(dmlAllocatorResource); };
    deleting_unique_ptr<void> dmlAllocatorResourceCleanup(dmlAllocatorResource, deleter);

    // Calculate the tensor byte size
    size_t tensorByteSize = static_cast<size_t>(d3dResource->GetDesc().Width * d3dResource->GetDesc().Height
        * 3 * 4);

    // Create the ORT Value
    Ort::Value newValue(
        Ort::Value::CreateTensor(
            memoryInformation,
            dmlAllocatorResource,
            tensorByteSize,
            tensorDimensions.data(),
            tensorDimensions.size(),
            elementDataType
        )
    );

    // Return values and the wrapped resource.
    *dmlEpResourceWrapper = dmlAllocatorResource;
    dmlAllocatorResourceCleanup.release();

    return newValue;
}