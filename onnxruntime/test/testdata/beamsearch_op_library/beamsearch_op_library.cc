#include "beamsearch_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdio.h>

#include <vector>
#include <cmath>
#include <mutex>
#include <ctime>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

static const char* c_OpDomain = "test.beamsearchop";

struct OrtCustomOpDomainDeleter {
  explicit OrtCustomOpDomainDeleter(const OrtApi* ort_api) {
    ort_api_ = ort_api;
  }
  void operator()(OrtCustomOpDomain* domain) const {
    ort_api_->ReleaseCustomOpDomain(domain);
  }

  const OrtApi* ort_api_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain* domain, const OrtApi* ort_api) {
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(domain, OrtCustomOpDomainDeleter(ort_api));
  ort_custom_op_domain_container.push_back(std::move(ptr));
}

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

struct CustomBearmsearchKernel {
  CustomBearmsearchKernel(OrtApi api, const OrtKernelInfo* info)
      : api_(api), ort_(api_) {
      //TODO how to free kernel info.
    OutputDebugStringA("Kernel Constructor called");

    api_.GetAllocatorWithDefaultOptions(&allocator_);
    api_.KernelInfoGetAttributeArray_void(info, allocator_, "customsubgraph", &graphProtoBufferPtr_, &size_);
    api_.KernelInfoGetAttribute_int64(info, "sampleint", &sampleint_);
    
    if (&api_ == nullptr) {
      OutputDebugStringA("api is nullptr");
    }

    session_ = nullptr;
 }

  void Compute(OrtKernelContext* context) {
   OutputDebugStringA("Compute is called");
   std::cout << "Size of internal model:"<< size_ << std::endl;
   std::cout << "Sampleint_:" << sampleint_ << std::endl;


   const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
   const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
   const float* X = ort_.GetTensorData<float>(input_X);
   const float* Y = ort_.GetTensorData<float>(input_Y);

   // session_ is getting created only once. TestInference<> actaully tests every test twice, which is good
   // for us, it verifies that it is resuing the earlier session.
   // => For verification, the test has to fail, uncomment temp = 1
   int temp = 0;
   if (session_ == nullptr) {
     std::cout << "Trying to create session" << std::endl;
     OrtEnv* env;
     api_.CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "customOp", &env);
     std::cout << "end session creation" << std::endl;

     OrtSessionOptions* sessionoptions;
     api_.CreateSessionOptions(&sessionoptions);

     std::filesystem::path model_path = "D:\\ai\\onnxruntime\\onnxruntime\\bart_mlp_megatron_basic_test.onnx";
     std::wstring model_path_wstring = model_path.wstring();
     std::time_t start_time = std::time(0);

     /*
     OrtCUDAProviderOptionsV2* provideroptions;
     std::vector<const char*> keys{"enable_cuda_graph", "deviceid"};
     std::vector<const char*> values{"1", "0"};
     api_.CreateCUDAProviderOptions(&provideroptions);
     api_.UpdateCUDAProviderOptions(provideroptions, keys.data(), values.data(), 2);
     api_.SessionOptionsAppendExecutionProvider_CUDA_V2(sessionoptions, provideroptions);
     */
     try {      
       api_.CreateSession(env, model_path_wstring.data(), sessionoptions, &session_);
       //api_.CreateSessionFromArray(env, graphProtoBufferPtr_, size_, sessionoptions, &session_);
     } catch (Ort::Exception& e) {
       std::cout << e.what() << std::endl;
     }
     std::time_t end_time = std::time(0);
     std::cout << "Time elapsed for creating a session:" << end_time - start_time << std::endl;
   } else {
     temp = 1;
   }

   std::array<int64_t, 3> inputShape = {1, 2, 4};
   std::array<int64_t, 3> outputShape = {1, 2, 4};
   std::array<float, 1 * 2 * 4> input1 = {1.0f, -1.2f, 1.0f, 0.0f, -1.2f, 1.0f, 1.0f, 1.0f};
   std::array<float, 1 * 2 * 4> output1;
   std::array<const char*, 1> inputNames = {"input"};
   std::array<const char*, 1> outputNames = {"output"};

    OrtMemoryInfo *ortmemoryinfo;
    // Must be freed explicitly
    api_.CreateMemoryInfo("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeCPU, &ortmemoryinfo);

    OrtValue* inputvalue;
    OrtValue* outputvalue;
    api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, input1.data(), 4*input1.size(), inputShape.data(),
        inputShape.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,  &inputvalue);
    api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, output1.data(), 4*output1.size(), outputShape.data(),
        outputShape.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &outputvalue);

    api_.Run(session_, nullptr, inputNames.data(), &inputvalue, 1, outputNames.data(), 1, &outputvalue);

    for (int i = 0; i < 1 * 2 * 4; i++) {
      std::cout << i << ":" << output1[i] << std::endl;
    }

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    int* out = ort_.GetTensorMutableData<int>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    for (int64_t i = 0; i < size; i++) {
      out[i] = static_cast<int>(X[i] + (*Y) + temp);
      std::cout << out[i] << std::endl;
    }
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
  
  OrtAllocator* allocator_;
  OrtSession* session_;

  //Subgraph variables
  size_t size_;
  void* graphProtoBufferPtr_;
  int64_t sampleint_;
};

struct CustomBeamSearchOP : Ort::CustomOpBase<CustomBeamSearchOP, CustomBearmsearchKernel> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {

    return new CustomBearmsearchKernel(api, info);
  };

  const char* GetName() const { return "CustomBeamsearchOp"; };
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };

  size_t GetInputTypeCount() const {
    // TODO Vish, how to count these?
    // There are many optional inputs
    return 2; };

  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    // TODO vish each index has a different type
    // There are some optional inputs as well, how to verify that node actually has these inputs? 
    // Is it up to the caller
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const {
    //TODO vish
    // what is the reason for this. Might change in the future.
    return 1; };
  
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    // TODO vish, same as GetInputType.
    // Optional outputs exist, how to verify that output actually exists. 
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; };

} c_CustomBeamSearchOP;

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomBeamSearchOP)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}
