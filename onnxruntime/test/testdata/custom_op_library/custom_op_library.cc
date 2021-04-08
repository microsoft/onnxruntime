#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include "core/common/common.h"

#include "sequence_pooling.h"
#include <cuda_fp16.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
//template <typename T1, typename T2, typename T3>
//void cuda_add(int64_t, T3*, const T1*, const T2*, cudaStream_t compute_stream);
#endif

static const char* c_OpDomain = "com.microsoft";

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

struct KernelOne {
  KernelOne(OrtApi api)
      : api_(api),
        ort_(api_) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
    const float* X = ort_.GetTensorData<float>(input_X);
    const float* Y = ort_.GetTensorData<float>(input_Y);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    float* out = ort_.GetTensorMutableData<float>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
    for (int64_t i = 0; i < size; i++) {
      out[i] = X[i] + Y[i];
    }
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};

struct KernelTwo {
  KernelTwo(OrtApi api)
      : api_(api),
        ort_(api_) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const float* X = ort_.GetTensorData<float>(input_X);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    int32_t* out = ort_.GetTensorMutableData<int32_t>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
    for (int64_t i = 0; i < size; i++) {
      out[i] = (int32_t)(round(X[i]));
    }
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};

struct CustomOpOne : Ort::CustomOpBase<CustomOpOne, KernelOne> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
    return new KernelOne(api);
  };

  const char* GetName() const { return "CustomOpOne"; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

} c_CustomOpOne;

struct CustomOpTwo : Ort::CustomOpBase<CustomOpTwo, KernelTwo> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
    return new KernelTwo(api);
  };

  const char* GetName() const { return "CustomOpTwo"; };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; };

} c_CustomOpTwo;

// SequencePooling-------------------------------------------
// SequencePooling-------------------------------------------
// SequencePooling-------------------------------------------
struct SequencePoolingKernel {
  SequencePoolingKernel(OrtApi api)
      : api_(api),
        ort_(api_) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* senlens = ort_.KernelContext_GetInput(context, 1);
    const float* input_data = ort_.GetTensorData<float>(input);
    const int64_t* senlens_data = ort_.GetTensorData<int64_t>(senlens);

    // Setup output
    OrtTensorDimensions input_dim(ort_, input);
    OrtTensorDimensions senlens_dim(ort_, senlens);

    int batch_size = static_cast<int>(input_dim[0]);
    int hidden_size = static_cast<int>(input_dim[2]);
    int num_sequences = static_cast<int>(senlens_dim[1]);
    int sequence_length_for_split = static_cast<int>(input_dim[1]);

    std::vector<int64_t> output_dims = input_dim;
    output_dims[1] = 256;

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size());
    float* output_data = ort_.GetTensorMutableData<float>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    SequencePoolingCuda(batch_size,
                        hidden_size,
                        num_sequences,
                        sequence_length_for_split,
                        input_data,
                        senlens_data,
                        output_data);

  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};


struct SequencePoolingKernel16 {
  SequencePoolingKernel16(OrtApi api)
      : api_(api),
        ort_(api_) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* senlens = ort_.KernelContext_GetInput(context, 1);
    const half* input_data = ort_.GetTensorData<half>(input);
    const int64_t* senlens_data = ort_.GetTensorData<int64_t>(senlens);

    // Setup output
    OrtTensorDimensions input_dim(ort_, input);
    OrtTensorDimensions senlens_dim(ort_, senlens);

    int batch_size = static_cast<int>(input_dim[0]);
    int hidden_size = static_cast<int>(input_dim[2]);
    int num_sequences = static_cast<int>(senlens_dim[1]);
    int sequence_length_for_split = static_cast<int>(input_dim[1]);

    std::vector<int64_t> output_dims = input_dim;
    output_dims[1] = 256;

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size());
    half* output_data = ort_.GetTensorMutableData<half>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
    SequencePoolingCuda(batch_size,
                        hidden_size,
                        num_sequences,
                        sequence_length_for_split,
                        input_data,
                        senlens_data,
                        output_data);
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};


struct SequencePooling : Ort::CustomOpBase<SequencePooling, SequencePoolingKernel> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
    return new SequencePoolingKernel(api);
  };

  const char* GetName() const { return "SequencePooling"; };
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/index) const {
    if (index == 0) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

} c_SequencePooling;

struct SequencePooling16 : Ort::CustomOpBase<SequencePooling16, SequencePoolingKernel16> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
    return new SequencePoolingKernel(api);
  };

  const char* GetName() const { return "SequencePooling"; };
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/index) const {
    if (index == 0) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };

} c_SequencePooling16;

// SequencePooling-------------------------------------------
// SequencePooling-------------------------------------------
// SequencePooling-------------------------------------------


OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomOpOne)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomOpTwo)) {
    return status;
  }

  //if (auto status = ortApi->CustomOpDomain_Add(domain, &c_SequencePooling)) {
  //  return status;
  //}

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_SequencePooling16)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}
