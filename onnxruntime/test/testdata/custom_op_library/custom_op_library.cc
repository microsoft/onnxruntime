#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>

#include "core/common/common.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
template <typename T1, typename T2, typename T3>
void cuda_add(int64_t, T3*, const T1*, const T2*, cudaStream_t compute_stream);
#endif

static const char* c_OpDomain = "test.customop";

struct KernelOne {
  void Compute(OrtKernelContext* context) {
    // Setup inputs
    Ort::KernelContext ctx(context);
    auto input_X = ctx.GetInput(0);
    auto input_Y = ctx.GetInput(1);
    const float* X = input_X.GetTensorData<float>();
    const float* Y = input_Y.GetTensorData<float>();

    // Setup output
    auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();

    auto output = ctx.GetOutput(0, dimensions);
    float* out = output.GetTensorMutableData<float>();

    const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();

    // Do computation
#ifdef USE_CUDA
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.GetGPUComputeStream());
    cuda_add(size, out, X, Y, stream);
#else
    for (size_t i = 0; i < size; i++) {
      out[i] = X[i] + Y[i];
    }
#endif
  }
};

struct KernelTwo {
  void Compute(OrtKernelContext* context) {
    // Setup inputs
    Ort::KernelContext ctx(context);
    auto input_X = ctx.GetInput(0);
    const float* X = input_X.GetTensorData<float>();

    // Setup output
    auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();

    auto output = ctx.GetOutput(0, dimensions);
    int32_t* out = output.GetTensorMutableData<int32_t>();

    const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();

    // Do computation
    for (size_t i = 0; i < size; i++) {
      out[i] = static_cast<int32_t>(round(X[i]));
    }
  }
};

struct CustomOpOne : Ort::CustomOpBase<CustomOpOne, KernelOne> {
  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* /* info */) const {
    return std::make_unique<KernelOne>().release();
  };

  const char* GetName() const { return "CustomOpOne"; };

#ifdef USE_CUDA
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; };
#endif

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};

struct CustomOpTwo : Ort::CustomOpBase<CustomOpTwo, KernelTwo> {
  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* /* info */) const {
    return std::make_unique<CustomOpTwo>().release();
  };

  const char* GetName() const { return "CustomOpTwo"; };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; };
};

////////////////////////////////////////////////

template <typename T>
T MulTopCompute(const T& input_0, const T& input_1) {
  return input_0 * input_1;
}

struct MulTopKernelFloat {
  MulTopKernelFloat(const OrtKernelInfo*){};
  ~MulTopKernelFloat() = default;
  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    auto tensor_in = ctx.GetInput(0);
    const float* float_in = tensor_in.GetTensorData<float>();
    int64_t output_shape = 1;
    auto tensor_out = ctx.GetOutput(0, &output_shape, 1);
    auto float_out = tensor_out.GetTensorMutableData<float>();
    *float_out = MulTopCompute(float_in[0], float_in[1]);
  }
};

struct MulTopOpFloat : Ort::CustomOpBase<MulTopOpFloat, MulTopKernelFloat> {
  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const { return new MulTopKernelFloat(info); }
  const char* GetName() const { return "MulTop"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};

////////////////////////////////////////////////

struct MulTopKernelInt32 {
  MulTopKernelInt32(const OrtKernelInfo*){};
  ~MulTopKernelInt32() = default;
  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    auto tensor_in = ctx.GetInput(0);
    const int32_t* int_in = tensor_in.GetTensorData<int32_t>();
    int64_t output_shape = 1;
    auto tensor_out = ctx.GetOutput(0, &output_shape, 1);
    auto int_out = tensor_out.GetTensorMutableData<int32_t>();
    *int_out = MulTopCompute(int_in[0], int_in[1]);
  }
};

struct MulTopOpInt32 : Ort::CustomOpBase<MulTopOpInt32, MulTopKernelInt32> {
  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const { return new MulTopKernelInt32(info); }
  const char* GetName() const { return "MulTop"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; }
};

////////////////////////////////////////////////

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  static const CustomOpOne c_CustomOpOne;
  static const CustomOpTwo c_CustomOpTwo;

  static const MulTopOpFloat c_MulTopOpFloat;
  static const MulTopOpInt32 c_MulTopOpInt32;

  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(&c_CustomOpOne);
    domain.Add(&c_CustomOpTwo);

    Ort::CustomOpDomain domain_v2{"v2"};
    domain_v2.Add(&c_MulTopOpFloat);
    domain_v2.Add(&c_MulTopOpInt32);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    session_options.Add(domain_v2);
    AddOrtCustomOpDomainToContainer(std::move(domain));
    AddOrtCustomOpDomainToContainer(std::move(domain_v2));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      Ort::Status status{e};
      result = status.release();
    });
  }
  return result;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}
