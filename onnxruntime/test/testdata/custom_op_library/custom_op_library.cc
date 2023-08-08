#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include <system_error>

#include "core/common/common.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ortmemoryinfo.h"

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
template <typename T1, typename T2, typename T3>
void cuda_add(int64_t, T3*, const T1*, const T2*, cudaStream_t compute_stream);
#endif

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_context.h"
#endif

#ifdef USE_DML
#include "core/providers/dml/dml_context.h"
#endif

#ifdef USE_ROCM
#include "core/providers/rocm/rocm_context.h"
void rocm_add(int64_t, float*, const float*, const float*, hipStream_t compute_stream);
#endif

#include "onnxruntime_lite_custom_op.h"

static const char* c_OpDomain = "test.customop";

#define CUSTOM_ENFORCE(cond, msg)  \
  if (!(cond)) {                   \
    throw std::runtime_error(msg); \
  }

#ifdef USE_DML
#include <wrl/client.h>
#include <core/providers/dml/dml_provider_factory.h>
using Microsoft::WRL::ComPtr;

#define DML_CALL(call, nm)                                                                                        \
  {                                                                                                               \
    HRESULT ret = (call);                                                                                         \
    if (FAILED(ret)) {                                                                                            \
      throw std::runtime_error(std::string{nm} + std::string{" failed, "} + std::system_category().message(ret)); \
    }                                                                                                             \
  }

struct IdentityDML {
  IdentityDML(const OrtApi* ort_api, const OrtKernelInfo*) : api(ort_api) {
    dml_buffer_tensor_desc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    dml_buffer_tensor_desc.Flags = DML_TENSOR_FLAG_NONE;
    dml_buffer_tensor_desc.Sizes = tensor_shape;
    dml_buffer_tensor_desc.Strides = nullptr;

    dml_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
    dml_tensor_desc.Desc = &dml_buffer_tensor_desc;

    dml_identity_op_desc.InputTensor = &dml_tensor_desc;
    dml_identity_op_desc.OutputTensor = &dml_tensor_desc;

    dml_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;
    dml_op_desc.Desc = &dml_identity_op_desc;
  }

  void Compute(OrtKernelContext& /*ctx*/,
               const Ort::Custom::DmlContext& dml_ctx,
               const Ort::Custom::Tensor<float>& /*input*/,
               Ort::Custom::Tensor<float>& /*output*/) {
    // step 1: get resources from dml context
    auto* dml_device = dml_ctx.dml_device;
    CUSTOM_ENFORCE(dml_device, "failed to get dml device");

    auto* d3d12_device = dml_ctx.d3d12_device;
    CUSTOM_ENFORCE(d3d12_device, "failed to get d3d12 device");

    auto* cmd_list = dml_ctx.cmd_list;
    CUSTOM_ENFORCE(cmd_list, "failed to get cmd list");

    auto* cmd_recorder = dml_ctx.cmd_recorder;
    CUSTOM_ENFORCE(cmd_recorder, "failed to get cmd recorder");

    // todo - more logic here to finish the compute ...
  }

  UINT tensor_shape[8U] = {};
  DML_TENSOR_DESC dml_tensor_desc{};
  DML_BUFFER_TENSOR_DESC dml_buffer_tensor_desc = {};
  DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC dml_identity_op_desc = {};
  DML_OPERATOR_DESC dml_op_desc = {};
  ComPtr<IDMLOperator> dml_op = {};
  ComPtr<IDMLCompiledOperator> dml_compiled_op = {};
  ComPtr<IDMLOperatorInitializer> dml_op_initializer = {};
  DML_BINDING_TABLE_DESC dml_binding_table_desc = {};
  ComPtr<IDMLBindingTable> dml_binding_table = {};
  D3D12_DESCRIPTOR_HEAP_DESC desc_heap_desc = {};
  ComPtr<ID3D12DescriptorHeap> d3d12_desc_heap = {};
  // ComPtr<IDMLCommandRecorder> dml_cmd_recorder = {};
  const OrtApi* api{};
};
#endif

void KernelOneCpu(const Ort::Custom::Tensor<float>& X,
                  const Ort::Custom::Tensor<float>& Y,
                  Ort::Custom::Tensor<float>& Z) {
  auto input_shape = X.Shape();
  auto x_raw = X.Data();
  auto y_raw = Y.Data();
  auto z_raw = Z.Allocate(input_shape);
  for (int64_t i = 0; i < Z.NumberOfElement(); ++i) {
    z_raw[i] = x_raw[i] + y_raw[i];
  }
}

#ifdef USE_CUDA
void KernelOneCuda(const Ort::Custom::CudaContext& cuda_ctx,
                   const Ort::Custom::Tensor<float>& X,
                   const Ort::Custom::Tensor<float>& Y,
                   Ort::Custom::Tensor<float>& Z) {
  auto input_shape = X.Shape();

  CUSTOM_ENFORCE(cuda_ctx.cuda_stream, "failed to fetch cuda stream");
  CUSTOM_ENFORCE(cuda_ctx.cudnn_handle, "failed to fetch cudnn handle");
  CUSTOM_ENFORCE(cuda_ctx.cublas_handle, "failed to fetch cublas handle");

  auto z_raw = Z.Allocate(input_shape);
  cuda_add(Z.NumberOfElement(), z_raw, X.Data(), Y.Data(), cuda_ctx.cuda_stream);
}
#endif

#ifdef USE_ROCM
#include <iostream>
void KernelOneRocm(const Ort::Custom::RocmContext& rocm_ctx,
                   const Ort::Custom::Tensor<float>& X,
                   const Ort::Custom::Tensor<float>& Y,
                   Ort::Custom::Tensor<float>& Z) {
  auto input_shape = X.Shape();

  CUSTOM_ENFORCE(rocm_ctx.hip_stream, "failed to fetch hip stream");
  CUSTOM_ENFORCE(rocm_ctx.miopen_handle, "failed to fetch miopen handle");
  CUSTOM_ENFORCE(rocm_ctx.rblas_handle, "failed to fetch rocblas handle");
  std::cout << "Running rocm kernel one" << std::endl;
  auto z_raw = Z.Allocate(input_shape);
  rocm_add(Z.NumberOfElement(), z_raw, X.Data(), Y.Data(), rocm_ctx.hip_stream);
}
#endif

#if !defined(DISABLE_FLOAT8_TYPES)

struct KernelOneFloat8 {
  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    auto input_X = ctx.GetInput(0);
    const Ort::Float8E4M3FN_t* X = input_X.GetTensorData<Ort::Float8E4M3FN_t>();
    auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();
    auto output = ctx.GetOutput(0, dimensions);
    Ort::Float8E4M3FN_t* out = output.GetTensorMutableData<Ort::Float8E4M3FN_t>();
    const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();
    for (size_t i = 0; i < size; i++) {
      out[i] = X[i];
    }
  }
};

// legacy custom op registration
struct CustomOpOneFloat8 : Ort::CustomOpBase<CustomOpOneFloat8, KernelOneFloat8> {
  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* /* info */) const {
    return std::make_unique<KernelOneFloat8>().release();
  };
  const char* GetName() const { return "CustomOpOneFloat8"; };
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };
  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN; };
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN; };
};

#endif

// lite custom op as a function
void KernelTwo(const Ort::Custom::Tensor<float>& X,
               Ort::Custom::Tensor<int32_t>& Y) {
  const auto& shape = X.Shape();
  auto X_raw = X.Data();
  auto Y_raw = Y.Allocate(shape);
  auto total = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  for (int64_t i = 0; i < total; i++) {
    Y_raw[i] = static_cast<int32_t>(round(X_raw[i]));
  }
}

template <typename T>
void MulTop(const Ort::Custom::Span<T>& in, Ort::Custom::Tensor<T>& out) {
  out.Allocate({1})[0] = in[0] * in[1];
}

void Fuse(
    OrtKernelContext*,
    const Ort::Custom::Span<float>& vector_1,
    const Ort::Custom::Span<float>& vector_2,
    int32_t alpha,
    Ort::Custom::Tensor<float>& vector_output) {
  auto len_output = std::min(vector_1.size(), vector_2.size());
  float* floats_out = static_cast<float*>(vector_output.Allocate({(int64_t)len_output}));
  for (size_t i = 0; i < len_output; ++i) {
    floats_out[i] = (vector_1[i] + vector_2[i]) * alpha;
  }
}

void Select(const Ort::Custom::Span<int32_t>& indices_in,
            Ort::Custom::Tensor<int32_t>& indices_out) {
  std::vector<int32_t> selected_indices;
  for (size_t i = 0; i < indices_in.size(); ++i) {
    if (indices_in[i] % 2 == 0) {
      selected_indices.push_back(indices_in[i]);
    }
  }

  int32_t* int_out = static_cast<int32_t*>(indices_out.Allocate({static_cast<int64_t>(selected_indices.size())}));
  for (size_t j = 0; j < selected_indices.size(); ++j) {
    int_out[j] = selected_indices[j];
  }
}

void Filter(const Ort::Custom::Tensor<float>& floats_in,
            Ort::Custom::Tensor<float>& floats_out) {
  const float* in = floats_in.Data();
  auto in_len = floats_in.NumberOfElement();

  std::vector<float> filter_floats;
  for (int64_t i = 0; i < in_len; ++i) {
    if (in[i] > 1.f) {
      filter_floats.push_back(in[i]);
    }
  }

  float* out = static_cast<float*>(floats_out.Allocate({static_cast<int64_t>(filter_floats.size())}));
  for (size_t j = 0; j < filter_floats.size(); ++j) {
    out[j] = filter_floats[j];
  }
}

#if !defined(DISABLE_FLOAT8_TYPES)

void FilterFloat8(const Ort::Custom::Tensor<Ort::Float8E4M3FN_t>& floats_in,
                  Ort::Custom::Tensor<Ort::Float8E4M3FN_t>& floats_out) {
  const Ort::Float8E4M3FN_t* in = floats_in.Data();
  auto in_len = floats_in.NumberOfElement();

  std::vector<Ort::Float8E4M3FN_t> filter_floats;
  for (int64_t i = 0; i < in_len; ++i) {
    if (in[i] > 1.f) {
      filter_floats.push_back(in[i]);
    }
  }

  Ort::Float8E4M3FN_t* out = static_cast<Ort::Float8E4M3FN_t*>(floats_out.Allocate({static_cast<int64_t>(filter_floats.size())}));
  for (size_t j = 0; j < filter_floats.size(); ++j) {
    out[j] = filter_floats[j];
  }
}

#endif

void Box(const Ort::Custom::Tensor<float>* float_in_1,
         const Ort::Custom::Tensor<float>* float_in_2,
         std::optional<const Ort::Custom::Tensor<float>*> float_in_3,
         Ort::Custom::Tensor<float>* float_out_1,
         std::optional<Ort::Custom::Tensor<float>*> float_out_2) {
  auto raw_in_1 = float_in_1->Data();
  auto raw_in_2 = float_in_2->Data();

  auto l_in_1 = float_in_1->Shape()[0];
  auto l_in_2 = float_in_2->Shape()[0];
  auto l_out_1 = l_in_1 + l_in_2;

  auto raw_out_1 = float_out_1->Allocate({l_out_1});

  for (int64_t i = 0; i < l_out_1; ++i) {
    raw_out_1[i] = i < l_in_1 ? raw_in_1[i] : raw_in_2[i - l_in_1];
  }

  if (float_in_3.has_value() && float_out_2.has_value()) {
    auto raw_in_3 = float_in_3.value()->Data();
    auto l_in_3 = float_in_3.value()->Shape()[0];
    auto l_out_2 = l_in_2 + l_in_3;
    auto raw_out_2 = float_out_2.value()->Allocate({l_out_2});
    for (int64_t i = 0; i < l_out_2; ++i) {
      raw_out_2[i] = i < l_in_2 ? raw_in_2[i] : raw_in_3[i - l_in_2];
    }
  }
}

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

#if !defined(DISABLE_FLOAT8_TYPES)
  static const CustomOpOneFloat8 c_CustomOpOneFloat8;
#endif
  using LiteOp = Ort::Custom::OrtLiteCustomOp;

  static const std::unique_ptr<LiteOp> c_CustomOpOneCpu{Ort::Custom::CreateLiteCustomOp("CustomOpOne", "CPUExecutionProvider", KernelOneCpu)};

#ifdef USE_CUDA
  static const std::unique_ptr<LiteOp> c_CustomOpOneCuda{Ort::Custom::CreateLiteCustomOp("CustomOpOne", "CUDAExecutionProvider", KernelOneCuda)};
#endif

#ifdef USE_ROCM
  static const std::unique_ptr<LiteOp> c_CustomOpOneRocm{Ort::Custom::CreateLiteCustomOp("CustomOpOne", "ROCMExecutionProvider", KernelOneRocm)};
#endif

  static const std::unique_ptr<LiteOp> c_CustomOpTwo{Ort::Custom::CreateLiteCustomOp("CustomOpTwo", "CPUExecutionProvider", KernelTwo)};
  static const std::unique_ptr<LiteOp> c_MulTopOpFloat{Ort::Custom::CreateLiteCustomOp("MulTop", "CPUExecutionProvider", MulTop<float>)};
  static const std::unique_ptr<LiteOp> c_MulTopOpInt32{Ort::Custom::CreateLiteCustomOp("MulTop", "CPUExecutionProvider", MulTop<int32_t>)};
  static const std::unique_ptr<LiteOp> fus_op_ptr{Ort::Custom::CreateLiteCustomOp("Fuse", "CPUExecutionProvider", Fuse)};
  static const std::unique_ptr<LiteOp> sel_op_ptr{Ort::Custom::CreateLiteCustomOp("Select", "CPUExecutionProvider", Select)};
  static const std::unique_ptr<LiteOp> fil_op_ptr{Ort::Custom::CreateLiteCustomOp("Filter", "CPUExecutionProvider", Filter)};
#if !defined(DISABLE_FLOAT8_TYPES)
  static const std::unique_ptr<LiteOp> fil8_op_ptr{Ort::Custom::CreateLiteCustomOp("FilterFloat8", "CPUExecutionProvider", FilterFloat8)};
#endif
  static const std::unique_ptr<LiteOp> box_op_ptr{Ort::Custom::CreateLiteCustomOp("Box", "CPUExecutionProvider", Box)};

#ifdef USE_DML
  static const std::unique_ptr<LiteOp> identity_dml_op_ptr{Ort::Custom::CreateLiteCustomOp<IdentityDML>("IdentityDML", "DmlExecutionProvider")};
#endif

  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};

    domain.Add(c_CustomOpOneCpu.get());

#ifdef USE_CUDA
    domain.Add(c_CustomOpOneCuda.get());
#endif

#ifdef USE_ROCM
    domain.Add(c_CustomOpOneRocm.get());
#endif

#if !defined(DISABLE_FLOAT8_TYPES)
    domain.Add(&c_CustomOpOneFloat8);
#endif
    domain.Add(c_CustomOpTwo.get());

#ifdef USE_DML
    domain.Add(identity_dml_op_ptr.get());
#endif

    Ort::CustomOpDomain domain_v2{"v2"};
    domain_v2.Add(c_MulTopOpFloat.get());
    domain_v2.Add(c_MulTopOpInt32.get());
    domain_v2.Add(fus_op_ptr.get());
    domain_v2.Add(sel_op_ptr.get());
    domain_v2.Add(fil_op_ptr.get());
#if !defined(DISABLE_FLOAT8_TYPES)
    domain_v2.Add(fil8_op_ptr.get());
#endif
    domain_v2.Add(box_op_ptr.get());

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
