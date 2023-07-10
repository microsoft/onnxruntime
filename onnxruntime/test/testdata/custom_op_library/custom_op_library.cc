#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>

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
#include "onnxruntime_cuda_context.h"
#endif

#ifdef USE_DML
#include "onnxruntime_dml_context.h"
#endif

#include "onnxruntime_lite_custom_op.h"

static const char* c_OpDomain = "test.customop";

#ifdef USE_DML
#include <wrl/client.h>
#include <core/providers/dml/dml_provider_factory.h>
using Microsoft::WRL::ComPtr;

struct IdentityDML {

  IdentityDML(const OrtApi* ort_api, const OrtKernelInfo* /*info*/) : api(ort_api) {

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

  void Compute(OrtKernelContext* ctx, Ort::Custom::OrtDmlContext* dml_ctx,
               const Ort::Custom::Tensor<float>& input,
               Ort::Custom::Tensor<float>& output) {

    auto* dml_device = dml_ctx->dml_device;
    auto* d3d12_device = dml_ctx->d3d12_device;
    auto* cmd_list = dml_ctx->cmd_list;
    ORT_UNUSED_PARAMETER(dml_device);
    ORT_UNUSED_PARAMETER(d3d12_device);
    ORT_UNUSED_PARAMETER(cmd_list);

    const auto& shape = input.Shape();
    // ORT_ENFORCE(shape.size() <= 8U);
    dml_buffer_tensor_desc.DimensionCount = static_cast<UINT>(shape.size());
    dml_buffer_tensor_desc.TotalTensorSizeInBytes = static_cast<UINT64>(input.NumberOfElement() * sizeof(float));
    for (size_t i = 0; i < shape.size(); ++i) {
      tensor_shape[i] = static_cast<UINT>(shape[i]);
    }

    // Compute will only be called once in the unit test
    dml_device->CreateOperator(
      &dml_op_desc,
      IID_PPV_ARGS(dml_op.GetAddressOf()));

    dml_device->CompileOperator(
      dml_op.Get(),
      DML_EXECUTION_FLAG_NONE,
      IID_PPV_ARGS(dml_compiled_op.GetAddressOf()));

    dml_device->CreateOperatorInitializer(1U,
        &dml_compiled_op,
        IID_PPV_ARGS(dml_op_initializer.GetAddressOf()));

    desc_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc_heap_desc.NumDescriptors = 1;
    desc_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    d3d12_device->CreateDescriptorHeap(
        &desc_heap_desc,
        IID_PPV_ARGS(d3d12_desc_heap.GetAddressOf()));

    // now bind input and output to the op
    // question - should get d3d12_desc_heap from cmd_list?
    dml_binding_table_desc.Dispatchable = dml_op_initializer.Get();
    //dml_binding_table_desc.Dispatchable = dml_op.Get();
    dml_binding_table_desc.CPUDescriptorHandle = d3d12_desc_heap->GetCPUDescriptorHandleForHeapStart();
    dml_binding_table_desc.GPUDescriptorHandle = d3d12_desc_heap->GetGPUDescriptorHandleForHeapStart();
    dml_binding_table_desc.SizeInDescriptors = 1;

    dml_device->CreateBindingTable(
        &dml_binding_table_desc,
        IID_PPV_ARGS(dml_binding_table.GetAddressOf()));

    OrtMemoryInfo mem_info("", OrtAllocatorType::OrtDeviceAllocator, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0));
    OrtAllocator* allocator = {};
    api->KernelContext_GetAllocator(ctx, &mem_info, &allocator);
    //assert(allocator)

    const OrtDmlApi* dml_api;
    api->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml_api));

    ID3D12Resource* input_resource = {};
    auto* input_addr = const_cast<float*>(input.Data());
    dml_api->GetD3D12ResourceFromAllocation(allocator, input_addr, &input_resource);
    //assert(input_resource)

    UINT64 tensorBufferSize = input.NumberOfElement() * sizeof(float);
    DML_BUFFER_BINDING inputBufferBinding{input_resource, 0, tensorBufferSize};
    DML_BINDING_DESC inputBindingDesc{DML_BINDING_TYPE_BUFFER, &inputBufferBinding};
    dml_binding_table->BindInputs(1, &inputBindingDesc);

    ID3D12Resource* output_resource = {};
    auto* output_addr = const_cast<float*>(output.Allocate(input.Shape()));
    dml_api->GetD3D12ResourceFromAllocation(allocator, output_addr, &output_resource);
    //assert(output_resource)

    DML_BUFFER_BINDING outputBufferBinding{output_resource, 0, tensorBufferSize};
    DML_BINDING_DESC outputBindingDesc{DML_BINDING_TYPE_BUFFER, &outputBufferBinding};
    dml_binding_table->BindOutputs(1, &outputBindingDesc);

    dml_cmd_recorder->RecordDispatch(
        cmd_list,
        dml_op_initializer.Get(),
        dml_binding_table.Get());
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

  //ComPtr<ID3D12Resource> input_resource = {};

  ComPtr<IDMLCommandRecorder> dml_cmd_recorder = {};

  const OrtApi* api{};
};

  //constexpr UINT tensorSizes[] = {2, 3};
  //DML_BUFFER_TENSOR_DESC dmlBufferTensorDesc = {};
  //dmlBufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
  //dmlBufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
  //dmlBufferTensorDesc.DimensionCount = 2;
  //dmlBufferTensorDesc.Sizes = tensorSizes;
  //dmlBufferTensorDesc.Strides = nullptr;
  //dmlBufferTensorDesc.TotalTensorSizeInBytes = 2 * 3 * sizeof(float);

  //ComPtr<IDMLOperator> dmlOperator;
  //DML_TENSOR_DESC dmlTensorDesc{};
  //dmlTensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
  //dmlTensorDesc.Desc = &dmlBufferTensorDesc;

  //DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC dmlIdentityOperatorDesc{};
  //dmlIdentityOperatorDesc.InputTensor = &dmlTensorDesc;
  //dmlIdentityOperatorDesc.OutputTensor = &dmlTensorDesc;

  //DML_OPERATOR_DESC dmlOperatorDesc{};
  //dmlOperatorDesc.Type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;
  //dmlOperatorDesc.Desc = &dmlIdentityOperatorDesc;

  //dml_device->CreateOperator(
  //    &dmlOperatorDesc,
  //    IID_PPV_ARGS(dmlOperator.GetAddressOf()));

  //ComPtr<IDMLCompiledOperator> dmlCompiledOperator;
  //dml_device->CompileOperator(
  //    dmlOperator.Get(),
  //    DML_EXECUTION_FLAG_NONE,
  //    IID_PPV_ARGS(dmlCompiledOperator.GetAddressOf()));

  //UINT64 tensorBufferSize{dmlBufferTensorDesc.TotalTensorSizeInBytes};

  //ComPtr<IDMLOperatorInitializer> dmlOperatorInitializer;
  //IDMLCompiledOperator* dmlCompiledOperators[] = {dmlCompiledOperator.Get()};
  //dml_device->CreateOperatorInitializer(
  //    ARRAYSIZE(dmlCompiledOperators),
  //    dmlCompiledOperators,
  //    IID_PPV_ARGS(dmlOperatorInitializer.GetAddressOf()));

  //DML_BINDING_PROPERTIES initializeBindingProperties = dmlOperatorInitializer->GetBindingProperties();
  //DML_BINDING_PROPERTIES executeBindingProperties = dmlCompiledOperator->GetBindingProperties();
  //UINT descriptorCount = std::max(
  //    initializeBindingProperties.RequiredDescriptorCount,
  //    executeBindingProperties.RequiredDescriptorCount);

  //ComPtr<ID3D12DescriptorHeap> descriptorHeap;

  //D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
  //descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  //descriptorHeapDesc.NumDescriptors = descriptorCount;
  //descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

  //d3d12_device->CreateDescriptorHeap(
  //    &descriptorHeapDesc,
  //    IID_PPV_ARGS(descriptorHeap.GetAddressOf()));

  //ID3D12DescriptorHeap* d3D12DescriptorHeaps[] = {descriptorHeap.Get()};
  //ComPtr<ID3D12GraphicsCommandList> command_list;

  //ComPtr<ID3D12CommandAllocator> command_allocator;
  //d3d12_device->CreateCommandAllocator(
  //    D3D12_COMMAND_LIST_TYPE_DIRECT,
  //    IID_PPV_ARGS(command_allocator.ReleaseAndGetAddressOf()));

  //d3d12_device->CreateCommandList(
  //    0,
  //    D3D12_COMMAND_LIST_TYPE_DIRECT,
  //    command_allocator.Get(),
  //    nullptr,
  //    IID_PPV_ARGS(command_list.ReleaseAndGetAddressOf()));

  //command_list->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

  //DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
  //dmlBindingTableDesc.Dispatchable = dmlOperatorInitializer.Get();
  //dmlBindingTableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
  //dmlBindingTableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
  //dmlBindingTableDesc.SizeInDescriptors = descriptorCount;

  //ComPtr<IDMLBindingTable> dmlBindingTable;
  //dml_device->CreateBindingTable(
  //    &dmlBindingTableDesc,
  //    IID_PPV_ARGS(dmlBindingTable.GetAddressOf()));

  //UINT64 temporaryResourceSize = std::max(
  //    initializeBindingProperties.TemporaryResourceSize,
  //    executeBindingProperties.TemporaryResourceSize);
  //UINT64 persistentResourceSize = executeBindingProperties.PersistentResourceSize;

  //ComPtr<ID3D12Resource> temporaryBuffer;
  //if (temporaryResourceSize != 0) {
  //  d3d12_device->CreateCommittedResource(
  //      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
  //      D3D12_HEAP_FLAG_NONE,
  //      &CD3DX12_RESOURCE_DESC::Buffer(temporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
  //      D3D12_RESOURCE_STATE_COMMON,
  //      nullptr,
  //      IID_GRAPHICS_PPV_ARGS(temporaryBuffer.GetAddressOf())));

  //  if (initializeBindingProperties.TemporaryResourceSize != 0) {
  //    DML_BUFFER_BINDING bufferBinding{temporaryBuffer.Get(), 0, temporaryResourceSize};
  //    DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
  //    dmlBindingTable->BindTemporaryResource(&bindingDesc);
  //  }
  //}
//}
#endif

#include <iostream>
#ifdef USE_CUDA
void KernelOne(Ort::Custom::OrtCudaContext* cuda_ctx,
               const Ort::Custom::Tensor<float>& X,
               const Ort::Custom::Tensor<float>& Y,
               Ort::Custom::Tensor<float>& Z) {
  auto input_shape = X.Shape();
  std::cout << "Fetch cuda stream from context" << std::endl;
  cudaStream_t cuda_stream = cuda_ctx->cuda_stream;
  auto z_raw = Z.Allocate(input_shape);
  cuda_add(Z.NumberOfElement(), z_raw, X.Data(), Y.Data(), cuda_stream);
}
#else
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
    for (size_t i = 0; i < size; i++) {
      out[i] = X[i] + Y[i];
    }
  }
};

// legacy custom op registration
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

  using LiteOp = Ort::Custom::OrtLiteCustomOp;

#ifdef USE_CUDA
  static const std::unique_ptr<LiteOp> c_CustomOpOne{Ort::Custom::CreateLiteCustomOp("CustomOpOne", "CUDAExecutionProvider", KernelOne)};
#else
  static const CustomOpOne c_CustomOpOne;
#endif  // !1

  static const std::unique_ptr<LiteOp> c_CustomOpTwo{Ort::Custom::CreateLiteCustomOp("CustomOpTwo", "CPUExecutionProvider", KernelTwo)};
  static const std::unique_ptr<LiteOp> c_MulTopOpFloat{Ort::Custom::CreateLiteCustomOp("MulTop", "CPUExecutionProvider", MulTop<float>)};
  static const std::unique_ptr<LiteOp> c_MulTopOpInt32{Ort::Custom::CreateLiteCustomOp("MulTop", "CPUExecutionProvider", MulTop<int32_t>)};
  static const std::unique_ptr<LiteOp> fus_op_ptr{Ort::Custom::CreateLiteCustomOp("Fuse", "CPUExecutionProvider", Fuse)};
  static const std::unique_ptr<LiteOp> sel_op_ptr{Ort::Custom::CreateLiteCustomOp("Select", "CPUExecutionProvider", Select)};
  static const std::unique_ptr<LiteOp> fil_op_ptr{Ort::Custom::CreateLiteCustomOp("Filter", "CPUExecutionProvider", Filter)};
  static const std::unique_ptr<LiteOp> box_op_ptr{Ort::Custom::CreateLiteCustomOp("Box", "CPUExecutionProvider", Box)};

#ifdef USE_DML
  static const std::unique_ptr<LiteOp> identity_dml_op_ptr{Ort::Custom::CreateLiteCustomOp<IdentityDML>("IdentityDML", "DmlExecutionProvider")};
#endif

  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};
#ifdef USE_CUDA
    domain.Add(c_CustomOpOne.get());
#else
    domain.Add(&c_CustomOpOne);
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
