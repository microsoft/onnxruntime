// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the Onnxruntime side of the bridge to allow providers to be built as a DLL
// It implements onnxruntime::ProviderHost

#include "core/framework/data_types.h"
#include "core/framework/allocatormgr.h"
#include "core/providers/dnnl/dnnl_provider_factory.h"
#include "core/session/inference_session.h"
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#ifdef USE_TENSORRT
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/providers/cuda/cuda_common.h"
#endif
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"
#include "core/platform/env.h"
#include "core/graph/model.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/compute_capability.h"
#include "core/framework/execution_provider.h"
#define PROVIDER_BRIDGE_ORT
#include "core/providers/shared_library/provider_interfaces.h"
#include "onnx/common/stl_backports.h"
#include "core/common/logging/logging.h"
#include "core/common/cpuid_info.h"

// The filename extension for a shared library is different per platform
#ifdef _WIN32
#define LIBRARY_PREFIX
#define LIBRARY_EXTENSION ".dll"
#elif defined(__APPLE__)
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".dylib"
#else
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".so"
#endif

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

namespace onnxruntime {

ProviderHost* g_host{};

struct Provider_OrtDevice_Impl : Provider_OrtDevice {
  OrtDevice v_;
};

struct Provider_OrtMemoryInfo_Impl : Provider_OrtMemoryInfo {
  Provider_OrtMemoryInfo_Impl(const char* name_, OrtAllocatorType type_, OrtDevice device_, int id_, OrtMemType mem_type_) : info_{onnxruntime::make_unique<OrtMemoryInfo>(name_, type_, device_, id_, mem_type_)} {}

  std::unique_ptr<OrtMemoryInfo> info_;
};

struct Provider_IAllocator_Impl : Provider_IAllocator {
  Provider_IAllocator_Impl(AllocatorPtr p) : p_{p} {}

  void* Alloc(size_t size) override { return p_->Alloc(size); }
  void Free(void* p) override { return p_->Free(p); }

  AllocatorPtr p_;
};

struct Provider_IDeviceAllocator_Impl : Provider_IDeviceAllocator {
  Provider_IDeviceAllocator_Impl(std::unique_ptr<IDeviceAllocator> p) : p_{std::move(p)} {}

  void* Alloc(size_t size) override { return p_->Alloc(size); }
  void Free(void* p) override { return p_->Free(p); }

  std::unique_ptr<IDeviceAllocator> p_;
};

struct Provider_TensorShapeProto_Dimension_Iterator_Impl : Provider_TensorShapeProto_Dimension_Iterator {
  Provider_TensorShapeProto_Dimension_Iterator_Impl(google::protobuf::internal::RepeatedPtrIterator<const onnx::TensorShapeProto_Dimension>&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_TensorShapeProto_Dimension_Iterator& p) const override { return v_ != static_cast<const Provider_TensorShapeProto_Dimension_Iterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Provider_TensorShapeProto_Dimension& operator*() override { return *reinterpret_cast<const Provider_TensorShapeProto_Dimension*>(&v_.operator*()); }

  google::protobuf::internal::RepeatedPtrIterator<const onnx::TensorShapeProto_Dimension> v_;
};

struct Provider_NodeAttributes_Iterator_Impl : Provider_NodeAttributes_Iterator {
  Provider_NodeAttributes_Iterator_Impl(NodeAttributes::const_iterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_NodeAttributes_Iterator& p) const override { return v_ != static_cast<const Provider_NodeAttributes_Iterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const std::string& first() const override { return v_->first; }
  const Provider_AttributeProto& second() override { return *reinterpret_cast<const Provider_AttributeProto*>(static_cast<const ONNX_NAMESPACE::AttributeProto*>(&v_->second)); }

  NodeAttributes::const_iterator v_;
};

struct Provider_Node__NodeIterator_Impl : Provider_Node__NodeIterator {
  Provider_Node__NodeIterator_Impl(Node::NodeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_Node__NodeIterator& p) const override { return v_ != static_cast<const Provider_Node__NodeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Provider_Node& operator*() override {
    return *reinterpret_cast<const Provider_Node*>(&*v_);
  }

  Node::NodeConstIterator v_;
};

struct Provider_Node__EdgeIterator_Impl : Provider_Node__EdgeIterator {
  Provider_Node__EdgeIterator_Impl(Node::EdgeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_Node__EdgeIterator& p) const override { return v_ != static_cast<const Provider_Node__EdgeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Provider_Node& GetNode() const override { return *reinterpret_cast<const Provider_Node*>(&v_->GetNode()); }

  int GetSrcArgIndex() const override {
    return v_->GetSrcArgIndex();
  }

  int GetDstArgIndex() const override {
    return v_->GetDstArgIndex();
  }

  Node::EdgeConstIterator v_;
};

struct Provider_OpKernel_Impl : Provider_OpKernel {
  OpKernelInfo op_kernel_info_;
};

struct OpKernel_Translator : OpKernel {
  OpKernel_Translator(const OpKernelInfo& info, Provider_OpKernel* p) : OpKernel{info}, p_{p} {
  }

  Status Compute(OpKernelContext* context) const override {
    return p_->Compute(reinterpret_cast<Provider_OpKernelContext*>(context), *reinterpret_cast<const Provider_OpKernel_Base*>(static_cast<const OpKernel*>(this)));
  }

  std::unique_ptr<Provider_OpKernel> p_;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OpKernel_Translator);
};

struct Provider_IExecutionProvider_Router_Impl : Provider_IExecutionProvider_Router, IExecutionProvider {
  Provider_IExecutionProvider_Router_Impl(Provider_IExecutionProvider* outer, const std::string& type) : IExecutionProvider(type), outer_(outer) {
  }

  virtual ~Provider_IExecutionProvider_Router_Impl() {}

  std::shared_ptr<Provider_KernelRegistry> Provider_GetKernelRegistry() const override {
    auto result = GetKernelRegistry();
    return *reinterpret_cast<std::shared_ptr<Provider_KernelRegistry>*>(&result);
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    auto result = outer_->Provider_GetKernelRegistry();
    return *reinterpret_cast<std::shared_ptr<KernelRegistry>*>(&result);
  }

  std::vector<std::unique_ptr<Provider_ComputeCapability>> Provider_GetCapability(const onnxruntime::Provider_GraphViewer& graph,
                                                                                  const std::vector<const Provider_KernelRegistry*>& kernel_registries) const override {
    auto capabilities_internal = IExecutionProvider::GetCapability(*reinterpret_cast<const GraphViewer*>(&graph), *reinterpret_cast<const std::vector<const KernelRegistry*>*>(&kernel_registries));
    return std::move(*reinterpret_cast<std::vector<std::unique_ptr<Provider_ComputeCapability>>*>(&capabilities_internal));
  }

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const onnxruntime::GraphViewer& graph,
                                                                const std::vector<const KernelRegistry*>& kernel_registries) const override {
    auto provider_result = outer_->Provider_GetCapability(*reinterpret_cast<const Provider_GraphViewer*>(&graph), *reinterpret_cast<const std::vector<const Provider_KernelRegistry*>*>(&kernel_registries));
    return std::move(*reinterpret_cast<std::vector<std::unique_ptr<ComputeCapability>>*>(&provider_result));
  }

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) override {
    return outer_->Provider_Compile(*reinterpret_cast<const std::vector<Provider_Node*>*>(&fused_nodes), node_compute_funcs);
  }

  Provider_AllocatorPtr Provider_GetAllocator(int id, OrtMemType mem_type) const override {
    return std::make_shared<Provider_IAllocator_Impl>(IExecutionProvider::GetAllocator(id, mem_type));
  }

  std::unique_ptr<Provider_IDataTransfer> Provider_GetDataTransfer() const override {
    return std::unique_ptr<Provider_IDataTransfer>(reinterpret_cast<Provider_IDataTransfer*>(IExecutionProvider::GetDataTransfer().release()));
  }

  std::unique_ptr<IDataTransfer> GetDataTransfer() const override {
    return std::unique_ptr<IDataTransfer>(reinterpret_cast<IDataTransfer*>(outer_->Provider_GetDataTransfer().release()));
  }

  void Provider_InsertAllocator(Provider_AllocatorPtr allocator) override {
    IExecutionProvider::InsertAllocator(static_cast<Provider_IAllocator_Impl*>(allocator.get())->p_);
  }

  const logging::Logger* GetLogger() const override { return IExecutionProvider::GetLogger(); }

  std::unique_ptr<Provider_IExecutionProvider> outer_;
};

struct ProviderHostImpl : ProviderHost {
  ProviderHostImpl() {
    DataTypeImpl_GetType_Tensor = &DataTypeImpl::GetType<Tensor>;
    DataTypeImpl_GetType_float = &DataTypeImpl::GetType<float>;
    DataTypeImpl_GetTensorType_float = &DataTypeImpl::GetTensorType<float>;
  }

  std::unique_ptr<Provider_OrtMemoryInfo> OrtMemoryInfo_Create(const char* name_, OrtAllocatorType type_, Provider_OrtDevice* device_, int id_, OrtMemType mem_type_) override {
    return onnxruntime::make_unique<Provider_OrtMemoryInfo_Impl>(name_, type_, device_ ? static_cast<Provider_OrtDevice_Impl*>(device_)->v_ : OrtDevice(), id_, mem_type_);
  }

  Provider_AllocatorPtr CreateAllocator(const Provider_DeviceAllocatorRegistrationInfo& info,
                                        OrtDevice::DeviceId device_id = 0,
                                        bool use_arena = true) override {
    DeviceAllocatorRegistrationInfo info_real{
        info.mem_type, [&info](int value) {
          return std::move(static_cast<Provider_IDeviceAllocator_Impl*>(&*info.factory(value))->p_);
        },
        info.max_mem};

    return std::make_shared<Provider_IAllocator_Impl>(onnxruntime::CreateAllocator(info_real, device_id, use_arena));
  }

  std::unique_ptr<Provider_IDeviceAllocator> CreateCPUAllocator(
      std::unique_ptr<Provider_OrtMemoryInfo> memory_info) override {
    return onnxruntime::make_unique<Provider_IDeviceAllocator_Impl>(
        onnxruntime::make_unique<CPUAllocator>(*static_cast<Provider_OrtMemoryInfo_Impl*>(memory_info.get())->info_));
  };

  std::unique_ptr<Provider_IExecutionProvider_Router> Create_IExecutionProvider_Router(
      Provider_IExecutionProvider* outer, const std::string& type) override {
    return onnxruntime::make_unique<Provider_IExecutionProvider_Router_Impl>(outer, type);
  };

#ifdef USE_TENSORRT
  std::unique_ptr<Provider_IDeviceAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) override {
    return onnxruntime::make_unique<Provider_IDeviceAllocator_Impl>(onnxruntime::make_unique<CUDAAllocator>(device_id, name));
  }

  std::unique_ptr<Provider_IDeviceAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) override {
    return onnxruntime::make_unique<Provider_IDeviceAllocator_Impl>(onnxruntime::make_unique<CUDAPinnedAllocator>(device_id, name));
  }

  std::unique_ptr<Provider_IDataTransfer> CreateGPUDataTransfer() override {
    return std::unique_ptr<Provider_IDataTransfer>(reinterpret_cast<Provider_IDataTransfer*>(new GPUDataTransfer()));
  }

  void cuda__Impl_Cast(const int64_t* input_data, int32_t* output_data, size_t count) override {
    return cuda::Impl_Cast(input_data, output_data, count);
  }

  void cuda__Impl_Cast(const int32_t* input_data, int64_t* output_data, size_t count) override {
    return cuda::Impl_Cast(input_data, output_data, count);
  }

  bool CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) override { return CudaCall<cudaError, false>(cudaError(retCode), exprString, libName, cudaError(successCode), msg); }
  bool CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) override { return CudaCall<cudaError, true>(cudaError(retCode), exprString, libName, cudaError(successCode), msg); }
#endif

  std::string GetEnvironmentVar(const std::string& var_name) override {
    return Env::Default().GetEnvironmentVar(var_name);
  }

  logging::Logger* LoggingManager_GetDefaultLogger() override {
    return const_cast<logging::Logger*>(&logging::LoggingManager::DefaultLogger());
  }

  const std::vector<MLDataType>& DataTypeImpl_AllFixedSizeTensorTypes() override {
    return DataTypeImpl::AllFixedSizeTensorTypes();
  }

  void* HeapAllocate(size_t size) override { return new uint8_t[size]; }
  void HeapFree(void* p) override { delete reinterpret_cast<uint8_t*>(p); }

  bool CPU_HasAVX2() override {
    return CPUIDInfo::GetCPUIDInfo().HasAVX2();
  }

  bool CPU_HasAVX512f() override {
    return CPUIDInfo::GetCPUIDInfo().HasAVX512f();
  }

  void LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file, const char* function, uint32_t line) override {
    return ::onnxruntime::LogRuntimeError(session_id, status, file, function, line);
  }

  // Provider_TypeProto_Tensor
  int32_t Provider_TypeProto_Tensor__elem_type(const Provider_TypeProto_Tensor* p) override { return reinterpret_cast<const ONNX_NAMESPACE::TypeProto_Tensor*>(p)->elem_type(); }

  // Provider_TypeProto
  const Provider_TypeProto_Tensor& Provider_TypeProto__tensor_type(const Provider_TypeProto* p) override { return *reinterpret_cast<const Provider_TypeProto_Tensor*>(&reinterpret_cast<const ONNX_NAMESPACE::TypeProto*>(p)->tensor_type()); }

  // Provider_AttributeProto
  std::unique_ptr<Provider_AttributeProto> Provider_AttributeProto__construct() override { return std::unique_ptr<Provider_AttributeProto>(reinterpret_cast<Provider_AttributeProto*>(new ONNX_NAMESPACE::AttributeProto())); }
  void Provider_AttributeProto__operator_delete(Provider_AttributeProto* p) override { delete reinterpret_cast<ONNX_NAMESPACE::AttributeProto*>(p); }
  void Provider_AttributeProto__operator_assign(Provider_AttributeProto* p, const Provider_AttributeProto& v) override { *reinterpret_cast<ONNX_NAMESPACE::AttributeProto*>(p) = *reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(&v); }

  ONNX_NAMESPACE::AttributeProto_AttributeType Provider_AttributeProto__type(const Provider_AttributeProto* p) override { return reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(p)->type(); }
  int Provider_AttributeProto__ints_size(const Provider_AttributeProto* p) override { return reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(p)->ints_size(); }
  int64_t Provider_AttributeProto__ints(const Provider_AttributeProto* p, int i) override { return reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(p)->ints(i); }
  int64_t Provider_AttributeProto__i(const Provider_AttributeProto* p) override { return reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(p)->i(); }
  float Provider_AttributeProto__f(const Provider_AttributeProto* p) override { return reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(p)->f(); }
  void Provider_AttributeProto__set_s(Provider_AttributeProto* p, const ::std::string& value) override { return reinterpret_cast<ONNX_NAMESPACE::AttributeProto*>(p)->set_s(value); }
  const ::std::string& Provider_AttributeProto__s(const Provider_AttributeProto* p) override { return reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(p)->s(); }
  void Provider_AttributeProto__set_name(Provider_AttributeProto* p, const ::std::string& value) override { return reinterpret_cast<ONNX_NAMESPACE::AttributeProto*>(p)->set_name(value); }
  void Provider_AttributeProto__set_type(Provider_AttributeProto* p, ONNX_NAMESPACE::AttributeProto_AttributeType value) override { return reinterpret_cast<ONNX_NAMESPACE::AttributeProto*>(p)->set_type(value); }
  Provider_TensorProto* Provider_AttributeProto__add_tensors(Provider_AttributeProto* p) override { return reinterpret_cast<Provider_TensorProto*>(reinterpret_cast<ONNX_NAMESPACE::AttributeProto*>(p)->add_tensors()); }

  // Provider_GraphProto
  void Provider_GraphProto__operator_delete(Provider_GraphProto* p) override { delete reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p); }

  Provider_ValueInfoProtos* Provider_GraphProto__mutable_input(Provider_GraphProto* p) override { return reinterpret_cast<Provider_ValueInfoProtos*>(reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p)->mutable_input()); }

  const Provider_ValueInfoProtos& Provider_GraphProto__output(const Provider_GraphProto* p) override { return *reinterpret_cast<const Provider_ValueInfoProtos*>(&reinterpret_cast<const ONNX_NAMESPACE::GraphProto*>(p)->output()); }
  Provider_ValueInfoProtos* Provider_GraphProto__mutable_output(Provider_GraphProto* p) override { return reinterpret_cast<Provider_ValueInfoProtos*>(reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p)->mutable_output()); }

  Provider_ValueInfoProtos* Provider_GraphProto__mutable_value_info(Provider_GraphProto* p) override { return reinterpret_cast<Provider_ValueInfoProtos*>(reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p)->mutable_value_info()); }
  Provider_TensorProtos* Provider_GraphProto__mutable_initializer(Provider_GraphProto* p) override { return reinterpret_cast<Provider_TensorProtos*>(reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p)->mutable_initializer()); }
  Provider_NodeProto* Provider_GraphProto__add_node(Provider_GraphProto* p) override { return reinterpret_cast<Provider_NodeProto*>(reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p)->add_node()); }

  void Provider_GraphProto__operator_assign(Provider_GraphProto* p, const Provider_GraphProto& v) override { *reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p) = *reinterpret_cast<const ONNX_NAMESPACE::GraphProto*>(&v); }

  // Provider_ModelProto
  void Provider_ModelProto__operator_delete(Provider_ModelProto* p) override { delete reinterpret_cast<ONNX_NAMESPACE::ModelProto*>(p); }

  bool Provider_ModelProto__SerializeToString(const Provider_ModelProto* p, std::string& string) override { return reinterpret_cast<const ONNX_NAMESPACE::ModelProto*>(p)->SerializeToString(&string); }
  bool Provider_ModelProto__SerializeToOstream(const Provider_ModelProto* p, std::ostream& output) override { return reinterpret_cast<const ONNX_NAMESPACE::ModelProto*>(p)->SerializeToOstream(&output); }

  const ONNX_NAMESPACE::Provider_GraphProto& Provider_ModelProto__graph(const Provider_ModelProto* p) override { return *reinterpret_cast<const ONNX_NAMESPACE::Provider_GraphProto*>(&reinterpret_cast<const ONNX_NAMESPACE::ModelProto*>(p)->graph()); }
  ONNX_NAMESPACE::Provider_GraphProto* Provider_ModelProto__mutable_graph(Provider_ModelProto* p) override { return reinterpret_cast<ONNX_NAMESPACE::Provider_GraphProto*>(reinterpret_cast<ONNX_NAMESPACE::ModelProto*>(p)->mutable_graph()); }

  void Provider_ModelProto__set_ir_version(Provider_ModelProto* p, int64_t value) override { reinterpret_cast<ONNX_NAMESPACE::ModelProto*>(p)->set_ir_version(value); }

  // Provider_TensorProto
  void Provider_TensorProto__operator_delete(Provider_TensorProto* p) override { delete reinterpret_cast<ONNX_NAMESPACE::TensorProto*>(p); }
  void Provider_TensorProto__operator_assign(Provider_TensorProto* p, const Provider_TensorProto& v) override { *reinterpret_cast<ONNX_NAMESPACE::TensorProto*>(p) = *reinterpret_cast<const ONNX_NAMESPACE::TensorProto*>(&v); }

  // Provider_TensorProtos
  Provider_TensorProto* Provider_TensorProtos__Add(Provider_TensorProtos* p) override { return reinterpret_cast<Provider_TensorProto*>(reinterpret_cast<google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::TensorProto>*>(p)->Add()); }

  // Provider_TensorShapeProto_Dimension
  const std::string& Provider_TensorShapeProto_Dimension__dim_param(const Provider_TensorShapeProto_Dimension* p) override {
    return reinterpret_cast<const ONNX_NAMESPACE::TensorShapeProto_Dimension*>(p)->dim_param();
  }

  // Provider_TensorShapeProto_Dimensions
  std::unique_ptr<Provider_TensorShapeProto_Dimension_Iterator> Provider_TensorShapeProto_Dimensions__begin(const Provider_TensorShapeProto_Dimensions* p) override {
    return onnxruntime::make_unique<Provider_TensorShapeProto_Dimension_Iterator_Impl>(reinterpret_cast<const google::protobuf::RepeatedPtrField<::ONNX_NAMESPACE::TensorShapeProto_Dimension>*>(p)->begin());
  }

  std::unique_ptr<Provider_TensorShapeProto_Dimension_Iterator> Provider_TensorShapeProto_Dimensions__end(const Provider_TensorShapeProto_Dimensions* p) override {
    return onnxruntime::make_unique<Provider_TensorShapeProto_Dimension_Iterator_Impl>(reinterpret_cast<const google::protobuf::RepeatedPtrField<::ONNX_NAMESPACE::TensorShapeProto_Dimension>*>(p)->end());
  }

  // Provider_TensorShapeProto
  int Provider_TensorShapeProto__dim_size(const Provider_TensorShapeProto* p) override { return reinterpret_cast<const ONNX_NAMESPACE::TensorShapeProto*>(p)->dim_size(); }
  const Provider_TensorShapeProto_Dimensions& Provider_TensorShapeProto__dim(const Provider_TensorShapeProto* p) override { return *reinterpret_cast<const Provider_TensorShapeProto_Dimensions*>(&reinterpret_cast<const ONNX_NAMESPACE::TensorShapeProto*>(p)->dim()); }

  // Provider_ValueInfoProto
  const Provider_TypeProto& Provider_ValueInfoProto__type(const Provider_ValueInfoProto* p) override { return *reinterpret_cast<const Provider_TypeProto*>(&reinterpret_cast<const ONNX_NAMESPACE::ValueInfoProto*>(p)->type()); }
  virtual void Provider_ValueInfoProto__operator_assign(Provider_ValueInfoProto* p, const Provider_ValueInfoProto& v) override { *reinterpret_cast<ONNX_NAMESPACE::ValueInfoProto*>(p) = *reinterpret_cast<const ONNX_NAMESPACE::ValueInfoProto*>(&v); }

  // Provider_ValueInfoProtos
  Provider_ValueInfoProto* Provider_ValueInfoProtos__Add(Provider_ValueInfoProtos* p) override { return reinterpret_cast<Provider_ValueInfoProto*>(reinterpret_cast<google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::ValueInfoProto>*>(p)->Add()); }

  const Provider_ValueInfoProto& Provider_ValueInfoProtos__operator_array(const Provider_ValueInfoProtos* p, int index) override { return *reinterpret_cast<const Provider_ValueInfoProto*>(&(*reinterpret_cast<const google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::ValueInfoProto>*>(p))[index]); }

  // Provider_ComputeCapability
  std::unique_ptr<Provider_ComputeCapability> Provider_ComputeCapability__construct(std::unique_ptr<Provider_IndexedSubGraph> t_sub_graph) override { return std::unique_ptr<Provider_ComputeCapability>(reinterpret_cast<Provider_ComputeCapability*>(new ComputeCapability(std::unique_ptr<IndexedSubGraph>(reinterpret_cast<IndexedSubGraph*>(t_sub_graph.release()))))); }
  void Provider_ComputeCapability__operator_delete(Provider_ComputeCapability* p) override { delete reinterpret_cast<ComputeCapability*>(p); }
  std::unique_ptr<Provider_IndexedSubGraph>& Provider_ComputeCapability__SubGraph(Provider_ComputeCapability* p) override { return *reinterpret_cast<std::unique_ptr<Provider_IndexedSubGraph>*>(&reinterpret_cast<ComputeCapability*>(p)->sub_graph); }

  // Provider_DataTransferManager
  Status Provider_DataTransferManager__CopyTensor(const Provider_DataTransferManager* p, const Provider_Tensor& src, Provider_Tensor& dst, int exec_queue_id) override { return reinterpret_cast<const DataTransferManager*>(p)->CopyTensor(*reinterpret_cast<const Tensor*>(&src), *reinterpret_cast<Tensor*>(&dst), exec_queue_id); }

  // Provider_IDataTransfer
  void Provider_IDataTransfer__operator_delete(Provider_IDataTransfer* p) override { delete reinterpret_cast<Provider_IDataTransfer*>(p); }

  // Provider_IndexedSubGraph_MetaDef
  std::unique_ptr<Provider_IndexedSubGraph_MetaDef> Provider_IndexedSubGraph_MetaDef__construct() override { return std::unique_ptr<Provider_IndexedSubGraph_MetaDef>(reinterpret_cast<Provider_IndexedSubGraph_MetaDef*>(new IndexedSubGraph::MetaDef())); }
  void Provider_IndexedSubGraph_MetaDef__operator_delete(Provider_IndexedSubGraph_MetaDef* p) override { delete reinterpret_cast<IndexedSubGraph::MetaDef*>(p); }

  std::string& Provider_IndexedSubGraph_MetaDef__name(Provider_IndexedSubGraph_MetaDef* p) override { return reinterpret_cast<IndexedSubGraph::MetaDef*>(p)->name; }
  std::string& Provider_IndexedSubGraph_MetaDef__domain(Provider_IndexedSubGraph_MetaDef* p) override { return reinterpret_cast<IndexedSubGraph::MetaDef*>(p)->domain; }
  int& Provider_IndexedSubGraph_MetaDef__since_version(Provider_IndexedSubGraph_MetaDef* p) override { return reinterpret_cast<IndexedSubGraph::MetaDef*>(p)->since_version; }
  ONNX_NAMESPACE::OperatorStatus& Provider_IndexedSubGraph_MetaDef__status(Provider_IndexedSubGraph_MetaDef* p) override { return reinterpret_cast<IndexedSubGraph::MetaDef*>(p)->status; }
  std::vector<std::string>& Provider_IndexedSubGraph_MetaDef__inputs(Provider_IndexedSubGraph_MetaDef* p) override { return reinterpret_cast<IndexedSubGraph::MetaDef*>(p)->inputs; }
  std::vector<std::string>& Provider_IndexedSubGraph_MetaDef__outputs(Provider_IndexedSubGraph_MetaDef* p) override { return reinterpret_cast<IndexedSubGraph::MetaDef*>(p)->outputs; }
  Provider_NodeAttributes& Provider_IndexedSubGraph_MetaDef__attributes(Provider_IndexedSubGraph_MetaDef* p) override { return *reinterpret_cast<Provider_NodeAttributes*>(&reinterpret_cast<IndexedSubGraph::MetaDef*>(p)->attributes); }
  std::string& Provider_IndexedSubGraph_MetaDef__doc_string(Provider_IndexedSubGraph_MetaDef* p) override { return reinterpret_cast<IndexedSubGraph::MetaDef*>(p)->doc_string; }

  // Provider_IndexedSubGraph
  std::unique_ptr<Provider_IndexedSubGraph> Provider_IndexedSubGraph__construct() override { return std::unique_ptr<Provider_IndexedSubGraph>(reinterpret_cast<Provider_IndexedSubGraph*>(new IndexedSubGraph())); }
  void Provider_IndexedSubGraph__operator_delete(Provider_IndexedSubGraph* p) override { delete reinterpret_cast<IndexedSubGraph*>(p); }

  std::vector<onnxruntime::NodeIndex>& Provider_IndexedSubGraph__Nodes(Provider_IndexedSubGraph* p) override { return reinterpret_cast<IndexedSubGraph*>(p)->nodes; }

  void Provider_IndexedSubGraph__SetMetaDef(Provider_IndexedSubGraph* p, std::unique_ptr<Provider_IndexedSubGraph_MetaDef>&& meta_def_) override { return reinterpret_cast<IndexedSubGraph*>(p)->SetMetaDef(std::move(*reinterpret_cast<std::unique_ptr<IndexedSubGraph::MetaDef>*>(&meta_def_))); }
  const Provider_IndexedSubGraph_MetaDef* Provider_IndexedSubGraph__GetMetaDef(const Provider_IndexedSubGraph* p) override { return reinterpret_cast<const Provider_IndexedSubGraph_MetaDef*>(reinterpret_cast<const IndexedSubGraph*>(p)->GetMetaDef()); }

  // Provider_KernelDef
  void Provider_KernelDef__operator_delete(Provider_KernelDef* p) override { delete reinterpret_cast<KernelDef*>(p); }

  // Provider_KernelDefBuilder
  std::unique_ptr<Provider_KernelDefBuilder> Provider_KernelDefBuilder__construct() override { return std::unique_ptr<Provider_KernelDefBuilder>(reinterpret_cast<Provider_KernelDefBuilder*>(new KernelDefBuilder())); }
  void Provider_KernelDefBuilder__operator_delete(Provider_KernelDefBuilder* p) override { delete reinterpret_cast<KernelDefBuilder*>(p); }

  void Provider_KernelDefBuilder__SetName(Provider_KernelDefBuilder* p, const char* op_name) override { reinterpret_cast<KernelDefBuilder*>(p)->SetName(op_name); }
  void Provider_KernelDefBuilder__SetDomain(Provider_KernelDefBuilder* p, const char* domain) override { reinterpret_cast<KernelDefBuilder*>(p)->SetDomain(domain); }
  void Provider_KernelDefBuilder__SinceVersion(Provider_KernelDefBuilder* p, int since_version) override { reinterpret_cast<KernelDefBuilder*>(p)->SinceVersion(since_version); }
  void Provider_KernelDefBuilder__Provider(Provider_KernelDefBuilder* p, const char* provider_type) override { reinterpret_cast<KernelDefBuilder*>(p)->Provider(provider_type); }
  void Provider_KernelDefBuilder__TypeConstraint(Provider_KernelDefBuilder* p, const char* arg_name, MLDataType supported_type) override { reinterpret_cast<KernelDefBuilder*>(p)->TypeConstraint(arg_name, supported_type); }
  void Provider_KernelDefBuilder__TypeConstraint(Provider_KernelDefBuilder* p, const char* arg_name, const std::vector<MLDataType>& supported_types) override { reinterpret_cast<KernelDefBuilder*>(p)->TypeConstraint(arg_name, supported_types); }
  void Provider_KernelDefBuilder__InputMemoryType(Provider_KernelDefBuilder* p, OrtMemType type, int input_index) override { reinterpret_cast<KernelDefBuilder*>(p)->InputMemoryType(type, input_index); }
  void Provider_KernelDefBuilder__OutputMemoryType(Provider_KernelDefBuilder* p, OrtMemType type, int input_index) override { reinterpret_cast<KernelDefBuilder*>(p)->OutputMemoryType(type, input_index); }
  void Provider_KernelDefBuilder__ExecQueueId(Provider_KernelDefBuilder* p, int queue_id) override { reinterpret_cast<KernelDefBuilder*>(p)->ExecQueueId(queue_id); }

  std::unique_ptr<Provider_KernelDef> Provider_KernelDefBuilder__Build(Provider_KernelDefBuilder* p) override {
    return std::unique_ptr<Provider_KernelDef>(reinterpret_cast<Provider_KernelDef*>(reinterpret_cast<KernelDefBuilder*>(p)->Build().release()));
  }

  // Provider_KernelRegistry
  std::shared_ptr<Provider_KernelRegistry> Provider_KernelRegistry__construct() override {
    auto result = std::make_shared<KernelRegistry>();
    return *reinterpret_cast<std::shared_ptr<Provider_KernelRegistry>*>(&result);
  }
  void Provider_KernelRegistry__operator_delete(Provider_KernelRegistry* p) override { delete reinterpret_cast<KernelRegistry*>(p); }
  Status Provider_KernelRegistry__Register(Provider_KernelRegistry* p, Provider_KernelCreateInfo&& create_info) override {
    KernelCreateInfo info_real(std::move(*reinterpret_cast<std::unique_ptr<KernelDef>*>(&create_info.kernel_def)),
                               [kernel_create_func = create_info.kernel_create_func](const OpKernelInfo& info) -> OpKernel* {
                                 return new OpKernel_Translator(info, kernel_create_func(*reinterpret_cast<const Provider_OpKernelInfo*>(&info)));
                               });
    return reinterpret_cast<KernelRegistry*>(p)->Register(std::move(info_real));
  }

  // Provider_Function
  const Provider_Graph& Provider_Function__Body(const Provider_Function* p) override { return *reinterpret_cast<const Provider_Graph*>(&reinterpret_cast<const Function*>(p)->Body()); }

  // Provider_Node
  const std::string& Provider_Node__Name(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->Name(); }
  const std::string& Provider_Node__Description(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->Description(); }
  const std::string& Provider_Node__Domain(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->Domain(); }
  const std::string& Provider_Node__OpType(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->OpType(); }

  const Provider_Function* Provider_Node__GetFunctionBody(const Provider_Node* p) noexcept override { return reinterpret_cast<const Provider_Function*>(reinterpret_cast<const Node*>(p)->GetFunctionBody()); }

  ConstPointerContainer<std::vector<Provider_NodeArg*>> Provider_Node__ImplicitInputDefs(const Provider_Node* p) noexcept override {
    auto result = reinterpret_cast<const Node*>(p)->ImplicitInputDefs();
    return *reinterpret_cast<ConstPointerContainer<std::vector<Provider_NodeArg*>>*>(&result);
  }

  ConstPointerContainer<std::vector<Provider_NodeArg*>> Provider_Node__InputDefs(const Provider_Node* p) noexcept override {
    auto result = reinterpret_cast<const Node*>(p)->InputDefs();
    return *reinterpret_cast<ConstPointerContainer<std::vector<Provider_NodeArg*>>*>(&result);
  }
  ConstPointerContainer<std::vector<Provider_NodeArg*>> Provider_Node__OutputDefs(const Provider_Node* p) noexcept override {
    auto result = reinterpret_cast<const Node*>(p)->OutputDefs();
    return *reinterpret_cast<ConstPointerContainer<std::vector<Provider_NodeArg*>>*>(&result);
  }

  NodeIndex Provider_Node__Index(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->Index(); }

  void Provider_Node__ToProto(const Provider_Node* p, Provider_NodeProto& proto, bool update_subgraphs = false) override { reinterpret_cast<const Node*>(p)->ToProto(*reinterpret_cast<ONNX_NAMESPACE::NodeProto*>(&proto), update_subgraphs); }

  const Provider_NodeAttributes& Provider_Node__GetAttributes(const Provider_Node* p) noexcept override { return *reinterpret_cast<const Provider_NodeAttributes*>(&reinterpret_cast<const Node*>(p)->GetAttributes()); }
  size_t Provider_Node__GetInputEdgesCount(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->GetInputEdgesCount(); }
  size_t Provider_Node__GetOutputEdgesCount(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->GetOutputEdgesCount(); }

  std::unique_ptr<Provider_Node__NodeIterator> Provider_Node__InputNodesBegin(const Provider_Node* p) noexcept override { return onnxruntime::make_unique<Provider_Node__NodeIterator_Impl>(reinterpret_cast<const Node*>(p)->InputNodesBegin()); }
  std::unique_ptr<Provider_Node__NodeIterator> Provider_Node__InputNodesEnd(const Provider_Node* p) noexcept override { return onnxruntime::make_unique<Provider_Node__NodeIterator_Impl>(reinterpret_cast<const Node*>(p)->InputNodesEnd()); }

  std::unique_ptr<Provider_Node__EdgeIterator> Provider_Node__OutputEdgesBegin(const Provider_Node* p) noexcept override { return onnxruntime::make_unique<Provider_Node__EdgeIterator_Impl>(reinterpret_cast<const Node*>(p)->OutputEdgesBegin()); }
  std::unique_ptr<Provider_Node__EdgeIterator> Provider_Node__OutputEdgesEnd(const Provider_Node* p) noexcept override { return onnxruntime::make_unique<Provider_Node__EdgeIterator_Impl>(reinterpret_cast<const Node*>(p)->OutputEdgesEnd()); }

  // Provider_NodeArg
  const std::string& Provider_NodeArg__Name(const Provider_NodeArg* p) noexcept override { return reinterpret_cast<const NodeArg*>(p)->Name(); }
  const ONNX_NAMESPACE::Provider_TensorShapeProto* Provider_NodeArg__Shape(const Provider_NodeArg* p) override { return reinterpret_cast<const ONNX_NAMESPACE::Provider_TensorShapeProto*>(reinterpret_cast<const NodeArg*>(p)->Shape()); }
  ONNX_NAMESPACE::DataType Provider_NodeArg__Type(const Provider_NodeArg* p) noexcept override { return reinterpret_cast<const NodeArg*>(p)->Type(); }
  const Provider_NodeArgInfo& Provider_NodeArg__ToProto(const Provider_NodeArg* p) noexcept override { return *reinterpret_cast<const Provider_NodeArgInfo*>(&reinterpret_cast<const NodeArg*>(p)->ToProto()); }
  bool Provider_NodeArg__Exists(const Provider_NodeArg* p) const noexcept override { return reinterpret_cast<const NodeArg*>(p)->Exists(); }
  const ONNX_NAMESPACE::Provider_TypeProto* Provider_NodeArg__TypeAsProto(const Provider_NodeArg* p) noexcept override { return reinterpret_cast<const ONNX_NAMESPACE::Provider_TypeProto*>(reinterpret_cast<const NodeArg*>(p)->TypeAsProto()); }

  // Provider_NodeAttributes
  std::unique_ptr<Provider_NodeAttributes> Provider_NodeAttributes__construct() override { return std::unique_ptr<Provider_NodeAttributes>(reinterpret_cast<Provider_NodeAttributes*>(new NodeAttributes())); }
  void Provider_NodeAttributes__operator_delete(Provider_NodeAttributes* p) noexcept override { delete reinterpret_cast<NodeAttributes*>(p); }
  size_t Provider_NodeAttributes__size(const Provider_NodeAttributes* p) override { return reinterpret_cast<const NodeAttributes*>(p)->size(); }
  void Provider_NodeAttributes__clear(Provider_NodeAttributes* p) noexcept override { return reinterpret_cast<NodeAttributes*>(p)->clear(); }
  Provider_AttributeProto& Provider_NodeAttributes__operator_array(Provider_NodeAttributes* p, const std::string& string) override { return *reinterpret_cast<Provider_AttributeProto*>(&(*reinterpret_cast<NodeAttributes*>(p))[string]); }
  void Provider_NodeAttributes__operator_assign(Provider_NodeAttributes* p, const Provider_NodeAttributes& v) override { *reinterpret_cast<NodeAttributes*>(p) = *reinterpret_cast<const NodeAttributes*>(&v); }

  std::unique_ptr<Provider_NodeAttributes_Iterator> Provider_NodeAttributes__begin(const Provider_NodeAttributes* p) override {
    return onnxruntime::make_unique<Provider_NodeAttributes_Iterator_Impl>(reinterpret_cast<const NodeAttributes*>(p)->begin());
  }
  std::unique_ptr<Provider_NodeAttributes_Iterator> Provider_NodeAttributes__end(const Provider_NodeAttributes* p) override {
    return onnxruntime::make_unique<Provider_NodeAttributes_Iterator_Impl>(reinterpret_cast<const NodeAttributes*>(p)->end());
  }
  std::unique_ptr<Provider_NodeAttributes_Iterator> Provider_NodeAttributes__find(const Provider_NodeAttributes* p, const std::string& key) override {
    return onnxruntime::make_unique<Provider_NodeAttributes_Iterator_Impl>(reinterpret_cast<const NodeAttributes*>(p)->find(key));
  }
  void Provider_NodeAttributes__insert(Provider_NodeAttributes* p, const Provider_NodeAttributes& v) override {
    auto& nodes = *reinterpret_cast<const NodeAttributes*>(&v);
    return reinterpret_cast<NodeAttributes*>(p)->insert(nodes.begin(), nodes.end());
  }

  // Provider_Model
  void Provider_Model__operator_delete(Provider_Model* p) override { delete reinterpret_cast<Model*>(p); }
  Provider_Graph& Provider_Model__MainGraph(Provider_Model* p) override { return *reinterpret_cast<Provider_Graph*>(&reinterpret_cast<Model*>(p)->MainGraph()); }
  std::unique_ptr<Provider_ModelProto> Provider_Model__ToProto(Provider_Model* p) override { return std::unique_ptr<Provider_ModelProto>(reinterpret_cast<Provider_ModelProto*>(new ONNX_NAMESPACE::ModelProto(reinterpret_cast<Model*>(p)->ToProto()))); }

  // Provider_Graph
  std::unique_ptr<Provider_GraphViewer> Provider_Graph__CreateGraphViewer(const Provider_Graph* p) override {
    return std::unique_ptr<Provider_GraphViewer>(reinterpret_cast<Provider_GraphViewer*>(new GraphViewer(*reinterpret_cast<const Graph*>(p))));
  }

  std::unique_ptr<Provider_GraphProto> Provider_Graph__ToGraphProto(const Provider_Graph* p) override {
    return std::unique_ptr<Provider_GraphProto>(reinterpret_cast<Provider_GraphProto*>(new ONNX_NAMESPACE::GraphProto(reinterpret_cast<const Graph*>(p)->ToGraphProto())));
  }

  Provider_NodeArg& Provider_Graph__GetOrCreateNodeArg(Provider_Graph* p, const std::string& name, const Provider_TypeProto* p_arg_type) override {
    return *reinterpret_cast<Provider_NodeArg*>(&reinterpret_cast<Graph*>(p)->GetOrCreateNodeArg(name, reinterpret_cast<const ONNX_NAMESPACE::TypeProto*>(p_arg_type)));
  }

  Status Provider_Graph__Resolve(Provider_Graph* p) override { return reinterpret_cast<Graph*>(p)->Resolve(); }
  void Provider_Graph__AddInitializedTensor(Provider_Graph* p, const Provider_TensorProto& tensor) override { reinterpret_cast<Graph*>(p)->AddInitializedTensor(*reinterpret_cast<const ONNX_NAMESPACE::TensorProto*>(&tensor)); }
  Provider_Node& Provider_Graph__AddNode(Provider_Graph* p, const std::string& name, const std::string& op_type, const std::string& description, const std::vector<Provider_NodeArg*>& input_args, const std::vector<Provider_NodeArg*>& output_args, const Provider_NodeAttributes* attributes, const std::string& domain) override {
    return *reinterpret_cast<Provider_Node*>(&reinterpret_cast<Graph*>(p)->AddNode(name, op_type, description, *reinterpret_cast<const std::vector<NodeArg*>*>(&input_args), *reinterpret_cast<const std::vector<NodeArg*>*>(&output_args), reinterpret_cast<const NodeAttributes*>(attributes), domain));
  }

  const std::vector<const Provider_NodeArg*>& Provider_Graph__GetOutputs(const Provider_Graph* p) noexcept override { return *reinterpret_cast<const std::vector<const Provider_NodeArg*>*>(&reinterpret_cast<const Graph*>(p)->GetOutputs()); }
  void Provider_Graph__SetOutputs(Provider_Graph* p, const std::vector<const Provider_NodeArg*>& outputs) override { reinterpret_cast<Graph*>(p)->SetOutputs(*reinterpret_cast<const std::vector<const NodeArg*>*>(&outputs)); }

  const std::vector<const Provider_NodeArg*>& Provider_Graph__GetInputs(const Provider_Graph* p) noexcept override { return *reinterpret_cast<const std::vector<const Provider_NodeArg*>*>(&reinterpret_cast<const Graph*>(p)->GetInputs()); }
  bool Provider_Graph__GetInitializedTensor(const Provider_Graph* p, const std::string& tensor_name, const Provider_TensorProto*& value) override { return reinterpret_cast<const Graph*>(p)->GetInitializedTensor(tensor_name, reinterpret_cast<const ONNX_NAMESPACE::TensorProto*&>(value)); }

  // Provider_GraphViewer
  void Provider_GraphViewer__operator_delete(Provider_GraphViewer* p) override { delete reinterpret_cast<GraphViewer*>(p); }
  std::unique_ptr<Provider_Model> Provider_GraphViewer__CreateModel(const Provider_GraphViewer* p, const logging::Logger& logger) override {
    auto& graph_viewer = *reinterpret_cast<const GraphViewer*>(p);
    return std::unique_ptr<Provider_Model>(reinterpret_cast<Provider_Model*>(new Model(graph_viewer.Name(), true, ModelMetaData(), PathString(),
                                                                                       IOnnxRuntimeOpSchemaRegistryList(), graph_viewer.DomainToVersionMap(),
                                                                                       std::vector<ONNX_NAMESPACE::FunctionProto>(), logger)));
  }

  const std::string& Provider_GraphViewer__Name(const Provider_GraphViewer* p) noexcept override { return reinterpret_cast<const GraphViewer*>(p)->Name(); }

  const Provider_Node* Provider_GraphViewer__GetNode(const Provider_GraphViewer* p, NodeIndex node_index) override { return reinterpret_cast<const Provider_Node*>(reinterpret_cast<const GraphViewer*>(p)->GetNode(node_index)); }
  const Provider_NodeArg* Provider_GraphViewer__GetNodeArg(const Provider_GraphViewer* p, const std::string& name) override { return reinterpret_cast<const Provider_NodeArg*>(reinterpret_cast<const GraphViewer*>(p)->GetNodeArg(name)); }

  bool Provider_GraphViewer__IsSubgraph(const Provider_GraphViewer* p) override { return reinterpret_cast<const GraphViewer*>(p)->IsSubgraph(); }
  int Provider_GraphViewer__NumberOfNodes(const Provider_GraphViewer* p) noexcept override { return reinterpret_cast<const GraphViewer*>(p)->NumberOfNodes(); }
  int Provider_GraphViewer__MaxNodeIndex(const Provider_GraphViewer* p) noexcept override { return reinterpret_cast<const GraphViewer*>(p)->MaxNodeIndex(); }

  const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetInputs(const Provider_GraphViewer* p) noexcept override { return *reinterpret_cast<const std::vector<const Provider_NodeArg*>*>(&reinterpret_cast<const GraphViewer*>(p)->GetInputs()); }
  const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetOutputs(const Provider_GraphViewer* p) noexcept override { return *reinterpret_cast<const std::vector<const Provider_NodeArg*>*>(&reinterpret_cast<const GraphViewer*>(p)->GetOutputs()); }
  const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetValueInfo(const Provider_GraphViewer* p) noexcept override { return *reinterpret_cast<const std::vector<const Provider_NodeArg*>*>(&reinterpret_cast<const GraphViewer*>(p)->GetValueInfo()); }

  const Provider_InitializedTensorSet& Provider_GraphViewer__GetAllInitializedTensors(const Provider_GraphViewer* p) override { return *reinterpret_cast<const Provider_InitializedTensorSet*>(&reinterpret_cast<const GraphViewer*>(p)->GetAllInitializedTensors()); }
  bool Provider_GraphViewer__GetInitializedTensor(const Provider_GraphViewer* p, const std::string& tensor_name, const Provider_TensorProto*& value) override { return reinterpret_cast<const GraphViewer*>(p)->GetInitializedTensor(tensor_name, reinterpret_cast<const ONNX_NAMESPACE::TensorProto*&>(value)); }

  const std::unordered_map<std::string, int>& Provider_GraphViewer__DomainToVersionMap(const Provider_GraphViewer* p) override { return reinterpret_cast<const GraphViewer*>(p)->DomainToVersionMap(); }

  const std::vector<NodeIndex>& Provider_GraphViewer__GetNodesInTopologicalOrder(const Provider_GraphViewer* p) override { return reinterpret_cast<const GraphViewer*>(p)->GetNodesInTopologicalOrder(); }

  // Provider_OpKernel_Base
  const Provider_OpKernelInfo& Provider_OpKernel_Base__GetInfo(const Provider_OpKernel_Base* p) override { return *reinterpret_cast<const Provider_OpKernelInfo*>(&reinterpret_cast<const OpKernel*>(p)->Info()); }

  // Provider_OpKernelContext
  const Provider_Tensor* Provider_OpKernelContext__Input_Tensor(const Provider_OpKernelContext* p, int index) override { return reinterpret_cast<const Provider_Tensor*>(reinterpret_cast<const OpKernelContext*>(p)->Input<Tensor>(index)); }
  Provider_Tensor* Provider_OpKernelContext__Output(Provider_OpKernelContext* p, int index, const TensorShape& shape) override { return reinterpret_cast<Provider_Tensor*>(reinterpret_cast<OpKernelContext*>(p)->Output(index, shape)); }

  // Provider_OpKernelInfo
  Status Provider_OpKernelInfo__GetAttr_int64(const Provider_OpKernelInfo* p, const std::string& name, int64_t* value) override { return reinterpret_cast<const OpKernelInfo*>(p)->GetAttr(name, value); }
  Status Provider_OpKernelInfo__GetAttr_float(const Provider_OpKernelInfo* p, const std::string& name, float* value) override { return reinterpret_cast<const OpKernelInfo*>(p)->GetAttr(name, value); }

  const Provider_DataTransferManager& Provider_OpKernelInfo__GetDataTransferManager(const Provider_OpKernelInfo* p) noexcept override { return *reinterpret_cast<const Provider_DataTransferManager*>(&reinterpret_cast<const OpKernelInfo*>(p)->GetDataTransferManager()); }
  int Provider_OpKernelInfo__GetKernelDef_ExecQueueId(const Provider_OpKernelInfo* p) noexcept override { return reinterpret_cast<const OpKernelInfo*>(p)->GetKernelDef().ExecQueueId(); }

  // Provider_Tensor
  float* Provider_Tensor__MutableData_float(Provider_Tensor* p) override { return reinterpret_cast<Tensor*>(p)->MutableData<float>(); }
  const float* Provider_Tensor__Data_float(const Provider_Tensor* p) override { return reinterpret_cast<const Tensor*>(p)->Data<float>(); }
  const TensorShape& Provider_Tensor__Shape(const Provider_Tensor* p) override { return reinterpret_cast<const Tensor*>(p)->Shape(); }

} provider_host_;

struct ProviderSharedLibrary {
  ProviderSharedLibrary() {
    std::string full_path = Env::Default().GetRuntimePath() + std::string(LIBRARY_PREFIX "onnxruntime_providers_shared" LIBRARY_EXTENSION);
    auto error = Env::Default().LoadDynamicLibrary(full_path, &handle_);
    if (!error.IsOK()) {
      LOGS_DEFAULT(ERROR) << error.ErrorMessage();
      return;
    }

    void (*PProvider_SetHost)(void*);
    Env::Default().GetSymbolFromLibrary(handle_, "Provider_SetHost", (void**)&PProvider_SetHost);

    PProvider_SetHost(&provider_host_);
  }

  ~ProviderSharedLibrary() {
    Env::Default().UnloadDynamicLibrary(handle_);
  }

  void* handle_{};

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ProviderSharedLibrary);
};

bool EnsureSharedProviderLibrary() {
  static ProviderSharedLibrary shared_library;
  return shared_library.handle_;
}

struct ProviderLibrary {
  ProviderLibrary(const char* filename) {
    if (!EnsureSharedProviderLibrary())
      return;

    std::string full_path = Env::Default().GetRuntimePath() + std::string(filename);
    auto error = Env::Default().LoadDynamicLibrary(full_path, &handle_);
    if (!error.IsOK()) {
      LOGS_DEFAULT(ERROR) << error.ErrorMessage();
      return;
    }

    Provider* (*PGetProvider)();
    Env::Default().GetSymbolFromLibrary(handle_, "GetProvider", (void**)&PGetProvider);

    provider_ = PGetProvider();
  }

  ~ProviderLibrary() {
    Env::Default().UnloadDynamicLibrary(handle_);
  }

  Provider* provider_{};
  void* handle_{};

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ProviderLibrary);
};

// This class translates the IExecutionProviderFactory interface to work with the interface providers implement
struct IExecutionProviderFactory_Translator : IExecutionProviderFactory {
  IExecutionProviderFactory_Translator(std::shared_ptr<Provider_IExecutionProviderFactory> p) : p_{p} {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    auto provider = p_->CreateProvider();
    return std::unique_ptr<IExecutionProvider>(static_cast<Provider_IExecutionProvider_Router_Impl*>(provider.release()->p_));
  }

  std::shared_ptr<Provider_IExecutionProviderFactory> p_;
};

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(int device_id) {
  static ProviderLibrary library(LIBRARY_PREFIX "onnxruntime_providers_dnnl" LIBRARY_EXTENSION);
  if (!library.provider_)
    return nullptr;

#if defined(_WIN32) && !defined(_OPENMP)
  {
    // We crash when unloading DNNL on Windows when OpenMP also unloads (As there are threads
    // still running code inside the openmp runtime DLL if OMP_WAIT_POLICY is set to ACTIVE).
    // To avoid this, we pin the OpenMP DLL so that it unloads as late as possible.
    HMODULE handle{};
#ifdef _DEBUG
    constexpr const char* dll_name = "vcomp140d.dll";
#else
    constexpr const char* dll_name = "vcomp140.dll";
#endif
    ::GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_PIN, dll_name, &handle);
    assert(handle);  // It should exist
  }
#endif

  //return std::make_shared<onnxruntime::MkldnnProviderFactory>(device_id);
  //TODO: This is apparently a bug. The constructor parameter is create-arena-flag, not the device-id
  return std::make_shared<IExecutionProviderFactory_Translator>(library.provider_->CreateExecutionProviderFactory(device_id));
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id) {
  static ProviderLibrary library(LIBRARY_PREFIX "onnxruntime_providers_tensorrt" LIBRARY_EXTENSION);
  if (!library.provider_)
    return nullptr;

  return std::make_shared<IExecutionProviderFactory_Translator>(library.provider_->CreateExecutionProviderFactory(device_id));
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Dnnl, _In_ OrtSessionOptions* options, int use_arena) {
  auto factory = onnxruntime::CreateExecutionProviderFactory_Dnnl(use_arena);
  if (!factory) {
    LOGS_DEFAULT(ERROR) << "OrtSessionOptionsAppendExecutionProvider_Dnnl: Failed to load shared library";
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Dnnl: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tensorrt, _In_ OrtSessionOptions* options, int device_id) {
  auto factory = onnxruntime::CreateExecutionProviderFactory_Tensorrt(device_id);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Tensorrt: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
}
