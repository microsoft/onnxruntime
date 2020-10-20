// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the Onnxruntime side of the bridge to allow providers to be built as a DLL
// It implements onnxruntime::ProviderHost

#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/framework/data_types.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/model.h"
#include "core/platform/env.h"
#include "core/providers/common.h"
#include "core/session/inference_session.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

#ifdef USE_TENSORRT
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/providers/cuda/cuda_common.h"
#endif

namespace onnxruntime {
// These Provider types are really just internal types, so we #define PROVIDER_BRIDGE_ORT so that only these definitions are seen by provider_interfaces.h
// Users of provider_interfaces.h (through provider_api.h) will see the wrappers that call through the provider shared interface which is implemented by this file
using Provider_AttributeProto = ONNX_NAMESPACE::AttributeProto;
using Provider_GraphProto = ONNX_NAMESPACE::GraphProto;
using Provider_ModelProto = ONNX_NAMESPACE::ModelProto;
using Provider_NodeProto = ONNX_NAMESPACE::NodeProto;
using Provider_TensorProto = ONNX_NAMESPACE::TensorProto;
using Provider_TensorProtos = google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::TensorProto>;
using Provider_TensorShapeProto_Dimension = ONNX_NAMESPACE::TensorShapeProto_Dimension;
using Provider_TensorShapeProto_Dimensions = google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::TensorShapeProto_Dimension>;
using Provider_TensorShapeProto = ONNX_NAMESPACE::TensorShapeProto;
using Provider_TypeProto_Tensor = ONNX_NAMESPACE::TypeProto_Tensor;
using Provider_TypeProto = ONNX_NAMESPACE::TypeProto;
using Provider_ValueInfoProto = ONNX_NAMESPACE::ValueInfoProto;
using Provider_ValueInfoProtos = google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::ValueInfoProto>;

using Provider_ComputeCapability = ComputeCapability;
using Provider_DataTransferManager = DataTransferManager;
using Provider_IDataTransfer = IDataTransfer;
using Provider_IndexedSubGraph = IndexedSubGraph;
using Provider_IndexedSubGraph_MetaDef = IndexedSubGraph::MetaDef;
using Provider_KernelDef = KernelDef;
using Provider_KernelDefBuilder = KernelDefBuilder;
using Provider_KernelRegistry = KernelRegistry;
using Provider_Function = Function;
using Provider_Graph = Graph;
using Provider_GraphViewer = GraphViewer;
using Provider_Model = Model;
using Provider_Node = Node;
using Provider_NodeArg = NodeArg;
using Provider_NodeAttributes = NodeAttributes;
using Provider_OpKernelContext = OpKernelContext;
using Provider_OpKernelInfo = OpKernelInfo;
using Provider_Tensor = Tensor;
}  // namespace onnxruntime

#define PROVIDER_BRIDGE_ORT
#include "core/common/cpuid_info.h"
#include "onnx/common/stl_backports.h"
#include "core/common/logging/logging.h"
#include "core/providers/shared_library/provider_interfaces.h"

#include "core/providers/dnnl/dnnl_provider_factory.h"
#include "core/providers/tensorrt/tensorrt_provider_factory.h"

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

struct Provider_AllocatorPtr_Impl : Provider_IAllocator {
  Provider_AllocatorPtr_Impl(AllocatorPtr p) : Provider_IAllocator{p->Info()}, p_{p} {}

  void* Alloc(size_t size) override { return p_->Alloc(size); }
  void Free(void* p) override { return p_->Free(p); }

  AllocatorPtr p_;
};

// This is really a IAllocator, but we wrap it with this class to make it into a Provider_IAllocator
struct Provider_IAllocator_Impl : Provider_IAllocator {
  Provider_IAllocator_Impl(std::unique_ptr<IAllocator> p) : Provider_IAllocator{p->Info()}, p_{std::move(p)} {}

  void* Alloc(size_t size) override { return p_->Alloc(size); }
  void Free(void* p) override { return p_->Free(p); }

  bool IsProviderInterface() const override { return false; }

  std::unique_ptr<IAllocator> p_;
};

// This is really a Provider_IAllocator, but we wrap it with this class to make it into a IAllocator
struct ProviderAllocator : IAllocator {
  ProviderAllocator(std::shared_ptr<Provider_IAllocator> p) : IAllocator{p->memory_info_}, p_{std::move(p)} {}

  void* Alloc(size_t size) override { return p_->Alloc(size); }
  void Free(void* p) override { return p_->Free(p); }

  std::shared_ptr<Provider_IAllocator> p_;
};

struct Provider_TensorShapeProto_Dimension_Iterator_Impl : Provider_TensorShapeProto_Dimension_Iterator {
  Provider_TensorShapeProto_Dimension_Iterator_Impl(google::protobuf::internal::RepeatedPtrIterator<const onnx::TensorShapeProto_Dimension>&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_TensorShapeProto_Dimension_Iterator& p) const override { return v_ != static_cast<const Provider_TensorShapeProto_Dimension_Iterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Provider_TensorShapeProto_Dimension& operator*() override { return *v_; }

  google::protobuf::internal::RepeatedPtrIterator<const onnx::TensorShapeProto_Dimension> v_;
};

struct Provider_NodeAttributes_Iterator_Impl : Provider_NodeAttributes_Iterator {
  Provider_NodeAttributes_Iterator_Impl(NodeAttributes::const_iterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_NodeAttributes_Iterator& p) const override { return v_ != static_cast<const Provider_NodeAttributes_Iterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const std::string& first() const override { return v_->first; }
  const Provider_AttributeProto& second() override { return v_->second; }

  NodeAttributes::const_iterator v_;
};

struct Provider_Node__NodeIterator_Impl : Provider_Node__NodeIterator {
  Provider_Node__NodeIterator_Impl(Node::NodeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_Node__NodeIterator& p) const override { return v_ != static_cast<const Provider_Node__NodeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Provider_Node& operator*() override { return *v_; }

  Node::NodeConstIterator v_;
};

struct Provider_Node__EdgeIterator_Impl : Provider_Node__EdgeIterator {
  Provider_Node__EdgeIterator_Impl(Node::EdgeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_Node__EdgeIterator& p) const override { return v_ != static_cast<const Provider_Node__EdgeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Provider_Node& GetNode() const override { return v_->GetNode(); }
  int GetSrcArgIndex() const override { return v_->GetSrcArgIndex(); }
  int GetDstArgIndex() const override { return v_->GetDstArgIndex(); }

  Node::EdgeConstIterator v_;
};

struct OpKernel_Translator : OpKernel {
  OpKernel_Translator(const OpKernelInfo& info, Provider_OpKernel* p) : OpKernel{info}, p_{p} {
  }

  Status Compute(OpKernelContext* context) const override {
    return p_->Compute(context, *reinterpret_cast<const Provider_OpKernel_Base*>(static_cast<const OpKernel*>(this)));
  }

  std::unique_ptr<Provider_OpKernel> p_;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OpKernel_Translator);
};

struct Provider_IExecutionProvider_Router_Impl : Provider_IExecutionProvider_Router, IExecutionProvider {
  Provider_IExecutionProvider_Router_Impl(Provider_IExecutionProvider* outer, const std::string& type) : IExecutionProvider(type), outer_(outer) {
  }

  virtual ~Provider_IExecutionProvider_Router_Impl() {}

  std::shared_ptr<Provider_KernelRegistry> Provider_GetKernelRegistry() const override { return GetKernelRegistry(); }
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override { return outer_->Provider_GetKernelRegistry(); }

  std::vector<std::unique_ptr<Provider_ComputeCapability>> Provider_GetCapability(const onnxruntime::Provider_GraphViewer& graph,
                                                                                  const std::vector<const Provider_KernelRegistry*>& kernel_registries) const override {
    return IExecutionProvider::GetCapability(graph, kernel_registries);
  }

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const onnxruntime::GraphViewer& graph,
                                                                const std::vector<const KernelRegistry*>& kernel_registries) const override {
    return outer_->Provider_GetCapability(graph, kernel_registries);
  }

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) override {
    return outer_->Provider_Compile(fused_nodes, node_compute_funcs);
  }

  Provider_AllocatorPtr Provider_GetAllocator(int id, OrtMemType mem_type) const override {
    return std::make_shared<Provider_AllocatorPtr_Impl>(IExecutionProvider::GetAllocator(id, mem_type));
  }

  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override {
    auto allocator = outer_->Provider_GetAllocator(id, mem_type);
    if (!allocator)
      return nullptr;
    return static_cast<Provider_AllocatorPtr_Impl*>(allocator.get())->p_;
  }

  std::unique_ptr<Provider_IDataTransfer> Provider_GetDataTransfer() const override { return IExecutionProvider::GetDataTransfer(); }
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override { return outer_->Provider_GetDataTransfer(); }

  void Provider_InsertAllocator(Provider_AllocatorPtr allocator) override {
    IExecutionProvider::InsertAllocator(static_cast<Provider_AllocatorPtr_Impl*>(allocator.get())->p_);
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

  Provider_AllocatorPtr CreateAllocator(const Provider_AllocatorCreationInfo& info) override {
    AllocatorCreationInfo info_real{
        [&info](int value) -> std::unique_ptr<IAllocator> {
          auto allocator = info.factory(value);
          // If the allocator is a provider interface, we need to wrap it with ProviderAllocator to turn it into an IAllocator
          // Otherwise it's really a Provider_IAllocator_Impl, so we can just unwrap it to get back to the IAllocator inside
          if (allocator->IsProviderInterface())
            return onnxruntime::make_unique<ProviderAllocator>(std::move(allocator));
          else
            return std::move(static_cast<Provider_IAllocator_Impl*>(&*allocator)->p_);
        },
        info.device_id,
        info.use_arena,
        info.arena_cfg};

    // info_real will always return a unique_ptr to an IAllocator, which might be a native IAllocator or a provider interface wrapped by ProviderAllocator.
    // Either way we wrap it in a Provider_IAllocator_Impl to be unwrapped by Provider_InsertAllocator
    return std::make_shared<Provider_AllocatorPtr_Impl>(onnxruntime::CreateAllocator(info_real));
  }

  std::unique_ptr<Provider_IAllocator> CreateCPUAllocator(
      const OrtMemoryInfo& memory_info) override {
    return onnxruntime::make_unique<Provider_IAllocator_Impl>(
        onnxruntime::make_unique<CPUAllocator>(memory_info));
  };

  std::unique_ptr<Provider_IExecutionProvider_Router> Create_IExecutionProvider_Router(
      Provider_IExecutionProvider* outer, const std::string& type) override {
    return onnxruntime::make_unique<Provider_IExecutionProvider_Router_Impl>(outer, type);
  };

#ifdef USE_TENSORRT
  std::unique_ptr<Provider_IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) override {
    return onnxruntime::make_unique<Provider_IAllocator_Impl>(onnxruntime::make_unique<CUDAAllocator>(device_id, name));
  }

  std::unique_ptr<Provider_IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) override {
    return onnxruntime::make_unique<Provider_IAllocator_Impl>(onnxruntime::make_unique<CUDAPinnedAllocator>(device_id, name));
  }

  std::unique_ptr<Provider_IDataTransfer> CreateGPUDataTransfer() override { return onnxruntime::make_unique<GPUDataTransfer>(); }

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
  void HeapFree(void* p) override { delete[] reinterpret_cast<uint8_t*>(p); }

  std::vector<std::string> GetStackTrace() override { return onnxruntime::GetStackTrace(); }

  AutoPadType StringToAutoPadType(const std::string& str) override { return onnxruntime::StringToAutoPadType(str); }

  void LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file, const char* function, uint32_t line) override {
    return ::onnxruntime::LogRuntimeError(session_id, status, file, function, line);
  }

  // CPUIDInfo
  const CPUIDInfo& CPUIDInfo__GetCPUIDInfo() override { return CPUIDInfo::GetCPUIDInfo(); }
  bool CPUIDInfo__HasAVX2(const CPUIDInfo* p) override { return p->HasAVX2(); }
  bool CPUIDInfo__HasAVX512f(const CPUIDInfo* p) override { return p->HasAVX512f(); }

  // logging::Logger
  bool logging__Logger__OutputIsEnabled(const logging::Logger* p, logging::Severity severity, logging::DataType data_type) override { return p->OutputIsEnabled(severity, data_type); }

  // logging::LoggingManager
  const logging::Logger& logging__LoggingManager__DefaultLogger() override { return logging::LoggingManager::DefaultLogger(); }

  // logging::Capture
  std::unique_ptr<logging::Capture> logging__Capture__construct(const logging::Logger& logger, logging::Severity severity, const char* category, logging::DataType dataType, const CodeLocation& location) override {
    return onnxruntime::make_unique<logging::Capture>(logger, severity, category, dataType, location);
  }
  void logging__Capture__operator_delete(logging::Capture* p) noexcept override { delete p; }
  std::ostream& logging__Capture__Stream(logging::Capture* p) noexcept override { return p->Stream();  }

  // Provider_TypeProto_Tensor
  int32_t Provider_TypeProto_Tensor__elem_type(const Provider_TypeProto_Tensor* p) override { return p->elem_type(); }

  // Provider_TypeProto
  const Provider_TypeProto_Tensor& Provider_TypeProto__tensor_type(const Provider_TypeProto* p) override { return p->tensor_type(); }

  // Provider_AttributeProto
  std::unique_ptr<Provider_AttributeProto> Provider_AttributeProto__construct() override { return onnxruntime::make_unique<ONNX_NAMESPACE::AttributeProto>(); }
  void Provider_AttributeProto__operator_delete(Provider_AttributeProto* p) override { delete p; }
  void Provider_AttributeProto__operator_assign(Provider_AttributeProto* p, const Provider_AttributeProto& v) override { *p = v; }

  ONNX_NAMESPACE::AttributeProto_AttributeType Provider_AttributeProto__type(const Provider_AttributeProto* p) override { return p->type(); }
  int Provider_AttributeProto__ints_size(const Provider_AttributeProto* p) override { return p->ints_size(); }
  int64_t Provider_AttributeProto__ints(const Provider_AttributeProto* p, int i) override { return p->ints(i); }
  int64_t Provider_AttributeProto__i(const Provider_AttributeProto* p) override { return p->i(); }
  float Provider_AttributeProto__f(const Provider_AttributeProto* p) override { return p->f(); }
  void Provider_AttributeProto__set_s(Provider_AttributeProto* p, const ::std::string& value) override { return p->set_s(value); }
  const ::std::string& Provider_AttributeProto__s(const Provider_AttributeProto* p) override { return p->s(); }
  void Provider_AttributeProto__set_name(Provider_AttributeProto* p, const ::std::string& value) override { return p->set_name(value); }
  void Provider_AttributeProto__set_type(Provider_AttributeProto* p, ONNX_NAMESPACE::AttributeProto_AttributeType value) override { return p->set_type(value); }
  Provider_TensorProto* Provider_AttributeProto__add_tensors(Provider_AttributeProto* p) override { return p->add_tensors(); }

  // Provider_GraphProto
  void Provider_GraphProto__operator_delete(Provider_GraphProto* p) override { delete p; }

  Provider_ValueInfoProtos* Provider_GraphProto__mutable_input(Provider_GraphProto* p) override { return p->mutable_input(); }

  const Provider_ValueInfoProtos& Provider_GraphProto__output(const Provider_GraphProto* p) override { return p->output(); }
  Provider_ValueInfoProtos* Provider_GraphProto__mutable_output(Provider_GraphProto* p) override { return p->mutable_output(); }

  Provider_ValueInfoProtos* Provider_GraphProto__mutable_value_info(Provider_GraphProto* p) override { return p->mutable_value_info(); }
  Provider_TensorProtos* Provider_GraphProto__mutable_initializer(Provider_GraphProto* p) override { return p->mutable_initializer(); }
  Provider_NodeProto* Provider_GraphProto__add_node(Provider_GraphProto* p) override { return p->add_node(); }

  void Provider_GraphProto__operator_assign(Provider_GraphProto* p, const Provider_GraphProto& v) override { *p = v; }

  // Provider_ModelProto
  void Provider_ModelProto__operator_delete(Provider_ModelProto* p) override { delete p; }

  bool Provider_ModelProto__SerializeToString(const Provider_ModelProto* p, std::string& string) override { return p->SerializeToString(&string); }
  bool Provider_ModelProto__SerializeToOstream(const Provider_ModelProto* p, std::ostream& output) override { return p->SerializeToOstream(&output); }

  const Provider_GraphProto& Provider_ModelProto__graph(const Provider_ModelProto* p) override { return p->graph(); }
  Provider_GraphProto* Provider_ModelProto__mutable_graph(Provider_ModelProto* p) override { return p->mutable_graph(); }

  void Provider_ModelProto__set_ir_version(Provider_ModelProto* p, int64_t value) override { p->set_ir_version(value); }

  // Provider_TensorProto
  void Provider_TensorProto__operator_delete(Provider_TensorProto* p) override { delete p; }
  void Provider_TensorProto__operator_assign(Provider_TensorProto* p, const Provider_TensorProto& v) override { *p = v; }

  // Provider_TensorProtos
  Provider_TensorProto* Provider_TensorProtos__Add(Provider_TensorProtos* p) override { return p->Add(); }

  // Provider_TensorShapeProto_Dimension
  const std::string& Provider_TensorShapeProto_Dimension__dim_param(const Provider_TensorShapeProto_Dimension* p) override {
    return p->dim_param();
  }

  // Provider_TensorShapeProto_Dimensions
  std::unique_ptr<Provider_TensorShapeProto_Dimension_Iterator> Provider_TensorShapeProto_Dimensions__begin(const Provider_TensorShapeProto_Dimensions* p) override {
    return onnxruntime::make_unique<Provider_TensorShapeProto_Dimension_Iterator_Impl>(p->begin());
  }

  std::unique_ptr<Provider_TensorShapeProto_Dimension_Iterator> Provider_TensorShapeProto_Dimensions__end(const Provider_TensorShapeProto_Dimensions* p) override {
    return onnxruntime::make_unique<Provider_TensorShapeProto_Dimension_Iterator_Impl>(p->end());
  }

  // Provider_TensorShapeProto
  int Provider_TensorShapeProto__dim_size(const Provider_TensorShapeProto* p) override { return p->dim_size(); }
  const Provider_TensorShapeProto_Dimensions& Provider_TensorShapeProto__dim(const Provider_TensorShapeProto* p) override { return p->dim(); }

  // Provider_ValueInfoProto
  const Provider_TypeProto& Provider_ValueInfoProto__type(const Provider_ValueInfoProto* p) override { return p->type(); }
  virtual void Provider_ValueInfoProto__operator_assign(Provider_ValueInfoProto* p, const Provider_ValueInfoProto& v) override { *p = v; }

  // Provider_ValueInfoProtos
  Provider_ValueInfoProto* Provider_ValueInfoProtos__Add(Provider_ValueInfoProtos* p) override { return p->Add(); }

  const Provider_ValueInfoProto& Provider_ValueInfoProtos__operator_array(const Provider_ValueInfoProtos* p, int index) override { return (*p)[index]; }

  // Provider_ComputeCapability
  std::unique_ptr<Provider_ComputeCapability> Provider_ComputeCapability__construct(std::unique_ptr<Provider_IndexedSubGraph> t_sub_graph) override { return onnxruntime::make_unique<ComputeCapability>(std::move(t_sub_graph)); }
  void Provider_ComputeCapability__operator_delete(Provider_ComputeCapability* p) override { delete p; }
  std::unique_ptr<Provider_IndexedSubGraph>& Provider_ComputeCapability__SubGraph(Provider_ComputeCapability* p) override { return p->sub_graph; }

  // Provider_DataTransferManager
  Status Provider_DataTransferManager__CopyTensor(const Provider_DataTransferManager* p, const Provider_Tensor& src, Provider_Tensor& dst, int exec_queue_id) override { return p->CopyTensor(src, dst, exec_queue_id); }

  // Provider_IDataTransfer
  void Provider_IDataTransfer__operator_delete(Provider_IDataTransfer* p) override { delete p; }

  // Provider_IndexedSubGraph_MetaDef
  std::unique_ptr<Provider_IndexedSubGraph_MetaDef> Provider_IndexedSubGraph_MetaDef__construct() override { return onnxruntime::make_unique<IndexedSubGraph::MetaDef>(); }
  void Provider_IndexedSubGraph_MetaDef__operator_delete(Provider_IndexedSubGraph_MetaDef* p) override { delete p; }

  std::string& Provider_IndexedSubGraph_MetaDef__name(Provider_IndexedSubGraph_MetaDef* p) override { return p->name; }
  std::string& Provider_IndexedSubGraph_MetaDef__domain(Provider_IndexedSubGraph_MetaDef* p) override { return p->domain; }
  int& Provider_IndexedSubGraph_MetaDef__since_version(Provider_IndexedSubGraph_MetaDef* p) override { return p->since_version; }
  ONNX_NAMESPACE::OperatorStatus& Provider_IndexedSubGraph_MetaDef__status(Provider_IndexedSubGraph_MetaDef* p) override { return p->status; }
  std::vector<std::string>& Provider_IndexedSubGraph_MetaDef__inputs(Provider_IndexedSubGraph_MetaDef* p) override { return p->inputs; }
  std::vector<std::string>& Provider_IndexedSubGraph_MetaDef__outputs(Provider_IndexedSubGraph_MetaDef* p) override { return p->outputs; }
  Provider_NodeAttributes& Provider_IndexedSubGraph_MetaDef__attributes(Provider_IndexedSubGraph_MetaDef* p) override { return p->attributes; }
  std::string& Provider_IndexedSubGraph_MetaDef__doc_string(Provider_IndexedSubGraph_MetaDef* p) override { return p->doc_string; }

  // Provider_IndexedSubGraph
  std::unique_ptr<Provider_IndexedSubGraph> Provider_IndexedSubGraph__construct() override { return onnxruntime::make_unique<IndexedSubGraph>(); }
  void Provider_IndexedSubGraph__operator_delete(Provider_IndexedSubGraph* p) override { delete p; }

  std::vector<onnxruntime::NodeIndex>& Provider_IndexedSubGraph__Nodes(Provider_IndexedSubGraph* p) override { return p->nodes; }

  void Provider_IndexedSubGraph__SetMetaDef(Provider_IndexedSubGraph* p, std::unique_ptr<Provider_IndexedSubGraph_MetaDef>&& meta_def_) override { return p->SetMetaDef(std::move(meta_def_)); }
  const Provider_IndexedSubGraph_MetaDef* Provider_IndexedSubGraph__GetMetaDef(const Provider_IndexedSubGraph* p) override { return p->GetMetaDef(); }

  // Provider_KernelDef
  void Provider_KernelDef__operator_delete(Provider_KernelDef* p) override { delete p; }

  // Provider_KernelDefBuilder
  std::unique_ptr<Provider_KernelDefBuilder> Provider_KernelDefBuilder__construct() override { return onnxruntime::make_unique<KernelDefBuilder>(); }
  void Provider_KernelDefBuilder__operator_delete(Provider_KernelDefBuilder* p) override { delete p; }

  void Provider_KernelDefBuilder__SetName(Provider_KernelDefBuilder* p, const char* op_name) override { p->SetName(op_name); }
  void Provider_KernelDefBuilder__SetDomain(Provider_KernelDefBuilder* p, const char* domain) override { p->SetDomain(domain); }
  void Provider_KernelDefBuilder__SinceVersion(Provider_KernelDefBuilder* p, int since_version) override { p->SinceVersion(since_version); }
  void Provider_KernelDefBuilder__Provider(Provider_KernelDefBuilder* p, const char* provider_type) override { p->Provider(provider_type); }
  void Provider_KernelDefBuilder__TypeConstraint(Provider_KernelDefBuilder* p, const char* arg_name, MLDataType supported_type) override { p->TypeConstraint(arg_name, supported_type); }
  void Provider_KernelDefBuilder__TypeConstraint(Provider_KernelDefBuilder* p, const char* arg_name, const std::vector<MLDataType>& supported_types) override { p->TypeConstraint(arg_name, supported_types); }
  void Provider_KernelDefBuilder__InputMemoryType(Provider_KernelDefBuilder* p, OrtMemType type, int input_index) override { p->InputMemoryType(type, input_index); }
  void Provider_KernelDefBuilder__OutputMemoryType(Provider_KernelDefBuilder* p, OrtMemType type, int input_index) override { p->OutputMemoryType(type, input_index); }
  void Provider_KernelDefBuilder__ExecQueueId(Provider_KernelDefBuilder* p, int queue_id) override { p->ExecQueueId(queue_id); }

  std::unique_ptr<Provider_KernelDef> Provider_KernelDefBuilder__Build(Provider_KernelDefBuilder* p) override { return p->Build(); }

  // Provider_KernelRegistry
  std::shared_ptr<Provider_KernelRegistry> Provider_KernelRegistry__construct() override { return std::make_shared<KernelRegistry>(); }
  void Provider_KernelRegistry__operator_delete(Provider_KernelRegistry* p) override { delete p; }
  Status Provider_KernelRegistry__Register(Provider_KernelRegistry* p, Provider_KernelCreateInfo&& create_info) override {
    KernelCreateInfo info_real(std::move(create_info.kernel_def),
                               [kernel_create_func = create_info.kernel_create_func](const OpKernelInfo& info) -> OpKernel* {
                                 return new OpKernel_Translator(info, kernel_create_func(info));
                               });
    return p->Register(std::move(info_real));
  }

  // Provider_Function
  const Provider_Graph& Provider_Function__Body(const Provider_Function* p) override { return p->Body(); }

  // Provider_Node
  const std::string& Provider_Node__Name(const Provider_Node* p) noexcept override { return p->Name(); }
  const std::string& Provider_Node__Description(const Provider_Node* p) noexcept override { return p->Description(); }
  const std::string& Provider_Node__Domain(const Provider_Node* p) noexcept override { return p->Domain(); }
  const std::string& Provider_Node__OpType(const Provider_Node* p) noexcept override { return p->OpType(); }

  const Provider_Function* Provider_Node__GetFunctionBody(const Provider_Node* p) noexcept override { return p->GetFunctionBody(); }

  ConstPointerContainer<std::vector<Provider_NodeArg*>> Provider_Node__ImplicitInputDefs(const Provider_Node* p) noexcept override { return p->ImplicitInputDefs(); }
  ConstPointerContainer<std::vector<Provider_NodeArg*>> Provider_Node__InputDefs(const Provider_Node* p) noexcept override { return p->InputDefs(); }
  ConstPointerContainer<std::vector<Provider_NodeArg*>> Provider_Node__OutputDefs(const Provider_Node* p) noexcept override { return p->OutputDefs(); }

  NodeIndex Provider_Node__Index(const Provider_Node* p) noexcept override { return p->Index(); }

  void Provider_Node__ToProto(const Provider_Node* p, Provider_NodeProto& proto, bool update_subgraphs = false) override { p->ToProto(proto, update_subgraphs); }

  const Provider_NodeAttributes& Provider_Node__GetAttributes(const Provider_Node* p) noexcept override { return p->GetAttributes(); }
  size_t Provider_Node__GetInputEdgesCount(const Provider_Node* p) noexcept override { return p->GetInputEdgesCount(); }
  size_t Provider_Node__GetOutputEdgesCount(const Provider_Node* p) noexcept override { return p->GetOutputEdgesCount(); }

  std::unique_ptr<Provider_Node__NodeIterator> Provider_Node__InputNodesBegin(const Provider_Node* p) noexcept override { return onnxruntime::make_unique<Provider_Node__NodeIterator_Impl>(p->InputNodesBegin()); }
  std::unique_ptr<Provider_Node__NodeIterator> Provider_Node__InputNodesEnd(const Provider_Node* p) noexcept override { return onnxruntime::make_unique<Provider_Node__NodeIterator_Impl>(p->InputNodesEnd()); }

  std::unique_ptr<Provider_Node__EdgeIterator> Provider_Node__OutputEdgesBegin(const Provider_Node* p) noexcept override { return onnxruntime::make_unique<Provider_Node__EdgeIterator_Impl>(p->OutputEdgesBegin()); }
  std::unique_ptr<Provider_Node__EdgeIterator> Provider_Node__OutputEdgesEnd(const Provider_Node* p) noexcept override { return onnxruntime::make_unique<Provider_Node__EdgeIterator_Impl>(p->OutputEdgesEnd()); }

  // Provider_NodeArg
  const std::string& Provider_NodeArg__Name(const Provider_NodeArg* p) noexcept override { return p->Name(); }
  const Provider_TensorShapeProto* Provider_NodeArg__Shape(const Provider_NodeArg* p) override { return p->Shape(); }
  ONNX_NAMESPACE::DataType Provider_NodeArg__Type(const Provider_NodeArg* p) noexcept override { return p->Type(); }
  const Provider_NodeArgInfo& Provider_NodeArg__ToProto(const Provider_NodeArg* p) noexcept override { return p->ToProto(); }
  bool Provider_NodeArg__Exists(const Provider_NodeArg* p) const noexcept override { return p->Exists(); }
  const Provider_TypeProto* Provider_NodeArg__TypeAsProto(const Provider_NodeArg* p) noexcept override { return p->TypeAsProto(); }

  // Provider_NodeAttributes
  std::unique_ptr<Provider_NodeAttributes> Provider_NodeAttributes__construct() override { return onnxruntime::make_unique<NodeAttributes>(); }
  void Provider_NodeAttributes__operator_delete(Provider_NodeAttributes* p) noexcept override { delete p; }
  size_t Provider_NodeAttributes__size(const Provider_NodeAttributes* p) override { return p->size(); }
  void Provider_NodeAttributes__clear(Provider_NodeAttributes* p) noexcept override { return p->clear(); }
  Provider_AttributeProto& Provider_NodeAttributes__operator_array(Provider_NodeAttributes* p, const std::string& string) override { return (*p)[string]; }
  void Provider_NodeAttributes__operator_assign(Provider_NodeAttributes* p, const Provider_NodeAttributes& v) override { *p = v; }

  std::unique_ptr<Provider_NodeAttributes_Iterator> Provider_NodeAttributes__begin(const Provider_NodeAttributes* p) override {
    return onnxruntime::make_unique<Provider_NodeAttributes_Iterator_Impl>(p->begin());
  }
  std::unique_ptr<Provider_NodeAttributes_Iterator> Provider_NodeAttributes__end(const Provider_NodeAttributes* p) override {
    return onnxruntime::make_unique<Provider_NodeAttributes_Iterator_Impl>(p->end());
  }
  std::unique_ptr<Provider_NodeAttributes_Iterator> Provider_NodeAttributes__find(const Provider_NodeAttributes* p, const std::string& key) override {
    return onnxruntime::make_unique<Provider_NodeAttributes_Iterator_Impl>(p->find(key));
  }
  void Provider_NodeAttributes__insert(Provider_NodeAttributes* p, const Provider_NodeAttributes& v) override { return p->insert(v.begin(), v.end()); }

  // Provider_Model
  void Provider_Model__operator_delete(Provider_Model* p) override { delete p; }
  Provider_Graph& Provider_Model__MainGraph(Provider_Model* p) override { return p->MainGraph(); }
  std::unique_ptr<Provider_ModelProto> Provider_Model__ToProto(Provider_Model* p) override { return onnxruntime::make_unique<ONNX_NAMESPACE::ModelProto>(p->ToProto()); }

  // Provider_Graph
  std::unique_ptr<Provider_GraphViewer> Provider_Graph__CreateGraphViewer(const Provider_Graph* p) override { return onnxruntime::make_unique<GraphViewer>(*p); }
  std::unique_ptr<Provider_GraphProto> Provider_Graph__ToGraphProto(const Provider_Graph* p) override { return onnxruntime::make_unique<ONNX_NAMESPACE::GraphProto>(p->ToGraphProto()); }

  Provider_NodeArg& Provider_Graph__GetOrCreateNodeArg(Provider_Graph* p, const std::string& name, const Provider_TypeProto* p_arg_type) override { return p->GetOrCreateNodeArg(name, p_arg_type); }

  Status Provider_Graph__Resolve(Provider_Graph* p) override { return p->Resolve(); }
  void Provider_Graph__AddInitializedTensor(Provider_Graph* p, const Provider_TensorProto& tensor) override { p->AddInitializedTensor(tensor); }
  Provider_Node& Provider_Graph__AddNode(Provider_Graph* p, const std::string& name, const std::string& op_type, const std::string& description, const std::vector<Provider_NodeArg*>& input_args, const std::vector<Provider_NodeArg*>& output_args, const Provider_NodeAttributes* attributes, const std::string& domain) override {
    return p->AddNode(name, op_type, description, input_args, output_args, attributes, domain);
  }

  const std::vector<const Provider_NodeArg*>& Provider_Graph__GetOutputs(const Provider_Graph* p) noexcept override { return p->GetOutputs(); }
  void Provider_Graph__SetOutputs(Provider_Graph* p, const std::vector<const Provider_NodeArg*>& outputs) override { p->SetOutputs(outputs); }

  const std::vector<const Provider_NodeArg*>& Provider_Graph__GetInputs(const Provider_Graph* p) noexcept override { return p->GetInputs(); }
  bool Provider_Graph__GetInitializedTensor(const Provider_Graph* p, const std::string& tensor_name, const Provider_TensorProto*& value) override { return p->GetInitializedTensor(tensor_name, value); }

  // Provider_GraphViewer
  void Provider_GraphViewer__operator_delete(Provider_GraphViewer* p) override { delete p; }
  std::unique_ptr<Provider_Model> Provider_GraphViewer__CreateModel(const Provider_GraphViewer* graph_viewer, const logging::Logger& logger) override {
    return onnxruntime::make_unique<Model>(graph_viewer->Name(), true, ModelMetaData(), PathString(),
                                           IOnnxRuntimeOpSchemaRegistryList(), graph_viewer->DomainToVersionMap(),
                                           std::vector<ONNX_NAMESPACE::FunctionProto>(), logger);
  }

  const std::string& Provider_GraphViewer__Name(const Provider_GraphViewer* p) noexcept override { return p->Name(); }

  const Provider_Node* Provider_GraphViewer__GetNode(const Provider_GraphViewer* p, NodeIndex node_index) override { return p->GetNode(node_index); }
  const Provider_NodeArg* Provider_GraphViewer__GetNodeArg(const Provider_GraphViewer* p, const std::string& name) override { return p->GetNodeArg(name); }

  bool Provider_GraphViewer__IsSubgraph(const Provider_GraphViewer* p) override { return p->IsSubgraph(); }
  int Provider_GraphViewer__NumberOfNodes(const Provider_GraphViewer* p) noexcept override { return p->NumberOfNodes(); }
  int Provider_GraphViewer__MaxNodeIndex(const Provider_GraphViewer* p) noexcept override { return p->MaxNodeIndex(); }

  const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetInputs(const Provider_GraphViewer* p) noexcept override { return p->GetInputs(); }
  const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetOutputs(const Provider_GraphViewer* p) noexcept override { return p->GetOutputs(); }
  const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetValueInfo(const Provider_GraphViewer* p) noexcept override { return p->GetValueInfo(); }

  const Provider_InitializedTensorSet& Provider_GraphViewer__GetAllInitializedTensors(const Provider_GraphViewer* p) override { return p->GetAllInitializedTensors(); }
  bool Provider_GraphViewer__GetInitializedTensor(const Provider_GraphViewer* p, const std::string& tensor_name, const Provider_TensorProto*& value) override { return p->GetInitializedTensor(tensor_name, value); }

  const std::unordered_map<std::string, int>& Provider_GraphViewer__DomainToVersionMap(const Provider_GraphViewer* p) override { return p->DomainToVersionMap(); }

  const std::vector<NodeIndex>& Provider_GraphViewer__GetNodesInTopologicalOrder(const Provider_GraphViewer* p) override { return p->GetNodesInTopologicalOrder(); }

  // Provider_OpKernel_Base
  const Provider_OpKernelInfo& Provider_OpKernel_Base__GetInfo(const Provider_OpKernel_Base* p) override { return reinterpret_cast<const OpKernel*>(p)->Info(); }

  // Provider_OpKernelContext
  const Provider_Tensor* Provider_OpKernelContext__Input_Tensor(const Provider_OpKernelContext* p, int index) override { return p->Input<Tensor>(index); }
  Provider_Tensor* Provider_OpKernelContext__Output(Provider_OpKernelContext* p, int index, const TensorShape& shape) override { return p->Output(index, shape); }

  // Provider_OpKernelInfo
  Status Provider_OpKernelInfo__GetAttr_int64(const Provider_OpKernelInfo* p, const std::string& name, int64_t* value) override { return p->GetAttr(name, value); }
  Status Provider_OpKernelInfo__GetAttr_float(const Provider_OpKernelInfo* p, const std::string& name, float* value) override { return p->GetAttr(name, value); }

  const Provider_DataTransferManager& Provider_OpKernelInfo__GetDataTransferManager(const Provider_OpKernelInfo* p) noexcept override { return p->GetDataTransferManager(); }
  int Provider_OpKernelInfo__GetKernelDef_ExecQueueId(const Provider_OpKernelInfo* p) noexcept override { return p->GetKernelDef().ExecQueueId(); }

  // Provider_Tensor
  float* Provider_Tensor__MutableData_float(Provider_Tensor* p) override { return p->MutableData<float>(); }
  const float* Provider_Tensor__Data_float(const Provider_Tensor* p) override { return p->Data<float>(); }

  void* Provider_Tensor__MutableDataRaw(Provider_Tensor* p) noexcept override { return p->MutableDataRaw(); }
  const void* Provider_Tensor__DataRaw(const Provider_Tensor* p) const noexcept override { return p->DataRaw(); }

  const TensorShape& Provider_Tensor__Shape(const Provider_Tensor* p) override { return p->Shape(); }
  size_t Provider_Tensor__SizeInBytes(const Provider_Tensor* p) override { return p->SizeInBytes(); }
  const OrtMemoryInfo& Provider_Tensor__Location(const Provider_Tensor* p) override { return p->Location(); }

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
