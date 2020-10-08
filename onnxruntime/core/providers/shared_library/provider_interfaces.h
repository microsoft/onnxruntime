// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Public wrappers around internal ort interfaces (currently)
// In the future the internal implementations could derive from these to remove the need for the wrapper implementations

#include "core/framework/func_api.h"

#define PROVIDER_DISALLOW_ALL(TypeName)     \
  TypeName() = delete;                      \
  TypeName(const TypeName&) = delete;       \
  void operator=(const TypeName&) = delete; \
  static void operator delete(void*) = delete;

namespace ONNX_NAMESPACE {
using namespace onnxruntime;

enum AttributeProto_AttributeType : int;
enum OperatorStatus : int;

// String pointer as unique TypeProto identifier.
using DataType = const std::string*;

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

// These types don't directly map to internal types
struct Provider_IExecutionProvider;
struct Provider_KernelCreateInfo;
struct Provider_OpKernel_Base;
struct ProviderHost;

class TensorShape;

template <typename T, typename TResult>
struct IteratorHolder {
  IteratorHolder(std::unique_ptr<T>&& p) : p_{std::move(p)} {}

  bool operator!=(const IteratorHolder& p) const { return p_->operator!=(*p.p_); }

  void operator++() { p_->operator++(); }
  TResult& operator*() { return p_->operator*(); }
  T* operator->() { return p_.get(); }

 private:
  std::unique_ptr<T> p_;
};

struct Provider_NodeAttributes_Iterator {
  virtual ~Provider_NodeAttributes_Iterator() {}

  virtual bool operator!=(const Provider_NodeAttributes_Iterator& p) const = 0;

  virtual void operator++() = 0;
  virtual const std::string& first() const = 0;
  virtual const Provider_AttributeProto& second() = 0;
};

struct Provider_TensorShapeProto_Dimension_Iterator {
  virtual ~Provider_TensorShapeProto_Dimension_Iterator() {}

  virtual bool operator!=(const Provider_TensorShapeProto_Dimension_Iterator& p) const = 0;

  virtual void operator++() = 0;
  virtual const Provider_TensorShapeProto_Dimension& operator*() = 0;
};

struct Provider_IExecutionProviderFactory {
  virtual ~Provider_IExecutionProviderFactory() = default;
  virtual std::unique_ptr<Provider_IExecutionProvider> CreateProvider() = 0;
};

class DataTypeImpl;
using MLDataType = const DataTypeImpl*;

template <typename T>
using Provider_IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

struct Provider_IAllocator {
  Provider_IAllocator(const OrtMemoryInfo& info) : memory_info_{info} {}
  virtual ~Provider_IAllocator() {}

  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p) = 0;
  const OrtMemoryInfo& Info() const { return memory_info_; };

  virtual bool IsProviderInterface() const { return true; }

  template <typename T>
  static Provider_IAllocatorUniquePtr<T> MakeUniquePtr(std::shared_ptr<Provider_IAllocator> allocator, size_t count_or_bytes) {
    if (allocator == nullptr) return nullptr;

    size_t alloc_size = count_or_bytes;

    // if T is not void, 'count_or_bytes' == number of items so allow for that
    if (!std::is_void<T>::value) {
      // TODO: Use internal implementation to get correct sizes
      return nullptr;
    }
    return Provider_IAllocatorUniquePtr<T>{
        static_cast<T*>(allocator->Alloc(alloc_size)),  // allocate
        [=](T* ptr) { allocator->Free(ptr); }};         // capture IAllocator so it's always valid, and use as deleter
  }

  const OrtMemoryInfo memory_info_;

  Provider_IAllocator(const Provider_IAllocator&) = delete;
  void operator=(const Provider_IAllocator&) = delete;
};

using Provider_AllocatorPtr = std::shared_ptr<Provider_IAllocator>;
using Provider_AllocatorFactory = std::function<std::unique_ptr<Provider_IAllocator>(int)>;

using DeviceId = int16_t;
struct Provider_AllocatorCreationInfo {
  Provider_AllocatorCreationInfo(Provider_AllocatorFactory device_alloc_factory0,
                                 DeviceId device_id0 = 0,
                                 bool use_arena0 = true,
                                 OrtArenaCfg arena_cfg0 = {0, -1, -1, -1})
      : factory(device_alloc_factory0),
        device_id(device_id0),
        use_arena(use_arena0),
        arena_cfg(arena_cfg0) {
  }

  Provider_AllocatorFactory factory;
  DeviceId device_id;
  bool use_arena;
  OrtArenaCfg arena_cfg;
};

struct Provider_OpKernel {
  Provider_OpKernel() {}
  virtual ~Provider_OpKernel() = default;

  virtual Status Compute(Provider_OpKernelContext* context, const Provider_OpKernel_Base& base) const = 0;

  Provider_OpKernel(const Provider_OpKernel&) = delete;
  void operator=(const Provider_OpKernel&) = delete;
};

using NodeIndex = size_t;
using Provider_NodeArgInfo = Provider_ValueInfoProto;
// We can't just reinterpret_cast this one, since it's an unordered_map of object BY VALUE (can't do anything by value on the real types)
//using Provider_NodeAttributes = std::unordered_map<std::string, ONNX_NAMESPACE::Provider_AttributeProto_Copyable>;

using Provider_InitializedTensorSet = std::unordered_map<std::string, const Provider_TensorProto*>;

struct Provider_Node__NodeIterator {
  virtual ~Provider_Node__NodeIterator() {}

  virtual bool operator!=(const Provider_Node__NodeIterator& p) const = 0;

  virtual void operator++() = 0;
  virtual const Provider_Node& operator*() = 0;
};

struct Provider_Node__EdgeIterator {
  virtual ~Provider_Node__EdgeIterator() {}
  virtual bool operator!=(const Provider_Node__EdgeIterator& p) const = 0;

  virtual void operator++() = 0;
  virtual const Provider_Node& GetNode() const = 0;
  virtual int GetSrcArgIndex() const = 0;
  virtual int GetDstArgIndex() const = 0;
};

#ifndef PROVIDER_BRIDGE_ORT
// TODO: These are from execution_provider.h and should be factored out in the future into a common header
using CreateFunctionStateFunc = std::function<int(ComputeContext*, FunctionState*)>;
using ComputeFunc = std::function<Status(FunctionState, const OrtApi*, OrtKernelContext*)>;
using DestroyFunctionStateFunc = std::function<void(FunctionState)>;

struct NodeComputeInfo {
  CreateFunctionStateFunc create_state_func;
  ComputeFunc compute_func;
  DestroyFunctionStateFunc release_state_func;
};
#endif

// Provides the base class implementations, since Provider_IExecutionProvider is just an interface. This is to fake the C++ inheritance used by internal IExecutionProvider implementations
struct Provider_IExecutionProvider_Router {
  virtual ~Provider_IExecutionProvider_Router() {}

  virtual std::shared_ptr<Provider_KernelRegistry> Provider_GetKernelRegistry() const = 0;

  virtual std::vector<std::unique_ptr<Provider_ComputeCapability>> Provider_GetCapability(const onnxruntime::Provider_GraphViewer& graph,
                                                                                          const std::vector<const Provider_KernelRegistry*>& kernel_registries) const = 0;

  virtual Provider_AllocatorPtr Provider_GetAllocator(int id, OrtMemType mem_type) const = 0;
  virtual std::unique_ptr<Provider_IDataTransfer> Provider_GetDataTransfer() const = 0;
  virtual void Provider_InsertAllocator(Provider_AllocatorPtr allocator) = 0;
  virtual const logging::Logger* GetLogger() const = 0;

  void operator=(const Provider_IExecutionProvider_Router&) = delete;
};

struct Provider_IExecutionProvider {
  Provider_IExecutionProvider(const std::string& type);
  virtual ~Provider_IExecutionProvider() {}

  virtual std::shared_ptr<Provider_KernelRegistry> Provider_GetKernelRegistry() const { return p_->Provider_GetKernelRegistry(); }

  virtual std::unique_ptr<Provider_IDataTransfer> Provider_GetDataTransfer() const { return p_->Provider_GetDataTransfer(); }

  virtual std::vector<std::unique_ptr<Provider_ComputeCapability>> Provider_GetCapability(const onnxruntime::Provider_GraphViewer& graph,
                                                                                          const std::vector<const Provider_KernelRegistry*>& kernel_registries) const { return p_->Provider_GetCapability(graph, kernel_registries); }

  virtual common::Status Provider_Compile(const std::vector<Provider_Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) = 0;

  virtual Provider_AllocatorPtr Provider_GetAllocator(int id, OrtMemType mem_type) const { return p_->Provider_GetAllocator(id, mem_type); }
  virtual void Provider_InsertAllocator(Provider_AllocatorPtr allocator) { return p_->Provider_InsertAllocator(allocator); }

  virtual const logging::Logger* GetLogger() const { return p_->GetLogger(); }

  Provider_IExecutionProvider_Router* p_;

  void operator=(const Provider_IExecutionProvider&) = delete;
};

struct Provider {
  virtual std::shared_ptr<Provider_IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) = 0;
};

// There are two ways to route a function, one is a virtual method and the other is a function pointer (or pointer to member function)
// The function pointers are nicer in that they directly call the target function, but they cannot be used in cases where we're calling
// a specific implementation of a virtual class member. Trying to get a pointer to member of a virtual function will return a thunk that
// calls the virtual function (which will lead to infinite recursion in the bridge). There is no known way to get the non virtual member
// function pointer implementation in this case.
struct ProviderHost {
  virtual Provider_AllocatorPtr CreateAllocator(const Provider_AllocatorCreationInfo& info) = 0;

  virtual logging::Logger* LoggingManager_GetDefaultLogger() = 0;

  virtual std::unique_ptr<Provider_IAllocator> CreateCPUAllocator(const OrtMemoryInfo& memory_info) = 0;

#ifdef USE_TENSORRT
  virtual std::unique_ptr<Provider_IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<Provider_IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<Provider_IDataTransfer> CreateGPUDataTransfer() = 0;

  virtual void cuda__Impl_Cast(const int64_t* input_data, int32_t* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(const int32_t* input_data, int64_t* output_data, size_t count) = 0;

  virtual bool CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) = 0;
  virtual bool CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) = 0;
#endif

  virtual std::unique_ptr<Provider_IExecutionProvider_Router> Create_IExecutionProvider_Router(Provider_IExecutionProvider* outer, const std::string& type) = 0;

  virtual std::string GetEnvironmentVar(const std::string& var_name) = 0;

  MLDataType (*DataTypeImpl_GetType_Tensor)();
  MLDataType (*DataTypeImpl_GetType_float)();
  MLDataType (*DataTypeImpl_GetTensorType_float)();
  virtual const std::vector<MLDataType>& DataTypeImpl_AllFixedSizeTensorTypes() = 0;

  virtual void* HeapAllocate(size_t size) = 0;
  virtual void HeapFree(void*) = 0;

  virtual AutoPadType StringToAutoPadType(const std::string& str) = 0;

  virtual void LogRuntimeError(uint32_t session_id, const common::Status& status,
                               const char* file, const char* function, uint32_t line) = 0;

  virtual std::vector<std::string> GetStackTrace() = 0;

  // CPUIDInfo
  virtual const CPUIDInfo& CPUIDInfo__GetCPUIDInfo() = 0;
  virtual bool CPUIDInfo__HasAVX2(const CPUIDInfo* p) = 0;
  virtual bool CPUIDInfo__HasAVX512f(const CPUIDInfo* p) = 0;

  // logging::Logger
  virtual bool logging__Logger__OutputIsEnabled(const logging::Logger* p, logging::Severity severity, logging::DataType data_type) = 0;

  // logging::LoggingManager
  virtual const logging::Logger& logging__LoggingManager__DefaultLogger() = 0;

  // logging::Capture
  virtual std::unique_ptr<logging::Capture> logging__Capture__construct(const logging::Logger& logger, logging::Severity severity, const char* category, logging::DataType dataType, const CodeLocation& location) = 0;
  virtual void logging__Capture__operator_delete(logging::Capture* p) noexcept = 0;
  virtual std::ostream& logging__Capture__Stream(logging::Capture* p) noexcept = 0;

  // Provider_TypeProto_Tensor
  virtual int32_t Provider_TypeProto_Tensor__elem_type(const Provider_TypeProto_Tensor* p) = 0;

  // Provider_TypeProto
  virtual const Provider_TypeProto_Tensor& Provider_TypeProto__tensor_type(const Provider_TypeProto* p) = 0;

  // Provider_AttributeProto
  virtual std::unique_ptr<Provider_AttributeProto> Provider_AttributeProto__construct() = 0;
  virtual void Provider_AttributeProto__operator_delete(Provider_AttributeProto* p) = 0;
  virtual void Provider_AttributeProto__operator_assign(Provider_AttributeProto* p, const Provider_AttributeProto& v) = 0;

  virtual ONNX_NAMESPACE::AttributeProto_AttributeType Provider_AttributeProto__type(const Provider_AttributeProto* p) = 0;
  virtual int Provider_AttributeProto__ints_size(const Provider_AttributeProto* p) = 0;
  virtual int64_t Provider_AttributeProto__ints(const Provider_AttributeProto* p, int i) = 0;
  virtual int64_t Provider_AttributeProto__i(const Provider_AttributeProto* p) = 0;
  virtual float Provider_AttributeProto__f(const Provider_AttributeProto* p) = 0;
  virtual void Provider_AttributeProto__set_s(Provider_AttributeProto* p, const ::std::string& value) = 0;
  virtual const ::std::string& Provider_AttributeProto__s(const Provider_AttributeProto* p) = 0;
  virtual void Provider_AttributeProto__set_name(Provider_AttributeProto* p, const ::std::string& value) = 0;
  virtual void Provider_AttributeProto__set_type(Provider_AttributeProto* p, ONNX_NAMESPACE::AttributeProto_AttributeType value) = 0;
  virtual Provider_TensorProto* Provider_AttributeProto__add_tensors(Provider_AttributeProto* p) = 0;

  // Provider_GraphProto
  virtual void Provider_GraphProto__operator_delete(Provider_GraphProto* p) = 0;
  virtual void Provider_GraphProto__operator_assign(Provider_GraphProto* p, const Provider_GraphProto& v) = 0;

  virtual Provider_ValueInfoProtos* Provider_GraphProto__mutable_input(Provider_GraphProto* p) = 0;

  virtual const Provider_ValueInfoProtos& Provider_GraphProto__output(const Provider_GraphProto* p) = 0;
  virtual Provider_ValueInfoProtos* Provider_GraphProto__mutable_output(Provider_GraphProto* p) = 0;

  virtual Provider_ValueInfoProtos* Provider_GraphProto__mutable_value_info(Provider_GraphProto* p) = 0;
  virtual Provider_TensorProtos* Provider_GraphProto__mutable_initializer(Provider_GraphProto* p) = 0;
  virtual Provider_NodeProto* Provider_GraphProto__add_node(Provider_GraphProto* p) = 0;

  // Provider_ModelProto
  virtual void Provider_ModelProto__operator_delete(Provider_ModelProto* p) = 0;

  virtual bool Provider_ModelProto__SerializeToString(const Provider_ModelProto* p, std::string& string) = 0;
  virtual bool Provider_ModelProto__SerializeToOstream(const Provider_ModelProto* p, std::ostream& output) = 0;

  virtual const Provider_GraphProto& Provider_ModelProto__graph(const Provider_ModelProto* p) = 0;
  virtual Provider_GraphProto* Provider_ModelProto__mutable_graph(Provider_ModelProto* p) = 0;

  virtual void Provider_ModelProto__set_ir_version(Provider_ModelProto* p, int64_t value) = 0;

  // Provider_TensorProto
  virtual void Provider_TensorProto__operator_delete(Provider_TensorProto* p) = 0;
  virtual void Provider_TensorProto__operator_assign(Provider_TensorProto* p, const Provider_TensorProto& v) = 0;

  // Provider_TensorProtos
  virtual Provider_TensorProto* Provider_TensorProtos__Add(Provider_TensorProtos* p) = 0;

  // Provider_TensorShapeProto_Dimension
  virtual const std::string& Provider_TensorShapeProto_Dimension__dim_param(const Provider_TensorShapeProto_Dimension* p) = 0;

  // Provider_TensorShapeProto_Dimensions
  virtual std::unique_ptr<Provider_TensorShapeProto_Dimension_Iterator> Provider_TensorShapeProto_Dimensions__begin(const Provider_TensorShapeProto_Dimensions* p) = 0;
  virtual std::unique_ptr<Provider_TensorShapeProto_Dimension_Iterator> Provider_TensorShapeProto_Dimensions__end(const Provider_TensorShapeProto_Dimensions* p) = 0;

  // Provider_TensorShapeProto
  virtual int Provider_TensorShapeProto__dim_size(const Provider_TensorShapeProto* p) = 0;
  virtual const Provider_TensorShapeProto_Dimensions& Provider_TensorShapeProto__dim(const Provider_TensorShapeProto* p) = 0;

  // Provider_ValueInfoProto
  virtual void Provider_ValueInfoProto__operator_assign(Provider_ValueInfoProto* p, const Provider_ValueInfoProto& v) = 0;
  virtual const Provider_TypeProto& Provider_ValueInfoProto__type(const Provider_ValueInfoProto* p) = 0;

  // Provider_ValueInfoProtos
  virtual Provider_ValueInfoProto* Provider_ValueInfoProtos__Add(Provider_ValueInfoProtos* p) = 0;

  virtual const Provider_ValueInfoProto& Provider_ValueInfoProtos__operator_array(const Provider_ValueInfoProtos* p, int index) = 0;

  // Provider_ComputeCapability
  virtual std::unique_ptr<Provider_ComputeCapability> Provider_ComputeCapability__construct(std::unique_ptr<Provider_IndexedSubGraph> t_sub_graph) = 0;
  virtual void Provider_ComputeCapability__operator_delete(Provider_ComputeCapability* p) = 0;
  virtual std::unique_ptr<Provider_IndexedSubGraph>& Provider_ComputeCapability__SubGraph(Provider_ComputeCapability* p) = 0;

  // Provider_DataTransferManager
  virtual Status Provider_DataTransferManager__CopyTensor(const Provider_DataTransferManager* p, const Provider_Tensor& src, Provider_Tensor& dst, int exec_queue_id) = 0;

  // Provider_IDataTransfer
  virtual void Provider_IDataTransfer__operator_delete(Provider_IDataTransfer* p) = 0;

  // Provider_IndexedSubGraph_MetaDef
  virtual std::unique_ptr<Provider_IndexedSubGraph_MetaDef> Provider_IndexedSubGraph_MetaDef__construct() = 0;
  virtual void Provider_IndexedSubGraph_MetaDef__operator_delete(Provider_IndexedSubGraph_MetaDef* p) = 0;

  virtual std::string& Provider_IndexedSubGraph_MetaDef__name(Provider_IndexedSubGraph_MetaDef* p) = 0;
  virtual std::string& Provider_IndexedSubGraph_MetaDef__domain(Provider_IndexedSubGraph_MetaDef* p) = 0;
  virtual int& Provider_IndexedSubGraph_MetaDef__since_version(Provider_IndexedSubGraph_MetaDef* p) = 0;
  virtual ONNX_NAMESPACE::OperatorStatus& Provider_IndexedSubGraph_MetaDef__status(Provider_IndexedSubGraph_MetaDef* p) = 0;
  virtual std::vector<std::string>& Provider_IndexedSubGraph_MetaDef__inputs(Provider_IndexedSubGraph_MetaDef* p) = 0;
  virtual std::vector<std::string>& Provider_IndexedSubGraph_MetaDef__outputs(Provider_IndexedSubGraph_MetaDef* p) = 0;
  virtual Provider_NodeAttributes& Provider_IndexedSubGraph_MetaDef__attributes(Provider_IndexedSubGraph_MetaDef* p) = 0;
  virtual std::string& Provider_IndexedSubGraph_MetaDef__doc_string(Provider_IndexedSubGraph_MetaDef* p) = 0;

  // Provider_IndexedSubGraph
  virtual std::unique_ptr<Provider_IndexedSubGraph> Provider_IndexedSubGraph__construct() = 0;
  virtual void Provider_IndexedSubGraph__operator_delete(Provider_IndexedSubGraph* p) = 0;

  virtual std::vector<onnxruntime::NodeIndex>& Provider_IndexedSubGraph__Nodes(Provider_IndexedSubGraph* p) = 0;

  virtual void Provider_IndexedSubGraph__SetMetaDef(Provider_IndexedSubGraph* p, std::unique_ptr<Provider_IndexedSubGraph_MetaDef>&& meta_def_) = 0;
  virtual const Provider_IndexedSubGraph_MetaDef* Provider_IndexedSubGraph__GetMetaDef(const Provider_IndexedSubGraph* p) = 0;

  // Provider_KernelDef
  virtual void Provider_KernelDef__operator_delete(Provider_KernelDef* p) = 0;

  // Provider_KernelDefBuilder
  virtual std::unique_ptr<Provider_KernelDefBuilder> Provider_KernelDefBuilder__construct() = 0;
  virtual void Provider_KernelDefBuilder__operator_delete(Provider_KernelDefBuilder* p) = 0;

  virtual void Provider_KernelDefBuilder__SetName(Provider_KernelDefBuilder* p, const char* op_name) = 0;
  virtual void Provider_KernelDefBuilder__SetDomain(Provider_KernelDefBuilder* p, const char* domain) = 0;
  virtual void Provider_KernelDefBuilder__SinceVersion(Provider_KernelDefBuilder* p, int since_version) = 0;
  virtual void Provider_KernelDefBuilder__Provider(Provider_KernelDefBuilder* p, const char* provider_type) = 0;
  virtual void Provider_KernelDefBuilder__TypeConstraint(Provider_KernelDefBuilder* p, const char* arg_name, MLDataType supported_type) = 0;
  virtual void Provider_KernelDefBuilder__TypeConstraint(Provider_KernelDefBuilder* p, const char* arg_name, const std::vector<MLDataType>& supported_types) = 0;
  virtual void Provider_KernelDefBuilder__InputMemoryType(Provider_KernelDefBuilder* p, OrtMemType type, int input_index) = 0;
  virtual void Provider_KernelDefBuilder__OutputMemoryType(Provider_KernelDefBuilder* p, OrtMemType type, int input_index) = 0;
  virtual void Provider_KernelDefBuilder__ExecQueueId(Provider_KernelDefBuilder* p, int queue_id) = 0;

  virtual std::unique_ptr<Provider_KernelDef> Provider_KernelDefBuilder__Build(Provider_KernelDefBuilder* p) = 0;

  // Provider_KernelRegistry
  virtual std::shared_ptr<Provider_KernelRegistry> Provider_KernelRegistry__construct() = 0;
  virtual void Provider_KernelRegistry__operator_delete(Provider_KernelRegistry* p) = 0;
  virtual Status Provider_KernelRegistry__Register(Provider_KernelRegistry* p, Provider_KernelCreateInfo&& create_info) = 0;

  // Provider_Function
  virtual const Provider_Graph& Provider_Function__Body(const Provider_Function* p) = 0;

  // Provider_Node
  virtual const std::string& Provider_Node__Name(const Provider_Node* p) noexcept = 0;
  virtual const std::string& Provider_Node__Description(const Provider_Node* p) noexcept = 0;
  virtual const std::string& Provider_Node__Domain(const Provider_Node* p) noexcept = 0;
  virtual const std::string& Provider_Node__OpType(const Provider_Node* p) noexcept = 0;

  virtual const Provider_Function* Provider_Node__GetFunctionBody(const Provider_Node* p) noexcept = 0;

  virtual ConstPointerContainer<std::vector<Provider_NodeArg*>> Provider_Node__ImplicitInputDefs(const Provider_Node* p) noexcept = 0;

  virtual ConstPointerContainer<std::vector<Provider_NodeArg*>> Provider_Node__InputDefs(const Provider_Node* p) noexcept = 0;
  virtual ConstPointerContainer<std::vector<Provider_NodeArg*>> Provider_Node__OutputDefs(const Provider_Node* p) noexcept = 0;
  virtual NodeIndex Provider_Node__Index(const Provider_Node* p) noexcept = 0;

  virtual void Provider_Node__ToProto(const Provider_Node* p, Provider_NodeProto& proto, bool update_subgraphs = false) = 0;

  virtual const Provider_NodeAttributes& Provider_Node__GetAttributes(const Provider_Node* p) noexcept = 0;
  virtual size_t Provider_Node__GetInputEdgesCount(const Provider_Node* p) noexcept = 0;
  virtual size_t Provider_Node__GetOutputEdgesCount(const Provider_Node* p) noexcept = 0;

  virtual std::unique_ptr<Provider_Node__NodeIterator> Provider_Node__InputNodesBegin(const Provider_Node* p) noexcept = 0;
  virtual std::unique_ptr<Provider_Node__NodeIterator> Provider_Node__InputNodesEnd(const Provider_Node* p) noexcept = 0;

  virtual std::unique_ptr<Provider_Node__EdgeIterator> Provider_Node__OutputEdgesBegin(const Provider_Node* p) noexcept = 0;
  virtual std::unique_ptr<Provider_Node__EdgeIterator> Provider_Node__OutputEdgesEnd(const Provider_Node* p) noexcept = 0;

  // Provider_NodeArg
  virtual const std::string& Provider_NodeArg__Name(const Provider_NodeArg* p) noexcept = 0;
  virtual const Provider_TensorShapeProto* Provider_NodeArg__Shape(const Provider_NodeArg* p) = 0;
  virtual ONNX_NAMESPACE::DataType Provider_NodeArg__Type(const Provider_NodeArg* p) noexcept = 0;
  virtual const Provider_NodeArgInfo& Provider_NodeArg__ToProto(const Provider_NodeArg* p) noexcept = 0;
  virtual bool Provider_NodeArg__Exists(const Provider_NodeArg* p) const noexcept = 0;
  virtual const Provider_TypeProto* Provider_NodeArg__TypeAsProto(const Provider_NodeArg* p) noexcept = 0;

  // Provider_NodeAttributes
  virtual std::unique_ptr<Provider_NodeAttributes> Provider_NodeAttributes__construct() = 0;
  virtual void Provider_NodeAttributes__operator_delete(Provider_NodeAttributes* p) noexcept = 0;
  virtual void Provider_NodeAttributes__operator_assign(Provider_NodeAttributes* p, const Provider_NodeAttributes& v) = 0;

  virtual size_t Provider_NodeAttributes__size(const Provider_NodeAttributes* p) = 0;
  virtual void Provider_NodeAttributes__clear(Provider_NodeAttributes* p) noexcept = 0;
  virtual Provider_AttributeProto& Provider_NodeAttributes__operator_array(Provider_NodeAttributes* p, const std::string& string) = 0;

  virtual std::unique_ptr<Provider_NodeAttributes_Iterator> Provider_NodeAttributes__begin(const Provider_NodeAttributes* p) = 0;
  virtual std::unique_ptr<Provider_NodeAttributes_Iterator> Provider_NodeAttributes__end(const Provider_NodeAttributes* p) = 0;
  virtual std::unique_ptr<Provider_NodeAttributes_Iterator> Provider_NodeAttributes__find(const Provider_NodeAttributes* p, const std::string& key) = 0;
  virtual void Provider_NodeAttributes__insert(Provider_NodeAttributes* p, const Provider_NodeAttributes& v) = 0;

  // Provider_Model
  virtual void Provider_Model__operator_delete(Provider_Model* p) = 0;
  virtual Provider_Graph& Provider_Model__MainGraph(Provider_Model* p) = 0;
  virtual std::unique_ptr<Provider_ModelProto> Provider_Model__ToProto(Provider_Model* p) = 0;

  // Provider_Graph
  virtual std::unique_ptr<Provider_GraphViewer> Provider_Graph__CreateGraphViewer(const Provider_Graph* p) = 0;
  virtual std::unique_ptr<Provider_GraphProto> Provider_Graph__ToGraphProto(const Provider_Graph* p) = 0;

  virtual Provider_NodeArg& Provider_Graph__GetOrCreateNodeArg(Provider_Graph* p, const std::string& name, const Provider_TypeProto* p_arg_type) = 0;

  virtual Status Provider_Graph__Resolve(Provider_Graph* p) = 0;
  virtual void Provider_Graph__AddInitializedTensor(Provider_Graph* p, const Provider_TensorProto& tensor) = 0;
  virtual Provider_Node& Provider_Graph__AddNode(Provider_Graph* p, const std::string& name, const std::string& op_type, const std::string& description, const std::vector<Provider_NodeArg*>& input_args, const std::vector<Provider_NodeArg*>& output_args, const Provider_NodeAttributes* attributes, const std::string& domain) = 0;

  virtual const std::vector<const Provider_NodeArg*>& Provider_Graph__GetOutputs(const Provider_Graph* p) noexcept = 0;
  virtual void Provider_Graph__SetOutputs(Provider_Graph* p, const std::vector<const Provider_NodeArg*>& outputs) = 0;

  virtual const std::vector<const Provider_NodeArg*>& Provider_Graph__GetInputs(const Provider_Graph* p) noexcept = 0;
  virtual bool Provider_Graph__GetInitializedTensor(const Provider_Graph* p, const std::string& tensor_name, const Provider_TensorProto*& value) = 0;

  // Provider_GraphViewer
  virtual void Provider_GraphViewer__operator_delete(Provider_GraphViewer* p) = 0;
  virtual std::unique_ptr<Provider_Model> Provider_GraphViewer__CreateModel(const Provider_GraphViewer* p, const logging::Logger& logger) = 0;

  virtual const std::string& Provider_GraphViewer__Name(const Provider_GraphViewer* p) noexcept = 0;

  virtual const Provider_Node* Provider_GraphViewer__GetNode(const Provider_GraphViewer* p, NodeIndex node_index) = 0;
  virtual const Provider_NodeArg* Provider_GraphViewer__GetNodeArg(const Provider_GraphViewer* p, const std::string& name) = 0;

  virtual bool Provider_GraphViewer__IsSubgraph(const Provider_GraphViewer* p) = 0;
  virtual int Provider_GraphViewer__NumberOfNodes(const Provider_GraphViewer* p) noexcept = 0;
  virtual int Provider_GraphViewer__MaxNodeIndex(const Provider_GraphViewer* p) noexcept = 0;

  virtual const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetInputs(const Provider_GraphViewer* p) noexcept = 0;
  virtual const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetOutputs(const Provider_GraphViewer* p) noexcept = 0;
  virtual const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetValueInfo(const Provider_GraphViewer* p) noexcept = 0;

  virtual const Provider_InitializedTensorSet& Provider_GraphViewer__GetAllInitializedTensors(const Provider_GraphViewer* p) = 0;
  virtual bool Provider_GraphViewer__GetInitializedTensor(const Provider_GraphViewer* p, const std::string& tensor_name, const Provider_TensorProto*& value) = 0;
  virtual const std::unordered_map<std::string, int>& Provider_GraphViewer__DomainToVersionMap(const Provider_GraphViewer* p) = 0;

  virtual const std::vector<NodeIndex>& Provider_GraphViewer__GetNodesInTopologicalOrder(const Provider_GraphViewer* p) = 0;

  // Provider_OpKernel_Base
  virtual const Provider_OpKernelInfo& Provider_OpKernel_Base__GetInfo(const Provider_OpKernel_Base* p) = 0;

  // Provider_OpKernelContext
  virtual const Provider_Tensor* Provider_OpKernelContext__Input_Tensor(const Provider_OpKernelContext* p, int index) = 0;
  virtual Provider_Tensor* Provider_OpKernelContext__Output(Provider_OpKernelContext* p, int index, const TensorShape& shape) = 0;

  // Provider_OpKernelInfo
  virtual Status Provider_OpKernelInfo__GetAttr_int64(const Provider_OpKernelInfo* p, const std::string& name, int64_t* value) = 0;
  virtual Status Provider_OpKernelInfo__GetAttr_float(const Provider_OpKernelInfo* p, const std::string& name, float* value) = 0;

  virtual const Provider_DataTransferManager& Provider_OpKernelInfo__GetDataTransferManager(const Provider_OpKernelInfo* p) noexcept = 0;
  virtual int Provider_OpKernelInfo__GetKernelDef_ExecQueueId(const Provider_OpKernelInfo* p) noexcept = 0;

  // Provider_Tensor
  virtual float* Provider_Tensor__MutableData_float(Provider_Tensor* p) = 0;
  virtual const float* Provider_Tensor__Data_float(const Provider_Tensor* p) = 0;

  virtual void* Provider_Tensor__MutableDataRaw(Provider_Tensor* p) noexcept = 0;
  virtual const void* Provider_Tensor__DataRaw(const Provider_Tensor* p) const noexcept = 0;

  virtual const TensorShape& Provider_Tensor__Shape(const Provider_Tensor* p) = 0;
  virtual size_t Provider_Tensor__SizeInBytes(const Provider_Tensor* p) = 0;
  virtual const OrtMemoryInfo& Provider_Tensor__Location(const Provider_Tensor* p) = 0;
};

extern ProviderHost* g_host;

#ifndef PROVIDER_BRIDGE_ORT

struct CPUIDInfo {
  static const CPUIDInfo& GetCPUIDInfo() { return g_host->CPUIDInfo__GetCPUIDInfo(); }

  bool HasAVX2() const { return g_host->CPUIDInfo__HasAVX2(this); }
  bool HasAVX512f() const { return g_host->CPUIDInfo__HasAVX512f(this); }

PROVIDER_DISALLOW_ALL(CPUIDInfo)
};

namespace logging {

struct Logger {
  bool OutputIsEnabled(Severity severity, DataType data_type) const noexcept { return g_host->logging__Logger__OutputIsEnabled(this, severity, data_type);  }

PROVIDER_DISALLOW_ALL(Logger)
};

struct LoggingManager {
  static const Logger& DefaultLogger() { return g_host->logging__LoggingManager__DefaultLogger();  }

  PROVIDER_DISALLOW_ALL(LoggingManager)
};

struct Capture {
    static std::unique_ptr<Capture> Create(const Logger& logger, logging::Severity severity, const char* category,
            logging::DataType dataType, const CodeLocation& location) { return g_host->logging__Capture__construct(logger, severity, category, dataType, location); }
    static void operator delete(void* p) { g_host->logging__Capture__operator_delete(reinterpret_cast<Capture*>(p)); }

    std::ostream& Stream() noexcept { return g_host->logging__Capture__Stream(this); }

    Capture() = delete;
    Capture(const Capture&) = delete;
    void operator=(const Capture&) = delete;
};
}

struct Provider_TypeProto_Tensor {
  int32_t elem_type() const { return g_host->Provider_TypeProto_Tensor__elem_type(this); }

  PROVIDER_DISALLOW_ALL(Provider_TypeProto_Tensor)
};

struct Provider_TypeProto {
  const Provider_TypeProto_Tensor& tensor_type() const { return g_host->Provider_TypeProto__tensor_type(this); }

  PROVIDER_DISALLOW_ALL(Provider_TypeProto)
};

struct Provider_AttributeProto {
  static std::unique_ptr<Provider_AttributeProto> Create() { return g_host->Provider_AttributeProto__construct(); }
  void operator=(const Provider_AttributeProto& v) { g_host->Provider_AttributeProto__operator_assign(this, v); }
  static void operator delete(void* p) { g_host->Provider_AttributeProto__operator_delete(reinterpret_cast<Provider_AttributeProto*>(p)); }

  ONNX_NAMESPACE::AttributeProto_AttributeType type() const { return g_host->Provider_AttributeProto__type(this); }
  int ints_size() const { return g_host->Provider_AttributeProto__ints_size(this); }
  int64_t ints(int i) const { return g_host->Provider_AttributeProto__ints(this, i); }
  int64_t i() const { return g_host->Provider_AttributeProto__i(this); }
  float f() const { return g_host->Provider_AttributeProto__f(this); }
  void set_s(const ::std::string& value) { return g_host->Provider_AttributeProto__set_s(this, value); }
  const ::std::string& s() const { return g_host->Provider_AttributeProto__s(this); }
  void set_name(const ::std::string& value) { return g_host->Provider_AttributeProto__set_name(this, value); }
  void set_type(ONNX_NAMESPACE::AttributeProto_AttributeType value) { return g_host->Provider_AttributeProto__set_type(this, value); }
  Provider_TensorProto* add_tensors() { return g_host->Provider_AttributeProto__add_tensors(this); }

  Provider_AttributeProto() = delete;
  Provider_AttributeProto(const Provider_AttributeProto&) = delete;
};

struct Provider_GraphProto {
  static void operator delete(void* p) { g_host->Provider_GraphProto__operator_delete(reinterpret_cast<Provider_GraphProto*>(p)); }
  void operator=(const Provider_GraphProto& v) { return g_host->Provider_GraphProto__operator_assign(this, v); }

  Provider_ValueInfoProtos* mutable_input() { return g_host->Provider_GraphProto__mutable_input(this); }

  const Provider_ValueInfoProtos& output() const { return g_host->Provider_GraphProto__output(this); }
  Provider_ValueInfoProtos* mutable_output() { return g_host->Provider_GraphProto__mutable_output(this); }

  Provider_ValueInfoProtos* mutable_value_info() { return g_host->Provider_GraphProto__mutable_value_info(this); }
  Provider_TensorProtos* mutable_initializer() { return g_host->Provider_GraphProto__mutable_initializer(this); }
  Provider_NodeProto* add_node() { return g_host->Provider_GraphProto__add_node(this); }

  Provider_GraphProto() = delete;
  Provider_GraphProto(const Provider_GraphProto&) = delete;
};

struct Provider_ModelProto {
  static void operator delete(void* p) { g_host->Provider_ModelProto__operator_delete(reinterpret_cast<Provider_ModelProto*>(p)); }

  bool SerializeToString(std::string& string) const { return g_host->Provider_ModelProto__SerializeToString(this, string); }
  bool SerializeToOstream(std::ostream& output) const { return g_host->Provider_ModelProto__SerializeToOstream(this, output); }

  const Provider_GraphProto& graph() const { return g_host->Provider_ModelProto__graph(this); }
  Provider_GraphProto* mutable_graph() { return g_host->Provider_ModelProto__mutable_graph(this); }

  void set_ir_version(int64_t value) { return g_host->Provider_ModelProto__set_ir_version(this, value); }

  Provider_ModelProto() = delete;
  Provider_ModelProto(const Provider_ModelProto&) = delete;
  void operator=(const Provider_ModelProto&) = delete;
};

struct Provider_TensorProto {
  static void operator delete(void* p) { g_host->Provider_TensorProto__operator_delete(reinterpret_cast<Provider_TensorProto*>(p)); }
  void operator=(const Provider_TensorProto& v) { g_host->Provider_TensorProto__operator_assign(this, v); }

  Provider_TensorProto() = delete;
  Provider_TensorProto(const Provider_TensorProto&) = delete;
};

struct Provider_TensorProtos {
  Provider_TensorProto* Add() { return g_host->Provider_TensorProtos__Add(this); }

  PROVIDER_DISALLOW_ALL(Provider_TensorProtos)
};

struct Provider_TensorShapeProto_Dimension {
  const std::string& dim_param() const { return g_host->Provider_TensorShapeProto_Dimension__dim_param(this); }

  PROVIDER_DISALLOW_ALL(Provider_TensorShapeProto_Dimension)
};

struct Provider_TensorShapeProto_Dimensions {
  IteratorHolder<Provider_TensorShapeProto_Dimension_Iterator, const Provider_TensorShapeProto_Dimension> begin() const { return g_host->Provider_TensorShapeProto_Dimensions__begin(this); }
  IteratorHolder<Provider_TensorShapeProto_Dimension_Iterator, const Provider_TensorShapeProto_Dimension> end() const { return g_host->Provider_TensorShapeProto_Dimensions__end(this); }

  PROVIDER_DISALLOW_ALL(Provider_TensorShapeProto_Dimensions)
};

struct Provider_TensorShapeProto {
  int dim_size() const { return g_host->Provider_TensorShapeProto__dim_size(this); }
  const Provider_TensorShapeProto_Dimensions& dim() const { return g_host->Provider_TensorShapeProto__dim(this); }

  PROVIDER_DISALLOW_ALL(Provider_TensorShapeProto)
};

struct Provider_ValueInfoProto {
  const Provider_TypeProto& type() const { return g_host->Provider_ValueInfoProto__type(this); }
  void operator=(const Provider_ValueInfoProto& v) { g_host->Provider_ValueInfoProto__operator_assign(this, v); }

  Provider_ValueInfoProto() = delete;
  Provider_ValueInfoProto(const Provider_ValueInfoProto&) = delete;
  static void operator delete(void*) = delete;
};

struct Provider_ValueInfoProtos {
  Provider_ValueInfoProto* Add() { return g_host->Provider_ValueInfoProtos__Add(this); }
  const Provider_ValueInfoProto& operator[](int index) const { return g_host->Provider_ValueInfoProtos__operator_array(this, index); }

  PROVIDER_DISALLOW_ALL(Provider_ValueInfoProtos)
};

struct Provider_ComputeCapability {
  static std::unique_ptr<Provider_ComputeCapability> Create(std::unique_ptr<Provider_IndexedSubGraph> t_sub_graph) { return g_host->Provider_ComputeCapability__construct(std::move(t_sub_graph)); }
  static void operator delete(void* p) { g_host->Provider_ComputeCapability__operator_delete(reinterpret_cast<Provider_ComputeCapability*>(p)); }

  std::unique_ptr<Provider_IndexedSubGraph>& SubGraph() { return g_host->Provider_ComputeCapability__SubGraph(this); }

  Provider_ComputeCapability() = delete;
  Provider_ComputeCapability(const Provider_ComputeCapability&) = delete;
  void operator=(const Provider_ComputeCapability&) = delete;
};

struct Provider_DataTransferManager {
  Status CopyTensor(const Provider_Tensor& src, Provider_Tensor& dst, int exec_queue_id) const { return g_host->Provider_DataTransferManager__CopyTensor(this, src, dst, exec_queue_id); }

  PROVIDER_DISALLOW_ALL(Provider_DataTransferManager)
};

struct Provider_IDataTransfer {
  static void operator delete(void* p) { g_host->Provider_IDataTransfer__operator_delete(reinterpret_cast<Provider_IDataTransfer*>(p)); }

  Provider_IDataTransfer() = delete;
  Provider_IDataTransfer(const Provider_IDataTransfer&) = delete;
  void operator=(const Provider_IDataTransfer&) = delete;
};

struct Provider_IndexedSubGraph_MetaDef {
  static std::unique_ptr<Provider_IndexedSubGraph_MetaDef> Create() { return g_host->Provider_IndexedSubGraph_MetaDef__construct(); }
  static void operator delete(void* p) { g_host->Provider_IndexedSubGraph_MetaDef__operator_delete(reinterpret_cast<Provider_IndexedSubGraph_MetaDef*>(p)); }

  const std::string& name() const { return g_host->Provider_IndexedSubGraph_MetaDef__name(const_cast<Provider_IndexedSubGraph_MetaDef*>(this)); }
  std::string& name() { return g_host->Provider_IndexedSubGraph_MetaDef__name(this); }
  const std::string& domain() const { return g_host->Provider_IndexedSubGraph_MetaDef__domain(const_cast<Provider_IndexedSubGraph_MetaDef*>(this)); }
  std::string& domain() { return g_host->Provider_IndexedSubGraph_MetaDef__domain(this); }
  int since_version() const { return g_host->Provider_IndexedSubGraph_MetaDef__since_version(const_cast<Provider_IndexedSubGraph_MetaDef*>(this)); }
  int& since_version() { return g_host->Provider_IndexedSubGraph_MetaDef__since_version(this); }

  ONNX_NAMESPACE::OperatorStatus& status() { return g_host->Provider_IndexedSubGraph_MetaDef__status(this); }

  const std::vector<std::string>& inputs() const { return g_host->Provider_IndexedSubGraph_MetaDef__inputs(const_cast<Provider_IndexedSubGraph_MetaDef*>(this)); }
  std::vector<std::string>& inputs() { return g_host->Provider_IndexedSubGraph_MetaDef__inputs(this); }
  const std::vector<std::string>& outputs() const { return g_host->Provider_IndexedSubGraph_MetaDef__outputs(const_cast<Provider_IndexedSubGraph_MetaDef*>(this)); }
  std::vector<std::string>& outputs() { return g_host->Provider_IndexedSubGraph_MetaDef__outputs(this); }
  Provider_NodeAttributes& attributes() { return g_host->Provider_IndexedSubGraph_MetaDef__attributes(this); }

  std::string& doc_string() { return g_host->Provider_IndexedSubGraph_MetaDef__doc_string(this); }

  Provider_IndexedSubGraph_MetaDef() = delete;
  Provider_IndexedSubGraph_MetaDef(const Provider_IndexedSubGraph_MetaDef&) = delete;
  void operator=(const Provider_IndexedSubGraph_MetaDef&) = delete;
};

struct Provider_IndexedSubGraph {
  static std::unique_ptr<Provider_IndexedSubGraph> Create() { return g_host->Provider_IndexedSubGraph__construct(); }
  static void operator delete(void* p) { g_host->Provider_IndexedSubGraph__operator_delete(reinterpret_cast<Provider_IndexedSubGraph*>(p)); }

  std::vector<onnxruntime::NodeIndex>& Nodes() { return g_host->Provider_IndexedSubGraph__Nodes(this); }

  void SetMetaDef(std::unique_ptr<Provider_IndexedSubGraph_MetaDef>&& meta_def_) { return g_host->Provider_IndexedSubGraph__SetMetaDef(this, std::move(*reinterpret_cast<std::unique_ptr<Provider_IndexedSubGraph_MetaDef>*>(&meta_def_))); }
  const Provider_IndexedSubGraph_MetaDef* GetMetaDef() const { return reinterpret_cast<const Provider_IndexedSubGraph_MetaDef*>(g_host->Provider_IndexedSubGraph__GetMetaDef(this)); }

  Provider_IndexedSubGraph() = delete;
  Provider_IndexedSubGraph(const Provider_IndexedSubGraph&) = delete;
  void operator=(const Provider_IndexedSubGraph&) = delete;
};

struct Provider_KernelDef {
  static void operator delete(void* p) { g_host->Provider_KernelDef__operator_delete(reinterpret_cast<Provider_KernelDef*>(p)); }

  Provider_KernelDef() = delete;
  Provider_KernelDef(const Provider_KernelDef*) = delete;
  void operator=(const Provider_KernelDef&) = delete;
};
#endif

using Provider_KernelCreateFn = std::function<Provider_OpKernel*(const Provider_OpKernelInfo& info)>;
using Provider_KernelCreatePtrFn = std::add_pointer<Provider_OpKernel*(const Provider_OpKernelInfo& info)>::type;

struct Provider_KernelCreateInfo {
  std::unique_ptr<Provider_KernelDef> kernel_def;  // Owned and stored in the global kernel registry.
  Provider_KernelCreateFn kernel_create_func;

  Provider_KernelCreateInfo(std::unique_ptr<Provider_KernelDef> definition,
                            Provider_KernelCreateFn create_func)
      : kernel_def(std::move(definition)),
        kernel_create_func(create_func) {}

  Provider_KernelCreateInfo(Provider_KernelCreateInfo&& other) noexcept
      : kernel_def(std::move(other.kernel_def)),
        kernel_create_func(std::move(other.kernel_create_func)) {}
};

using Provider_BuildKernelCreateInfoFn = Provider_KernelCreateInfo (*)();

#ifndef PROVIDER_BRIDGE_ORT
struct Provider_KernelDefBuilder {
  static std::unique_ptr<Provider_KernelDefBuilder> Create() { return g_host->Provider_KernelDefBuilder__construct(); }
  static void operator delete(void* p) { g_host->Provider_KernelDefBuilder__operator_delete(reinterpret_cast<Provider_KernelDefBuilder*>(p)); }

  Provider_KernelDefBuilder& SetName(const char* op_name) {
    g_host->Provider_KernelDefBuilder__SetName(this, op_name);
    return *this;
  }
  Provider_KernelDefBuilder& SetDomain(const char* domain) {
    g_host->Provider_KernelDefBuilder__SetDomain(this, domain);
    return *this;
  }
  Provider_KernelDefBuilder& SinceVersion(int since_version) {
    g_host->Provider_KernelDefBuilder__SinceVersion(this, since_version);
    return *this;
  }
  Provider_KernelDefBuilder& Provider(const char* provider_type) {
    g_host->Provider_KernelDefBuilder__Provider(this, provider_type);
    return *this;
  }
  Provider_KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType supported_type) {
    g_host->Provider_KernelDefBuilder__TypeConstraint(this, arg_name, supported_type);
    return *this;
  }
  Provider_KernelDefBuilder& TypeConstraint(const char* arg_name, const std::vector<MLDataType>& supported_types) {
    g_host->Provider_KernelDefBuilder__TypeConstraint(this, arg_name, supported_types);
    return *this;
  }
  Provider_KernelDefBuilder& InputMemoryType(OrtMemType type, int input_index) {
    g_host->Provider_KernelDefBuilder__InputMemoryType(this, type, input_index);
    return *this;
  }
  Provider_KernelDefBuilder& OutputMemoryType(OrtMemType type, int input_index) {
    g_host->Provider_KernelDefBuilder__OutputMemoryType(this, type, input_index);
    return *this;
  }
  Provider_KernelDefBuilder& ExecQueueId(int queue_id) {
    g_host->Provider_KernelDefBuilder__ExecQueueId(this, queue_id);
    return *this;
  }

  std::unique_ptr<Provider_KernelDef> Build() { return g_host->Provider_KernelDefBuilder__Build(this); }

  Provider_KernelDefBuilder() = delete;
  Provider_KernelDefBuilder(const Provider_KernelDefBuilder&) = delete;
  void operator=(const Provider_KernelDefBuilder&) = delete;
};

struct Provider_KernelRegistry {
  static std::shared_ptr<Provider_KernelRegistry> Create() { return g_host->Provider_KernelRegistry__construct(); }
  static void operator delete(void* p) { g_host->Provider_KernelRegistry__operator_delete(reinterpret_cast<Provider_KernelRegistry*>(p)); }

  Status Register(Provider_KernelCreateInfo&& create_info) { return g_host->Provider_KernelRegistry__Register(this, std::move(create_info)); }

  Provider_KernelRegistry() = delete;
  Provider_KernelRegistry(const Provider_KernelRegistry&) = delete;
  void operator=(const Provider_KernelRegistry&) = delete;
};

struct Provider_Function {
  const Provider_Graph& Body() const { return g_host->Provider_Function__Body(this); }

  PROVIDER_DISALLOW_ALL(Provider_Function)
};

struct Provider_Node {
  const std::string& Name() const noexcept { return g_host->Provider_Node__Name(this); }
  const std::string& Description() const noexcept { return g_host->Provider_Node__Description(this); }
  const std::string& Domain() const noexcept { return g_host->Provider_Node__Domain(this); }
  const std::string& OpType() const noexcept { return g_host->Provider_Node__OpType(this); }

  const Provider_Function* GetFunctionBody() const noexcept { return g_host->Provider_Node__GetFunctionBody(this); }

  ConstPointerContainer<std::vector<Provider_NodeArg*>> ImplicitInputDefs() const noexcept { return g_host->Provider_Node__ImplicitInputDefs(this); }

  ConstPointerContainer<std::vector<Provider_NodeArg*>> InputDefs() const noexcept { return g_host->Provider_Node__InputDefs(this); }
  ConstPointerContainer<std::vector<Provider_NodeArg*>> OutputDefs() const noexcept { return g_host->Provider_Node__OutputDefs(this); }
  NodeIndex Index() const noexcept { return g_host->Provider_Node__Index(this); }

  void ToProto(Provider_NodeProto& proto, bool update_subgraphs = false) const { return g_host->Provider_Node__ToProto(this, proto, update_subgraphs); }

  const Provider_NodeAttributes& GetAttributes() const noexcept { return g_host->Provider_Node__GetAttributes(this); }
  size_t GetInputEdgesCount() const noexcept { return g_host->Provider_Node__GetInputEdgesCount(this); }
  size_t GetOutputEdgesCount() const noexcept { return g_host->Provider_Node__GetOutputEdgesCount(this); }

  struct NodeConstIterator {
    NodeConstIterator(std::unique_ptr<Provider_Node__NodeIterator> p) : impl_{std::move(p)} {}

    bool operator!=(const NodeConstIterator& p_other) const { return *impl_ != *p_other.impl_; }

    void operator++() { impl_->operator++(); }

    const Provider_Node& operator*() const { return impl_->operator*(); }
    const Provider_Node* operator->() const { return &impl_->operator*(); }

    std::unique_ptr<Provider_Node__NodeIterator> impl_;
  };

  NodeConstIterator InputNodesBegin() const noexcept { return g_host->Provider_Node__InputNodesBegin(this); }
  NodeConstIterator InputNodesEnd() const noexcept { return g_host->Provider_Node__InputNodesEnd(this); }

  struct EdgeConstIterator {
    EdgeConstIterator(std::unique_ptr<Provider_Node__EdgeIterator> p) : impl_{std::move(p)} {}

    bool operator!=(const EdgeConstIterator& p_other) const {
      return *impl_ != *p_other.impl_;
    }

    void operator++() { impl_->operator++(); }
    const Provider_Node__EdgeIterator* operator->() const { return impl_.get(); }

    std::unique_ptr<Provider_Node__EdgeIterator> impl_;
  };

  EdgeConstIterator OutputEdgesBegin() const noexcept { return g_host->Provider_Node__OutputEdgesBegin(this); }
  EdgeConstIterator OutputEdgesEnd() const noexcept { return g_host->Provider_Node__OutputEdgesEnd(this); }

  PROVIDER_DISALLOW_ALL(Provider_Node)
};

struct Provider_NodeArg {
  const std::string& Name() const noexcept { return g_host->Provider_NodeArg__Name(this); }
  const Provider_TensorShapeProto* Shape() const { return g_host->Provider_NodeArg__Shape(this); }
  ONNX_NAMESPACE::DataType Type() const noexcept { return g_host->Provider_NodeArg__Type(this); }
  const Provider_NodeArgInfo& ToProto() const noexcept { return g_host->Provider_NodeArg__ToProto(this); }
  bool Exists() const noexcept { return g_host->Provider_NodeArg__Exists(this); }
  const Provider_TypeProto* TypeAsProto() const noexcept { return g_host->Provider_NodeArg__TypeAsProto(this); }

  PROVIDER_DISALLOW_ALL(Provider_NodeArg)
};

struct Provider_NodeAttributes {
  static std::unique_ptr<Provider_NodeAttributes> Create() { return g_host->Provider_NodeAttributes__construct(); }
  void operator=(const Provider_NodeAttributes& v) { return g_host->Provider_NodeAttributes__operator_assign(this, v); }
  static void operator delete(void* p) { g_host->Provider_NodeAttributes__operator_delete(reinterpret_cast<Provider_NodeAttributes*>(p)); }

  size_t size() const { return g_host->Provider_NodeAttributes__size(this); }
  void clear() noexcept { g_host->Provider_NodeAttributes__clear(this); }
  Provider_AttributeProto& operator[](const std::string& string) { return g_host->Provider_NodeAttributes__operator_array(this, string); }

  IteratorHolder<Provider_NodeAttributes_Iterator, std::pair<std::string&, Provider_AttributeProto&>> begin() const { return g_host->Provider_NodeAttributes__begin(this); }
  IteratorHolder<Provider_NodeAttributes_Iterator, std::pair<std::string&, Provider_AttributeProto&>> end() const { return g_host->Provider_NodeAttributes__end(this); }
  IteratorHolder<Provider_NodeAttributes_Iterator, std::pair<std::string&, Provider_AttributeProto&>> find(const std::string& key) const { return g_host->Provider_NodeAttributes__find(this, key); }
  void insert(const Provider_NodeAttributes& v) { return g_host->Provider_NodeAttributes__insert(this, v); }

  Provider_NodeAttributes() = delete;
  Provider_NodeAttributes(const Provider_NodeAttributes&) = delete;
};

struct Provider_Model {
  static void operator delete(void* p) { g_host->Provider_Model__operator_delete(reinterpret_cast<Provider_Model*>(p)); }

  Provider_Graph& MainGraph() { return g_host->Provider_Model__MainGraph(this); }

  std::unique_ptr<Provider_ModelProto> ToProto() { return g_host->Provider_Model__ToProto(this); }

  Provider_Model() = delete;
  Provider_Model(const Provider_Model&) = delete;
  void operator=(const Provider_Model&) = delete;
};

struct Provider_Graph {
  std::unique_ptr<Provider_GraphViewer> CreateGraphViewer() const { return g_host->Provider_Graph__CreateGraphViewer(this); }
  std::unique_ptr<Provider_GraphProto> ToGraphProto() const { return g_host->Provider_Graph__ToGraphProto(this); }

  Provider_NodeArg& GetOrCreateNodeArg(const std::string& name, const ONNX_NAMESPACE::Provider_TypeProto* p_arg_type) { return g_host->Provider_Graph__GetOrCreateNodeArg(this, name, p_arg_type); }

  Status Resolve() { return g_host->Provider_Graph__Resolve(this); }
  void AddInitializedTensor(const ONNX_NAMESPACE::Provider_TensorProto& tensor) { return g_host->Provider_Graph__AddInitializedTensor(this, tensor); }
  Provider_Node& AddNode(const std::string& name, const std::string& op_type, const std::string& description, const std::vector<Provider_NodeArg*>& input_args, const std::vector<Provider_NodeArg*>& output_args, const Provider_NodeAttributes* attributes, const std::string& domain) { return g_host->Provider_Graph__AddNode(this, name, op_type, description, input_args, output_args, attributes, domain); }

  const std::vector<const Provider_NodeArg*>& GetOutputs() const noexcept { return g_host->Provider_Graph__GetOutputs(this); }
  void SetOutputs(const std::vector<const Provider_NodeArg*>& outputs) { return g_host->Provider_Graph__SetOutputs(this, outputs); }

  const std::vector<const Provider_NodeArg*>& GetInputs() const noexcept { return g_host->Provider_Graph__GetInputs(this); }

  bool GetInitializedTensor(const std::string& tensor_name, const Provider_TensorProto*& value) const { return g_host->Provider_Graph__GetInitializedTensor(this, tensor_name, value); }

  PROVIDER_DISALLOW_ALL(Provider_Graph)
};

struct Provider_GraphViewer {
  static void operator delete(void* p) { g_host->Provider_GraphViewer__operator_delete(reinterpret_cast<Provider_GraphViewer*>(p)); }

  std::unique_ptr<Provider_Model> CreateModel(const logging::Logger& logger) const { return g_host->Provider_GraphViewer__CreateModel(this, logger); }

  const std::string& Name() const noexcept { return g_host->Provider_GraphViewer__Name(this); }

  const Provider_Node* GetNode(NodeIndex node_index) const { return g_host->Provider_GraphViewer__GetNode(this, node_index); }
  const Provider_NodeArg* GetNodeArg(const std::string& name) const { return g_host->Provider_GraphViewer__GetNodeArg(this, name); }

  bool IsSubgraph() const { return g_host->Provider_GraphViewer__IsSubgraph(this); }

  int NumberOfNodes() const noexcept { return g_host->Provider_GraphViewer__NumberOfNodes(this); }
  int MaxNodeIndex() const noexcept { return g_host->Provider_GraphViewer__MaxNodeIndex(this); }

  const std::vector<const Provider_NodeArg*>& GetInputs() const noexcept { return g_host->Provider_GraphViewer__GetInputs(this); }
  const std::vector<const Provider_NodeArg*>& GetOutputs() const noexcept { return g_host->Provider_GraphViewer__GetOutputs(this); }
  const std::vector<const Provider_NodeArg*>& GetValueInfo() const noexcept { return g_host->Provider_GraphViewer__GetValueInfo(this); }

  const Provider_InitializedTensorSet& GetAllInitializedTensors() const noexcept { return g_host->Provider_GraphViewer__GetAllInitializedTensors(this); }
  bool GetInitializedTensor(const std::string& tensor_name, const Provider_TensorProto*& value) const { return g_host->Provider_GraphViewer__GetInitializedTensor(this, tensor_name, value); }

  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept { return g_host->Provider_GraphViewer__DomainToVersionMap(this); }

  const std::vector<NodeIndex>& GetNodesInTopologicalOrder() const { return g_host->Provider_GraphViewer__GetNodesInTopologicalOrder(this); }

  Provider_GraphViewer() = delete;
  Provider_GraphViewer(const Provider_GraphViewer&) = delete;
  void operator=(const Provider_GraphViewer&) = delete;
};
#endif

struct Provider_OpKernel_Base {
  const Provider_OpKernelInfo& GetInfo() const { return g_host->Provider_OpKernel_Base__GetInfo(this); }

  PROVIDER_DISALLOW_ALL(Provider_OpKernel_Base)
};

#ifndef PROVIDER_BRIDGE_ORT
struct Provider_OpKernelContext {
  const Provider_Tensor* Input_Tensor(int index) const { return g_host->Provider_OpKernelContext__Input_Tensor(this, index); }

  template <typename T>
  const T* Input(int index) const;

  Provider_Tensor* Output(int index, const TensorShape& shape) { return g_host->Provider_OpKernelContext__Output(this, index, shape); }

  PROVIDER_DISALLOW_ALL(Provider_OpKernelContext)
};

template <>
inline const Provider_Tensor* Provider_OpKernelContext::Input<Provider_Tensor>(int index) const {
  return Input_Tensor(index);
}

struct Provider_OpKernelInfo {
  template <typename T>
  Status GetAttr(const std::string& name, T* value) const;

  Status GetAttr(const std::string& name, int64_t* value) const { return g_host->Provider_OpKernelInfo__GetAttr_int64(this, name, value); }
  Status GetAttr(const std::string& name, float* value) const { return g_host->Provider_OpKernelInfo__GetAttr_float(this, name, value); }

  const Provider_DataTransferManager& GetDataTransferManager() const noexcept { return g_host->Provider_OpKernelInfo__GetDataTransferManager(this); }
  int GetKernelDef_ExecQueueId() const noexcept { return g_host->Provider_OpKernelInfo__GetKernelDef_ExecQueueId(this); }

  PROVIDER_DISALLOW_ALL(Provider_OpKernelInfo)
};

template <>
inline Status Provider_OpKernelInfo::GetAttr<int64_t>(const std::string& name, int64_t* value) const {
  return GetAttr(name, value);
}

template <>
inline Status Provider_OpKernelInfo::GetAttr<float>(const std::string& name, float* value) const {
  return GetAttr(name, value);
}

struct Provider_Tensor {
  float* MutableData_float() { return g_host->Provider_Tensor__MutableData_float(this); }
  const float* Data_float() const { return g_host->Provider_Tensor__Data_float(this); }

  template <typename T>
  T* MutableData();

  template <typename T>
  const T* Data() const;

  void* MutableDataRaw() noexcept { return g_host->Provider_Tensor__MutableDataRaw(this); }
  const void* DataRaw() const noexcept { return g_host->Provider_Tensor__DataRaw(this); }

  const TensorShape& Shape() const { return g_host->Provider_Tensor__Shape(this); }
  size_t SizeInBytes() const { return g_host->Provider_Tensor__SizeInBytes(this); }
  const OrtMemoryInfo& Location() const { return g_host->Provider_Tensor__Location(this); }

  PROVIDER_DISALLOW_ALL(Provider_Tensor)
};

template <>
inline float* Provider_Tensor::MutableData<float>() { return MutableData_float(); }

template <>
inline const float* Provider_Tensor::Data<float>() const { return Data_float(); }
#endif

}  // namespace onnxruntime
