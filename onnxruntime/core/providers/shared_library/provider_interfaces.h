// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Public wrappers around internal ort interfaces (currently)
// In the future the internal implementations could derive from these to remove the need for the wrapper implementations

#include "core/framework/func_api.h"

namespace onnxruntime {

struct Provider_GraphProto;
struct Provider_ModelProto;
struct Provider_NodeProto;
struct Provider_TensorProto;
struct Provider_TensorProtos;
struct Provider_TensorShapeProto_Dimension;
struct Provider_TensorShapeProto_Dimensions;
struct Provider_TensorShapeProto;
struct Provider_TypeProto_Tensor;
struct Provider_TypeProto;
struct Provider_ValueInfoProto;
struct Provider_ValueInfoProtos;

template <typename T, typename TResult>
struct IteratorHolder {
  IteratorHolder(std::unique_ptr<T>&& p) : p_{std::move(p)} {}

  bool operator!=(const IteratorHolder& p) const { return p_->operator!=(*p.p_); }

  void operator++() { p_->operator++(); }
  TResult& operator*() { return p_->operator*(); }

  std::unique_ptr<T> p_;
};

struct Provider_TensorShapeProto_Dimension_Iterator {
  virtual ~Provider_TensorShapeProto_Dimension_Iterator() {}

  virtual bool operator!=(const Provider_TensorShapeProto_Dimension_Iterator& p) const = 0;

  virtual void operator++() = 0;
  virtual const Provider_TensorShapeProto_Dimension& operator*() = 0;
};

}  // namespace onnxruntime

namespace ONNX_NAMESPACE {
using namespace onnxruntime;

enum AttributeProto_AttributeType : int;
enum OperatorStatus : int;

// String pointer as unique TypeProto identifier.
using DataType = const std::string*;

#if 0
struct Provider_TypeProto_Tensor {
  virtual int32_t elem_type() const = 0;
};

struct Provider_TypeProto {
  virtual const Provider_TypeProto_Tensor& tensor_type() const = 0;
};

struct Provider_TensorProto {
  virtual ~Provider_TensorProto() = default;

  virtual void CopyFrom(const Provider_TensorProto& v) = 0;

  void operator=(const Provider_TensorProto& v) { CopyFrom(v); }
};
#endif

struct Provider_AttributeProto {
  static std::unique_ptr<Provider_AttributeProto> Create();

  virtual ~Provider_AttributeProto() = default;
  virtual std::unique_ptr<Provider_AttributeProto> Clone() const = 0;

  virtual AttributeProto_AttributeType type() const = 0;
  virtual int ints_size() const = 0;
  virtual int64_t ints(int i) const = 0;
  virtual int64_t i() const = 0;
  virtual float f() const = 0;
  virtual void set_s(const ::std::string& value) = 0;
  virtual const ::std::string& s() const = 0;
  virtual void set_name(const ::std::string& value) = 0;
  virtual void set_type(AttributeProto_AttributeType value) = 0;
  virtual Provider_TensorProto* add_tensors() = 0;

  void operator=(const Provider_AttributeProto& v) = delete;
};

// This is needed since Provider_NodeAttributes is a map of unique_ptr to Provider_AttributeProto and that won't work since unique_ptrs are not copyable
// (supposedly this should work in the latest C++ STL but it didn't for me so I used this to make it copyable)
struct Provider_AttributeProto_Copyable {
  Provider_AttributeProto_Copyable() = default;
  Provider_AttributeProto_Copyable(const Provider_AttributeProto_Copyable& copy) : p_{copy->Clone()} {}

  void operator=(std::unique_ptr<Provider_AttributeProto>&& p) { p_ = std::move(p); }
  void operator=(const Provider_AttributeProto_Copyable& p) { p_ = p->Clone(); }

  Provider_AttributeProto& operator*() const { return *p_.get(); }
  Provider_AttributeProto* operator->() const { return p_.get(); }

  std::unique_ptr<Provider_AttributeProto> p_;
};

#if 0
struct Provider_TensorShapeProto {
  virtual int dim_size() const = 0;
  virtual const Provider_TensorShapeProto_Dimensions& dim() const = 0;
};

struct Provider_ValueInfoProto {
  virtual const Provider_TypeProto& type() const = 0;
};

struct Provider_ValueInfoProtos {
  virtual Provider_ValueInfoProto* Add() = 0;

  virtual const Provider_ValueInfoProto& operator[](int index) const = 0;
};

struct Provider_TensorProtos {
  virtual Provider_TensorProto* Add() = 0;
};

struct Provider_NodeProto {
};

struct Provider_GraphProto {
  virtual ~Provider_GraphProto() {}

  virtual Provider_ValueInfoProtos& mutable_input() = 0;

  virtual const Provider_ValueInfoProtos& output() const = 0;
  virtual Provider_ValueInfoProtos& mutable_output() = 0;

  virtual Provider_ValueInfoProtos& mutable_value_info() = 0;
  virtual Provider_TensorProtos& mutable_initializer() = 0;
  virtual Provider_NodeProto& add_node() = 0;

  virtual void operator=(Provider_GraphProto& v) = 0;
};

struct Provider_ModelProto {
  virtual ~Provider_ModelProto() {}

  virtual bool SerializeToString(std::string& string) const = 0;
  virtual bool SerializeToOstream(std::ostream& output) const = 0;

  virtual const Provider_GraphProto& graph() const = 0;
  virtual Provider_GraphProto& mutable_graph() = 0;

  virtual void set_ir_version(int64_t value) = 0;
};
#endif

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

struct ProviderHost;
struct Provider_ComputeCapability;
struct Provider_IExecutionProvider;
struct Provider_IndexedSubGraph;
struct Provider_IndexedSubGraph_MetaDef;
struct Provider_KernelCreateInfo;
struct Provider_KernelDef;
struct Provider_KernelDefBuilder;
struct Provider_KernelRegistry;
struct Provider_Function;
struct Provider_Graph;
struct Provider_GraphViewer;
struct Provider_Model;
struct Provider_Node;
struct Provider_NodeArg;

struct Provider_IExecutionProviderFactory {
  virtual ~Provider_IExecutionProviderFactory() = default;
  virtual std::unique_ptr<Provider_IExecutionProvider> CreateProvider() = 0;
};

//struct KernelCreateInfo;

class DataTypeImpl;
using MLDataType = const DataTypeImpl*;

struct Provider_OrtDevice {
  virtual ~Provider_OrtDevice() {}
};

struct Provider_OrtMemoryInfo {
  static std::unique_ptr<Provider_OrtMemoryInfo> Create(const char* name_, OrtAllocatorType type_, Provider_OrtDevice* device_ = nullptr, int id_ = 0, OrtMemType mem_type_ = OrtMemTypeDefault);
  virtual ~Provider_OrtMemoryInfo() {}

  void operator=(const Provider_OrtMemoryInfo& v) = delete;
};

template <typename T>
using Provider_IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

struct Provider_IAllocator {
  virtual ~Provider_IAllocator() {}

  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p) = 0;

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

  void operator=(const Provider_IAllocator& v) = delete;
};

struct Provider_IDeviceAllocator : Provider_IAllocator {
  virtual bool AllowsArena() const = 0;
};

using Provider_AllocatorPtr = std::shared_ptr<Provider_IAllocator>;
using Provider_DeviceAllocatorFactory = std::function<std::unique_ptr<Provider_IDeviceAllocator>(int)>;

struct Provider_DeviceAllocatorRegistrationInfo {
  OrtMemType mem_type;
  Provider_DeviceAllocatorFactory factory;
  size_t max_mem;
};

class TensorShape;

struct Provider_Tensor {
  virtual float* MutableData_float() = 0;
  virtual const float* Data_float() const = 0;

  template <typename T>
  T* MutableData();

  template <typename T>
  const T* Data() const;

  virtual const TensorShape& Shape() const = 0;
};

template <>
inline float* Provider_Tensor::MutableData<float>() { return MutableData_float(); }

template <>
inline const float* Provider_Tensor::Data<float>() const { return Data_float(); }

struct Provider_DataTransferManager {
  virtual Status CopyTensor(const Provider_Tensor& src, Provider_Tensor& dst, int exec_queue_id) const = 0;
};

struct Provider_OpKernelInfo {
  virtual Status GetAttr(const std::string& name, int64_t* value) const = 0;
  virtual Status GetAttr(const std::string& name, float* value) const = 0;

  template <typename T>
  Status GetAttr(const std::string& name, T* value) const;

  virtual const Provider_DataTransferManager& GetDataTransferManager() const noexcept = 0;
  virtual int GetKernelDef_ExecQueueId() const noexcept = 0;
};

template <>
inline Status Provider_OpKernelInfo::GetAttr<int64_t>(const std::string& name, int64_t* value) const {
  return GetAttr(name, value);
}

template <>
inline Status Provider_OpKernelInfo::GetAttr<float>(const std::string& name, float* value) const {
  return GetAttr(name, value);
}

struct Provider_OpKernelContext {
  virtual const Provider_Tensor* Input_Tensor(int index) const = 0;

  template <typename T>
  const T* Input(int index) const;

  virtual Provider_Tensor* Output(int index, const TensorShape& shape) = 0;
};

template <>
inline const Provider_Tensor* Provider_OpKernelContext::Input<Provider_Tensor>(int index) const {
  return Input_Tensor(index);
}

struct Provider_OpKernel {
  Provider_OpKernel(const Provider_OpKernelInfo& /*info*/) {}
  virtual ~Provider_OpKernel() = default;

  virtual Status Compute(Provider_OpKernelContext* context) const = 0;
};

#if 0
struct Provider_KernelDef {
  virtual ~Provider_KernelDef() = default;
};

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

struct Provider_KernelDefBuilder {
  static std::unique_ptr<Provider_KernelDefBuilder> Create();

  virtual ~Provider_KernelDefBuilder() = default;
  virtual Provider_KernelDefBuilder& SetName(const char* op_name) = 0;
  virtual Provider_KernelDefBuilder& SetDomain(const char* domain) = 0;
  virtual Provider_KernelDefBuilder& SinceVersion(int since_version) = 0;
  virtual Provider_KernelDefBuilder& Provider(const char* provider_type) = 0;
  virtual Provider_KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType supported_type) = 0;
  virtual Provider_KernelDefBuilder& TypeConstraint(const char* arg_name, const std::vector<MLDataType>& supported_types) = 0;
  virtual Provider_KernelDefBuilder& InputMemoryType(OrtMemType type, int input_index) = 0;
  virtual Provider_KernelDefBuilder& OutputMemoryType(OrtMemType type, int input_index) = 0;
  virtual Provider_KernelDefBuilder& ExecQueueId(int queue_id) = 0;

  virtual std::unique_ptr<Provider_KernelDef> Build() = 0;

  void operator=(const Provider_KernelDefBuilder& v) = delete;
};
#endif

using NodeIndex = size_t;
using Provider_NodeArgInfo = Provider_ValueInfoProto;
using Provider_NodeAttributes = std::unordered_map<std::string, ONNX_NAMESPACE::Provider_AttributeProto_Copyable>;

using Provider_InitializedTensorSet = std::unordered_map<std::string, const ONNX_NAMESPACE::Provider_TensorProto*>;

#if 0
struct Provider_NodeArg {
  virtual ~Provider_NodeArg() = default;
  virtual const std::string& Name() const noexcept = 0;
  virtual const ONNX_NAMESPACE::Provider_TensorShapeProto* Shape() const = 0;
  virtual ONNX_NAMESPACE::DataType Type() const noexcept = 0;
  virtual const Provider_NodeArgInfo& ToProto() const noexcept = 0;
  virtual bool Exists() const noexcept = 0;
  virtual const ONNX_NAMESPACE::Provider_TypeProto* TypeAsProto() const noexcept = 0;

  void operator=(const Provider_NodeArg& v) = delete;
};

struct Provider_Graph {
  virtual std::unique_ptr<Provider_GraphViewer> CreateGraphViewer() const = 0;
  virtual std::unique_ptr<Provider_GraphProto> CreateGraphProto() const = 0;

  virtual Provider_NodeArg& GetOrCreateNodeArg(const std::string& name, const ONNX_NAMESPACE::Provider_TypeProto* p_arg_type) = 0;

  virtual Status Resolve() = 0;
  virtual void AddInitializedTensor(const ONNX_NAMESPACE::Provider_TensorProto& tensor) = 0;
  virtual Provider_Node& AddNode(const std::string& name, const std::string& op_type, const std::string& description, const std::vector<Provider_NodeArg*>& input_args, const std::vector<Provider_NodeArg*>& output_args, const Provider_NodeAttributes* attributes, const std::string& domain) = 0;

  virtual const std::vector<const Provider_NodeArg*>& GetOutputs() const noexcept = 0;
  virtual void SetOutputs(const std::vector<const Provider_NodeArg*>& outputs) = 0;

  virtual const std::vector<const Provider_NodeArg*>& GetInputs() const noexcept = 0;
};

struct Provider_Function {
  //  virtual const ONNX_NAMESPACE::OpSchema& OpSchema() const = 0;

  /** Gets the Graph instance for the Function body subgraph. */
  virtual const Provider_Graph& Body() const = 0;

  /** Gets the IndexedSubGraph for the Function. */
  //  virtual const IndexedSubGraph& GetIndexedSubGraph() const = 0;
};

struct Provider_Node {
  virtual ~Provider_Node() = default;

  virtual const std::string& Name() const noexcept = 0;
  virtual const std::string& Description() const noexcept = 0;
  virtual const std::string& Domain() const noexcept = 0;
  virtual const std::string& OpType() const noexcept = 0;

  virtual const Provider_Function* GetFunctionBody() const noexcept = 0;

  virtual ConstPointerContainer<std::vector<Provider_NodeArg*>> InputDefs() const noexcept = 0;
  virtual ConstPointerContainer<std::vector<Provider_NodeArg*>> OutputDefs() const noexcept = 0;
  virtual NodeIndex Index() const noexcept = 0;

  virtual void ToProto(Provider_NodeProto& proto, bool update_subgraphs = false) const = 0;

  virtual const Provider_NodeAttributes& GetAttributes() const noexcept = 0;
  virtual size_t GetInputEdgesCount() const noexcept = 0;
  virtual size_t GetOutputEdgesCount() const noexcept = 0;

  struct Provider_NodeIterator {
    virtual ~Provider_NodeIterator() {}
    virtual bool operator!=(const Provider_NodeIterator& p) const = 0;

    virtual void operator++() = 0;
    virtual const Provider_Node& operator*() = 0;
  };

  struct NodeConstIterator {
    NodeConstIterator(std::unique_ptr<Provider_NodeIterator> p) : impl_{std::move(p)} {}

    bool operator==(const NodeConstIterator& p_other) const;
    bool operator!=(const NodeConstIterator& p_other) const {
      return *impl_ != *p_other.impl_;
    }

    void operator++() {
      impl_->operator++();
    }

    const Provider_Node& operator*() const {
      return impl_->operator*();
    }
    const Provider_Node* operator->() const;

    std::unique_ptr<Provider_NodeIterator> impl_;
  };

  NodeConstIterator InputNodesBegin() const noexcept { return NodeConstIterator(InputNodesBegin_internal()); }
  NodeConstIterator InputNodesEnd() const noexcept { return NodeConstIterator(InputNodesEnd_internal()); }

  virtual std::unique_ptr<Provider_NodeIterator> InputNodesBegin_internal() const noexcept = 0;
  virtual std::unique_ptr<Provider_NodeIterator> InputNodesEnd_internal() const noexcept = 0;

  struct Provider_EdgeIterator {
    virtual ~Provider_EdgeIterator() {}
    virtual bool operator!=(const Provider_EdgeIterator& p) const = 0;

    virtual void operator++() = 0;
    virtual const Provider_Node& GetNode() const = 0;
    virtual int GetSrcArgIndex() const = 0;
    virtual int GetDstArgIndex() const = 0;
  };

  struct EdgeConstIterator {
    EdgeConstIterator(std::unique_ptr<Provider_EdgeIterator> p) : impl_{std::move(p)} {}

    bool operator!=(const EdgeConstIterator& p_other) const {
      return *impl_ != *p_other.impl_;
    }

    void operator++() { impl_->operator++(); }

    const Provider_EdgeIterator* operator->() const { return impl_.get(); }

    std::unique_ptr<Provider_EdgeIterator> impl_;
  };

  EdgeConstIterator OutputEdgesBegin() const noexcept { return EdgeConstIterator(OutputEdgesBegin_internal()); }
  EdgeConstIterator OutputEdgesEnd() const noexcept { return EdgeConstIterator(OutputEdgesEnd_internal()); }

  virtual std::unique_ptr<Provider_EdgeIterator> OutputEdgesBegin_internal() const noexcept = 0;
  virtual std::unique_ptr<Provider_EdgeIterator> OutputEdgesEnd_internal() const noexcept = 0;
};
#endif

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

#if 0
struct Provider_Model {
  virtual ~Provider_Model() {}

  virtual Provider_Graph& MainGraph() = 0;

  virtual std::unique_ptr<ONNX_NAMESPACE::Provider_ModelProto> CreateModelProto() const = 0;
};
#endif

#if 0
struct Provider_IndexedSubGraph {
  static std::unique_ptr<Provider_IndexedSubGraph> Create();
  virtual ~Provider_IndexedSubGraph() = default;

  struct MetaDef {
    std::string name;    ///< Name of customized SubGraph/FunctionProto
    std::string domain;  ///< Domain of customized SubGraph/FunctionProto
    int since_version;   ///< Since version of customized SubGraph/FunctionProto.

    ONNX_NAMESPACE::OperatorStatus status;  ///< Status of customized SubGraph/FunctionProto.

    std::vector<std::string> inputs;     ///< Inputs of customized SubGraph/FunctionProto.
    std::vector<std::string> outputs;    ///< Outputs of customized SubGraph/FunctionProto.
    Provider_NodeAttributes attributes;  ///< Attributes of customized SubGraph/FunctionProto.

    std::string doc_string;  ///< Doc string of customized SubGraph/FunctionProto.
  };

  /** Nodes covered by this subgraph. The NodeIndex values are from the parent Graph.*/
  virtual std::vector<onnxruntime::NodeIndex>& Nodes() = 0;

  virtual void SetMetaDef(std::unique_ptr<MetaDef>& meta_def_) = 0;
  virtual const MetaDef* GetMetaDef() = 0;

  void operator=(const Provider_IndexedSubGraph& v) = delete;
};

struct Provider_KernelRegistry {
  static std::shared_ptr<Provider_KernelRegistry> Create();

  virtual ~Provider_KernelRegistry() = default;
  virtual Status Register(Provider_KernelCreateInfo&& create_info) = 0;

  void operator=(const Provider_KernelRegistry& v) = delete;
};

struct Provider_ComputeCapability {
  Provider_ComputeCapability(std::unique_ptr<Provider_IndexedSubGraph> t_sub_graph) : t_sub_graph_{std::move(t_sub_graph)} {}

  std::unique_ptr<Provider_IndexedSubGraph> t_sub_graph_;

  void operator=(const Provider_ComputeCapability& v) = delete;
};
#endif

struct Provider_IDataTransfer {
  virtual ~Provider_IDataTransfer() {}
};

namespace logging {
class Logger;
}

// Provides the base class implementations, since Provider_IExecutionProvider is just an interface. This is to fake the C++ inheritance used by internal IExecutionProvider implementations
struct Provider_IExecutionProvider_Router {
  virtual ~Provider_IExecutionProvider_Router() {}

  virtual std::shared_ptr<Provider_KernelRegistry> Provider_GetKernelRegistry() const = 0;

  virtual std::vector<std::unique_ptr<Provider_ComputeCapability>> Provider_GetCapability(const onnxruntime::Provider_GraphViewer& graph,
                                                                                          const std::vector<const Provider_KernelRegistry*>& kernel_registries) const = 0;

  virtual Provider_AllocatorPtr Provider_GetAllocator(int id, OrtMemType mem_type) const = 0;
  virtual void Provider_InsertAllocator(Provider_AllocatorPtr allocator) = 0;
  virtual const logging::Logger* GetLogger() const = 0;

  void operator=(const Provider_IExecutionProvider_Router& v) = delete;
};

struct Provider_IExecutionProvider {
  Provider_IExecutionProvider(const std::string& type);
  virtual ~Provider_IExecutionProvider() {}

  virtual std::shared_ptr<Provider_KernelRegistry> Provider_GetKernelRegistry() const { return p_->Provider_GetKernelRegistry(); }

  virtual std::unique_ptr<Provider_IDataTransfer> GetDataTransfer() const { return nullptr; }

  virtual std::vector<std::unique_ptr<Provider_ComputeCapability>> Provider_GetCapability(const onnxruntime::Provider_GraphViewer& graph,
                                                                                          const std::vector<const Provider_KernelRegistry*>& kernel_registries) const { return p_->Provider_GetCapability(graph, kernel_registries); }

  virtual common::Status Provider_Compile(const std::vector<Provider_Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) = 0;

  virtual Provider_AllocatorPtr Provider_GetAllocator(int id, OrtMemType mem_type) const { return p_->Provider_GetAllocator(id, mem_type); }
  virtual void Provider_InsertAllocator(Provider_AllocatorPtr allocator) { return p_->Provider_InsertAllocator(allocator); }

  virtual const logging::Logger* GetLogger() const { return p_->GetLogger(); }

  Provider_IExecutionProvider_Router* p_;

  void operator=(const Provider_IExecutionProvider& v) = delete;
};

struct Provider {
  virtual std::shared_ptr<Provider_IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) = 0;
  virtual void SetProviderHost(ProviderHost& host) = 0;
};

// There are two ways to route a function, one is a virtual method and the other is a function pointer (or pointer to member function)
// The function pointers are nicer in that they directly call the target function, but they cannot be used in cases where we're calling
// a specific implementation of a virtual class member. Trying to get a pointer to member of a virtual function will return a thunk that
// calls the virtual function (which will lead to infinite recursion in the bridge). There is no known way to get the non virtual member
// function pointer implementation in this case.
struct ProviderHost {
  virtual Provider_AllocatorPtr CreateAllocator(Provider_DeviceAllocatorRegistrationInfo& info, int16_t device_id = 0) = 0;

  virtual logging::Logger* LoggingManager_GetDefaultLogger() = 0;

  virtual std::unique_ptr<ONNX_NAMESPACE::Provider_AttributeProto> AttributeProto_Create() = 0;

  virtual std::unique_ptr<Provider_OrtMemoryInfo> OrtMemoryInfo_Create(const char* name_, OrtAllocatorType type_, Provider_OrtDevice* device_, int id_, OrtMemType mem_type_) = 0;
#if 0
  virtual std::unique_ptr<Provider_KernelDefBuilder> KernelDefBuilder_Create() = 0;
  virtual std::shared_ptr<Provider_KernelRegistry> KernelRegistry_Create() = 0;
  virtual std::unique_ptr<Provider_IndexedSubGraph> IndexedSubGraph_Create() = 0;
#endif

  virtual std::unique_ptr<Provider_IDeviceAllocator> CreateCPUAllocator(std::unique_ptr<Provider_OrtMemoryInfo> memory_info) = 0;
#if 0
  virtual std::unique_ptr<Provider_IDeviceAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<Provider_IDeviceAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<Provider_IDataTransfer> CreateGPUDataTransfer() = 0;
#endif

  virtual Provider_AllocatorPtr CreateDummyArenaAllocator(std::unique_ptr<Provider_IDeviceAllocator> resource_allocator) = 0;
  virtual std::unique_ptr<Provider_IExecutionProvider_Router> Create_IExecutionProvider_Router(Provider_IExecutionProvider* outer, const std::string& type) = 0;

  virtual std::string GetEnvironmentVar(const std::string& var_name) = 0;

  MLDataType (*DataTypeImpl_GetType_Tensor)();
  MLDataType (*DataTypeImpl_GetType_float)();
  MLDataType (*DataTypeImpl_GetTensorType_float)();
  virtual const std::vector<MLDataType>& DataTypeImpl_AllFixedSizeTensorTypes() = 0;

  virtual void* HeapAllocate(size_t size) = 0;
  virtual void HeapFree(void*) = 0;

  virtual void LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file, const char* function, uint32_t line) = 0;

  virtual bool CPU_HasAVX2() = 0;
  virtual bool CPU_HasAVX512f() = 0;

  // Provider_TypeProto_Tensor
  virtual int32_t Provider_TypeProto_Tensor__elem_type(const Provider_TypeProto_Tensor* p) = 0;

  // Provider_TypeProto
  virtual const Provider_TypeProto_Tensor& Provider_TypeProto__tensor_type(const Provider_TypeProto* p) = 0;

  // Provider_GraphProto
  virtual void Provider_GraphProto_destructor(Provider_GraphProto* p) = 0;

  virtual Provider_ValueInfoProtos* Provider_GraphProto__mutable_input(Provider_GraphProto* p) = 0;

  virtual const Provider_ValueInfoProtos& Provider_GraphProto__output(const Provider_GraphProto* p) = 0;
  virtual Provider_ValueInfoProtos* Provider_GraphProto__mutable_output(Provider_GraphProto* p) = 0;

  virtual Provider_ValueInfoProtos* Provider_GraphProto__mutable_value_info(Provider_GraphProto* p) = 0;
  virtual Provider_TensorProtos* Provider_GraphProto__mutable_initializer(Provider_GraphProto* p) = 0;
  virtual Provider_NodeProto* Provider_GraphProto__add_node(Provider_GraphProto* p) = 0;

  virtual void Provider_GraphProto__operator_assign(Provider_GraphProto* p, const Provider_GraphProto& v) = 0;

  // Provider_ModelProto
  virtual void Provider_ModelProto__destructor(Provider_ModelProto* p) = 0;

  virtual bool Provider_ModelProto__SerializeToString(const Provider_ModelProto* p, std::string& string) = 0;
  virtual bool Provider_ModelProto__SerializeToOstream(const Provider_ModelProto* p, std::ostream& output) = 0;

  virtual const Provider_GraphProto& Provider_ModelProto__graph(const Provider_ModelProto* p) = 0;
  virtual Provider_GraphProto* Provider_ModelProto__mutable_graph(Provider_ModelProto* p) = 0;

  virtual void Provider_ModelProto__set_ir_version(Provider_ModelProto* p, int64_t value) = 0;

  // Provider_TensorProto
  virtual void Provider_TensorProto__destructor(Provider_TensorProto* p) = 0;
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
  virtual const Provider_TypeProto& Provider_ValueInfoProto__type(const Provider_ValueInfoProto* p) = 0;
  virtual void Provider_ValueInfoProto__operator_assign(Provider_ValueInfoProto* p, const Provider_ValueInfoProto& v) = 0;

  // Provider_ValueInfoProtos
  virtual Provider_ValueInfoProto* Provider_ValueInfoProtos__Add(Provider_ValueInfoProtos* p) = 0;

  virtual const Provider_ValueInfoProto& Provider_ValueInfoProtos__operator_array(const Provider_ValueInfoProtos* p, int index) = 0;

  // Provider_ComputeCapability
  virtual std::unique_ptr<Provider_ComputeCapability> Provider_ComputeCapability__construct(std::unique_ptr<Provider_IndexedSubGraph> t_sub_graph) = 0;
  virtual std::unique_ptr<Provider_IndexedSubGraph>& Provider_ComputeCapability__SubGraph(Provider_ComputeCapability* p) = 0;

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
  virtual const ONNX_NAMESPACE::Provider_TypeProto* Provider_NodeArg__TypeAsProto(const Provider_NodeArg* p) noexcept = 0;

  // Provider_Model
  virtual void Provider_Model__destructor(Provider_Model* p) = 0;
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

  // Provider_GraphViewer
  virtual void Provider_GraphViewer__destructor(Provider_GraphViewer* p) = 0;
  virtual std::unique_ptr<Provider_Model> Provider_GraphViewer__CreateModel(const Provider_GraphViewer* p, const logging::Logger& logger) = 0;

  virtual const std::string& Provider_GraphViewer__Name(const Provider_GraphViewer* p) noexcept = 0;

  virtual const Provider_Node* Provider_GraphViewer__GetNode(const Provider_GraphViewer* p, NodeIndex node_index) = 0;
  virtual const Provider_NodeArg* Provider_GraphViewer__GetNodeArg(const Provider_GraphViewer* p, const std::string& name) = 0;

  virtual bool Provider_GraphViewer__IsSubgraph(const Provider_GraphViewer* p) = 0;
  virtual int Provider_GraphViewer__MaxNodeIndex(const Provider_GraphViewer* p) noexcept = 0;

  virtual const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetInputs(const Provider_GraphViewer* p) noexcept = 0;
  virtual const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetOutputs(const Provider_GraphViewer* p) noexcept = 0;
  virtual const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetValueInfo(const Provider_GraphViewer* p) noexcept = 0;

  virtual const Provider_InitializedTensorSet& Provider_GraphViewer__GetAllInitializedTensors(const Provider_GraphViewer* p) = 0;
  virtual const std::unordered_map<std::string, int>& Provider_GraphViewer__DomainToVersionMap(const Provider_GraphViewer* p) = 0;

  virtual const std::vector<NodeIndex>& Provider_GraphViewer__GetNodesInTopologicalOrder(const Provider_GraphViewer* p) = 0;
};

extern ProviderHost* g_host;

struct Provider_TypeProto_Tensor {
  int32_t elem_type() const { return g_host->Provider_TypeProto_Tensor__elem_type(this); }

  void operator=(const Provider_TypeProto_Tensor& v) = delete;
};

struct Provider_TypeProto {
  const Provider_TypeProto_Tensor& tensor_type() const { return g_host->Provider_TypeProto__tensor_type(this); }

  void operator=(const Provider_TypeProto& v) = delete;
};

struct Provider_GraphProto {
  static void operator delete(void* p) { g_host->Provider_GraphProto_destructor(reinterpret_cast<Provider_GraphProto*>(p)); }

  Provider_ValueInfoProtos* mutable_input() { return g_host->Provider_GraphProto__mutable_input(this); }

  const Provider_ValueInfoProtos& output() const { return g_host->Provider_GraphProto__output(this); }
  Provider_ValueInfoProtos* mutable_output() { return g_host->Provider_GraphProto__mutable_output(this); }

  Provider_ValueInfoProtos* mutable_value_info() { return g_host->Provider_GraphProto__mutable_value_info(this); }
  Provider_TensorProtos* mutable_initializer() { return g_host->Provider_GraphProto__mutable_initializer(this); }
  Provider_NodeProto* add_node() { return g_host->Provider_GraphProto__add_node(this); }

  void operator=(const Provider_GraphProto& v) { return g_host->Provider_GraphProto__operator_assign(this, v); }
};

struct Provider_ModelProto {
  static void operator delete(void* p) { g_host->Provider_ModelProto__destructor(reinterpret_cast<Provider_ModelProto*>(p)); }

  bool SerializeToString(std::string& string) const { return g_host->Provider_ModelProto__SerializeToString(this, string); }
  bool SerializeToOstream(std::ostream& output) const { return g_host->Provider_ModelProto__SerializeToOstream(this, output); }

  const ONNX_NAMESPACE::Provider_GraphProto& graph() const { return g_host->Provider_ModelProto__graph(this); }
  ONNX_NAMESPACE::Provider_GraphProto* mutable_graph() { return g_host->Provider_ModelProto__mutable_graph(this); }

  void set_ir_version(int64_t value) { return g_host->Provider_ModelProto__set_ir_version(this, value); }

  void operator=(const Provider_ModelProto& v) = delete;
};

struct Provider_TensorProto {
  static void operator delete(void* p) { g_host->Provider_TensorProto__destructor(reinterpret_cast<Provider_TensorProto*>(p)); }
  void operator=(const Provider_TensorProto& v) { g_host->Provider_TensorProto__operator_assign(this, v); }
};

struct Provider_TensorProtos {
  Provider_TensorProto* Add() { return g_host->Provider_TensorProtos__Add(this); }

  void operator=(const Provider_TensorProtos& v) = delete;
};

struct Provider_TensorShapeProto_Dimension {
  const std::string& dim_param() const { return g_host->Provider_TensorShapeProto_Dimension__dim_param(this); }

  void operator=(const Provider_TensorShapeProto_Dimension& v) = delete;
};

struct Provider_TensorShapeProto_Dimensions {
  IteratorHolder<Provider_TensorShapeProto_Dimension_Iterator, const Provider_TensorShapeProto_Dimension> begin() const { return g_host->Provider_TensorShapeProto_Dimensions__begin(this); }
  IteratorHolder<Provider_TensorShapeProto_Dimension_Iterator, const Provider_TensorShapeProto_Dimension> end() const { return g_host->Provider_TensorShapeProto_Dimensions__end(this); }
};

struct Provider_TensorShapeProto {
  int dim_size() const { return g_host->Provider_TensorShapeProto__dim_size(this); }
  const Provider_TensorShapeProto_Dimensions& dim() const { return g_host->Provider_TensorShapeProto__dim(this); }

  void operator=(const Provider_TensorShapeProto& v) = delete;
};

struct Provider_ValueInfoProto {
  const Provider_TypeProto& type() const { return g_host->Provider_ValueInfoProto__type(this); }
  void operator=(const Provider_ValueInfoProto& v) { g_host->Provider_ValueInfoProto__operator_assign(this, v); }
};

struct Provider_ValueInfoProtos {
  Provider_ValueInfoProto* Add() { return g_host->Provider_ValueInfoProtos__Add(this); }
  const Provider_ValueInfoProto& operator[](int index) const { return g_host->Provider_ValueInfoProtos__operator_array(this, index); }

  void operator=(const Provider_ValueInfoProtos& v) = delete;
};

struct Provider_ComputeCapability {
  static std::unique_ptr<Provider_ComputeCapability> Create(std::unique_ptr<Provider_IndexedSubGraph> t_sub_graph) { return g_host->Provider_ComputeCapability__construct(std::move(t_sub_graph)); }

  std::unique_ptr<Provider_IndexedSubGraph>& SubGraph() { return g_host->Provider_ComputeCapability__SubGraph(this); }
};

struct Provider_IndexedSubGraph {
  static std::unique_ptr<Provider_IndexedSubGraph> Create() { return g_host->Provider_IndexedSubGraph__construct(); }
  static void operator delete(void* p) { g_host->Provider_IndexedSubGraph__operator_delete(reinterpret_cast<Provider_IndexedSubGraph*>(p)); }

  struct MetaDef {
    std::string name;    ///< Name of customized SubGraph/FunctionProto
    std::string domain;  ///< Domain of customized SubGraph/FunctionProto
    int since_version;   ///< Since version of customized SubGraph/FunctionProto.

    ONNX_NAMESPACE::OperatorStatus status;  ///< Status of customized SubGraph/FunctionProto.

    std::vector<std::string> inputs;     ///< Inputs of customized SubGraph/FunctionProto.
    std::vector<std::string> outputs;    ///< Outputs of customized SubGraph/FunctionProto.
    Provider_NodeAttributes attributes;  ///< Attributes of customized SubGraph/FunctionProto.

    std::string doc_string;  ///< Doc string of customized SubGraph/FunctionProto.
  };

  /** Nodes covered by this subgraph. The NodeIndex values are from the parent Graph.*/
  std::vector<onnxruntime::NodeIndex>& Nodes() { return g_host->Provider_IndexedSubGraph__Nodes(this); }

  void SetMetaDef(std::unique_ptr<MetaDef>&& meta_def_) { return g_host->Provider_IndexedSubGraph__SetMetaDef(this, std::move(*reinterpret_cast<std::unique_ptr<Provider_IndexedSubGraph_MetaDef>*>(&meta_def_))); }
  const MetaDef* GetMetaDef() const { return reinterpret_cast<const MetaDef*>(g_host->Provider_IndexedSubGraph__GetMetaDef(this)); }

  Provider_IndexedSubGraph() = delete;
  void operator=(const Provider_IndexedSubGraph& v) = delete;
};

struct Provider_KernelDef {
  static void operator delete(void* p) { g_host->Provider_KernelDef__operator_delete(reinterpret_cast<Provider_KernelDef*>(p)); }

  Provider_KernelDef() = delete;
  void operator=(const Provider_KernelDef& v) = delete;
};

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
};

struct Provider_KernelRegistry {
  static std::shared_ptr<Provider_KernelRegistry> Create() { return g_host->Provider_KernelRegistry__construct(); }
  static void operator delete(void* p) { g_host->Provider_KernelRegistry__operator_delete(reinterpret_cast<Provider_KernelRegistry*>(p)); }

  Status Register(Provider_KernelCreateInfo&& create_info) { return g_host->Provider_KernelRegistry__Register(this, std::move(create_info)); }

  void operator=(const Provider_KernelRegistry& v) = delete;
};

struct Provider_Function {
  const Provider_Graph& Body() const { return g_host->Provider_Function__Body(this); }

  void operator=(const Provider_Function& v) = delete;
};

struct Provider_Node {
  const std::string& Name() const noexcept { return g_host->Provider_Node__Name(this); }
  const std::string& Description() const noexcept { return g_host->Provider_Node__Description(this); }
  const std::string& Domain() const noexcept { return g_host->Provider_Node__Domain(this); }
  const std::string& OpType() const noexcept { return g_host->Provider_Node__OpType(this); }

  const Provider_Function* GetFunctionBody() const noexcept { return g_host->Provider_Node__GetFunctionBody(this); }

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

  void operator=(const Provider_Node& v) = delete;
};

struct Provider_NodeArg {
  const std::string& Name() const noexcept { return g_host->Provider_NodeArg__Name(this); }
  const Provider_TensorShapeProto* Shape() const { return g_host->Provider_NodeArg__Shape(this); }
  ONNX_NAMESPACE::DataType Type() const noexcept { return g_host->Provider_NodeArg__Type(this); }
  const Provider_NodeArgInfo& ToProto() const noexcept { return g_host->Provider_NodeArg__ToProto(this); }
  bool Exists() const noexcept { return g_host->Provider_NodeArg__Exists(this); }
  const Provider_TypeProto* TypeAsProto() const noexcept { return g_host->Provider_NodeArg__TypeAsProto(this); }

  void operator=(const Provider_NodeArg& v) = delete;
};

struct Provider_Model {
  static void operator delete(void* p) { g_host->Provider_Model__destructor(reinterpret_cast<Provider_Model*>(p)); }

  Provider_Graph& MainGraph() { return g_host->Provider_Model__MainGraph(this); }

  std::unique_ptr<Provider_ModelProto> ToProto() { return g_host->Provider_Model__ToProto(this); }

  void operator=(const Provider_Model& v) = delete;
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

  void operator=(const Provider_Graph& v) = delete;
};

struct Provider_GraphViewer {
  static void operator delete(void* p) { g_host->Provider_GraphViewer__destructor(reinterpret_cast<Provider_GraphViewer*>(p)); }

  std::unique_ptr<Provider_Model> CreateModel(const logging::Logger& logger) const { return g_host->Provider_GraphViewer__CreateModel(this, logger); }

  const std::string& Name() const noexcept { return g_host->Provider_GraphViewer__Name(this); }

  const Provider_Node* GetNode(NodeIndex node_index) const { return g_host->Provider_GraphViewer__GetNode(this, node_index); }
  const Provider_NodeArg* GetNodeArg(const std::string& name) const { return g_host->Provider_GraphViewer__GetNodeArg(this, name); }

  bool IsSubgraph() const { return g_host->Provider_GraphViewer__IsSubgraph(this); }

  int MaxNodeIndex() const noexcept { return g_host->Provider_GraphViewer__MaxNodeIndex(this); }

  const std::vector<const Provider_NodeArg*>& GetInputs() const noexcept { return g_host->Provider_GraphViewer__GetInputs(this); }
  const std::vector<const Provider_NodeArg*>& GetOutputs() const noexcept { return g_host->Provider_GraphViewer__GetOutputs(this); }
  const std::vector<const Provider_NodeArg*>& GetValueInfo() const noexcept { return g_host->Provider_GraphViewer__GetValueInfo(this); }

  const Provider_InitializedTensorSet& GetAllInitializedTensors() const noexcept { return g_host->Provider_GraphViewer__GetAllInitializedTensors(this); }

  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept { return g_host->Provider_GraphViewer__DomainToVersionMap(this); }

  const std::vector<NodeIndex>& GetNodesInTopologicalOrder() const { return g_host->Provider_GraphViewer__GetNodesInTopologicalOrder(this); }

  void operator=(const Provider_GraphViewer& v) = delete;
};

}  // namespace onnxruntime
