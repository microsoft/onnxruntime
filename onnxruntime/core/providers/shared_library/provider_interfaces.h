// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Public wrappers around internal ort interfaces (currently)
// In the future the internal implementations could derive from these to remove the need for the wrapper implementations

#include "core/framework/func_api.h"

namespace ONNX_NAMESPACE {
enum AttributeProto_AttributeType : int;
enum OperatorStatus : int;

// String pointer as unique TypeProto identifier.
using DataType = const std::string*;

struct Provider_TensorProto {
  virtual ~Provider_TensorProto() = default;

  virtual void CopyFrom(const Provider_TensorProto& v) = 0;

  void operator=(const Provider_TensorProto& v) { CopyFrom(v); }
};

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

struct Provider_TensorShapeProto {
  int dim_size() const { return dim_size_; }

  int dim_size_{};
};

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

struct ProviderHost;
struct Provider_IExecutionProvider;

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

// This can probably be deleted
struct Provider_IDeviceAllocator : Provider_IAllocator {
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

struct Provider_OpKernelInfo {
  virtual Status GetAttr(const std::string& name, int64_t* value) const = 0;
  virtual Status GetAttr(const std::string& name, float* value) const = 0;

  template <typename T>
  Status GetAttr(const std::string& name, T* value) const;
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

  virtual std::unique_ptr<Provider_KernelDef> Build() = 0;

  void operator=(const Provider_KernelDefBuilder& v) = delete;
};

using NodeIndex = size_t;
using Provider_NodeAttributes = std::unordered_map<std::string, ONNX_NAMESPACE::Provider_AttributeProto_Copyable>;

using Provider_InitializedTensorSet = std::unordered_map<std::string, const ONNX_NAMESPACE::Provider_TensorProto*>;

struct Provider_NodeArg {
  virtual ~Provider_NodeArg() = default;
  virtual const std::string& Name() const noexcept = 0;
  virtual const ONNX_NAMESPACE::Provider_TensorShapeProto* Shape() const = 0;
  virtual ONNX_NAMESPACE::DataType Type() const noexcept = 0;

  void operator=(const Provider_NodeArg& v) = delete;
};

struct Provider_Node {
  virtual ~Provider_Node() = default;

  virtual const std::string& OpType() const noexcept = 0;

  virtual ConstPointerContainer<std::vector<Provider_NodeArg*>> InputDefs() const noexcept = 0;
  virtual ConstPointerContainer<std::vector<Provider_NodeArg*>> OutputDefs() const noexcept = 0;
  virtual NodeIndex Index() const noexcept = 0;

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
    void operator--();

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

struct Provider_GraphViewer {
  virtual ~Provider_GraphViewer() = default;
  virtual const std::string& Name() const noexcept = 0;

  virtual const Provider_Node* GetNode(NodeIndex node_index) const = 0;

  virtual int MaxNodeIndex() const noexcept = 0;

  virtual const Provider_InitializedTensorSet& GetAllInitializedTensors() const noexcept = 0;

  virtual const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept = 0;

  void operator=(const Provider_GraphViewer& v) = delete;
};

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

// Provides the base class implementations, since Provider_IExecutionProvider is just an interface. This is to fake the C++ inheritance used by internal IExecutionProvider implementations
struct Provider_IExecutionProvider_Router {
  virtual ~Provider_IExecutionProvider_Router() {}

  virtual std::shared_ptr<Provider_KernelRegistry> Provider_GetKernelRegistry() const = 0;

  virtual std::vector<std::unique_ptr<Provider_ComputeCapability>> Provider_GetCapability(const onnxruntime::Provider_GraphViewer& graph,
                                                                                          const std::vector<const Provider_KernelRegistry*>& kernel_registries) const = 0;

  virtual Provider_AllocatorPtr Provider_GetAllocator(int id, OrtMemType mem_type) const = 0;
  virtual void Provider_InsertAllocator(Provider_AllocatorPtr allocator) = 0;

  void operator=(const Provider_IExecutionProvider_Router& v) = delete;
};

struct Provider_IExecutionProvider {
  Provider_IExecutionProvider(const std::string& type);
  virtual ~Provider_IExecutionProvider() {}

  virtual std::shared_ptr<Provider_KernelRegistry> Provider_GetKernelRegistry() const { return p_->Provider_GetKernelRegistry(); }

  virtual std::vector<std::unique_ptr<Provider_ComputeCapability>> Provider_GetCapability(const onnxruntime::Provider_GraphViewer& graph,
                                                                                          const std::vector<const Provider_KernelRegistry*>& kernel_registries) const { return p_->Provider_GetCapability(graph, kernel_registries); }

  virtual common::Status Provider_Compile(const std::vector<Provider_Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) = 0;

  virtual Provider_AllocatorPtr Provider_GetAllocator(int id, OrtMemType mem_type) const { return p_->Provider_GetAllocator(id, mem_type); }
  virtual void Provider_InsertAllocator(Provider_AllocatorPtr allocator) { return p_->Provider_InsertAllocator(allocator); }

  Provider_IExecutionProvider_Router* p_;

  void operator=(const Provider_IExecutionProvider& v) = delete;
};

namespace logging {
class Logger;
}

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
  virtual Provider_AllocatorPtr CreateAllocator(const Provider_DeviceAllocatorRegistrationInfo& info,
                                                int16_t device_id = 0, bool use_arena = true) = 0;

  virtual logging::Logger* LoggingManager_GetDefaultLogger() = 0;

  virtual std::unique_ptr<ONNX_NAMESPACE::Provider_AttributeProto> AttributeProto_Create() = 0;

  virtual std::unique_ptr<Provider_OrtMemoryInfo> OrtMemoryInfo_Create(
      const char* name_, OrtAllocatorType type_, Provider_OrtDevice* device_, int id_, OrtMemType mem_type_) = 0;

  virtual std::unique_ptr<Provider_KernelDefBuilder> KernelDefBuilder_Create() = 0;

  virtual std::shared_ptr<Provider_KernelRegistry> KernelRegistry_Create() = 0;

  virtual std::unique_ptr<Provider_IndexedSubGraph> IndexedSubGraph_Create() = 0;

  virtual std::unique_ptr<Provider_IDeviceAllocator> CreateCPUAllocator(
      std::unique_ptr<Provider_OrtMemoryInfo> memory_info) = 0;

  virtual std::unique_ptr<Provider_IExecutionProvider_Router> Create_IExecutionProvider_Router(
      Provider_IExecutionProvider* outer, const std::string& type) = 0;

  MLDataType (*DataTypeImpl_GetType_Tensor)();
  MLDataType (*DataTypeImpl_GetType_float)();
  MLDataType (*DataTypeImpl_GetTensorType_float)();

  virtual void* HeapAllocate(size_t size) = 0;
  virtual void HeapFree(void*) = 0;

  virtual void LogRuntimeError(uint32_t session_id, const common::Status& status,
                               const char* file, const char* function, uint32_t line) = 0;

  virtual bool CPU_HasAVX2() = 0;
  virtual bool CPU_HasAVX512f() = 0;
};

}  // namespace onnxruntime
