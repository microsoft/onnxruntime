// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Public wrappers around internal ort interfaces (currently)
// In the future the internal implementations could derive from these to remove the need for the wrapper implementations

#include "core/framework/func_api.h"

#if _MSC_VER
#define PROVIDER_NOT_IMPLEMENTED __debugbreak();
#else
#define PROVIDER_NOT_IMPLEMENTED raise(SIGTRAP);
#endif

namespace ONNX_NAMESPACE {
enum AttributeProto_AttributeType;
enum OperatorStatus;

// String pointer as unique TypeProto identifier.
using DataType = const std::string*;

struct Prov_TensorProto {
  virtual ~Prov_TensorProto() = default;

  virtual void CopyFrom(const Prov_TensorProto& v) = 0;

  void operator=(const Prov_TensorProto& v) { CopyFrom(v); }
};

struct Prov_AttributeProto {
  static std::unique_ptr<Prov_AttributeProto> Create();

  virtual ~Prov_AttributeProto() = default;
  virtual std::unique_ptr<Prov_AttributeProto> Clone() const = 0;

  virtual ::onnx::AttributeProto_AttributeType type() const = 0;
  virtual int ints_size() const = 0;
  virtual int64_t ints(int i) const = 0;
  virtual int64_t i() const = 0;
  virtual float f() const = 0;
  virtual void set_s(const ::std::string& value) = 0;
  virtual const ::std::string& s() const = 0;
  virtual void set_name(const ::std::string& value) = 0;
  virtual void set_type(::onnx::AttributeProto_AttributeType value) = 0;
  virtual ::onnx::Prov_TensorProto* add_tensors() = 0;

  void operator=(const Prov_AttributeProto& v) = delete;
};

// This is needed since Prov_NodeAttributes is a map of unique_ptr to Prov_AttributeProto and that won't work since unique_ptrs are not copyable
// (supposedly this should work in the latest C++ STL but it didn't for me so I used this to make it copyable)
struct Prov_AttributeProto_Copyable {
  Prov_AttributeProto_Copyable() = default;
  Prov_AttributeProto_Copyable(const Prov_AttributeProto_Copyable& copy) : p_{copy->Clone()} {}

  void operator=(std::unique_ptr<Prov_AttributeProto>&& p) { p_ = std::move(p); }
  void operator=(const Prov_AttributeProto_Copyable& p) { p_ = p->Clone(); }

  Prov_AttributeProto& operator*() const { return *p_.get(); }
  Prov_AttributeProto* operator->() const { return p_.get(); }

  std::unique_ptr<Prov_AttributeProto> p_;
};

struct Prov_TensorShapeProto {
  int dim_size() const { return dim_size_; }

  int dim_size_;
};

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

struct ProviderHost;
struct Prov_IExecutionProvider;

struct Prov_IExecutionProviderFactory {
  virtual ~Prov_IExecutionProviderFactory() = default;
  virtual std::unique_ptr<Prov_IExecutionProvider> CreateProvider() = 0;
};

//struct KernelCreateInfo;

class DataTypeImpl;
using MLDataType = const DataTypeImpl*;

struct Prov_OrtDevice {
  virtual ~Prov_OrtDevice() {}
};

struct Prov_OrtMemoryInfo {
  static std::unique_ptr<Prov_OrtMemoryInfo> Create(const char* name_, OrtAllocatorType type_, Prov_OrtDevice* device_ = nullptr, int id_ = 0, OrtMemType mem_type_ = OrtMemTypeDefault);
  virtual ~Prov_OrtMemoryInfo() {}

  void operator=(const Prov_OrtMemoryInfo& v) = delete;
};

template <typename T>
using Prov_IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

struct Prov_IAllocator {
  virtual ~Prov_IAllocator() {}

  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p) = 0;
  virtual const Prov_OrtMemoryInfo& Info() const = 0;

  static bool CalcMemSizeForArray(size_t nmemb, size_t size, size_t* out) noexcept {
    return CalcMemSizeForArrayWithAlignment<0>(nmemb, size, out);
  }

  template <size_t alignment>
  static bool CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t* out) noexcept ORT_MUST_USE_RESULT;

  template <typename T>
  static Prov_IAllocatorUniquePtr<T> MakeUniquePtr(std::shared_ptr<Prov_IAllocator> allocator, size_t count_or_bytes) {
    if (allocator == nullptr) return nullptr;
    // for now limit to fundamental types. we could support others, but to do so either we or the caller
    // needs to call the dtor for the objects, for buffers allocated on device we don't have destructor
    //static_assert(std::is_fundamental<T>::value, "Fundamental type required as no destructors are called.");

    size_t alloc_size = count_or_bytes;

    // if T is not void, 'count_or_bytes' == number of items so allow for that
    if (!std::is_void<T>::value) {
      // sizeof(void) isn't valid, but the compiler isn't smart enough to ignore that this line isn't
      // reachable if T is void. use std::conditional to 'use' void* in the sizeof call
      if (!CalcMemSizeForArray(count_or_bytes, sizeof(typename std::conditional<std::is_void<T>::value, void*, T>::type),
                               &alloc_size)) return nullptr;
    }
    return IAllocatorUniquePtr<T>{
        static_cast<T*>(allocator->Alloc(alloc_size)),  // allocate
        [=](T* ptr) { allocator->Free(ptr); }};         // capture IAllocator so it's always valid, and use as deleter
  }

  void operator=(const Prov_IAllocator& v) = delete;
};

template <size_t alignment>
bool Prov_IAllocator::CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t* out) noexcept {
  static constexpr size_t max_allowed = (static_cast<size_t>(1) << (static_cast<size_t>(std::numeric_limits<size_t>::digits >> 1))) - alignment;
  static constexpr size_t max_size = std::numeric_limits<size_t>::max() - alignment;
  static constexpr size_t alignment_mask = alignment - 1;
  //Indeed, we only need to check if max_size / nmemb < size
  //max_allowed is for avoiding unnecessary DIV.
  if (nmemb >= max_allowed && max_size / nmemb < size) {
    return false;
  }
  if (size >= max_allowed &&
      nmemb > 0 && max_size / nmemb < size) {
    return false;
  }
  if (alignment == 0)
    *out = size * nmemb;
  else
    *out = (size * nmemb + alignment_mask) & ~static_cast<size_t>(alignment_mask);
  return true;
}

struct Prov_IDeviceAllocator : Prov_IAllocator {
  virtual bool AllowsArena() const = 0;
};

using Prov_AllocatorPtr = std::shared_ptr<Prov_IAllocator>;
using Prov_DeviceAllocatorFactory = std::function<std::unique_ptr<Prov_IDeviceAllocator>(int)>;

struct Prov_DeviceAllocatorRegistrationInfo {
  OrtMemType mem_type;
  Prov_DeviceAllocatorFactory factory;
  size_t max_mem;
};

class OpKernel;      // TODO
class OpKernelInfo;  // TODO

struct Prov_KernelDef {
  virtual ~Prov_KernelDef() {}
};

using Prov_KernelCreateFn = std::function<OpKernel*(const OpKernelInfo& info)>;
using Prov_KernelCreatePtrFn = std::add_pointer<OpKernel*(const OpKernelInfo& info)>::type;

struct Prov_KernelCreateInfo {
  std::unique_ptr<Prov_KernelDef> kernel_def;  // Owned and stored in the global kernel registry.
  Prov_KernelCreateFn kernel_create_func;

  Prov_KernelCreateInfo(std::unique_ptr<Prov_KernelDef> definition,
                        Prov_KernelCreateFn create_func)
      : kernel_def(std::move(definition)),
        kernel_create_func(create_func) {}

  Prov_KernelCreateInfo(Prov_KernelCreateInfo&& other) noexcept
      : kernel_def(std::move(other.kernel_def)),
        kernel_create_func(std::move(other.kernel_create_func)) {}
};

using Prov_BuildKernelCreateInfoFn = Prov_KernelCreateInfo (*)();

struct Prov_KernelDefBuilder {
  static std::unique_ptr<Prov_KernelDefBuilder> Create();

  virtual ~Prov_KernelDefBuilder() = default;
  virtual Prov_KernelDefBuilder& SetName(const char* op_name) = 0;
  virtual Prov_KernelDefBuilder& SetDomain(const char* domain) = 0;
  virtual Prov_KernelDefBuilder& SinceVersion(int since_version) = 0;
  virtual Prov_KernelDefBuilder& Provider(const char* provider_type) = 0;
  virtual Prov_KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType supported_type) = 0;

  virtual std::unique_ptr<Prov_KernelDef> Build() = 0;

  void operator=(const Prov_KernelDefBuilder& v) = delete;
};

using NodeIndex = size_t;
using Prov_NodeAttributes = std::unordered_map<std::string, ONNX_NAMESPACE::Prov_AttributeProto_Copyable>;

using Prov_InitializedTensorSet = std::unordered_map<std::string, const ONNX_NAMESPACE::Prov_TensorProto*>;

struct Prov_NodeArg {
  virtual ~Prov_NodeArg() = default;
  virtual const std::string& Name() const noexcept = 0;
  virtual const ONNX_NAMESPACE::Prov_TensorShapeProto* Shape() const = 0;
  virtual ONNX_NAMESPACE::DataType Type() const noexcept = 0;

  void operator=(const Prov_NodeArg& v) = delete;
};

struct Prov_Node {
  virtual ~Prov_Node() = default;

  virtual const std::string& OpType() const noexcept = 0;
#if 0
  const ONNX_NAMESPACE::OpSchema* Op() const noexcept;
#endif

  virtual ConstPointerContainer<std::vector<Prov_NodeArg*>> InputDefs() const noexcept = 0;
  virtual ConstPointerContainer<std::vector<Prov_NodeArg*>> OutputDefs() const noexcept = 0;
  virtual NodeIndex Index() const noexcept = 0;

  virtual const Prov_NodeAttributes& GetAttributes() const noexcept = 0;
  virtual size_t GetInputEdgesCount() const noexcept = 0;
  virtual size_t GetOutputEdgesCount() const noexcept = 0;

  struct Prov_NodeIterator {
    virtual ~Prov_NodeIterator() {}
    virtual bool operator!=(const Prov_NodeIterator& p) const = 0;

    virtual void operator++() = 0;
    virtual const Prov_Node& operator*() = 0;
  };

  struct NodeConstIterator {
    NodeConstIterator(std::unique_ptr<Prov_NodeIterator> p) : impl_{std::move(p)} {}

    bool operator==(const NodeConstIterator& p_other) const;
    bool operator!=(const NodeConstIterator& p_other) const {
      return *impl_ != *p_other.impl_;
    }

    void operator++() {
      impl_->operator++();
    }
    void operator--();

    const Prov_Node& operator*() const {
      return impl_->operator*();
    }
    const Prov_Node* operator->() const;

    std::unique_ptr<Prov_NodeIterator> impl_;
  };

  NodeConstIterator InputNodesBegin() const noexcept { return NodeConstIterator(InputNodesBegin_internal()); }
  NodeConstIterator InputNodesEnd() const noexcept { return NodeConstIterator(InputNodesEnd_internal()); }

  virtual std::unique_ptr<Prov_NodeIterator> InputNodesBegin_internal() const noexcept = 0;
  virtual std::unique_ptr<Prov_NodeIterator> InputNodesEnd_internal() const noexcept = 0;

};  // namespace onnxruntime

#ifndef PROVIDER_BRIDGE_ORT
// if we are export the fused function to dll, the function will still in the same binary as lotus
// use std function to give execution provider some chance to capture some state.
using CreateFunctionStateFunc = std::function<int(ComputeContext*, FunctionState*)>;
using ComputeFunc = std::function<Status(FunctionState, const OrtApi*, OrtKernelContext*)>;
using DestroyFunctionStateFunc = std::function<void(FunctionState)>;

struct NodeComputeInfo {
  CreateFunctionStateFunc create_state_func;
  ComputeFunc compute_func;
  DestroyFunctionStateFunc release_state_func;
};
#endif

struct Prov_GraphViewer {
  virtual ~Prov_GraphViewer() = default;
  virtual const std::string& Name() const noexcept = 0;

#if 0
  virtual const std::string& Description() const noexcept = 0;
  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const;
  bool CanOverrideInitializer() const noexcept;
  const std::vector<const NodeArg*>& GetInputs() const noexcept;
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept;
  const std::vector<const NodeArg*>& GetOutputs() const noexcept;
  const std::vector<const NodeArg*>& GetValueInfo() const noexcept;
#endif

  virtual const Prov_Node* GetNode(NodeIndex node_index) const = 0;

#if 0
  const GraphNodes& Nodes() const noexcept;
  int NumberOfNodes() const noexcept;
#endif

  virtual int MaxNodeIndex() const noexcept = 0;

#if 0
  const std::vector<NodeIndex>& GetNodesInTopologicalOrder() const;
  const std::vector<NodeIndex>& GetRootNodes() const;
#endif
  virtual const Prov_InitializedTensorSet& GetAllInitializedTensors() const noexcept = 0;

#if 0
  const NodeArg* GetNodeArg(const std::string& name) const;
#endif

  virtual const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept = 0;
#if 0
  bool IsSubgraph() const;
  bool IsConstantInitializer(const std::string& name, bool check_outer_scope) const;
#endif

  void operator=(const Prov_GraphViewer& v) = delete;
};

struct Prov_IndexedSubGraph {
  static std::unique_ptr<Prov_IndexedSubGraph> Create();
  virtual ~Prov_IndexedSubGraph() = default;

  struct MetaDef {
    std::string name;    ///< Name of customized SubGraph/FunctionProto
    std::string domain;  ///< Domain of customized SubGraph/FunctionProto
    int since_version;   ///< Since version of customized SubGraph/FunctionProto.

    ONNX_NAMESPACE::OperatorStatus status;  ///< Status of customized SubGraph/FunctionProto.

    std::vector<std::string> inputs;   ///< Inputs of customized SubGraph/FunctionProto.
    std::vector<std::string> outputs;  ///< Outputs of customized SubGraph/FunctionProto.
    Prov_NodeAttributes attributes;    ///< Attributes of customized SubGraph/FunctionProto.

    std::string doc_string;  ///< Doc string of customized SubGraph/FunctionProto.
  };

  /** Nodes covered by this subgraph. The NodeIndex values are from the parent Graph.*/
  virtual std::vector<onnxruntime::NodeIndex>& Nodes() = 0;

  virtual void SetMetaDef(std::unique_ptr<MetaDef>& meta_def_) = 0;

  void operator=(const Prov_IndexedSubGraph& v) = delete;
};

struct Prov_KernelRegistry {
  static std::shared_ptr<Prov_KernelRegistry> Create();

  virtual ~Prov_KernelRegistry() = default;
  virtual Status Register(Prov_KernelCreateInfo&& create_info) = 0;

  void operator=(const Prov_KernelRegistry& v) = delete;
};

struct Prov_ComputeCapability {
  Prov_ComputeCapability(std::unique_ptr<Prov_IndexedSubGraph> t_sub_graph) : t_sub_graph_{std::move(t_sub_graph)} {}

  std::unique_ptr<Prov_IndexedSubGraph> t_sub_graph_;

  void operator=(const Prov_ComputeCapability& v) = delete;
};

// Provides the base class implementations, since Prov_IExecutionProvider is just an interface. This is to fake the C++ inheritance used by internal IExecutionProvider implementations
struct Prov_IExecutionProvider_Router {
  virtual ~Prov_IExecutionProvider_Router() {}

  virtual std::shared_ptr<Prov_KernelRegistry> Prov_GetKernelRegistry() const = 0;

  virtual std::vector<std::unique_ptr<Prov_ComputeCapability>> Prov_GetCapability(const onnxruntime::Prov_GraphViewer& graph,
                                                                                  const std::vector<const Prov_KernelRegistry*>& kernel_registries) const = 0;

  virtual Prov_AllocatorPtr Prov_GetAllocator(int id, OrtMemType mem_type) const = 0;
  virtual void Prov_InsertAllocator(Prov_AllocatorPtr allocator) = 0;

  void operator=(const Prov_IExecutionProvider_Router& v) = delete;
};

struct Prov_IExecutionProvider {
  Prov_IExecutionProvider(const std::string& type);
  virtual ~Prov_IExecutionProvider() {}

  virtual std::shared_ptr<Prov_KernelRegistry> Prov_GetKernelRegistry() const { return p_->Prov_GetKernelRegistry(); }

  virtual std::vector<std::unique_ptr<Prov_ComputeCapability>> Prov_GetCapability(const onnxruntime::Prov_GraphViewer& graph,
                                                                                  const std::vector<const Prov_KernelRegistry*>& kernel_registries) const { return p_->Prov_GetCapability(graph, kernel_registries); }

  virtual common::Status Prov_Compile(const std::vector<Prov_Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) = 0;

  virtual Prov_AllocatorPtr Prov_GetAllocator(int id, OrtMemType mem_type) const { return p_->Prov_GetAllocator(id, mem_type); }
  virtual void Prov_InsertAllocator(Prov_AllocatorPtr allocator) { return p_->Prov_InsertAllocator(allocator); }

  Prov_IExecutionProvider_Router* p_;

  void operator=(const Prov_IExecutionProvider& v) = delete;
};

namespace logging {
class Logger;
}

struct Provider {
  virtual std::shared_ptr<Prov_IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) = 0;
  virtual void SetProviderHost(ProviderHost& host) = 0;
};

// There are two ways to route a function, one is a virtual method and the other is a function pointer (or pointer to member function)
// The function pointers are nicer in that they directly call the target function, but they cannot be used in cases where we're calling
// a specific implementation of a virtual class member. Trying to get a pointer to member of a virtual function will return a thunk that
// calls the virtual function (which will lead to infinite recursion in the bridge). There is no known way to get the non virtual member
// function pointer implementation in this case.
struct ProviderHost {
  virtual Prov_AllocatorPtr CreateAllocator(Prov_DeviceAllocatorRegistrationInfo& info, int16_t device_id = 0) = 0;

  virtual logging::Logger* LoggingManager_GetDefaultLogger() = 0;

  virtual std::unique_ptr<ONNX_NAMESPACE::Prov_AttributeProto> AttributeProto_Create() = 0;

  virtual std::unique_ptr<Prov_OrtMemoryInfo> OrtMemoryInfo_Create(const char* name_, OrtAllocatorType type_, Prov_OrtDevice* device_, int id_, OrtMemType mem_type_) = 0;
  virtual std::unique_ptr<Prov_KernelDefBuilder> KernelDefBuilder_Create() = 0;

  virtual std::shared_ptr<Prov_KernelRegistry> KernelRegistry_Create() = 0;

  virtual std::unique_ptr<Prov_IndexedSubGraph> IndexedSubGraph_Create() = 0;

  virtual std::unique_ptr<Prov_IDeviceAllocator> CreateCPUAllocator(std::unique_ptr<Prov_OrtMemoryInfo> memory_info) = 0;
  virtual std::unique_ptr<Prov_IExecutionProvider_Router> Create_IExecutionProvider_Router(Prov_IExecutionProvider* outer, const std::string& type) = 0;

  virtual void SessionOptions_AddProviderFactory(OrtSessionOptions& options, std::shared_ptr<Prov_IExecutionProviderFactory> provider) = 0;

  MLDataType (*DataTypeImpl_GetType_Tensor)();
  MLDataType (*DataTypeImpl_GetType_float)();
  MLDataType (*DataTypeImpl_GetTensorType_float)();

  virtual void* HeapAllocate(size_t size) = 0;
  virtual void HeapFree(void*) = 0;

  virtual void LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file, const char* function, uint32_t line) = 0;

  virtual bool CPU_HasAVX2() = 0;
  virtual bool CPU_HasAVX512f() = 0;
};

}  // namespace onnxruntime
