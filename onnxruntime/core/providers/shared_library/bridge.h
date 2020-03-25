namespace ONNX_NAMESPACE {
class ValueInfoProto;
class TensorProto;
class TensorShapeProto;
class TypeProto;
class AttributeProto;
class OpSchema;
// String pointer as unique TypeProto identifier.
using DataType = const std::string*;
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

using NodeIndex = size_t;
class Graph;
class NodeArg;
class Node;
class GraphNodes;
class TensorShape;

struct Prov_IExecutionProvider;

struct Prov_IExecutionProviderFactory {
  virtual ~Prov_IExecutionProviderFactory() = default;
  virtual std::unique_ptr<Prov_IExecutionProvider> CreateProvider() = 0;
};

struct ProviderHost;
struct KernelCreateInfo;

class DataTypeImpl;
using MLDataType = const DataTypeImpl*;

struct Prov_OrtDevice {
  virtual ~Prov_OrtDevice() {}
};

struct Prov_OrtMemoryInfo {
  static std::unique_ptr<Prov_OrtMemoryInfo> Create(const char* name_, OrtAllocatorType type_, Prov_OrtDevice* device_ = nullptr, int id_ = 0, OrtMemType mem_type_ = OrtMemTypeDefault);
  virtual ~Prov_OrtMemoryInfo() {}
};

template <typename T>
using Prov_IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

struct Prov_IAllocator {
  virtual ~Prov_IAllocator() {}

  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p) = 0;
  virtual const Prov_OrtMemoryInfo& Info() const = 0;

  template <typename T>
  static Prov_IAllocatorUniquePtr<T> MakeUniquePtr(std::shared_ptr<Prov_IAllocator> allocator, size_t count_or_bytes) {
    __debugbreak();
    allocator;
    count_or_bytes;
    return nullptr;
  }
};

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

  virtual Prov_KernelDefBuilder& SetName(const char* op_name) = 0;
  virtual Prov_KernelDefBuilder& SetDomain(const char* domain) = 0;
  virtual Prov_KernelDefBuilder& SinceVersion(int since_version) = 0;
  virtual Prov_KernelDefBuilder& Provider(const char* provider_type) = 0;
  virtual Prov_KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType supported_type) = 0;

  virtual std::unique_ptr<Prov_KernelDef> Build() = 0;
};

class GraphViewer;
struct IndexedSubGraph;

struct Prov_KernelRegistry {
  static std::shared_ptr<Prov_KernelRegistry> Create();
  virtual Status Register(Prov_KernelCreateInfo&& create_info) = 0;
};

struct Prov_ComputeCapability {
  Prov_ComputeCapability(std::unique_ptr<IndexedSubGraph> t_sub_graph);
};

// Provides the base class implementations, since Prov_IExecutionProvider is just an interface. This is to fake the C++ inheritance used by internal IExecutionProvider implementations
struct Prov_IExecutionProvider_Router {
  virtual ~Prov_IExecutionProvider_Router() {}

  virtual std::shared_ptr<Prov_KernelRegistry> Prov_GetKernelRegistry() const = 0;

  virtual std::vector<std::unique_ptr<Prov_ComputeCapability>> Prov_GetCapability(const onnxruntime::GraphViewer& graph,
                                                                                  const std::vector<const Prov_KernelRegistry*>& kernel_registries) const = 0;

  virtual Prov_AllocatorPtr Prov_GetAllocator(int id, OrtMemType mem_type) const = 0;
  virtual void Prov_InsertAllocator(Prov_AllocatorPtr allocator) = 0;
};

struct Prov_IExecutionProvider {
  Prov_IExecutionProvider(const std::string& type);
  virtual ~Prov_IExecutionProvider() {}

  virtual std::shared_ptr<Prov_KernelRegistry> Prov_GetKernelRegistry() const { return p_->Prov_GetKernelRegistry(); }

  virtual std::vector<std::unique_ptr<Prov_ComputeCapability>> Prov_GetCapability(const onnxruntime::GraphViewer& graph,
                                                                                  const std::vector<const Prov_KernelRegistry*>& kernel_registries) const { return p_->Prov_GetCapability(graph, kernel_registries); }
#if 0
  virtual common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) = 0;
#endif

  virtual Prov_AllocatorPtr Prov_GetAllocator(int id, OrtMemType mem_type) const { return p_->Prov_GetAllocator(id, mem_type); }
  virtual void Prov_InsertAllocator(Prov_AllocatorPtr allocator) { return p_->Prov_InsertAllocator(allocator); }

  std::unique_ptr<Prov_IExecutionProvider_Router> p_;
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
  virtual void* IExecutionProvider_constructor(const std::string& type) = 0;
  virtual void IExecutionProvider_destructor(void* proxy) = 0;

  virtual void IExecutionProvider_InsertAllocator(Prov_AllocatorPtr allocator) = 0;

  virtual Prov_AllocatorPtr CreateAllocator(Prov_DeviceAllocatorRegistrationInfo& info, int device_id = 0) = 0;

  virtual logging::Logger* LoggingManager_GetDefaultLogger() = 0;

  virtual std::unique_ptr<Prov_OrtMemoryInfo> OrtMemoryInfo_Create(const char* name_, OrtAllocatorType type_, Prov_OrtDevice* device_, int id_, OrtMemType mem_type_) = 0;
  virtual std::unique_ptr<Prov_KernelDefBuilder> KernelDefBuilder_Create() = 0;

  virtual std::shared_ptr<Prov_KernelRegistry> KernelRegistry_Create() = 0;

  //  virtual OrtMemoryInfo* OrtMemoryInfo_constructor(const char* name_, OrtAllocatorType type_, OrtDevice& device_, int id_, OrtMemType mem_type_) = 0;
  //  virtual void OrtMemoryInfo_destructor(OrtMemoryInfo* proxy) = 0;

  virtual std::unique_ptr<Prov_IDeviceAllocator> CreateCPUAllocator(std::unique_ptr<Prov_OrtMemoryInfo> memory_info) = 0;
  virtual std::unique_ptr<Prov_IExecutionProvider_Router> Create_IExecutionProvider_Router(Prov_IExecutionProvider* outer, const std::string& type) = 0;

  virtual void SessionOptions_AddProviderFactory(OrtSessionOptions& options, std::shared_ptr<Prov_IExecutionProviderFactory> provider) = 0;

  MLDataType (*DataTypeImpl_GetType_Tensor)();
  MLDataType (*DataTypeImpl_GetType_float)();
  MLDataType (*DataTypeImpl_GetTensorType_float)();

  virtual void* HeapAllocate(size_t size) = 0;
  virtual void HeapFree(void*) = 0;

  //  const ::std::string& (*google_protobuf_internal_GetEmptyStringAlreadyInited)();
  //  void (google::protobuf::internal::RepeatedPtrFieldBase::*google_protobuf_internal_RepeatedPtrFieldBase_Reserve)(int new_size);
  //  onnx::TensorProto* (*google_protobuf_Arena_CreateMaybeMessage_onnx_TensorProto)(google::protobuf::Arena*);

  virtual const TensorShape& Tensor_Shape(const void* _this) = 0;

#if 0
  // Special functions in bridge_special.h that route through these
  virtual void onnx_AttributeProto_constructor(void* _this) = 0;
  virtual void onnx_AttributeProto_copy_constructor(void* _this, void* copy) = 0;
  virtual void onnx_AttributeProto_destructor(void* _this) = 0;

  virtual void onnxruntime_Node_NodeConstIterator_constructor(void* _this, void* param) = 0;

  virtual void onnxruntime_Status_constructor_1(void* _this, const void* category, int code, char const* msg) = 0;
  virtual void onnxruntime_Status_constructor_2(void* _this, const void* category, int code, const void* std_string_msg) = 0;

  virtual void onnxruntime_TensorShape_constructor(void* _this, int64_t const* p1, uint64_t p2) = 0;

  virtual void onnxruntime_OpKernelInfo_constructor(void* _this, void* p1, void* p2, void* p3, void* p4, void* p5, void* p6, void* p7) = 0;
  virtual void onnxruntime_OpKernelInfo_copy_constructor(void* _this, void* copy) = 0;
  //
#endif

  //  void (onnx::AttributeProto::*onnx_AttributeProto_CopyFrom)(onnx::AttributeProto const& p1);

  //  bool (*onnx_AttributeProto_AttributeType_IsValid)(int p1);

  //  void (onnx::TensorProto::*onnx_TensorProto_CopyFrom)(onnx::TensorProto const& p1);

  //  std::shared_ptr<IAllocator> (*CreateAllocator)(DeviceAllocatorRegistrationInfo info, int device_id);

  //  const CPUIDInfo& (*CPUIDInfo_GetCPUIDInfo)();

  virtual void LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file, const char* function, uint32_t line) = 0;

#if 0
  virtual void* CPUAllocator_Alloc(CPUAllocator* _this, uint64_t p1) = 0;
  virtual void CPUAllocator_Free(CPUAllocator* _this, void* p1) = 0;
  virtual const OrtMemoryInfo& CPUAllocator_Info(const CPUAllocator* _this) = 0;

  virtual std::shared_ptr<IAllocator> IExecutionProvider_GetAllocator(const IExecutionProvider* _this, int, OrtMemType) = 0;
  virtual std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider_GetCapability(const IExecutionProvider* _this, const GraphViewer& p1, const std::vector<const KernelRegistry*>& p2) = 0;
  virtual void IExecutionProvider_InsertAllocator(IExecutionProvider* _this, std::shared_ptr<IAllocator> p1) = 0;
  virtual Status IExecutionProvider_OnRunEnd(IExecutionProvider* _this) = 0;
  virtual Status IExecutionProvider_OnRunStart(IExecutionProvider* _this) = 0;

  virtual const InitializedTensorSet& GraphViewer_GetAllInitializedTensors(const GraphViewer* _this) = 0;
  virtual const Node* GraphViewer_GetNode(const GraphViewer* _this, NodeIndex p1) = 0;
  virtual int GraphViewer_MaxNodeIndex(const GraphViewer* _this) = 0;
  virtual const std::string& GraphViewer_Name(const GraphViewer* _this) = 0;

  virtual Status KernelRegistry_Register(KernelRegistry* _this, KernelCreateInfo&& p1) = 0;

  virtual const NodeAttributes& Node_GetAttributes(const Node* _this) = 0;
  virtual NodeIndex Node_Index(const Node* _this) = 0;
  virtual const ONNX_NAMESPACE::OpSchema* Node_Op(const Node* _this) = 0;
  virtual const std::string& Node_OpType(const Node* _this) = 0;

  virtual bool Node_NodeConstIterator_operator_not_equal(const void* _this, const void* p1) = 0;
  virtual void Node_NodeConstIterator_operator_plusplus(void* _this) = 0;
  virtual const Node& Node_NodeConstIterator_operator_star(const void* _this) = 0;

  virtual const std::string& NodeArg_Name(const NodeArg* _this) = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto* NodeArg_Shape(const NodeArg* _this) = 0;
  virtual ONNX_NAMESPACE::DataType NodeArg_Type(const NodeArg* _this) = 0;

  virtual void* TensorShape_TensorShape(const std::initializer_list<int64_t>& dims) override {
    virtual int64_t TensorShape_Size(const void* _this) = 0;
    //  virtual TensorShape TensorShape_Slice(const TensorShape* _this, uint64_t) = 0;
#endif
};

}  // namespace onnxruntime
