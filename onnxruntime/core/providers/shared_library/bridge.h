namespace onnxruntime {

struct IExecutionProviderFactory;
struct ProviderHost;
struct KernelCreateInfo;
class CPUIDInfo;

namespace logging {
class Logger;
}

struct Provider {
  virtual std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) = 0;
  virtual void SetProviderHost(ProviderHost& host) = 0;
};

// There are two ways to route a function, one is a virtual method and the other is a function pointer (or pointer to member function)
// The function pointers are nicer in that they directly call the target function, but they cannot be used in cases where we're calling
// a specific implementation of a virtual class member. Trying to get a pointer to member of a virtual function will return a thunk that
// calls the virtual function (which will lead to infinite recursion in the bridge). There is no known way to get the non virtual member
// function pointer implementation in this case.
struct ProviderHost {
  virtual logging::Logger* LoggingManager_GetDefaultLogger() = 0;

  MLDataType (*DataTypeImpl_GetType_Tensor)();
  MLDataType (*DataTypeImpl_GetType_float)();
  MLDataType (*DataTypeImpl_GetTensorType_float)();

  virtual void* HeapAllocate(size_t size) = 0;
  virtual void HeapFree(void*) = 0;

  const ::std::string& (*google_protobuf_internal_GetEmptyStringAlreadyInited)();
  void (google::protobuf::internal::RepeatedPtrFieldBase::*google_protobuf_internal_RepeatedPtrFieldBase_Reserve)(int new_size);
  onnx::TensorProto* (*google_protobuf_Arena_CreateMaybeMessage_onnx_TensorProto)(google::protobuf::Arena*);

  // Special functions in bridge_special.h that route through these
  virtual void onnx_AttributeProto_constructor(void* _this) = 0;
  virtual void onnx_AttributeProto_copy_constructor(void* _this, void* copy) = 0;
  virtual void onnx_AttributeProto_destructor(void* _this) = 0;

  virtual void onnxruntime_TensorShape_constructor(void* _this, __int64 const* p1, unsigned __int64 p2) = 0;

  virtual void onnxruntime_OpKernelInfo_constructor(void* _this, void* p1, void* p2, void* p3, void* p4, void* p5, void* p6, void* p7) = 0;
  virtual void onnxruntime_OpKernelInfo_copy_constructor(void* _this, void* copy) = 0;
  //

  void (onnx::AttributeProto::*onnx_AttributeProto_CopyFrom)(onnx::AttributeProto const& p1);

  bool (*onnx_AttributeProto_AttributeType_IsValid)(int p1);

  void (onnx::TensorProto::*onnx_TensorProto_CopyFrom)(onnx::TensorProto const& p1);

  std::shared_ptr<IAllocator> (*CreateAllocator)(DeviceAllocatorRegistrationInfo info, int device_id);

  const CPUIDInfo& (*CPUIDInfo_GetCPUIDInfo)();

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

  KernelDefBuilder& (KernelDefBuilder::*KernelDefBuilder_Provider)(char const* p1);
  KernelDefBuilder& (KernelDefBuilder::*KernelDefBuilder_SetName)(char const* p1);
  KernelDefBuilder& (KernelDefBuilder::*KernelDefBuilder_SetDomain)(char const* p1);
  KernelDefBuilder& (KernelDefBuilder::*KernelDefBuilder_TypeConstraint)(char const* p1, const DataTypeImpl* p2);

  virtual Status KernelRegistry_Register(KernelRegistry* _this, KernelCreateInfo&& p1) = 0;

  virtual const NodeAttributes& Node_GetAttributes(const Node* _this) = 0;
  virtual NodeIndex Node_Index(const Node* _this) = 0;
  virtual const ONNX_NAMESPACE::OpSchema* Node_Op(const Node* _this) = 0;
  virtual const std::string& Node_OpType(const Node* _this) = 0;

  virtual const std::string& NodeArg_Name(const NodeArg* _this) = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto* NodeArg_Shape(const NodeArg* _this) = 0;
  virtual ONNX_NAMESPACE::DataType NodeArg_Type(const NodeArg* _this) = 0;

  virtual int64_t TensorShape_Size(const TensorShape* _this) = 0;
  virtual TensorShape TensorShape_Slice(const TensorShape* _this, unsigned __int64) = 0;
};

}  // namespace onnxruntime
