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

struct ProviderHost {
  virtual logging::Logger* LoggingManager_GetDefaultLogger() = 0;

  MLDataType (*DataTypeImpl_GetType_Tensor)();
  MLDataType (*DataTypeImpl_GetType_float)();
  MLDataType (*DataTypeImpl_GetTensorType_float)();

  virtual void* HeapAllocate(size_t size) = 0;
  virtual void HeapFree(void*) = 0;

  virtual const ::std::string& google_protobuf_internal_GetEmptyStringAlreadyInited() = 0;
  void (google::protobuf::internal::RepeatedPtrFieldBase::*google_protobuf_internal_RepeatedPtrFieldBase_Reserve)(int new_size);
  onnx::TensorProto* (*google_protobuf_Arena_CreateMaybeMessage_onnx_TensorProto)(google::protobuf::Arena*);

  virtual void onnx_AttributeProto_constructor(void* _this) = 0;
  virtual void onnx_AttributeProto_copy_constructor(void* _this, void* copy) = 0;
  virtual void onnx_AttributeProto_destructor(void* _this) = 0;
  void (onnx::AttributeProto::*onnx_AttributeProto_CopyFrom)(onnx::AttributeProto const& p1);

  virtual bool onnx_AttributeProto_AttributeType_IsValid(int p1) = 0;

  void (onnx::TensorProto::*onnx_TensorProto_CopyFrom)(onnx::TensorProto const& p1);

  virtual std::shared_ptr<IAllocator> CreateAllocator(DeviceAllocatorRegistrationInfo&& info, int device_id) = 0;

  virtual const CPUIDInfo& CPUIDInfo_GetCPUIDInfo() = 0;

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

  virtual KernelDefBuilder& KernelDefBuilder_Provider(KernelDefBuilder* _this, char const* p1) = 0;
  virtual KernelDefBuilder& KernelDefBuilder_SetName(KernelDefBuilder* _this, char const* p1) = 0;
  virtual KernelDefBuilder& KernelDefBuilder_SetDomain(KernelDefBuilder* _this, char const* p1) = 0;
  virtual KernelDefBuilder& KernelDefBuilder_TypeConstraint(KernelDefBuilder* _this, char const* p1, const DataTypeImpl* p2) = 0;

  virtual Status KernelRegistry_Register(KernelRegistry* _this, KernelCreateInfo&& p1) = 0;

  virtual const NodeAttributes& Node_GetAttributes(const Node* _this) = 0;
  virtual NodeIndex Node_Index(const Node* _this) = 0;
  virtual const ONNX_NAMESPACE::OpSchema* Node_Op(const Node* _this) = 0;
  virtual const std::string& Node_OpType(const Node* _this) = 0;

  virtual const std::string& NodeArg_Name(const NodeArg* _this) = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto* NodeArg_Shape(const NodeArg* _this) = 0;
  virtual ONNX_NAMESPACE::DataType NodeArg_Type(const NodeArg* _this) = 0;

  virtual void onnxruntime_TensorShape_constructor(void* _this, __int64 const* p1, unsigned __int64 p2) = 0;
  virtual int64_t TensorShape_Size(const TensorShape* _this) = 0;
  virtual TensorShape TensorShape_Slice(const TensorShape* _this, unsigned __int64) = 0;
};

}  // namespace onnxruntime
