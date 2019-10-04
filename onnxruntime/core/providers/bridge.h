namespace onnxruntime {

struct IExecutionProviderFactory;
struct ProviderHost;

struct Provider {
  virtual std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) = 0;
  virtual void SetProviderHost(ProviderHost& host) = 0;
};

struct ProviderHost {
  virtual MLDataType DataTypeImpl_GetType_Tensor() = 0;
  virtual MLDataType DataTypeImpl_GetType_float() = 0;

  virtual void* HeapAllocate(size_t size) = 0;
  virtual void HeapFree(void*) = 0;

  virtual std::shared_ptr<IAllocator> CreateAllocator(DeviceAllocatorRegistrationInfo&& info, int device_id) = 0;

  virtual const OrtMemoryInfo& CPUAllocator_Info(const CPUAllocator* _this) = 0;

#if 0

  virtual MLDataType GetType(DataTypes::Type type) = 0;
  virtual MLDataType GetTensorType(TensorDataTypes::Type type) = 0;

  virtual std::basic_ostream<char>& operator_leftshift(std::basic_ostream<char>& stream, DataTypeImpl const*) = 0;

  virtual __int64 TensorShape_Size(const TensorShape* pThis) = 0;

  virtual const MLValue* OpKernelContext_GetInputMLValue(const OpKernelContext* pThis, int index) = 0;
  virtual Tensor* OpKernelContext_Output(OpKernelContext* pThis, int index, const TensorShape& shape) = 0;

  virtual void AttributeProto_Destructor(::onnx::AttributeProto* pThis) = 0;

  virtual ::onnx::OpSchema& OpSchema_SinceVersion(::onnx::OpSchema* pThis, ::onnx::OperatorSetVersion n) = 0;

  virtual ::onnx::OpSchema& OpSchema_Input(::onnx::OpSchema* pThis,
                                           int n,
                                           const char* name,
                                           const char* description,
                                           const char* type_str,
                                           ::onnx::OpSchema::FormalParameterOption param_option) = 0;

  virtual ::onnx::OpSchema& OpSchema_Output(::onnx::OpSchema* pThis,
                                            int n,
                                            const char* name,
                                            const char* description,
                                            const char* type_str,
                                            ::onnx::OpSchema::FormalParameterOption param_option) = 0;

  virtual ::onnx::OpSchema& OpSchema_TypeConstraint(
      ::onnx::OpSchema* pThis,
      std::string type_str,
      std::vector<std::string> constraints,
      std::string description) = 0;

  virtual void AttributeProto_AttributeProto(::onnx::AttributeProto* pThis, const ::onnx::AttributeProto& copy) = 0;
  virtual ::google::protobuf::uint8* AttributeProto_InternalSerializeWithCachedSizesToArray(const ::onnx::AttributeProto* pThis,
                                                                                            bool deterministic, ::google::protobuf::uint8* target) = 0;
  virtual ::google::protobuf::Metadata AttributeProto_GetMetadata(const ::onnx::AttributeProto* pThis) = 0;
  virtual bool AttributeProto_MergePartialFromCodedStream(::onnx::AttributeProto* pThis, ::google::protobuf::io::CodedInputStream* input) = 0;
  virtual void AttributeProto_SerializeWithCachedSizes(const ::onnx::AttributeProto* pThis, ::google::protobuf::io::CodedOutputStream* output) = 0;
  virtual void AttributeProto_SetCachedSize(const ::onnx::AttributeProto* pThis, int size) = 0;
  virtual size_t AttributeProto_ByteSizeLong(const ::onnx::AttributeProto* pThis) = 0;
  virtual bool AttributeProto_IsInitialized(const ::onnx::AttributeProto* pThis) = 0;
  virtual void AttributeProto_Clear(::onnx::AttributeProto* pThis) = 0;
  virtual void AttributeProto_CopyFrom(::onnx::AttributeProto* pThis, const ::google::protobuf::Message& message) = 0;
  virtual void AttributeProto_MergeFrom(::onnx::AttributeProto* pThis, const ::google::protobuf::Message& message) = 0;
#endif
};

}  // namespace onnxruntime
