// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {

using ProviderType = const std::string&;

struct CPUIDInfo final {
  static const CPUIDInfo& GetCPUIDInfo() { return Provider_GetHost()->CPUIDInfo__GetCPUIDInfo(); }

  bool HasAVX2() const { return Provider_GetHost()->CPUIDInfo__HasAVX2(this); }
  bool HasAVX512f() const { return Provider_GetHost()->CPUIDInfo__HasAVX512f(this); }
  bool HasAVX512_BF16() const { return Provider_GetHost()->CPUIDInfo__HasAVX512_BF16(this); }
  bool HasAMX_BF16() const { return Provider_GetHost()->CPUIDInfo__HasAMX_BF16(this); }
  bool HasAVX512Skylake() const { return Provider_GetHost()->CPUIDInfo__HasAVX512Skylake(this); }

  PROVIDER_DISALLOW_ALL(CPUIDInfo)
};

namespace logging {

struct Logger final {
  bool OutputIsEnabled(Severity severity, DataType data_type) const noexcept { return Provider_GetHost()->logging__Logger__OutputIsEnabled(this, severity, data_type); }

  PROVIDER_DISALLOW_ALL(Logger)
};

struct LoggingManager final {
  static const Logger& DefaultLogger() { return Provider_GetHost()->logging__LoggingManager__DefaultLogger(); }

  PROVIDER_DISALLOW_ALL(LoggingManager)
};

struct Capture final {
  static std::unique_ptr<Capture> Create(const Logger& logger, logging::Severity severity, const char* category,
                                         logging::DataType dataType, const CodeLocation& location) { return Provider_GetHost()->logging__Capture__construct(logger, severity, category, dataType, location); }
  static void operator delete(void* p) { Provider_GetHost()->logging__Capture__operator_delete(reinterpret_cast<Capture*>(p)); }

  std::ostream& Stream() noexcept { return Provider_GetHost()->logging__Capture__Stream(this); }

  Capture() = delete;
  Capture(const Capture&) = delete;
  void operator=(const Capture&) = delete;
};
}  // namespace logging
}  // namespace onnxruntime

namespace ONNX_NAMESPACE {

struct int64s final {
  int size() const { return Provider_GetHost()->int64s__size(this); }
  const int64_t& Get(int index) const { return Provider_GetHost()->int64s__Get(this, index); }
  const int64_t& operator[](int index) const { return Get(index); }

  PROVIDER_DISALLOW_ALL(int64s)
};

struct AttributeProto final {
  static std::unique_ptr<AttributeProto> Create() { return Provider_GetHost()->AttributeProto__construct(); }
  void operator=(const AttributeProto& v) { Provider_GetHost()->AttributeProto__operator_assign(this, v); }
  static void operator delete(void* p) { Provider_GetHost()->AttributeProto__operator_delete(reinterpret_cast<AttributeProto*>(p)); }

  const std::string& name() const { return Provider_GetHost()->AttributeProto__name(this); }
  AttributeProto_AttributeType type() const { return Provider_GetHost()->AttributeProto__type(this); }
  int ints_size() const { return Provider_GetHost()->AttributeProto__ints_size(this); }
  int floats_size() const { return Provider_GetHost()->AttributeProto__floats_size(this); }
  int strings_size() const { return Provider_GetHost()->AttributeProto__strings_size(this); }
  int64_t ints(int i) const { return Provider_GetHost()->AttributeProto__ints(this, i); }
  float floats(int i) const { return Provider_GetHost()->AttributeProto__floats(this, i); }
  const std::string& strings(int i) const { return Provider_GetHost()->AttributeProto__strings(this, i); }
  const int64s& ints() const { return Provider_GetHost()->AttributeProto__ints(this); }
  int64_t i() const { return Provider_GetHost()->AttributeProto__i(this); }
  float f() const { return Provider_GetHost()->AttributeProto__f(this); }
  void set_s(const ::std::string& value) { return Provider_GetHost()->AttributeProto__set_s(this, value); }
  const ::std::string& s() const { return Provider_GetHost()->AttributeProto__s(this); }
  void set_name(const ::std::string& value) { return Provider_GetHost()->AttributeProto__set_name(this, value); }
  void set_type(AttributeProto_AttributeType value) { return Provider_GetHost()->AttributeProto__set_type(this, value); }
  TensorProto* add_tensors() { return Provider_GetHost()->AttributeProto__add_tensors(this); }

  typedef AttributeProto_AttributeType AttributeType;
  static constexpr AttributeType UNDEFINED = AttributeProto_AttributeType_UNDEFINED;
  static constexpr AttributeType FLOAT = AttributeProto_AttributeType_FLOAT;
  static constexpr AttributeType INT = AttributeProto_AttributeType_INT;
  static constexpr AttributeType STRING = AttributeProto_AttributeType_STRING;
  static constexpr AttributeType TENSOR = AttributeProto_AttributeType_TENSOR;
  static constexpr AttributeType GRAPH = AttributeProto_AttributeType_GRAPH;
#if !defined(DISABLE_SPARSE_TENSORS)
  static constexpr AttributeType SPARSE_TENSOR = AttributeProto_AttributeType_SPARSE_TENSOR;
#endif
  static constexpr AttributeType FLOATS = AttributeProto_AttributeType_FLOATS;
  static constexpr AttributeType INTS = AttributeProto_AttributeType_INTS;
  static constexpr AttributeType STRINGS = AttributeProto_AttributeType_STRINGS;
  static constexpr AttributeType TENSORS = AttributeProto_AttributeType_TENSORS;
  static constexpr AttributeType GRAPHS = AttributeProto_AttributeType_GRAPHS;
#if !defined(DISABLE_SPARSE_TENSORS)
  static constexpr AttributeType SPARSE_TENSORS = AttributeProto_AttributeType_SPARSE_TENSORS;
#endif

  AttributeProto() = delete;
  AttributeProto(const AttributeProto&) = delete;
};

struct GraphProto final {
  static void operator delete(void* p) { Provider_GetHost()->GraphProto__operator_delete(reinterpret_cast<GraphProto*>(p)); }
  void operator=(const GraphProto& v) { return Provider_GetHost()->GraphProto__operator_assign(this, v); }

  const ValueInfoProto& input(int index) const { return Provider_GetHost()->GraphProto__input(this, index); }
  ValueInfoProtos* mutable_input() { return Provider_GetHost()->GraphProto__mutable_input(this); }
  ValueInfoProto* mutable_input(int index) { return Provider_GetHost()->GraphProto__mutable_input(this, index); }
  int input_size() const { return Provider_GetHost()->GraphProto__input_size(this); }

  const ValueInfoProtos& output() const { return Provider_GetHost()->GraphProto__output(this); }
  const ValueInfoProto& output(int index) const { return Provider_GetHost()->GraphProto__output(this, index); }
  ValueInfoProtos* mutable_output() { return Provider_GetHost()->GraphProto__mutable_output(this); }

  ValueInfoProtos* mutable_value_info() { return Provider_GetHost()->GraphProto__mutable_value_info(this); }
  TensorProtos* mutable_initializer() { return Provider_GetHost()->GraphProto__mutable_initializer(this); }
  NodeProto* add_node() { return Provider_GetHost()->GraphProto__add_node(this); }

  GraphProto() = delete;
  GraphProto(const GraphProto&) = delete;
};

struct ModelProto final {
  static std::unique_ptr<ModelProto> Create() { return Provider_GetHost()->ModelProto__construct(); }
  static void operator delete(void* p) { Provider_GetHost()->ModelProto__operator_delete(reinterpret_cast<ModelProto*>(p)); }

  bool SerializeToString(std::string& string) const { return Provider_GetHost()->ModelProto__SerializeToString(this, string); }
  bool SerializeToOstream(std::ostream& output) const { return Provider_GetHost()->ModelProto__SerializeToOstream(this, output); }
  bool ParseFromString(const std::string& data) { return Provider_GetHost()->ModelProto__ParseFromString(this, data); }
  std::string SerializeAsString() const { return Provider_GetHost()->ModelProto__SerializeAsString(this); }

  const GraphProto& graph() const { return Provider_GetHost()->ModelProto__graph(this); }
  GraphProto* mutable_graph() { return Provider_GetHost()->ModelProto__mutable_graph(this); }

  void set_ir_version(int64_t value) { return Provider_GetHost()->ModelProto__set_ir_version(this, value); }

  ModelProto() = delete;
  ModelProto(const ModelProto&) = delete;
  void operator=(const ModelProto&) = delete;
};

struct NodeProto final {
  static std::unique_ptr<NodeProto> Create() { return Provider_GetHost()->NodeProto__construct(); }
  static void operator delete(void* p) { Provider_GetHost()->NodeProto__operator_delete(reinterpret_cast<NodeProto*>(p)); }
  void operator=(const NodeProto& v) { Provider_GetHost()->NodeProto__operator_assign(this, v); }
  int attribute_size() { return Provider_GetHost()->NodeProto__attribute_size(this); }
  const AttributeProto& attribute(int index) const { return Provider_GetHost()->NodeProto__attribute(this, index); }

  NodeProto() = delete;
  NodeProto(const NodeProto&) = delete;
};

struct TensorProto final {
  static std::unique_ptr<TensorProto> Create() { return Provider_GetHost()->TensorProto__construct(); }
  static void operator delete(void* p) { Provider_GetHost()->TensorProto__operator_delete(reinterpret_cast<TensorProto*>(p)); }
  void operator=(const TensorProto& v) { Provider_GetHost()->TensorProto__operator_assign(this, v); }

  bool has_name() const { return Provider_GetHost()->TensorProto__has_name(this); }

  int dims_size() const { return Provider_GetHost()->TensorProto__dims_size(this); }
  const int64s& dims() const { return Provider_GetHost()->TensorProto__dims(this); }

  bool has_data_location() const { return Provider_GetHost()->TensorProto__has_data_location(this); }
  TensorProto_DataLocation data_location() const { return TensorProto_DataLocation(Provider_GetHost()->TensorProto__data_location(this)); }

  bool has_raw_data() const { return Provider_GetHost()->TensorProto__has_raw_data(this); }
  const std::string& raw_data() const { return Provider_GetHost()->TensorProto__raw_data(this); }

  int32_t data_type() const { return Provider_GetHost()->TensorProto__data_type(this); }

  typedef TensorProto_DataType DataType;
  static constexpr DataType UNDEFINED = TensorProto_DataType_UNDEFINED;

  static bool DataType_IsValid(int value) { return Provider_GetHost()->TensorProto_DataType_IsValid(value); }

  void copy_from(const TensorProto* other) { return Provider_GetHost()->TensorProto__CopyFrom(this, other); }

  TensorProto() = delete;
  TensorProto(const TensorProto&) = delete;
};

struct TensorProtos final {
  TensorProto* Add() { return Provider_GetHost()->TensorProtos__Add(this); }

  PROVIDER_DISALLOW_ALL(TensorProtos)
};

struct TensorShapeProto_Dimension final {
  enum ValueCase {
    kDimValue = 1,
    kDimParam = 2,
    VALUE_NOT_SET = 0,
  };

  ValueCase value_case() const { return ValueCase(Provider_GetHost()->TensorShapeProto_Dimension__value_case(this)); }
  const std::string& dim_param() const { return Provider_GetHost()->TensorShapeProto_Dimension__dim_param(this); }
  int64_t dim_value() const { return Provider_GetHost()->TensorShapeProto_Dimension__dim_value(this); }
  void set_dim_value(int64_t value) { return Provider_GetHost()->TensorShapeProto_Dimension__set_dim_value(this, value); }
  bool has_dim_value() const { return Provider_GetHost()->TensorShapeProto_Dimension__has_dim_value(this); }
  bool has_dim_param() const { return Provider_GetHost()->TensorShapeProto_Dimension__has_dim_param(this); }
  void clear_dim_value() { return Provider_GetHost()->TensorShapeProto_Dimension__clear_dim_value(this); }

  PROVIDER_DISALLOW_ALL(TensorShapeProto_Dimension)
};

struct TensorShapeProto_Dimensions final {
  IteratorHolder<TensorShapeProto_Dimension_Iterator, const TensorShapeProto_Dimension> begin() const { return Provider_GetHost()->TensorShapeProto_Dimensions__begin(this); }
  IteratorHolder<TensorShapeProto_Dimension_Iterator, const TensorShapeProto_Dimension> end() const { return Provider_GetHost()->TensorShapeProto_Dimensions__end(this); }

  PROVIDER_DISALLOW_ALL(TensorShapeProto_Dimensions)
};

struct TensorShapeProto final {
  int dim_size() const { return Provider_GetHost()->TensorShapeProto__dim_size(this); }
  const TensorShapeProto_Dimensions& dim() const { return Provider_GetHost()->TensorShapeProto__dim(this); }
  const TensorShapeProto_Dimension& dim(int index) const { return Provider_GetHost()->TensorShapeProto__dim(this, index); }
  TensorShapeProto_Dimension* mutable_dim(int index) { return Provider_GetHost()->TensorShapeProto__mutable_dim(this, index); }
  void clear_dim() { return Provider_GetHost()->TensorShapeProto__clear_dim(this); }
  TensorShapeProto_Dimension* add_dim() { return Provider_GetHost()->TensorShapeProto__add_dim(this); }

  PROVIDER_DISALLOW_ALL(TensorShapeProto)
};

struct TypeProto_Tensor final {
  bool has_shape() const { return Provider_GetHost()->TypeProto_Tensor__has_shape(this); }
  const TensorShapeProto& shape() const { return Provider_GetHost()->TypeProto_Tensor__shape(this); }
  TensorShapeProto* mutable_shape() { return Provider_GetHost()->TypeProto_Tensor__mutable_shape(this); }
  int32_t elem_type() const { return Provider_GetHost()->TypeProto_Tensor__elem_type(this); }

  PROVIDER_DISALLOW_ALL(TypeProto_Tensor)
};

#if !defined(DISABLE_SPARSE_TENSORS)
struct TypeProto_SparseTensor final {
  bool has_shape() const { return Provider_GetHost()->TypeProto_SparseTensor__has_shape(this); }
  const TensorShapeProto& shape() const { return Provider_GetHost()->TypeProto_SparseTensor__shape(this); }
  TensorShapeProto* mutable_shape() { return Provider_GetHost()->TypeProto_SparseTensor__mutable_shape(this); }
  int32_t elem_type() const { return Provider_GetHost()->TypeProto_SparseTensor__elem_type(this); }

  PROVIDER_DISALLOW_ALL(TypeProto_SparseTensor)
};
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)
struct TypeProto_Optional final {
  const TypeProto& elem_type() const { return Provider_GetHost()->TypeProto_Optional__elem_type(this); }
  TypeProto* mutable_elem_type() { return Provider_GetHost()->TypeProto_Optional__mutable_elem_type(this); }
  PROVIDER_DISALLOW_ALL(TypeProto_Optional)
};
#endif

struct TypeProto_Sequence final {
  const TypeProto& elem_type() const { return Provider_GetHost()->TypeProto_Sequence__elem_type(this); }
  TypeProto* mutable_elem_type() { return Provider_GetHost()->TypeProto_Sequence__mutable_elem_type(this); }
  PROVIDER_DISALLOW_ALL(TypeProto_Sequence)
};

struct TypeProto final {
  static std::unique_ptr<TypeProto> Create() { return Provider_GetHost()->TypeProto__construct(); }

  const TypeProto_Tensor& tensor_type() const { return Provider_GetHost()->TypeProto__tensor_type(this); }
  TypeProto_Tensor* mutable_tensor_type() { return Provider_GetHost()->TypeProto__mutable_tensor_type(this); }

#if !defined(DISABLE_SPARSE_TENSORS)
  const TypeProto_SparseTensor& sparse_tensor_type() const { return Provider_GetHost()->TypeProto__sparse_tensor_type(this); }
  TypeProto_SparseTensor* mutable_sparse_tensor_type() { return Provider_GetHost()->TypeProto__mutable_sparse_tensor_type(this); }
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)
  const TypeProto_Optional& optional_type() const { return Provider_GetHost()->TypeProto__optional_type(this); }
  TypeProto_Optional* mutable_optional_type() { return Provider_GetHost()->TypeProto__mutable_optional_type(this); }
#endif

  const TypeProto_Sequence& sequence_type() const { return Provider_GetHost()->TypeProto__sequence_type(this); }
  TypeProto_Sequence* mutable_sequence_type() { return Provider_GetHost()->TypeProto__mutable_sequence_type(this); }

  enum ValueCase {
    kTensorType = 1,
    kSequenceType = 4,
    kMapType = 5,
    kOptionalType = 9,
    kSparseTensorType = 8,
    kOpaqueType = 7,
    VALUE_NOT_SET = 0,
  };

  ValueCase value_case() const { return ValueCase(Provider_GetHost()->TypeProto__value_case(this)); }

  void copy_from(const TypeProto* other) { return Provider_GetHost()->TypeProto__CopyFrom(this, other); }

  TypeProto() = delete;
  TypeProto(const TypeProto&) = delete;
};

struct ValueInfoProto final {
  const TypeProto& type() const { return Provider_GetHost()->ValueInfoProto__type(this); }
  TypeProto* mutable_type() { return Provider_GetHost()->ValueInfoProto__mutable_type(this); }

  void operator=(const ValueInfoProto& v) { Provider_GetHost()->ValueInfoProto__operator_assign(this, v); }

  ValueInfoProto() = delete;
  ValueInfoProto(const ValueInfoProto&) = delete;
  static void operator delete(void*) = delete;
};

struct ValueInfoProtos final {
  ValueInfoProto* Add() { return Provider_GetHost()->ValueInfoProtos__Add(this); }
  const ValueInfoProto& operator[](int index) const { return Provider_GetHost()->ValueInfoProtos__operator_array(this, index); }

  PROVIDER_DISALLOW_ALL(ValueInfoProtos)
};

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

namespace utils {
bool IsDataTypeString(MLDataType dt_type);

}  // namespace utils

namespace Utils {

struct DataTypeUtils final {
  static const std::string* ToType(const ONNX_NAMESPACE::TypeProto& type_proto) { return Provider_GetHost()->Utils__DataTypeUtils__ToType(type_proto); }

  PROVIDER_DISALLOW_ALL(DataTypeUtils)
};

}  // namespace Utils

struct ComputeCapability final {
  static std::unique_ptr<ComputeCapability> Create(std::unique_ptr<IndexedSubGraph> t_sub_graph) { return Provider_GetHost()->ComputeCapability__construct(std::move(t_sub_graph)); }
  static void operator delete(void* p) { Provider_GetHost()->ComputeCapability__operator_delete(reinterpret_cast<ComputeCapability*>(p)); }

  std::unique_ptr<IndexedSubGraph>& SubGraph() { return Provider_GetHost()->ComputeCapability__SubGraph(this); }

  ComputeCapability() = delete;
  ComputeCapability(const ComputeCapability&) = delete;
  void operator=(const ComputeCapability&) = delete;
};

struct DataTransferManager final {
  Status CopyTensor(const Tensor& src, Tensor& dst) const { return Provider_GetHost()->DataTransferManager__CopyTensor(this, src, dst); }
#if !defined(DISABLE_SPARSE_TENSORS)
  Status CopySparseTensor(const SparseTensor& src, SparseTensor& dst) const { return Provider_GetHost()->DataTransferManager__CopySparseTensor(this, src, dst); }
  Status CopySparseTensors(const std::vector<IDataTransfer::SparseSrcDstPair>& src_dst_pairs) const { return Provider_GetHost()->DataTransferManager__CopySparseTensors(this, src_dst_pairs); }
#endif
  const IDataTransfer* GetDataTransfer(const OrtDevice& src_device, const OrtDevice& dst_device) const { return Provider_GetHost()->DataTransferManager__GetDataTransfer(this, src_device, dst_device); }

  PROVIDER_DISALLOW_ALL(DataTransferManager)
};

struct IndexedSubGraph_MetaDef final {
  static std::unique_ptr<IndexedSubGraph_MetaDef> Create() { return Provider_GetHost()->IndexedSubGraph_MetaDef__construct(); }
  static void operator delete(void* p) { Provider_GetHost()->IndexedSubGraph_MetaDef__operator_delete(reinterpret_cast<IndexedSubGraph_MetaDef*>(p)); }

  const std::string& name() const { return Provider_GetHost()->IndexedSubGraph_MetaDef__name(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  std::string& name() { return Provider_GetHost()->IndexedSubGraph_MetaDef__name(this); }
  const std::string& domain() const { return Provider_GetHost()->IndexedSubGraph_MetaDef__domain(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  std::string& domain() { return Provider_GetHost()->IndexedSubGraph_MetaDef__domain(this); }
  int since_version() const { return Provider_GetHost()->IndexedSubGraph_MetaDef__since_version(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  int& since_version() { return Provider_GetHost()->IndexedSubGraph_MetaDef__since_version(this); }

  ONNX_NAMESPACE::OperatorStatus& status() { return Provider_GetHost()->IndexedSubGraph_MetaDef__status(this); }

  const std::vector<std::string>& inputs() const { return Provider_GetHost()->IndexedSubGraph_MetaDef__inputs(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  std::vector<std::string>& inputs() { return Provider_GetHost()->IndexedSubGraph_MetaDef__inputs(this); }
  const std::vector<std::string>& outputs() const { return Provider_GetHost()->IndexedSubGraph_MetaDef__outputs(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  const std::vector<std::string>& constant_initializers() const { return Provider_GetHost()->IndexedSubGraph_MetaDef__constant_initializers(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  std::vector<std::string>& constant_initializers() { return Provider_GetHost()->IndexedSubGraph_MetaDef__constant_initializers(this); }
  std::vector<std::string>& outputs() { return Provider_GetHost()->IndexedSubGraph_MetaDef__outputs(this); }
  NodeAttributes& attributes() { return Provider_GetHost()->IndexedSubGraph_MetaDef__attributes(this); }

  std::string& doc_string() { return Provider_GetHost()->IndexedSubGraph_MetaDef__doc_string(this); }

  IndexedSubGraph_MetaDef() = delete;
  IndexedSubGraph_MetaDef(const IndexedSubGraph_MetaDef&) = delete;
  void operator=(const IndexedSubGraph_MetaDef&) = delete;
};

struct IndexedSubGraph final {
  static std::unique_ptr<IndexedSubGraph> Create() { return Provider_GetHost()->IndexedSubGraph__construct(); }
  static void operator delete(void* p) { Provider_GetHost()->IndexedSubGraph__operator_delete(reinterpret_cast<IndexedSubGraph*>(p)); }

  std::vector<onnxruntime::NodeIndex>& Nodes() { return Provider_GetHost()->IndexedSubGraph__Nodes(this); }

  void SetMetaDef(std::unique_ptr<IndexedSubGraph_MetaDef>&& meta_def_) { return Provider_GetHost()->IndexedSubGraph__SetMetaDef(this, std::move(*reinterpret_cast<std::unique_ptr<IndexedSubGraph_MetaDef>*>(&meta_def_))); }
  const IndexedSubGraph_MetaDef* GetMetaDef() const { return reinterpret_cast<const IndexedSubGraph_MetaDef*>(Provider_GetHost()->IndexedSubGraph__GetMetaDef(this)); }

  IndexedSubGraph() = delete;
  IndexedSubGraph(const IndexedSubGraph&) = delete;
  void operator=(const IndexedSubGraph&) = delete;
};

struct KernelDef final {
  static void operator delete(void* p) { Provider_GetHost()->KernelDef__operator_delete(reinterpret_cast<KernelDef*>(p)); }

  int ExecQueueId() const { return Provider_GetHost()->KernelDef__ExecQueueId(this); }

  void SinceVersion(/*out*/ int* start, /*out*/ int* end) const { Provider_GetHost()->KernelDef__SinceVersion(this, start, end); }
  const std::string& Domain() const { return Provider_GetHost()->KernelDef__Domain(this); }
  const std::string& OpName() const { return Provider_GetHost()->KernelDef__OpName(this); }

  KernelDef() = delete;
  KernelDef(const KernelDef*) = delete;
  void operator=(const KernelDef&) = delete;
};

using BuildKernelCreateInfoFn = KernelCreateInfo (*)();

struct KernelDefBuilder final {
  static std::unique_ptr<KernelDefBuilder> Create() { return Provider_GetHost()->KernelDefBuilder__construct(); }
  static void operator delete(void* p) { Provider_GetHost()->KernelDefBuilder__operator_delete(reinterpret_cast<KernelDefBuilder*>(p)); }

  KernelDefBuilder& SetName(const char* op_name) {
    Provider_GetHost()->KernelDefBuilder__SetName(this, op_name);
    return *this;
  }
  KernelDefBuilder& SetDomain(const char* domain) {
    Provider_GetHost()->KernelDefBuilder__SetDomain(this, domain);
    return *this;
  }
  KernelDefBuilder& SinceVersion(int since_version) {
    Provider_GetHost()->KernelDefBuilder__SinceVersion(this, since_version);
    return *this;
  }
  KernelDefBuilder& SinceVersion(int since_version_start, int since_version_end) {
    Provider_GetHost()->KernelDefBuilder__SinceVersion(this, since_version_start, since_version_end);
    return *this;
  }
  KernelDefBuilder& Provider(const char* provider_type) {
    Provider_GetHost()->KernelDefBuilder__Provider(this, provider_type);
    return *this;
  }
  KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType supported_type) {
    Provider_GetHost()->KernelDefBuilder__TypeConstraint(this, arg_name, supported_type);
    return *this;
  }
  KernelDefBuilder& TypeConstraint(const char* arg_name, const std::vector<MLDataType>& supported_types) {
    Provider_GetHost()->KernelDefBuilder__TypeConstraint(this, arg_name, supported_types);
    return *this;
  }
  KernelDefBuilder& InputMemoryType(OrtMemType type, int input_index) {
    Provider_GetHost()->KernelDefBuilder__InputMemoryType(this, type, input_index);
    return *this;
  }
  KernelDefBuilder& InputMemoryType(OrtMemType type, const std::vector<int>& input_indexes) {
    Provider_GetHost()->KernelDefBuilder__InputMemoryType(this, type, input_indexes);
    return *this;
  }
  KernelDefBuilder& OutputMemoryType(OrtMemType type, int input_index) {
    Provider_GetHost()->KernelDefBuilder__OutputMemoryType(this, type, input_index);
    return *this;
  }
  KernelDefBuilder& ExecQueueId(int queue_id) {
    Provider_GetHost()->KernelDefBuilder__ExecQueueId(this, queue_id);
    return *this;
  }
  KernelDefBuilder& MayInplace(int input_index, int output_index) {
    Provider_GetHost()->KernelDefBuilder__MayInplace(this, input_index, output_index);
    return *this;
  }
  KernelDefBuilder& Alias(const std::vector<std::pair<int, int>>& aliases) {
    Provider_GetHost()->KernelDefBuilder__Alias(this, aliases);
    return *this;
  }
  KernelDefBuilder& Alias(int input_index, int output_index) {
    Provider_GetHost()->KernelDefBuilder__Alias(this, input_index, output_index);
    return *this;
  }
  KernelDefBuilder& VariadicAlias(int input_offset, int output_offset) {
    Provider_GetHost()->KernelDefBuilder__VariadicAlias(this, input_offset, output_offset);
    return *this;
  }

  KernelDefBuilder& ExternalOutputs() {
    Provider_GetHost()->KernelDefBuilder__ExternalOutputs(this);
    return *this;
  }

  KernelDefBuilder& AllocateInputsContiguously() {
    Provider_GetHost()->KernelDefBuilder__AllocateInputsContiguously(this);
    return *this;
  }

#ifdef ENABLE_STRIDED_TENSORS
  KernelDefBuilder& MayStridedInput(int input_index) {
    Provider_GetHost()->KernelDefBuilder__MayStridedInput(this, input_index);
    return *this;
  }

  KernelDefBuilder& MayStridedOutput(int input_index, int output_index) {
    Provider_GetHost()->KernelDefBuilder__MayStridedOutput(this, input_index, output_index);
    return *this;
  }
#endif

  std::unique_ptr<KernelDef> Build() {
    return Provider_GetHost()->KernelDefBuilder__Build(this);
  }

  KernelDefBuilder() = delete;
  KernelDefBuilder(const KernelDefBuilder&) = delete;
  void operator=(const KernelDefBuilder&) = delete;
};

struct KernelRegistry final {
  static std::shared_ptr<KernelRegistry> Create() { return Provider_GetHost()->KernelRegistry__construct(); }
  static void operator delete(void* p) { Provider_GetHost()->KernelRegistry__operator_delete(reinterpret_cast<KernelRegistry*>(p)); }

  Status Register(KernelCreateInfo&& create_info) { return Provider_GetHost()->KernelRegistry__Register(this, std::move(create_info)); }

  KernelRegistry() = delete;
  KernelRegistry(const KernelRegistry&) = delete;
  void operator=(const KernelRegistry&) = delete;
};

struct PrimitiveDataTypeBase final {
  int32_t GetDataType() const { return Provider_GetHost()->PrimitiveDataTypeBase__GetDataType(this); }

  PROVIDER_DISALLOW_ALL(PrimitiveDataTypeBase)
};

class DataTypeImpl final {
 public:
  size_t Size() const { return Provider_GetHost()->DataTypeImpl__Size(this); }

  template <typename T>
  static MLDataType GetType();
  template <typename elemT>
  static MLDataType GetTensorType();
#if !defined(DISABLE_SPARSE_TENSORS)
  template <typename elemT>
  static MLDataType GetSparseTensorType();
#endif

  static MLDataType GetTypeFromOnnxType(int);

  bool IsTensorType() const { return Provider_GetHost()->DataTypeImpl__IsTensorType(this); }
  bool IsTensorSequenceType() const { return Provider_GetHost()->DataTypeImpl__IsTensorSequenceType(this); }
#if !defined(DISABLE_SPARSE_TENSORS)
  bool IsSparseTensorType() const { return Provider_GetHost()->DataTypeImpl__IsSparseTensorType(this); }
#endif
  DeleteFunc GetDeleteFunc() const { return Provider_GetHost()->DataTypeImpl__GetDeleteFunc(this); }

  static const std::vector<MLDataType>& AllFixedSizeTensorTypes() { return Provider_GetHost()->DataTypeImpl__AllFixedSizeTensorTypes(); }
  static const std::vector<MLDataType>& AllTensorTypes() { return Provider_GetHost()->DataTypeImpl__AllTensorTypes(); }
  static const std::vector<MLDataType>& AllIEEEFloatTensorTypes() { return Provider_GetHost()->DataTypeImpl__AllIEEEFloatTensorTypes(); }
  static const std::vector<MLDataType>& AllTensorAndSequenceTensorTypes() { return Provider_GetHost()->DataTypeImpl__AllTensorAndSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllFixedSizeTensorAndSequenceTensorTypes() { return Provider_GetHost()->DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllSequenceTensorTypes() { return Provider_GetHost()->DataTypeImpl__AllSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllFixedSizeSequenceTensorTypes() { return Provider_GetHost()->DataTypeImpl__AllFixedSizeSequenceTensorTypes(); }

  const PrimitiveDataTypeBase* AsPrimitiveDataType() const { return Provider_GetHost()->DataTypeImpl__AsPrimitiveDataType(this); }

  static const char* ToString(MLDataType type) { return Provider_GetHost()->DataTypeImpl__ToString(type); }

  PROVIDER_DISALLOW_ALL(DataTypeImpl)
};

struct Function final {
  const Graph& Body() const { return Provider_GetHost()->Function__Body(this); }

  PROVIDER_DISALLOW_ALL(Function)
};

struct Node final {
  const std::string& Name() const noexcept { return Provider_GetHost()->Node__Name(this); }
  const std::string& Description() const noexcept { return Provider_GetHost()->Node__Description(this); }
  const std::string& Domain() const noexcept { return Provider_GetHost()->Node__Domain(this); }
  const std::string& OpType() const noexcept { return Provider_GetHost()->Node__OpType(this); }

  int SinceVersion() const noexcept { return Provider_GetHost()->Node__SinceVersion(this); }

  const Function* GetFunctionBody() const noexcept { return Provider_GetHost()->Node__GetFunctionBody(this); }
  ProviderType GetExecutionProviderType() const noexcept { return Provider_GetHost()->Node__GetExecutionProviderType(this); }

  ConstPointerContainer<std::vector<NodeArg*>> ImplicitInputDefs() const noexcept { return Provider_GetHost()->Node__ImplicitInputDefs(this); }

  const std::vector<int>& InputArgCount() const noexcept { return Provider_GetHost()->Node__InputArgCount(this); }

  ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept { return Provider_GetHost()->Node__InputDefs(this); }
  ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept { return Provider_GetHost()->Node__OutputDefs(this); }
  NodeIndex Index() const noexcept { return Provider_GetHost()->Node__Index(this); }

  std::vector<gsl::not_null<const Graph*>> GetSubgraphs() const noexcept { return Provider_GetHost()->Node__GetSubgraphs(this); }

  void ToProto(ONNX_NAMESPACE::NodeProto& proto, bool update_subgraphs = false) const { return Provider_GetHost()->Node__ToProto(this, proto, update_subgraphs); }

  const NodeAttributes& GetAttributes() const noexcept { return Provider_GetHost()->Node__GetAttributes(this); }
  size_t GetInputEdgesCount() const noexcept { return Provider_GetHost()->Node__GetInputEdgesCount(this); }
  size_t GetOutputEdgesCount() const noexcept { return Provider_GetHost()->Node__GetOutputEdgesCount(this); }

  struct NodeConstIterator {
    NodeConstIterator(std::unique_ptr<Node__NodeIterator> p) : impl_{std::move(p)} {}

    bool operator!=(const NodeConstIterator& p_other) const { return *impl_ != *p_other.impl_; }

    void operator++() { impl_->operator++(); }

    const Node& operator*() const { return impl_->operator*(); }
    const Node* operator->() const { return &impl_->operator*(); }

    std::unique_ptr<Node__NodeIterator> impl_;
  };

  NodeConstIterator InputNodesBegin() const noexcept { return Provider_GetHost()->Node__InputNodesBegin(this); }
  NodeConstIterator InputNodesEnd() const noexcept { return Provider_GetHost()->Node__InputNodesEnd(this); }

  NodeConstIterator OutputNodesBegin() const noexcept { return Provider_GetHost()->Node__OutputNodesBegin(this); }
  NodeConstIterator OutputNodesEnd() const noexcept { return Provider_GetHost()->Node__OutputNodesEnd(this); }

  struct EdgeConstIterator {
    EdgeConstIterator(std::unique_ptr<Node__EdgeIterator> p) : impl_{std::move(p)} {}

    bool operator!=(const EdgeConstIterator& p_other) const {
      return *impl_ != *p_other.impl_;
    }

    void operator++() { impl_->operator++(); }
    const Node__EdgeIterator* operator->() const { return impl_.get(); }

    std::unique_ptr<Node__EdgeIterator> impl_;
  };

  EdgeConstIterator OutputEdgesBegin() const noexcept { return Provider_GetHost()->Node__OutputEdgesBegin(this); }
  EdgeConstIterator OutputEdgesEnd() const noexcept { return Provider_GetHost()->Node__OutputEdgesEnd(this); }

  void ForEachDef(std::function<void(const NodeArg&, bool is_input)> func, bool include_missing_optional_defs = false) const { Provider_GetHost()->Node__ForEachDef(this, func, std::move(include_missing_optional_defs)); }

  PROVIDER_DISALLOW_ALL(Node)
};

struct NodeArg final {
  const std::string& Name() const noexcept { return Provider_GetHost()->NodeArg__Name(this); }
  const ONNX_NAMESPACE::TensorShapeProto* Shape() const { return Provider_GetHost()->NodeArg__Shape(this); }
  ONNX_NAMESPACE::DataType Type() const noexcept { return Provider_GetHost()->NodeArg__Type(this); }
  const NodeArgInfo& ToProto() const noexcept { return Provider_GetHost()->NodeArg__ToProto(this); }
  bool Exists() const noexcept { return Provider_GetHost()->NodeArg__Exists(this); }
  const ONNX_NAMESPACE::TypeProto* TypeAsProto() const noexcept { return Provider_GetHost()->NodeArg__TypeAsProto(this); }

  PROVIDER_DISALLOW_ALL(NodeArg)
};

struct NodeAttributes final {
  static std::unique_ptr<NodeAttributes> Create() { return Provider_GetHost()->NodeAttributes__construct(); }
  void operator=(const NodeAttributes& v) { return Provider_GetHost()->NodeAttributes__operator_assign(this, v); }
  static void operator delete(void* p) { Provider_GetHost()->NodeAttributes__operator_delete(reinterpret_cast<NodeAttributes*>(p)); }

  size_t size() const { return Provider_GetHost()->NodeAttributes__size(this); }
  void clear() noexcept { Provider_GetHost()->NodeAttributes__clear(this); }
  size_t count(const std::string& keyval) const { return Provider_GetHost()->NodeAttributes__count(this, keyval); }
  ONNX_NAMESPACE::AttributeProto& operator[](const std::string& string) { return Provider_GetHost()->NodeAttributes__operator_array(this, string); }
  const ONNX_NAMESPACE::AttributeProto& at(const std::string& string) const { return Provider_GetHost()->NodeAttributes__at(this, string); }

  IteratorHolder<NodeAttributes_Iterator, std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>> begin() const { return Provider_GetHost()->NodeAttributes__begin(this); }
  IteratorHolder<NodeAttributes_Iterator, std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>> end() const { return Provider_GetHost()->NodeAttributes__end(this); }
  IteratorHolder<NodeAttributes_Iterator, std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>> find(const std::string& key) const { return Provider_GetHost()->NodeAttributes__find(this, key); }
  void insert(const NodeAttributes& v) { return Provider_GetHost()->NodeAttributes__insert(this, v); }
  void emplace(const std::string& k, const ONNX_NAMESPACE::AttributeProto& v) { Provider_GetHost()->NodeAttributes__emplace(this, k, v); }
  void reserve(size_t size) { Provider_GetHost()->NodeAttributes__reserve(this, size); }

  NodeAttributes() = delete;
  NodeAttributes(const NodeAttributes&) = delete;
};

struct Model final {
  static void operator delete(void* p) { Provider_GetHost()->Model__operator_delete(reinterpret_cast<Model*>(p)); }

  Graph& MainGraph() { return Provider_GetHost()->Model__MainGraph(this); }

  std::unique_ptr<ONNX_NAMESPACE::ModelProto> ToProto() { return Provider_GetHost()->Model__ToProto(this); }

  Model() = delete;
  Model(const Model&) = delete;
  void operator=(const Model&) = delete;
};

struct Graph final {
  std::unique_ptr<GraphViewer> CreateGraphViewer() const { return Provider_GetHost()->Graph__CreateGraphViewer(this); }
  std::unique_ptr<ONNX_NAMESPACE::GraphProto> ToGraphProto() const { return Provider_GetHost()->Graph__ToGraphProto(this); }

  NodeArg& GetOrCreateNodeArg(const std::string& name, const ONNX_NAMESPACE::TypeProto* p_arg_type) { return Provider_GetHost()->Graph__GetOrCreateNodeArg(this, name, p_arg_type); }

  Status Resolve() { return Provider_GetHost()->Graph__Resolve(this); }
  void AddInitializedTensor(const ONNX_NAMESPACE::TensorProto& tensor) { return Provider_GetHost()->Graph__AddInitializedTensor(this, tensor); }
  Node& AddNode(const std::string& name, const std::string& op_type, const std::string& description, gsl::span<NodeArg* const> input_args, gsl::span<NodeArg* const> output_args, const NodeAttributes* attributes, const std::string& domain) { return Provider_GetHost()->Graph__AddNode(this, name, op_type, description, input_args, output_args, attributes, domain); }

  const std::vector<const NodeArg*>& GetOutputs() const noexcept { return Provider_GetHost()->Graph__GetOutputs(this); }
  void SetOutputs(gsl::span<const NodeArg* const> outputs) { return Provider_GetHost()->Graph__SetOutputs(this, outputs); }

  const std::vector<const NodeArg*>& GetInputs() const noexcept { return Provider_GetHost()->Graph__GetInputs(this); }

  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const { return Provider_GetHost()->Graph__GetInitializedTensor(this, tensor_name, value); }

  const Node* ParentNode() const { return Provider_GetHost()->Graph__ParentNode(this); }
  const Graph* ParentGraph() const { return Provider_GetHost()->Graph__ParentGraph(this); }
  const std::string& Name() const noexcept { return Provider_GetHost()->Graph__Name(this); }
  const Path& ModelPath() const { return Provider_GetHost()->Graph__ModelPath(this); }
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept { return Provider_GetHost()->Graph__GetInputsIncludingInitializers(this); }
  bool IsSubgraph() const { return Provider_GetHost()->Graph__IsSubgraph(this); }

  PROVIDER_DISALLOW_ALL(Graph)
};

struct GraphViewer final {
  static void operator delete(void* p) { Provider_GetHost()->GraphViewer__operator_delete(reinterpret_cast<GraphViewer*>(p)); }

  std::unique_ptr<Model> CreateModel(const logging::Logger& logger) const { return Provider_GetHost()->GraphViewer__CreateModel(this, logger); }

  const std::string& Name() const noexcept { return Provider_GetHost()->GraphViewer__Name(this); }
  const Path& ModelPath() const noexcept { return Provider_GetHost()->GraphViewer__ModelPath(this); }

  const Node* GetNode(NodeIndex node_index) const { return Provider_GetHost()->GraphViewer__GetNode(this, node_index); }
  const NodeArg* GetNodeArg(const std::string& name) const { return Provider_GetHost()->GraphViewer__GetNodeArg(this, name); }

  bool IsSubgraph() const { return Provider_GetHost()->GraphViewer__IsSubgraph(this); }
  const Graph& GetGraph() const { return Provider_GetHost()->GraphViewer__GetGraph(this); }
  bool IsConstantInitializer(const std::string& name, bool check_outer_scope) const { return Provider_GetHost()->GraphViewer__IsConstantInitializer(this, name, check_outer_scope); }
  const Node* ParentNode() const { return Provider_GetHost()->GraphViewer__ParentNode(this); }

  int NumberOfNodes() const noexcept { return Provider_GetHost()->GraphViewer__NumberOfNodes(this); }
  int MaxNodeIndex() const noexcept { return Provider_GetHost()->GraphViewer__MaxNodeIndex(this); }

  const std::vector<const NodeArg*>& GetInputs() const noexcept { return Provider_GetHost()->GraphViewer__GetInputs(this); }
  const std::vector<const NodeArg*>& GetOutputs() const noexcept { return Provider_GetHost()->GraphViewer__GetOutputs(this); }
  const std::unordered_set<const NodeArg*>& GetValueInfo() const noexcept { return Provider_GetHost()->GraphViewer__GetValueInfo(this); }

  const InitializedTensorSet& GetAllInitializedTensors() const noexcept { return Provider_GetHost()->GraphViewer__GetAllInitializedTensors(this); }
  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const { return Provider_GetHost()->GraphViewer__GetInitializedTensor(this, tensor_name, value); }

  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept { return Provider_GetHost()->GraphViewer__DomainToVersionMap(this); }

  const std::vector<NodeIndex>& GetNodesInTopologicalOrder() const { return Provider_GetHost()->GraphViewer__GetNodesInTopologicalOrder(this); }
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept { return Provider_GetHost()->GraphViewer__GetInputsIncludingInitializers(this); }

  void ToProto(ONNX_NAMESPACE::GraphProto& graph_proto, bool include_initializers, bool include_outer_scope_args) const { Provider_GetHost()->GraphViewer__ToProto(this, graph_proto, include_initializers, include_outer_scope_args); }

  GraphViewer() = delete;
  GraphViewer(const GraphViewer&) = delete;
  void operator=(const GraphViewer&) = delete;
};

struct Path final {
  PathString ToPathString() const noexcept { return Provider_GetHost()->Path__ToPathString(this); }
  const std::vector<PathString>& GetComponents() const noexcept { return Provider_GetHost()->Path__GetComponents(this); }
  bool IsEmpty() const noexcept { return Provider_GetHost()->Path__IsEmpty(this); }

  PROVIDER_DISALLOW_ALL(Path)
};

struct OpKernelContext final {
  template <typename T>
  const T& RequiredInput(int index) const;
  Tensor& RequiredOutput(int index, const TensorShape& shape) { return Provider_GetHost()->OpKernelContext__RequiredOutput(this, index, shape); }

  template <typename T>
  const T* Input(int index) const;
  int InputCount() const { return Provider_GetHost()->OpKernelContext__InputCount(this); }

  MLDataType InputType(int index) const { return Provider_GetHost()->OpKernelContext__InputType(this, index); }

  template <typename T>
  T* Output(int index);

  Tensor* Output(int index, const TensorShape& shape) { return Provider_GetHost()->OpKernelContext__Output(this, index, shape); }
#if !defined(DISABLE_SPARSE_TENSORS)
  SparseTensor* OutputSparse(int index, const TensorShape& shape) { return Provider_GetHost()->OpKernelContext__OutputSparse(this, index, shape); }
#endif
  int OutputCount() const { return Provider_GetHost()->OpKernelContext__OutputCount(this); }

  Status GetTempSpaceAllocator(AllocatorPtr* output) const { return Provider_GetHost()->OpKernelContext__GetTempSpaceAllocator(this, output); }

  Status GetTempSpaceCPUAllocator(AllocatorPtr* output) const { return Provider_GetHost()->OpKernelContext__GetTempSpaceCPUAllocator(this, output); }

  bool GetUseDeterministicCompute() const { return Provider_GetHost()->OpKernelContext__GetUseDeterministicCompute(this); }

  bool TryGetInferredOutputShape(int index, TensorShape& shape) const { return Provider_GetHost()->OpKernelContext__TryGetInferredOutputShape(this, index, shape); }
  bool TryGetInferredInputShape(int index, TensorShape& shape) const { return Provider_GetHost()->OpKernelContext__TryGetInferredInputShape(this, index, shape); }
  Stream* GetComputeStream() const { return Provider_GetHost()->OpKernelContext__GetComputeStream(this); }

  PROVIDER_DISALLOW_ALL(OpKernelContext)
};

template <>
inline const Tensor* OpKernelContext::Input<Tensor>(int index) const {
  return Provider_GetHost()->OpKernelContext__Input_Tensor(this, index);
}

#if !defined(DISABLE_SPARSE_TENSORS)
template <>
inline const SparseTensor* OpKernelContext::Input<SparseTensor>(int index) const {
  return Provider_GetHost()->OpKernelContext__Input_SparseTensor(this, index);
}
#endif

template <>
inline const TensorSeq* OpKernelContext::Input<TensorSeq>(int index) const {
  return Provider_GetHost()->OpKernelContext__Input_TensorSeq(this, index);
}

template <>
inline Tensor* OpKernelContext::Output<Tensor>(int index) {
  return Provider_GetHost()->OpKernelContext__Output_Tensor(this, index);
}

template <>
inline TensorSeq* OpKernelContext::Output<TensorSeq>(int index) {
  return Provider_GetHost()->OpKernelContext__Output_TensorSeq(this, index);
}

template <>
inline const Tensor& OpKernelContext::RequiredInput(int index) const {
  return Provider_GetHost()->OpKernelContext__RequiredInput_Tensor(this, index);
}

struct OpKernelInfo final {
  static void operator delete(void* p) { Provider_GetHost()->OpKernelInfo__operator_delete(reinterpret_cast<OpKernelInfo*>(p)); }

  AllocatorPtr GetAllocator(OrtMemType mem_type) const { return Provider_GetHost()->OpKernelInfo__GetAllocator(this, mem_type); }

  const IExecutionProvider* GetExecutionProvider() const noexcept { return Provider_GetHost()->OpKernelInfo__GetExecutionProvider(this); }

  template <typename T>
  Status GetAttr(const std::string& name, T* value) const;

  template <typename T>
  Status GetAttrs(const std::string& name, std::vector<T>& values) const;

  template <typename T>
  T GetAttrOrDefault(const std::string& name, const T& default_value) const {
    T tmp;
    return GetAttr<T>(name, &tmp).IsOK() ? tmp : default_value;
  }

  template <typename T>
  void GetAttrOrDefault(const std::string& name, T* value, const T& default_value) const {
    if (!GetAttr<T>(name, value).IsOK())
      *value = default_value;
  }

  template <typename T>
  std::vector<T> GetAttrsOrDefault(const std::string& name, const std::vector<T>& default_value = std::vector<T>{}) const {
    std::vector<T> tmp;
    return GetAttrs<T>(name, tmp).IsOK() ? tmp : default_value;
  }

  template <typename T>
  Status GetAttrsAsSpan(const std::string& name, gsl::span<const T>& out) const;

  Status GetAttrs(const std::string& name, TensorShapeVector& out) const;

  TensorShapeVector GetAttrsOrDefault(const std::string& name, const TensorShapeVector& default_value = TensorShapeVector{}) const;

  bool TryGetConstantInput(int input_index, const Tensor** constant_input_value) const { return Provider_GetHost()->OpKernelInfo__TryGetConstantInput(this, input_index, constant_input_value); }

  const DataTransferManager& GetDataTransferManager() const noexcept { return Provider_GetHost()->OpKernelInfo__GetDataTransferManager(this); }
  const KernelDef& GetKernelDef() const { return Provider_GetHost()->OpKernelInfo__GetKernelDef(this); }

  uint32_t GetInputCount() const { return Provider_GetHost()->OpKernelInfo__GetInputCount(this); }
  uint32_t GetOutputCount() const { return Provider_GetHost()->OpKernelInfo__GetOutputCount(this); }

  const Node& node() const noexcept { return Provider_GetHost()->OpKernelInfo__node(this); }

  OpKernelInfo() = delete;
  OpKernelInfo(const OpKernelInfo&) = delete;
  void operator=(const OpKernelInfo&) = delete;
};

template <>
inline Status OpKernelInfo::GetAttr<int64_t>(const std::string& name, int64_t* value) const { return Provider_GetHost()->OpKernelInfo__GetAttr_int64(this, name, value); }
template <>
inline Status OpKernelInfo::GetAttr<float>(const std::string& name, float* value) const { return Provider_GetHost()->OpKernelInfo__GetAttr_float(this, name, value); }
template <>
inline Status OpKernelInfo::GetAttr<std::string>(const std::string& name, std::string* value) const { return Provider_GetHost()->OpKernelInfo__GetAttr_string(this, name, value); }
template <>
inline Status OpKernelInfo::GetAttr<ONNX_NAMESPACE::TensorProto>(const std::string& name, ONNX_NAMESPACE::TensorProto* value) const { return Provider_GetHost()->OpKernelInfo__GetAttr_TensorProto(this, name, value); }
template <>
inline Status OpKernelInfo::GetAttrs<int64_t>(const std::string& name, std::vector<int64_t>& values) const { return Provider_GetHost()->OpKernelInfo__GetAttrs(this, name, values); }
template <>
inline Status OpKernelInfo::GetAttrs<float>(const std::string& name, std::vector<float>& values) const { return Provider_GetHost()->OpKernelInfo__GetAttrs(this, name, values); }
template <>
inline Status OpKernelInfo::GetAttrs<std::string>(const std::string& name, std::vector<std::string>& values) const { return Provider_GetHost()->OpKernelInfo__GetAttrs(this, name, values); }
template <>
inline Status OpKernelInfo::GetAttrsAsSpan<int64_t>(const std::string& name, gsl::span<const int64_t>& values) const { return Provider_GetHost()->OpKernelInfo__GetAttrsAsSpan(this, name, values); }

inline Status OpKernelInfo::GetAttrs(const std::string& name, TensorShapeVector& out) const {
  gsl::span<const int64_t> span;
  Status status = this->GetAttrsAsSpan<int64_t>(name, span);
  if (status.IsOK()) {
    out.reserve(span.size());
    out.assign(span.begin(), span.end());
  }
  return status;
}

inline TensorShapeVector OpKernelInfo::GetAttrsOrDefault(const std::string& name, const TensorShapeVector& default_value) const {
  TensorShapeVector tmp;
  return GetAttrs(name, tmp).IsOK() ? tmp : default_value;
}

class SessionState {
 public:
  const DataTransferManager& GetDataTransferMgr() const noexcept { return Provider_GetHost()->SessionState__GetDataTransferMgr(this); }

  PROVIDER_DISALLOW_ALL(SessionState)
};

struct Tensor final {
  static std::unique_ptr<Tensor> CreateDefault() { return Provider_GetHost()->Tensor__construct_default(); }
  static std::unique_ptr<Tensor> Create(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator) { return Provider_GetHost()->Tensor__construct(p_type, shape, std::move(allocator)); }
  static std::unique_ptr<Tensor> Create(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc, ptrdiff_t offset = 0) { return Provider_GetHost()->Tensor__construct(p_type, shape, p_data, alloc, offset); }

  static void operator delete(void* p) noexcept { Provider_GetHost()->Tensor__operator_delete(reinterpret_cast<Tensor*>(p)); }

  static void InitOrtValue(MLDataType elt_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator, OrtValue& ort_value) {
    Provider_GetHost()->Tensor__InitOrtValue(elt_type, shape, std::move(allocator), ort_value);
  }

  static void InitOrtValue(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& location, OrtValue& ort_value) {
    Provider_GetHost()->Tensor__InitOrtValue(p_type, shape, p_data, location, ort_value);
  }

  template <typename T>
  T* MutableData();

  template <typename T>
  const T* Data() const;

  template <typename T>
  gsl::span<const T> DataAsSpan() const;

  void* MutableDataRaw(MLDataType type) { return Provider_GetHost()->Tensor__MutableDataRaw(this, type); }
  const void* DataRaw(MLDataType type) const { return Provider_GetHost()->Tensor__DataRaw(this, type); }

  void* MutableDataRaw() noexcept { return Provider_GetHost()->Tensor__MutableDataRaw(this); }
  const void* DataRaw() const noexcept { return Provider_GetHost()->Tensor__DataRaw(this); }

  const TensorShape& Shape() const { return Provider_GetHost()->Tensor__Shape(this); }
  void Reshape(const TensorShape& new_shape) { Provider_GetHost()->Tensor__Reshape(this, new_shape); }
  void SetByteOffset(ptrdiff_t byte_offset) { return Provider_GetHost()->Tensor__SetByteOffset(this, byte_offset); }
  ptrdiff_t ByteOffset() const { return Provider_GetHost()->Tensor__ByteOffset(this); }
  size_t SizeInBytes() const { return Provider_GetHost()->Tensor__SizeInBytes(this); }
  const OrtMemoryInfo& Location() const { return Provider_GetHost()->Tensor__Location(this); }

  int32_t GetElementType() const { return Provider_GetHost()->Tensor__GetElementType(this); }
  MLDataType DataType() const { return Provider_GetHost()->Tensor__DataType(this); }
  bool IsDataTypeString() const { return Provider_GetHost()->Tensor__IsDataTypeString(this); }

#ifdef ENABLE_STRIDED_TENSORS
  gsl::span<const int64_t> Strides() const noexcept { return Provider_GetHost()->Tensor__Strides(this); }
  bool IsContiguous() const { return Provider_GetHost()->Tensor__IsContiguous(this); }
  void SetShapeAndStrides(const TensorShape& new_shape, gsl::span<const int64_t> new_strides) {
    return Provider_GetHost()->Tensor__SetShapeAndStrides(this, new_shape, new_strides);
  }
#endif

  template <class T>
  bool IsDataType() const;

  Tensor() = delete;
  Tensor(const Tensor&) = delete;
  void operator=(const Tensor&) = delete;
  Tensor& operator=(Tensor&& o) noexcept {
    Provider_GetHost()->Tensor__move_assign(*this, std::move(o));
    return *this;
  }
};

template <>
inline bool Tensor::IsDataType<bool>() const { return Provider_GetHost()->Tensor__IsDataType_bool(this); }
template <>
inline bool Tensor::IsDataType<int8_t>() const { return Provider_GetHost()->Tensor__IsDataType_int8(this); }
template <>
inline bool Tensor::IsDataType<uint8_t>() const { return Provider_GetHost()->Tensor__IsDataType_uint8(this); }
template <>
inline bool Tensor::IsDataType<int16_t>() const { return Provider_GetHost()->Tensor__IsDataType_int16(this); }
template <>
inline bool Tensor::IsDataType<uint16_t>() const { return Provider_GetHost()->Tensor__IsDataType_uint16(this); }
template <>
inline bool Tensor::IsDataType<int32_t>() const { return Provider_GetHost()->Tensor__IsDataType_int32(this); }
template <>
inline bool Tensor::IsDataType<uint32_t>() const { return Provider_GetHost()->Tensor__IsDataType_uint32(this); }
template <>
inline bool Tensor::IsDataType<int64_t>() const { return Provider_GetHost()->Tensor__IsDataType_int64(this); }
template <>
inline bool Tensor::IsDataType<uint64_t>() const { return Provider_GetHost()->Tensor__IsDataType_uint64(this); }
template <>
inline bool Tensor::IsDataType<float>() const { return Provider_GetHost()->Tensor__IsDataType_float(this); }
template <>
inline bool Tensor::IsDataType<double>() const { return Provider_GetHost()->Tensor__IsDataType_double(this); }
template <>
inline bool Tensor::IsDataType<MLFloat16>() const { return Provider_GetHost()->Tensor__IsDataType_MLFloat16(this); }
template <>
inline bool Tensor::IsDataType<BFloat16>() const { return Provider_GetHost()->Tensor__IsDataType_BFloat16(this); }

template <>
inline bool* Tensor::MutableData<bool>() { return Provider_GetHost()->Tensor__MutableData_bool(this); }
template <>
inline int8_t* Tensor::MutableData<int8_t>() { return Provider_GetHost()->Tensor__MutableData_int8(this); }
template <>
inline uint8_t* Tensor::MutableData<uint8_t>() { return Provider_GetHost()->Tensor__MutableData_uint8(this); }
template <>
inline int16_t* Tensor::MutableData<int16_t>() { return Provider_GetHost()->Tensor__MutableData_int16(this); }
template <>
inline uint16_t* Tensor::MutableData<uint16_t>() { return Provider_GetHost()->Tensor__MutableData_uint16(this); }
template <>
inline int32_t* Tensor::MutableData<int32_t>() { return Provider_GetHost()->Tensor__MutableData_int32(this); }
template <>
inline uint32_t* Tensor::MutableData<uint32_t>() { return Provider_GetHost()->Tensor__MutableData_uint32(this); }
template <>
inline int64_t* Tensor::MutableData<int64_t>() { return Provider_GetHost()->Tensor__MutableData_int64(this); }
template <>
inline uint64_t* Tensor::MutableData<uint64_t>() { return Provider_GetHost()->Tensor__MutableData_uint64(this); }
template <>
inline float* Tensor::MutableData<float>() { return Provider_GetHost()->Tensor__MutableData_float(this); }
template <>
inline double* Tensor::MutableData<double>() { return Provider_GetHost()->Tensor__MutableData_double(this); }
template <>
inline BFloat16* Tensor::MutableData<BFloat16>() { return Provider_GetHost()->Tensor__MutableData_BFloat16(this); }
template <>
inline MLFloat16* Tensor::MutableData<MLFloat16>() { return Provider_GetHost()->Tensor__MutableData_MLFloat16(this); }

template <>
inline const bool* Tensor::Data<bool>() const { return Provider_GetHost()->Tensor__Data_bool(this); }
template <>
inline const int8_t* Tensor::Data<int8_t>() const { return Provider_GetHost()->Tensor__Data_int8(this); }
template <>
inline const uint8_t* Tensor::Data<uint8_t>() const { return Provider_GetHost()->Tensor__Data_uint8(this); }
template <>
inline const int16_t* Tensor::Data<int16_t>() const { return Provider_GetHost()->Tensor__Data_int16(this); }
template <>
inline const uint16_t* Tensor::Data<uint16_t>() const { return Provider_GetHost()->Tensor__Data_uint16(this); }
template <>
inline const int32_t* Tensor::Data<int32_t>() const { return Provider_GetHost()->Tensor__Data_int32(this); }
template <>
inline const uint32_t* Tensor::Data<uint32_t>() const { return Provider_GetHost()->Tensor__Data_uint32(this); }
template <>
inline const int64_t* Tensor::Data<int64_t>() const { return Provider_GetHost()->Tensor__Data_int64(this); }
template <>
inline const uint64_t* Tensor::Data<uint64_t>() const { return Provider_GetHost()->Tensor__Data_uint64(this); }
template <>
inline const float* Tensor::Data<float>() const { return Provider_GetHost()->Tensor__Data_float(this); }
template <>
inline const double* Tensor::Data<double>() const { return Provider_GetHost()->Tensor__Data_double(this); }
template <>
inline const BFloat16* Tensor::Data<BFloat16>() const { return Provider_GetHost()->Tensor__Data_BFloat16(this); }
template <>
inline const MLFloat16* Tensor::Data<MLFloat16>() const { return Provider_GetHost()->Tensor__Data_MLFloat16(this); }

// SparseTensor
#if !defined(DISABLE_SPARSE_TENSORS)
struct SparseTensor final {
  const TensorShape& DenseShape() const noexcept { return Provider_GetHost()->SparseTensor__DenseShape(this); }
  Status Copy(const DataTransferManager& dtm, SparseTensor& dst) const { return Provider_GetHost()->SparseTensor__Copy(this, dtm, dst); }
};
#endif

// TensorSeq
class TensorSeq final {
public:
  MLDataType DataType() const noexcept { return Provider_GetHost()->TensorSeq__DataType(this); }
  void SetType(MLDataType elem_type) { Provider_GetHost()->TensorSeq__SetType(this, elem_type); }
  size_t Size() const noexcept { return Provider_GetHost()->TensorSeq__Size(this); }
  const Tensor& Get(size_t i) const { return Provider_GetHost()->TensorSeq__Get(this, i); }
  const OrtValue& GetAt(size_t i) const { return Provider_GetHost()->TensorSeq__GetAt(this, i); }
  void Add(const OrtValue& tensor) { Provider_GetHost()->TensorSeq__Add(this, tensor); }
  void Add(OrtValue&& tensor) { Provider_GetHost()->TensorSeq__Add(this, std::move(tensor)); }
  void Add(Tensor&& tensor) { Provider_GetHost()->TensorSeq__Add(this, std::move(tensor)); }
  void Reserve(size_t capacity) { Provider_GetHost()->TensorSeq__Reserve(this, capacity); }
};

template <>
inline gsl::span<const int64_t> Tensor::DataAsSpan() const { return Provider_GetHost()->Tensor__DataAsSpan_int64(this); }

}  // namespace onnxruntime
