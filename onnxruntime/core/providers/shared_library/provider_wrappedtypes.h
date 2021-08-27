// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {

extern ProviderHost* g_host;

struct CPUIDInfo final {
  static const CPUIDInfo& GetCPUIDInfo() { return g_host->CPUIDInfo__GetCPUIDInfo(); }

  bool HasAVX2() const { return g_host->CPUIDInfo__HasAVX2(this); }
  bool HasAVX512f() const { return g_host->CPUIDInfo__HasAVX512f(this); }

  PROVIDER_DISALLOW_ALL(CPUIDInfo)
};

namespace logging {

struct Logger final {
  bool OutputIsEnabled(Severity severity, DataType data_type) const noexcept { return g_host->logging__Logger__OutputIsEnabled(this, severity, data_type); }

  PROVIDER_DISALLOW_ALL(Logger)
};

struct LoggingManager final {
  static const Logger& DefaultLogger() { return g_host->logging__LoggingManager__DefaultLogger(); }

  PROVIDER_DISALLOW_ALL(LoggingManager)
};

struct Capture final {
  static std::unique_ptr<Capture> Create(const Logger& logger, logging::Severity severity, const char* category,
                                         logging::DataType dataType, const CodeLocation& location) { return g_host->logging__Capture__construct(logger, severity, category, dataType, location); }
  static void operator delete(void* p) { g_host->logging__Capture__operator_delete(reinterpret_cast<Capture*>(p)); }

  std::ostream& Stream() noexcept { return g_host->logging__Capture__Stream(this); }

  Capture() = delete;
  Capture(const Capture&) = delete;
  void operator=(const Capture&) = delete;
};
}  // namespace logging
}  // namespace onnxruntime

namespace ONNX_NAMESPACE {

struct int64s final {
  int size() const { return g_host->int64s__size(this); }
  const int64_t& Get(int index) const { return g_host->int64s__Get(this, index); }
  const int64_t& operator[](int index) const { return Get(index); }

  PROVIDER_DISALLOW_ALL(int64s)
};

struct AttributeProto final {
  static std::unique_ptr<AttributeProto> Create() { return g_host->AttributeProto__construct(); }
  void operator=(const AttributeProto& v) { g_host->AttributeProto__operator_assign(this, v); }
  static void operator delete(void* p) { g_host->AttributeProto__operator_delete(reinterpret_cast<AttributeProto*>(p)); }

  AttributeProto_AttributeType type() const { return g_host->AttributeProto__type(this); }
  int ints_size() const { return g_host->AttributeProto__ints_size(this); }
  int floats_size() const { return g_host->AttributeProto__floats_size(this); }
  int strings_size() const { return g_host->AttributeProto__strings_size(this); }
  int64_t ints(int i) const { return g_host->AttributeProto__ints(this, i); }
  float floats(int i) const { return g_host->AttributeProto__floats(this, i); }
  const std::string& strings(int i) const { return g_host->AttributeProto__strings(this, i); }
  const int64s& ints() const { return g_host->AttributeProto__ints(this); }
  int64_t i() const { return g_host->AttributeProto__i(this); }
  float f() const { return g_host->AttributeProto__f(this); }
  void set_s(const ::std::string& value) { return g_host->AttributeProto__set_s(this, value); }
  const ::std::string& s() const { return g_host->AttributeProto__s(this); }
  void set_name(const ::std::string& value) { return g_host->AttributeProto__set_name(this, value); }
  void set_type(AttributeProto_AttributeType value) { return g_host->AttributeProto__set_type(this, value); }
  TensorProto* add_tensors() { return g_host->AttributeProto__add_tensors(this); }

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
  static void operator delete(void* p) { g_host->GraphProto__operator_delete(reinterpret_cast<GraphProto*>(p)); }
  void operator=(const GraphProto& v) { return g_host->GraphProto__operator_assign(this, v); }

  const ValueInfoProto& input(int index) const { return g_host->GraphProto__input(this, index); }
  ValueInfoProtos* mutable_input() { return g_host->GraphProto__mutable_input(this); }
  ValueInfoProto* mutable_input(int index) { return g_host->GraphProto__mutable_input(this, index); }
  int input_size() const { return g_host->GraphProto__input_size(this); }

  const ValueInfoProtos& output() const { return g_host->GraphProto__output(this); }
  const ValueInfoProto& output(int index) const { return g_host->GraphProto__output(this, index); }
  ValueInfoProtos* mutable_output() { return g_host->GraphProto__mutable_output(this); }

  ValueInfoProtos* mutable_value_info() { return g_host->GraphProto__mutable_value_info(this); }
  TensorProtos* mutable_initializer() { return g_host->GraphProto__mutable_initializer(this); }
  NodeProto* add_node() { return g_host->GraphProto__add_node(this); }

  GraphProto() = delete;
  GraphProto(const GraphProto&) = delete;
};

struct ModelProto final {
  static std::unique_ptr<ModelProto> Create() { return g_host->ModelProto__construct(); }
  static void operator delete(void* p) { g_host->ModelProto__operator_delete(reinterpret_cast<ModelProto*>(p)); }

  bool SerializeToString(std::string& string) const { return g_host->ModelProto__SerializeToString(this, string); }
  bool SerializeToOstream(std::ostream& output) const { return g_host->ModelProto__SerializeToOstream(this, output); }
  bool ParseFromString(const std::string& data) { return g_host->ModelProto__ParseFromString(this, data); }
  std::string SerializeAsString() const { return g_host->ModelProto__SerializeAsString(this); }

  const GraphProto& graph() const { return g_host->ModelProto__graph(this); }
  GraphProto* mutable_graph() { return g_host->ModelProto__mutable_graph(this); }

  void set_ir_version(int64_t value) { return g_host->ModelProto__set_ir_version(this, value); }

  ModelProto() = delete;
  ModelProto(const ModelProto&) = delete;
  void operator=(const ModelProto&) = delete;
};

struct TensorProto final {
  static std::unique_ptr<TensorProto> Create() { return g_host->TensorProto__construct(); }
  static void operator delete(void* p) { g_host->TensorProto__operator_delete(reinterpret_cast<TensorProto*>(p)); }
  void operator=(const TensorProto& v) { g_host->TensorProto__operator_assign(this, v); }

  bool has_name() const { return g_host->TensorProto__has_name(this); }

  int dims_size() const { return g_host->TensorProto__dims_size(this); }
  const int64s& dims() const { return g_host->TensorProto__dims(this); }

  bool has_data_location() const { return g_host->TensorProto__has_data_location(this); }
  TensorProto_DataLocation data_location() const { return TensorProto_DataLocation(g_host->TensorProto__data_location(this)); }

  bool has_raw_data() const { return g_host->TensorProto__has_raw_data(this); }
  const std::string& raw_data() const { return g_host->TensorProto__raw_data(this); }

  int32_t data_type() const { return g_host->TensorProto__data_type(this); }

  typedef TensorProto_DataType DataType;
  static constexpr DataType UNDEFINED = TensorProto_DataType_UNDEFINED;

  static bool DataType_IsValid(int value) { return g_host->TensorProto_DataType_IsValid(value); }

  TensorProto() = delete;
  TensorProto(const TensorProto&) = delete;
};

struct TensorProtos final {
  TensorProto* Add() { return g_host->TensorProtos__Add(this); }

  PROVIDER_DISALLOW_ALL(TensorProtos)
};

struct TensorShapeProto_Dimension final {
  enum ValueCase {
    kDimValue = 1,
    kDimParam = 2,
    VALUE_NOT_SET = 0,
  };

  ValueCase value_case() const { return ValueCase(g_host->TensorShapeProto_Dimension__value_case(this)); }
  const std::string& dim_param() const { return g_host->TensorShapeProto_Dimension__dim_param(this); }
  int64_t dim_value() const { return g_host->TensorShapeProto_Dimension__dim_value(this); }
  void set_dim_value(int64_t value) { return g_host->TensorShapeProto_Dimension__set_dim_value(this, value); }
  bool has_dim_value() const { return g_host->TensorShapeProto_Dimension__has_dim_value(this); }
  bool has_dim_param() const { return g_host->TensorShapeProto_Dimension__has_dim_param(this); }
  void clear_dim_value() { return g_host->TensorShapeProto_Dimension__clear_dim_value(this); }

  PROVIDER_DISALLOW_ALL(TensorShapeProto_Dimension)
};

struct TensorShapeProto_Dimensions final {
  IteratorHolder<TensorShapeProto_Dimension_Iterator, const TensorShapeProto_Dimension> begin() const { return g_host->TensorShapeProto_Dimensions__begin(this); }
  IteratorHolder<TensorShapeProto_Dimension_Iterator, const TensorShapeProto_Dimension> end() const { return g_host->TensorShapeProto_Dimensions__end(this); }

  PROVIDER_DISALLOW_ALL(TensorShapeProto_Dimensions)
};

struct TensorShapeProto final {
  int dim_size() const { return g_host->TensorShapeProto__dim_size(this); }
  const TensorShapeProto_Dimensions& dim() const { return g_host->TensorShapeProto__dim(this); }
  const TensorShapeProto_Dimension& dim(int index) const { return g_host->TensorShapeProto__dim(this, index); }
  TensorShapeProto_Dimension* mutable_dim(int index) { return g_host->TensorShapeProto__mutable_dim(this, index); }
  void clear_dim() { return g_host->TensorShapeProto__clear_dim(this); }
  TensorShapeProto_Dimension* add_dim() { return g_host->TensorShapeProto__add_dim(this); }

  PROVIDER_DISALLOW_ALL(TensorShapeProto)
};

struct TypeProto_Tensor final {
  bool has_shape() const { return g_host->TypeProto_Tensor__has_shape(this); }
  const TensorShapeProto& shape() const { return g_host->TypeProto_Tensor__shape(this); }
  TensorShapeProto* mutable_shape() { return g_host->TypeProto_Tensor__mutable_shape(this); }
  int32_t elem_type() const { return g_host->TypeProto_Tensor__elem_type(this); }

  PROVIDER_DISALLOW_ALL(TypeProto_Tensor)
};

#if !defined(DISABLE_SPARSE_TENSORS)
struct TypeProto_SparseTensor final {
  bool has_shape() const { return g_host->TypeProto_SparseTensor__has_shape(this); }
  const TensorShapeProto& shape() const { return g_host->TypeProto_SparseTensor__shape(this); }
  TensorShapeProto* mutable_shape() { return g_host->TypeProto_SparseTensor__mutable_shape(this); }
  int32_t elem_type() const { return g_host->TypeProto_SparseTensor__elem_type(this); }

  PROVIDER_DISALLOW_ALL(TypeProto_SparseTensor)
};
#endif

struct TypeProto final {
  const TypeProto_Tensor& tensor_type() const { return g_host->TypeProto__tensor_type(this); }
  TypeProto_Tensor* mutable_tensor_type() { return g_host->TypeProto__mutable_tensor_type(this); }

#if !defined(DISABLE_SPARSE_TENSORS)
  const TypeProto_SparseTensor& sparse_tensor_type() const { return g_host->TypeProto__sparse_tensor_type(this); }
  TypeProto_SparseTensor* mutable_sparse_tensor_type() { return g_host->TypeProto__mutable_sparse_tensor_type(this); }
#endif

  enum ValueCase {
    kTensorType = 1,
    kSequenceType = 4,
    kMapType = 5,
    kSparseTensorType = 8,
    kOpaqueType = 7,
    VALUE_NOT_SET = 0,
  };

  ValueCase value_case() const { return ValueCase(g_host->TypeProto__value_case(this)); }

  PROVIDER_DISALLOW_ALL(TypeProto)
};

struct ValueInfoProto final {
  const TypeProto& type() const { return g_host->ValueInfoProto__type(this); }
  TypeProto* mutable_type() { return g_host->ValueInfoProto__mutable_type(this); }

  void operator=(const ValueInfoProto& v) { g_host->ValueInfoProto__operator_assign(this, v); }

  ValueInfoProto() = delete;
  ValueInfoProto(const ValueInfoProto&) = delete;
  static void operator delete(void*) = delete;
};

struct ValueInfoProtos final {
  ValueInfoProto* Add() { return g_host->ValueInfoProtos__Add(this); }
  const ValueInfoProto& operator[](int index) const { return g_host->ValueInfoProtos__operator_array(this, index); }

  PROVIDER_DISALLOW_ALL(ValueInfoProtos)
};

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

namespace utils {
bool IsDataTypeString(MLDataType dt_type);

}  // namespace utils

namespace Utils {

struct DataTypeUtils final {
  static const std::string* ToType(const ONNX_NAMESPACE::TypeProto& type_proto) { return g_host->Utils__DataTypeUtils__ToType(type_proto); }

  PROVIDER_DISALLOW_ALL(DataTypeUtils)
};

}  // namespace Utils

struct ComputeCapability final {
  static std::unique_ptr<ComputeCapability> Create(std::unique_ptr<IndexedSubGraph> t_sub_graph) { return g_host->ComputeCapability__construct(std::move(t_sub_graph)); }
  static void operator delete(void* p) { g_host->ComputeCapability__operator_delete(reinterpret_cast<ComputeCapability*>(p)); }

  std::unique_ptr<IndexedSubGraph>& SubGraph() { return g_host->ComputeCapability__SubGraph(this); }

  ComputeCapability() = delete;
  ComputeCapability(const ComputeCapability&) = delete;
  void operator=(const ComputeCapability&) = delete;
};

struct DataTransferManager final {
  Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const { return g_host->DataTransferManager__CopyTensor(this, src, dst, exec_queue_id); }
  Status CopyTensor(const Tensor& src, Tensor& dst) const { return g_host->DataTransferManager__CopyTensor(this, src, dst); }
#if !defined(DISABLE_SPARSE_TENSORS)
  Status CopySparseTensor(const SparseTensor& src, SparseTensor& dst) const { return g_host->DataTransferManager__CopySparseTensor(this, src, dst); }
  Status CopySparseTensor(const SparseTensor& src, SparseTensor& dst, int exec_queue_id) const { return g_host->DataTransferManager__CopySparseTensor(this, src, dst, exec_queue_id); }
  Status CopySparseTensors(const std::vector<IDataTransfer::SparseSrcDstPair>& src_dst_pairs) const { return g_host->DataTransferManager__CopySparseTensors(this, src_dst_pairs); }
#endif
  const IDataTransfer* GetDataTransfer(const OrtDevice& src_device, const OrtDevice& dst_device) const { return g_host->DataTransferManager__GetDataTransfer(this, src_device, dst_device); }

  PROVIDER_DISALLOW_ALL(DataTransferManager)
};

struct IndexedSubGraph_MetaDef final {
  static std::unique_ptr<IndexedSubGraph_MetaDef> Create() { return g_host->IndexedSubGraph_MetaDef__construct(); }
  static void operator delete(void* p) { g_host->IndexedSubGraph_MetaDef__operator_delete(reinterpret_cast<IndexedSubGraph_MetaDef*>(p)); }

  const std::string& name() const { return g_host->IndexedSubGraph_MetaDef__name(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  std::string& name() { return g_host->IndexedSubGraph_MetaDef__name(this); }
  const std::string& domain() const { return g_host->IndexedSubGraph_MetaDef__domain(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  std::string& domain() { return g_host->IndexedSubGraph_MetaDef__domain(this); }
  int since_version() const { return g_host->IndexedSubGraph_MetaDef__since_version(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  int& since_version() { return g_host->IndexedSubGraph_MetaDef__since_version(this); }

  ONNX_NAMESPACE::OperatorStatus& status() { return g_host->IndexedSubGraph_MetaDef__status(this); }

  const std::vector<std::string>& inputs() const { return g_host->IndexedSubGraph_MetaDef__inputs(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  std::vector<std::string>& inputs() { return g_host->IndexedSubGraph_MetaDef__inputs(this); }
  const std::vector<std::string>& outputs() const { return g_host->IndexedSubGraph_MetaDef__outputs(const_cast<IndexedSubGraph_MetaDef*>(this)); }
  std::vector<std::string>& outputs() { return g_host->IndexedSubGraph_MetaDef__outputs(this); }
  NodeAttributes& attributes() { return g_host->IndexedSubGraph_MetaDef__attributes(this); }

  std::string& doc_string() { return g_host->IndexedSubGraph_MetaDef__doc_string(this); }

  IndexedSubGraph_MetaDef() = delete;
  IndexedSubGraph_MetaDef(const IndexedSubGraph_MetaDef&) = delete;
  void operator=(const IndexedSubGraph_MetaDef&) = delete;
};

struct IndexedSubGraph final {
  static std::unique_ptr<IndexedSubGraph> Create() { return g_host->IndexedSubGraph__construct(); }
  static void operator delete(void* p) { g_host->IndexedSubGraph__operator_delete(reinterpret_cast<IndexedSubGraph*>(p)); }

  std::vector<onnxruntime::NodeIndex>& Nodes() { return g_host->IndexedSubGraph__Nodes(this); }

  void SetMetaDef(std::unique_ptr<IndexedSubGraph_MetaDef>&& meta_def_) { return g_host->IndexedSubGraph__SetMetaDef(this, std::move(*reinterpret_cast<std::unique_ptr<IndexedSubGraph_MetaDef>*>(&meta_def_))); }
  const IndexedSubGraph_MetaDef* GetMetaDef() const { return reinterpret_cast<const IndexedSubGraph_MetaDef*>(g_host->IndexedSubGraph__GetMetaDef(this)); }

  IndexedSubGraph() = delete;
  IndexedSubGraph(const IndexedSubGraph&) = delete;
  void operator=(const IndexedSubGraph&) = delete;
};

struct KernelDef final {
  static void operator delete(void* p) { g_host->KernelDef__operator_delete(reinterpret_cast<KernelDef*>(p)); }

  int ExecQueueId() const { return g_host->KernelDef__ExecQueueId(this); }

  void SinceVersion(/*out*/ int* start, /*out*/ int* end) const { g_host->KernelDef__SinceVersion(this, start, end); }
  const std::string& Domain() const { return g_host->KernelDef__Domain(this); }
  const std::string& OpName() const { return g_host->KernelDef__OpName(this); }

  KernelDef() = delete;
  KernelDef(const KernelDef*) = delete;
  void operator=(const KernelDef&) = delete;
};

using BuildKernelCreateInfoFn = KernelCreateInfo (*)();

struct KernelDefBuilder final {
  static std::unique_ptr<KernelDefBuilder> Create() { return g_host->KernelDefBuilder__construct(); }
  static void operator delete(void* p) { g_host->KernelDefBuilder__operator_delete(reinterpret_cast<KernelDefBuilder*>(p)); }

  KernelDefBuilder& SetName(const char* op_name) {
    g_host->KernelDefBuilder__SetName(this, op_name);
    return *this;
  }
  KernelDefBuilder& SetDomain(const char* domain) {
    g_host->KernelDefBuilder__SetDomain(this, domain);
    return *this;
  }
  KernelDefBuilder& SinceVersion(int since_version) {
    g_host->KernelDefBuilder__SinceVersion(this, since_version);
    return *this;
  }
  KernelDefBuilder& SinceVersion(int since_version_start, int since_version_end) {
    g_host->KernelDefBuilder__SinceVersion(this, since_version_start, since_version_end);
    return *this;
  }
  KernelDefBuilder& Provider(const char* provider_type) {
    g_host->KernelDefBuilder__Provider(this, provider_type);
    return *this;
  }
  KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType supported_type) {
    g_host->KernelDefBuilder__TypeConstraint(this, arg_name, supported_type);
    return *this;
  }
  KernelDefBuilder& TypeConstraint(const char* arg_name, const std::vector<MLDataType>& supported_types) {
    g_host->KernelDefBuilder__TypeConstraint(this, arg_name, supported_types);
    return *this;
  }
  KernelDefBuilder& InputMemoryType(OrtMemType type, int input_index) {
    g_host->KernelDefBuilder__InputMemoryType(this, type, input_index);
    return *this;
  }
  KernelDefBuilder& InputMemoryType(OrtMemType type, const std::vector<int>& input_indexes) {
    g_host->KernelDefBuilder__InputMemoryType(this, type, input_indexes);
    return *this;
  }
  KernelDefBuilder& OutputMemoryType(OrtMemType type, int input_index) {
    g_host->KernelDefBuilder__OutputMemoryType(this, type, input_index);
    return *this;
  }
  KernelDefBuilder& ExecQueueId(int queue_id) {
    g_host->KernelDefBuilder__ExecQueueId(this, queue_id);
    return *this;
  }
  KernelDefBuilder& MayInplace(int input_index, int output_index) {
    g_host->KernelDefBuilder__MayInplace(this, input_index, output_index);
    return *this;
  }
  KernelDefBuilder& Alias(const std::vector<std::pair<int, int>>& aliases) {
    g_host->KernelDefBuilder__Alias(this, aliases);
    return *this;
  }
  KernelDefBuilder& Alias(int input_index, int output_index) {
    g_host->KernelDefBuilder__Alias(this, input_index, output_index);
    return *this;
  }
  KernelDefBuilder& VariadicAlias(int input_offset, int output_offset) {
    g_host->KernelDefBuilder__VariadicAlias(this, input_offset, output_offset);
    return *this;
  }

  KernelDefBuilder& ExternalOutputs() {
    g_host->KernelDefBuilder__ExternalOutputs(this);
    return *this;
  }

  KernelDefBuilder& AllocateInputsContiguously() {
    g_host->KernelDefBuilder__AllocateInputsContiguously(this);
    return *this;
  }

  std::unique_ptr<KernelDef> Build() {
    return g_host->KernelDefBuilder__Build(this);
  }

  KernelDefBuilder() = delete;
  KernelDefBuilder(const KernelDefBuilder&) = delete;
  void operator=(const KernelDefBuilder&) = delete;
};

struct KernelRegistry final {
  static std::shared_ptr<KernelRegistry> Create() { return g_host->KernelRegistry__construct(); }
  static void operator delete(void* p) { g_host->KernelRegistry__operator_delete(reinterpret_cast<KernelRegistry*>(p)); }

  Status Register(KernelCreateInfo&& create_info) { return g_host->KernelRegistry__Register(this, std::move(create_info)); }

  Status TryFindKernel(const Node& node, ProviderType exec_provider, const KernelCreateInfo** out) const { return g_host->KernelRegistry__TryFindKernel(this, node, exec_provider, out); }

  KernelRegistry() = delete;
  KernelRegistry(const KernelRegistry&) = delete;
  void operator=(const KernelRegistry&) = delete;
};

struct PrimitiveDataTypeBase final {
  int32_t GetDataType() const { return g_host->PrimitiveDataTypeBase__GetDataType(this); }

  PROVIDER_DISALLOW_ALL(PrimitiveDataTypeBase)
};

class DataTypeImpl final {
 public:
  size_t Size() const { return g_host->DataTypeImpl__Size(this); }

  template <typename T>
  static MLDataType GetType();
  template <typename elemT>
  static MLDataType GetTensorType();
#if !defined(DISABLE_SPARSE_TENSORS)
  template <typename elemT>
  static MLDataType GetSparseTensorType();
#endif

  static MLDataType GetTypeFromOnnxType(int);

  bool IsTensorType() const { return g_host->DataTypeImpl__IsTensorType(this); }
  bool IsTensorSequenceType() const { return g_host->DataTypeImpl__IsTensorSequenceType(this); }
#if !defined(DISABLE_SPARSE_TENSORS)
  bool IsSparseTensorType() const { return g_host->DataTypeImpl__IsSparseTensorType(this); }
#endif
  DeleteFunc GetDeleteFunc() const { return g_host->DataTypeImpl__GetDeleteFunc(this); }

  static const std::vector<MLDataType>& AllFixedSizeTensorTypes() { return g_host->DataTypeImpl__AllFixedSizeTensorTypes(); }
  static const std::vector<MLDataType>& AllTensorTypes() { return g_host->DataTypeImpl__AllTensorTypes(); }
  static const std::vector<MLDataType>& AllIEEEFloatTensorTypes() { return g_host->DataTypeImpl__AllIEEEFloatTensorTypes(); }
  static const std::vector<MLDataType>& AllTensorAndSequenceTensorTypes() { return g_host->DataTypeImpl__AllTensorAndSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllFixedSizeTensorAndSequenceTensorTypes() { return g_host->DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllSequenceTensorTypes() { return g_host->DataTypeImpl__AllSequenceTensorTypes(); }
  static const std::vector<MLDataType>& AllFixedSizeSequenceTensorTypes() { return g_host->DataTypeImpl__AllFixedSizeSequenceTensorTypes(); }

  const PrimitiveDataTypeBase* AsPrimitiveDataType() const { return g_host->DataTypeImpl__AsPrimitiveDataType(this); }

  static const char* ToString(MLDataType type) { return g_host->DataTypeImpl__ToString(type); }

  PROVIDER_DISALLOW_ALL(DataTypeImpl)
};

struct Function final {
  const Graph& Body() const { return g_host->Function__Body(this); }

  PROVIDER_DISALLOW_ALL(Function)
};

struct Node final {
  const std::string& Name() const noexcept { return g_host->Node__Name(this); }
  const std::string& Description() const noexcept { return g_host->Node__Description(this); }
  const std::string& Domain() const noexcept { return g_host->Node__Domain(this); }
  const std::string& OpType() const noexcept { return g_host->Node__OpType(this); }

  int SinceVersion() const noexcept { return g_host->Node__SinceVersion(this); }

  const Function* GetFunctionBody() const noexcept { return g_host->Node__GetFunctionBody(this); }
  ProviderType GetExecutionProviderType() const noexcept { return g_host->Node__GetExecutionProviderType(this); }

  ConstPointerContainer<std::vector<NodeArg*>> ImplicitInputDefs() const noexcept { return g_host->Node__ImplicitInputDefs(this); }

  const std::vector<int>& InputArgCount() const noexcept { return g_host->Node__InputArgCount(this); }

  ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept { return g_host->Node__InputDefs(this); }
  ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept { return g_host->Node__OutputDefs(this); }
  NodeIndex Index() const noexcept { return g_host->Node__Index(this); }

  void ToProto(ONNX_NAMESPACE::NodeProto& proto, bool update_subgraphs = false) const { return g_host->Node__ToProto(this, proto, update_subgraphs); }

  const NodeAttributes& GetAttributes() const noexcept { return g_host->Node__GetAttributes(this); }
  size_t GetInputEdgesCount() const noexcept { return g_host->Node__GetInputEdgesCount(this); }
  size_t GetOutputEdgesCount() const noexcept { return g_host->Node__GetOutputEdgesCount(this); }

  struct NodeConstIterator {
    NodeConstIterator(std::unique_ptr<Node__NodeIterator> p) : impl_{std::move(p)} {}

    bool operator!=(const NodeConstIterator& p_other) const { return *impl_ != *p_other.impl_; }

    void operator++() { impl_->operator++(); }

    const Node& operator*() const { return impl_->operator*(); }
    const Node* operator->() const { return &impl_->operator*(); }

    std::unique_ptr<Node__NodeIterator> impl_;
  };

  NodeConstIterator InputNodesBegin() const noexcept { return g_host->Node__InputNodesBegin(this); }
  NodeConstIterator InputNodesEnd() const noexcept { return g_host->Node__InputNodesEnd(this); }

  NodeConstIterator OutputNodesBegin() const noexcept { return g_host->Node__OutputNodesBegin(this); }
  NodeConstIterator OutputNodesEnd() const noexcept { return g_host->Node__OutputNodesEnd(this); }

  struct EdgeConstIterator {
    EdgeConstIterator(std::unique_ptr<Node__EdgeIterator> p) : impl_{std::move(p)} {}

    bool operator!=(const EdgeConstIterator& p_other) const {
      return *impl_ != *p_other.impl_;
    }

    void operator++() { impl_->operator++(); }
    const Node__EdgeIterator* operator->() const { return impl_.get(); }

    std::unique_ptr<Node__EdgeIterator> impl_;
  };

  EdgeConstIterator OutputEdgesBegin() const noexcept { return g_host->Node__OutputEdgesBegin(this); }
  EdgeConstIterator OutputEdgesEnd() const noexcept { return g_host->Node__OutputEdgesEnd(this); }

  void ForEachDef(std::function<void(const NodeArg&, bool is_input)> func, bool include_missing_optional_defs = false) const { g_host->Node__ForEachDef(this, func, std::move(include_missing_optional_defs)); }

  PROVIDER_DISALLOW_ALL(Node)
};

struct NodeArg final {
  const std::string& Name() const noexcept { return g_host->NodeArg__Name(this); }
  const ONNX_NAMESPACE::TensorShapeProto* Shape() const { return g_host->NodeArg__Shape(this); }
  ONNX_NAMESPACE::DataType Type() const noexcept { return g_host->NodeArg__Type(this); }
  const NodeArgInfo& ToProto() const noexcept { return g_host->NodeArg__ToProto(this); }
  bool Exists() const noexcept { return g_host->NodeArg__Exists(this); }
  const ONNX_NAMESPACE::TypeProto* TypeAsProto() const noexcept { return g_host->NodeArg__TypeAsProto(this); }

  PROVIDER_DISALLOW_ALL(NodeArg)
};

struct NodeAttributes final {
  static std::unique_ptr<NodeAttributes> Create() { return g_host->NodeAttributes__construct(); }
  void operator=(const NodeAttributes& v) { return g_host->NodeAttributes__operator_assign(this, v); }
  static void operator delete(void* p) { g_host->NodeAttributes__operator_delete(reinterpret_cast<NodeAttributes*>(p)); }

  size_t size() const { return g_host->NodeAttributes__size(this); }
  void clear() noexcept { g_host->NodeAttributes__clear(this); }
  size_t count(const std::string& keyval) const { return g_host->NodeAttributes__count(this, keyval); }
  ONNX_NAMESPACE::AttributeProto& operator[](const std::string& string) { return g_host->NodeAttributes__operator_array(this, string); }
  const ONNX_NAMESPACE::AttributeProto& at(const std::string& string) const { return g_host->NodeAttributes__at(this, string); }

  IteratorHolder<NodeAttributes_Iterator, std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>> begin() const { return g_host->NodeAttributes__begin(this); }
  IteratorHolder<NodeAttributes_Iterator, std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>> end() const { return g_host->NodeAttributes__end(this); }
  IteratorHolder<NodeAttributes_Iterator, std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>> find(const std::string& key) const { return g_host->NodeAttributes__find(this, key); }
  void insert(const NodeAttributes& v) { return g_host->NodeAttributes__insert(this, v); }

  NodeAttributes() = delete;
  NodeAttributes(const NodeAttributes&) = delete;
};

struct Model final {
  static void operator delete(void* p) { g_host->Model__operator_delete(reinterpret_cast<Model*>(p)); }

  Graph& MainGraph() { return g_host->Model__MainGraph(this); }

  std::unique_ptr<ONNX_NAMESPACE::ModelProto> ToProto() { return g_host->Model__ToProto(this); }

  Model() = delete;
  Model(const Model&) = delete;
  void operator=(const Model&) = delete;
};

struct Graph final {
  std::unique_ptr<GraphViewer> CreateGraphViewer() const { return g_host->Graph__CreateGraphViewer(this); }
  std::unique_ptr<ONNX_NAMESPACE::GraphProto> ToGraphProto() const { return g_host->Graph__ToGraphProto(this); }

  NodeArg& GetOrCreateNodeArg(const std::string& name, const ONNX_NAMESPACE::TypeProto* p_arg_type) { return g_host->Graph__GetOrCreateNodeArg(this, name, p_arg_type); }

  Status Resolve() { return g_host->Graph__Resolve(this); }
  void AddInitializedTensor(const ONNX_NAMESPACE::TensorProto& tensor) { return g_host->Graph__AddInitializedTensor(this, tensor); }
  Node& AddNode(const std::string& name, const std::string& op_type, const std::string& description, const std::vector<NodeArg*>& input_args, const std::vector<NodeArg*>& output_args, const NodeAttributes* attributes, const std::string& domain) { return g_host->Graph__AddNode(this, name, op_type, description, input_args, output_args, attributes, domain); }

  const std::vector<const NodeArg*>& GetOutputs() const noexcept { return g_host->Graph__GetOutputs(this); }
  void SetOutputs(const std::vector<const NodeArg*>& outputs) { return g_host->Graph__SetOutputs(this, outputs); }

  const std::vector<const NodeArg*>& GetInputs() const noexcept { return g_host->Graph__GetInputs(this); }

  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const { return g_host->Graph__GetInitializedTensor(this, tensor_name, value); }

  PROVIDER_DISALLOW_ALL(Graph)
};

struct GraphViewer final {
  static void operator delete(void* p) { g_host->GraphViewer__operator_delete(reinterpret_cast<GraphViewer*>(p)); }

  std::unique_ptr<Model> CreateModel(const logging::Logger& logger) const { return g_host->GraphViewer__CreateModel(this, logger); }

  const std::string& Name() const noexcept { return g_host->GraphViewer__Name(this); }
  const Path& ModelPath() const noexcept { return g_host->GraphViewer__ModelPath(this); }

  const Node* GetNode(NodeIndex node_index) const { return g_host->GraphViewer__GetNode(this, node_index); }
  const NodeArg* GetNodeArg(const std::string& name) const { return g_host->GraphViewer__GetNodeArg(this, name); }

  bool IsSubgraph() const { return g_host->GraphViewer__IsSubgraph(this); }
  bool IsConstantInitializer(const std::string& name, bool check_outer_scope) const { return g_host->GraphViewer__IsConstantInitializer(this, name, check_outer_scope); }

  int NumberOfNodes() const noexcept { return g_host->GraphViewer__NumberOfNodes(this); }
  int MaxNodeIndex() const noexcept { return g_host->GraphViewer__MaxNodeIndex(this); }

  const std::vector<const NodeArg*>& GetInputs() const noexcept { return g_host->GraphViewer__GetInputs(this); }
  const std::vector<const NodeArg*>& GetOutputs() const noexcept { return g_host->GraphViewer__GetOutputs(this); }
  const std::unordered_set<const NodeArg*>& GetValueInfo() const noexcept { return g_host->GraphViewer__GetValueInfo(this); }

  const InitializedTensorSet& GetAllInitializedTensors() const noexcept { return g_host->GraphViewer__GetAllInitializedTensors(this); }
  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const { return g_host->GraphViewer__GetInitializedTensor(this, tensor_name, value); }

  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept { return g_host->GraphViewer__DomainToVersionMap(this); }

  const std::vector<NodeIndex>& GetNodesInTopologicalOrder() const { return g_host->GraphViewer__GetNodesInTopologicalOrder(this); }
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept { return g_host->GraphViewer__GetInputsIncludingInitializers(this); }

  GraphViewer() = delete;
  GraphViewer(const GraphViewer&) = delete;
  void operator=(const GraphViewer&) = delete;
};

struct Path final {
  PathString ToPathString() const noexcept { return g_host->Path__ToPathString(this); }

  PROVIDER_DISALLOW_ALL(Path)
};

struct OpKernelContext final {
  template <typename T>
  const T& RequiredInput(int index) const;
  Tensor& RequiredOutput(int index, const TensorShape& shape) { return g_host->OpKernelContext__RequiredOutput(this, index, shape); }

  template <typename T>
  const T* Input(int index) const;
  int InputCount() const { return g_host->OpKernelContext__InputCount(this); }

  MLDataType InputType(int index) const { return g_host->OpKernelContext__InputType(this, index); }

  template <typename T>
  T* Output(int index);

  Tensor* Output(int index, const TensorShape& shape) { return g_host->OpKernelContext__Output(this, index, shape); }
#if !defined(DISABLE_SPARSE_TENSORS)
  SparseTensor* OutputSparse(int index, const TensorShape& shape) { return g_host->OpKernelContext__OutputSparse(this, index, shape); }
#endif
  int OutputCount() const { return g_host->OpKernelContext__OutputCount(this); }

  Status GetTempSpaceAllocator(AllocatorPtr* output) const { return g_host->OpKernelContext__GetTempSpaceAllocator(this, output); }

  bool GetUseDeterministicCompute() const { return g_host->OpKernelContext__GetUseDeterministicCompute(this); }

  bool TryGetInferredOutputShape(int index, TensorShape& shape) const { return g_host->OpKernelContext__TryGetInferredOutputShape(this, index, shape); }
  bool TryGetInferredInputShape(int index, TensorShape& shape) const { return g_host->OpKernelContext__TryGetInferredInputShape(this, index, shape); }

  PROVIDER_DISALLOW_ALL(OpKernelContext)
};

template <>
inline const Tensor* OpKernelContext::Input<Tensor>(int index) const {
  return g_host->OpKernelContext__Input_Tensor(this, index);
}

#if !defined(DISABLE_SPARSE_TENSORS)
template <>
inline const SparseTensor* OpKernelContext::Input<SparseTensor>(int index) const {
  return g_host->OpKernelContext__Input_SparseTensor(this, index);
}
#endif

template <>
inline const TensorSeq* OpKernelContext::Input<TensorSeq>(int index) const {
  return g_host->OpKernelContext__Input_TensorSeq(this, index);
}

template <>
inline Tensor* OpKernelContext::Output<Tensor>(int index) {
  return g_host->OpKernelContext__Output_Tensor(this, index);
}

template <>
inline TensorSeq* OpKernelContext::Output<TensorSeq>(int index) {
  return g_host->OpKernelContext__Output_TensorSeq(this, index);
}

template <>
inline const Tensor& OpKernelContext::RequiredInput(int index) const {
  return g_host->OpKernelContext__RequiredInput_Tensor(this, index);
}

struct OpKernelInfo final {
  static void operator delete(void* p) { g_host->OpKernelInfo__operator_delete(reinterpret_cast<OpKernelInfo*>(p)); }

  AllocatorPtr GetAllocator(int device_id, OrtMemType mem_type) const { return g_host->OpKernelInfo__GetAllocator(this, device_id, mem_type); }

  const IExecutionProvider* GetExecutionProvider() const noexcept { return g_host->OpKernelInfo__GetExecutionProvider(this); }

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

  bool TryGetConstantInput(int input_index, const Tensor** constant_input_value) const { return g_host->OpKernelInfo__TryGetConstantInput(this, input_index, constant_input_value); }

  const DataTransferManager& GetDataTransferManager() const noexcept { return g_host->OpKernelInfo__GetDataTransferManager(this); }
  const KernelDef& GetKernelDef() const { return g_host->OpKernelInfo__GetKernelDef(this); }

  uint32_t GetInputCount() const { return g_host->OpKernelInfo__GetInputCount(this); }
  uint32_t GetOutputCount() const { return g_host->OpKernelInfo__GetOutputCount(this); }

  const Node& node() const noexcept { return g_host->OpKernelInfo__node(this); }

  OpKernelInfo() = delete;
  OpKernelInfo(const OpKernelInfo&) = delete;
  void operator=(const OpKernelInfo&) = delete;
};

template <>
inline Status OpKernelInfo::GetAttr<int64_t>(const std::string& name, int64_t* value) const { return g_host->OpKernelInfo__GetAttr_int64(this, name, value); }
template <>
inline Status OpKernelInfo::GetAttr<float>(const std::string& name, float* value) const { return g_host->OpKernelInfo__GetAttr_float(this, name, value); }
template <>
inline Status OpKernelInfo::GetAttr<std::string>(const std::string& name, std::string* value) const { return g_host->OpKernelInfo__GetAttr_string(this, name, value); }
template <>
inline Status OpKernelInfo::GetAttr<ONNX_NAMESPACE::TensorProto>(const std::string& name, ONNX_NAMESPACE::TensorProto* value) const { return g_host->OpKernelInfo__GetAttr_TensorProto(this, name, value); }
template <>
inline Status OpKernelInfo::GetAttrs<int64_t>(const std::string& name, std::vector<int64_t>& values) const { return g_host->OpKernelInfo__GetAttrs(this, name, values); }
template <>
inline Status OpKernelInfo::GetAttrs<float>(const std::string& name, std::vector<float>& values) const { return g_host->OpKernelInfo__GetAttrs(this, name, values); }
template <>
inline Status OpKernelInfo::GetAttrs<std::string>(const std::string& name, std::vector<std::string>& values) const { return g_host->OpKernelInfo__GetAttrs(this, name, values); }

class SessionState {
 public:
  const DataTransferManager& GetDataTransferMgr() const noexcept { return g_host->SessionState__GetDataTransferMgr(this); }

  PROVIDER_DISALLOW_ALL(SessionState)
};

struct Tensor final {
  static std::unique_ptr<Tensor> Create(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator) { return g_host->Tensor__construct(p_type, shape, std::move(allocator)); }
  static std::unique_ptr<Tensor> Create(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc, ptrdiff_t offset = 0) { return g_host->Tensor__construct(p_type, shape, p_data, alloc, offset); }

  static void operator delete(void* p) { g_host->Tensor__operator_delete(reinterpret_cast<Tensor*>(p)); }

  static void InitOrtValue(MLDataType elt_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator, OrtValue& ort_value) {
    g_host->Tensor__InitOrtValue(elt_type, shape, std::move(allocator), ort_value);
  }

  static void InitOrtValue(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& location, OrtValue& ort_value) {
    g_host->Tensor__InitOrtValue(p_type, shape, p_data, location, ort_value);
  }

  template <typename T>
  T* MutableData();

  template <typename T>
  const T* Data() const;

  template <typename T>
  gsl::span<const T> DataAsSpan() const;

  void* MutableDataRaw(MLDataType type) { return g_host->Tensor__MutableDataRaw(this, type); }
  const void* DataRaw(MLDataType type) const { return g_host->Tensor__DataRaw(this, type); }

  void* MutableDataRaw() noexcept { return g_host->Tensor__MutableDataRaw(this); }
  const void* DataRaw() const noexcept { return g_host->Tensor__DataRaw(this); }

  const TensorShape& Shape() const { return g_host->Tensor__Shape(this); }
  void Reshape(const TensorShape& new_shape) { g_host->Tensor__Reshape(this, new_shape); }
  void SetByteOffset(ptrdiff_t byte_offset) { return g_host->Tensor__SetByteOffset(this, byte_offset); }
  ptrdiff_t ByteOffset() const { return g_host->Tensor__ByteOffset(this); }
  size_t SizeInBytes() const { return g_host->Tensor__SizeInBytes(this); }
  const OrtMemoryInfo& Location() const { return g_host->Tensor__Location(this); }

  int32_t GetElementType() const { return g_host->Tensor__GetElementType(this); }
  MLDataType DataType() const { return g_host->Tensor__DataType(this); }
  bool IsDataTypeString() const { return g_host->Tensor__IsDataTypeString(this); }

  template <class T>
  bool IsDataType() const;

  Tensor() = delete;
  Tensor(const Tensor&) = delete;
  void operator=(const Tensor&) = delete;
};

template <>
inline bool Tensor::IsDataType<bool>() const { return g_host->Tensor__IsDataType_bool(this); }
template <>
inline bool Tensor::IsDataType<int8_t>() const { return g_host->Tensor__IsDataType_int8(this); }
template <>
inline bool Tensor::IsDataType<uint8_t>() const { return g_host->Tensor__IsDataType_uint8(this); }
template <>
inline bool Tensor::IsDataType<int16_t>() const { return g_host->Tensor__IsDataType_int16(this); }
template <>
inline bool Tensor::IsDataType<uint16_t>() const { return g_host->Tensor__IsDataType_uint16(this); }
template <>
inline bool Tensor::IsDataType<int32_t>() const { return g_host->Tensor__IsDataType_int32(this); }
template <>
inline bool Tensor::IsDataType<uint32_t>() const { return g_host->Tensor__IsDataType_uint32(this); }
template <>
inline bool Tensor::IsDataType<int64_t>() const { return g_host->Tensor__IsDataType_int64(this); }
template <>
inline bool Tensor::IsDataType<uint64_t>() const { return g_host->Tensor__IsDataType_uint64(this); }
template <>
inline bool Tensor::IsDataType<float>() const { return g_host->Tensor__IsDataType_float(this); }
template <>
inline bool Tensor::IsDataType<double>() const { return g_host->Tensor__IsDataType_double(this); }
template <>
inline bool Tensor::IsDataType<MLFloat16>() const { return g_host->Tensor__IsDataType_MLFloat16(this); }

template <>
inline bool* Tensor::MutableData<bool>() { return g_host->Tensor__MutableData_bool(this); }
template <>
inline int8_t* Tensor::MutableData<int8_t>() { return g_host->Tensor__MutableData_int8(this); }
template <>
inline uint8_t* Tensor::MutableData<uint8_t>() { return g_host->Tensor__MutableData_uint8(this); }
template <>
inline int16_t* Tensor::MutableData<int16_t>() { return g_host->Tensor__MutableData_int16(this); }
template <>
inline uint16_t* Tensor::MutableData<uint16_t>() { return g_host->Tensor__MutableData_uint16(this); }
template <>
inline int32_t* Tensor::MutableData<int32_t>() { return g_host->Tensor__MutableData_int32(this); }
template <>
inline uint32_t* Tensor::MutableData<uint32_t>() { return g_host->Tensor__MutableData_uint32(this); }
template <>
inline int64_t* Tensor::MutableData<int64_t>() { return g_host->Tensor__MutableData_int64(this); }
template <>
inline uint64_t* Tensor::MutableData<uint64_t>() { return g_host->Tensor__MutableData_uint64(this); }
template <>
inline float* Tensor::MutableData<float>() { return g_host->Tensor__MutableData_float(this); }
template <>
inline double* Tensor::MutableData<double>() { return g_host->Tensor__MutableData_double(this); }
template <>
inline BFloat16* Tensor::MutableData<BFloat16>() { return g_host->Tensor__MutableData_BFloat16(this); }
template <>
inline MLFloat16* Tensor::MutableData<MLFloat16>() { return g_host->Tensor__MutableData_MLFloat16(this); }

template <>
inline const bool* Tensor::Data<bool>() const { return g_host->Tensor__Data_bool(this); }
template <>
inline const int8_t* Tensor::Data<int8_t>() const { return g_host->Tensor__Data_int8(this); }
template <>
inline const uint8_t* Tensor::Data<uint8_t>() const { return g_host->Tensor__Data_uint8(this); }
template <>
inline const int16_t* Tensor::Data<int16_t>() const { return g_host->Tensor__Data_int16(this); }
template <>
inline const uint16_t* Tensor::Data<uint16_t>() const { return g_host->Tensor__Data_uint16(this); }
template <>
inline const int32_t* Tensor::Data<int32_t>() const { return g_host->Tensor__Data_int32(this); }
template <>
inline const uint32_t* Tensor::Data<uint32_t>() const { return g_host->Tensor__Data_uint32(this); }
template <>
inline const int64_t* Tensor::Data<int64_t>() const { return g_host->Tensor__Data_int64(this); }
template <>
inline const uint64_t* Tensor::Data<uint64_t>() const { return g_host->Tensor__Data_uint64(this); }
template <>
inline const float* Tensor::Data<float>() const { return g_host->Tensor__Data_float(this); }
template <>
inline const double* Tensor::Data<double>() const { return g_host->Tensor__Data_double(this); }
template <>
inline const BFloat16* Tensor::Data<BFloat16>() const { return g_host->Tensor__Data_BFloat16(this); }
template <>
inline const MLFloat16* Tensor::Data<MLFloat16>() const { return g_host->Tensor__Data_MLFloat16(this); }

// SparseTensor
#if !defined(DISABLE_SPARSE_TENSORS)
struct SparseTensor final {
  const TensorShape& DenseShape() const noexcept { return g_host->SparseTensor__DenseShape(this); }
  Status Copy(const DataTransferManager& dtm, int exec_q_id, SparseTensor& dst) const { return g_host->SparseTensor__Copy(this, dtm, exec_q_id, dst); }
};
#endif

//TensorSeq
struct TensorSeq final {
  MLDataType DataType() const noexcept { return g_host->TensorSeq__DataType(this); }
  void SetType(MLDataType elem_type) { g_host->TensorSeq__SetType(this, elem_type); }
  size_t Size() const noexcept { return g_host->TensorSeq__Size(this); }
  const Tensor& Get(size_t i) const { return g_host->TensorSeq__Get(this, i); }
  void Add(Tensor&& tensor) { g_host->TensorSeq__Add(this, std::move(tensor)); }
};

template <>
inline gsl::span<const int64_t> Tensor::DataAsSpan() const { return g_host->Tensor__DataAsSpan_int64(this); }

}  // namespace onnxruntime