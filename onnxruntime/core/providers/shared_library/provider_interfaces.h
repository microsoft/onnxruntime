// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Public wrappers around internal ort interfaces (currently)
// In the future the internal implementations could derive from these to remove the need for the wrapper implementations
#include "core/providers/shared_library/provider_host_api.h"

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

struct NodeAttributes_Iterator {
  virtual ~NodeAttributes_Iterator() {}

  virtual bool operator!=(const NodeAttributes_Iterator& p) const = 0;

  virtual void operator++() = 0;
  virtual const std::string& first() const = 0;
  virtual const ONNX_NAMESPACE::AttributeProto& second() = 0;
};

struct TensorShapeProto_Dimension_Iterator {
  virtual ~TensorShapeProto_Dimension_Iterator() {}

  virtual bool operator!=(const TensorShapeProto_Dimension_Iterator& p) const = 0;

  virtual void operator++() = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto_Dimension& operator*() = 0;
};

using NodeIndex = size_t;
// We can't just reinterpret_cast this one, since it's an unordered_map of object BY VALUE (can't do anything by value on the real types)
// using NodeAttributes = std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto_Copyable>;

using Provider_InitializedTensorSet = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>;

struct Node__NodeIterator {
  virtual ~Node__NodeIterator() {}

  virtual bool operator!=(const Node__NodeIterator& p) const = 0;

  virtual void operator++() = 0;
  virtual const Node& operator*() = 0;
};

struct Node__EdgeIterator {
  virtual ~Node__EdgeIterator() {}
  virtual bool operator!=(const Node__EdgeIterator& p) const = 0;

  virtual void operator++() = 0;
  virtual const Node& GetNode() const = 0;
  virtual int GetSrcArgIndex() const = 0;
  virtual int GetDstArgIndex() const = 0;
};

// There are two ways to route a function, one is a virtual method and the other is a function pointer (or pointer to member function)
// The function pointers are nicer in that they directly call the target function, but they cannot be used in cases where we're calling
// a specific implementation of a virtual class member. Trying to get a pointer to member of a virtual function will return a thunk that
// calls the virtual function (which will lead to infinite recursion in the bridge). There is no known way to get the non virtual member
// function pointer implementation in this case.
struct ProviderHost {
  virtual AllocatorPtr CreateAllocator(const AllocatorCreationInfo& info) = 0;

  virtual logging::Logger* LoggingManager_GetDefaultLogger() = 0;

  virtual std::unique_ptr<IAllocator> CreateCPUAllocator(const OrtMemoryInfo& memory_info) = 0;

#ifdef USE_TENSORRT
  virtual std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<IDataTransfer> CreateGPUDataTransfer(void* stream) = 0;

  virtual void cuda__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(void* stream, const double* input_data, float* output_data, size_t count) = 0;//slx
  virtual void cuda__Impl_Cast(void* stream, const float* input_data, double* output_data, size_t count) = 0;//slx

  virtual bool CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) = 0;
  virtual bool CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) = 0;
#endif

  virtual std::string GetEnvironmentVar(const std::string& var_name) = 0;

  // PrimitiveDataTypeBase
  virtual int32_t PrimitiveDataTypeBase__GetDataType(const PrimitiveDataTypeBase* p) = 0;

  // DataTypeImpl
  MLDataType (*DataTypeImpl_GetType_Tensor)();
  MLDataType (*DataTypeImpl_GetType_float)();
  MLDataType (*DataTypeImpl_GetTensorType_float)();

  virtual const char* DataTypeImpl__ToString(MLDataType type) = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorTypes() = 0;
  virtual const std::vector<MLDataType>& DataTypeImpl__AllTensorTypes() = 0;
  virtual size_t DataTypeImpl__Size(const DataTypeImpl* p) = 0;
  virtual const PrimitiveDataTypeBase* DataTypeImpl__AsPrimitiveDataType(const DataTypeImpl* p) = 0;

  virtual void* HeapAllocate(size_t size) = 0;
  virtual void HeapFree(void*) = 0;

  virtual AutoPadType StringToAutoPadType(const std::string& str) = 0;

  virtual void LogRuntimeError(uint32_t session_id, const common::Status& status,
                               const char* file, const char* function, uint32_t line) = 0;

  virtual std::vector<std::string> GetStackTrace() = 0;

  // IAllocator
  virtual bool IAllocator__CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment, size_t* out) = 0;

  // IExecutionProvider
  virtual AllocatorPtr IExecutionProvider__GetAllocator(const IExecutionProvider* p, int id, OrtMemType mem_type) = 0;
  virtual void IExecutionProvider__InsertAllocator(IExecutionProvider* p, AllocatorPtr allocator) = 0;
  virtual void IExecutionProvider__TryInsertAllocator(IExecutionProvider* p, AllocatorPtr allocator) = 0;
  virtual std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider__GetCapability(const IExecutionProvider* p, const onnxruntime::GraphViewer& graph_viewer,
                                                                                            const std::vector<const KernelRegistry*>& kernel_registries) = 0;
  virtual common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<onnxruntime::Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) = 0;
  virtual common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<onnxruntime::Node*>& fused_nodes, std::string& dll_path) = 0;
  virtual common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs, std::vector<NodeComputeInfo>& node_compute_funcs) = 0;

  virtual int IExecutionProvider__GenerateMetaDefId(const IExecutionProvider* p, const onnxruntime::GraphViewer& graph_viewer, uint64_t& model_hash) = 0;

  virtual void IExecutionProvider__RegisterAllocator(IExecutionProvider* p, std::shared_ptr<AllocatorManager> allocator_manager) = 0;
  // Status
  virtual std::string Status__ToString(const Status* p) = 0;

  // TensorShape
  virtual int64_t TensorShape__SizeHelper(const TensorShape* p, size_t start, size_t end) = 0;
  virtual std::string TensorShape__ToString(const TensorShape* p) = 0;

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

  // Utils::DataTypeUtils
  virtual const std::string* Utils__DataTypeUtils__ToType(const ONNX_NAMESPACE::TypeProto& type_proto) = 0;

  // int64s
  virtual int int64s__size(const ONNX_NAMESPACE::int64s* p) = 0;
  virtual const int64_t& int64s__Get(const ONNX_NAMESPACE::int64s* p, int index) = 0;

  // TypeProto_Tensor
  virtual const ONNX_NAMESPACE::TensorShapeProto& TypeProto_Tensor__shape(const ONNX_NAMESPACE::TypeProto_Tensor* p) = 0;
  virtual ONNX_NAMESPACE::TensorShapeProto* TypeProto_Tensor__mutable_shape(ONNX_NAMESPACE::TypeProto_Tensor* p) = 0;
  virtual int32_t TypeProto_Tensor__elem_type(const ONNX_NAMESPACE::TypeProto_Tensor* p) = 0;

  // TypeProto
  virtual const ONNX_NAMESPACE::TypeProto_Tensor& TypeProto__tensor_type(const ONNX_NAMESPACE::TypeProto* p) = 0;
  virtual ONNX_NAMESPACE::TypeProto_Tensor* TypeProto__mutable_tensor_type(ONNX_NAMESPACE::TypeProto* p) = 0;

  // AttributeProto
  virtual std::unique_ptr<ONNX_NAMESPACE::AttributeProto> AttributeProto__construct() = 0;
  virtual void AttributeProto__operator_delete(ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual void AttributeProto__operator_assign(ONNX_NAMESPACE::AttributeProto* p, const ONNX_NAMESPACE::AttributeProto& v) = 0;

  virtual ONNX_NAMESPACE::AttributeProto_AttributeType AttributeProto__type(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual int AttributeProto__ints_size(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual int AttributeProto__floats_size(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual int64_t AttributeProto__ints(const ONNX_NAMESPACE::AttributeProto* p, int i) = 0;
  virtual float AttributeProto__floats(const ONNX_NAMESPACE::AttributeProto* p, int i) = 0;
  virtual const ONNX_NAMESPACE::int64s& AttributeProto__ints(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual int64_t AttributeProto__i(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual float AttributeProto__f(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual void AttributeProto__set_s(ONNX_NAMESPACE::AttributeProto* p, const ::std::string& value) = 0;
  virtual const ::std::string& AttributeProto__s(const ONNX_NAMESPACE::AttributeProto* p) = 0;
  virtual void AttributeProto__set_name(ONNX_NAMESPACE::AttributeProto* p, const ::std::string& value) = 0;
  virtual void AttributeProto__set_type(ONNX_NAMESPACE::AttributeProto* p, ONNX_NAMESPACE::AttributeProto_AttributeType value) = 0;
  virtual ONNX_NAMESPACE::TensorProto* AttributeProto__add_tensors(ONNX_NAMESPACE::AttributeProto* p) = 0;

  // GraphProto
  virtual void GraphProto__operator_delete(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual void GraphProto__operator_assign(ONNX_NAMESPACE::GraphProto* p, const ONNX_NAMESPACE::GraphProto& v) = 0;

  virtual const ONNX_NAMESPACE::ValueInfoProto& GraphProto__input(const ONNX_NAMESPACE::GraphProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_input(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual ONNX_NAMESPACE::ValueInfoProto* GraphProto__mutable_input(ONNX_NAMESPACE::GraphProto* p, int index) = 0;
  virtual int GraphProto__input_size(const ONNX_NAMESPACE::GraphProto* p) = 0;

  virtual const ONNX_NAMESPACE::ValueInfoProtos& GraphProto__output(const ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual const ONNX_NAMESPACE::ValueInfoProto& GraphProto__output(const ONNX_NAMESPACE::GraphProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_output(ONNX_NAMESPACE::GraphProto* p) = 0;

  virtual ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_value_info(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual ONNX_NAMESPACE::TensorProtos* GraphProto__mutable_initializer(ONNX_NAMESPACE::GraphProto* p) = 0;
  virtual ONNX_NAMESPACE::NodeProto* GraphProto__add_node(ONNX_NAMESPACE::GraphProto* p) = 0;

  // ModelProto
  virtual std::unique_ptr<ONNX_NAMESPACE::ModelProto> ModelProto__construct() = 0;
  virtual void ModelProto__operator_delete(ONNX_NAMESPACE::ModelProto* p) = 0;

  virtual bool ModelProto__SerializeToString(const ONNX_NAMESPACE::ModelProto* p, std::string& string) = 0;
  virtual bool ModelProto__SerializeToOstream(const ONNX_NAMESPACE::ModelProto* p, std::ostream& output) = 0;
  virtual bool ModelProto__ParseFromString(ONNX_NAMESPACE::ModelProto* p, const std::string& data) = 0;
  virtual std::string ModelProto__SerializeAsString(const ONNX_NAMESPACE::ModelProto* p) = 0;

  virtual const ONNX_NAMESPACE::GraphProto& ModelProto__graph(const ONNX_NAMESPACE::ModelProto* p) = 0;
  virtual ONNX_NAMESPACE::GraphProto* ModelProto__mutable_graph(ONNX_NAMESPACE::ModelProto* p) = 0;

  virtual void ModelProto__set_ir_version(ONNX_NAMESPACE::ModelProto* p, int64_t value) = 0;

  // TensorProto
  virtual void TensorProto__operator_delete(ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual void TensorProto__operator_assign(ONNX_NAMESPACE::TensorProto* p, const ONNX_NAMESPACE::TensorProto& v) = 0;
  virtual bool TensorProto__has_data_location(const ONNX_NAMESPACE::TensorProto* p) = 0;
  virtual int TensorProto__data_location(const ONNX_NAMESPACE::TensorProto* p) = 0;

  // TensorProtos
  virtual ONNX_NAMESPACE::TensorProto* TensorProtos__Add(ONNX_NAMESPACE::TensorProtos* p) = 0;

  // TensorShapeProto_Dimension
  virtual int TensorShapeProto_Dimension__value_case(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual const std::string& TensorShapeProto_Dimension__dim_param(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual int64_t TensorShapeProto_Dimension__dim_value(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual void TensorShapeProto_Dimension__set_dim_value(ONNX_NAMESPACE::TensorShapeProto_Dimension* p, int64_t value) = 0;
  virtual bool TensorShapeProto_Dimension__has_dim_value(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual bool TensorShapeProto_Dimension__has_dim_param(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;
  virtual void TensorShapeProto_Dimension__clear_dim_value(ONNX_NAMESPACE::TensorShapeProto_Dimension* p) = 0;

  // TensorShapeProto_Dimensions
  virtual std::unique_ptr<TensorShapeProto_Dimension_Iterator> TensorShapeProto_Dimensions__begin(const ONNX_NAMESPACE::TensorShapeProto_Dimensions* p) = 0;
  virtual std::unique_ptr<TensorShapeProto_Dimension_Iterator> TensorShapeProto_Dimensions__end(const ONNX_NAMESPACE::TensorShapeProto_Dimensions* p) = 0;

  // TensorShapeProto
  virtual int TensorShapeProto__dim_size(const ONNX_NAMESPACE::TensorShapeProto* p) = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto_Dimensions& TensorShapeProto__dim(const ONNX_NAMESPACE::TensorShapeProto* p) = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto_Dimension& TensorShapeProto__dim(const ONNX_NAMESPACE::TensorShapeProto* p, int index) = 0;
  virtual ONNX_NAMESPACE::TensorShapeProto_Dimension* TensorShapeProto__mutable_dim(ONNX_NAMESPACE::TensorShapeProto* p, int index) = 0;
  virtual void TensorShapeProto__clear_dim(ONNX_NAMESPACE::TensorShapeProto* p) = 0;
  virtual ONNX_NAMESPACE::TensorShapeProto_Dimension* TensorShapeProto__add_dim(ONNX_NAMESPACE::TensorShapeProto* p) = 0;

  // ValueInfoProto
  virtual void ValueInfoProto__operator_assign(ONNX_NAMESPACE::ValueInfoProto* p, const ONNX_NAMESPACE::ValueInfoProto& v) = 0;
  virtual const ONNX_NAMESPACE::TypeProto& ValueInfoProto__type(const ONNX_NAMESPACE::ValueInfoProto* p) = 0;
  virtual ONNX_NAMESPACE::TypeProto* ValueInfoProto__mutable_type(ONNX_NAMESPACE::ValueInfoProto* p) = 0;

  // ValueInfoProtos
  virtual ONNX_NAMESPACE::ValueInfoProto* ValueInfoProtos__Add(ONNX_NAMESPACE::ValueInfoProtos* p) = 0;

  virtual const ONNX_NAMESPACE::ValueInfoProto& ValueInfoProtos__operator_array(const ONNX_NAMESPACE::ValueInfoProtos* p, int index) = 0;

  // ComputeCapability
  virtual std::unique_ptr<ComputeCapability> ComputeCapability__construct(std::unique_ptr<IndexedSubGraph> t_sub_graph) = 0;
  virtual void ComputeCapability__operator_delete(ComputeCapability* p) = 0;
  virtual std::unique_ptr<IndexedSubGraph>& ComputeCapability__SubGraph(ComputeCapability* p) = 0;

  // DataTransferManager
  virtual Status DataTransferManager__CopyTensor(const DataTransferManager* p, const Tensor& src, Tensor& dst, int exec_queue_id) = 0;

  // IDataTransfer
  virtual Status IDataTransfer__CopyTensor(const IDataTransfer* p, const Tensor& src, Tensor& dst) = 0;
  virtual Status IDataTransfer__CopyTensors(const IDataTransfer* p, const std::vector<IDataTransfer::SrcDstPair>& src_dst_pairs) = 0;

  // IndexedSubGraph_MetaDef
  virtual std::unique_ptr<IndexedSubGraph_MetaDef> IndexedSubGraph_MetaDef__construct() = 0;
  virtual void IndexedSubGraph_MetaDef__operator_delete(IndexedSubGraph_MetaDef* p) = 0;

  virtual std::string& IndexedSubGraph_MetaDef__name(IndexedSubGraph_MetaDef* p) = 0;
  virtual std::string& IndexedSubGraph_MetaDef__domain(IndexedSubGraph_MetaDef* p) = 0;
  virtual int& IndexedSubGraph_MetaDef__since_version(IndexedSubGraph_MetaDef* p) = 0;
  virtual ONNX_NAMESPACE::OperatorStatus& IndexedSubGraph_MetaDef__status(IndexedSubGraph_MetaDef* p) = 0;
  virtual std::vector<std::string>& IndexedSubGraph_MetaDef__inputs(IndexedSubGraph_MetaDef* p) = 0;
  virtual std::vector<std::string>& IndexedSubGraph_MetaDef__outputs(IndexedSubGraph_MetaDef* p) = 0;
  virtual NodeAttributes& IndexedSubGraph_MetaDef__attributes(IndexedSubGraph_MetaDef* p) = 0;
  virtual std::string& IndexedSubGraph_MetaDef__doc_string(IndexedSubGraph_MetaDef* p) = 0;

  // IndexedSubGraph
  virtual std::unique_ptr<IndexedSubGraph> IndexedSubGraph__construct() = 0;
  virtual void IndexedSubGraph__operator_delete(IndexedSubGraph* p) = 0;

  virtual std::vector<onnxruntime::NodeIndex>& IndexedSubGraph__Nodes(IndexedSubGraph* p) = 0;

  virtual void IndexedSubGraph__SetMetaDef(IndexedSubGraph* p, std::unique_ptr<IndexedSubGraph_MetaDef>&& meta_def_) = 0;
  virtual const IndexedSubGraph_MetaDef* IndexedSubGraph__GetMetaDef(const IndexedSubGraph* p) = 0;

  // KernelDef
  virtual void KernelDef__operator_delete(KernelDef* p) = 0;
  virtual int KernelDef__ExecQueueId(const KernelDef* p) = 0;

  // KernelDefBuilder
  virtual std::unique_ptr<KernelDefBuilder> KernelDefBuilder__construct() = 0;
  virtual void KernelDefBuilder__operator_delete(KernelDefBuilder* p) = 0;

  virtual void KernelDefBuilder__SetName(KernelDefBuilder* p, const char* op_name) = 0;
  virtual void KernelDefBuilder__SetDomain(KernelDefBuilder* p, const char* domain) = 0;
  virtual void KernelDefBuilder__SinceVersion(KernelDefBuilder* p, int since_version) = 0;
  virtual void KernelDefBuilder__Provider(KernelDefBuilder* p, const char* provider_type) = 0;
  virtual void KernelDefBuilder__TypeConstraint(KernelDefBuilder* p, const char* arg_name, MLDataType supported_type) = 0;
  virtual void KernelDefBuilder__TypeConstraint(KernelDefBuilder* p, const char* arg_name, const std::vector<MLDataType>& supported_types) = 0;
  virtual void KernelDefBuilder__InputMemoryType(KernelDefBuilder* p, OrtMemType type, int input_index) = 0;
  virtual void KernelDefBuilder__OutputMemoryType(KernelDefBuilder* p, OrtMemType type, int input_index) = 0;
  virtual void KernelDefBuilder__ExecQueueId(KernelDefBuilder* p, int queue_id) = 0;

  virtual std::unique_ptr<KernelDef> KernelDefBuilder__Build(KernelDefBuilder* p) = 0;

  // KernelRegistry
  virtual std::shared_ptr<KernelRegistry> KernelRegistry__construct() = 0;
  virtual void KernelRegistry__operator_delete(KernelRegistry* p) = 0;
  virtual Status KernelRegistry__Register(KernelRegistry* p, KernelCreateInfo&& create_info) = 0;

  // Function
  virtual const Graph& Function__Body(const Function* p) = 0;

  // Node
  virtual const std::string& Node__Name(const Node* p) noexcept = 0;
  virtual const std::string& Node__Description(const Node* p) noexcept = 0;
  virtual const std::string& Node__Domain(const Node* p) noexcept = 0;
  virtual const std::string& Node__OpType(const Node* p) noexcept = 0;

  virtual const Function* Node__GetFunctionBody(const Node* p) noexcept = 0;

  virtual ConstPointerContainer<std::vector<NodeArg*>> Node__ImplicitInputDefs(const Node* p) noexcept = 0;

  virtual ConstPointerContainer<std::vector<NodeArg*>> Node__InputDefs(const Node* p) noexcept = 0;
  virtual ConstPointerContainer<std::vector<NodeArg*>> Node__OutputDefs(const Node* p) noexcept = 0;
  virtual NodeIndex Node__Index(const Node* p) noexcept = 0;

  virtual void Node__ToProto(const Node* p, ONNX_NAMESPACE::NodeProto& proto, bool update_subgraphs = false) = 0;

  virtual const NodeAttributes& Node__GetAttributes(const Node* p) noexcept = 0;
  virtual size_t Node__GetInputEdgesCount(const Node* p) noexcept = 0;
  virtual size_t Node__GetOutputEdgesCount(const Node* p) noexcept = 0;

  virtual std::unique_ptr<Node__NodeIterator> Node__InputNodesBegin(const Node* p) noexcept = 0;
  virtual std::unique_ptr<Node__NodeIterator> Node__InputNodesEnd(const Node* p) noexcept = 0;

  virtual std::unique_ptr<Node__NodeIterator> Node__OutputNodesBegin(const Node* p) noexcept = 0;
  virtual std::unique_ptr<Node__NodeIterator> Node__OutputNodesEnd(const Node* p) noexcept = 0;

  virtual std::unique_ptr<Node__EdgeIterator> Node__OutputEdgesBegin(const Node* p) noexcept = 0;
  virtual std::unique_ptr<Node__EdgeIterator> Node__OutputEdgesEnd(const Node* p) noexcept = 0;

  virtual void Node__ForEachDef(const Node* p, std::function<void(const NodeArg&, bool is_input)> func, bool include_missing_optional_defs) = 0;

  // NodeArg
  virtual const std::string& NodeArg__Name(const NodeArg* p) noexcept = 0;
  virtual const ONNX_NAMESPACE::TensorShapeProto* NodeArg__Shape(const NodeArg* p) = 0;
  virtual ONNX_NAMESPACE::DataType NodeArg__Type(const NodeArg* p) noexcept = 0;
  virtual const ONNX_NAMESPACE::NodeArgInfo& NodeArg__ToProto(const NodeArg* p) noexcept = 0;
  virtual bool NodeArg__Exists(const NodeArg* p) const noexcept = 0;
  virtual const ONNX_NAMESPACE::TypeProto* NodeArg__TypeAsProto(const NodeArg* p) noexcept = 0;

  // NodeAttributes
  virtual std::unique_ptr<NodeAttributes> NodeAttributes__construct() = 0;
  virtual void NodeAttributes__operator_delete(NodeAttributes* p) noexcept = 0;
  virtual void NodeAttributes__operator_assign(NodeAttributes* p, const NodeAttributes& v) = 0;

  virtual size_t NodeAttributes__size(const NodeAttributes* p) = 0;
  virtual void NodeAttributes__clear(NodeAttributes* p) noexcept = 0;
  virtual size_t NodeAttributes__count(const NodeAttributes* p, const std::string& keyval) = 0;
  virtual ONNX_NAMESPACE::AttributeProto& NodeAttributes__operator_array(NodeAttributes* p, const std::string& string) = 0;
  virtual const ONNX_NAMESPACE::AttributeProto& NodeAttributes__at(const NodeAttributes* p, const std::string& string) = 0;

  virtual std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__begin(const NodeAttributes* p) = 0;
  virtual std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__end(const NodeAttributes* p) = 0;
  virtual std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__find(const NodeAttributes* p, const std::string& key) = 0;
  virtual void NodeAttributes__insert(NodeAttributes* p, const NodeAttributes& v) = 0;

  // Model
  virtual void Model__operator_delete(Model* p) = 0;
  virtual Graph& Model__MainGraph(Model* p) = 0;
  virtual std::unique_ptr<ONNX_NAMESPACE::ModelProto> Model__ToProto(Model* p) = 0;

  // Graph
  virtual std::unique_ptr<GraphViewer> Graph__CreateGraphViewer(const Graph* p) = 0;
  virtual std::unique_ptr<ONNX_NAMESPACE::GraphProto> Graph__ToGraphProto(const Graph* p) = 0;

  virtual NodeArg& Graph__GetOrCreateNodeArg(Graph* p, const std::string& name, const ONNX_NAMESPACE::TypeProto* p_arg_type) = 0;

  virtual Status Graph__Resolve(Graph* p) = 0;
  virtual void Graph__AddInitializedTensor(Graph* p, const ONNX_NAMESPACE::TensorProto& tensor) = 0;
  virtual Node& Graph__AddNode(Graph* p, const std::string& name, const std::string& op_type, const std::string& description, const std::vector<NodeArg*>& input_args, const std::vector<NodeArg*>& output_args, const NodeAttributes* attributes, const std::string& domain) = 0;

  virtual const std::vector<const NodeArg*>& Graph__GetOutputs(const Graph* p) noexcept = 0;
  virtual void Graph__SetOutputs(Graph* p, const std::vector<const NodeArg*>& outputs) = 0;

  virtual const std::vector<const NodeArg*>& Graph__GetInputs(const Graph* p) noexcept = 0;
  virtual bool Graph__GetInitializedTensor(const Graph* p, const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) = 0;

  // GraphViewer
  virtual void GraphViewer__operator_delete(GraphViewer* p) = 0;
  virtual std::unique_ptr<Model> GraphViewer__CreateModel(const GraphViewer* p, const logging::Logger& logger) = 0;

  virtual const std::string& GraphViewer__Name(const GraphViewer* p) noexcept = 0;
  virtual const Path& GraphViewer__ModelPath(const GraphViewer* p) noexcept = 0;

  virtual const Node* GraphViewer__GetNode(const GraphViewer* p, NodeIndex node_index) = 0;
  virtual const NodeArg* GraphViewer__GetNodeArg(const GraphViewer* p, const std::string& name) = 0;

  virtual bool GraphViewer__IsSubgraph(const GraphViewer* p) = 0;
  virtual bool GraphViewer__IsConstantInitializer(const GraphViewer* p, const std::string& name, bool check_outer_scope) = 0;
  virtual int GraphViewer__NumberOfNodes(const GraphViewer* p) noexcept = 0;
  virtual int GraphViewer__MaxNodeIndex(const GraphViewer* p) noexcept = 0;

  virtual const std::vector<const NodeArg*>& GraphViewer__GetInputs(const GraphViewer* p) noexcept = 0;
  virtual const std::vector<const NodeArg*>& GraphViewer__GetOutputs(const GraphViewer* p) noexcept = 0;
  virtual const std::vector<const NodeArg*>& GraphViewer__GetValueInfo(const GraphViewer* p) noexcept = 0;

  virtual const Provider_InitializedTensorSet& GraphViewer__GetAllInitializedTensors(const GraphViewer* p) = 0;
  virtual bool GraphViewer__GetInitializedTensor(const GraphViewer* p, const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) = 0;
  virtual const std::unordered_map<std::string, int>& GraphViewer__DomainToVersionMap(const GraphViewer* p) = 0;

  virtual const std::vector<NodeIndex>& GraphViewer__GetNodesInTopologicalOrder(const GraphViewer* p) = 0;
  virtual const std::vector<const NodeArg*>& GraphViewer__GetInputsIncludingInitializers(const GraphViewer* p) noexcept = 0;

  // Path
  virtual PathString Path__ToPathString(const Path* p) noexcept = 0;

  // OpKernelContext
  virtual const Tensor* OpKernelContext__Input_Tensor(const OpKernelContext* p, int index) = 0;
  virtual Tensor* OpKernelContext__Output(OpKernelContext* p, int index, const TensorShape& shape) = 0;

  // OpKernelInfo
  virtual std::unique_ptr<OpKernelInfo> CopyOpKernelInfo(const OpKernelInfo& info) = 0;
  virtual void OpKernelInfo__operator_delete(OpKernelInfo* p) = 0;
  virtual Status OpKernelInfo__GetAttr_int64(const OpKernelInfo* p, const std::string& name, int64_t* value) = 0;
  virtual Status OpKernelInfo__GetAttr_float(const OpKernelInfo* p, const std::string& name, float* value) = 0;

  virtual const DataTransferManager& OpKernelInfo__GetDataTransferManager(const OpKernelInfo* p) noexcept = 0;
  virtual const KernelDef& OpKernelInfo__GetKernelDef(const OpKernelInfo* p) = 0;

  // Tensor
  virtual float* Tensor__MutableData_float(Tensor* p) = 0;
  virtual const float* Tensor__Data_float(const Tensor* p) = 0;

  virtual void* Tensor__MutableDataRaw(Tensor* p) noexcept = 0;
  virtual const void* Tensor__DataRaw(const Tensor* p) const noexcept = 0;

  virtual const TensorShape& Tensor__Shape(const Tensor* p) = 0;
  virtual size_t Tensor__SizeInBytes(const Tensor* p) = 0;
  virtual const OrtMemoryInfo& Tensor__Location(const Tensor* p) = 0;

  // AllocatorManager
  virtual void AllocatorManager__InsertAllocator(AllocatorManager* p, AllocatorPtr allocator) = 0;
  virtual AllocatorPtr AllocatorManager__GetAllocator(AllocatorManager* p, int id, OrtMemType mem_type) = 0;
};

extern ProviderHost* g_host;

#ifdef SHARED_PROVIDER

struct CPUIDInfo {
  static const CPUIDInfo& GetCPUIDInfo() { return g_host->CPUIDInfo__GetCPUIDInfo(); }

  bool HasAVX2() const { return g_host->CPUIDInfo__HasAVX2(this); }
  bool HasAVX512f() const { return g_host->CPUIDInfo__HasAVX512f(this); }

  PROVIDER_DISALLOW_ALL(CPUIDInfo)
};

namespace logging {

struct Logger {
  bool OutputIsEnabled(Severity severity, DataType data_type) const noexcept { return g_host->logging__Logger__OutputIsEnabled(this, severity, data_type); }

  PROVIDER_DISALLOW_ALL(Logger)
};

struct LoggingManager {
  static const Logger& DefaultLogger() { return g_host->logging__LoggingManager__DefaultLogger(); }

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
}  // namespace logging

namespace Utils {

struct DataTypeUtils {
  static const std::string* ToType(const ONNX_NAMESPACE::TypeProto& type_proto) { return g_host->Utils__DataTypeUtils__ToType(type_proto); }

  PROVIDER_DISALLOW_ALL(DataTypeUtils)
};

}  // namespace Utils
}

namespace ONNX_NAMESPACE {

struct int64s {
  int size() const { return g_host->int64s__size(this); }
  const int64_t& Get(int index) const { return g_host->int64s__Get(this, index); }

  PROVIDER_DISALLOW_ALL(int64s)
};

struct AttributeProto {
  static std::unique_ptr<AttributeProto> Create() { return g_host->AttributeProto__construct(); }
  void operator=(const AttributeProto& v) { g_host->AttributeProto__operator_assign(this, v); }
  static void operator delete(void* p) { g_host->AttributeProto__operator_delete(reinterpret_cast<AttributeProto*>(p)); }

  AttributeProto_AttributeType type() const { return g_host->AttributeProto__type(this); }
  int ints_size() const { return g_host->AttributeProto__ints_size(this); }
  int floats_size() const { return g_host->AttributeProto__floats_size(this); }
  int64_t ints(int i) const { return g_host->AttributeProto__ints(this, i); }
  float floats(int i) const { return g_host->AttributeProto__floats(this, i); }
  const int64s& ints() const { return g_host->AttributeProto__ints(this); }
  int64_t i() const { return g_host->AttributeProto__i(this); }
  float f() const { return g_host->AttributeProto__f(this); }
  void set_s(const ::std::string& value) { return g_host->AttributeProto__set_s(this, value); }
  const ::std::string& s() const { return g_host->AttributeProto__s(this); }
  void set_name(const ::std::string& value) { return g_host->AttributeProto__set_name(this, value); }
  void set_type(AttributeProto_AttributeType value) { return g_host->AttributeProto__set_type(this, value); }
  TensorProto* add_tensors() { return g_host->AttributeProto__add_tensors(this); }

  AttributeProto() = delete;
  AttributeProto(const AttributeProto&) = delete;
};

struct GraphProto {
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

struct ModelProto {
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

struct TensorProto {
  static void operator delete(void* p) { g_host->TensorProto__operator_delete(reinterpret_cast<TensorProto*>(p)); }
  void operator=(const TensorProto& v) { g_host->TensorProto__operator_assign(this, v); }

  bool has_data_location() const { return g_host->TensorProto__has_data_location(this); }
  TensorProto_DataLocation data_location() const { return TensorProto_DataLocation(g_host->TensorProto__data_location(this)); }

  TensorProto() = delete;
  TensorProto(const TensorProto&) = delete;
};

struct TensorProtos {
  TensorProto* Add() { return g_host->TensorProtos__Add(this); }

  PROVIDER_DISALLOW_ALL(TensorProtos)
};

struct TensorShapeProto_Dimension {
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

struct TensorShapeProto_Dimensions {
  IteratorHolder<TensorShapeProto_Dimension_Iterator, const TensorShapeProto_Dimension> begin() const { return g_host->TensorShapeProto_Dimensions__begin(this); }
  IteratorHolder<TensorShapeProto_Dimension_Iterator, const TensorShapeProto_Dimension> end() const { return g_host->TensorShapeProto_Dimensions__end(this); }

  PROVIDER_DISALLOW_ALL(TensorShapeProto_Dimensions)
};

struct TensorShapeProto {
  int dim_size() const { return g_host->TensorShapeProto__dim_size(this); }
  const TensorShapeProto_Dimensions& dim() const { return g_host->TensorShapeProto__dim(this); }
  const TensorShapeProto_Dimension& dim(int index) const { return g_host->TensorShapeProto__dim(this, index); }
  TensorShapeProto_Dimension* mutable_dim(int index) { return g_host->TensorShapeProto__mutable_dim(this, index); }
  void clear_dim() { return g_host->TensorShapeProto__clear_dim(this); }
  TensorShapeProto_Dimension* add_dim() { return g_host->TensorShapeProto__add_dim(this); }

  PROVIDER_DISALLOW_ALL(TensorShapeProto)
};

struct TypeProto_Tensor {
  const TensorShapeProto& shape() const { return g_host->TypeProto_Tensor__shape(this); }
  TensorShapeProto* mutable_shape() { return g_host->TypeProto_Tensor__mutable_shape(this); }
  int32_t elem_type() const { return g_host->TypeProto_Tensor__elem_type(this); }

  PROVIDER_DISALLOW_ALL(TypeProto_Tensor)
};

struct TypeProto {
  const TypeProto_Tensor& tensor_type() const { return g_host->TypeProto__tensor_type(this); }
  TypeProto_Tensor* mutable_tensor_type() { return g_host->TypeProto__mutable_tensor_type(this); }

  PROVIDER_DISALLOW_ALL(TypeProto)
};

struct ValueInfoProto {
  const TypeProto& type() const { return g_host->ValueInfoProto__type(this); }
  TypeProto* mutable_type() { return g_host->ValueInfoProto__mutable_type(this); }

  void operator=(const ValueInfoProto& v) { g_host->ValueInfoProto__operator_assign(this, v); }

  ValueInfoProto() = delete;
  ValueInfoProto(const ValueInfoProto&) = delete;
  static void operator delete(void*) = delete;
};

struct ValueInfoProtos {
  ValueInfoProto* Add() { return g_host->ValueInfoProtos__Add(this); }
  const ValueInfoProto& operator[](int index) const { return g_host->ValueInfoProtos__operator_array(this, index); }

  PROVIDER_DISALLOW_ALL(ValueInfoProtos)
};

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

struct ComputeCapability {
  static std::unique_ptr<ComputeCapability> Create(std::unique_ptr<IndexedSubGraph> t_sub_graph) { return g_host->ComputeCapability__construct(std::move(t_sub_graph)); }
  static void operator delete(void* p) { g_host->ComputeCapability__operator_delete(reinterpret_cast<ComputeCapability*>(p)); }

  std::unique_ptr<IndexedSubGraph>& SubGraph() { return g_host->ComputeCapability__SubGraph(this); }

  ComputeCapability() = delete;
  ComputeCapability(const ComputeCapability&) = delete;
  void operator=(const ComputeCapability&) = delete;
};

struct DataTransferManager {
  Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const { return g_host->DataTransferManager__CopyTensor(this, src, dst, exec_queue_id); }

  PROVIDER_DISALLOW_ALL(DataTransferManager)
};

struct IndexedSubGraph_MetaDef {
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

struct IndexedSubGraph {
  static std::unique_ptr<IndexedSubGraph> Create() { return g_host->IndexedSubGraph__construct(); }
  static void operator delete(void* p) { g_host->IndexedSubGraph__operator_delete(reinterpret_cast<IndexedSubGraph*>(p)); }

  std::vector<onnxruntime::NodeIndex>& Nodes() { return g_host->IndexedSubGraph__Nodes(this); }

  void SetMetaDef(std::unique_ptr<IndexedSubGraph_MetaDef>&& meta_def_) { return g_host->IndexedSubGraph__SetMetaDef(this, std::move(*reinterpret_cast<std::unique_ptr<IndexedSubGraph_MetaDef>*>(&meta_def_))); }
  const IndexedSubGraph_MetaDef* GetMetaDef() const { return reinterpret_cast<const IndexedSubGraph_MetaDef*>(g_host->IndexedSubGraph__GetMetaDef(this)); }

  IndexedSubGraph() = delete;
  IndexedSubGraph(const IndexedSubGraph&) = delete;
  void operator=(const IndexedSubGraph&) = delete;
};

struct KernelDef {
  static void operator delete(void* p) { g_host->KernelDef__operator_delete(reinterpret_cast<KernelDef*>(p)); }

  int ExecQueueId() const { return g_host->KernelDef__ExecQueueId(this); }

  KernelDef() = delete;
  KernelDef(const KernelDef*) = delete;
  void operator=(const KernelDef&) = delete;
};
#endif

using BuildKernelCreateInfoFn = KernelCreateInfo (*)();

#ifdef SHARED_PROVIDER
struct KernelDefBuilder {
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
  KernelDefBuilder& OutputMemoryType(OrtMemType type, int input_index) {
    g_host->KernelDefBuilder__OutputMemoryType(this, type, input_index);
    return *this;
  }
  KernelDefBuilder& ExecQueueId(int queue_id) {
    g_host->KernelDefBuilder__ExecQueueId(this, queue_id);
    return *this;
  }

  std::unique_ptr<KernelDef> Build() { return g_host->KernelDefBuilder__Build(this); }

  KernelDefBuilder() = delete;
  KernelDefBuilder(const KernelDefBuilder&) = delete;
  void operator=(const KernelDefBuilder&) = delete;
};

struct KernelRegistry {
  static std::shared_ptr<KernelRegistry> Create() { return g_host->KernelRegistry__construct(); }
  static void operator delete(void* p) { g_host->KernelRegistry__operator_delete(reinterpret_cast<KernelRegistry*>(p)); }

  Status Register(KernelCreateInfo&& create_info) { return g_host->KernelRegistry__Register(this, std::move(create_info)); }

  KernelRegistry() = delete;
  KernelRegistry(const KernelRegistry&) = delete;
  void operator=(const KernelRegistry&) = delete;
};

struct PrimitiveDataTypeBase {
  int32_t GetDataType() const { return g_host->PrimitiveDataTypeBase__GetDataType(this); }

  PROVIDER_DISALLOW_ALL(PrimitiveDataTypeBase)
};

class DataTypeImpl {
 public:
  size_t Size() const { return g_host->DataTypeImpl__Size(this); }

  template <typename T>
  static MLDataType GetType();
  template <typename elemT>
  static MLDataType GetTensorType();

  static const std::vector<MLDataType>& AllFixedSizeTensorTypes() { return g_host->DataTypeImpl__AllFixedSizeTensorTypes(); }
  static const std::vector<MLDataType>& AllTensorTypes() { return g_host->DataTypeImpl__AllTensorTypes(); }

  const PrimitiveDataTypeBase* AsPrimitiveDataType() const { return g_host->DataTypeImpl__AsPrimitiveDataType(this); }

  static const char* ToString(MLDataType type) { return g_host->DataTypeImpl__ToString(type); }

  PROVIDER_DISALLOW_ALL(DataTypeImpl)
};

struct Function {
  const Graph& Body() const { return g_host->Function__Body(this); }

  PROVIDER_DISALLOW_ALL(Function)
};

struct Node {
  const std::string& Name() const noexcept { return g_host->Node__Name(this); }
  const std::string& Description() const noexcept { return g_host->Node__Description(this); }
  const std::string& Domain() const noexcept { return g_host->Node__Domain(this); }
  const std::string& OpType() const noexcept { return g_host->Node__OpType(this); }

  const Function* GetFunctionBody() const noexcept { return g_host->Node__GetFunctionBody(this); }

  ConstPointerContainer<std::vector<NodeArg*>> ImplicitInputDefs() const noexcept { return g_host->Node__ImplicitInputDefs(this); }

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

struct NodeArg {
  const std::string& Name() const noexcept { return g_host->NodeArg__Name(this); }
  const ONNX_NAMESPACE::TensorShapeProto* Shape() const { return g_host->NodeArg__Shape(this); }
  ONNX_NAMESPACE::DataType Type() const noexcept { return g_host->NodeArg__Type(this); }
  const NodeArgInfo& ToProto() const noexcept { return g_host->NodeArg__ToProto(this); }
  bool Exists() const noexcept { return g_host->NodeArg__Exists(this); }
  const ONNX_NAMESPACE::TypeProto* TypeAsProto() const noexcept { return g_host->NodeArg__TypeAsProto(this); }

  PROVIDER_DISALLOW_ALL(NodeArg)
};

struct NodeAttributes {
  static std::unique_ptr<NodeAttributes> Create() { return g_host->NodeAttributes__construct(); }
  void operator=(const NodeAttributes& v) { return g_host->NodeAttributes__operator_assign(this, v); }
  static void operator delete(void* p) { g_host->NodeAttributes__operator_delete(reinterpret_cast<NodeAttributes*>(p)); }

  size_t size() const { return g_host->NodeAttributes__size(this); }
  void clear() noexcept { g_host->NodeAttributes__clear(this); }
  size_t count(const std::string& keyval) const { return g_host->NodeAttributes__count(this, keyval); }
  ONNX_NAMESPACE::AttributeProto& operator[](const std::string& string) { return g_host->NodeAttributes__operator_array(this, string); }
  const ONNX_NAMESPACE::AttributeProto& at(const std::string& string) const { return g_host->NodeAttributes__at(this, string); }

  IteratorHolder<NodeAttributes_Iterator, std::pair<std::string&, ONNX_NAMESPACE::AttributeProto&>> begin() const { return g_host->NodeAttributes__begin(this); }
  IteratorHolder<NodeAttributes_Iterator, std::pair<std::string&, ONNX_NAMESPACE::AttributeProto&>> end() const { return g_host->NodeAttributes__end(this); }
  IteratorHolder<NodeAttributes_Iterator, std::pair<std::string&, ONNX_NAMESPACE::AttributeProto&>> find(const std::string& key) const { return g_host->NodeAttributes__find(this, key); }
  void insert(const NodeAttributes& v) { return g_host->NodeAttributes__insert(this, v); }

  NodeAttributes() = delete;
  NodeAttributes(const NodeAttributes&) = delete;
};

struct Model {
  static void operator delete(void* p) { g_host->Model__operator_delete(reinterpret_cast<Model*>(p)); }

  Graph& MainGraph() { return g_host->Model__MainGraph(this); }

  std::unique_ptr<ONNX_NAMESPACE::ModelProto> ToProto() { return g_host->Model__ToProto(this); }

  Model() = delete;
  Model(const Model&) = delete;
  void operator=(const Model&) = delete;
};

struct Graph {
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

struct GraphViewer {
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
  const std::vector<const NodeArg*>& GetValueInfo() const noexcept { return g_host->GraphViewer__GetValueInfo(this); }

  const Provider_InitializedTensorSet& GetAllInitializedTensors() const noexcept { return g_host->GraphViewer__GetAllInitializedTensors(this); }
  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const { return g_host->GraphViewer__GetInitializedTensor(this, tensor_name, value); }

  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept { return g_host->GraphViewer__DomainToVersionMap(this); }

  const std::vector<NodeIndex>& GetNodesInTopologicalOrder() const { return g_host->GraphViewer__GetNodesInTopologicalOrder(this); }
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept { return g_host->GraphViewer__GetInputsIncludingInitializers(this); }

  GraphViewer() = delete;
  GraphViewer(const GraphViewer&) = delete;
  void operator=(const GraphViewer&) = delete;
};

struct Path {
  PathString ToPathString() const noexcept { return g_host->Path__ToPathString(this); }

  PROVIDER_DISALLOW_ALL(Path)
};

#endif

#ifdef SHARED_PROVIDER
struct OpKernelContext {
  const Tensor* Input_Tensor(int index) const { return g_host->OpKernelContext__Input_Tensor(this, index); }

  template <typename T>
  const T* Input(int index) const;

  Tensor* Output(int index, const TensorShape& shape) { return g_host->OpKernelContext__Output(this, index, shape); }

  PROVIDER_DISALLOW_ALL(OpKernelContext)
};

template <>
inline const Tensor* OpKernelContext::Input<Tensor>(int index) const {
  return Input_Tensor(index);
}

struct OpKernelInfo {
  static void operator delete(void* p) { g_host->OpKernelInfo__operator_delete(reinterpret_cast<OpKernelInfo*>(p)); }

  template <typename T>
  Status GetAttr(const std::string& name, T* value) const;

  Status GetAttr(const std::string& name, int64_t* value) const { return g_host->OpKernelInfo__GetAttr_int64(this, name, value); }
  Status GetAttr(const std::string& name, float* value) const { return g_host->OpKernelInfo__GetAttr_float(this, name, value); }

  const DataTransferManager& GetDataTransferManager() const noexcept { return g_host->OpKernelInfo__GetDataTransferManager(this); }
  const KernelDef& GetKernelDef() const { return g_host->OpKernelInfo__GetKernelDef(this); }

  OpKernelInfo() = delete;
  OpKernelInfo(const OpKernelInfo&) = delete;
  void operator=(const OpKernelInfo&) = delete;
};

template <>
inline Status OpKernelInfo::GetAttr<int64_t>(const std::string& name, int64_t* value) const {
  return GetAttr(name, value);
}

template <>
inline Status OpKernelInfo::GetAttr<float>(const std::string& name, float* value) const {
  return GetAttr(name, value);
}

struct Tensor {
  float* MutableData_float() { return g_host->Tensor__MutableData_float(this); }
  const float* Data_float() const { return g_host->Tensor__Data_float(this); }

  template <typename T>
  T* MutableData();

  template <typename T>
  const T* Data() const;

  void* MutableDataRaw() noexcept { return g_host->Tensor__MutableDataRaw(this); }
  const void* DataRaw() const noexcept { return g_host->Tensor__DataRaw(this); }

  const TensorShape& Shape() const { return g_host->Tensor__Shape(this); }
  size_t SizeInBytes() const { return g_host->Tensor__SizeInBytes(this); }
  const OrtMemoryInfo& Location() const { return g_host->Tensor__Location(this); }

  PROVIDER_DISALLOW_ALL(Tensor)
};

template <>
inline float* Tensor::MutableData<float>() { return MutableData_float(); }

template <>
inline const float* Tensor::Data<float>() const { return Data_float(); }

namespace utils {

inline bool HasDimValue(const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim) {
  return dim.value_case() == ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimValue;
}

}  // namespace utils

#endif

}  // namespace onnxruntime
