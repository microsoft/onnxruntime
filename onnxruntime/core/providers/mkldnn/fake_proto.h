// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#define NO_PROTOBUF
#define ONNX_NAMESPACE onnx

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include "core/common/common.h"
#include "core/common/const_pointer_container.h"
#include "core/session/onnxruntime_c_api.h"
//#include "core/framework/op_node_proto_helper.h"
//#include "core/graph/graph.h"
//#include "core/providers/providers.h"

namespace google {
namespace protobuf {
template <typename T>
struct RepeatedPtrField {};
}  // namespace protobuf
}  // namespace google

namespace onnx {
enum AttributeProto_AttributeType;

using DataType = const std::string*;
using OperatorSetVersion = int;

enum AttributeProto_AttributeType {
  AttributeProto_AttributeType_UNDEFINED = 0,
  AttributeProto_AttributeType_FLOAT = 1,
  AttributeProto_AttributeType_INT = 2,
  AttributeProto_AttributeType_STRING = 3,
  AttributeProto_AttributeType_TENSOR = 4,
  AttributeProto_AttributeType_GRAPH = 5,
  AttributeProto_AttributeType_SPARSE_TENSOR = 11,
  AttributeProto_AttributeType_FLOATS = 6,
  AttributeProto_AttributeType_INTS = 7,
  AttributeProto_AttributeType_STRINGS = 8,
  AttributeProto_AttributeType_TENSORS = 9,
  AttributeProto_AttributeType_GRAPHS = 10,
  AttributeProto_AttributeType_SPARSE_TENSORS = 12
};

enum Version {
  _START_VERSION = 0,
  IR_VERSION_2017_10_10 = 1,
  IR_VERSION_2017_10_30 = 2,
  IR_VERSION_2017_11_3 = 3,
  IR_VERSION_2019_1_22 = 4,
  IR_VERSION_2019_3_18 = 5,
  IR_VERSION = 6
};

enum OperatorStatus {
  EXPERIMENTAL = 0,
  STABLE = 1
};

class ValueInfoProto {};
class TensorProto {
};

class TensorShapeProto {
 public:
  int dim_size() const;
};
class TypeProto;
class OpSchema {
 public:
  OperatorSetVersion SinceVersion() const;
};

class AttributeProto {
 public:
  ::onnx::AttributeProto_AttributeType type() const;
  int ints_size() const;
  int64_t ints(int i) const;
  int64_t i() const;
  float f() const;
  void set_s(const ::std::string& value);
  const ::std::string& s() const;
  void set_name(const ::std::string& value);
  void set_type(::onnx::AttributeProto_AttributeType value);
  ::onnx::TensorProto* add_tensors();
};

class GraphProto {
};

class SparseTensorProto {
};

class NodeProto {};

class FunctionProto {};
}  // namespace onnx

namespace onnxruntime {

constexpr const char* kOnnxDomain = "";

class Graph;
class DataTypeImpl;
using MLDataType = const DataTypeImpl*;

/**
 * \brief Base class for MLDataType
 *
 */
class DataTypeImpl {
 public:
  virtual ~DataTypeImpl() = default;

  template <typename T>
  static MLDataType GetType();
  template <typename elemT>
  static MLDataType GetTensorType();
};

struct IExecutionProviderFactory;

using NodeIndex = size_t;
using NodeAttributes = std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto>;

struct IndexedSubGraph {
  struct MetaDef {
    std::string name;    ///< Name of customized SubGraph/FunctionProto
    std::string domain;  ///< Domain of customized SubGraph/FunctionProto
    int since_version;   ///< Since version of customized SubGraph/FunctionProto.

    ONNX_NAMESPACE::OperatorStatus status;  ///< Status of customized SubGraph/FunctionProto.

    std::vector<std::string> inputs;   ///< Inputs of customized SubGraph/FunctionProto.
    std::vector<std::string> outputs;  ///< Outputs of customized SubGraph/FunctionProto.
    NodeAttributes attributes;         ///< Attributes of customized SubGraph/FunctionProto.

    std::string doc_string;  ///< Doc string of customized SubGraph/FunctionProto.
  };

  /** Nodes covered by this subgraph. The NodeIndex values are from the parent Graph.*/
  std::vector<onnxruntime::NodeIndex> nodes;

  void SetMetaDef(std::unique_ptr<MetaDef>& meta_def_);
};

class NodeArg {
 public:
  /** Gets the name. */
  const std::string& Name() const noexcept;

  /** Gets the shape if NodeArg is for a Tensor.
  @returns TensorShapeProto if shape is set. nullptr if there's no shape specified. */
  const ONNX_NAMESPACE::TensorShapeProto* Shape() const;

  /** Gets the data type. */
  ONNX_NAMESPACE::DataType Type() const noexcept;
};

class Node {
 public:
  /** Gets the Node's operator type. */
  const std::string& OpType() const noexcept;

  /** Gets the Node's OpSchema.
  @remarks The graph containing this node must be resolved, otherwise nullptr will be returned. */
  const ONNX_NAMESPACE::OpSchema* Op() const noexcept;

  /** Gets the Node's input definitions.
  @remarks requires ConstPointerContainer wrapper to apply const to the NodeArg pointers so access is read-only. */
  ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept;
  //  { return ConstPointerContainer<std::vector<NodeArg*>>(definitions_.input_defs); }

  ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept;
  //  { return ConstPointerContainer<std::vector<NodeArg*>>(definitions_.output_defs);  }

  /** Gets the Node's NodeIndex. */
  NodeIndex Index() const noexcept;

  /** Gets the Node's attributes. */
  const NodeAttributes& GetAttributes() const noexcept;

  /** Gets the number of input edges to this Node */
  size_t GetInputEdgesCount() const noexcept;

  /** Gets the number of output edges from this Node */
  size_t GetOutputEdgesCount() const noexcept;

  class NodeConstIterator {
   public:
    NodeConstIterator();

    bool operator==(const NodeConstIterator& p_other) const;

    bool operator!=(const NodeConstIterator& p_other) const;

    void operator++();
    void operator--();

    const Node& operator*() const;
    const Node* operator->() const;
  };

  /** Gets an iterator to the beginning of the input nodes to this Node. */
  NodeConstIterator InputNodesBegin() const noexcept;
  /** Gets an iterator to the end of the input nodes to this Node. */
  NodeConstIterator InputNodesEnd() const noexcept;
};

class GraphNodes;

using InitializedTensorSet = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>;

class GraphViewer {
 public:
  /**
	Construct a GraphViewer from the provided Graph instance.
	*/
  explicit GraphViewer(const Graph& graph);

  /** Gets the Graph name. */
  const std::string& Name() const noexcept;

  /** Gets the Graph description. */
  const std::string& Description() const noexcept;

  /**
	Gets a tensor created from an initializer.
	@param tensor_name The tensor name
	@param[out] value Sets the pointer to the TensorProto if found, or nullptr if not.
	@returns True if found. False if not.
	*/
  bool GetInitializedTensor(const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) const;

  /** Returns true if an initializer value can be overridden by a graph input with the same name. */
  bool CanOverrideInitializer() const noexcept;

  /**
	Gets the Graph inputs, excluding initializers.
	@returns Collection of NodeArg pointers for the graph inputs, excluding inputs that have matching initializers.
	@remarks No nullptr values in the returned collection. The order will be the same as in the GraphProto.
	*/
  const std::vector<const NodeArg*>& GetInputs() const noexcept;

  /**
	Gets the Graph inputs, including any initializers.
	@returns Collection of NodeArg pointers for all the graph inputs.
	@remarks No nullptr values in the returned collection. The order will be the same as in the GraphProto.
	*/
  const std::vector<const NodeArg*>& GetInputsIncludingInitializers() const noexcept;

  /**
	Gets the Graph outputs.
	@returns Collection of NodeArg pointers for all the graph outputs.
	@remarks No nullptr values in the returned collection. The order will be the same as in the GraphProto.
	*/
  const std::vector<const NodeArg*>& GetOutputs() const noexcept;

  /** Gets all ValueInfo NodeArg instances in the Graph. */
  const std::vector<const NodeArg*>& GetValueInfo() const noexcept;

  /**
	Gets the Node instance at the specified index.
	@param node_index Index to retrieve Node from.
	@remarks May return nullptr if index no longer points to a valid node due to the node being freed.
	*/
  const Node* GetNode(NodeIndex node_index) const;

  /**  Gets an iterator over all the valid Nodes in the Graph. */
  const GraphNodes& Nodes() const noexcept;

  /** Gets the number of valid nodes in the Graph. */
  int NumberOfNodes() const noexcept;

  /** Gets the maximum NodeIndex value used by Nodes in the Graph. */
  int MaxNodeIndex() const noexcept;

  /** Gets the NodeIndex values for the Graph nodes, sorted into topological order. */
  const std::vector<NodeIndex>& GetNodesInTopologicalOrder() const;

  /**
	Gets the NodeIndex values for the root nodes in the Graph.
	The root nodes are the topmost nodes in the Graph that receive inputs from the Graph inputs
	and no other nodes in the Graph.
	*/
  const std::vector<NodeIndex>& GetRootNodes() const;

  /** Gets all tensors created from initializers. */
  const InitializedTensorSet& GetAllInitializedTensors() const noexcept;

  /**
	Gets the NodeArg instance for the given name.
	@returns A NodeArg if found, a nullptr if not.
	*/
  const NodeArg* GetNodeArg(const std::string& name) const;

  /** Gets the map of operator domains to their opset versions. */
  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept;
  //{
  //  return graph_->DomainToVersionMap();
  //}

  /** Checks if this is a Subgraph */
  bool IsSubgraph() const;

  /**
	returns true if 'name' is an initializer, and is constant and cannot be overridden at runtime.
	@param check_outer_scope If true and the 'graph_' is a subgraph, check parent graph/s for 'name' if not found in 'graph_'.
	*/
  bool IsConstantInitializer(const std::string& name, bool check_outer_scope) const;

  /** Get the Node containing this Graph if IsSubgraph is true. Returns nullptr otherwise. */
  //const Node* ParentNode() const noexcept { return graph_->ParentNode(); }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphViewer);

  const Graph* graph_;

  // The NodeIndex values of the graph nodes sorted in topological order.
  std::vector<NodeIndex> nodes_in_topological_order_;
  // Graph root nodes.
  std::vector<NodeIndex> root_nodes_;
};

class TensorShape {
 public:
  TensorShape();

#if 0
  TensorShape(const TensorShape& /*other*/) = default;
  TensorShape& operator=(const TensorShape& /*other*/) = default;

  TensorShape(TensorShape&& /*other*/) = default;
  TensorShape& operator=(TensorShape&& /*other*/) = default;
  TensorShape(const std::vector<int64_t>& dims) : std::vector<int64_t>(dims) {}
  TensorShape(std::vector<int64_t>&& dims) : std::vector<int64_t>(std::move(dims)) {}
#endif

  TensorShape(const std::initializer_list<int64_t>& dims);

#if 0
  TensorShape(const int64_t* dimension_sizes, size_t dimension_count);
  TensorShape(const std::vector<int64_t>& dims, size_t start, size_t end);
#endif

  const int64_t& operator[](size_t idx) const;
  int64_t& operator[](size_t idx);

#if 0
  bool operator==(const TensorShape& other) const noexcept;
  bool operator!=(const TensorShape& other) const noexcept;
#endif
  size_t NumDimensions() const noexcept;

#if 0

  void CopyDims(int64_t* dims, size_t num_dims) const;
#endif

  const std::vector<int64_t>& GetDims() const;

  int64_t Size() const;

#if 0

  /**
     Return the total number of elements up to the specified dimension.
     If the dimension interval is empty (dimension == 0), return 1.
     @param dimension Return size up to this dimension. Value must be between 0 and this->NumDimensions(), inclusive.
  */
  int64_t SizeToDimension(size_t dimension) const;

  /**
     Return the total number of elements from the specified dimension to the end of the tensor shape.
     If the dimension interval is empty (dimension == this->NumDimensions()), return 1.
     @param dimension Return size from this dimension to the end. Value must be between 0 and this->NumDimensions(),
                      inclusive.
  */
  int64_t SizeFromDimension(size_t dimension) const;

  /**
     Return a new TensorShape of the dimensions from dimstart to dimend.
  */
  TensorShape Slice(size_t dimstart, size_t dimend) const;

  /**
     Return a new TensorShape of the dimensions from dimstart to end.
  */
  TensorShape Slice(size_t dimstart) const;

  /**
     output dimensions nicely formatted
  */
  std::string ToString() const;

  /**
     Calculate size between start and end.
     Assumes start and end are between 0 and this->NumDimensions(), inclusive, and that
     start < end.
  */
  int64_t SizeHelper(size_t start, size_t end) const;

  /**
     empty shape or 1D shape (1) is regarded as scalar tensor
  */
  bool IsScalar() const;

  static const TensorShape& ReinterpretBaseType(const std::vector<int64_t>& dimensions);
#endif

  void* this_;
};

class Tensor final {
 public:
  Tensor() = default;

  const TensorShape& Shape() const noexcept;

  template <typename T>
  T* MutableData();

  template <typename T>
  const T* Data() const;
};

class OpKernelInfo {
 public:
  template <typename T>
  Status GetAttr(const std::string& name, T* value) const;
};

class OpKernel {
 public:
  explicit OpKernel(const OpKernelInfo& info);
};

class OpKernelContext {
 public:
  template <typename T>
  const T* Input(int index) const;
  template <typename T>
  T* Output(int index);
  Tensor* Output(int index, const TensorShape& shape);
};

constexpr const char* kMSDomain = "com.microsoft";
constexpr const char* kMklDnnExecutionProvider = "MKLDNNExecutionProvider";

class KernelDef {
};

using KernelCreateFn = std::function<OpKernel*(const OpKernelInfo& info)>;
using KernelCreatePtrFn = std::add_pointer<OpKernel*(const OpKernelInfo& info)>::type;

struct KernelCreateInfo {
  std::unique_ptr<KernelDef> kernel_def;  // Owned and stored in the global kernel registry.
  KernelCreateFn kernel_create_func;
  Status status;

  KernelCreateInfo(std::unique_ptr<KernelDef> definition,
                   KernelCreateFn create_func)
      : kernel_def(std::move(definition)),
        kernel_create_func(create_func) {}

  KernelCreateInfo(KernelCreateInfo&& other) noexcept
      : kernel_def(std::move(other.kernel_def)),
        kernel_create_func(std::move(other.kernel_create_func)) {}
};

using BuildKernelCreateInfoFn = KernelCreateInfo (*)();

class KernelDefBuilder {
 public:
  explicit KernelDefBuilder();

  KernelDefBuilder& SetName(const char* op_name);
  KernelDefBuilder& SetDomain(const char* domain);
  KernelDefBuilder& SinceVersion(int since_version);
  KernelDefBuilder& Provider(const char* provider_type);
  KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType supported_type);

  std::unique_ptr<KernelDef> Build();

  void* this_;
};

class KernelRegistry {
 public:
  Status Register(KernelCreateInfo&& create_info);
};

struct ComputeCapability {
  ComputeCapability(std::unique_ptr<IndexedSubGraph> t_sub_graph);
};

using AllocateFunc = void* (*)(void*, size_t, size_t);
using DestroyFunc = void (*)(void*, void*);
using AllocatorHandle = void*;

typedef struct {
  //right now we only include allocation for host memory
  AllocateFunc allocate_func;
  DestroyFunc release_func;
  AllocatorHandle allocator_handle;
  const char* node_name;
} ComputeContext;

using FunctionState = void*;

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

struct OrtDevice {
};

class OrtMemoryInfo {
 public:
  constexpr OrtMemoryInfo(const char* name_, OrtAllocatorType type_, OrtDevice device_ = OrtDevice(), int id_ = 0, OrtMemType mem_type_ = OrtMemTypeDefault);
};

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

class IAllocator {
 public:
  virtual ~IAllocator() = default;
  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p) = 0;
  virtual const OrtMemoryInfo& Info() const = 0;

  template <typename T>
  static IAllocatorUniquePtr<T> MakeUniquePtr(std::shared_ptr<IAllocator> allocator, size_t count_or_bytes);
};

class IDeviceAllocator : public IAllocator {
 public:
  ~IDeviceAllocator() override = default;
  void* Alloc(size_t size) override = 0;
  void Free(void* p) override = 0;
  const OrtMemoryInfo& Info() const override = 0;
  virtual bool AllowsArena() const { return true; }
};

class CPUIDInfo {
 public:
  static const CPUIDInfo& GetCPUIDInfo();

  bool HasAVX2() const;
  bool HasAVX512f() const;
};

class CPUAllocator : public IDeviceAllocator {
 public:
  explicit CPUAllocator(std::unique_ptr<OrtMemoryInfo> memory_info);
  CPUAllocator();

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  const OrtMemoryInfo& Info() const override;
};

using DeviceAllocatorFactory = std::function<std::unique_ptr<IDeviceAllocator>(int)>;

struct DeviceAllocatorRegistrationInfo {
  OrtMemType mem_type;
  DeviceAllocatorFactory factory;
  size_t max_mem;
};

using AllocatorPtr = std::shared_ptr<IAllocator>;

AllocatorPtr CreateAllocator(DeviceAllocatorRegistrationInfo info, int device_id = 0);

class IArenaAllocator : public IAllocator {
};

class DummyArena : public IArenaAllocator {
 public:
  explicit DummyArena(std::unique_ptr<IDeviceAllocator> resource_allocator);

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  const OrtMemoryInfo& Info() const override;
};

class IExecutionProvider {
 protected:
  IExecutionProvider(const std::string& type);

 public:
  virtual ~IExecutionProvider() = default;

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const;

  virtual std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const onnxruntime::GraphViewer& graph,
                                                                        const std::vector<const KernelRegistry*>& /*kernel_registries*/) const;

  virtual common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                 std::vector<NodeComputeInfo>& node_compute_funcs);

  virtual AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const;
  void InsertAllocator(AllocatorPtr allocator);
};

namespace logging {

enum class Severity {
  kVERBOSE = 0,
  kINFO = 1,
  kWARNING = 2,
  kERROR = 3,
  kFATAL = 4
};

enum class DataType {
  SYSTEM = 0,  ///< System data.
  USER = 1     ///< Contains potentially sensitive user data.
};

struct Category {
  static const char* onnxruntime;  ///< General output
  static const char* System;       ///< Log output regarding interactions with the host system
  // TODO: What other high level categories are meaningful? Model? Optimizer? Execution?
};

constexpr const char* SEVERITY_PREFIX = "VIWEF";

class Logger {
 public:
  bool OutputIsEnabled(Severity severity, DataType data_type) const noexcept;
};

class LoggingManager {
 public:
  static const Logger& DefaultLogger();
};

class Capture {
 public:
  Capture(const Logger& logger, logging::Severity severity, const char* category,
          logging::DataType dataType, const CodeLocation& location);

  std::ostream& Stream() noexcept;
};
}  // namespace logging

enum class AutoPadType {
  NOTSET = 0,
  VALID = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
};

inline AutoPadType StringToAutoPadType(const std::string& str) {
  if (str.empty()) {
    return AutoPadType::NOTSET;
  }
  if (str == "NOTSET") {  // in onnx spec, default value is "NOTSET"
    return AutoPadType::NOTSET;
  }
  if (str == "VALID") {
    return AutoPadType::VALID;
  }
  if (str == "SAME_UPPER") {
    return AutoPadType::SAME_UPPER;
  }
  if (str == "SAME_LOWER") {
    return AutoPadType::SAME_LOWER;
  }
  ORT_ENFORCE(false, "Unknown AutoPadType String");
}

}  // namespace onnxruntime

struct OrtSessionOptions {
  std::vector<std::shared_ptr<onnxruntime::IExecutionProviderFactory>> provider_factories;
};

#define ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name) \
  provider##_##name##_##domain##_ver##ver

#define ONNX_OPERATOR_KERNEL_EX(name, domain, ver, provider, builder, ...)                                            \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name);                                                 \
  template <>                                                                                                         \
  KernelCreateInfo                                                                                                    \
  BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)>() {                             \
    return KernelCreateInfo(                                                                                          \
        builder.SetName(#name)                                                                                        \
            .SetDomain(domain)                                                                                        \
            .SinceVersion(ver)                                                                                        \
            .Provider(provider)                                                                                       \
            .Build(),                                                                                                 \
        static_cast<KernelCreatePtrFn>([](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); })); \
  }

#define CREATE_MESSAGE(logger, severity, category, datatype) \
  ::onnxruntime::logging::Capture(logger, ::onnxruntime::logging::Severity::k##severity, category, datatype, ORT_WHERE)

// iostream style logging. Capture log info in Message, and push to the logger in ~Message.
#define LOGS_CATEGORY(logger, severity, category)                                                                        \
  if ((logger).OutputIsEnabled(::onnxruntime::logging::Severity::k##severity, ::onnxruntime::logging::DataType::SYSTEM)) \
  CREATE_MESSAGE(logger, severity, category, ::onnxruntime::logging::DataType::SYSTEM).Stream()

#define LOGS_DEFAULT_CATEGORY(severity, category) \
  LOGS_CATEGORY(::onnxruntime::logging::LoggingManager::DefaultLogger(), severity, category)

#define LOGS_DEFAULT(severity) \
  LOGS_DEFAULT_CATEGORY(severity, ::onnxruntime::logging::Category::onnxruntime)
