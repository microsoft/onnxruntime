// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the Onnxruntime side of the bridge to allow providers to be built as a DLL
// It implements onnxruntime::ProviderHost

#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/framework/data_types.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/provider_shutdown.h"
#include "core/graph/model.h"
#include "core/platform/env.h"
#include "core/providers/common.h"
#include "core/session/inference_session.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

#ifdef USE_TENSORRT
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/providers/cuda/cuda_common.h"
#endif

namespace onnxruntime {
// These Provider types are really just internal types, so we #define PROVIDER_BRIDGE_ORT so that only these definitions are seen by provider_interfaces.h
// Users of provider_interfaces.h (through provider_api.h) will see the wrappers that call through the provider shared interface which is implemented by this file
using Provider_int64s = google::protobuf::RepeatedField<int64_t>;
using Provider_AttributeProto = ONNX_NAMESPACE::AttributeProto;
using Provider_GraphProto = ONNX_NAMESPACE::GraphProto;
using Provider_ModelProto = ONNX_NAMESPACE::ModelProto;
using Provider_NodeProto = ONNX_NAMESPACE::NodeProto;
using Provider_TensorProto = ONNX_NAMESPACE::TensorProto;
using Provider_TensorProtos = google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::TensorProto>;
using Provider_TensorShapeProto_Dimension = ONNX_NAMESPACE::TensorShapeProto_Dimension;
using Provider_TensorShapeProto_Dimensions = google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::TensorShapeProto_Dimension>;
using Provider_TensorShapeProto = ONNX_NAMESPACE::TensorShapeProto;
using Provider_TypeProto_Tensor = ONNX_NAMESPACE::TypeProto_Tensor;
using Provider_TypeProto = ONNX_NAMESPACE::TypeProto;
using Provider_ValueInfoProto = ONNX_NAMESPACE::ValueInfoProto;
using Provider_ValueInfoProtos = google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::ValueInfoProto>;

using IndexedSubGraph_MetaDef = IndexedSubGraph::MetaDef;
}  // namespace onnxruntime

#define PROVIDER_BRIDGE_ORT
#include "core/common/cpuid_info.h"
#include "onnx/common/stl_backports.h"
#include "core/common/logging/logging.h"
#include "core/providers/shared_library/provider_interfaces.h"

#include "core/providers/dnnl/dnnl_provider_factory.h"
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#include "core/providers/openvino/openvino_provider_factory.h"

// The filename extension for a shared library is different per platform
#ifdef _WIN32
#define LIBRARY_PREFIX
#define LIBRARY_EXTENSION ".dll"
#elif defined(__APPLE__)
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".dylib"
#else
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".so"
#endif

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

namespace onnxruntime {

ProviderHost* g_host{};

struct Provider_TensorShapeProto_Dimension_Iterator_Impl : Provider_TensorShapeProto_Dimension_Iterator {
  Provider_TensorShapeProto_Dimension_Iterator_Impl(google::protobuf::internal::RepeatedPtrIterator<const onnx::TensorShapeProto_Dimension>&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_TensorShapeProto_Dimension_Iterator& p) const override { return v_ != static_cast<const Provider_TensorShapeProto_Dimension_Iterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Provider_TensorShapeProto_Dimension& operator*() override { return *v_; }

  google::protobuf::internal::RepeatedPtrIterator<const onnx::TensorShapeProto_Dimension> v_;
};

struct NodeAttributes_Iterator_Impl : NodeAttributes_Iterator {
  NodeAttributes_Iterator_Impl(NodeAttributes::const_iterator&& v) : v_{std::move(v)} {}

  bool operator!=(const NodeAttributes_Iterator& p) const override { return v_ != static_cast<const NodeAttributes_Iterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const std::string& first() const override { return v_->first; }
  const Provider_AttributeProto& second() override { return v_->second; }

  NodeAttributes::const_iterator v_;
};

struct Node__NodeIterator_Impl : Node__NodeIterator {
  Node__NodeIterator_Impl(Node::NodeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Node__NodeIterator& p) const override { return v_ != static_cast<const Node__NodeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Node& operator*() override { return *v_; }

  Node::NodeConstIterator v_;
};

struct Node__EdgeIterator_Impl : Node__EdgeIterator {
  Node__EdgeIterator_Impl(Node::EdgeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Node__EdgeIterator& p) const override { return v_ != static_cast<const Node__EdgeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Node& GetNode() const override { return v_->GetNode(); }
  int GetSrcArgIndex() const override { return v_->GetSrcArgIndex(); }
  int GetDstArgIndex() const override { return v_->GetDstArgIndex(); }

  Node::EdgeConstIterator v_;
};

struct OpKernel_Translator : OpKernel {
  OpKernel_Translator(const OpKernelInfo& info, Provider_OpKernel* p) : OpKernel{info}, p_{p} {
  }

  Status Compute(OpKernelContext* context) const override {
    return p_->Compute(context, *reinterpret_cast<const Provider_OpKernel_Base*>(static_cast<const OpKernel*>(this)));
  }

  std::unique_ptr<Provider_OpKernel> p_;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OpKernel_Translator);
};

struct ProviderHostImpl : ProviderHost {
  ProviderHostImpl() {
    DataTypeImpl_GetType_Tensor = &DataTypeImpl::GetType<Tensor>;
    DataTypeImpl_GetType_float = &DataTypeImpl::GetType<float>;
    DataTypeImpl_GetTensorType_float = &DataTypeImpl::GetTensorType<float>;
  }

  AllocatorPtr CreateAllocator(const AllocatorCreationInfo& info) override {
    return onnxruntime::CreateAllocator(info);
  }

  std::unique_ptr<IAllocator> CreateCPUAllocator(const OrtMemoryInfo& memory_info) override {
    return onnxruntime::make_unique<CPUAllocator>(memory_info);
  };

#ifdef USE_TENSORRT
  std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) override {
    return onnxruntime::make_unique<CUDAAllocator>(device_id, name);
  }

  std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) override {
    return onnxruntime::make_unique<CUDAPinnedAllocator>(device_id, name);
  }

  std::unique_ptr<IDataTransfer> CreateGPUDataTransfer() override { return onnxruntime::make_unique<GPUDataTransfer>(); }

  void cuda__Impl_Cast(const int64_t* input_data, int32_t* output_data, size_t count) override {
    return cuda::Impl_Cast(input_data, output_data, count);
  }

  void cuda__Impl_Cast(const int32_t* input_data, int64_t* output_data, size_t count) override {
    return cuda::Impl_Cast(input_data, output_data, count);
  }

  bool CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) override { return CudaCall<cudaError, false>(cudaError(retCode), exprString, libName, cudaError(successCode), msg); }
  bool CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) override { return CudaCall<cudaError, true>(cudaError(retCode), exprString, libName, cudaError(successCode), msg); }
#endif

  std::string GetEnvironmentVar(const std::string& var_name) override {
    return Env::Default().GetEnvironmentVar(var_name);
  }

  logging::Logger* LoggingManager_GetDefaultLogger() override {
    return const_cast<logging::Logger*>(&logging::LoggingManager::DefaultLogger());
  }

  const std::vector<MLDataType>& DataTypeImpl_AllFixedSizeTensorTypes() override {
    return DataTypeImpl::AllFixedSizeTensorTypes();
  }

  void* HeapAllocate(size_t size) override { return new uint8_t[size]; }
  void HeapFree(void* p) override { delete[] reinterpret_cast<uint8_t*>(p); }

  std::vector<std::string> GetStackTrace() override { return onnxruntime::GetStackTrace(); }

  AutoPadType StringToAutoPadType(const std::string& str) override { return onnxruntime::StringToAutoPadType(str); }

  void LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file, const char* function, uint32_t line) override {
    return ::onnxruntime::LogRuntimeError(session_id, status, file, function, line);
  }

  // IAllocator
  bool IAllocator__CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment, size_t* out) override { return IAllocator::CalcMemSizeForArrayWithAlignment(nmemb, size, alignment, out); }

  // IExecutionProvider
  AllocatorPtr IExecutionProvider__GetAllocator(const IExecutionProvider* p, int id, OrtMemType mem_type) override { return p->IExecutionProvider::GetAllocator(id, mem_type); }
  void IExecutionProvider__InsertAllocator(IExecutionProvider* p, AllocatorPtr allocator) override { return p->IExecutionProvider::InsertAllocator(allocator); }
  void IExecutionProvider__TryInsertAllocator(IExecutionProvider* p, AllocatorPtr allocator) override { return p->IExecutionProvider::TryInsertAllocator(allocator); }
  std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider__GetCapability(const IExecutionProvider* p, const onnxruntime::GraphViewer& graph_viewer,
                                                                                    const std::vector<const KernelRegistry*>& kernel_registries) override { return p->IExecutionProvider::GetCapability(graph_viewer, kernel_registries); }
  common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<onnxruntime::Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) override {
    return p->IExecutionProvider::Compile(fused_nodes, node_compute_funcs);
  }

  common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<onnxruntime::Node*>& fused_nodes, std::string& dll_path) override {
    return p->IExecutionProvider::Compile(fused_nodes, dll_path);
  }

  common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs, std::vector<NodeComputeInfo>& node_compute_funcs) override {
    return p->IExecutionProvider::Compile(fused_nodes_and_graphs, node_compute_funcs);
  }

  int IExecutionProvider__GenerateMetaDefId(const IExecutionProvider* p, const onnxruntime::GraphViewer& graph_viewer, uint64_t& model_hash) override {
    return p->IExecutionProvider::GenerateMetaDefId(graph_viewer, model_hash);
  }

  void IExecutionProvider__RegisterAllocator(IExecutionProvider* p, std::shared_ptr<AllocatorManager> allocator_manager) override {
    return p->IExecutionProvider::RegisterAllocator(allocator_manager);
  }

  // Status
  std::string Status__ToString(const Status* p) override { return p->ToString(); }

  // TensorShape
  int64_t TensorShape__SizeHelper(const TensorShape* p, size_t start, size_t end) override { return p->SizeHelper(start, end); }
  std::string TensorShape__ToString(const TensorShape* p) override { return p->ToString(); }

  // CPUIDInfo
  const CPUIDInfo& CPUIDInfo__GetCPUIDInfo() override { return CPUIDInfo::GetCPUIDInfo(); }
  bool CPUIDInfo__HasAVX2(const CPUIDInfo* p) override { return p->HasAVX2(); }
  bool CPUIDInfo__HasAVX512f(const CPUIDInfo* p) override { return p->HasAVX512f(); }

  // logging::Logger
  bool logging__Logger__OutputIsEnabled(const logging::Logger* p, logging::Severity severity, logging::DataType data_type) override { return p->OutputIsEnabled(severity, data_type); }

  // logging::LoggingManager
  const logging::Logger& logging__LoggingManager__DefaultLogger() override { return logging::LoggingManager::DefaultLogger(); }

  // logging::Capture
  std::unique_ptr<logging::Capture> logging__Capture__construct(const logging::Logger& logger, logging::Severity severity, const char* category, logging::DataType dataType, const CodeLocation& location) override {
    return onnxruntime::make_unique<logging::Capture>(logger, severity, category, dataType, location);
  }
  void logging__Capture__operator_delete(logging::Capture* p) noexcept override { delete p; }
  std::ostream& logging__Capture__Stream(logging::Capture* p) noexcept override { return p->Stream(); }

  // Utils::DataTypeUtils
  const std::string* Utils__DataTypeUtils__ToType(const Provider_TypeProto& type_proto) override { return ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(type_proto); }

  // Provider_int64s
  int Provider_int64s__size(const Provider_int64s* p) override { return p->size(); }
  const int64_t& Provider_int64s__Get(const Provider_int64s* p, int index) override { return p->Get(index); }

  // Provider_TypeProto_Tensor
  const Provider_TensorShapeProto& Provider_TypeProto_Tensor__shape(const Provider_TypeProto_Tensor* p) override { return p->shape(); }
  Provider_TensorShapeProto* Provider_TypeProto_Tensor__mutable_shape(Provider_TypeProto_Tensor* p) override { return p->mutable_shape(); }
  int32_t Provider_TypeProto_Tensor__elem_type(const Provider_TypeProto_Tensor* p) override { return p->elem_type(); }

  // Provider_TypeProto
  const Provider_TypeProto_Tensor& Provider_TypeProto__tensor_type(const Provider_TypeProto* p) override { return p->tensor_type(); }
  Provider_TypeProto_Tensor* Provider_TypeProto__mutable_tensor_type(Provider_TypeProto* p) override { return p->mutable_tensor_type(); }

  // Provider_AttributeProto
  std::unique_ptr<Provider_AttributeProto> Provider_AttributeProto__construct() override { return onnxruntime::make_unique<ONNX_NAMESPACE::AttributeProto>(); }
  void Provider_AttributeProto__operator_delete(Provider_AttributeProto* p) override { delete p; }
  void Provider_AttributeProto__operator_assign(Provider_AttributeProto* p, const Provider_AttributeProto& v) override { *p = v; }

  ONNX_NAMESPACE::AttributeProto_AttributeType Provider_AttributeProto__type(const Provider_AttributeProto* p) override { return p->type(); }
  int Provider_AttributeProto__ints_size(const Provider_AttributeProto* p) override { return p->ints_size(); }
  int Provider_AttributeProto__floats_size(const Provider_AttributeProto* p) override { return p->floats_size(); }
  int64_t Provider_AttributeProto__ints(const Provider_AttributeProto* p, int i) override { return p->ints(i); }
  float Provider_AttributeProto__floats(const Provider_AttributeProto* p, int i) override { return p->floats(i); }
  const Provider_int64s& Provider_AttributeProto__ints(const Provider_AttributeProto* p) override { return p->ints(); }
  int64_t Provider_AttributeProto__i(const Provider_AttributeProto* p) override { return p->i(); }
  float Provider_AttributeProto__f(const Provider_AttributeProto* p) override { return p->f(); }
  void Provider_AttributeProto__set_s(Provider_AttributeProto* p, const ::std::string& value) override { return p->set_s(value); }
  const ::std::string& Provider_AttributeProto__s(const Provider_AttributeProto* p) override { return p->s(); }
  void Provider_AttributeProto__set_name(Provider_AttributeProto* p, const ::std::string& value) override { return p->set_name(value); }
  void Provider_AttributeProto__set_type(Provider_AttributeProto* p, ONNX_NAMESPACE::AttributeProto_AttributeType value) override { return p->set_type(value); }
  Provider_TensorProto* Provider_AttributeProto__add_tensors(Provider_AttributeProto* p) override { return p->add_tensors(); }

  // Provider_GraphProto
  void Provider_GraphProto__operator_delete(Provider_GraphProto* p) override { delete p; }

  const Provider_ValueInfoProto& Provider_GraphProto__input(const Provider_GraphProto* p, int index) override { return p->input(index); }
  Provider_ValueInfoProto* Provider_GraphProto__mutable_input(Provider_GraphProto* p, int index) override { return p->mutable_input(index); }
  Provider_ValueInfoProtos* Provider_GraphProto__mutable_input(Provider_GraphProto* p) override { return p->mutable_input(); }
  int Provider_GraphProto__input_size(const Provider_GraphProto* p) override { return p->input_size(); }

  const Provider_ValueInfoProtos& Provider_GraphProto__output(const Provider_GraphProto* p) override { return p->output(); }
  const Provider_ValueInfoProto& Provider_GraphProto__output(const Provider_GraphProto* p, int index) override { return p->output(index); }
  Provider_ValueInfoProtos* Provider_GraphProto__mutable_output(Provider_GraphProto* p) override { return p->mutable_output(); }

  Provider_ValueInfoProtos* Provider_GraphProto__mutable_value_info(Provider_GraphProto* p) override { return p->mutable_value_info(); }
  Provider_TensorProtos* Provider_GraphProto__mutable_initializer(Provider_GraphProto* p) override { return p->mutable_initializer(); }
  Provider_NodeProto* Provider_GraphProto__add_node(Provider_GraphProto* p) override { return p->add_node(); }

  void Provider_GraphProto__operator_assign(Provider_GraphProto* p, const Provider_GraphProto& v) override { *p = v; }

  // Provider_ModelProto
  std::unique_ptr<Provider_ModelProto> Provider_ModelProto__construct() override { return onnxruntime::make_unique<ONNX_NAMESPACE::ModelProto>(); }
  void Provider_ModelProto__operator_delete(Provider_ModelProto* p) override { delete p; }

  bool Provider_ModelProto__SerializeToString(const Provider_ModelProto* p, std::string& string) override { return p->SerializeToString(&string); }
  bool Provider_ModelProto__SerializeToOstream(const Provider_ModelProto* p, std::ostream& output) override { return p->SerializeToOstream(&output); }
  bool Provider_ModelProto__ParseFromString(Provider_ModelProto* p, const std::string& data) override { return p->ParseFromString(data); }
  std::string Provider_ModelProto__SerializeAsString(const Provider_ModelProto* p) override { return p->SerializeAsString(); }

  const Provider_GraphProto& Provider_ModelProto__graph(const Provider_ModelProto* p) override { return p->graph(); }
  Provider_GraphProto* Provider_ModelProto__mutable_graph(Provider_ModelProto* p) override { return p->mutable_graph(); }

  void Provider_ModelProto__set_ir_version(Provider_ModelProto* p, int64_t value) override { p->set_ir_version(value); }

  // Provider_TensorProto
  void Provider_TensorProto__operator_delete(Provider_TensorProto* p) override { delete p; }
  void Provider_TensorProto__operator_assign(Provider_TensorProto* p, const Provider_TensorProto& v) override { *p = v; }
  bool Provider_TensorProto__has_data_location(const Provider_TensorProto* p) override { return p->has_data_location(); }
  int Provider_TensorProto__data_location(const Provider_TensorProto* p) override { return p->data_location(); }

  // Provider_TensorProtos
  Provider_TensorProto* Provider_TensorProtos__Add(Provider_TensorProtos* p) override { return p->Add(); }

  // Provider_TensorShapeProto_Dimension
  int Provider_TensorShapeProto_Dimension__value_case(const Provider_TensorShapeProto_Dimension* p) override { return p->value_case(); }
  const std::string& Provider_TensorShapeProto_Dimension__dim_param(const Provider_TensorShapeProto_Dimension* p) override { return p->dim_param(); }
  int64_t Provider_TensorShapeProto_Dimension__dim_value(const Provider_TensorShapeProto_Dimension* p) override { return p->dim_value(); }
  void Provider_TensorShapeProto_Dimension__clear_dim_value(Provider_TensorShapeProto_Dimension* p) override { return p->clear_dim_value(); }
  void Provider_TensorShapeProto_Dimension__set_dim_value(Provider_TensorShapeProto_Dimension* p, int64_t value) override { return p->set_dim_value(value); }

  // Provider_TensorShapeProto_Dimensions
  std::unique_ptr<Provider_TensorShapeProto_Dimension_Iterator> Provider_TensorShapeProto_Dimensions__begin(const Provider_TensorShapeProto_Dimensions* p) override {
    return onnxruntime::make_unique<Provider_TensorShapeProto_Dimension_Iterator_Impl>(p->begin());
  }

  std::unique_ptr<Provider_TensorShapeProto_Dimension_Iterator> Provider_TensorShapeProto_Dimensions__end(const Provider_TensorShapeProto_Dimensions* p) override {
    return onnxruntime::make_unique<Provider_TensorShapeProto_Dimension_Iterator_Impl>(p->end());
  }

  // Provider_TensorShapeProto
  int Provider_TensorShapeProto__dim_size(const Provider_TensorShapeProto* p) override { return p->dim_size(); }
  const Provider_TensorShapeProto_Dimensions& Provider_TensorShapeProto__dim(const Provider_TensorShapeProto* p) override { return p->dim(); }
  const Provider_TensorShapeProto_Dimension& Provider_TensorShapeProto__dim(const Provider_TensorShapeProto* p, int index) override { return p->dim(index); }
  Provider_TensorShapeProto_Dimension* Provider_TensorShapeProto__mutable_dim(Provider_TensorShapeProto* p, int index) override { return p->mutable_dim(index); }
  void Provider_TensorShapeProto__clear_dim(Provider_TensorShapeProto* p) override { return p->clear_dim(); }
  Provider_TensorShapeProto_Dimension* Provider_TensorShapeProto__add_dim(Provider_TensorShapeProto* p) override { return p->add_dim(); }

  // Provider_ValueInfoProto
  const Provider_TypeProto& Provider_ValueInfoProto__type(const Provider_ValueInfoProto* p) override { return p->type(); }
  Provider_TypeProto* Provider_ValueInfoProto__mutable_type(Provider_ValueInfoProto* p) override { return p->mutable_type(); }
  virtual void Provider_ValueInfoProto__operator_assign(Provider_ValueInfoProto* p, const Provider_ValueInfoProto& v) override { *p = v; }

  // Provider_ValueInfoProtos
  Provider_ValueInfoProto* Provider_ValueInfoProtos__Add(Provider_ValueInfoProtos* p) override { return p->Add(); }

  const Provider_ValueInfoProto& Provider_ValueInfoProtos__operator_array(const Provider_ValueInfoProtos* p, int index) override { return (*p)[index]; }

  // ComputeCapability
  std::unique_ptr<ComputeCapability> ComputeCapability__construct(std::unique_ptr<IndexedSubGraph> t_sub_graph) override { return onnxruntime::make_unique<ComputeCapability>(std::move(t_sub_graph)); }
  void ComputeCapability__operator_delete(ComputeCapability* p) override { delete p; }
  std::unique_ptr<IndexedSubGraph>& ComputeCapability__SubGraph(ComputeCapability* p) override { return p->sub_graph; }

  // DataTransferManager
  Status DataTransferManager__CopyTensor(const DataTransferManager* p, const Tensor& src, Tensor& dst, int exec_queue_id) override { return p->CopyTensor(src, dst, exec_queue_id); }

  // IDataTransfer
  void IDataTransfer__operator_delete(IDataTransfer* p) override { delete p; }

  // IndexedSubGraph_MetaDef
  std::unique_ptr<IndexedSubGraph_MetaDef> IndexedSubGraph_MetaDef__construct() override { return onnxruntime::make_unique<IndexedSubGraph::MetaDef>(); }
  void IndexedSubGraph_MetaDef__operator_delete(IndexedSubGraph_MetaDef* p) override { delete p; }

  std::string& IndexedSubGraph_MetaDef__name(IndexedSubGraph_MetaDef* p) override { return p->name; }
  std::string& IndexedSubGraph_MetaDef__domain(IndexedSubGraph_MetaDef* p) override { return p->domain; }
  int& IndexedSubGraph_MetaDef__since_version(IndexedSubGraph_MetaDef* p) override { return p->since_version; }
  ONNX_NAMESPACE::OperatorStatus& IndexedSubGraph_MetaDef__status(IndexedSubGraph_MetaDef* p) override { return p->status; }
  std::vector<std::string>& IndexedSubGraph_MetaDef__inputs(IndexedSubGraph_MetaDef* p) override { return p->inputs; }
  std::vector<std::string>& IndexedSubGraph_MetaDef__outputs(IndexedSubGraph_MetaDef* p) override { return p->outputs; }
  NodeAttributes& IndexedSubGraph_MetaDef__attributes(IndexedSubGraph_MetaDef* p) override { return p->attributes; }
  std::string& IndexedSubGraph_MetaDef__doc_string(IndexedSubGraph_MetaDef* p) override { return p->doc_string; }

  // IndexedSubGraph
  std::unique_ptr<IndexedSubGraph> IndexedSubGraph__construct() override { return onnxruntime::make_unique<IndexedSubGraph>(); }
  void IndexedSubGraph__operator_delete(IndexedSubGraph* p) override { delete p; }

  std::vector<onnxruntime::NodeIndex>& IndexedSubGraph__Nodes(IndexedSubGraph* p) override { return p->nodes; }

  void IndexedSubGraph__SetMetaDef(IndexedSubGraph* p, std::unique_ptr<IndexedSubGraph_MetaDef>&& meta_def_) override { return p->SetMetaDef(std::move(meta_def_)); }
  const IndexedSubGraph_MetaDef* IndexedSubGraph__GetMetaDef(const IndexedSubGraph* p) override { return p->GetMetaDef(); }

  // KernelDef
  void KernelDef__operator_delete(KernelDef* p) override { delete p; }

  // KernelDefBuilder
  std::unique_ptr<KernelDefBuilder> KernelDefBuilder__construct() override { return onnxruntime::make_unique<KernelDefBuilder>(); }
  void KernelDefBuilder__operator_delete(KernelDefBuilder* p) override { delete p; }

  void KernelDefBuilder__SetName(KernelDefBuilder* p, const char* op_name) override { p->SetName(op_name); }
  void KernelDefBuilder__SetDomain(KernelDefBuilder* p, const char* domain) override { p->SetDomain(domain); }
  void KernelDefBuilder__SinceVersion(KernelDefBuilder* p, int since_version) override { p->SinceVersion(since_version); }
  void KernelDefBuilder__Provider(KernelDefBuilder* p, const char* provider_type) override { p->Provider(provider_type); }
  void KernelDefBuilder__TypeConstraint(KernelDefBuilder* p, const char* arg_name, MLDataType supported_type) override { p->TypeConstraint(arg_name, supported_type); }
  void KernelDefBuilder__TypeConstraint(KernelDefBuilder* p, const char* arg_name, const std::vector<MLDataType>& supported_types) override { p->TypeConstraint(arg_name, supported_types); }
  void KernelDefBuilder__InputMemoryType(KernelDefBuilder* p, OrtMemType type, int input_index) override { p->InputMemoryType(type, input_index); }
  void KernelDefBuilder__OutputMemoryType(KernelDefBuilder* p, OrtMemType type, int input_index) override { p->OutputMemoryType(type, input_index); }
  void KernelDefBuilder__ExecQueueId(KernelDefBuilder* p, int queue_id) override { p->ExecQueueId(queue_id); }

  std::unique_ptr<KernelDef> KernelDefBuilder__Build(KernelDefBuilder* p) override { return p->Build(); }

  // KernelRegistry
  std::shared_ptr<KernelRegistry> KernelRegistry__construct() override { return std::make_shared<KernelRegistry>(); }
  void KernelRegistry__operator_delete(KernelRegistry* p) override { delete p; }
  Status KernelRegistry__Register(KernelRegistry* p, Provider_KernelCreateInfo&& create_info) override {
    KernelCreateInfo info_real(std::move(create_info.kernel_def),
                               [kernel_create_func = create_info.kernel_create_func](const OpKernelInfo& info) -> OpKernel* {
                                 return new OpKernel_Translator(info, kernel_create_func(info));
                               });
    return p->Register(std::move(info_real));
  }

  // Function
  const Graph& Function__Body(const Function* p) override { return p->Body(); }

  // Node
  const std::string& Node__Name(const Node* p) noexcept override { return p->Name(); }
  const std::string& Node__Description(const Node* p) noexcept override { return p->Description(); }
  const std::string& Node__Domain(const Node* p) noexcept override { return p->Domain(); }
  const std::string& Node__OpType(const Node* p) noexcept override { return p->OpType(); }

  const Function* Node__GetFunctionBody(const Node* p) noexcept override { return p->GetFunctionBody(); }

  ConstPointerContainer<std::vector<NodeArg*>> Node__ImplicitInputDefs(const Node* p) noexcept override { return p->ImplicitInputDefs(); }
  ConstPointerContainer<std::vector<NodeArg*>> Node__InputDefs(const Node* p) noexcept override { return p->InputDefs(); }
  ConstPointerContainer<std::vector<NodeArg*>> Node__OutputDefs(const Node* p) noexcept override { return p->OutputDefs(); }

  NodeIndex Node__Index(const Node* p) noexcept override { return p->Index(); }

  void Node__ToProto(const Node* p, Provider_NodeProto& proto, bool update_subgraphs = false) override { p->ToProto(proto, update_subgraphs); }

  const NodeAttributes& Node__GetAttributes(const Node* p) noexcept override { return p->GetAttributes(); }
  size_t Node__GetInputEdgesCount(const Node* p) noexcept override { return p->GetInputEdgesCount(); }
  size_t Node__GetOutputEdgesCount(const Node* p) noexcept override { return p->GetOutputEdgesCount(); }

  std::unique_ptr<Node__NodeIterator> Node__InputNodesBegin(const Node* p) noexcept override { return onnxruntime::make_unique<Node__NodeIterator_Impl>(p->InputNodesBegin()); }
  std::unique_ptr<Node__NodeIterator> Node__InputNodesEnd(const Node* p) noexcept override { return onnxruntime::make_unique<Node__NodeIterator_Impl>(p->InputNodesEnd()); }

  std::unique_ptr<Node__NodeIterator> Node__OutputNodesBegin(const Node* p) noexcept override { return onnxruntime::make_unique<Node__NodeIterator_Impl>(p->OutputNodesBegin()); }
  std::unique_ptr<Node__NodeIterator> Node__OutputNodesEnd(const Node* p) noexcept override { return onnxruntime::make_unique<Node__NodeIterator_Impl>(p->OutputNodesEnd()); }

  std::unique_ptr<Node__EdgeIterator> Node__OutputEdgesBegin(const Node* p) noexcept override { return onnxruntime::make_unique<Node__EdgeIterator_Impl>(p->OutputEdgesBegin()); }
  std::unique_ptr<Node__EdgeIterator> Node__OutputEdgesEnd(const Node* p) noexcept override { return onnxruntime::make_unique<Node__EdgeIterator_Impl>(p->OutputEdgesEnd()); }

  void Node__ForEachDef(const Node* p, std::function<void(const NodeArg&, bool is_input)> func, bool include_missing_optional_defs) override { p->ForEachDef(func, std::move(include_missing_optional_defs)); }

  // NodeArg
  const std::string& NodeArg__Name(const NodeArg* p) noexcept override { return p->Name(); }
  const Provider_TensorShapeProto* NodeArg__Shape(const NodeArg* p) override { return p->Shape(); }
  ONNX_NAMESPACE::DataType NodeArg__Type(const NodeArg* p) noexcept override { return p->Type(); }
  const Provider_NodeArgInfo& NodeArg__ToProto(const NodeArg* p) noexcept override { return p->ToProto(); }
  bool NodeArg__Exists(const NodeArg* p) const noexcept override { return p->Exists(); }
  const Provider_TypeProto* NodeArg__TypeAsProto(const NodeArg* p) noexcept override { return p->TypeAsProto(); }

  // NodeAttributes
  std::unique_ptr<NodeAttributes> NodeAttributes__construct() override { return onnxruntime::make_unique<NodeAttributes>(); }
  void NodeAttributes__operator_delete(NodeAttributes* p) noexcept override { delete p; }
  size_t NodeAttributes__size(const NodeAttributes* p) override { return p->size(); }
  void NodeAttributes__clear(NodeAttributes* p) noexcept override { return p->clear(); }
  size_t NodeAttributes__count(const NodeAttributes* p, const std::string& keyval) override { return p->count(keyval); }
  Provider_AttributeProto& NodeAttributes__operator_array(NodeAttributes* p, const std::string& string) override { return (*p)[string]; }
  const Provider_AttributeProto& NodeAttributes__at(const NodeAttributes* p, const std::string& string) override { return p->at(string); }
  void NodeAttributes__operator_assign(NodeAttributes* p, const NodeAttributes& v) override { *p = v; }

  std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__begin(const NodeAttributes* p) override {
    return onnxruntime::make_unique<NodeAttributes_Iterator_Impl>(p->begin());
  }
  std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__end(const NodeAttributes* p) override {
    return onnxruntime::make_unique<NodeAttributes_Iterator_Impl>(p->end());
  }
  std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__find(const NodeAttributes* p, const std::string& key) override {
    return onnxruntime::make_unique<NodeAttributes_Iterator_Impl>(p->find(key));
  }
  void NodeAttributes__insert(NodeAttributes* p, const NodeAttributes& v) override { return p->insert(v.begin(), v.end()); }

  // Model
  void Model__operator_delete(Model* p) override { delete p; }
  Graph& Model__MainGraph(Model* p) override { return p->MainGraph(); }
  std::unique_ptr<Provider_ModelProto> Model__ToProto(Model* p) override { return onnxruntime::make_unique<ONNX_NAMESPACE::ModelProto>(p->ToProto()); }

  // Graph
  std::unique_ptr<GraphViewer> Graph__CreateGraphViewer(const Graph* p) override { return onnxruntime::make_unique<GraphViewer>(*p); }
  std::unique_ptr<Provider_GraphProto> Graph__ToGraphProto(const Graph* p) override { return onnxruntime::make_unique<ONNX_NAMESPACE::GraphProto>(p->ToGraphProto()); }

  NodeArg& Graph__GetOrCreateNodeArg(Graph* p, const std::string& name, const Provider_TypeProto* p_arg_type) override { return p->GetOrCreateNodeArg(name, p_arg_type); }

  Status Graph__Resolve(Graph* p) override { return p->Resolve(); }
  void Graph__AddInitializedTensor(Graph* p, const Provider_TensorProto& tensor) override { p->AddInitializedTensor(tensor); }
  Node& Graph__AddNode(Graph* p, const std::string& name, const std::string& op_type, const std::string& description, const std::vector<NodeArg*>& input_args, const std::vector<NodeArg*>& output_args, const NodeAttributes* attributes, const std::string& domain) override {
    return p->AddNode(name, op_type, description, input_args, output_args, attributes, domain);
  }

  const std::vector<const NodeArg*>& Graph__GetOutputs(const Graph* p) noexcept override { return p->GetOutputs(); }
  void Graph__SetOutputs(Graph* p, const std::vector<const NodeArg*>& outputs) override { p->SetOutputs(outputs); }

  const std::vector<const NodeArg*>& Graph__GetInputs(const Graph* p) noexcept override { return p->GetInputs(); }
  bool Graph__GetInitializedTensor(const Graph* p, const std::string& tensor_name, const Provider_TensorProto*& value) override { return p->GetInitializedTensor(tensor_name, value); }

  // GraphViewer
  void GraphViewer__operator_delete(GraphViewer* p) override { delete p; }
  std::unique_ptr<Model> GraphViewer__CreateModel(const GraphViewer* graph_viewer, const logging::Logger& logger) override {
    return onnxruntime::make_unique<Model>(graph_viewer->Name(), true, ModelMetaData(), PathString(),
                                           IOnnxRuntimeOpSchemaRegistryList(), graph_viewer->DomainToVersionMap(),
                                           std::vector<ONNX_NAMESPACE::FunctionProto>(), logger);
  }

  const std::string& GraphViewer__Name(const GraphViewer* p) noexcept override { return p->Name(); }
  const Path& GraphViewer__ModelPath(const GraphViewer* p) noexcept override { return p->ModelPath(); }

  const Node* GraphViewer__GetNode(const GraphViewer* p, NodeIndex node_index) override { return p->GetNode(node_index); }
  const NodeArg* GraphViewer__GetNodeArg(const GraphViewer* p, const std::string& name) override { return p->GetNodeArg(name); }

  bool GraphViewer__IsSubgraph(const GraphViewer* p) override { return p->IsSubgraph(); }
  bool GraphViewer__IsConstantInitializer(const GraphViewer* p, const std::string& name, bool check_outer_scope) override { return p->IsConstantInitializer(name, check_outer_scope); }
  int GraphViewer__NumberOfNodes(const GraphViewer* p) noexcept override { return p->NumberOfNodes(); }
  int GraphViewer__MaxNodeIndex(const GraphViewer* p) noexcept override { return p->MaxNodeIndex(); }

  const std::vector<const NodeArg*>& GraphViewer__GetInputs(const GraphViewer* p) noexcept override { return p->GetInputs(); }
  const std::vector<const NodeArg*>& GraphViewer__GetOutputs(const GraphViewer* p) noexcept override { return p->GetOutputs(); }
  const std::vector<const NodeArg*>& GraphViewer__GetValueInfo(const GraphViewer* p) noexcept override { return p->GetValueInfo(); }

  const Provider_InitializedTensorSet& GraphViewer__GetAllInitializedTensors(const GraphViewer* p) override { return p->GetAllInitializedTensors(); }
  bool GraphViewer__GetInitializedTensor(const GraphViewer* p, const std::string& tensor_name, const Provider_TensorProto*& value) override { return p->GetInitializedTensor(tensor_name, value); }

  const std::unordered_map<std::string, int>& GraphViewer__DomainToVersionMap(const GraphViewer* p) override { return p->DomainToVersionMap(); }

  const std::vector<NodeIndex>& GraphViewer__GetNodesInTopologicalOrder(const GraphViewer* p) override { return p->GetNodesInTopologicalOrder(); }
  const std::vector<const NodeArg*>& GraphViewer__GetInputsIncludingInitializers(const GraphViewer* p) noexcept override { return p->GetInputsIncludingInitializers(); }

  //Path
  //Gets a string representation of the path.
  PathString Path__ToPathString(const Path* p) noexcept override { return p->ToPathString(); }

  // Provider_OpKernel_Base
  const OpKernelInfo& Provider_OpKernel_Base__GetInfo(const Provider_OpKernel_Base* p) override { return reinterpret_cast<const OpKernel*>(p)->Info(); }

  // OpKernelContext
  const Tensor* OpKernelContext__Input_Tensor(const OpKernelContext* p, int index) override { return p->Input<Tensor>(index); }
  Tensor* OpKernelContext__Output(OpKernelContext* p, int index, const TensorShape& shape) override { return p->Output(index, shape); }

  // OpKernelInfo
  Status OpKernelInfo__GetAttr_int64(const OpKernelInfo* p, const std::string& name, int64_t* value) override { return p->GetAttr(name, value); }
  Status OpKernelInfo__GetAttr_float(const OpKernelInfo* p, const std::string& name, float* value) override { return p->GetAttr(name, value); }

  const DataTransferManager& OpKernelInfo__GetDataTransferManager(const OpKernelInfo* p) noexcept override { return p->GetDataTransferManager(); }
  int OpKernelInfo__GetKernelDef_ExecQueueId(const OpKernelInfo* p) noexcept override { return p->GetKernelDef().ExecQueueId(); }

  // Tensor
  float* Tensor__MutableData_float(Tensor* p) override { return p->MutableData<float>(); }
  const float* Tensor__Data_float(const Tensor* p) override { return p->Data<float>(); }

  void* Tensor__MutableDataRaw(Tensor* p) noexcept override { return p->MutableDataRaw(); }
  const void* Tensor__DataRaw(const Tensor* p) const noexcept override { return p->DataRaw(); }

  const TensorShape& Tensor__Shape(const Tensor* p) override { return p->Shape(); }
  size_t Tensor__SizeInBytes(const Tensor* p) override { return p->SizeInBytes(); }
  const OrtMemoryInfo& Tensor__Location(const Tensor* p) override { return p->Location(); }  

  // AllocatorManager
  void AllocatorManager__InsertAllocator(AllocatorManager* p, AllocatorPtr allocator) override { p->InsertAllocator(allocator); }
  AllocatorPtr AllocatorManager__GetAllocator(AllocatorManager* p, int id, OrtMemType mem_type) override { return p->GetAllocator(id, mem_type); };

} provider_host_;

struct ProviderSharedLibrary {
  bool Ensure() {
    if (handle_)
      return true;

    std::string full_path = Env::Default().GetRuntimePath() + std::string(LIBRARY_PREFIX "onnxruntime_providers_shared" LIBRARY_EXTENSION);
    auto error = Env::Default().LoadDynamicLibrary(full_path, &handle_);
    if (!error.IsOK()) {
      LOGS_DEFAULT(ERROR) << error.ErrorMessage();
      return false;
    }

    void (*PProvider_SetHost)(void*);
    Env::Default().GetSymbolFromLibrary(handle_, "Provider_SetHost", (void**)&PProvider_SetHost);

    PProvider_SetHost(&provider_host_);
    return true;
  }

  void Unload() {
    if (handle_) {
      Env::Default().UnloadDynamicLibrary(handle_);
      handle_ = nullptr;
    }
  }

  ProviderSharedLibrary() = default;
  ~ProviderSharedLibrary() { /*assert(!handle_);*/
  }                          // We should already be unloaded at this point (disabled until Python shuts down deterministically)

 private:
  void* handle_{};

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ProviderSharedLibrary);
};

static ProviderSharedLibrary s_library_shared;

struct ProviderLibrary {
  ProviderLibrary(const char* filename) : filename_{filename} {}
  ~ProviderLibrary() { /*assert(!handle_);*/
  }                    // We should already be unloaded at this point (disabled until Python shuts down deterministically)

  Provider* Get() {
    if (provider_)
      return provider_;

    if (!s_library_shared.Ensure())
      return nullptr;

    std::string full_path = Env::Default().GetRuntimePath() + std::string(filename_);
    auto error = Env::Default().LoadDynamicLibrary(full_path, &handle_);
    if (!error.IsOK()) {
      LOGS_DEFAULT(ERROR) << error.ErrorMessage();
      return nullptr;
    }

    Provider* (*PGetProvider)();
    Env::Default().GetSymbolFromLibrary(handle_, "GetProvider", (void**)&PGetProvider);

    provider_ = PGetProvider();
    return provider_;
  }

  void Unload() {
    if (handle_) {
      if (provider_)
        provider_->Shutdown();

      Env::Default().UnloadDynamicLibrary(handle_);
      handle_ = nullptr;
      provider_ = nullptr;
    }
  }

 private:
  const char* filename_;
  Provider* provider_{};
  void* handle_{};

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ProviderLibrary);
};

static ProviderLibrary s_library_dnnl(LIBRARY_PREFIX "onnxruntime_providers_dnnl" LIBRARY_EXTENSION);
static ProviderLibrary s_library_openvino(LIBRARY_PREFIX "onnxruntime_providers_openvino" LIBRARY_EXTENSION);
static ProviderLibrary s_library_tensorrt(LIBRARY_PREFIX "onnxruntime_providers_tensorrt" LIBRARY_EXTENSION);

void UnloadSharedProviders() {
  s_library_dnnl.Unload();
  s_library_openvino.Unload();
  s_library_tensorrt.Unload();
  s_library_shared.Unload();
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(int use_arena) {
  if (auto provider = s_library_dnnl.Get())
    return provider->CreateExecutionProviderFactory(use_arena);

  return nullptr;
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id) {
  if (auto provider = s_library_tensorrt.Get())
    return provider->CreateExecutionProviderFactory(device_id);

  return nullptr;
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(const OrtOpenVINOProviderOptions* provider_options) {
  if (auto provider = s_library_openvino.Get())
    return provider->CreateExecutionProviderFactory(provider_options);

  return nullptr;
}

const ProviderInfo_OpenVINO* GetProviderInfo_OpenVINO() {
  if (auto provider = s_library_openvino.Get())
    return reinterpret_cast<const ProviderInfo_OpenVINO*>(provider->GetInfo());
  return nullptr;
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Dnnl, _In_ OrtSessionOptions* options, int use_arena) {
  auto factory = onnxruntime::CreateExecutionProviderFactory_Dnnl(use_arena);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Dnnl: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tensorrt, _In_ OrtSessionOptions* options, int device_id) {
  auto factory = onnxruntime::CreateExecutionProviderFactory_Tensorrt(device_id);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Tensorrt: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_OpenVINO, _In_ OrtSessionOptions* options, _In_ const OrtOpenVINOProviderOptions* provider_options) {
  auto factory = onnxruntime::CreateExecutionProviderFactory_OpenVINO(provider_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "SessionOptionsAppendExecutionProvider_OpenVINO: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_OpenVINO, _In_ OrtSessionOptions* options, _In_ const char* device_type) {
  OrtOpenVINOProviderOptions provider_options;
  provider_options.device_type = device_type;
  auto factory = onnxruntime::CreateExecutionProviderFactory_OpenVINO(&provider_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_OpenVINO: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
}
