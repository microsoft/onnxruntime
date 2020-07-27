// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the Onnxruntime side of the bridge to allow providers to be built as a DLL
// It implements onnxruntime::ProviderHost

#include "core/framework/data_types.h"
#include "core/framework/allocatormgr.h"
#include "core/providers/dnnl/dnnl_provider_factory.h"
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
//#include "core/providers/cuda/cuda_allocator.h"
//#include "core/providers/cuda/cuda_common.h"
//#include "core/providers/cuda/cuda_fence.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"
#include "core/platform/env.h"
#include "core/graph/model.h"
#include "core/framework/execution_provider.h"
#include "core/framework/compute_capability.h"
#define PROVIDER_BRIDGE_ORT
#include "core/providers/shared_library/provider_interfaces.h"
#include "onnx/common/stl_backports.h"
#include "core/common/logging/logging.h"
#include "core/common/cpuid_info.h"

namespace onnxruntime {

struct Provider_OrtDevice_Impl : Provider_OrtDevice {
  OrtDevice v_;
};

struct Provider_OrtMemoryInfo_Impl : Provider_OrtMemoryInfo {
  Provider_OrtMemoryInfo_Impl(const char* name_, OrtAllocatorType type_, OrtDevice device_, int id_, OrtMemType mem_type_) : info_{onnxruntime::make_unique<OrtMemoryInfo>(name_, type_, device_, id_, mem_type_)} {}

  std::unique_ptr<OrtMemoryInfo> info_;
};

struct Provider_IAllocator_Impl : Provider_IAllocator {
  Provider_IAllocator_Impl(AllocatorPtr p) : p_{p} {}

  void* Alloc(size_t size) override { return p_->Alloc(size); }
  void Free(void* p) override { return p_->Free(p); }

  AllocatorPtr p_;
};

struct Provider_IDeviceAllocator_Impl : Provider_IDeviceAllocator {
  Provider_IDeviceAllocator_Impl(std::unique_ptr<IDeviceAllocator> p) : p_{std::move(p)} {}

  void* Alloc(size_t size) override { return p_->Alloc(size); }
  void Free(void* p) override { return p_->Free(p); }

  std::unique_ptr<IDeviceAllocator> p_;
};

#if 0
struct Provider_NodeProto_Impl : ONNX_NAMESPACE::Provider_NodeProto {
  Provider_NodeProto_Impl(ONNX_NAMESPACE::NodeProto& v) : v_{v} {}

  ONNX_NAMESPACE::NodeProto& v_;
};

struct Provider_TypeProto_Tensor_Impl : ONNX_NAMESPACE::Provider_TypeProto_Tensor {
  Provider_TypeProto_Tensor_Impl(const ONNX_NAMESPACE::TypeProto_Tensor& v) : v_{v} {}

  int32_t elem_type() const override { return v_.elem_type(); }

  const ONNX_NAMESPACE::TypeProto_Tensor& v_;
};

struct Provider_TypeProto_Impl : ONNX_NAMESPACE::Provider_TypeProto {
  Provider_TypeProto_Impl(const ONNX_NAMESPACE::TypeProto& v) : v_{v} {}

  const ONNX_NAMESPACE::Provider_TypeProto_Tensor& tensor_type() const override { return tensor_type_; }

  const ONNX_NAMESPACE::TypeProto& v_;
  Provider_TypeProto_Tensor_Impl tensor_type_{v_.tensor_type()};
};

struct Provider_TensorProto_Impl : ONNX_NAMESPACE::Provider_TensorProto {
  Provider_TensorProto_Impl(ONNX_NAMESPACE::TensorProto* p) : p_{p} {}

  void CopyFrom(const Provider_TensorProto& v) override {
    *p_ = *static_cast<const Provider_TensorProto_Impl*>(&v)->p_;
  }

  ONNX_NAMESPACE::TensorProto* p_;
};
#endif

struct Provider_TensorShapeProto_Dimension_Iterator_Impl : Provider_TensorShapeProto_Dimension_Iterator {
  Provider_TensorShapeProto_Dimension_Iterator_Impl(google::protobuf::internal::RepeatedPtrIterator<const onnx::TensorShapeProto_Dimension>&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_TensorShapeProto_Dimension_Iterator& p) const override { return v_ != static_cast<const Provider_TensorShapeProto_Dimension_Iterator_Impl*>(&p)->v_; }

  void operator++() { v_.operator++(); }
  const Provider_TensorShapeProto_Dimension& operator*() { return *reinterpret_cast<const Provider_TensorShapeProto_Dimension*>(&v_.operator*()); }

  google::protobuf::internal::RepeatedPtrIterator<const onnx::TensorShapeProto_Dimension> v_;
};

#if 0
struct Provider_TensorShapeProto_Dimension_Impl : ONNX_NAMESPACE::Provider_TensorShapeProto_Dimension {
  Provider_TensorShapeProto_Dimension_Impl(const ONNX_NAMESPACE::TensorShapeProto_Dimension& v) : v_{v} {}
    
  const std::string& dim_param() const override { return v_.dim_param();  }

  const ONNX_NAMESPACE::TensorShapeProto_Dimension& v_;
};

struct Provider_TensorShapeProto_Dimensions_Impl : ONNX_NAMESPACE::Provider_TensorShapeProto_Dimensions {
    Provider_TensorShapeProto_Dimensions_Impl(const google::protobuf::RepeatedPtrField<::ONNX_NAMESPACE::TensorShapeProto_Dimension>& v) : v_{ v } {
    for (auto iterator = v_.begin(); iterator < v_.end(); iterator++)
      dimension_impls_.emplace_back(*iterator);
    for (auto& impl : dimension_impls_)
      dimensions_.push_back(&impl);
  }

  const ONNX_NAMESPACE::Provider_TensorShapeProto_Dimension* begin() const override { return *dimensions_.begin(); } 
  const ONNX_NAMESPACE::Provider_TensorShapeProto_Dimension* end() const override { return *dimensions_.end(); }

  const google::protobuf::RepeatedPtrField<::ONNX_NAMESPACE::TensorShapeProto_Dimension>& v_;

  std::vector<Provider_TensorShapeProto_Dimension_Impl> dimension_impls_;
  std::vector<ONNX_NAMESPACE::Provider_TensorShapeProto_Dimension*> dimensions_;
};

struct Provider_TensorShapeProto_Impl : ONNX_NAMESPACE::Provider_TensorShapeProto {
  Provider_TensorShapeProto_Impl(const ONNX_NAMESPACE::TensorShapeProto& v) : v_{v} {}

  int dim_size() const override { return v_.dim_size(); }
  const Provider_TensorShapeProto_Dimensions& dim() const override { return *reinterpret_cast<const Provider_TensorShapeProto_Dimensions*>(&v_.dim()); }

  const ONNX_NAMESPACE::TensorShapeProto& v_;
};

struct Provider_ValueInfoProto_Impl : ONNX_NAMESPACE::Provider_ValueInfoProto {
  Provider_ValueInfoProto_Impl(const ONNX_NAMESPACE::ValueInfoProto& v) : v_{v} {}

  const ONNX_NAMESPACE::Provider_TypeProto& type() const override { return type_; }

  const ONNX_NAMESPACE::ValueInfoProto& v_;
  Provider_TypeProto_Impl type_{v_.type()};
};

struct Provider_ValueInfoProtos_Impl : ONNX_NAMESPACE::Provider_ValueInfoProtos {
  Provider_ValueInfoProto* Add() override {}

  virtual const Provider_ValueInfoProto& operator[](int index) const = 0;
};
#endif

struct Provider_AttributeProto_Impl : ONNX_NAMESPACE::Provider_AttributeProto {
  Provider_AttributeProto_Impl() = default;
  Provider_AttributeProto_Impl(const ONNX_NAMESPACE::AttributeProto& copy) : v_{copy} {}

  std::unique_ptr<Provider_AttributeProto> Clone() const override {
    return onnxruntime::make_unique<Provider_AttributeProto_Impl>(v_);
  }

  ::onnx::AttributeProto_AttributeType type() const override { return v_.type(); }

  int ints_size() const override {
    return v_.ints_size();
  }

  int64_t ints(int i) const override { return v_.ints(i); }
  int64_t i() const override { return v_.i(); }
  float f() const override { return v_.f(); }
  void set_s(const ::std::string& value) override { v_.set_s(value); }
  const ::std::string& s() const override { return v_.s(); }
  void set_name(const ::std::string& value) override { v_.set_name(value); }
  void set_type(::onnx::AttributeProto_AttributeType value) override { v_.set_type(value); }
#if 0
  ::onnx::Provider_TensorProto* add_tensors() override {
    // Kind of a hack, but the pointer is only valid until the next add_tensors call
    tensors_ = onnxruntime::make_unique<Provider_TensorProto_Impl>(v_.add_tensors());
    return tensors_.get();
  }
#endif

  Provider_TensorProto* add_tensors() override { return reinterpret_cast<Provider_TensorProto*>(v_.add_tensors()); }

  ONNX_NAMESPACE::AttributeProto v_;
  //  std::unique_ptr<Provider_TensorProto_Impl> tensors_;
};

#if 0
struct Provider_KernelDef_Impl : Provider_KernelDef {
  Provider_KernelDef_Impl(std::unique_ptr<KernelDef> p) : p_(std::move(p)) {}
  std::unique_ptr<KernelDef> p_;
};

struct Provider_KernelDefBuilder_Impl : Provider_KernelDefBuilder {
  Provider_KernelDefBuilder& SetName(const char* op_name) override {
    v_.SetName(op_name);
    return *this;
  }
  Provider_KernelDefBuilder& SetDomain(const char* domain) override {
    v_.SetDomain(domain);
    return *this;
  }

  Provider_KernelDefBuilder& SinceVersion(int since_version) override {
    v_.SinceVersion(since_version);
    return *this;
  }
  Provider_KernelDefBuilder& Provider(const char* provider_type) override {
    v_.Provider(provider_type);
    return *this;
  }

  Provider_KernelDefBuilder& TypeConstraint(const char* arg_name, const std::vector<MLDataType>& supported_types) override {
    v_.TypeConstraint(arg_name, supported_types);
    return *this;
  }

  Provider_KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType supported_type) override {
    v_.TypeConstraint(arg_name, supported_type);
    return *this;
  }

  Provider_KernelDefBuilder& InputMemoryType(OrtMemType type, int input_index) override {
    v_.InputMemoryType(type, input_index);
    return *this;
  }

  Provider_KernelDefBuilder& OutputMemoryType(OrtMemType type, int input_index) override {
    v_.OutputMemoryType(type, input_index);
    return *this;
  }

  Provider_KernelDefBuilder& ExecQueueId(int queue_id) override {
    v_.ExecQueueId(queue_id);
    return *this;
  }

  std::unique_ptr<Provider_KernelDef> Build() override {
    return onnxruntime::make_unique<Provider_KernelDef_Impl>(v_.Build());
  }

  KernelDefBuilder v_;
};
#endif

#if 0
struct Provider_NodeArg_Impl : Provider_NodeArg {
  Provider_NodeArg_Impl(const NodeArg* p) : p_{p} {
  }

  const std::string& Name() const noexcept override { return p_->Name(); }
  const ONNX_NAMESPACE::Provider_TensorShapeProto* Shape() const override { return p_->Shape() ? &tensor_shape_proto_ : nullptr; }
  ONNX_NAMESPACE::DataType Type() const noexcept override { return p_->Type(); }

  const Provider_NodeArgInfo& ToProto() const noexcept override { return proto_; }
  bool Exists() const noexcept override { return p_->Exists(); }
  const ONNX_NAMESPACE::Provider_TypeProto* TypeAsProto() const noexcept override { return &typeproto_; }

  const NodeArg* p_;
  Provider_TensorShapeProto_Impl tensor_shape_proto_{*p_->Shape()};
  Provider_ValueInfoProto_Impl proto_{p_->ToProto()};
  Provider_TypeProto_Impl typeproto_{*p_->TypeAsProto()};
};
#endif

#if 0
#pragma warning(disable : 4100)

struct Provider_Graph_Impl : Provider_Graph {
  Provider_Graph_Impl(Graph& v) : v_{v} {}
  Provider_Graph_Impl(const Graph& v) : v_{const_cast<Graph&>(v)} {}

  std::unique_ptr<Provider_GraphViewer> CreateGraphViewer() const override {
    __debugbreak();
    __assume(false);
  }
  std::unique_ptr<ONNX_NAMESPACE::Provider_GraphProto> CreateGraphProto() const {
    __debugbreak();
    __assume(false);
  }

  Provider_NodeArg& GetOrCreateNodeArg(const std::string& name, const ONNX_NAMESPACE::Provider_TypeProto* p_arg_type) override {
    __debugbreak();
    __assume(false);
  }

  Status Resolve() override { return v_.Resolve(); }
  void AddInitializedTensor(const ONNX_NAMESPACE::Provider_TensorProto& tensor) override {
    __debugbreak();
    __assume(false);
  }
  Provider_Node& AddNode(const std::string& name, const std::string& op_type, const std::string& description, const std::vector<Provider_NodeArg*>& input_args, const std::vector<Provider_NodeArg*>& output_args, const Provider_NodeAttributes* attributes, const std::string& domain) override {
    __debugbreak();
    __assume(false);
  }

  const std::vector<const Provider_NodeArg*>& GetOutputs() const noexcept override {
    __debugbreak();
    __assume(false);
  }
  void SetOutputs(const std::vector<const Provider_NodeArg*>& outputs) override {
    __debugbreak();
    __assume(false);
  }

  const std::vector<const Provider_NodeArg*>& GetInputs() const noexcept {
    __debugbreak();
    __assume(false);
  }

  Graph& v_;
};

struct Provider_Function_Impl : Provider_Function {
  Provider_Function_Impl(const Function& v) : v_{v} {}

  const Provider_Graph& Body() const override { return graph_; }

  const Function& v_;
  Provider_Graph_Impl graph_{v_.Body()};
};
#endif

#if 0
struct Provider_Node_Impl : Provider_Node {
  Provider_Node_Impl(const Node* p) : p_{p} {}
  ~Provider_Node_Impl() override {
#if 0
    for (auto p : input_defs_)
      delete p;
    for (auto p : output_defs_)
      delete p;
#endif
  }

  const std::string& Name() const noexcept override { return p_->Name(); }
  const std::string& Description() const noexcept override { return p_->Description(); }
  const std::string& Domain() const noexcept override { return p_->Domain(); }
  const std::string& OpType() const noexcept override { return p_->OpType(); }
  //  const ONNX_NAMESPACE::OpSchema* Op() const noexcept

#if 0
  const Provider_Function_Impl* GetFunctionBody() const noexcept override {
    if (function_)
      return function_.get();

    if (auto p = p_->GetFunctionBody())
      function_ = onnxruntime::make_unique<Provider_Function_Impl>(*p);

    return function_.get();
  }

  mutable std::unique_ptr<Provider_Function_Impl> function_;
#endif

  const Provider_Function* GetFunctionBody() const noexcept override { return reinterpret_cast<const Provider_Function*>(p_->GetFunctionBody()); }

  ConstPointerContainer<std::vector<Provider_NodeArg*>> InputDefs() const noexcept override {
    auto result = p_->InputDefs();
    return *reinterpret_cast<ConstPointerContainer<std::vector<Provider_NodeArg*>>*>(&result);
  }

  ConstPointerContainer<std::vector<Provider_NodeArg*>> OutputDefs() const noexcept override {
    auto result = p_->OutputDefs();
    return *reinterpret_cast<ConstPointerContainer<std::vector<Provider_NodeArg*>>*>(&result);
  }

#if 0
  ConstPointerContainer<std::vector<Provider_NodeArg*>> InputDefs() const noexcept override {
    if (input_defs_.empty()) {
      for (auto p : p_->InputDefs())
        input_defs_.push_back(new Provider_NodeArg_Impl(p));
    }

    return ConstPointerContainer<std::vector<Provider_NodeArg*>>(input_defs_);
  }

  ConstPointerContainer<std::vector<Provider_NodeArg*>> OutputDefs() const noexcept override {
    if (output_defs_.empty()) {
      for (auto p : p_->OutputDefs())
        output_defs_.push_back(new Provider_NodeArg_Impl(p));
    }

    return ConstPointerContainer<std::vector<Provider_NodeArg*>>(output_defs_);
  }
#endif

  NodeIndex Index() const noexcept override { return p_->Index(); }

#if 0
  void ToProto(ONNX_NAMESPACE::Provider_NodeProto& proto, bool update_subgraphs = false) const override {
    p_->ToProto(static_cast<Provider_NodeProto_Impl*>(&proto)->v_, update_subgraphs);
  }

    const Provider_NodeAttributes& GetAttributes() const noexcept override {
    if (!attributes_) {
      attributes_ = onnxruntime::make_unique<Provider_NodeAttributes>();
      for (auto& v : p_->GetAttributes())
        (*attributes_)[v.first] = onnxruntime::make_unique<Provider_AttributeProto_Impl>(v.second);
    }
    return *attributes_;
  }

#endif

  void ToProto(ONNX_NAMESPACE::Provider_NodeProto& proto, bool update_subgraphs = false) const override {
    p_->ToProto(*reinterpret_cast<ONNX_NAMESPACE::NodeProto*>(&proto), update_subgraphs);
  }

  const Provider_NodeAttributes& GetAttributes() const noexcept override {
    return *reinterpret_cast<const Provider_NodeAttributes*>(&p_->GetAttributes());
  }

  size_t GetInputEdgesCount() const noexcept override {
    return p_->GetInputEdgesCount();
  }
  size_t GetOutputEdgesCount() const noexcept override { return p_->GetOutputEdgesCount(); }

  std::unique_ptr<Provider_NodeIterator> InputNodesBegin_internal() const noexcept override;
  std::unique_ptr<Provider_NodeIterator> InputNodesEnd_internal() const noexcept override;

  std::unique_ptr<Provider_EdgeIterator> OutputEdgesBegin_internal() const noexcept override;
  std::unique_ptr<Provider_EdgeIterator> OutputEdgesEnd_internal() const noexcept override;

  const Node* p_;
#if 0
  mutable std::vector<Provider_NodeArg*> input_defs_;
  mutable std::vector<Provider_NodeArg*> output_defs_;
  mutable std::unique_ptr<Provider_NodeAttributes> attributes_;
#endif
};

struct Provider_NodeIterator_Impl : Provider_Node::Provider_NodeIterator {
  Provider_NodeIterator_Impl(Node::NodeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_NodeIterator& p) const override { return v_ != static_cast<const Provider_NodeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Provider_Node& operator*() override {
    node_.p_ = &*v_;
    return node_;
  }

  Node::NodeConstIterator v_;
  Provider_Node_Impl node_{nullptr};
};

std::unique_ptr<Provider_Node::Provider_NodeIterator> Provider_Node_Impl::InputNodesBegin_internal() const noexcept {
  return onnxruntime::make_unique<Provider_NodeIterator_Impl>(p_->InputNodesBegin());
}

std::unique_ptr<Provider_Node::Provider_NodeIterator> Provider_Node_Impl::InputNodesEnd_internal() const noexcept {
  return onnxruntime::make_unique<Provider_NodeIterator_Impl>(p_->InputNodesEnd());
}
#endif

struct Provider_Node__NodeIterator_Impl : Provider_Node__NodeIterator {
  Provider_Node__NodeIterator_Impl(Node::NodeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_Node__NodeIterator& p) const override { return v_ != static_cast<const Provider_Node__NodeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Provider_Node& operator*() override {
    return *reinterpret_cast<const Provider_Node*>(&*v_);
  }

  Node::NodeConstIterator v_;
};

struct Provider_Node__EdgeIterator_Impl : Provider_Node__EdgeIterator {
  Provider_Node__EdgeIterator_Impl(Node::EdgeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_Node__EdgeIterator& p) const override { return v_ != static_cast<const Provider_Node__EdgeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Provider_Node& GetNode() const override { return *reinterpret_cast<const Provider_Node*>(&*v_); }

  int GetSrcArgIndex() const override {
    return v_->GetSrcArgIndex();
  }

  int GetDstArgIndex() const override {
    return v_->GetDstArgIndex();
  }

  Node::EdgeConstIterator v_;
};

#if 0
struct Provider_EdgeIterator_Impl : Provider_Node::Provider_EdgeIterator {
  Provider_EdgeIterator_Impl(Node::EdgeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_EdgeIterator& p) const override { return v_ != static_cast<const Provider_EdgeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Provider_Node& GetNode() const override {
    node_ = onnxruntime::make_unique<Provider_Node_Impl>(&v_->GetNode());
    return *node_;
  }

  int GetSrcArgIndex() const override {
    return v_->GetSrcArgIndex();
  }

  int GetDstArgIndex() const override {
    return v_->GetDstArgIndex();
  }

  Node::EdgeConstIterator v_;
  mutable std::unique_ptr<Provider_Node_Impl> node_;
};

std::unique_ptr<Provider_Node::Provider_EdgeIterator> Provider_Node_Impl::OutputEdgesBegin_internal() const noexcept {
  return onnxruntime::make_unique<Provider_EdgeIterator_Impl>(p_->OutputEdgesBegin());
}

std::unique_ptr<Provider_Node::Provider_EdgeIterator> Provider_Node_Impl::OutputEdgesEnd_internal() const noexcept {
  return onnxruntime::make_unique<Provider_EdgeIterator_Impl>(p_->OutputEdgesEnd());
}

struct Provider_IndexedSubGraph_Impl : Provider_IndexedSubGraph {
  Provider_IndexedSubGraph_Impl() = default;
  Provider_IndexedSubGraph_Impl(std::unique_ptr<IndexedSubGraph> p) : p_{std::move(p)} {}

  void SetMetaDef(std::unique_ptr<MetaDef>& def_) override {
    auto real = onnxruntime::make_unique<IndexedSubGraph::MetaDef>();

    real->name = std::move(def_->name);
    real->domain = std::move(def_->domain);
    real->since_version = def_->since_version;
    real->status = def_->status;
    real->inputs = std::move(def_->inputs);
    real->outputs = std::move(def_->outputs);

    for (const auto& v : def_->attributes)
      real->attributes.emplace(v.first, static_cast<Provider_AttributeProto_Impl*>(v.second.p_.get())->v_);

    real->doc_string = std::move(def_->doc_string);

    p_->SetMetaDef(real);
  }

  const MetaDef* GetMetaDef() override {
    __debugbreak();
    __assume(false);
  }

  std::vector<onnxruntime::NodeIndex>& Nodes() override { return p_->nodes; }

  std::unique_ptr<IndexedSubGraph> p_{onnxruntime::make_unique<IndexedSubGraph>()};
};

struct Provider_GraphViewer_Impl : Provider_GraphViewer {
  Provider_GraphViewer_Impl(const GraphViewer& v) : v_(v) {
    for (int i = 0; i < v_.MaxNodeIndex(); i++)
      provider_nodes_.emplace_back(onnxruntime::make_unique<Provider_Node_Impl>(v_.GetNode(i)));
  }

  std::unique_ptr<Provider_Model> CreateModel() const override {
    __debugbreak();
    __assume(false);
  }

  const std::string& Name() const noexcept override { return v_.Name(); }

  const Provider_Node* GetNode(NodeIndex node_index) const override {
    auto& node = *provider_nodes_[node_index];
    if (node.p_)
      return &node;
    return nullptr;
  }

  const Provider_NodeArg* GetNodeArg(const std::string& name) const override {
    __debugbreak();
    __assume(false);
  }

  bool IsSubgraph() const override { return v_.IsSubgraph(); }

  int MaxNodeIndex() const noexcept override { return v_.MaxNodeIndex(); }

  const std::vector<const Provider_NodeArg*>& GetInputs() const noexcept override {
    __debugbreak();
    __assume(false);
  }

  const std::vector<const Provider_NodeArg*>& GetOutputs() const noexcept override {
    __debugbreak();
    __assume(false);
  }

  const std::vector<const Provider_NodeArg*>& GetValueInfo() const noexcept override {
    __debugbreak();
    __assume(false);
  }

  const Provider_InitializedTensorSet& GetAllInitializedTensors() const noexcept override {
    if (initialized_tensor_set_.empty()) {
      initialized_tensors_.reserve(v_.GetAllInitializedTensors().size());

      for (auto& v : v_.GetAllInitializedTensors()) {
        initialized_tensors_.emplace_back(const_cast<ONNX_NAMESPACE::TensorProto*>(v.second));
        initialized_tensor_set_.emplace(v.first, &initialized_tensors_.back());
      }
    }

    return initialized_tensor_set_;
  }

  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept override { return v_.DomainToVersionMap(); }

  const std::vector<NodeIndex>& GetNodesInTopologicalOrder() const override { return v_.GetNodesInTopologicalOrder(); }

  const GraphViewer& v_;

  std::vector<std::unique_ptr<Provider_Node_Impl>> provider_nodes_;

  mutable std::vector<Provider_TensorProto_Impl> initialized_tensors_;
  mutable Provider_InitializedTensorSet initialized_tensor_set_;

  mutable std::vector<Provider_NodeArg_Impl> inputs_impl_;
  mutable std::vector<Provider_NodeArg*> inputs_;
};
#endif

struct Provider_Tensor_Impl final : Provider_Tensor {
  Provider_Tensor_Impl(const Tensor* p) : p_(const_cast<Tensor*>(p)) {}

  float* MutableData_float() override { return p_->MutableData<float>(); }
  const float* Data_float() const override { return p_->Data<float>(); }

  const TensorShape& Shape() const override { return p_->Shape(); }

  Tensor* p_;
};

struct Provider_IDataTransfer_Impl : Provider_IDataTransfer {
  Provider_IDataTransfer_Impl(std::unique_ptr<IDataTransfer> v) : v_{std::move(v)} {}

  std::unique_ptr<IDataTransfer> v_;
};

struct Provider_DataTransferManager_Impl : Provider_DataTransferManager {
  Provider_DataTransferManager_Impl(const DataTransferManager& v) : v_{v} {}

  Status CopyTensor(const Provider_Tensor& src, Provider_Tensor& dst, int exec_queue_id) const override {
    return v_.CopyTensor(*static_cast<const Provider_Tensor_Impl*>(&src)->p_, *static_cast<const Provider_Tensor_Impl*>(&dst)->p_, exec_queue_id);
  }

  const DataTransferManager& v_;
};

struct Provider_OpKernelInfo_Impl : Provider_OpKernelInfo {
  Provider_OpKernelInfo_Impl(const OpKernelInfo& info) : info_(info), data_transfer_manager_{info_.GetDataTransferManager()} {}

  Status GetAttr(const std::string& name, int64_t* value) const override {
    return info_.GetAttr<int64_t>(name, value);
  }

  Status GetAttr(const std::string& name, float* value) const override {
    return info_.GetAttr<float>(name, value);
  }

  const Provider_DataTransferManager& GetDataTransferManager() const noexcept override {
    return data_transfer_manager_;
  }

  int GetKernelDef_ExecQueueId() const noexcept override {
    return info_.GetKernelDef().ExecQueueId();
  }

  const Provider_DataTransferManager_Impl& data_transfer_manager_;
  const OpKernelInfo& info_;
};

struct Provider_OpKernelContext_Impl : Provider_OpKernelContext {
  Provider_OpKernelContext_Impl(OpKernelContext* context) : p_(context) {}

  const Provider_Tensor* Input_Tensor(int index) const override {
    tensors_.push_back(onnxruntime::make_unique<Provider_Tensor_Impl>(p_->Input<Tensor>(index)));
    return tensors_.back().get();
  }

  Provider_Tensor* Output(int index, const TensorShape& shape) override {
    tensors_.push_back(onnxruntime::make_unique<Provider_Tensor_Impl>(p_->Output(index, shape)));
    return tensors_.back().get();
  }

  OpKernelContext* p_;
  mutable std::vector<std::unique_ptr<Provider_Tensor_Impl>> tensors_;
};

struct Provider_OpKernel_Impl : Provider_OpKernel {
  OpKernelInfo op_kernel_info_;
};

struct OpKernel_Translator : OpKernel {
  OpKernel_Translator(Provider_OpKernelInfo_Impl& info, Provider_OpKernel* p) : OpKernel(info.info_), p_(p) {}
  ~OpKernel_Translator() {
    delete p_;
  }

  Status Compute(OpKernelContext* context) const override {
    Provider_OpKernelContext_Impl provider_context(context);
    return p_->Compute(&provider_context);
  }

  Provider_OpKernel* p_;
};

#if 0
struct Provider_KernelRegistry_Impl : Provider_KernelRegistry {
  Provider_KernelRegistry_Impl(std::shared_ptr<KernelRegistry> p) : p_owned_(p) {}
  Provider_KernelRegistry_Impl(KernelRegistry* p) : p_(p) {}
  Provider_KernelRegistry_Impl() : p_owned_(std::make_shared<KernelRegistry>()) {}

  Status Register(Provider_KernelCreateInfo&& create_info) override {
    KernelCreateInfo info_real(std::move(*reinterpret_cast<std::unique_ptr<KernelDef>*>(&create_info.kernel_def)),
                               [kernel_create_func = create_info.kernel_create_func](const OpKernelInfo& info) -> OpKernel* {
                                 Provider_OpKernelInfo_Impl provider_info(info);
                                 return new OpKernel_Translator(provider_info, kernel_create_func(provider_info));
                               });

    return p_->Register(std::move(info_real));
  }

  std::shared_ptr<KernelRegistry> p_owned_;
  KernelRegistry* p_{&*p_owned_};
};
#endif

struct Provider_IExecutionProvider_Router_Impl : Provider_IExecutionProvider_Router, IExecutionProvider {
  Provider_IExecutionProvider_Router_Impl(Provider_IExecutionProvider* outer, const std::string& type) : IExecutionProvider(type), outer_(outer) {
  }

  virtual ~Provider_IExecutionProvider_Router_Impl() {}

  std::shared_ptr<Provider_KernelRegistry> Provider_GetKernelRegistry() const override {
    auto result = GetKernelRegistry();
    return *reinterpret_cast<std::shared_ptr<Provider_KernelRegistry>*>(&result);
    //    return std::make_shared<Provider_KernelRegistry_Impl>(GetKernelRegistry());
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    auto result = outer_->Provider_GetKernelRegistry();
    return *reinterpret_cast<std::shared_ptr<KernelRegistry>*>(&result);
    //    return static_cast<Provider_KernelRegistry_Impl*>(&*outer_->Provider_GetKernelRegistry())->p_owned_;
  }

#if 0
  std::vector<std::unique_ptr<Provider_ComputeCapability>> Provider_GetCapability(const onnxruntime::Provider_GraphViewer& graph,
                                                                                  const std::vector<const Provider_KernelRegistry*>& kernel_registries) const override {
    std::vector<const KernelRegistry*> kernel_registries_internal;
    for (auto& v : kernel_registries)
      kernel_registries_internal.emplace_back(static_cast<const Provider_KernelRegistry_Impl*>(v)->p_);

    auto capabilities_internal = IExecutionProvider::GetCapability(*reinterpret_cast<const GraphViewer*>(&graph), kernel_registries_internal);

    std::vector<std::unique_ptr<Provider_ComputeCapability>> capabilities;
    for (auto& v : capabilities_internal)
      capabilities.emplace_back(onnxruntime::make_unique<Provider_ComputeCapability>(std::move(*reinterpret_cast<std::unique_ptr<Provider_IndexedSubGraph>*>(&v->sub_graph))));
    return capabilities;
  }

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const onnxruntime::GraphViewer& graph,
                                                                const std::vector<const KernelRegistry*>& kernel_registries) const override {
    std::vector<const Provider_KernelRegistry*> registries;
    for (auto p : kernel_registries)
      registries.push_back(new Provider_KernelRegistry_Impl(const_cast<KernelRegistry*>(p)));

    auto provider_result = outer_->Provider_GetCapability(*reinterpret_cast<const Provider_GraphViewer*>(&graph), registries);
    std::vector<std::unique_ptr<ComputeCapability>> result;

    for (auto& p : provider_result)
      result.emplace_back(onnxruntime::make_unique<ComputeCapability>(std::move(*reinterpret_cast<std::unique_ptr<IndexedSubGraph>*>(&p->t_sub_graph_))));

    for (auto p : registries)
      delete p;

    return result;
  }
#endif

  std::vector<std::unique_ptr<Provider_ComputeCapability>> Provider_GetCapability(const onnxruntime::Provider_GraphViewer& graph,
                                                                                  const std::vector<const Provider_KernelRegistry*>& kernel_registries) const override {
    auto capabilities_internal = IExecutionProvider::GetCapability(*reinterpret_cast<const GraphViewer*>(&graph), *reinterpret_cast<const std::vector<const KernelRegistry*>*>(&kernel_registries));
    return std::move(*reinterpret_cast<std::vector<std::unique_ptr<Provider_ComputeCapability>>*>(&capabilities_internal));
  }

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const onnxruntime::GraphViewer& graph,
                                                                const std::vector<const KernelRegistry*>& kernel_registries) const override {
    auto provider_result = outer_->Provider_GetCapability(*reinterpret_cast<const Provider_GraphViewer*>(&graph), *reinterpret_cast<const std::vector<const Provider_KernelRegistry*>*>(&kernel_registries));
    return std::move(*reinterpret_cast<std::vector<std::unique_ptr<ComputeCapability>>*>(&provider_result));
  }

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) override {
    return outer_->Provider_Compile(*reinterpret_cast<const std::vector<Provider_Node*>*>(&fused_nodes), node_compute_funcs);
  }

  Provider_AllocatorPtr Provider_GetAllocator(int id, OrtMemType mem_type) const override {
    return std::make_shared<Provider_IAllocator_Impl>(IExecutionProvider::GetAllocator(id, mem_type));
  }

  void Provider_InsertAllocator(Provider_AllocatorPtr allocator) override {
    IExecutionProvider::InsertAllocator(static_cast<Provider_IAllocator_Impl*>(allocator.get())->p_);
  }

  const logging::Logger* GetLogger() const override { return IExecutionProvider::GetLogger(); }

  std::unique_ptr<Provider_IExecutionProvider> outer_;
};

static_assert(sizeof(Provider_IndexedSubGraph::MetaDef) == sizeof(IndexedSubGraph::MetaDef), "These must match for the interface to work");

struct ProviderHostImpl : ProviderHost {
  ProviderHostImpl() {
    DataTypeImpl_GetType_Tensor = &DataTypeImpl::GetType<Tensor>;
    DataTypeImpl_GetType_float = &DataTypeImpl::GetType<float>;
    DataTypeImpl_GetTensorType_float = &DataTypeImpl::GetTensorType<float>;
  }

  std::unique_ptr<ONNX_NAMESPACE::Provider_AttributeProto> AttributeProto_Create() override {
    return onnxruntime::make_unique<Provider_AttributeProto_Impl>();
  }

  std::unique_ptr<Provider_OrtMemoryInfo> OrtMemoryInfo_Create(const char* name_, OrtAllocatorType type_, Provider_OrtDevice* device_, int id_, OrtMemType mem_type_) override {
    return onnxruntime::make_unique<Provider_OrtMemoryInfo_Impl>(name_, type_, device_ ? static_cast<Provider_OrtDevice_Impl*>(device_)->v_ : OrtDevice(), id_, mem_type_);
  }

#if 0
  std::unique_ptr<Provider_KernelDefBuilder> KernelDefBuilder_Create() override {
    return onnxruntime::make_unique<Provider_KernelDefBuilder_Impl>();
  }

  std::shared_ptr<Provider_KernelRegistry> KernelRegistry_Create() override {
    return std::make_shared<Provider_KernelRegistry_Impl>();
  }

  std::unique_ptr<Provider_IndexedSubGraph> IndexedSubGraph_Create() override {
    return onnxruntime::make_unique<Provider_IndexedSubGraph_Impl>();
  }
#endif

  Provider_AllocatorPtr CreateAllocator(const Provider_DeviceAllocatorRegistrationInfo& info,
                                        OrtDevice::DeviceId device_id = 0,
                                        bool use_arena = true) override {
    DeviceAllocatorRegistrationInfo info_real{
        info.mem_type, [&info](int value) {
          return std::move(static_cast<Provider_IDeviceAllocator_Impl*>(&*info.factory(value))->p_);
        },
        info.max_mem};

    return std::make_shared<Provider_IAllocator_Impl>(onnxruntime::CreateAllocator(info_real, device_id, use_arena));
  }

  std::unique_ptr<Provider_IDeviceAllocator> CreateCPUAllocator(
      std::unique_ptr<Provider_OrtMemoryInfo> memory_info) override {
    return onnxruntime::make_unique<Provider_IDeviceAllocator_Impl>(
        onnxruntime::make_unique<CPUAllocator>(*static_cast<Provider_OrtMemoryInfo_Impl*>(memory_info.get())->info_));
  };

  std::unique_ptr<Provider_IExecutionProvider_Router> Create_IExecutionProvider_Router(
      Provider_IExecutionProvider* outer, const std::string& type) override {
    return onnxruntime::make_unique<Provider_IExecutionProvider_Router_Impl>(outer, type);
  };

#if 0
  std::unique_ptr<Provider_IDeviceAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) override {
    return onnxruntime::make_unique<Provider_IDeviceAllocator_Impl>(onnxruntime::make_unique<CUDAAllocator>(device_id, name));
  }

  std::unique_ptr<Provider_IDeviceAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) override {
    return onnxruntime::make_unique<Provider_IDeviceAllocator_Impl>(onnxruntime::make_unique<CUDAPinnedAllocator>(device_id, name));
  }

  std::unique_ptr<Provider_IDataTransfer> CreateGPUDataTransfer() override {
    return onnxruntime::make_unique<Provider_IDataTransfer_Impl>(onnxruntime::make_unique<GPUDataTransfer>());
  }
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
  void HeapFree(void* p) override { delete reinterpret_cast<uint8_t*>(p); }

  bool CPU_HasAVX2() override {
    return CPUIDInfo::GetCPUIDInfo().HasAVX2();
  }

  bool CPU_HasAVX512f() override {
    return CPUIDInfo::GetCPUIDInfo().HasAVX512f();
  }

  void LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file, const char* function, uint32_t line) override {
    return ::onnxruntime::LogRuntimeError(session_id, status, file, function, line);
  }

  // Provider_TypeProto_Tensor
  int32_t Provider_TypeProto_Tensor__elem_type(const Provider_TypeProto_Tensor* p) override { return reinterpret_cast<const ONNX_NAMESPACE::TypeProto_Tensor*>(p)->elem_type(); }

  // Provider_TypeProto
  const Provider_TypeProto_Tensor& Provider_TypeProto__tensor_type(const Provider_TypeProto* p) override { return *reinterpret_cast<const Provider_TypeProto_Tensor*>(&reinterpret_cast<const ONNX_NAMESPACE::TypeProto*>(p)->tensor_type()); }

  // Provider_GraphProto
  void Provider_GraphProto_destructor(Provider_GraphProto* p) override { delete reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p); }

  Provider_ValueInfoProtos* Provider_GraphProto__mutable_input(Provider_GraphProto* p) override { return reinterpret_cast<Provider_ValueInfoProtos*>(reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p)->mutable_input()); }

  const Provider_ValueInfoProtos& Provider_GraphProto__output(const Provider_GraphProto* p) override { return *reinterpret_cast<const Provider_ValueInfoProtos*>(&reinterpret_cast<const ONNX_NAMESPACE::GraphProto*>(p)->output()); }
  Provider_ValueInfoProtos* Provider_GraphProto__mutable_output(Provider_GraphProto* p) override { return reinterpret_cast<Provider_ValueInfoProtos*>(reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p)->mutable_output()); }

  Provider_ValueInfoProtos* Provider_GraphProto__mutable_value_info(Provider_GraphProto* p) override { return reinterpret_cast<Provider_ValueInfoProtos*>(reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p)->mutable_value_info()); }
  Provider_TensorProtos* Provider_GraphProto__mutable_initializer(Provider_GraphProto* p) override { return reinterpret_cast<Provider_TensorProtos*>(reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p)->mutable_initializer()); }
  Provider_NodeProto* Provider_GraphProto__add_node(Provider_GraphProto* p) override { return reinterpret_cast<Provider_NodeProto*>(reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p)->add_node()); }

  void Provider_GraphProto__operator_assign(Provider_GraphProto* p, const Provider_GraphProto& v) override { *reinterpret_cast<ONNX_NAMESPACE::GraphProto*>(p) = *reinterpret_cast<const ONNX_NAMESPACE::GraphProto*>(&v); }

  // Provider_ModelProto
  void Provider_ModelProto__destructor(Provider_ModelProto* p) override { delete reinterpret_cast<ONNX_NAMESPACE::ModelProto*>(p); }

  bool Provider_ModelProto__SerializeToString(const Provider_ModelProto* p, std::string& string) override { return reinterpret_cast<const ONNX_NAMESPACE::ModelProto*>(p)->SerializeToString(&string); }
  bool Provider_ModelProto__SerializeToOstream(const Provider_ModelProto* p, std::ostream& output) override { return reinterpret_cast<const ONNX_NAMESPACE::ModelProto*>(p)->SerializeToOstream(&output); }

  const ONNX_NAMESPACE::Provider_GraphProto& Provider_ModelProto__graph(const Provider_ModelProto* p) override { return *reinterpret_cast<const ONNX_NAMESPACE::Provider_GraphProto*>(&reinterpret_cast<const ONNX_NAMESPACE::ModelProto*>(p)->graph()); }
  ONNX_NAMESPACE::Provider_GraphProto* Provider_ModelProto__mutable_graph(Provider_ModelProto* p) override { return reinterpret_cast<ONNX_NAMESPACE::Provider_GraphProto*>(reinterpret_cast<ONNX_NAMESPACE::ModelProto*>(p)->mutable_graph()); }

  void Provider_ModelProto__set_ir_version(Provider_ModelProto* p, int64_t value) override { reinterpret_cast<ONNX_NAMESPACE::ModelProto*>(p)->set_ir_version(value); }

  // Provider_TensorProto
  void Provider_TensorProto__destructor(Provider_TensorProto* p) override { delete reinterpret_cast<ONNX_NAMESPACE::TensorProto*>(p); }
  void Provider_TensorProto__operator_assign(Provider_TensorProto* p, const Provider_TensorProto& v) override { *reinterpret_cast<ONNX_NAMESPACE::TensorProto*>(p) = *reinterpret_cast<const ONNX_NAMESPACE::TensorProto*>(&v); }

  // Provider_TensorProtos
  Provider_TensorProto* Provider_TensorProtos__Add(Provider_TensorProtos* p) override { return reinterpret_cast<Provider_TensorProto*>(reinterpret_cast<google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::TensorProto>*>(p)->Add()); }

  // Provider_TensorShapeProto_Dimension
  const std::string& Provider_TensorShapeProto_Dimension__dim_param(const Provider_TensorShapeProto_Dimension* p) override {
    return reinterpret_cast<const ONNX_NAMESPACE::TensorShapeProto_Dimension*>(p)->dim_param();
  }

  // Provider_TensorShapeProto_Dimensions
  std::unique_ptr<Provider_TensorShapeProto_Dimension_Iterator> Provider_TensorShapeProto_Dimensions__begin(const Provider_TensorShapeProto_Dimensions* p) override {
    return onnxruntime::make_unique<Provider_TensorShapeProto_Dimension_Iterator_Impl>(reinterpret_cast<const google::protobuf::RepeatedPtrField<::ONNX_NAMESPACE::TensorShapeProto_Dimension>*>(p)->begin());
  }

  std::unique_ptr<Provider_TensorShapeProto_Dimension_Iterator> Provider_TensorShapeProto_Dimensions__end(const Provider_TensorShapeProto_Dimensions* p) override {
    return onnxruntime::make_unique<Provider_TensorShapeProto_Dimension_Iterator_Impl>(reinterpret_cast<const google::protobuf::RepeatedPtrField<::ONNX_NAMESPACE::TensorShapeProto_Dimension>*>(p)->end());
  }

  // Provider_TensorShapeProto
  int Provider_TensorShapeProto__dim_size(const Provider_TensorShapeProto* p) override { return reinterpret_cast<const ONNX_NAMESPACE::TensorShapeProto*>(p)->dim_size(); }
  const Provider_TensorShapeProto_Dimensions& Provider_TensorShapeProto__dim(const Provider_TensorShapeProto* p) override { return *reinterpret_cast<const Provider_TensorShapeProto_Dimensions*>(&reinterpret_cast<const ONNX_NAMESPACE::TensorShapeProto*>(p)->dim()); }

  // Provider_ValueInfoProto
  const Provider_TypeProto& Provider_ValueInfoProto__type(const Provider_ValueInfoProto* p) override { return *reinterpret_cast<const Provider_TypeProto*>(&reinterpret_cast<const ONNX_NAMESPACE::ValueInfoProto*>(p)->type()); }
  virtual void Provider_ValueInfoProto__operator_assign(Provider_ValueInfoProto* p, const Provider_ValueInfoProto& v) override { *reinterpret_cast<ONNX_NAMESPACE::ValueInfoProto*>(p) = *reinterpret_cast<const ONNX_NAMESPACE::ValueInfoProto*>(&v); }

  // Provider_ValueInfoProtos
  Provider_ValueInfoProto* Provider_ValueInfoProtos__Add(Provider_ValueInfoProtos* p) override { return reinterpret_cast<Provider_ValueInfoProto*>(reinterpret_cast<google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::ValueInfoProto>*>(p)->Add()); }

  const Provider_ValueInfoProto& Provider_ValueInfoProtos__operator_array(const Provider_ValueInfoProtos* p, int index) override { return *reinterpret_cast<const Provider_ValueInfoProto*>(&(*reinterpret_cast<const google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::ValueInfoProto>*>(p))[index]); }

  // Provider_ComputeCapability
  std::unique_ptr<Provider_ComputeCapability> Provider_ComputeCapability__construct(std::unique_ptr<Provider_IndexedSubGraph> t_sub_graph) override { return std::unique_ptr<Provider_ComputeCapability>(reinterpret_cast<Provider_ComputeCapability*>(new ComputeCapability(std::unique_ptr<IndexedSubGraph>(reinterpret_cast<IndexedSubGraph*>(t_sub_graph.release()))))); }
  std::unique_ptr<Provider_IndexedSubGraph>& Provider_ComputeCapability__SubGraph(Provider_ComputeCapability* p) override { return *reinterpret_cast<std::unique_ptr<Provider_IndexedSubGraph>*>(&reinterpret_cast<ComputeCapability*>(p)->sub_graph); }

  // Provider_IndexedSubGraph
  std::unique_ptr<Provider_IndexedSubGraph> Provider_IndexedSubGraph__construct() override { return std::unique_ptr<Provider_IndexedSubGraph>(reinterpret_cast<Provider_IndexedSubGraph*>(new IndexedSubGraph())); }
  void Provider_IndexedSubGraph__operator_delete(Provider_IndexedSubGraph* p) override { delete reinterpret_cast<IndexedSubGraph*>(p); }

  std::vector<onnxruntime::NodeIndex>& Provider_IndexedSubGraph__Nodes(Provider_IndexedSubGraph* p) override { return reinterpret_cast<IndexedSubGraph*>(p)->nodes; }

  void Provider_IndexedSubGraph__SetMetaDef(Provider_IndexedSubGraph* p, std::unique_ptr<Provider_IndexedSubGraph_MetaDef>&& meta_def_) override { return reinterpret_cast<IndexedSubGraph*>(p)->SetMetaDef(std::move(*reinterpret_cast<std::unique_ptr<IndexedSubGraph::MetaDef>*>(&meta_def_))); }
  const Provider_IndexedSubGraph_MetaDef* Provider_IndexedSubGraph__GetMetaDef(const Provider_IndexedSubGraph* p) override { return reinterpret_cast<const Provider_IndexedSubGraph_MetaDef*>(reinterpret_cast<const IndexedSubGraph*>(p)->GetMetaDef()); }

  // Provider_KernelDef
  void Provider_KernelDef__operator_delete(Provider_KernelDef* p) override { delete reinterpret_cast<KernelDef*>(p); }

  // Provider_KernelDefBuilder
  std::unique_ptr<Provider_KernelDefBuilder> Provider_KernelDefBuilder__construct() override { return std::unique_ptr<Provider_KernelDefBuilder>(reinterpret_cast<Provider_KernelDefBuilder*>(new KernelDefBuilder())); }
  void Provider_KernelDefBuilder__operator_delete(Provider_KernelDefBuilder* p) override { delete reinterpret_cast<KernelDefBuilder*>(p); }

  void Provider_KernelDefBuilder__SetName(Provider_KernelDefBuilder* p, const char* op_name) override { reinterpret_cast<KernelDefBuilder*>(p)->SetName(op_name); }
  void Provider_KernelDefBuilder__SetDomain(Provider_KernelDefBuilder* p, const char* domain) override { reinterpret_cast<KernelDefBuilder*>(p)->SetDomain(domain); }
  void Provider_KernelDefBuilder__SinceVersion(Provider_KernelDefBuilder* p, int since_version) override { reinterpret_cast<KernelDefBuilder*>(p)->SinceVersion(since_version); }
  void Provider_KernelDefBuilder__Provider(Provider_KernelDefBuilder* p, const char* provider_type) override { reinterpret_cast<KernelDefBuilder*>(p)->Provider(provider_type); }
  void Provider_KernelDefBuilder__TypeConstraint(Provider_KernelDefBuilder* p, const char* arg_name, MLDataType supported_type) override { reinterpret_cast<KernelDefBuilder*>(p)->TypeConstraint(arg_name, supported_type); }
  void Provider_KernelDefBuilder__TypeConstraint(Provider_KernelDefBuilder* p, const char* arg_name, const std::vector<MLDataType>& supported_types) override { reinterpret_cast<KernelDefBuilder*>(p)->TypeConstraint(arg_name, supported_types); }
  void Provider_KernelDefBuilder__InputMemoryType(Provider_KernelDefBuilder* p, OrtMemType type, int input_index) override { reinterpret_cast<KernelDefBuilder*>(p)->InputMemoryType(type, input_index); }
  void Provider_KernelDefBuilder__OutputMemoryType(Provider_KernelDefBuilder* p, OrtMemType type, int input_index) override { reinterpret_cast<KernelDefBuilder*>(p)->OutputMemoryType(type, input_index); }
  void Provider_KernelDefBuilder__ExecQueueId(Provider_KernelDefBuilder* p, int queue_id) override { reinterpret_cast<KernelDefBuilder*>(p)->ExecQueueId(queue_id); }

  std::unique_ptr<Provider_KernelDef> Provider_KernelDefBuilder__Build(Provider_KernelDefBuilder* p) override {
    return std::unique_ptr<Provider_KernelDef>(reinterpret_cast<Provider_KernelDef*>(reinterpret_cast<KernelDefBuilder*>(p)->Build().release()));
  }

  // Provider_KernelRegistry
  std::shared_ptr<Provider_KernelRegistry> Provider_KernelRegistry__construct() override {
    auto result = std::make_shared<KernelRegistry>();
    return *reinterpret_cast<std::shared_ptr<Provider_KernelRegistry>*>(&result);
  }
  void Provider_KernelRegistry__operator_delete(Provider_KernelRegistry* p) override { delete reinterpret_cast<KernelRegistry*>(p); }
  Status Provider_KernelRegistry__Register(Provider_KernelRegistry* p, Provider_KernelCreateInfo&& create_info) override {
    KernelCreateInfo info_real(std::move(*reinterpret_cast<std::unique_ptr<KernelDef>*>(&create_info.kernel_def)),
                               [kernel_create_func = create_info.kernel_create_func](const OpKernelInfo& info) -> OpKernel* {
                                 Provider_OpKernelInfo_Impl provider_info(info);
                                 return new OpKernel_Translator(provider_info, kernel_create_func(provider_info));
                               });
    return reinterpret_cast<KernelRegistry*>(p)->Register(std::move(info_real));
  }

  // Provider_Function
  const Provider_Graph& Provider_Function__Body(const Provider_Function* p) override { return *reinterpret_cast<const Provider_Graph*>(&reinterpret_cast<const Function*>(p)->Body()); }

  // Provider_Node
  const std::string& Provider_Node__Name(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->Name(); }
  const std::string& Provider_Node__Description(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->Description(); }
  const std::string& Provider_Node__Domain(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->Domain(); }
  const std::string& Provider_Node__OpType(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->OpType(); }

  const Provider_Function* Provider_Node__GetFunctionBody(const Provider_Node* p) noexcept override { return reinterpret_cast<const Provider_Function*>(reinterpret_cast<const Node*>(p)->GetFunctionBody()); }

  ConstPointerContainer<std::vector<Provider_NodeArg*>> Provider_Node__InputDefs(const Provider_Node* p) noexcept override {
    auto result = reinterpret_cast<const Node*>(p)->InputDefs();
    return *reinterpret_cast<ConstPointerContainer<std::vector<Provider_NodeArg*>>*>(&result);
  }
  ConstPointerContainer<std::vector<Provider_NodeArg*>> Provider_Node__OutputDefs(const Provider_Node* p) noexcept override {
    auto result = reinterpret_cast<const Node*>(p)->OutputDefs();
    return *reinterpret_cast<ConstPointerContainer<std::vector<Provider_NodeArg*>>*>(&result);
  }

  NodeIndex Provider_Node__Index(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->Index(); }

  void Provider_Node__ToProto(const Provider_Node* p, Provider_NodeProto& proto, bool update_subgraphs = false) override { reinterpret_cast<const Node*>(p)->ToProto(*reinterpret_cast<ONNX_NAMESPACE::NodeProto*>(&proto), update_subgraphs); }

  const Provider_NodeAttributes& Provider_Node__GetAttributes(const Provider_Node* p) noexcept override { return *reinterpret_cast<const Provider_NodeAttributes*>(&reinterpret_cast<const Node*>(p)->GetAttributes()); }
  size_t Provider_Node__GetInputEdgesCount(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->GetInputEdgesCount(); }
  size_t Provider_Node__GetOutputEdgesCount(const Provider_Node* p) noexcept override { return reinterpret_cast<const Node*>(p)->GetOutputEdgesCount(); }

  std::unique_ptr<Provider_Node__NodeIterator> Provider_Node__InputNodesBegin(const Provider_Node* p) noexcept override { return onnxruntime::make_unique<Provider_Node__NodeIterator_Impl>(reinterpret_cast<const Node*>(p)->InputNodesBegin()); }
  std::unique_ptr<Provider_Node__NodeIterator> Provider_Node__InputNodesEnd(const Provider_Node* p) noexcept override { return onnxruntime::make_unique<Provider_Node__NodeIterator_Impl>(reinterpret_cast<const Node*>(p)->InputNodesEnd()); }

  std::unique_ptr<Provider_Node__EdgeIterator> Provider_Node__OutputEdgesBegin(const Provider_Node* p) noexcept override { return onnxruntime::make_unique<Provider_Node__EdgeIterator_Impl>(reinterpret_cast<const Node*>(p)->OutputEdgesBegin()); }
  std::unique_ptr<Provider_Node__EdgeIterator> Provider_Node__OutputEdgesEnd(const Provider_Node* p) noexcept override { return onnxruntime::make_unique<Provider_Node__EdgeIterator_Impl>(reinterpret_cast<const Node*>(p)->OutputEdgesEnd()); }

  // Provider_NodeArg
  const std::string& Provider_NodeArg__Name(const Provider_NodeArg* p) noexcept override { return reinterpret_cast<const NodeArg*>(p)->Name(); }
  const ONNX_NAMESPACE::Provider_TensorShapeProto* Provider_NodeArg__Shape(const Provider_NodeArg* p) override { return reinterpret_cast<const ONNX_NAMESPACE::Provider_TensorShapeProto*>(reinterpret_cast<const NodeArg*>(p)->Shape()); }
  ONNX_NAMESPACE::DataType Provider_NodeArg__Type(const Provider_NodeArg* p) noexcept override { return reinterpret_cast<const NodeArg*>(p)->Type(); }
  const Provider_NodeArgInfo& Provider_NodeArg__ToProto(const Provider_NodeArg* p) noexcept override { return *reinterpret_cast<const Provider_NodeArgInfo*>(&reinterpret_cast<const NodeArg*>(p)->ToProto()); }
  bool Provider_NodeArg__Exists(const Provider_NodeArg* p) const noexcept override { return reinterpret_cast<const NodeArg*>(p)->Exists(); }
  const ONNX_NAMESPACE::Provider_TypeProto* Provider_NodeArg__TypeAsProto(const Provider_NodeArg* p) noexcept override { return reinterpret_cast<const ONNX_NAMESPACE::Provider_TypeProto*>(reinterpret_cast<const NodeArg*>(p)->TypeAsProto()); }

  // Provider_Model
  void Provider_Model__destructor(Provider_Model* p) override { delete reinterpret_cast<Model*>(p); }
  Provider_Graph& Provider_Model__MainGraph(Provider_Model* p) override { return *reinterpret_cast<Provider_Graph*>(&reinterpret_cast<Model*>(p)->MainGraph()); }
  std::unique_ptr<Provider_ModelProto> Provider_Model__ToProto(Provider_Model* p) override { return std::unique_ptr<Provider_ModelProto>(reinterpret_cast<Provider_ModelProto*>(new ONNX_NAMESPACE::ModelProto(reinterpret_cast<Model*>(p)->ToProto()))); }

  // Provider_Graph
  std::unique_ptr<Provider_GraphViewer> Provider_Graph__CreateGraphViewer(const Provider_Graph* p) override {
    return std::unique_ptr<Provider_GraphViewer>(reinterpret_cast<Provider_GraphViewer*>(new GraphViewer(*reinterpret_cast<const Graph*>(p))));
  }

  std::unique_ptr<Provider_GraphProto> Provider_Graph__ToGraphProto(const Provider_Graph* p) override {
    return std::unique_ptr<Provider_GraphProto>(reinterpret_cast<Provider_GraphProto*>(new ONNX_NAMESPACE::GraphProto(reinterpret_cast<const Graph*>(p)->ToGraphProto())));
  }

  Provider_NodeArg& Provider_Graph__GetOrCreateNodeArg(Provider_Graph* p, const std::string& name, const Provider_TypeProto* p_arg_type) override {
    return *reinterpret_cast<Provider_NodeArg*>(&reinterpret_cast<Graph*>(p)->GetOrCreateNodeArg(name, reinterpret_cast<const ONNX_NAMESPACE::TypeProto*>(p_arg_type)));
  }

  Status Provider_Graph__Resolve(Provider_Graph* p) override { return reinterpret_cast<Graph*>(p)->Resolve(); }
  void Provider_Graph__AddInitializedTensor(Provider_Graph* p, const Provider_TensorProto& tensor) override { reinterpret_cast<Graph*>(p)->AddInitializedTensor(*reinterpret_cast<const ONNX_NAMESPACE::TensorProto*>(&tensor)); }
  Provider_Node& Provider_Graph__AddNode(Provider_Graph* p, const std::string& name, const std::string& op_type, const std::string& description, const std::vector<Provider_NodeArg*>& input_args, const std::vector<Provider_NodeArg*>& output_args, const Provider_NodeAttributes* attributes, const std::string& domain) override {
    return *reinterpret_cast<Provider_Node*>(&reinterpret_cast<Graph*>(p)->AddNode(name, op_type, description, *reinterpret_cast<const std::vector<NodeArg*>*>(&input_args), *reinterpret_cast<const std::vector<NodeArg*>*>(&output_args), reinterpret_cast<const NodeAttributes*>(attributes), domain));
  }

  const std::vector<const Provider_NodeArg*>& Provider_Graph__GetOutputs(const Provider_Graph* p) noexcept override { return *reinterpret_cast<const std::vector<const Provider_NodeArg*>*>(&reinterpret_cast<const Graph*>(p)->GetOutputs()); }
  void Provider_Graph__SetOutputs(Provider_Graph* p, const std::vector<const Provider_NodeArg*>& outputs) override { reinterpret_cast<Graph*>(p)->SetOutputs(*reinterpret_cast<const std::vector<const NodeArg*>*>(&outputs)); }

  const std::vector<const Provider_NodeArg*>& Provider_Graph__GetInputs(const Provider_Graph* p) noexcept override { return *reinterpret_cast<const std::vector<const Provider_NodeArg*>*>(&reinterpret_cast<const Graph*>(p)->GetInputs()); }

  // Provider_GraphViewer
  void Provider_GraphViewer__destructor(Provider_GraphViewer* p) override { delete reinterpret_cast<GraphViewer*>(p); }
  std::unique_ptr<Provider_Model> Provider_GraphViewer__CreateModel(const Provider_GraphViewer* p, const logging::Logger& logger) override {
    auto& graph_viewer = *reinterpret_cast<const GraphViewer*>(p);
    return std::unique_ptr<Provider_Model>(reinterpret_cast<Provider_Model*>(new Model(graph_viewer.Name(), true, ModelMetaData(), PathString(),
                                                                                       IOnnxRuntimeOpSchemaRegistryList(), graph_viewer.DomainToVersionMap(),
                                                                                       std::vector<ONNX_NAMESPACE::FunctionProto>(), logger)));
  }

  const std::string& Provider_GraphViewer__Name(const Provider_GraphViewer* p) noexcept override { return reinterpret_cast<const GraphViewer*>(p)->Name(); }

  const Provider_Node* Provider_GraphViewer__GetNode(const Provider_GraphViewer* p, NodeIndex node_index) override { return reinterpret_cast<const Provider_Node*>(reinterpret_cast<const GraphViewer*>(p)->GetNode(node_index)); }
  const Provider_NodeArg* Provider_GraphViewer__GetNodeArg(const Provider_GraphViewer* p, const std::string& name) override { return reinterpret_cast<const Provider_NodeArg*>(reinterpret_cast<const GraphViewer*>(p)->GetNodeArg(name)); }

  bool Provider_GraphViewer__IsSubgraph(const Provider_GraphViewer* p) override { return reinterpret_cast<const GraphViewer*>(p)->IsSubgraph(); }
  int Provider_GraphViewer__MaxNodeIndex(const Provider_GraphViewer* p) noexcept override { return reinterpret_cast<const GraphViewer*>(p)->MaxNodeIndex(); }

  const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetInputs(const Provider_GraphViewer* p) noexcept override { return *reinterpret_cast<const std::vector<const Provider_NodeArg*>*>(&reinterpret_cast<const GraphViewer*>(p)->GetInputs()); }
  const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetOutputs(const Provider_GraphViewer* p) noexcept override { return *reinterpret_cast<const std::vector<const Provider_NodeArg*>*>(&reinterpret_cast<const GraphViewer*>(p)->GetOutputs()); }
  const std::vector<const Provider_NodeArg*>& Provider_GraphViewer__GetValueInfo(const Provider_GraphViewer* p) noexcept { return *reinterpret_cast<const std::vector<const Provider_NodeArg*>*>(&reinterpret_cast<const GraphViewer*>(p)->GetValueInfo()); }

  const Provider_InitializedTensorSet& Provider_GraphViewer__GetAllInitializedTensors(const Provider_GraphViewer* p) override { return *reinterpret_cast<const Provider_InitializedTensorSet*>(&reinterpret_cast<const GraphViewer*>(p)->GetAllInitializedTensors()); }

  const std::unordered_map<std::string, int>& Provider_GraphViewer__DomainToVersionMap(const Provider_GraphViewer* p) override { return reinterpret_cast<const GraphViewer*>(p)->DomainToVersionMap(); }

  const std::vector<NodeIndex>& Provider_GraphViewer__GetNodesInTopologicalOrder(const Provider_GraphViewer* p) override { return reinterpret_cast<const GraphViewer*>(p)->GetNodesInTopologicalOrder(); }
} provider_host_;

struct ProviderLibrary {
  ProviderLibrary(const char* filename) {
    std::string full_path = Env::Default().GetRuntimePath() + std::string(filename);
    Env::Default().LoadDynamicLibrary(full_path, &handle_);
    if (!handle_)
      return;

#if defined(_WIN32) && !defined(_OPENMP)
    {
      // We crash when unloading DNNL on Windows when OpenMP also unloads (As there are threads
      // still running code inside the openmp runtime DLL if OMP_WAIT_POLICY is set to ACTIVE).
      // To avoid this, we pin the OpenMP DLL so that it unloads as late as possible.
      HMODULE handle{};
#ifdef _DEBUG
      constexpr const char* dll_name = "vcomp140d.dll";
#else
      constexpr const char* dll_name = "vcomp140.dll";
#endif
      ::GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_PIN, dll_name, &handle);
      assert(handle);  // It should exist
    }
#endif

    Provider* (*PGetProvider)();
    Env::Default().GetSymbolFromLibrary(handle_, "GetProvider", (void**)&PGetProvider);

    provider_ = PGetProvider();
    provider_->SetProviderHost(provider_host_);
  }

  ProviderLibrary() {
    //    Provider* (*PGetProvider)() = &Tensorrt_GetProvider;
    //    provider_ = PGetProvider();
    provider_->SetProviderHost(provider_host_);
  }

  ~ProviderLibrary() {
    Env::Default().UnloadDynamicLibrary(handle_);
  }

  Provider* provider_{};
  void* handle_{};
};

// This class translates the IExecutionProviderFactory interface to work with the interface providers implement
struct IExecutionProviderFactory_Translator : IExecutionProviderFactory {
  IExecutionProviderFactory_Translator(std::shared_ptr<Provider_IExecutionProviderFactory> p) : p_{p} {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    auto provider = p_->CreateProvider();
    return std::unique_ptr<IExecutionProvider>(static_cast<Provider_IExecutionProvider_Router_Impl*>(provider.release()->p_));
  }

  std::shared_ptr<Provider_IExecutionProviderFactory> p_;
};

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(int device_id) {
#ifdef _WIN32
  static ProviderLibrary library("onnxruntime_providers_dnnl.dll");
#elif defined(__APPLE__)
  static ProviderLibrary library("libonnxruntime_providers_dnnl.dylib");
#else
  static ProviderLibrary library("libonnxruntime_providers_dnnl.so");
#endif
  if (!library.provider_) {
    LOGS_DEFAULT(ERROR) << "Failed to load provider shared library";
    return nullptr;
  }

  //return std::make_shared<onnxruntime::MkldnnProviderFactory>(device_id);
  //TODO: This is apparently a bug. The constructor parameter is create-arena-flag, not the device-id
  return std::make_shared<IExecutionProviderFactory_Translator>(library.provider_->CreateExecutionProviderFactory(device_id));
}

extern "C" {
ORT_API(onnxruntime::Provider*, GetProvider_Tensorrt);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id) {
#if 0
#ifdef _WIN32
  static ProviderLibrary library("onnxruntime_providers_tensorrt.dll");
#elif defined(__APPLE__)
  static ProviderLibrary library("libonnxruntime_providers_tensorrt.dylib");
#else
  static ProviderLibrary library("libonnxruntime_providers_tensorrt.so");
#endif
  if (!library.provider_) {
    LOGS_DEFAULT(ERROR) << "Failed to load provider shared library";
    return nullptr;
  }

  //return std::make_shared<onnxruntime::MkldnnProviderFactory>(device_id);
  //TODO: This is apparently a bug. The constructor parameter is create-arena-flag, not the device-id
  return std::make_shared<IExecutionProviderFactory_Translator>(library.provider_->CreateExecutionProviderFactory(device_id));
#endif

  auto provider = GetProvider_Tensorrt();
  provider->SetProviderHost(provider_host_);

  return std::make_shared<IExecutionProviderFactory_Translator>(provider->CreateExecutionProviderFactory(device_id));
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
