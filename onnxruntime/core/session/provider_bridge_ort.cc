// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the Onnxruntime side of the bridge to allow providers to be built as a DLL
// It implements onnxruntime::ProviderHost

#include "core/framework/data_types.h"
#include "core/framework/allocatormgr.h"
#include "core/providers/dnnl/dnnl_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "core/platform/env.h"
#include "core/framework/execution_provider.h"
#include "core/framework/compute_capability.h"
#define PROVIDER_BRIDGE_ORT
#include "core/providers/shared_library/provider_interfaces.h"
#include "onnx/common/stl_backports.h"
#include "core/common/logging/logging.h"
#include "core/common/cpuid_info.h"

namespace onnxruntime {

struct Prov_OrtDevice_Impl : Prov_OrtDevice {
  OrtDevice v_;
};

struct Prov_OrtMemoryInfo_Impl : Prov_OrtMemoryInfo {
  Prov_OrtMemoryInfo_Impl(const char* name_, OrtAllocatorType type_, OrtDevice device_, int id_, OrtMemType mem_type_) : info_{onnxruntime::make_unique<OrtMemoryInfo>(name_, type_, device_, id_, mem_type_)} {}

  std::unique_ptr<OrtMemoryInfo> info_;
};

struct Prov_IAllocator_Impl : Prov_IAllocator {
  Prov_IAllocator_Impl(AllocatorPtr p) : p_{p} {}

  void* Alloc(size_t size) override { return p_->Alloc(size); }
  void Free(void* p) override { return p_->Free(p); }
  const Prov_OrtMemoryInfo& Info() const override {
    PROVIDER_NOT_IMPLEMENTED
    return *(Prov_OrtMemoryInfo*)nullptr;
    //return p_->Info();
  }

  AllocatorPtr p_;
};

struct Prov_IDeviceAllocator_Impl : Prov_IDeviceAllocator {
  Prov_IDeviceAllocator_Impl(std::unique_ptr<IDeviceAllocator> p) : p_{std::move(p)} {}

  void* Alloc(size_t size) override { return p_->Alloc(size); }
  void Free(void* p) override { return p_->Free(p); }
  const Prov_OrtMemoryInfo& Info() const override {
    PROVIDER_NOT_IMPLEMENTED
    return *(Prov_OrtMemoryInfo*)nullptr;
    //return p_->Info();
  }

  bool AllowsArena() const override { return p_->AllowsArena(); }

  std::unique_ptr<IDeviceAllocator> p_;
};

struct Prov_TensorProto_Impl : ONNX_NAMESPACE::Prov_TensorProto {
  Prov_TensorProto_Impl(ONNX_NAMESPACE::TensorProto* p) : p_{p} {}

  void CopyFrom(const Prov_TensorProto& v) override {
    *p_ = *static_cast<const Prov_TensorProto_Impl*>(&v)->p_;
  }

  ONNX_NAMESPACE::TensorProto* p_{};
};  // namespace onnxruntime

struct Prov_AttributeProto_Impl : ONNX_NAMESPACE::Prov_AttributeProto {
  Prov_AttributeProto_Impl() = default;
  Prov_AttributeProto_Impl(const ONNX_NAMESPACE::AttributeProto& copy) : v_{copy} {}

  std::unique_ptr<Prov_AttributeProto> Clone() const override {
    return onnxruntime::make_unique<Prov_AttributeProto_Impl>(v_);
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
  ::onnx::Prov_TensorProto* add_tensors() override {
    // Kind of a hack, but the pointer is only valid until the next add_tensors call
    tensors_ = onnxruntime::make_unique<Prov_TensorProto_Impl>(v_.add_tensors());
    return tensors_.get();
  }

  ONNX_NAMESPACE::AttributeProto v_;
  std::unique_ptr<Prov_TensorProto_Impl> tensors_;
};

struct Prov_KernelDef_Impl : Prov_KernelDef {
  Prov_KernelDef_Impl(std::unique_ptr<KernelDef> p) : p_(std::move(p)) {}
  std::unique_ptr<KernelDef> p_;
};

struct Prov_KernelDefBuilder_Impl : Prov_KernelDefBuilder {
  Prov_KernelDefBuilder& SetName(const char* op_name) override {
    v_.SetName(op_name);
    return *this;
  }
  Prov_KernelDefBuilder& SetDomain(const char* domain) override {
    v_.SetDomain(domain);
    return *this;
  }

  Prov_KernelDefBuilder& SinceVersion(int since_version) override {
    v_.SinceVersion(since_version);
    return *this;
  }
  Prov_KernelDefBuilder& Provider(const char* provider_type) override {
    v_.Provider(provider_type);
    return *this;
  }
  Prov_KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType supported_type) override {
    v_.TypeConstraint(arg_name, supported_type);
    return *this;
  }

  std::unique_ptr<Prov_KernelDef> Build() override {
    return onnxruntime::make_unique<Prov_KernelDef_Impl>(v_.Build());
  }

  KernelDefBuilder v_;
};

struct Prov_NodeArg_Impl : Prov_NodeArg {
  Prov_NodeArg_Impl(const NodeArg* p) : p_{p}, tensor_shape_proto_{p_->Shape()->dim_size()} {}

  const std::string& Name() const noexcept override { return p_->Name(); }
  const ONNX_NAMESPACE::Prov_TensorShapeProto* Shape() const override { return &tensor_shape_proto_; }
  virtual ONNX_NAMESPACE::DataType Type() const noexcept override { return p_->Type(); }

  const NodeArg* p_;
  ONNX_NAMESPACE::Prov_TensorShapeProto tensor_shape_proto_;
};

struct Prov_Node_Impl : Prov_Node {
  Prov_Node_Impl(const Node* p) : p_{p} {}
  ~Prov_Node_Impl() override {
    for (auto p : input_defs_)
      delete p;
    for (auto p : output_defs_)
      delete p;
  }

  const std::string& OpType() const noexcept override { return p_->OpType(); }
  //  const ONNX_NAMESPACE::OpSchema* Op() const noexcept

  ConstPointerContainer<std::vector<Prov_NodeArg*>> InputDefs() const noexcept override {
    if (input_defs_.empty()) {
      for (auto p : p_->InputDefs())
        input_defs_.push_back(new Prov_NodeArg_Impl(p));
    }

    return ConstPointerContainer<std::vector<Prov_NodeArg*>>(input_defs_);
  }

  ConstPointerContainer<std::vector<Prov_NodeArg*>> OutputDefs() const noexcept override {
    if (output_defs_.empty()) {
      for (auto p : p_->OutputDefs())
        output_defs_.push_back(new Prov_NodeArg_Impl(p));
    }

    return ConstPointerContainer<std::vector<Prov_NodeArg*>>(output_defs_);
  }

  NodeIndex Index() const noexcept override { return p_->Index(); }

  const Prov_NodeAttributes& GetAttributes() const noexcept override {
    if (attributes_.empty()) {
      for (auto& v : p_->GetAttributes())
        attributes_[v.first] = onnxruntime::make_unique<Prov_AttributeProto_Impl>(v.second);
    }
    return attributes_;
  }

  size_t GetInputEdgesCount() const noexcept override {
    return p_->GetInputEdgesCount();
  }
  size_t GetOutputEdgesCount() const noexcept override { return p_->GetOutputEdgesCount(); }

  std::unique_ptr<Prov_NodeIterator> InputNodesBegin_internal() const noexcept override;
  std::unique_ptr<Prov_NodeIterator> InputNodesEnd_internal() const noexcept override;

  const Node* p_;
  mutable std::vector<Prov_NodeArg*> input_defs_;
  mutable std::vector<Prov_NodeArg*> output_defs_;
  mutable Prov_NodeAttributes attributes_;
};

struct Prov_NodeIterator_Impl : Prov_Node::Prov_NodeIterator {
  Prov_NodeIterator_Impl(Node::NodeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Prov_NodeIterator& p) const override { return v_ != static_cast<const Prov_NodeIterator_Impl*>(&p)->v_; }

  void operator++() override { return v_.operator++(); }
  const Prov_Node& operator*() override {
    node_ = Prov_Node_Impl(&*v_);
    return node_;
  }

  Node::NodeConstIterator v_;
  Prov_Node_Impl node_{nullptr};
};

std::unique_ptr<Prov_Node::Prov_NodeIterator> Prov_Node_Impl::InputNodesBegin_internal() const noexcept {
  return onnxruntime::make_unique<Prov_NodeIterator_Impl>(p_->InputNodesBegin());
}

std::unique_ptr<Prov_Node::Prov_NodeIterator> Prov_Node_Impl::InputNodesEnd_internal() const noexcept {
  return onnxruntime::make_unique<Prov_NodeIterator_Impl>(p_->InputNodesEnd());
}

struct Prov_IndexedSubGraph_Impl : Prov_IndexedSubGraph {
  Prov_IndexedSubGraph_Impl() = default;
  Prov_IndexedSubGraph_Impl(std::unique_ptr<IndexedSubGraph> p) : p_{std::move(p)} {}

  void SetMetaDef(std::unique_ptr<MetaDef>& def_) override {
    auto real = onnxruntime::make_unique<IndexedSubGraph::MetaDef>();

    real->name = std::move(def_->name);
    real->domain = std::move(def_->domain);
    real->since_version = def_->since_version;
    real->status = def_->status;
    real->inputs = std::move(def_->inputs);
    real->outputs = std::move(def_->outputs);

    for (auto& v : def_->attributes)
      real->attributes.emplace(v.first, static_cast<Prov_AttributeProto_Impl*>(v.second.p_.get())->v_);

    real->doc_string = std::move(def_->doc_string);

    p_->SetMetaDef(real);
  }

  std::vector<onnxruntime::NodeIndex>& Nodes() override { return p_->nodes; }

  std::unique_ptr<IndexedSubGraph> p_{onnxruntime::make_unique<IndexedSubGraph>()};
};

struct Prov_GraphViewer_Impl : Prov_GraphViewer {
  Prov_GraphViewer_Impl(const GraphViewer& v) : v_(v) {
    for (int i = 0; i < v_.MaxNodeIndex(); i++)
      prov_nodes_.emplace_back(v_.GetNode(i));
  }

  const std::string& Name() const noexcept override { return v_.Name(); }
  const Prov_Node* GetNode(NodeIndex node_index) const override {
    auto& node = prov_nodes_[node_index];
    if (node.p_)
      return &node;
    return nullptr;
  }
  int MaxNodeIndex() const noexcept override { return v_.MaxNodeIndex(); }
  const Prov_InitializedTensorSet& GetAllInitializedTensors() const noexcept override {
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

  const GraphViewer& v_;

  std::vector<Prov_Node_Impl> prov_nodes_;

  mutable std::vector<Prov_TensorProto_Impl> initialized_tensors_;
  mutable Prov_InitializedTensorSet initialized_tensor_set_;
};

struct Prov_KernelRegistry_Impl : Prov_KernelRegistry {
  Prov_KernelRegistry_Impl(std::shared_ptr<KernelRegistry> p) : p_owned_(p) {}
  Prov_KernelRegistry_Impl(KernelRegistry* p) : p_(p) {}
  Prov_KernelRegistry_Impl() : p_owned_(std::make_shared<KernelRegistry>()) {}

  Status Register(Prov_KernelCreateInfo&& create_info) override {
    KernelCreateInfo info_real(std::move(static_cast<Prov_KernelDef_Impl*>(create_info.kernel_def.get())->p_),
                               [](const OpKernelInfo& /*info*/) -> OpKernel* {
      PROVIDER_NOT_IMPLEMENTED // So far providers use the function API, not creating kernels this way
      return nullptr;  /*create_info.kernel_create_func);*/ });

    return p_->Register(std::move(info_real));
  }

  std::shared_ptr<KernelRegistry> p_owned_;
  KernelRegistry* p_{&*p_owned_};
};

struct Prov_IExecutionProvider_Router_Impl : Prov_IExecutionProvider_Router, IExecutionProvider {
  Prov_IExecutionProvider_Router_Impl(Prov_IExecutionProvider* outer, const std::string& type) : IExecutionProvider(type), outer_(outer) {
  }

  virtual ~Prov_IExecutionProvider_Router_Impl() {}

  std::shared_ptr<Prov_KernelRegistry> Prov_GetKernelRegistry() const override {
    return std::make_shared<Prov_KernelRegistry_Impl>(GetKernelRegistry());
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    return static_cast<Prov_KernelRegistry_Impl*>(&*outer_->Prov_GetKernelRegistry())->p_owned_;
  }

  std::vector<std::unique_ptr<Prov_ComputeCapability>> Prov_GetCapability(const onnxruntime::Prov_GraphViewer& graph,
                                                                          const std::vector<const Prov_KernelRegistry*>& kernel_registries) const override {
    std::vector<const KernelRegistry*> kernel_registries_internal;
    for (auto& v : kernel_registries)
      kernel_registries_internal.emplace_back(static_cast<const Prov_KernelRegistry_Impl*>(v)->p_);

    auto capabilities_internal = IExecutionProvider::GetCapability(static_cast<const Prov_GraphViewer_Impl*>(&graph)->v_, kernel_registries_internal);

    std::vector<std::unique_ptr<Prov_ComputeCapability>> capabilities;
    for (auto& v : capabilities_internal)
      capabilities.emplace_back(onnxruntime::make_unique<Prov_ComputeCapability>(onnxruntime::make_unique<Prov_IndexedSubGraph_Impl>(std::move(v->sub_graph))));
    return capabilities;
  }

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const onnxruntime::GraphViewer& graph,
                                                                const std::vector<const KernelRegistry*>& kernel_registries) const override {
    std::vector<const Prov_KernelRegistry*> registries;
    for (auto p : kernel_registries)
      registries.push_back(new Prov_KernelRegistry_Impl(const_cast<KernelRegistry*>(p)));

    auto prov_result = outer_->Prov_GetCapability(Prov_GraphViewer_Impl(graph), registries);
    std::vector<std::unique_ptr<ComputeCapability>> result;

    for (auto& p : prov_result)
      result.emplace_back(onnxruntime::make_unique<ComputeCapability>(std::move(static_cast<Prov_IndexedSubGraph_Impl*>(p->t_sub_graph_.get())->p_)));

    for (auto p : registries)
      delete p;

    return result;
  }

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) override {
    std::vector<Prov_Node_Impl> prov_fused_nodes_values;
    std::vector<Prov_Node*> prov_fused_nodes;
    prov_fused_nodes_values.reserve(fused_nodes.size());
    for (auto& p : fused_nodes) {
      prov_fused_nodes_values.emplace_back(p);
      prov_fused_nodes.emplace_back(&prov_fused_nodes_values.back());
    }

    return outer_->Prov_Compile(prov_fused_nodes, node_compute_funcs);
  }

  Prov_AllocatorPtr Prov_GetAllocator(int id, OrtMemType mem_type) const override {
    return std::make_shared<Prov_IAllocator_Impl>(IExecutionProvider::GetAllocator(id, mem_type));
  }

  void Prov_InsertAllocator(Prov_AllocatorPtr allocator) override {
    IExecutionProvider::InsertAllocator(static_cast<Prov_IAllocator_Impl*>(allocator.get())->p_);
  }

  std::unique_ptr<Prov_IExecutionProvider> outer_;
};

struct ProviderHostImpl : ProviderHost {
  ProviderHostImpl() {
    DataTypeImpl_GetType_Tensor = &DataTypeImpl::GetType<Tensor>;
    DataTypeImpl_GetType_float = &DataTypeImpl::GetType<float>;
    DataTypeImpl_GetTensorType_float = &DataTypeImpl::GetTensorType<float>;
  }

  std::unique_ptr<ONNX_NAMESPACE::Prov_AttributeProto> AttributeProto_Create() override {
    return onnxruntime::make_unique<Prov_AttributeProto_Impl>();
  }

  std::unique_ptr<Prov_OrtMemoryInfo> OrtMemoryInfo_Create(const char* name_, OrtAllocatorType type_, Prov_OrtDevice* device_, int id_, OrtMemType mem_type_) override {
    return onnxruntime::make_unique<Prov_OrtMemoryInfo_Impl>(name_, type_, device_ ? static_cast<Prov_OrtDevice_Impl*>(device_)->v_ : OrtDevice(), id_, mem_type_);
  }

  std::unique_ptr<Prov_KernelDefBuilder> KernelDefBuilder_Create() override {
    return onnxruntime::make_unique<Prov_KernelDefBuilder_Impl>();
  }

  std::shared_ptr<Prov_KernelRegistry> KernelRegistry_Create() override {
    return std::make_shared<Prov_KernelRegistry_Impl>();
  }

  std::unique_ptr<Prov_IndexedSubGraph> IndexedSubGraph_Create() override {
    return onnxruntime::make_unique<Prov_IndexedSubGraph_Impl>();
  }

  Prov_AllocatorPtr CreateAllocator(Prov_DeviceAllocatorRegistrationInfo& info, OrtDevice::DeviceId device_id = 0) override {
    DeviceAllocatorRegistrationInfo info_real{
        info.mem_type, [&info](int value) { return std::move(static_cast<Prov_IDeviceAllocator_Impl*>(&*info.factory(value))->p_); },
        info.max_mem};

    return std::make_shared<Prov_IAllocator_Impl>(onnxruntime::CreateAllocator(info_real, device_id));
  }

  std::unique_ptr<Prov_IDeviceAllocator>
  CreateCPUAllocator(std::unique_ptr<Prov_OrtMemoryInfo> memory_info) override {
    return onnxruntime::make_unique<Prov_IDeviceAllocator_Impl>(onnxruntime::make_unique<CPUAllocator>(std::move(static_cast<Prov_OrtMemoryInfo_Impl*>(memory_info.get())->info_)));
  };

  std::unique_ptr<Prov_IExecutionProvider_Router> Create_IExecutionProvider_Router(Prov_IExecutionProvider* outer, const std::string& type) override {
    return onnxruntime::make_unique<Prov_IExecutionProvider_Router_Impl>(outer, type);
  };

  void SessionOptions_AddProviderFactory(OrtSessionOptions& /*options*/, std::shared_ptr<Prov_IExecutionProviderFactory> /*provider*/) override {
    // options.provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Dnnl(use_arena));
    PROVIDER_NOT_IMPLEMENTED
  }

  logging::Logger* LoggingManager_GetDefaultLogger() override {
    return const_cast<logging::Logger*>(&logging::LoggingManager::DefaultLogger());
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

}  // namespace onnxruntime
provider_host_;

struct IExecutionProviderFactory_Translator : IExecutionProviderFactory {
  IExecutionProviderFactory_Translator(std::shared_ptr<Prov_IExecutionProviderFactory> p, void* handle) : p_{p}, handle_{handle} {}
  ~IExecutionProviderFactory_Translator() override {
    p_ = nullptr;  // Release before we unload
    Env::Default().UnloadDynamicLibrary(handle_);
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    auto provider = p_->CreateProvider();
    return std::unique_ptr<IExecutionProvider>(static_cast<Prov_IExecutionProvider_Router_Impl*>(provider.release()->p_));
  }

  std::shared_ptr<Prov_IExecutionProviderFactory> p_;
  void* handle_;
};

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(int device_id) {
  void* handle;
  Env::Default().LoadDynamicLibrary("C:\\code\\github\\onnxrt\\build\\windows\\debug\\debug\\onnxruntime_providers_dnnl.dll", &handle);
  if (!handle)
    return nullptr;

  Provider* (*PGetProvider)();

  Env::Default().GetSymbolFromLibrary(handle, "GetProvider", (void**)&PGetProvider);
  //return std::make_shared<onnxruntime::MkldnnProviderFactory>(device_id);
  //TODO: This is apparently a bug. The constructor parameter is create-arena-flag, not the device-id

  Provider* provider = PGetProvider();
  provider->SetProviderHost(provider_host_);

  return std::make_shared<IExecutionProviderFactory_Translator>(provider->CreateExecutionProviderFactory(device_id), handle);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Dnnl, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Dnnl(use_arena));
  return nullptr;
}
