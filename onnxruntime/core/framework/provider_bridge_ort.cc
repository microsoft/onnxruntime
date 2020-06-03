// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the Onnxruntime side of the bridge to allow providers to be built as a DLL
// It implements onnxruntime::ProviderHost

#include "core/framework/data_types.h"
#include "core/framework/allocatormgr.h"
#include "core/providers/dnnl/dnnl_provider_factory.h"
#include "core/session/inference_session.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"
#include "core/platform/env.h"
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

  bool AllowsArena() const override { return p_->AllowsArena(); }

  std::unique_ptr<IDeviceAllocator> p_;
};

struct Provider_TensorProto_Impl : ONNX_NAMESPACE::Provider_TensorProto {
  Provider_TensorProto_Impl(ONNX_NAMESPACE::TensorProto* p) : p_{p} {}

  void CopyFrom(const Provider_TensorProto& v) override {
    *p_ = *static_cast<const Provider_TensorProto_Impl*>(&v)->p_;
  }

  ONNX_NAMESPACE::TensorProto* p_{};
};

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
  ::onnx::Provider_TensorProto* add_tensors() override {
    // Kind of a hack, but the pointer is only valid until the next add_tensors call
    tensors_ = onnxruntime::make_unique<Provider_TensorProto_Impl>(v_.add_tensors());
    return tensors_.get();
  }

  ONNX_NAMESPACE::AttributeProto v_;
  std::unique_ptr<Provider_TensorProto_Impl> tensors_;
};

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
  Provider_KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType supported_type) override {
    v_.TypeConstraint(arg_name, supported_type);
    return *this;
  }

  std::unique_ptr<Provider_KernelDef> Build() override {
    return onnxruntime::make_unique<Provider_KernelDef_Impl>(v_.Build());
  }

  KernelDefBuilder v_;
};

struct Provider_NodeArg_Impl : Provider_NodeArg {
  Provider_NodeArg_Impl(const NodeArg* p) : p_{p} {
    if (p_->Shape())
      tensor_shape_proto_.dim_size_ = p_->Shape()->dim_size();
  }

  const std::string& Name() const noexcept override { return p_->Name(); }
  const ONNX_NAMESPACE::Provider_TensorShapeProto* Shape() const override { return &tensor_shape_proto_; }
  virtual ONNX_NAMESPACE::DataType Type() const noexcept override { return p_->Type(); }

  const NodeArg* p_;
  ONNX_NAMESPACE::Provider_TensorShapeProto tensor_shape_proto_;
};

struct Provider_Node_Impl : Provider_Node {
  Provider_Node_Impl(const Node* p) : p_{p} {}
  ~Provider_Node_Impl() override {
    for (auto p : input_defs_)
      delete p;
    for (auto p : output_defs_)
      delete p;
  }

  const std::string& OpType() const noexcept override { return p_->OpType(); }
  //  const ONNX_NAMESPACE::OpSchema* Op() const noexcept

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

  NodeIndex Index() const noexcept override { return p_->Index(); }

  const Provider_NodeAttributes& GetAttributes() const noexcept override {
    if (attributes_.empty()) {
      for (auto& v : p_->GetAttributes())
        attributes_[v.first] = onnxruntime::make_unique<Provider_AttributeProto_Impl>(v.second);
    }
    return attributes_;
  }

  size_t GetInputEdgesCount() const noexcept override {
    return p_->GetInputEdgesCount();
  }
  size_t GetOutputEdgesCount() const noexcept override { return p_->GetOutputEdgesCount(); }

  std::unique_ptr<Provider_NodeIterator> InputNodesBegin_internal() const noexcept override;
  std::unique_ptr<Provider_NodeIterator> InputNodesEnd_internal() const noexcept override;

  const Node* p_;
  mutable std::vector<Provider_NodeArg*> input_defs_;
  mutable std::vector<Provider_NodeArg*> output_defs_;
  mutable Provider_NodeAttributes attributes_;
};

struct Provider_NodeIterator_Impl : Provider_Node::Provider_NodeIterator {
  Provider_NodeIterator_Impl(Node::NodeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Provider_NodeIterator& p) const override { return v_ != static_cast<const Provider_NodeIterator_Impl*>(&p)->v_; }

  void operator++() override { return v_.operator++(); }
  const Provider_Node& operator*() override {
    node_ = Provider_Node_Impl(&*v_);
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

  std::vector<onnxruntime::NodeIndex>& Nodes() override { return p_->nodes; }

  std::unique_ptr<IndexedSubGraph> p_{onnxruntime::make_unique<IndexedSubGraph>()};
};

struct Provider_GraphViewer_Impl : Provider_GraphViewer {
  Provider_GraphViewer_Impl(const GraphViewer& v) : v_(v) {
    for (int i = 0; i < v_.MaxNodeIndex(); i++)
      provider_nodes_.emplace_back(v_.GetNode(i));
  }

  const std::string& Name() const noexcept override { return v_.Name(); }

  const Provider_Node* GetNode(NodeIndex node_index) const override {
    auto& node = provider_nodes_[node_index];
    if (node.p_)
      return &node;
    return nullptr;
  }

  int MaxNodeIndex() const noexcept override { return v_.MaxNodeIndex(); }

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

  const GraphViewer& v_;

  std::vector<Provider_Node_Impl> provider_nodes_;

  mutable std::vector<Provider_TensorProto_Impl> initialized_tensors_;
  mutable Provider_InitializedTensorSet initialized_tensor_set_;
};

struct Provider_OpKernelInfo_Impl : Provider_OpKernelInfo {
  Provider_OpKernelInfo_Impl(const OpKernelInfo& info) : info_(info) {}

  Status GetAttr(const std::string& name, int64_t* value) const override {
    return info_.GetAttr<int64_t>(name, value);
  }

  Status GetAttr(const std::string& name, float* value) const override {
    return info_.GetAttr<float>(name, value);
  }

  const OpKernelInfo& info_;
};

struct Provider_Tensor_Impl final : Provider_Tensor {
  Provider_Tensor_Impl(const Tensor* p) : p_(const_cast<Tensor*>(p)) {}

  float* MutableData_float() override { return p_->MutableData<float>(); }
  const float* Data_float() const override { return p_->Data<float>(); }

  const TensorShape& Shape() const override { return p_->Shape(); }

  Tensor* p_;
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

struct Provider_KernelRegistry_Impl : Provider_KernelRegistry {
  Provider_KernelRegistry_Impl(std::shared_ptr<KernelRegistry> p) : p_owned_(p) {}
  Provider_KernelRegistry_Impl(KernelRegistry* p) : p_(p) {}
  Provider_KernelRegistry_Impl() : p_owned_(std::make_shared<KernelRegistry>()) {}

  Status Register(Provider_KernelCreateInfo&& create_info) override {
    KernelCreateInfo info_real(std::move(static_cast<Provider_KernelDef_Impl*>(create_info.kernel_def.get())->p_),
                               [kernel_create_func = create_info.kernel_create_func](const OpKernelInfo& info) -> OpKernel* {
                                 Provider_OpKernelInfo_Impl provider_info(info);
                                 return new OpKernel_Translator(provider_info, kernel_create_func(provider_info));
                               });

    return p_->Register(std::move(info_real));
  }

  std::shared_ptr<KernelRegistry> p_owned_;
  KernelRegistry* p_{&*p_owned_};
};

struct Provider_IExecutionProvider_Router_Impl : Provider_IExecutionProvider_Router, IExecutionProvider {
  Provider_IExecutionProvider_Router_Impl(Provider_IExecutionProvider* outer, const std::string& type) : IExecutionProvider(type), outer_(outer) {
  }

  virtual ~Provider_IExecutionProvider_Router_Impl() {}

  std::shared_ptr<Provider_KernelRegistry> Provider_GetKernelRegistry() const override {
    return std::make_shared<Provider_KernelRegistry_Impl>(GetKernelRegistry());
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    return static_cast<Provider_KernelRegistry_Impl*>(&*outer_->Provider_GetKernelRegistry())->p_owned_;
  }

  std::vector<std::unique_ptr<Provider_ComputeCapability>> Provider_GetCapability(const onnxruntime::Provider_GraphViewer& graph,
                                                                                  const std::vector<const Provider_KernelRegistry*>& kernel_registries) const override {
    std::vector<const KernelRegistry*> kernel_registries_internal;
    for (auto& v : kernel_registries)
      kernel_registries_internal.emplace_back(static_cast<const Provider_KernelRegistry_Impl*>(v)->p_);

    auto capabilities_internal = IExecutionProvider::GetCapability(static_cast<const Provider_GraphViewer_Impl*>(&graph)->v_, kernel_registries_internal);

    std::vector<std::unique_ptr<Provider_ComputeCapability>> capabilities;
    for (auto& v : capabilities_internal)
      capabilities.emplace_back(onnxruntime::make_unique<Provider_ComputeCapability>(onnxruntime::make_unique<Provider_IndexedSubGraph_Impl>(std::move(v->sub_graph))));
    return capabilities;
  }

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const onnxruntime::GraphViewer& graph,
                                                                const std::vector<const KernelRegistry*>& kernel_registries) const override {
    std::vector<const Provider_KernelRegistry*> registries;
    for (auto p : kernel_registries)
      registries.push_back(new Provider_KernelRegistry_Impl(const_cast<KernelRegistry*>(p)));

    auto provider_result = outer_->Provider_GetCapability(Provider_GraphViewer_Impl(graph), registries);
    std::vector<std::unique_ptr<ComputeCapability>> result;

    for (auto& p : provider_result)
      result.emplace_back(onnxruntime::make_unique<ComputeCapability>(std::move(static_cast<Provider_IndexedSubGraph_Impl*>(p->t_sub_graph_.get())->p_)));

    for (auto p : registries)
      delete p;

    return result;
  }

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes, std::vector<NodeComputeInfo>& node_compute_funcs) override {
    std::vector<Provider_Node_Impl> provider_fused_nodes_values;
    std::vector<Provider_Node*> provider_fused_nodes;
    provider_fused_nodes_values.reserve(fused_nodes.size());
    for (auto& p : fused_nodes) {
      provider_fused_nodes_values.emplace_back(p);
      provider_fused_nodes.emplace_back(&provider_fused_nodes_values.back());
    }

    return outer_->Provider_Compile(provider_fused_nodes, node_compute_funcs);
  }

  Provider_AllocatorPtr Provider_GetAllocator(int id, OrtMemType mem_type) const override {
    return std::make_shared<Provider_IAllocator_Impl>(IExecutionProvider::GetAllocator(id, mem_type));
  }

  void Provider_InsertAllocator(Provider_AllocatorPtr allocator) override {
    IExecutionProvider::InsertAllocator(static_cast<Provider_IAllocator_Impl*>(allocator.get())->p_);
  }

  std::unique_ptr<Provider_IExecutionProvider> outer_;
};

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

  std::unique_ptr<Provider_KernelDefBuilder> KernelDefBuilder_Create() override {
    return onnxruntime::make_unique<Provider_KernelDefBuilder_Impl>();
  }

  std::shared_ptr<Provider_KernelRegistry> KernelRegistry_Create() override {
    return std::make_shared<Provider_KernelRegistry_Impl>();
  }

  std::unique_ptr<Provider_IndexedSubGraph> IndexedSubGraph_Create() override {
    return onnxruntime::make_unique<Provider_IndexedSubGraph_Impl>();
  }

  Provider_AllocatorPtr CreateAllocator(Provider_DeviceAllocatorRegistrationInfo& info, OrtDevice::DeviceId device_id = 0) override {
    DeviceAllocatorRegistrationInfo info_real{
        info.mem_type, [&info](int value) { return std::move(static_cast<Provider_IDeviceAllocator_Impl*>(&*info.factory(value))->p_); },
        info.max_mem};

    return std::make_shared<Provider_IAllocator_Impl>(onnxruntime::CreateAllocator(info_real, device_id));
  }

  std::unique_ptr<Provider_IDeviceAllocator>
  CreateCPUAllocator(std::unique_ptr<Provider_OrtMemoryInfo> memory_info) override {
    return onnxruntime::make_unique<Provider_IDeviceAllocator_Impl>(onnxruntime::make_unique<CPUAllocator>(std::move(static_cast<Provider_OrtMemoryInfo_Impl*>(memory_info.get())->info_)));
  };

  Provider_AllocatorPtr
  CreateDummyArenaAllocator(std::unique_ptr<Provider_IDeviceAllocator> resource_allocator) override {
    return std::make_shared<Provider_IAllocator_Impl>(onnxruntime::make_unique<DummyArena>(std::move(static_cast<Provider_IDeviceAllocator_Impl*>(resource_allocator.get())->p_)));
  };

  std::unique_ptr<Provider_IExecutionProvider_Router> Create_IExecutionProvider_Router(Provider_IExecutionProvider* outer, const std::string& type) override {
    return onnxruntime::make_unique<Provider_IExecutionProvider_Router_Impl>(outer, type);
  };

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

}  // namespace onnxruntime

// TODO: Right now Dnnl is the only provider in here, but this will be made more generic and support more providers in the future
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Dnnl, _In_ OrtSessionOptions* options, int use_arena) {
  auto factory = onnxruntime::CreateExecutionProviderFactory_Dnnl(use_arena);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Dnnl: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
}
