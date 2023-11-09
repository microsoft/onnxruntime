// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "interface/framework/kernel.h"
#include "interface/provider/provider.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/framework_provider_common.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/allocator_adapters.h"
#include "core/session/custom_ops.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph_view_api_impl.h"
#include <memory>

namespace onnxruntime {
class ExternalDataTransfer : public IDataTransfer {
 public:
  ExternalDataTransfer(interface::ExecutionProvider* external_ep) : external_ep_impl_(external_ep) {}
  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override {
    return external_ep_impl_->CanCopy(src_device, dst_device);
  }

  common::Status CopyTensor(const Tensor& /*src*/, Tensor& /*dst*/) const override {
    //OrtValue src_value, dst_value;
    //const void* src_raw = src.DataRaw();
    //Tensor::InitOrtValue(src.DataType(), src.Shape(), const_cast<void*>(src_raw), src.Location(), src_value, src.ByteOffset());
    //Tensor::InitOrtValue(dst.DataType(), dst.Shape(), dst.MutableDataRaw(), dst.Location(), dst_value, dst.ByteOffset());

    //Ort::ConstValue src_cv{&src_value};
    //Ort::UnownedValue dst_uv{&dst_value};
    ////external_ep_impl_->MemoryCpy(dst_uv, src_cv);
    //return Status::OK();
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED);
  }

 private:
  interface::ExecutionProvider* external_ep_impl_;
};

//////////////////////////////////////////////// Kernel Adapters ////////////////////////////////////////////////

struct KernelInfoAdapter : public interface::IKernelInfo {
  KernelInfoAdapter(const OpKernelInfo&) {}
};

struct KernelBuilderAdapter : public interface::IKernelBuilder {
  IKernelBuilder& Provider(const char* provider) override {
    builder_.Provider(provider);
    return *this;
  }
  IKernelBuilder& SetDomain(const char* domain) override {
    builder_.SetDomain(domain);
    return *this;
  }
  IKernelBuilder& SetName(const char* op_name) override {
    builder_.SetName(op_name);
    return *this;
  }
  IKernelBuilder& SinceVersion(int since_ver, int end_ver) override {
    builder_.SinceVersion(since_ver, end_ver);
    return *this;
  }
  IKernelBuilder& Alias(int input_indice, int output_indice) override {
    builder_.Alias(input_indice, output_indice);
    return *this;
  }
  IKernelBuilder& TypeConstraint(const char* name, interface::TensorDataType type) override {
    switch (type) {
      case interface::TensorDataType::float_tp:
        builder_.TypeConstraint(name, DataTypeImpl::GetDataType("float"));
        break;
      case interface::TensorDataType::double_tp:
        builder_.TypeConstraint(name, DataTypeImpl::GetDataType("double"));
        break;
      case interface::TensorDataType::int8_tp:
        builder_.TypeConstraint(name, DataTypeImpl::GetDataType("int8"));
        break;
      case interface::TensorDataType::uint8_tp:
        builder_.TypeConstraint(name, DataTypeImpl::GetDataType("uint8"));
        break;
      case interface::TensorDataType::int16_tp:
        builder_.TypeConstraint(name, DataTypeImpl::GetDataType("int16"));
        break;
      case interface::TensorDataType::uint16_tp:
        builder_.TypeConstraint(name, DataTypeImpl::GetDataType("uint16"));
        break;
      case interface::TensorDataType::int32_tp:
        builder_.TypeConstraint(name, DataTypeImpl::GetDataType("int32"));
        break;
      case interface::TensorDataType::uint32_tp:
        builder_.TypeConstraint(name, DataTypeImpl::GetDataType("uint32"));
        break;
      case interface::TensorDataType::int64_tp:
        builder_.TypeConstraint(name, DataTypeImpl::GetDataType("int64"));
        break;
      case interface::TensorDataType::uint64_tp:
        builder_.TypeConstraint(name, DataTypeImpl::GetDataType("uint64"));
        break;
      case interface::TensorDataType::bool_tp:
        builder_.TypeConstraint(name, DataTypeImpl::GetDataType("bool"));
        break;
      default:
        break;
    }
    return *this;
  }
  KernelDefBuilder builder_;
};

#define INT(st) (static_cast<int>(st))

struct KernelContextAdapter : public interface::IKernelContext {
  KernelContextAdapter(OpKernelContext& context) : context_(context) {
    input_count_ = context_.InputCount();
    output_count_ = context_.OutputCount();
  }
  const void* InputData(int index) const override {
    ORT_ENFORCE(INT(index) < input_count_);
    const auto* tensor = context_.Input<onnxruntime::Tensor>(INT(index));
    ORT_ENFORCE(tensor);
    return tensor->DataRaw();
  };
  const int64_t* InputShape(int index, size_t* num_dims) const override {
    ORT_ENFORCE(INT(index) < input_count_);
    const auto* tensor = context_.Input<onnxruntime::Tensor>(INT(index));
    ORT_ENFORCE(tensor);
    const auto dims = tensor->Shape().GetDims();
    *num_dims = dims.size();
    return dims.data();
  };
  void* AllocateOutput(int index, const interface::TensorShape& shape) override {
    ORT_ENFORCE(INT(index) < output_count_);
    auto* tensor = context_.Output(INT(index), shape);
    ORT_ENFORCE(tensor);
    return tensor->MutableDataRaw();
  };
  OpKernelContext& context_;
  int input_count_ = 0;
  int output_count_ = 0;
};

struct OpKernelAdapter : public OpKernel {
  OpKernelAdapter(const OpKernelInfo& info,
                  std::unique_ptr<interface::IKernel> kernel) : OpKernel(info), kernel_(std::move(kernel)) {}
  Status Compute(_Inout_ OpKernelContext* context) const override {
    KernelContextAdapter context_adapter(*context);
    return kernel_->Compute(&context_adapter);
  };
  std::unique_ptr<interface::IKernel> kernel_;
};

struct KernelRegistryAdapter : public interface::IKernelRegistry {
  KernelRegistryAdapter() {
    registry_ = std::make_shared<KernelRegistry>();
  }
  interface::IKernelBuilder& CreateBuilder() override {
    builders_.push_back(std::make_unique<KernelBuilderAdapter>());
    return *builders_.back();
  }
  void BuildKernels() {
    for (auto& b : builders_) {
      KernelBuilderAdapter* builder = b.get();
      KernelCreateInfo kci(builder->builder_.Build(),
                           [=](FuncManager&,
                               const OpKernelInfo& info,
                               std::unique_ptr<OpKernel>& out) -> onnxruntime::common::Status {
                             KernelInfoAdapter info_adapter(info);
                             out = std::make_unique<OpKernelAdapter>(info, builder->create_kernel_fn_(info_adapter));
                             return onnxruntime::common::Status::OK();
                           });
      ORT_ENFORCE(registry_->Register(std::move(kci)).IsOK());
    }
  }
  std::shared_ptr<KernelRegistry> registry_;
  using BuilderPtr = std::unique_ptr<KernelBuilderAdapter>;
  std::vector<BuilderPtr> builders_;
};

struct AllocatorAdapter : public OrtAllocator {
  AllocatorAdapter(interface::Allocator* impl) : impl_(impl),
                                                 mem_info_("",
                                                           OrtDeviceAllocator,
                                                           OrtDevice(static_cast<OrtDevice::DeviceType>(impl->dev_type),
                                                                     OrtDevice::MemType::DEFAULT, 0)) {
    version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](struct OrtAllocator* this_, size_t size) -> void* {
      auto self = reinterpret_cast<AllocatorAdapter*>(this_);
      return self->impl_->Alloc(size);
    };
    OrtAllocator::Free = [](struct OrtAllocator* this_, void* p) {
      auto self = reinterpret_cast<AllocatorAdapter*>(this_);
      return self->impl_->Free(p);
    };
    OrtAllocator::Info = [](const struct OrtAllocator* this_) {
      auto self = reinterpret_cast<const AllocatorAdapter*>(this_);
      return &self->mem_info_;
    };
  }

 private:
  interface::Allocator* impl_;
  struct OrtMemoryInfo mem_info_;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ExecutionProviderAdapter : public IExecutionProvider {
 public:
  ExecutionProviderAdapter(interface::ExecutionProvider* external_ep)
      : IExecutionProvider(external_ep->GetType(), external_ep->GetDevice()), external_ep_impl_(external_ep) {
    external_ep_impl_->RegisterKernels(kernel_registry_);
    kernel_registry_.BuildKernels();
    for (auto& allocator : external_ep_impl_->GetAllocators()) {
      allocators_.push_back(std::make_unique<AllocatorAdapter>(allocator.get()));
    }
  }

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    return kernel_registry_.registry_;
  }

  std::vector<AllocatorPtr> CreatePreferredAllocators() override {
    std::vector<AllocatorPtr> ret;
    std::for_each(allocators_.begin(), allocators_.end(), [&](std::unique_ptr<AllocatorAdapter>& impl) {
      ret.push_back(std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(impl.get()));
    });
    return ret;
  }

  std::unique_ptr<IDataTransfer> GetDataTransfer() const override { return std::make_unique<ExternalDataTransfer>(external_ep_impl_.get()); }

  // TODO: 1. should we reserve allocators? 2. when to release OrtAllocator*?
  virtual void RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap& allocators) const override {
    std::map<OrtDevice, OrtAllocator*> ort_allocators;
    for (auto& [device, allocator] : allocators) {
      std::shared_ptr<IAllocator> iallocator(allocator);
      std::unique_ptr<OrtAllocatorImplWrappingIAllocator> ort_allocator = std::make_unique<OrtAllocatorImplWrappingIAllocator>(std::move(iallocator));
      ort_allocators.insert({device, ort_allocator.release()});
    }
    external_ep_impl_->RegisterStreamHandlers(stream_handle_registry, ort_allocators);
  }

  virtual std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const GraphViewer& graph_viewer, const IKernelLookup& kernel_lookup) const override {
    return IExecutionProvider::GetCapability(graph_viewer, kernel_lookup);
  }

  /*
  * disable logic below to skip stack segfault returning vec object
  virtual std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const GraphViewer& graph_viewer, const IKernelLookup& kernel_lookup) const override {
  AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  ApiGraphView api_graph_view(graph_viewer.GetGraph(), std::move(cpu_allocator), graph_viewer.GetFilterInfo());
  std::vector<std::unique_ptr<SubGraphDef>> sub_graphs = external_ep_impl_->GetCapability(&api_graph_view);
  if (sub_graphs.size() == 0) return IExecutionProvider::GetCapability(graph_viewer, kernel_lookup);

  std::vector<std::unique_ptr<ComputeCapability>> ret;
  for (auto& sub_graph : sub_graphs) {
    std::unique_ptr<IndexedSubGraph> sb = std::make_unique<IndexedSubGraph>();
    sb->nodes = sub_graph->nodes;
    auto meta_def = std::make_unique<onnxruntime::IndexedSubGraph::MetaDef>();
    meta_def->name = sub_graph->GetMetaDef()->name;
    meta_def->doc_string = sub_graph->GetMetaDef()->doc_string;
    meta_def->domain = sub_graph->GetMetaDef()->domain;
    meta_def->since_version = sub_graph->GetMetaDef()->since_version;
    meta_def->inputs = sub_graph->GetMetaDef()->inputs;
    meta_def->outputs = sub_graph->GetMetaDef()->outputs;
    meta_def->constant_initializers = sub_graph->GetMetaDef()->constant_initializers;
    sb->SetMetaDef(std::move(meta_def));

    ret.push_back(std::make_unique<ComputeCapability>(std::move(sb)));
  }
  return ret;
}*/

  virtual common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs, std::vector<NodeComputeInfo>& node_compute_funcs) override {
    std::vector<std::unique_ptr<interface::GraphViewRef>> fused_node_as_graph;
    std::vector<std::unique_ptr<interface::NodeViewRef>> fused_node_view;
    for (auto& fused_node_graph : fused_nodes_and_graphs) {
      const GraphViewer& graph_viewer = fused_node_graph.filtered_graph;
      const Node& fused_node = fused_node_graph.fused_node;

      AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
      fused_node_as_graph.push_back(std::make_unique<ApiGraphView>(graph_viewer.GetGraph(), std::move(cpu_allocator), graph_viewer.GetFilterInfo()));
      fused_node_view.push_back(std::make_unique<ApiNodeView>(fused_node));
    }
    common::Status ret = external_ep_impl_->Compile(fused_node_as_graph, fused_node_view, node_compute_funcs);
    return ret;
  }

 private:
  std::vector<std::unique_ptr<AllocatorAdapter>> allocators_;
  std::unique_ptr<interface::ExecutionProvider> external_ep_impl_;
  KernelRegistryAdapter kernel_registry_;
};
}  // namespace onnxruntime
