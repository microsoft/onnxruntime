#pragma once
#include "core/framework/custom_execution_provider.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/framework_provider_common.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/allocator_adapters.h"
#include "core/session/custom_ops.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/compute_capability.h"
#include "core/optimizer/transpose_optimization/ort_optimizer_api_impl.h"
#include <memory>

namespace onnxruntime {
    class ExternalDataTransfer : public IDataTransfer {
    public:
        ExternalDataTransfer(CustomExecutionProvider* external_ep) : external_ep_impl_(external_ep) {}
        bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override {
            return external_ep_impl_->CanCopy(src_device, dst_device);
        }

        common::Status CopyTensor(const Tensor& src, Tensor& dst) const override {
            OrtValue src_value, dst_value;
            const void* src_raw = src.DataRaw();
            Tensor::InitOrtValue(src.DataType(), src.Shape(), const_cast<void*>(src_raw), src.Location(), src_value, src.ByteOffset());
            Tensor::InitOrtValue(dst.DataType(), dst.Shape(), dst.MutableDataRaw(), dst.Location(), dst_value, dst.ByteOffset());

            Ort::ConstValue src_cv{&src_value};
            Ort::UnownedValue dst_uv{&dst_value};
            external_ep_impl_->MemoryCpy(dst_uv, src_cv);
            return Status::OK();
        }

    private:
        CustomExecutionProvider* external_ep_impl_;
    };

    class ExternalExecutionProvider : public IExecutionProvider{
    public:
        ExternalExecutionProvider(CustomExecutionProvider* external_ep)
            : IExecutionProvider(external_ep->GetType(), external_ep->GetDevice()), external_ep_impl_(external_ep){
                kernel_registry_ = std::make_shared<KernelRegistry>();
                size_t kernelsCount = external_ep_impl_->GetKernelDefinitionCount();
                for (size_t i = 0; i < kernelsCount; i++) {
                    OrtLiteCustomOp2KernelRegistry(external_ep_impl_->GetKernelDefinition(i));
                }
            }

        virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
            return kernel_registry_;
        }

        std::vector<AllocatorPtr> CreatePreferredAllocators() override {
            std::vector<AllocatorPtr> ret;
            std::vector<OrtAllocator*> allocators = external_ep_impl_->GetAllocators();
            for (auto& allocator : allocators) {
                AllocatorPtr alloc_ptr = std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(allocator);
                ret.push_back(std::move(alloc_ptr));
            }
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
            AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
            ApiGraphView api_graph_view(graph_viewer, std::move(cpu_allocator));
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
            }
            return ret;
        }

        virtual common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs, std::vector<NodeComputeInfo>& node_compute_funcs) override {
            std::vector<std::unique_ptr<GraphViewRef>> fused_node_as_graph;
            std::vector<std::unique_ptr<NodeViewRef>> fused_node_view;
            for (auto& fused_node_graph : fused_nodes_and_graphs) {
                const GraphViewer& graph_viewer = fused_node_graph.filtered_graph;
                const Node& fused_node = fused_node_graph.fused_node;

                AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
                fused_node_as_graph.push_back(std::make_unique<ApiGraphView>(graph_viewer, std::move(cpu_allocator)));
                fused_node_view.push_back(std::make_unique<ApiNodeView>(fused_node));
            }
            common::Status ret = external_ep_impl_->Compile(fused_node_as_graph, fused_node_view, node_compute_funcs);
            return ret;
        }

    private:
        std::unique_ptr<CustomExecutionProvider> external_ep_impl_;
        std::shared_ptr<KernelRegistry> kernel_registry_;

        void OrtLiteCustomOp2KernelRegistry(Ort::Custom::ExternalKernelDef* kernel_definition) {
            KernelCreateInfo kernel_create_info = CreateKernelCreateInfo(kernel_definition->domain_, kernel_definition->custom_op_.get(), kernel_definition->op_since_version_start_, kernel_definition->op_since_version_end_);
            ORT_THROW_IF_ERROR(kernel_registry_->Register(std::move(kernel_create_info)));
        }
    };
}
