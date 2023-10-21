#pragma once
#include "core/framework/custom_execution_provider.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/framework_provider_common.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/allocator_adapters.h"
#include "core/session/custom_ops.h"
#include "core/session/onnxruntime_cxx_api.h"
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

    //////////////////////////////////////////////// Lite Interfaces ////////////////////////////////////////////////

    struct KernelBuilderLite : public lite::IKernelBuilder {
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
        IKernelBuilder& TypeConstraint(const char* name, lite::DataType type) override {
            switch (type) {
              case lite::DataType::float_tp:
                builder_.TypeConstraint(name, DataTypeImpl::GetDataType("float"));
                break;
              case lite::DataType::double_tp:
                builder_.TypeConstraint(name, DataTypeImpl::GetDataType("double"));
                break;
              case lite::DataType::int8_tp:
                builder_.TypeConstraint(name, DataTypeImpl::GetDataType("int8"));
                break;
              case lite::DataType::uint8_tp:
                builder_.TypeConstraint(name, DataTypeImpl::GetDataType("uint8"));
                break;
              case lite::DataType::int16_tp:
                builder_.TypeConstraint(name, DataTypeImpl::GetDataType("int16"));
                break;
              case lite::DataType::uint16_tp:
                builder_.TypeConstraint(name, DataTypeImpl::GetDataType("uint16"));
                break;
              case lite::DataType::int32_tp:
                builder_.TypeConstraint(name, DataTypeImpl::GetDataType("int32"));
                break;
              case lite::DataType::uint32_tp:
                builder_.TypeConstraint(name, DataTypeImpl::GetDataType("uint32"));
                break;
              case lite::DataType::int64_tp:
                builder_.TypeConstraint(name, DataTypeImpl::GetDataType("int64"));
                break;
              case lite::DataType::uint64_tp:
                builder_.TypeConstraint(name, DataTypeImpl::GetDataType("uint64"));
                break;
              case lite::DataType::bool_tp:
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

    struct KernelContextLite : public lite::IKernelContext {
        KernelContextLite(OpKernelContext& context): context_(context) {
            input_count_ = context_.InputCount();
            output_count_ = context_.OutputCount();
        }
        const void* InputData(size_t index) const override {
            ORT_ENFORCE(INT(index) < input_count_);
            const auto* tensor = context_.Input<onnxruntime::Tensor>(INT(index));
            ORT_ENFORCE(tensor);
            return tensor->DataRaw();
        };
        const int64_t* InputShape(size_t index, size_t* num_dims) const override {
            ORT_ENFORCE(INT(index) < input_count_);
            const auto* tensor = context_.Input<onnxruntime::Tensor>(INT(index));
            ORT_ENFORCE(tensor);
            const auto dims = tensor->Shape().GetDims();
            *num_dims = dims.size();
            return dims.data();
        };
        void* AllocateOutput(size_t index, const lite::TensorShape& shape) override {
            ORT_ENFORCE(INT(index) < output_count_);
            auto* tensor = context_.Output(INT(index), shape);
            ORT_ENFORCE(tensor);
            return tensor->MutableDataRaw();
        };
        OpKernelContext& context_;
        int input_count_ = 0;
        int output_count_ = 0;
    };

    struct OpKernelLite : public OpKernel {
        OpKernelLite(const OpKernelInfo& info, lite::IKernel& lite_kernel) : OpKernel(info), lite_kernel_(lite_kernel) {}
        Status Compute(_Inout_ OpKernelContext* context) const override {
            KernelContextLite context_lite(*context);
            return lite_kernel_.Compute(&context_lite);
        };
        lite::IKernel& lite_kernel_;
    };

    struct KernelRegistryLite : public lite::IKernelRegistry {
        KernelRegistryLite() {
            registry_ = std::make_shared<KernelRegistry>();
        }
        lite::IKernelBuilder& CreateBuilder() override {
            builders_.emplace_back();
            return builders_.back();
        }
        void BuildKernels() {
            for (auto& builder : builders_) {
              KernelCreateInfo kci(builder.builder_.Build(),
                                   [&](FuncManager&,
                                       const OpKernelInfo& info,
                                       std::unique_ptr<OpKernel>& out) -> onnxruntime::common::Status {
                                     out = std::make_unique<OpKernelLite>(info, *builder.kernel_);
                                     return onnxruntime::common::Status::OK();
                                   });
              ORT_ENFORCE(registry_->Register(std::move(kci)).IsOK());
            }
        }
        std::shared_ptr<KernelRegistry> registry_;
        std::vector<KernelBuilderLite> builders_;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    class ExternalExecutionProvider : public IExecutionProvider {
       public:
        ExternalExecutionProvider(lite::IExecutionProvider* external_ep)
            : IExecutionProvider(external_ep->GetType(), external_ep->GetDevice()), external_ep_impl_(external_ep) {
            external_ep_impl_->RegisterKernels(kernel_registry_);
            kernel_registry_.BuildKernels();
        }

        virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
            return kernel_registry_.registry_;
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

    private:
        std::unique_ptr<lite::IExecutionProvider> external_ep_impl_;
        KernelRegistryLite kernel_registry_;

        //void OrtLiteCustomOp2KernelRegistry(Ort::Custom::ExternalKernelDef* kernel_definition) {
        //    KernelCreateInfo kernel_create_info = CreateKernelCreateInfo(kernel_definition->domain_, kernel_definition->custom_op_.get(), kernel_definition->op_since_version_start_, kernel_definition->op_since_version_end_);
        //    ORT_THROW_IF_ERROR(kernel_registry_->Register(std::move(kernel_create_info)));
        //}
    };
}
