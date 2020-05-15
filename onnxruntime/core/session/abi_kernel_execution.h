#pragma once

#include <core/framework/op_kernel.h>
#include "core/graph/graph.h"

namespace onnxruntime {

    struct ExecutableKernelImpl {
        ExecutableKernelImpl(std::unique_ptr<Model> model, std::unique_ptr<OpKernel> op_kernel) : model(
                std::move(model)), op_kernel(std::move(op_kernel)) {}

        // the model who's MainGraph holds the single node for this op_kernel
        std::unique_ptr<Model> model;
        std::unique_ptr<OpKernel> op_kernel;
    };

    // An ExecutionFrame that only executes a single kernel
    class ExecutableKernelContextImpl final : public IExecutionFrame {
    public:
        class Info {
        public:
            Info(std::unique_ptr<OpKernel> const &kernel, const logging::Logger &logger)
                    : kernel_(kernel),
                      logger_(&logger) {
                ORT_ENFORCE(kernel, "kernel cannot be null");

                if (kernel->KernelDef().Provider() == kCpuExecutionProvider) {
                    provider_ = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
                    transfer_manager_ = std::make_unique<DataTransferManager>();
                    transfer_manager_->RegisterDataTransfer(std::make_unique<CPUDataTransfer>());

                    // as in OptimizerExectutionFrame
                    allocator_ = provider_->GetAllocator(0, OrtMemTypeDefault);
                } else {
                    throw NotImplementedException("Provider type is not supported");
                }
            }

            AllocatorPtr GetAllocator() const {
                return allocator_;
            }

            Status AddOutput(OrtValue &value, int index) {
                int mlvalue_idx = value_name_idx_map_.Add("output" + std::to_string(index));
                ort_value_idx_nodearg_map_[mlvalue_idx] = kernel_->Info().GetOutputType(index);

                // TODO check that OrtValue is not too big
                fetches_.push_back(value);

                // TODO shoudl this be num_args++? or mlvalue_idx? Double check that.
                fetches_mlvalue_idxs_.push_back(mlvalue_idx);
                return Status::OK();
            }

            Status AddInput(OrtValue &value, int index) {
                int mlvalue_idx = value_name_idx_map_.Add("input" + std::to_string(index));

                ort_value_idx_nodearg_map_[mlvalue_idx] = kernel_->Info().GetInputType(index);
                // TODO check that OrtValue is not too big
                feeds_.push_back(value);
                feed_mlvalue_idxs_.push_back(mlvalue_idx);
                return Status::OK();
            }

            Status AddImplicitInput(OrtValue &value, int index) {
                int mlvalue_idx = value_name_idx_map_.Add("imp_input" + std::to_string(index));

                // TODO, probably need to offset this?
                ort_value_idx_nodearg_map_[mlvalue_idx] = kernel_->Info().GetInputType(index);

                // TODO check that OrtValue is not too big
                feeds_.push_back(value);
                feed_mlvalue_idxs_.push_back(mlvalue_idx);
                return Status::OK();
            }


        protected:

            int num_args = 0;

            NodeIndexInfo &GetNodeIndexInfo() {

                std::cout << "All OrtValues:" << std::endl;

                for (auto const &value : value_name_idx_map_) {
                    std::cout << value.first << " : " << value.second << std::endl;
                }
                node_index_info_ = std::unique_ptr<NodeIndexInfo>(
                        new NodeIndexInfo({&kernel_->Node()}, value_name_idx_map_));

                return *node_index_info_;
            }

            std::unique_ptr<OpKernel> const &kernel_;
            const logging::Logger *const logger_;

            friend ExecutableKernelContextImpl;

            OrtValueNameIdxMap value_name_idx_map_;
            std::unordered_map<int, const ONNX_NAMESPACE::TypeProto *> ort_value_idx_nodearg_map_;

            std::unique_ptr<NodeIndexInfo> node_index_info_;

            std::vector<int> fetches_mlvalue_idxs_;
            std::vector<OrtValue> fetches_;
            std::vector<int> feed_mlvalue_idxs_;
            std::vector<OrtValue> feeds_;

            std::unique_ptr<IExecutionProvider> provider_;
            std::unique_ptr<DataTransferManager> transfer_manager_;
            AllocatorPtr allocator_;
        };


        ExecutableKernelContextImpl(Info &info) :
                IExecutionFrame(info.value_name_idx_map_, info.GetNodeIndexInfo(), info.fetches_mlvalue_idxs_),
                info_(info) {
            Init(info.feed_mlvalue_idxs_, info.feeds_, std::unordered_map<int, OrtValue>(), info.fetches_);
        };

        Status Compute() {
            OpKernelContext context(this, info_.kernel_.get(), nullptr, *info_.logger_);
            Status status = info_.kernel_->Compute(&context);
            return status;
        }


    protected:
        AllocatorPtr GetAllocatorImpl(const OrtMemoryInfo &info) const override {
            return info_.provider_->GetAllocator(info.id, info.mem_type);
        }

        Status CopyTensor(const Tensor &src, Tensor &dest) const override {
            // TODO throw error
            return info_.transfer_manager_->CopyTensor(src, dest);
        }


        Status CreateNodeOutputMLValueImpl(OrtValue &ort_value, int ort_value_idx, const TensorShape *shape,
                                           size_t nnz) override;

    private:
        const Info &info_;

        const int device_id_{0};
        const OrtMemType mem_type_{OrtMemTypeDefault};

    };

}  // namespace onnxruntime
