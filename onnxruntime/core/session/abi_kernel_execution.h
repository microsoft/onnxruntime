#pragma once

#include <core/framework/op_kernel.h>
#include "core/graph/graph.h"

namespace onnxruntime {

    struct KernelSessionImpl {
        KernelSessionImpl(std::unique_ptr<Model> model) : model(
                std::move(model)) {}

        // the model who's MainGraph holds the nodes for the kernels that we will execute
        std::unique_ptr<Model> model;
    };

    // An ExecutionFrame that only executes a single kernel
    class ExecutableKernelContextImpl final : public IExecutionFrame {
    public:

        // the struct OrtExecutableKernelContext actually points to this class
        class Info {
        public:
            Info(std::unique_ptr<OpKernel> kernel, const logging::Logger &logger)
                    : kernel_(std::move(kernel)),
                      logger_(&logger) {
                ORT_ENFORCE(kernel_, "kernel cannot be null");

                if (kernel_->KernelDef().Provider() == kCpuExecutionProvider) {
                    provider_ = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

                    // as in OptimizerExectutionFrame
                    allocator_ = provider_->GetAllocator(0, OrtMemTypeDefault);
                } else {
                    throw NotImplementedException(
                            "Provider type (" + kernel_->KernelDef().Provider() + ") is not supported");
                }

                auto &node = kernel_->Node();

                input_index_to_mlvalue_map_ = std::vector<int> (node.InputDefs().size(), -1);
                output_index_to_mlvalue_map_ = std::vector<int> (node.OutputDefs().size(), -1);

                if (node.ImplicitInputDefs().size()) {
                    // not sure how to handle this correctly
                    throw new NotImplementedException("Implicit inputs are not supporterted");
                }

                // initialize inputs and outputs with null values
                OrtValue null_value;
                node.ForEachWithIndex(node.InputDefs(),
                                      [this](const NodeArg &arg, size_t index) {
                                          this->AddInput(OrtValue(), index, arg.Name());
                                          return Status::OK();
                                      });

                node.ForEachWithIndex(node.OutputDefs(),
                                      [this](const NodeArg &arg, size_t index) {
                                          this->AddOutput(OrtValue(), index, arg.Name());
                                          return Status::OK();
                                      });
            }

            AllocatorPtr GetAllocator() const {
                return allocator_;
            }

            Status AddOutput(OrtValue value, int index, const std::string &name) {
                int mlvalue_idx = value_name_idx_map_.Add(name);
                ort_value_idx_nodearg_map_[mlvalue_idx] = kernel_->Info().GetOutputType(index);

                output_index_to_mlvalue_map_[index] = mlvalue_idx;
                fetches_.push_back(value);
                fetches_mlvalue_idxs_.push_back(mlvalue_idx);
                return Status::OK();
            }

            Status AddInput(OrtValue value, int index, const std::string &name) {
                int mlvalue_idx = value_name_idx_map_.Add(name);

                input_index_to_mlvalue_map_[index] = mlvalue_idx;
                ort_value_idx_nodearg_map_[mlvalue_idx] = kernel_->Info().GetInputType(index);
                feeds_.push_back(value);
                feed_mlvalue_idxs_.push_back(mlvalue_idx);
                return Status::OK();
            }

        protected:

            NodeIndexInfo &GetNodeIndexInfo() {
                if (!node_index_info_) {
                    node_index_info_ = std::unique_ptr<NodeIndexInfo>(
                            new NodeIndexInfo({&kernel_->Node()}, value_name_idx_map_));
                }
                return *node_index_info_;
            }

            std::unique_ptr<OpKernel> kernel_;
            const logging::Logger *const logger_;

            friend ExecutableKernelContextImpl;

            OrtValueNameIdxMap value_name_idx_map_;
            std::unordered_map<int, const ONNX_NAMESPACE::TypeProto *> ort_value_idx_nodearg_map_;

            std::unique_ptr<NodeIndexInfo> node_index_info_;

            std::vector<int> input_index_to_mlvalue_map_;
            std::vector<int> output_index_to_mlvalue_map_;
            std::vector<int> fetches_mlvalue_idxs_;
            std::vector<OrtValue> fetches_;
            std::vector<int> feed_mlvalue_idxs_;
            std::vector<OrtValue> feeds_;

            std::unique_ptr<IExecutionProvider> provider_;
            AllocatorPtr allocator_;
        };

        ExecutableKernelContextImpl(std::unique_ptr<Info> info) :
        // Ideally we would remove the NodeIndexInfo from the constructor, since we only have one node
                IExecutionFrame(info->value_name_idx_map_, info->GetNodeIndexInfo(), info->fetches_mlvalue_idxs_),
                info_(std::move(info)) {
            Init(info_->feed_mlvalue_idxs_, info_->feeds_, std::unordered_map<int, OrtValue>(), info_->fetches_);
        };

        Status SetInput(OrtValue &value, int index) {
            return SetOrtValue(value, info_->input_index_to_mlvalue_map_[index]);
        }

        Status SetOutput(OrtValue &value, int index) {
            return SetOrtValue(value, info_->output_index_to_mlvalue_map_[index]);
        }

        Status Compute() {
            OpKernelContext context(this, info_->kernel_.get(), nullptr, *info_->logger_);
            Status status = info_->kernel_->Compute(&context);
            return status;
        }


    protected:

        AllocatorPtr GetAllocatorImpl(const OrtMemoryInfo &info) const override {
            return info_->provider_->GetAllocator(info.id, info.mem_type);
        }

        Status
        CopyTensor(__attribute__((unused)) const Tensor &src, __attribute__((unused)) Tensor &dest) const override {
            return Status(ONNXRUNTIME, NOT_IMPLEMENTED, "CopyTensor is not implemented for Single Kernel Execution.");
        }

        Status CreateNodeOutputMLValueImpl(OrtValue &ort_value, int ort_value_idx, const TensorShape *shape,
                                           size_t nnz) override;

    private:
        const std::unique_ptr<const Info> info_;

        const int device_id_{0};
        const OrtMemType mem_type_{OrtMemTypeDefault};

    };

}  // namespace onnxruntime
