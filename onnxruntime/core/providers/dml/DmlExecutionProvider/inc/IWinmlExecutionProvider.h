// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <functional>
#include <variant>
#include <optional>

#include "core/framework/op_kernel.h"

struct AbstractOperatorDesc;
interface IMLOperatorTensor;
struct DML_INPUT_GRAPH_EDGE_DESC;
struct DML_OUTPUT_GRAPH_EDGE_DESC;
struct DML_INTERMEDIATE_GRAPH_EDGE_DESC;

namespace onnxruntime
{
    class KernelDef;
    class Node;
}

namespace Windows::AI::MachineLearning::Adapter
{
    interface __declspec(uuid("5b19a18a-5ed5-4df2-a363-21b89380a698"))
    IWinmlExecutionProvider : public IUnknown
    {
    public:
        // Hold a reference to an object until preceding work in the queue is complete.  This
        // only needs to be handled by providers which hide the asynchronous nature of
        // computation, and involve resoures which cannot be automatically by work in the
        // the provider's underlying queues.
        virtual void QueueReference(IUnknown *object) = 0;

        virtual void GetShadowCopyIfRequired(
            bool isInternalOperator,
            IUnknown* data,
            IUnknown** dataCopy) const = 0;

        virtual void GetABIDataInterface(
            bool isInternalOperator,
            IUnknown* data,
            IUnknown** abiData) const = 0;

        virtual uint64_t TryGetPooledAllocationId(
            IUnknown* data,
            bool isInternalOperator) = 0;

        virtual void GetABIExecutionInterfaceAndInvalidateState(
            bool isInternalOperator,
            IUnknown** abiExecutionObject) const = 0;

        // Whether TransitionResourcesForOperator should be called before and after executing
        // an operator registered to this provider with the specified flags
        virtual bool TransitionsRequiredForOperator(bool isInternalOperator) = 0;

        // If TransitionsRequiredForOperator returns true, should be called before and after executing
        // an operator to transition its resources to and from the appropriate state.
        virtual void TransitionResourcesForOperator(
            bool isBeforeOp,
            uint32_t resourceCount,
            IUnknown** resources) = 0;

        // Waits for flushed work, discards unflushed work, and discards associated references to
        // prevent circular references.  Must be the last call on the object before destruction.
        virtual void Close() = 0;
    };

    using MLOperatorTensorGetter = std::function<std::variant<Microsoft::WRL::ComPtr<IMLOperatorTensor>, std::vector<Microsoft::WRL::ComPtr<IMLOperatorTensor>>>(uint32_t index)>;

    struct DmlOperatorParams
    {
        Microsoft::WRL::ComPtr<IDMLOperator> op;
        std::unique_ptr<AbstractOperatorDesc> desc;
    };

    // This is the counterpart to the MLOperatorGraphDesc ABI struct which owns its memory and uses containers.
    // Either nodesAsOperatorDesc or nodesAsIDMLOperator can have non-zero size.
    struct DmlGraphNodeCreateInfo
    {
        uint32_t nodeCount;
        std::vector<std::unique_ptr<AbstractOperatorDesc>> nodesAsOperatorDesc;

        // TODO: Remove this
        std::vector<Microsoft::WRL::ComPtr<IDMLOperator>> nodesAsIDMLOperator;
        
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
    };

    using GraphNodeFactory = std::function<void(
        const onnxruntime::Node& node,
        MLOperatorTensorGetter& constantInputGetter,
        const void* executionHandle,
        /*out*/ DmlGraphNodeCreateInfo* graphNodeCreateInfo
        )>;

    struct GraphNodeFactoryRegistration
    {
        GraphNodeFactory factory;
        std::optional<uint32_t> requiredInputCount;
    };

    using KernelSupportQuery = std::function<bool(const onnxruntime::Node& node)>;

    struct InternalRegistrationInfo
    {
        std::vector<uint32_t> requiredConstantCpuInputs;
        std::optional<GraphNodeFactoryRegistration> graphNodeFactoryRegistration;
        KernelSupportQuery supportQuery;
    };

    using InternalRegistrationInfoMap = std::unordered_map<onnxruntime::KernelDef*, std::shared_ptr<InternalRegistrationInfo>>;
}
