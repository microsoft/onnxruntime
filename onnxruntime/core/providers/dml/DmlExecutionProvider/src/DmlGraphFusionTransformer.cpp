#pragma once

#include "precomp.h"
#include "GraphDescBuilder.h"
#include "ExecutionProvider.h"
#include "DmlGraphFusionTransformer.h"
#include "GraphPartitioner.h"

namespace Dml
{
    DmlGraphFusionTransformer::DmlGraphFusionTransformer(
        const std::string& name,
        const onnxruntime::IExecutionProvider* provider
    )
        :onnxruntime::GraphTransformer(name),
         m_providerImpl(static_cast<const ExecutionProvider*>(provider)->GetImpl())
    {

    }

    std::unordered_map<const onnx::TensorProto*, std::vector<uint32_t>>
    GetInitializerToPartitionMap(
        const onnxruntime::Graph& graph,
        gsl::span<std::unique_ptr<GraphPartition>> partitions
    )
    {
        std::unordered_map<const onnx::TensorProto*, std::vector<uint32_t>> initializerPartitionMap;

        for (uint32_t partitionIndex = 0; partitionIndex < gsl::narrow_cast<uint32_t>(partitions.size()); ++partitionIndex)
        {
            auto& partition = partitions[partitionIndex];

            // Skip partitions which have been merged into other partitions
            if (partition->GetRootMergedPartition() != partition.get())
            {
                continue;
            }

            std::unordered_map<std::string, onnx::TensorProto> transferredInitializerMap;

            for (const std::string& input : partition->GetInputs())
            {
                const onnx::TensorProto* tensor = nullptr;
                if (graph.GetInitializedTensor(input, tensor))
                {
                    initializerPartitionMap[tensor].push_back(partitionIndex);
                }
            }
        }

        return initializerPartitionMap;
    }
	
    onnxruntime::common::Status DmlGraphFusionTransformer::ApplyImpl(
        onnxruntime::Graph& graph,
        bool& modified,
        int graph_level,
        const onnxruntime::logging::Logger& logger) const
    {
        
        //std::vector<std::unique_ptr<onnxruntime::ComputeCapability>> result;

        //// Initializers needed by any graph partition
        //std::unordered_set<std::string> requiredInitializerMap;

        //std::unordered_map<const onnxruntime::Node*, GraphNodeProperties> graphNodePropertyMap;
        //std::vector<std::unique_ptr<GraphPartition>> partitions = BuildPartitions(
        //    graph,
        //    *m_providerImpl->GetInternalRegistrationInfoMap(), 
        //    m_providerImpl->GetKernelRegistry().get(),
        //    m_providerImpl->GetSupportedDeviceDataTypeMask(),
        //    graphNodePropertyMap, 
        //    requiredInitializerMap);

        //// Create a map between each initialized tensor and the partition(s) it is part of.
        //auto initializerPartitionMap = GetInitializerToPartitionMap(graph, partitions);

        //for (uint32_t partitionIndex = 0; partitionIndex < partitions.size(); ++partitionIndex)
        //{
        //    auto& partition = partitions[partitionIndex];

        //    if (partition->GetRootMergedPartition() != partition.get() ||
        //        !partition->IsDmlPartition())
        //    {
        //        continue;
        //    }

        //    // Create a map which will store by name each initializer which should be transferred to the 
        //    // partition.  This prevents OnnxRuntime from allocating GPU resources and uploading those initializers,
        //    // so the partiton's kernel can do so.  In the process, it will pre-process weights while consuming a CPU
        //    // backed resource, avoiding an extra set of GPU resources in memory.
        //    // A shared pointer is used so the functor and contained initializer captures can be cheaply copied within ORT.
        //    auto transferredInitializerMap = std::make_shared<std::unordered_map<std::string, onnx::TensorProto>>();

        //    
        //    if (partition->IsDmlGraphPartition())
        //    {
        //        for (const auto& input : partition->GetInputs())
        //        {
        //            const onnx::TensorProto* tensor = nullptr;
        //            if (graph.GetInitializedTensor(input, tensor))
        //            {
        //                // It's only safe to transfer tensors which are used by this partition alone.
        //                auto iter = initializerPartitionMap.find(tensor);
        //                assert(iter != initializerPartitionMap.end());
        //                if (iter->second.size() > 1)
        //                {
        //                    if (requiredInitializerMap.find(input) != requiredInitializerMap.end())
        //                    {
        //                        // The kernel relies on this input to be initialized, and it should be small enough to copy
        //                        // cheaply. FusedGraphKernel only handles constant CPU inputs through transferred initializers,
        //                        // rather than ORT, to avoid mismatches in policy or implementation causing failures.
        //                        (*transferredInitializerMap)[input] = const_cast<onnx::TensorProto&>(*tensor);
        //                    }

        //                    continue;
        //                }

        //                // Transfer the initializer
        //                auto& graphTensor = const_cast<onnx::TensorProto&>(*tensor);

        //                onnx::TensorProto partitionTensor;
        //                graphTensor.Swap(&partitionTensor);
        //                (*transferredInitializerMap)[input] = std::move(partitionTensor);
        //        
        //                const_cast<onnxruntime::InitializedTensorSet&>(graph.GetAllInitializedTensors()).erase(graph.GetAllInitializedTensors().find(input));
        //            }
        //        }
        //    }

        //    result.push_back(ComputationCapacityFromPartition(
        //        partition.get(), 
        //        partitionIndex, 
        //        graph, 
        //        std::move(graphNodePropertyMap),
        //        registryForPartitionKernels,
        //        partitionKernelPrefix,
        //        transferredInitializerMap
        //    ));
        //}

        return onnxruntime::common::Status::OK();
    }
}
