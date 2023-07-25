#pragma once

#include "../../../OperatorAuthorHelper/OperatorHelper.h"
#include "../MLOperatorAuthorImpl.h"

#include "../External/D3DX12/d3dx12.h"
#include <d3d12.h>

// NOTE: When this operator's implementation is moved into DML, the associated FP16 fallback
//       should be removed from IsCustomOpShader(...) in
//       onnxruntime\core\providers\dml\DmlExecutionProvider\src\ExecutionProvider.cpp

// The shader headers are produced using "GeneratedShaders/GenerateShaders.bat"
namespace GridSample_uint16_float
{
    #include "GeneratedShaders/grid_sample_uint16_float.h"
}

namespace GridSample_uint_float
{
    #include "GeneratedShaders/grid_sample_uint_float.h"
}

namespace GridSample_uint64_float
{
    #include "GeneratedShaders/grid_sample_uint64_float.h"
}

namespace GridSample_int16_float
{
    #include "GeneratedShaders/grid_sample_int16_float.h"
}

namespace GridSample_int_float
{
    #include "GeneratedShaders/grid_sample_int_float.h"
}

namespace GridSample_int64_float
{
    #include "GeneratedShaders/grid_sample_int64_float.h"
}

namespace GridSample_fp16_float
{
    #include "GeneratedShaders/grid_sample_fp16_float.h"
}

namespace GridSample_float_float
{
    #include "GeneratedShaders/grid_sample_float_float.h"
}

namespace GridSample_double_float
{
    #include "GeneratedShaders/grid_sample_double_float.h"
}

namespace GridSample_bool_float
{
    #include "GeneratedShaders/grid_sample_bool_float.h"
}

namespace GridSample_uint16_fp16
{
    #include "GeneratedShaders/grid_sample_uint16_fp16.h"
}

namespace GridSample_uint_fp16
{
    #include "GeneratedShaders/grid_sample_uint_fp16.h"
}

namespace GridSample_uint64_fp16
{
    #include "GeneratedShaders/grid_sample_uint64_fp16.h"
}

namespace GridSample_int16_fp16
{
    #include "GeneratedShaders/grid_sample_int16_fp16.h"
}

namespace GridSample_int_fp16
{
    #include "GeneratedShaders/grid_sample_int_fp16.h"
}

namespace GridSample_int64_fp16
{
    #include "GeneratedShaders/grid_sample_int64_fp16.h"
}

namespace GridSample_fp16_fp16
{
    #include "GeneratedShaders/grid_sample_fp16_fp16.h"
}

namespace GridSample_float_fp16
{
    #include "GeneratedShaders/grid_sample_float_fp16.h"
}

namespace GridSample_double_fp16
{
    #include "GeneratedShaders/grid_sample_double_fp16.h"
}

namespace GridSample_bool_fp16
{
    #include "GeneratedShaders/grid_sample_bool_fp16.h"
}

namespace GridSample_uint16_double
{
    #include "GeneratedShaders/grid_sample_uint16_double.h"
}

namespace GridSample_uint_double
{
    #include "GeneratedShaders/grid_sample_uint_double.h"
}

namespace GridSample_uint64_double
{
    #include "GeneratedShaders/grid_sample_uint64_double.h"
}

namespace GridSample_int16_double
{
    #include "GeneratedShaders/grid_sample_int16_double.h"
}

namespace GridSample_int_double
{
    #include "GeneratedShaders/grid_sample_int_double.h"
}

namespace GridSample_int64_double
{
    #include "GeneratedShaders/grid_sample_int64_double.h"
}

namespace GridSample_fp16_double
{
    #include "GeneratedShaders/grid_sample_fp16_double.h"
}

namespace GridSample_float_double
{
    #include "GeneratedShaders/grid_sample_float_double.h"
}

namespace GridSample_double_double
{
    #include "GeneratedShaders/grid_sample_double_double.h"
}

namespace GridSample_bool_double
{
    #include "GeneratedShaders/grid_sample_bool_double.h"
}


#include <wrl/client.h>
#include <wrl/implements.h>

#include <sstream>

using namespace Microsoft::WRL;

enum DmlGridSampleKernelInputIndex : uint32_t
{
    Input,
    Grid,
};

enum DmlGridSampleMode : uint32_t
{
    Bilinear,
    Nearest,
    Bicubic,
};

enum DmlGridSamplePaddingMode : uint32_t
{
    Zeros,
    Border,
    Reflection
};

// Helper to derive dimensions and attributes from either the GridSample shape inferrer or the GridSample kernel constructor.
struct DmlGridSampleParameters
{
    uint32_t batchSize = 0;
    uint32_t channelSize = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    int64_t alignCorners = 0;
    DmlGridSampleMode mode = DmlGridSampleMode::Bilinear;
    DmlGridSamplePaddingMode paddingMode = DmlGridSamplePaddingMode::Zeros;

    DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_UNKNOWN;

    DmlGridSampleParameters(){}

    DmlGridSampleParameters(
        const OperatorHelper::IKernelInformationAdapter& kernelInfo,
        const OperatorHelper::IShapeInformationAdapter& shapeInfo)
    {
        auto& attributes = kernelInfo.GetAttributes();

        alignCorners = attributes.GetOptionalAttribute<int64_t>(AttrName::AlignCorners, 0);

        std::string str_attrib = attributes.GetOptionalAttribute<std::string>(AttrName::Mode, "bilinear");
        ML_CHECK_VALID_ARGUMENT(str_attrib == "bilinear" || str_attrib == "nearest" || str_attrib == "bicubic");
        if (str_attrib == "bilinear")
        {
            mode = DmlGridSampleMode::Bilinear;
        }
        else if (str_attrib == "nearest")
        {
            mode = DmlGridSampleMode::Nearest;
        }
        else if (str_attrib == "bicubic")
        {
            mode = DmlGridSampleMode::Bicubic;
        }

        str_attrib = attributes.GetOptionalAttribute<std::string>(AttrName::PaddingMode,  "zeros");
        ML_CHECK_VALID_ARGUMENT(str_attrib == "zeros" || str_attrib == "border" || str_attrib == "reflection");
        if (str_attrib == "zeros")
        {
            paddingMode = DmlGridSamplePaddingMode::Zeros;
        }
        else if (str_attrib == "border")
        {
            paddingMode = DmlGridSamplePaddingMode::Border;
        }
        else if (str_attrib == "reflection")
        {
            paddingMode = DmlGridSamplePaddingMode::Reflection;
        }

        // input 0: signal (required; tensor)
        {
            // Input shape is expected to be [batch_size, channels, height, width]
            // 4-D tensor of shape (N, C, H_out, W_out) of sampled values.
            // For integer input types, intermediate values are computed as floating point and cast to integer at the end.            uint32_t rank = shapeInfo.GetInputTensorDimensionCount(DmlGridSampleKernelInputIndex::Input);
            uint32_t rank = shapeInfo.GetInputTensorDimensionCount(DmlGridSampleKernelInputIndex::Input);
            ML_CHECK_VALID_ARGUMENT(rank == 4, "Input shape must be 4D.");

            auto dims = shapeInfo.GetInputTensorShape(DmlGridSampleKernelInputIndex::Input);
            assert(dims.size() == rank);
            this->batchSize = dims[0];
            this->channelSize = dims[1];

            MLOperatorEdgeDescription edgeDesc = kernelInfo.GetInputEdgeDescription(DmlGridSampleKernelInputIndex::Input);

            assert(edgeDesc.edgeType == MLOperatorEdgeType::Tensor);
            this->dataType = Dml::GetDmlDataTypeFromMlDataType(edgeDesc.tensorDataType);
        }

        // input 1: grid (required; tensor)
        {
            // Grid shape is expected to be [batch_size, height_out, width_out, 2]
            uint32_t rank = shapeInfo.GetInputTensorDimensionCount(DmlGridSampleKernelInputIndex::Grid);
            ML_CHECK_VALID_ARGUMENT(rank == 4, "Input shape must be 4D.");

            auto dims = shapeInfo.GetInputTensorShape(DmlGridSampleKernelInputIndex::Grid);
            assert(dims.size() == rank);
            this->height = dims[1];
            this->width = dims[2];
        }
    }

};

namespace GridSampleHelpers
{
    // Divides and rounds
    inline uint32_t CeilDivide(uint32_t dividend, uint32_t divisor)
    {
        uint64_t temp = static_cast<uint64_t>(dividend) + divisor - 1;
        return static_cast<uint32_t>(temp / divisor);
    }

    // Gets the next number of elements to dispatch to the GPU within a loop handling a large
    // total number of tensor elements and threads.
    void GetNextDispatchSize(
        uint32_t elementCount,
        uint32_t elementsPerThread,
        uint32_t numThreads,
        _Out_ uint32_t& dispatch,
        _Out_ uint32_t& pendingElementCount
    )
    {
        // Max threads per workgroup is 2^10 (1024). Max dispatch per dimension is 2^16. Taken together, we can dispatch a maximum of
        // 2^26 (268,435,456) threads along a single dimension. This should suffice for a majority of the workload. Therefore, even
        // though it is possible to dispatch up to (2^16)^3 workgroups simultaneously, we stick to the simpler 1D dispatch alternative.
        assert(numThreads <= D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP);

        const uint32_t maxThreadsPerDispatch = numThreads * D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;

        const uint32_t requiredThreadCount = CeilDivide(elementCount, elementsPerThread);

        // Compute max dispatchable elements
        const uint32_t availableThreadCount = std::min(requiredThreadCount, maxThreadsPerDispatch);

        // Compute required thread group count
        uint32_t workGroupCount1D = CeilDivide(availableThreadCount, numThreads);

        // Compute min dispatch size
        dispatch = workGroupCount1D;

        // With the dispatch size computed, compute the dispatched element count
        const uint32_t dispatchedElementCount = workGroupCount1D * numThreads * elementsPerThread;

        // Update the pending element count
        pendingElementCount = (dispatchedElementCount < elementCount) ? elementCount - dispatchedElementCount : 0;
    }
}

class DmlGridSampleOperator : public WRL::Base<IMLOperatorKernel>
{
private:
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12RootSignature> m_gridSampleRootSignature;
    ComPtr<ID3D12PipelineState> m_gridSamplePipelineState;
    DmlGridSampleParameters m_params = {};


    // Allocate temporary buffers if needed
    struct ResourceDesc
    {
        ComPtr<ID3D12Resource> Resource;
        std::array<uint32_t, 4> Sizes;
        std::array<uint32_t, 4> Strides;
    };

    struct GridSampleShaderConstants
    {
        uint32_t StartIndex;
        uint32_t ElementCount;
        uint32_t Mode;
        uint32_t PaddingMode;
        uint32_t InputSizes[4];
        uint32_t InputStrides[4];
        uint32_t GridSizes[4];
        uint32_t GridStrides[4];
        uint32_t OutputSizes[4];
        uint32_t OutputStrides[4];
        uint32_t AlignCorners;
    };

public:

    DmlGridSampleOperator(IMLOperatorKernelCreationContext* context)
    {
        ComPtr<IUnknown> executionObject;
        context->GetExecutionInterface(executionObject.GetAddressOf());

        ComPtr<ID3D12GraphicsCommandList> commandList;
        executionObject.As(&commandList);

        ORT_THROW_IF_FAILED(commandList->GetDevice(IID_ID3D12Device, &m_device));

        MLOperatorKernelCreationContext creationContext(context);
        OperatorHelper::KernelInformationAdapter kernelInfo{creationContext};
        OperatorHelper::ShapeInformationAdapter shapeInfo{creationContext};
        m_params = DmlGridSampleParameters(kernelInfo, shapeInfo);

        MLOperatorEdgeDescription inputEdgeDesc;
        ORT_THROW_IF_FAILED(context->GetInputEdgeDescription(0, &inputEdgeDesc));
        assert(inputEdgeDesc.edgeType == MLOperatorEdgeType::Tensor);

        MLOperatorEdgeDescription gridEdgeDesc;
        ORT_THROW_IF_FAILED(context->GetInputEdgeDescription(0, &gridEdgeDesc));
        assert(gridEdgeDesc.edgeType == MLOperatorEdgeType::Tensor);

        PrepareGridSample(inputEdgeDesc.tensorDataType, gridEdgeDesc.tensorDataType);
    }

    void PrepareGridSample(MLOperatorTensorDataType inputDataType, MLOperatorTensorDataType gridDataType)
    {
        // Compute root signature.
        const int uavCount = 3; // 3 bound UAVs: input, grid & output
        std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters;
        rootParameters.resize(uavCount + 1);

        for (uint32_t i = 0; i < uavCount; i++)
        {
            rootParameters[i].InitAsUnorderedAccessView(i);
        }

        // cbuffer Constants
        // {
        //     uint StartIndex;
        //     uint ElementCount;
        //     uint Mode;
        //     uint PaddingMode;
        //     uint4 InputSizes;
        //     uint4 InputStrides;
        //     uint4 GridSizes;
        //     uint4 GridStrides;
        //     uint4 OutputSizes;
        //     uint4 OutputStrides;
        //     uint AlignCorners;
        // };
        int constantCount = 29;
        rootParameters[uavCount].InitAsConstants(constantCount, 0);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
        desc.Init_1_1(static_cast<uint32_t>(rootParameters.size()), rootParameters.data());

        ComPtr<ID3DBlob> rootSignatureBlob;
        ComPtr<ID3DBlob> rootSignatureErrorBlob;
        ORT_THROW_IF_FAILED(D3D12SerializeVersionedRootSignature(
            &desc,
            rootSignatureBlob.GetAddressOf(),
            rootSignatureErrorBlob.GetAddressOf()
        ));

        ORT_THROW_IF_FAILED(m_device->CreateRootSignature(
            0,
            rootSignatureBlob->GetBufferPointer(),
            rootSignatureBlob->GetBufferSize(),
            IID_ID3D12RootSignature,
            &m_gridSampleRootSignature
        ));

        // Describe and create the compute pipeline state object (PSO).
        D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
        computePsoDesc.pRootSignature = m_gridSampleRootSignature.Get();

        switch (gridDataType)
        {
            case MLOperatorTensorDataType::Float:
            {
                switch (inputDataType)
                {
                case MLOperatorTensorDataType::UInt16:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_uint16_float::g_GridSample, sizeof(GridSample_uint16_float::g_GridSample));
                break;

                case MLOperatorTensorDataType::UInt32:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_uint_float::g_GridSample, sizeof(GridSample_uint_float::g_GridSample));
                break;

                case MLOperatorTensorDataType::UInt64:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_uint64_float::g_GridSample, sizeof(GridSample_uint64_float::g_GridSample));
                break;

                case MLOperatorTensorDataType::Int16:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_int16_float::g_GridSample, sizeof(GridSample_int16_float::g_GridSample));
                break;

                case MLOperatorTensorDataType::Int32:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_int_float::g_GridSample, sizeof(GridSample_int_float::g_GridSample));
                break;

                case MLOperatorTensorDataType::Int64:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_int64_float::g_GridSample, sizeof(GridSample_int64_float::g_GridSample));
                break;

                case MLOperatorTensorDataType::Float16:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_fp16_float::g_GridSample, sizeof(GridSample_fp16_float::g_GridSample));
                break;

                case MLOperatorTensorDataType::Float:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_float_float::g_GridSample, sizeof(GridSample_float_float::g_GridSample));
                break;

                case MLOperatorTensorDataType::Double:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_double_float::g_GridSample, sizeof(GridSample_double_float::g_GridSample));
                break;

                case MLOperatorTensorDataType::Bool:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_bool_float::g_GridSample, sizeof(GridSample_bool_float::g_GridSample));
                break;

                default:
                ORT_THROW_HR(E_INVALIDARG);
                }
                break;
            }
            case MLOperatorTensorDataType::Float16:
            {
                switch (inputDataType)
                {
                case MLOperatorTensorDataType::UInt16:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_uint16_fp16::g_GridSample, sizeof(GridSample_uint16_fp16::g_GridSample));
                break;

                case MLOperatorTensorDataType::UInt32:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_uint_fp16::g_GridSample, sizeof(GridSample_uint_fp16::g_GridSample));
                break;

                case MLOperatorTensorDataType::UInt64:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_uint64_fp16::g_GridSample, sizeof(GridSample_uint64_fp16::g_GridSample));
                break;

                case MLOperatorTensorDataType::Int16:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_int16_fp16::g_GridSample, sizeof(GridSample_int16_fp16::g_GridSample));
                break;

                case MLOperatorTensorDataType::Int32:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_int_fp16::g_GridSample, sizeof(GridSample_int_fp16::g_GridSample));
                break;

                case MLOperatorTensorDataType::Int64:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_int64_fp16::g_GridSample, sizeof(GridSample_int64_fp16::g_GridSample));
                break;

                case MLOperatorTensorDataType::Float16:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_fp16_fp16::g_GridSample, sizeof(GridSample_fp16_fp16::g_GridSample));
                break;

                case MLOperatorTensorDataType::Float:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_float_fp16::g_GridSample, sizeof(GridSample_float_fp16::g_GridSample));
                break;

                case MLOperatorTensorDataType::Double:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_double_fp16::g_GridSample, sizeof(GridSample_double_fp16::g_GridSample));
                break;

                case MLOperatorTensorDataType::Bool:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_bool_fp16::g_GridSample, sizeof(GridSample_bool_fp16::g_GridSample));
                break;

                default:
                ORT_THROW_HR(E_INVALIDARG);
                }
                break;
            }
            case MLOperatorTensorDataType::Double:
            {
                switch (inputDataType)
                {
                case MLOperatorTensorDataType::UInt16:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_uint16_double::g_GridSample, sizeof(GridSample_uint16_double::g_GridSample));
                break;

                case MLOperatorTensorDataType::UInt32:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_uint_double::g_GridSample, sizeof(GridSample_uint_double::g_GridSample));
                break;

                case MLOperatorTensorDataType::UInt64:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_uint64_double::g_GridSample, sizeof(GridSample_uint64_double::g_GridSample));
                break;

                case MLOperatorTensorDataType::Int16:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_int16_double::g_GridSample, sizeof(GridSample_int16_double::g_GridSample));
                break;

                case MLOperatorTensorDataType::Int32:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_int_double::g_GridSample, sizeof(GridSample_int_double::g_GridSample));
                break;

                case MLOperatorTensorDataType::Int64:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_int64_double::g_GridSample, sizeof(GridSample_int64_double::g_GridSample));
                break;

                case MLOperatorTensorDataType::Float16:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_fp16_double::g_GridSample, sizeof(GridSample_fp16_double::g_GridSample));
                break;

                case MLOperatorTensorDataType::Float:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_float_double::g_GridSample, sizeof(GridSample_float_double::g_GridSample));
                break;

                case MLOperatorTensorDataType::Double:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_double_double::g_GridSample, sizeof(GridSample_double_double::g_GridSample));
                break;

                case MLOperatorTensorDataType::Bool:
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(GridSample_bool_double::g_GridSample, sizeof(GridSample_bool_double::g_GridSample));
                break;

                default:
                ORT_THROW_HR(E_INVALIDARG);
                }
                break;
            }
            default:
            ORT_THROW_HR(E_INVALIDARG);
        }

        ORT_THROW_IF_FAILED(m_device->CreateComputePipelineState(&computePsoDesc, IID_ID3D12PipelineState, &m_gridSamplePipelineState));
    }

    // Computes the outputs of the kernel.  This may be called multiple times
    // simultaneously within the same instance of the class.  Implementations
    // of this method must be thread-safe.
    STDMETHOD(Compute)(IMLOperatorKernelContext* context)
    {
        try
        {
            // Get the input tensor
            ComPtr<IMLOperatorTensor> inputTensor;
            ORT_THROW_IF_FAILED(context->GetInputTensor(0, inputTensor.GetAddressOf()));

            // Get the grid tensor
            ComPtr<IMLOperatorTensor> gridTensor;
            ORT_THROW_IF_FAILED(context->GetInputTensor(1, gridTensor.GetAddressOf()));

            // Get the output tensor
            ComPtr<IMLOperatorTensor> outputTensor;
            context->GetOutputTensor(0, outputTensor.GetAddressOf());

            if (outputTensor->IsCpuData() || inputTensor->IsCpuData() || gridTensor->IsCpuData())
            {
                return E_UNEXPECTED;
            }

            ComPtr<IUnknown> executionObject;
            ComPtr<ID3D12GraphicsCommandList> commandList;
            context->GetExecutionInterface(executionObject.GetAddressOf());
            executionObject.As(&commandList);

            // Get the input and output shape sizes
            auto inputDims = GetTensorDimensions(inputTensor.Get());
            auto gridDims = GetTensorDimensions(gridTensor.Get());
            auto outputDims = GetTensorDimensions(outputTensor.Get());

            ComPtr<IUnknown> inputUnknown;
            ComPtr<ID3D12Resource> inputResource;
            inputTensor->GetDataInterface(inputUnknown.GetAddressOf());
            ORT_THROW_IF_FAILED(inputUnknown.As(&inputResource));

            ComPtr<IUnknown> gridUnknown;
            ComPtr<ID3D12Resource> gridResource;
            gridTensor->GetDataInterface(gridUnknown.GetAddressOf());
            ORT_THROW_IF_FAILED(gridUnknown.As(&gridResource));

            ComPtr<IUnknown> outputUnknown;
            ComPtr<ID3D12Resource> outputResource;
            outputTensor->GetDataInterface(outputUnknown.GetAddressOf());
            ORT_THROW_IF_FAILED(outputUnknown.As(&outputResource));

            return Compute(
                commandList.Get(),
                context,
                inputResource.Get(),
                inputDims,
                gridResource.Get(),
                gridDims,
                outputResource.Get(),
                outputDims
            );
        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }

    HRESULT Compute(
        ID3D12GraphicsCommandList* commandList,
        IMLOperatorKernelContext* context,
        ID3D12Resource* inputResource,
        gsl::span<const uint32_t> inputDims,
        ID3D12Resource* gridResource,
        gsl::span<const uint32_t> gridDims,
        ID3D12Resource* outputResource,
        gsl::span<const uint32_t> outputDims)
    {
        try
        {
            GridSample(
                inputResource,
                inputDims,
                gridResource,
                gridDims,
                outputResource,
                outputDims,
                commandList);
        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }

    void GridSample(
        ID3D12Resource* inputResource,
        gsl::span<const uint32_t> inputDims,
        ID3D12Resource* gridResource,
        gsl::span<const uint32_t> gridDims,
        ID3D12Resource* outputResource,
        gsl::span<const uint32_t> outputDims,
        ID3D12GraphicsCommandList* commandList)
    {
        std::array<uint32_t, 4> inputStrides;
        std::array<uint32_t, 4> gridStrides;
        std::array<uint32_t, 4> outputStrides;
        Dml::GetDescendingPackedStrides(inputDims, inputStrides);
        Dml::GetDescendingPackedStrides(gridDims, gridStrides);
        Dml::GetDescendingPackedStrides(outputDims, outputStrides);

        // Transition resources from common to UAV state
        D3D12_RESOURCE_BARRIER barriers[3];

        barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
            inputResource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

        barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
            gridResource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

        barriers[2] = CD3DX12_RESOURCE_BARRIER::Transition(
            outputResource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

        inputResource->SetName(L"InputResource");
        outputResource->SetName(L"OutputResource");
        gridResource->SetName(L"GridResource");

        commandList->ResourceBarrier(3, barriers);

        // Set the root signature and pipeline state
        commandList->SetComputeRootSignature(m_gridSampleRootSignature.Get());
        commandList->SetPipelineState(m_gridSamplePipelineState.Get());

        // Each iteration of the below loop represents 1 level in the Stockham DFT
        // Dispatch in a loop
        GridSampleShaderConstants constants = {};
        constants.AlignCorners = static_cast<uint32_t>(m_params.alignCorners);
        constants.Mode = static_cast<uint32_t>(m_params.mode);
        constants.PaddingMode = static_cast<uint32_t>(m_params.paddingMode);
        std::copy(inputDims.begin(), inputDims.end(), constants.InputSizes);
        std::copy(inputStrides.begin(), inputStrides.end(), constants.InputStrides);
        std::copy(gridDims.begin(), gridDims.end(), constants.GridSizes);
        std::copy(gridStrides.begin(), gridStrides.end(), constants.GridStrides);
        std::copy(outputDims.begin(), outputDims.end(), constants.OutputSizes);
        std::copy(outputStrides.begin(), outputStrides.end(), constants.OutputStrides);

        constants.ElementCount = ComputeElementCountFromDimensions(constants.OutputSizes);
        std::array<ID3D12Resource*, 3> uav_resources = { inputResource, gridResource, outputResource };
        Dispatch(uav_resources, constants, commandList);

        // Transition resources to common state
        barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
                inputResource,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
                );

        barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
                gridResource,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
                );

        barriers[2] = CD3DX12_RESOURCE_BARRIER::Transition(
                outputResource,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
                );

        commandList->ResourceBarrier(3, barriers);
    }

    std::vector<uint32_t> GetTensorDimensions(IMLOperatorTensor* tensor)
    {
        auto inputDimsSize = tensor->GetDimensionCount();
        auto dims = std::vector<uint32_t>(inputDimsSize);
        ORT_THROW_IF_FAILED(tensor->GetShape(static_cast<uint32_t>(dims.size()), dims.data()));
        return dims;
    }

    template <typename TConstants, uint32_t TSize>
    void Dispatch(
        std::array<ID3D12Resource*, TSize>& resources,
        TConstants& constants,
        ID3D12GraphicsCommandList* commandList)
    {
        D3D12_RESOURCE_BARRIER uav_barriers[TSize];

        std::transform(
            resources.begin(), resources.end(),
            uav_barriers,
            [](auto& resource) { return CD3DX12_RESOURCE_BARRIER::UAV(resource); } );
        commandList->ResourceBarrier(TSize, uav_barriers);

        for (uint32_t i = 0; i < TSize; i++)
        {
            // Set resource views
            if (resources[i]) {
                commandList->SetComputeRootUnorderedAccessView(
                    i, // root parameter index
                    resources[i]->GetGPUVirtualAddress()
                );
            }
            else
            {
                commandList->SetComputeRootUnorderedAccessView(
                    i, // root parameter index
                    {}
                );

            }
        }

        auto pendingElementCount = constants.ElementCount;

        // Dispatch up to the maximum number of threads per iteration until
        // all elements are completed
        while (pendingElementCount > 0)
        {
            constants.StartIndex = constants.ElementCount - pendingElementCount;

            uint32_t dispatchSizeX;

            GridSampleHelpers::GetNextDispatchSize(
                pendingElementCount,
                1,
                64,
                dispatchSizeX,
                pendingElementCount
            );

            // Set root constants
            commandList->SetComputeRoot32BitConstants(
                TSize, // root parameter index
                29, // Constant count
                &constants,
                0 // offset
            );

            commandList->Dispatch(dispatchSizeX, 1, 1);
        }

        commandList->ResourceBarrier(2, uav_barriers);
    }
};

struct GridSampleShapeInferrer : public WRL::Base<IMLOperatorShapeInferrer>
{
    STDMETHOD(InferOutputShapes)(IMLOperatorShapeInferenceContext* context) noexcept
    {
        try
        {
            ComPtr<IMLOperatorShapeInferenceContextPrivate> contextPrivate;
            ORT_THROW_IF_FAILED(context->QueryInterface(IID_PPV_ARGS(&contextPrivate)));

            MLShapeInferenceContext inferenceContext(context);
            OperatorHelper::KernelInformationAdapter kernelInfo{inferenceContext};
            OperatorHelper::ShapeInformationAdapter shapeInfo{inferenceContext};
            DmlGridSampleParameters params(kernelInfo, shapeInfo);

            std::array<uint32_t, 4> outputDims = { params.batchSize, params.channelSize, params.height, params.width };

            ORT_THROW_IF_FAILED(context->SetOutputTensorShape(0, onnxruntime::narrow<uint32_t>(outputDims.size()), outputDims.data()));
        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }
};

class DmlGridSampleOperatorFactory : public WRL::Base<IMLOperatorKernelFactory>
{
public:
    STDMETHOD(CreateKernel)(
        IMLOperatorKernelCreationContext* context,
        IMLOperatorKernel** kernel)
    {
        try
        {
            auto dftOperator = wil::MakeOrThrow<DmlGridSampleOperator>(context);
            dftOperator.CopyTo(kernel);
            return S_OK;
        }
        catch (...)
        {
            return E_FAIL;
        }
    }

    static void RegisterGridSampleKernel(IMLOperatorRegistry* registry)
    {
        MLOperatorKernelDescription kernelDescription = {};
        kernelDescription.domain = "";
        kernelDescription.name = "GridSample";
        kernelDescription.minimumOperatorSetVersion = 16;
        kernelDescription.executionType = MLOperatorExecutionType::D3D12;

        // T1: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
        MLOperatorEdgeTypeConstrant t1Constraint;
        t1Constraint.typeLabel = "T1";
        std::vector<MLOperatorEdgeDescription> t1AllowedEdges
        {
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float16 },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Int8 },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Int16 },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Int32 },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Int64 },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::UInt8 },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::UInt16 },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::UInt32 },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::UInt64 },
        };
        t1Constraint.allowedTypes = t1AllowedEdges.data();
        t1Constraint.allowedTypeCount = static_cast<uint32_t>(t1AllowedEdges.size());

        // T2 : tensor(int32), tensor(int64)
        MLOperatorEdgeTypeConstrant t2Constraint;
        t2Constraint.typeLabel = "T2";
        std::vector<MLOperatorEdgeDescription> t2AllowedEdges
        {
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float16 },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float },
        };
        t2Constraint.allowedTypes = t2AllowedEdges.data();
        t2Constraint.allowedTypeCount = static_cast<uint32_t>(t2AllowedEdges.size());

        std::vector<MLOperatorEdgeTypeConstrant> typeConstraints{ t1Constraint, t2Constraint };
        kernelDescription.typeConstraints = typeConstraints.data();
        kernelDescription.typeConstraintCount = static_cast<uint32_t>(typeConstraints.size());

        MLOperatorAttributeNameValue alignedCornersAttributeValue;
        alignedCornersAttributeValue.name = AttrName::AlignCorners;
        alignedCornersAttributeValue.type = MLOperatorAttributeType::Int;
        alignedCornersAttributeValue.valueCount = 1;
        static const int64_t alignedCorners[] = { 0 };
        alignedCornersAttributeValue.ints = alignedCorners;

        MLOperatorAttributeNameValue modeAttributeValue;
        modeAttributeValue.name = AttrName::Mode;
        modeAttributeValue.type = MLOperatorAttributeType::String;
        modeAttributeValue.valueCount = 1;
        static const char* modes[] = { "bilinear" };
        modeAttributeValue.strings = modes;

        MLOperatorAttributeNameValue paddingModeAttributeValue;
        paddingModeAttributeValue.name = AttrName::Mode;
        paddingModeAttributeValue.type = MLOperatorAttributeType::String;
        paddingModeAttributeValue.valueCount = 1;
        static const char* paddingModes[] = { "zeros" };
        paddingModeAttributeValue.strings = paddingModes;

        std::vector<MLOperatorAttributeNameValue> attributeDefaultValues{
            alignedCornersAttributeValue,
            modeAttributeValue,
            paddingModeAttributeValue
        };

        kernelDescription.defaultAttributes = attributeDefaultValues.data();
        kernelDescription.defaultAttributeCount = static_cast<uint32_t>(attributeDefaultValues.size());
        kernelDescription.options = MLOperatorKernelOptions::None;
        kernelDescription.executionOptions = 0;

        auto shareInferrer = wil::MakeOrThrow<GridSampleShapeInferrer>();
        auto factory = wil::MakeOrThrow<DmlGridSampleOperatorFactory>();

        ComPtr<IMLOperatorRegistryPrivate> registryPrivate;
        ORT_THROW_IF_FAILED(registry->QueryInterface(IID_PPV_ARGS(&registryPrivate)));

        ORT_THROW_IF_FAILED(registryPrivate->RegisterOperatorKernel(
            &kernelDescription,
            factory.Get(),
            shareInferrer.Get(),
            nullptr,
            false, // isInternalOperator
            false, // alias
            false, // supportsGraph
            nullptr,
            nullptr,
            0));

    }
};
