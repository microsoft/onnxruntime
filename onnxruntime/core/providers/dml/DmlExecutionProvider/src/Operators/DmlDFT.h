#pragma once

#include "../MLOperatorAuthorImpl.h"
#include "../../../OperatorAuthorHelper/OperatorHelper.h"

#include "../External/D3DX12/d3dx12.h"
#include <d3d12.h>

// NOTE: When this operator's implementation is moved into DML, the associated FP16 fallback
//       should be removed from IsCustomOpShader(...) in
//       onnxruntime\core\providers\dml\DmlExecutionProvider\src\ExecutionProvider.cpp

// The shader headers are produced using "GeneratedShaders/GenerateShaders.bat"
namespace StockhamFFT_Float32
{
    #include "GeneratedShaders/stockham.h"
}
namespace StockhamFFT_Float16
{
    #include "GeneratedShaders/stockham_fp16.h"
}

namespace BluesteinChirp_Float32
{
    #include "GeneratedShaders/bluestein_chirp.h"
}
namespace BluesteinChirp_Float16
{
    #include "GeneratedShaders/bluestein_chirp_fp16.h"
}

#include <wrl/client.h>
#include <wrl/implements.h>

#include <sstream>

using namespace Microsoft::WRL;

namespace DFTHelpers {
    // Divides and rounds up
    inline uint32_t CeilDivide(uint32_t dividend, uint32_t divisor)
    {
        uint64_t temp = static_cast<uint64_t>(dividend) + divisor - 1;
        return static_cast<uint32_t>(temp / divisor);
    }

    inline bool IsPowerOfTwo(uint32_t x)
    {
        return (x != 0) && ((x & (x - 1)) == 0);
    }

    template <typename T>
    T NextPowerOf2(T in) {
        in--;
        T out = 1;
        while (out <= in) {
            out <<= 1;
        }
        return out;
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

class GpuDFTOperator : public WRL::Base<IMLOperatorKernel>
{
private:
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12RootSignature> m_stockhamFFTRootSignature;
    ComPtr<ID3D12PipelineState> m_stockhamFFTPipelineState;
    ComPtr<ID3D12RootSignature> m_bluesteinChirpRootSignature;
    ComPtr<ID3D12PipelineState> m_bluesteinChirpPipelineState;

    int64_t m_axis;
    bool m_isOnesided;
    bool m_isInverse;

    // Allocate temporary buffers if needed
    struct ResourceDesc
    {
        ComPtr<ID3D12Resource> Resource;
        std::array<uint32_t, 4> Sizes;
        std::array<uint32_t, 4> Strides;
    };

    struct StockhamParameters
    {
        struct LoopRangeCalculator
        {
            unsigned Left;
            unsigned Right;
            unsigned End;
            unsigned CalculateIndex(unsigned index) const
            {
                if (index > 0 && index < End)
                {
                    unsigned range = Right - Left + 1;
                    index = Left + (index - 1) % range;
                }
                else if (index == End)
                {
                    index = Right + 1;
                }
                return index;
            }
        };

        ResourceDesc Window = {};
        std::vector<ResourceDesc> ResourceLoopList = {};
        LoopRangeCalculator LoopRange = {};
        uint32_t OutputIndex = 0;
        uint32_t NumberOfPasses = 0;
    };

    struct BluesteinZChirpParameters
    {
        ResourceDesc ZChirp = {};
        ResourceDesc AFFT = {};
        ResourceDesc B = {};
        ResourceDesc BFFT = {};

        StockhamParameters AFFTParams = {};
        StockhamParameters AFFTInverseParams = {};
        StockhamParameters BFFTParams = {};
    };

    enum class DFTType
    {
        Stockham = 0,
        BluesteinZChirp,
    };

    struct DFTParameters
    {
        DFTType Type = DFTType::Stockham;
        StockhamParameters StockhamParams = {};
        BluesteinZChirpParameters BluesteinZChirpParams = {};
        uint32_t DFTLength = 0;
    };

    struct DFTShaderConstants
    {
        uint32_t StartIndex;
        uint32_t ElementCount;
        uint32_t DFTIteration;
        uint32_t IsInverse;
        uint32_t InputSizes[4];
        uint32_t InputStrides[4];
        uint32_t OutputSizes[4];
        uint32_t OutputStrides[4];
        uint32_t WindowSizes[4];
        uint32_t WindowStrides[4];
        uint32_t HasWindow;
        float ChirpLength;
        float Scale;
        uint32_t DFTLength;
    };

    struct BluesteinZChirpShaderConstants
    {
        uint32_t StartIndex;
        uint32_t ElementCount;
        uint32_t DFTLength;
        uint32_t IsInverse;
    };

public:
    GpuDFTOperator(ID3D12Device* device, uint32_t axis = 1, bool isOnesided = true, bool isInverse = false, MLOperatorTensorDataType dataType = MLOperatorTensorDataType::Float)
     : m_device(device)
     , m_axis(axis)
     , m_isOnesided(isOnesided)
     , m_isInverse(isInverse)
    {
        PrepareStockhamFFT(dataType);
        PrepareBluesteinZChirp(dataType);
    }

    GpuDFTOperator(IMLOperatorKernelCreationContext* context)
    {
        ComPtr<IUnknown> executionObject;
        context->GetExecutionInterface(executionObject.GetAddressOf());

        ComPtr<ID3D12GraphicsCommandList> commandList;
        executionObject.As(&commandList);

        ORT_THROW_IF_FAILED(commandList->GetDevice(IID_ID3D12Device, &m_device));

        ORT_THROW_IF_FAILED(context->GetAttribute("axis", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&m_axis)));

        int64_t isInverseInt;
        ORT_THROW_IF_FAILED(context->GetAttribute("inverse", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&isInverseInt)));
        m_isInverse = static_cast<bool>(isInverseInt);

        int64_t isOnesidedInt;
        ORT_THROW_IF_FAILED(context->GetAttribute("onesided", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&isOnesidedInt)));
        m_isOnesided = static_cast<bool>(isOnesidedInt);

        MLOperatorEdgeDescription edgeDesc;
        ORT_THROW_IF_FAILED(context->GetInputEdgeDescription(0, &edgeDesc));
        assert(edgeDesc.edgeType == MLOperatorEdgeType::Tensor);

        PrepareStockhamFFT(edgeDesc.tensorDataType);
        PrepareBluesteinZChirp(edgeDesc.tensorDataType);
    }

    void PrepareBluesteinZChirp(MLOperatorTensorDataType dataType)
    {
        // Compute root signature.
        const int uavCount = 2; // 2 outputs: chirp, and the reflected chirp conjugate (B)
        std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters;
        rootParameters.resize(uavCount + 1);

        for (uint32_t i = 0; i < uavCount; i++)
        {
            rootParameters[i].InitAsUnorderedAccessView(i);
        }

        // cbuffer Constants // BluesteinZChirpShaderConstants
        // {
        //     uint StartIndex;
        //     uint ElementCount;
        //     uint DFTLength;
        //     uint IsInverse;
        // };
        int constantCount = 4;
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
            &m_bluesteinChirpRootSignature
        ));

        // Describe and create the compute pipeline state object (PSO).
        D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
        computePsoDesc.pRootSignature = m_bluesteinChirpRootSignature.Get();

        switch (dataType)
        {
            case MLOperatorTensorDataType::Float:
            computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(BluesteinChirp_Float32::g_BluesteinZChirp, sizeof(BluesteinChirp_Float32::g_BluesteinZChirp));
            break;

            case MLOperatorTensorDataType::Float16:
            {
                D3D12_FEATURE_DATA_D3D12_OPTIONS4 featureOptions = {};
                ORT_THROW_IF_FAILED(m_device->CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS4,
                    &featureOptions,
                    sizeof(featureOptions))
                );

                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(BluesteinChirp_Float16::g_BluesteinZChirp, sizeof(BluesteinChirp_Float16::g_BluesteinZChirp));
            }
            break;

            default:
            ORT_THROW_HR(E_INVALIDARG);
        }

        ORT_THROW_IF_FAILED(m_device->CreateComputePipelineState(&computePsoDesc, IID_ID3D12PipelineState, &m_bluesteinChirpPipelineState));
    }

    void PrepareStockhamFFT(MLOperatorTensorDataType dataType)
    {
        // Compute root signature.
        const int uavCount = 3;
        std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters;
        rootParameters.resize(uavCount + 1);

        for (uint32_t i = 0; i < uavCount; i++)
        {
            rootParameters[i].InitAsUnorderedAccessView(i);
        }

        int constantCount = 32;
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
            &m_stockhamFFTRootSignature
        ));

        // Describe and create the compute pipeline state object (PSO).
        D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
        computePsoDesc.pRootSignature = m_stockhamFFTRootSignature.Get();

        switch (dataType)
        {
            case MLOperatorTensorDataType::Float:
            computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(StockhamFFT_Float32::g_DFT, sizeof(StockhamFFT_Float32::g_DFT));
            break;

            case MLOperatorTensorDataType::Float16:
            {
                D3D12_FEATURE_DATA_D3D12_OPTIONS4 featureOptions = {};
                ORT_THROW_IF_FAILED(m_device->CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS4,
                    &featureOptions,
                    sizeof(featureOptions))
                );

                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(StockhamFFT_Float16::g_DFT, sizeof(StockhamFFT_Float16::g_DFT));
            }
            break;

            default:
            ORT_THROW_HR(E_INVALIDARG);
        }

        ORT_THROW_IF_FAILED(m_device->CreateComputePipelineState(&computePsoDesc, IID_ID3D12PipelineState, &m_stockhamFFTPipelineState));
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

            // Get the output tensor
            ComPtr<IMLOperatorTensor> outputTensor;
            context->GetOutputTensor(0, outputTensor.GetAddressOf());

            if (outputTensor->IsCpuData() || inputTensor->IsCpuData())
            {
                return E_UNEXPECTED;
            }

            ComPtr<IUnknown> executionObject;
            ComPtr<ID3D12GraphicsCommandList> commandList;
            context->GetExecutionInterface(executionObject.GetAddressOf());
            executionObject.As(&commandList);

            // Get the input and output shape sizes
            auto inputDims = GetTensorDimensions(inputTensor.Get());
            ML_CHECK_VALID_ARGUMENT(static_cast<size_t>(m_axis) < inputDims.size())
            auto outputDims = GetTensorDimensions(outputTensor.Get());
            ORT_THROW_HR_IF(E_FAIL, inputDims.size() != outputDims.size());

            ComPtr<IUnknown> inputUnknown;
            ComPtr<ID3D12Resource> inputResource;
            inputTensor->GetDataInterface(inputUnknown.GetAddressOf());
            ORT_THROW_IF_FAILED(inputUnknown.As(&inputResource));

            ComPtr<IUnknown> outputUnknown;
            ComPtr<ID3D12Resource> outputResource;
            outputTensor->GetDataInterface(outputUnknown.GetAddressOf());
            ORT_THROW_IF_FAILED(outputUnknown.As(&outputResource));

            // Get optional dft_length input
            uint32_t dftLength = inputDims[onnxruntime::narrow<size_t>(m_axis)];
            ComPtr<IMLOperatorTensor> dftLengthTensor;
            if (SUCCEEDED(context->GetInputTensor(1, &dftLengthTensor)) && dftLengthTensor != nullptr)
            {
                MLOperatorTensor tensor(dftLengthTensor.Get());
                dftLength = onnxruntime::narrow<uint32_t>(OperatorHelper::ReadScalarTensorCastToInt64(tensor));
            }

            return Compute(
                commandList.Get(),
                context,
                inputResource.Get(),
                inputDims,
                outputResource.Get(),
                outputDims,
                dftLength
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
        ID3D12Resource* outputResource,
        gsl::span<const uint32_t> outputDims,
        uint32_t dftLength
        )
    {
        try
        {
            auto dftParams = PrepareDFT(context, inputResource, inputDims, outputResource, outputDims, dftLength);

            switch (dftParams.Type)
            {
                case DFTType::Stockham:
                {
                    StockhamFFT(dftParams, m_isInverse, 0 /*chirpLength*/, 1 /*scale*/, commandList);
                    break;
                }
                case DFTType::BluesteinZChirp:
                    BluesteinZChirp(dftParams, m_isInverse,commandList);
                    break;
                default:
                    return E_NOTIMPL;
            }
        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }

    // { before_dft_axis, axis, after_dft_axis, real_or_complex }
    std::array<uint32_t, 4> GetReshapedDimensions(gsl::span<const uint32_t> dims, int64_t axis)
    {
        std::array<uint32_t, 4> reshapedDims = { 1, 1, 1, 1 };
        size_t reshapedIndex = 0;
        for (int i = 0; i < static_cast<int>(dims.size()) - 1; i++)
        {
            if (i == axis || i == (axis + 1))
            {
                reshapedIndex++;
            }

            reshapedDims[reshapedIndex] *= dims[i];
        }
        return reshapedDims;
    }

    void PrepareStockhamFFTParams(
        IMLOperatorKernelContext* context,
        ID3D12Resource* inputResource,
        gsl::span<const uint32_t> inputDims,
        ID3D12Resource* outputResource,
        gsl::span<const uint32_t> outputDims,
        uint32_t dftLength,
        int64_t inAxis,
        int64_t outAxis,
        StockhamParameters& params)
    {
        params = {};

        auto reshapedInputSize = GetReshapedDimensions(inputDims, inAxis);
        reshapedInputSize.back() = inputDims.back();
        auto reshapedOutputSize = GetReshapedDimensions(outputDims, outAxis);
        reshapedOutputSize.back() = outputDims.back();

        auto temporarySize = reshapedInputSize;
        // In the case where the dft length does not match the dft size, the temporary output should
        // match the dft length
        temporarySize[1] = dftLength;
        temporarySize.back() = reshapedOutputSize.back();
        auto temporaryBufferByteSize = sizeof(float) * ComputeElementCountFromDimensions(temporarySize);

        // Calculate elements and strides
        std::array<uint32_t, 4> reshapedInputStrides = { 1, 1, 1, 1 };
        std::array<uint32_t, 4> reshapedOutputStrides = { 1, 1, 1, 1 };
        std::array<uint32_t, 4> temporaryStrides = { 1, 1, 1, 1 };
        for (int i = static_cast<int>(reshapedInputSize.size()) - 2; i >= 0; i--)
        {
            reshapedInputStrides[i] = reshapedInputSize[i + 1] * reshapedInputStrides[i + 1];
            reshapedOutputStrides[i] = reshapedOutputSize[i + 1] * reshapedOutputStrides[i + 1];
            temporaryStrides[i] = temporarySize[i + 1] * temporaryStrides[i + 1];
        }

        bool doesTemporaryShapeMatchOutput = true;
        for (uint32_t i = 0; i < temporarySize.size(); i++)
        {
            doesTemporaryShapeMatchOutput &= (temporarySize[i] == reshapedOutputSize[i]);
            if (!doesTemporaryShapeMatchOutput)
            {
                break;
            }
        }

        // Calculate passes
        params.NumberOfPasses = static_cast<unsigned>(log2(dftLength));
        bool hasOnePass = params.NumberOfPasses == 1;
        bool hasOddPasses = params.NumberOfPasses % 2;
        bool hasEvenPasses = !hasOddPasses;

        // write directly input buffer to output buffer, dont create temps
        bool writeToOutput = hasOnePass;
        // First and final are input/output buffers, but all else ocillate between 2 temp buffers
        bool oscillateBetweenTwoTemporaries = !hasOnePass && (m_isOnesided || !doesTemporaryShapeMatchOutput);
        // First is input buffer, all else ocillate between temp and output, causing the final pass to write to the output buffer
        bool oscillateFirstOutputThenTemporary = hasOddPasses && (!m_isOnesided && doesTemporaryShapeMatchOutput);
        // First is input buffer, all else ocillate between output and temp, causing the final pass to write to the output buffer
        bool oscillateFirstTemporaryThenOutput = hasEvenPasses && (!m_isOnesided && doesTemporaryShapeMatchOutput);

        // Create the resource loop list
        // Add the input resource to the loop list
        params.ResourceLoopList.push_back({});
        params.ResourceLoopList.back().Resource = inputResource;
        params.ResourceLoopList.back().Sizes = reshapedInputSize;
        params.ResourceLoopList.back().Strides = reshapedInputStrides;

        // If 1 temporary should be placed first, or multiple temporaries, then
        // Add a temp in the list
        if (oscillateFirstTemporaryThenOutput || oscillateBetweenTwoTemporaries)
        {
            params.ResourceLoopList.push_back({});
            params.ResourceLoopList.back().Sizes = temporarySize;
            params.ResourceLoopList.back().Strides = temporaryStrides;

            auto& resource = params.ResourceLoopList.back().Resource;
            ORT_THROW_IF_FAILED(context->AllocateTemporaryData(temporaryBufferByteSize, &resource));
        }

        // If 2 temps, add another
        if (oscillateBetweenTwoTemporaries)
        {
            params.ResourceLoopList.push_back({});
            params.ResourceLoopList.back().Sizes = temporarySize;
            params.ResourceLoopList.back().Strides = temporaryStrides;

            auto& resource = params.ResourceLoopList.back().Resource;
            ORT_THROW_IF_FAILED(context->AllocateTemporaryData(temporaryBufferByteSize, &resource));
        }

        // Add output resource
        params.ResourceLoopList.push_back({});
        params.ResourceLoopList.back().Resource = outputResource;
        params.ResourceLoopList.back().Sizes = reshapedOutputSize;
        params.ResourceLoopList.back().Strides = reshapedOutputStrides;
        params.OutputIndex = static_cast<uint32_t>(params.ResourceLoopList.size() - 1);

        // Add the temporary after output incase of odd number of passes
        if (oscillateFirstOutputThenTemporary)
        {
            params.ResourceLoopList.push_back({});
            params.ResourceLoopList.back().Sizes = temporarySize;
            params.ResourceLoopList.back().Strides = temporaryStrides;

            auto& resource = params.ResourceLoopList.back().Resource;
            ORT_THROW_IF_FAILED(context->AllocateTemporaryData(temporaryBufferByteSize, &resource));
        }

        // Define the loop range
        if (writeToOutput) { params.LoopRange = { 0, 1, params.NumberOfPasses }; }
        if (oscillateBetweenTwoTemporaries) { params.LoopRange = { 1, 2, params.NumberOfPasses }; }
        if (oscillateFirstOutputThenTemporary) { params.LoopRange = { 1, 2, params.NumberOfPasses + 1 }; }
        if (oscillateFirstTemporaryThenOutput) { params.LoopRange = { 1, 2, params.NumberOfPasses + 1 }; }

        params.Window.Resource = nullptr;
        params.Window.Sizes = std::array<uint32_t, 4> {0, 0, 0, 0};
        params.Window.Strides = std::array<uint32_t, 4> {0, 0, 0, 0};
    }

    DFTParameters PrepareDFT(
        IMLOperatorKernelContext* context,
        ID3D12Resource* inputResource,
        gsl::span<const uint32_t> inputDims,
        ID3D12Resource* outputResource,
        gsl::span<const uint32_t> outputDims,
        uint32_t dftLength
        )
    {
        DFTParameters params = {};

        params.DFTLength = dftLength;
        params.StockhamParams = {};
        params.BluesteinZChirpParams = {};

        if (DFTHelpers::IsPowerOfTwo(params.DFTLength))
        {
            params.Type = DFTType::Stockham;
            PrepareStockhamFFTParams(
                context,
                inputResource,
                inputDims,
                outputResource,
                outputDims,
                dftLength,
                m_axis,
                m_axis,
                params.StockhamParams);
        }
        else
        {
            params.Type = DFTType::BluesteinZChirp;
            auto M = DFTHelpers::NextPowerOf2((2*dftLength) - 1);
            auto batchSize = inputDims[0];

            // Compute intermediate tensor strides
            params.BluesteinZChirpParams.ZChirp.Sizes = std::array<uint32_t, 4> { 1, 1, dftLength, 2 };
            params.BluesteinZChirpParams.ZChirp.Strides = std::array<uint32_t, 4> { dftLength * 2, dftLength * 2, 2, 1 };

            params.BluesteinZChirpParams.AFFT.Sizes = GetReshapedDimensions(inputDims, m_axis);
            params.BluesteinZChirpParams.AFFT.Sizes[1] = M;
            params.BluesteinZChirpParams.AFFT.Sizes.back() = 2;
            Dml::GetDescendingPackedStrides(params.BluesteinZChirpParams.AFFT.Sizes, params.BluesteinZChirpParams.AFFT.Strides);

            params.BluesteinZChirpParams.B.Sizes = std::array<uint32_t, 4> { 1, 1, M, 2 };
            params.BluesteinZChirpParams.B.Strides = std::array<uint32_t, 4> { M * 2, M * 2, 2, 1 };

            params.BluesteinZChirpParams.BFFT.Sizes = std::array<uint32_t, 4> { 1, 1, M, 2 };
            params.BluesteinZChirpParams.BFFT.Strides = std::array<uint32_t, 4> { M * 2, M * 2, 2, 1 };

            auto zChirpBufferByteSize = sizeof(float) * ComputeElementCountFromDimensions(params.BluesteinZChirpParams.ZChirp.Sizes);
            auto aIntermediateBufferByteSize = sizeof(float) * ComputeElementCountFromDimensions(params.BluesteinZChirpParams.AFFT.Sizes);
            auto bIntermediateBufferByteSize = sizeof(float) * ComputeElementCountFromDimensions(params.BluesteinZChirpParams.BFFT.Sizes);

            auto& zChirpResource = params.BluesteinZChirpParams.ZChirp.Resource;
            auto& aFFTResource = params.BluesteinZChirpParams.AFFT.Resource;
            auto& bResource = params.BluesteinZChirpParams.B.Resource;
            auto& bFFTResource = params.BluesteinZChirpParams.BFFT.Resource;
            ORT_THROW_IF_FAILED(context->AllocateTemporaryData(zChirpBufferByteSize, &zChirpResource));
            ORT_THROW_IF_FAILED(context->AllocateTemporaryData(aIntermediateBufferByteSize, &aFFTResource));
            ORT_THROW_IF_FAILED(context->AllocateTemporaryData(bIntermediateBufferByteSize, &bResource));
            ORT_THROW_IF_FAILED(context->AllocateTemporaryData(bIntermediateBufferByteSize, &bFFTResource));

            // The AFFT call takes input A, and produces output A_FFT.
            //
            // Input A: This is a pow-2 padded and chirp-weighted representation of the signal represented by "inputResource"
            // Therefore the dftLength is not correct for AFFT, it should be NextPowerOf2(2*dftLength-1)
            //
            // The weighted representation should be calculated by passing in the chirp to the dft (like a window function).
            // Padding should be handled by the shader.
            PrepareStockhamFFTParams(
                context,
                inputResource, inputDims,
                aFFTResource.Get(), params.BluesteinZChirpParams.AFFT.Sizes,
                M,
                m_axis,
                1,
                params.BluesteinZChirpParams.AFFTParams);
            params.BluesteinZChirpParams.AFFTParams.Window = params.BluesteinZChirpParams.ZChirp;


            // This shader will be used to calculate the inverse of the A_FFT, after complex multiplication with the B_FFT.
            // Therefore the window function logic shold hangle complex multiplication, and B_FTT should be used like a window function.
            PrepareStockhamFFTParams(
                context,
                aFFTResource.Get(), params.BluesteinZChirpParams.AFFT.Sizes,
                outputResource, outputDims,
                M,
                1,
                m_axis,
                params.BluesteinZChirpParams.AFFTInverseParams);
            // The BFFT Window is described with the reshaped sizes and strides, which is incompatible with the
            // window parameter expected by the stockham shader.
            // We need to reinterpret it with the same BFFT size and strides above to make it conform to the shape
            // expected by the shader.
            params.BluesteinZChirpParams.AFFTInverseParams.Window = params.BluesteinZChirpParams.BFFT;
            params.BluesteinZChirpParams.AFFTInverseParams.Window.Sizes = params.BluesteinZChirpParams.BFFT.Sizes;
            params.BluesteinZChirpParams.AFFTInverseParams.Window.Strides = params.BluesteinZChirpParams.BFFT.Strides;

            // The BFFT call takes input B, and produces output B_FFT.
            PrepareStockhamFFTParams(
                context,
                bResource.Get(), params.BluesteinZChirpParams.B.Sizes,
                bFFTResource.Get(), params.BluesteinZChirpParams.BFFT.Sizes,
                M,
                2,
                2,
                params.BluesteinZChirpParams.BFFTParams);
        }

        return params;
    }

    void BluesteinZChirp(const DFTParameters& dftParams, bool isInverse, ID3D12GraphicsCommandList* commandList)
    {
        const auto& bluesteinZChirpParams = dftParams.BluesteinZChirpParams;

        // Get input and output resources
        auto inputResource =  bluesteinZChirpParams.AFFTParams.ResourceLoopList.front().Resource.Get();
        auto outputResource = bluesteinZChirpParams.AFFTInverseParams.ResourceLoopList[bluesteinZChirpParams.AFFTInverseParams.OutputIndex].Resource.Get();
        auto zChirpResource = bluesteinZChirpParams.ZChirp.Resource.Get();
        auto aFFTResource = bluesteinZChirpParams.AFFT.Resource.Get();
        auto bResource = bluesteinZChirpParams.B.Resource.Get();
        auto bFFTResource = bluesteinZChirpParams.BFFT.Resource.Get();

        // Transition resources from common to UAV state
        D3D12_RESOURCE_BARRIER barriers[2];

        barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
            inputResource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

        barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
            outputResource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

        commandList->ResourceBarrier(2, barriers);

        // Set the root signature and pipeline state
        commandList->SetComputeRootSignature(m_bluesteinChirpRootSignature.Get());
        commandList->SetPipelineState(m_bluesteinChirpPipelineState.Get());

        // Create ZChirp and B Tensors
        BluesteinZChirpShaderConstants constants = {};
        constants.DFTLength = dftParams.DFTLength;
        constants.IsInverse = isInverse;

        auto totalElementCount = ComputeElementCountFromDimensions(bluesteinZChirpParams.B.Sizes);
        constants.ElementCount = totalElementCount / bluesteinZChirpParams.B.Sizes[3];

        std::array<ID3D12Resource*, 2> uav_resources = { zChirpResource, bResource };
        Dispatch(uav_resources, constants, commandList);

        DFTParameters fft_params = {};
        fft_params.Type = DFTType::Stockham;
        fft_params.BluesteinZChirpParams = {};
        fft_params.DFTLength = bluesteinZChirpParams.AFFT.Sizes[1];

        // Create BFFT Tensors
        fft_params.StockhamParams = bluesteinZChirpParams.BFFTParams;
        StockhamFFT(fft_params, false, 0 /*chirpLength*/, 1 /*scale*/, commandList);

        // Create AFFT Tensors
        fft_params.StockhamParams = bluesteinZChirpParams.AFFTParams;
        StockhamFFT(fft_params, false, 0 /*chirpLength*/, 1 /*scale*/, commandList);

        // Should include the BFFT tensor as the window function
        fft_params.StockhamParams = bluesteinZChirpParams.AFFTInverseParams;
        float chirpLength = static_cast<float>(bluesteinZChirpParams.ZChirp.Sizes[2]);
        chirpLength *= (m_isInverse ? 1 : -1);
        float scale = isInverse ? 1.f / dftParams.DFTLength : 1.f;
        StockhamFFT(fft_params, true,  chirpLength, scale, commandList);

        // Transition resources to common state
        barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
                inputResource,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
                );

        barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
                outputResource,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
                );

        commandList->ResourceBarrier(2, barriers);
    }

    void StockhamFFT(
        const DFTParameters& dftParams,
        bool isInverse,
        float chirpLength,
        float scale,
        ID3D12GraphicsCommandList* commandList)
    {
        const auto& stockhamParams = dftParams.StockhamParams;

        // Create resource loop list
        const auto& loopList = stockhamParams.ResourceLoopList;

        // Get input and output resources
        auto inputResource = loopList[0].Resource.Get();
        auto outputResource = loopList[stockhamParams.OutputIndex].Resource.Get();
        auto windowResource = dftParams.StockhamParams.Window.Resource.Get();

        // Transition resources from common to UAV state
        D3D12_RESOURCE_BARRIER barriers[2];

        barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
            inputResource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

        barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
            outputResource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

        commandList->ResourceBarrier(2, barriers);

        // Set the root signature and pipeline state
        commandList->SetComputeRootSignature(m_stockhamFFTRootSignature.Get());
        commandList->SetPipelineState(m_stockhamFFTPipelineState.Get());

        // Each iteration of the below loop represents 1 level in the Stockham DFT
        // Dispatch in a loop
        DFTShaderConstants constants = {};
        constants.DFTLength = dftParams.DFTLength;
        constants.DFTIteration = 0;
        constants.IsInverse = isInverse;
        std::copy(dftParams.StockhamParams.Window.Sizes.begin(), dftParams.StockhamParams.Window.Sizes.end(), constants.WindowSizes);
        std::copy(dftParams.StockhamParams.Window.Strides.begin(), dftParams.StockhamParams.Window.Strides.end(), constants.WindowStrides);

        for (unsigned index = 0; index < stockhamParams.NumberOfPasses; index++)
        {
            auto inIdx = stockhamParams.LoopRange.CalculateIndex(index);
            auto outIdx = stockhamParams.LoopRange.CalculateIndex(index + 1);

            auto in = loopList[inIdx].Resource.Get();
            std::copy(loopList[inIdx].Sizes.begin(), loopList[inIdx].Sizes.end(), constants.InputSizes);
            std::copy(loopList[inIdx].Strides.begin(), loopList[inIdx].Strides.end(), constants.InputStrides);

            auto out = loopList[outIdx].Resource.Get();
            std::copy(loopList[outIdx].Sizes.begin(), loopList[outIdx].Sizes.end(), constants.OutputSizes);
            std::copy(loopList[outIdx].Strides.begin(), loopList[outIdx].Strides.end(), constants.OutputStrides);

            auto isFirstPass = (index == 0);
            auto isLastPass = (index == stockhamParams.NumberOfPasses - 1);
            auto isLastInversePass = isLastPass && isInverse;
            auto dftLength = 1 << stockhamParams.NumberOfPasses;
            constants.Scale = isLastInversePass ? (scale / dftLength) : 1;

            auto totalElementCount = ComputeElementCountFromDimensions(constants.OutputSizes);
            constants.ElementCount = totalElementCount / constants.OutputSizes[3];
            constants.DFTIteration = index + 1;
            constants.ChirpLength = isLastPass ? chirpLength : 0;
            constants.HasWindow = isFirstPass && windowResource != nullptr;
            auto window = constants.HasWindow ? windowResource : out;
            std::array<ID3D12Resource*, 3> uav_resources = { in, out, window };
            Dispatch(uav_resources, constants, commandList);
        }

        // Transition resources to common state
        barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
                inputResource,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
                );

        barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
                outputResource,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
                );

        commandList->ResourceBarrier(2, barriers);
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

            DFTHelpers::GetNextDispatchSize(
                pendingElementCount,
                1,
                64,
                dispatchSizeX,
                pendingElementCount
            );

            // Set root constants
            commandList->SetComputeRoot32BitConstants(
                TSize, // root parameter index
                32, // Constant count
                &constants,
                0 // offset
            );

            commandList->Dispatch(dispatchSizeX, 1, 1);
        }

        commandList->ResourceBarrier(2, uav_barriers);
    }
};

struct DFTShapeInferrer : public WRL::Base<IMLOperatorShapeInferrer>
{
    STDMETHOD(InferOutputShapes)(IMLOperatorShapeInferenceContext* context) noexcept
    {
        try
        {
            int64_t axis;
            ORT_THROW_IF_FAILED(context->GetAttribute("axis", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&axis)));
            int64_t isInverseInt;
            ORT_THROW_IF_FAILED(context->GetAttribute("inverse", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&isInverseInt)));
            int64_t isOnesidedInt;
            ORT_THROW_IF_FAILED(context->GetAttribute("onesided", MLOperatorAttributeType::Int, 1, sizeof(int64_t), reinterpret_cast<void*>(&isOnesidedInt)));
            bool isOnesided = static_cast<bool>(isOnesidedInt);
            bool isInverse = static_cast<bool>(isInverseInt);

            if (isInverse && isOnesided)
            {
                throw new std::exception("onesided and inverse attributes cannot be enabled at the same time");
            }

            uint32_t rank;
            ORT_THROW_IF_FAILED(context->GetInputTensorDimensionCount(0, &rank));
            if (rank == 0)
            {
                // If no shape is available for the input, skip shape inference...
                throw;
            }

            auto axisIdx = OperatorHelper::HandleNegativeAxis(static_cast<int32_t>(axis), rank);

            // In general the output shape will match the input shape exactly
            // So initialize the output shape with the input shape
            std::vector<uint32_t> inputDims(rank);
            ORT_THROW_IF_FAILED(context->GetInputTensorShape(0, rank, inputDims.data()));
            auto outputDims = inputDims;
            // The last dimension of the output shape is always 2.
            // It corresponds to the real and imaginary parts of the DFT output.
            outputDims.back() = 2;

            if (context->IsInputValid(1))
            {
                // If dft_length is specified, then we should honor the shape.
                // If onesided this will be adjusted later on.
                ComPtr<IMLOperatorShapeInferenceContextPrivate> contextPrivate;
                ORT_THROW_IF_FAILED(context->QueryInterface(IID_PPV_ARGS(&contextPrivate)));
                ComPtr<IMLOperatorTensor> dftLengthTensor;
                ORT_THROW_IF_FAILED(contextPrivate->GetConstantInputTensor(1, &dftLengthTensor));
                MLOperatorTensor tensor(dftLengthTensor.Get());
                auto dftLength = onnxruntime::narrow<uint32_t>(OperatorHelper::ReadScalarTensorCastToInt64(tensor));
                outputDims[axisIdx] = dftLength;
            }

            // When DFT is onesided, the output shape is half the size of the input shape
            // along the specified axis.
            if (isOnesided)
            {
                auto axisDimension = outputDims.at(axisIdx);
                // We need to update the output shape dimension along the specified axis,
                // but sometimes the dimension will be a free dimension or be otherwise unset.
                // Only perform inference when a input dimension value exists.
                auto originalSignalSize = axisDimension;
                auto halfSignalSize = (originalSignalSize >> 1) + 1;
                outputDims.at(axisIdx) = halfSignalSize;
            }

            ORT_THROW_IF_FAILED(context->SetOutputTensorShape(0, rank, outputDims.data()));
        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }
};

class GpuDFTOperatorFactory : public WRL::Base<IMLOperatorKernelFactory>
{
public:
    STDMETHOD(CreateKernel)(
        IMLOperatorKernelCreationContext* context,
        IMLOperatorKernel** kernel)
    {
        try
        {
            auto dftOperator = wil::MakeOrThrow<GpuDFTOperator>(context);
            dftOperator.CopyTo(kernel);
            return S_OK;
        }
        catch (...)
        {
            return E_FAIL;
        }
    }

    static void RegisterDFTKernel(IMLOperatorRegistry* registry)
    {
        MLOperatorKernelDescription kernelDescription = {};
        kernelDescription.domain = "";
        kernelDescription.name = "DFT";
        kernelDescription.minimumOperatorSetVersion = 17;
        kernelDescription.executionType = MLOperatorExecutionType::D3D12;

        // T1: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
        MLOperatorEdgeTypeConstrant t1Constraint;
        t1Constraint.typeLabel = "T1";
        std::vector<MLOperatorEdgeDescription> t1AllowedEdges
        {
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float16 },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float },
            //MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Double },
        };
        t1Constraint.allowedTypes = t1AllowedEdges.data();
        t1Constraint.allowedTypeCount = static_cast<uint32_t>(t1AllowedEdges.size());

        // T2 : tensor(int32), tensor(int64)
        MLOperatorEdgeTypeConstrant t2Constraint;
        t2Constraint.typeLabel = "T2";
        std::vector<MLOperatorEdgeDescription> t2AllowedEdges
        {
          //  MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Int32 },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Int64 },
        };
        t2Constraint.allowedTypes = t2AllowedEdges.data();
        t2Constraint.allowedTypeCount = static_cast<uint32_t>(t2AllowedEdges.size());

        std::vector<MLOperatorEdgeTypeConstrant> typeConstraints{ t1Constraint, t2Constraint };
        kernelDescription.typeConstraints = typeConstraints.data();
        kernelDescription.typeConstraintCount = static_cast<uint32_t>(typeConstraints.size());

        MLOperatorAttributeNameValue axisAttributeValue;
        axisAttributeValue.name = "axis";
        axisAttributeValue.type = MLOperatorAttributeType::Int;
        axisAttributeValue.valueCount = 1;
        static const int64_t axis[] = { 1 };
        axisAttributeValue.ints = axis;

        MLOperatorAttributeNameValue inverseAttributeValue;
        inverseAttributeValue.name = "inverse";
        inverseAttributeValue.type = MLOperatorAttributeType::Int;
        inverseAttributeValue.valueCount = 1;
        static const int64_t inverse[] = { 0 };
        inverseAttributeValue.ints = inverse;

        MLOperatorAttributeNameValue onesidedAttributeValue;
        onesidedAttributeValue.name = "onesided";
        onesidedAttributeValue.type = MLOperatorAttributeType::Int;
        onesidedAttributeValue.valueCount = 1;
        static const int64_t onesided[] = { 0 };
        onesidedAttributeValue.ints = onesided;

        std::vector<MLOperatorAttributeNameValue> attributeDefaultValues{
            axisAttributeValue,
            inverseAttributeValue,
            onesidedAttributeValue
        };

        kernelDescription.defaultAttributes = attributeDefaultValues.data();
        kernelDescription.defaultAttributeCount = static_cast<uint32_t>(attributeDefaultValues.size());
        kernelDescription.options = MLOperatorKernelOptions::None;
        kernelDescription.executionOptions = 0;

        auto shareInferrer = wil::MakeOrThrow<DFTShapeInferrer>();
        auto factory = wil::MakeOrThrow<GpuDFTOperatorFactory>();

        std::array<uint32_t, 1> requiredConstantCpuInputs = { 1 };

        ComPtr<IMLOperatorRegistryPrivate> registryPrivate;
        ORT_THROW_IF_FAILED(registry->QueryInterface(IID_PPV_ARGS(&registryPrivate)));

        ORT_THROW_IF_FAILED(registryPrivate->RegisterOperatorKernel(
            &kernelDescription,
            factory.Get(),
            shareInferrer.Get(),
            nullptr,
            false, // isInternalOperator
            false, // supportsGraph
            nullptr,
            requiredConstantCpuInputs.data(),
            static_cast<uint32_t>(requiredConstantCpuInputs.size())
        ));

    }
};
