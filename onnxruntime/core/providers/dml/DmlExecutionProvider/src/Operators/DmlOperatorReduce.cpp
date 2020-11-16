// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorReduce : public DmlOperator, public ReduceHelperBase
{
public:
    DmlOperatorReduce(
        const MLOperatorKernelCreationContext& kernelInfo,
        DML_REDUCE_FUNCTION function
        )
    :   DmlOperator(kernelInfo),
        ReduceHelperBase(kernelInfo,
                         kernelInfo.GetTensorShapeDescription(),
                        (function != DML_REDUCE_FUNCTION_ARGMAX && function != DML_REDUCE_FUNCTION_ARGMIN))
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);
        DmlOperator::Initialize(kernelInfo);

        std::vector<uint32_t> dmlAxes;
        std::vector<DimensionType> reducedDims = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(0);
        int dimOffset = gsl::narrow_cast<int>(m_inputTensorDescs[0].GetDimensionCount() - reducedDims.size());
        for (auto& dim : m_axes)
        {
            assert(dim < static_cast<int32_t>(reducedDims.size())); // ReduceHelperBase already validated this.
            reducedDims[dim] = 1;
            dmlAxes.push_back(static_cast<uint32_t>(dim + dimOffset));
        }

        if (!m_keepDims)
        {
            // DML doesn't know about keepDim and always assume the dim is preserved after reduce.
            // So if m_keepDims is false, the ONNX output dim is different than DML tensor desc dim.
            // ReduceSum example:
            // input dims: {3, 2, 2}
            // axes: 1
            // keepDims: 0
            // 
            // the ONNX output expect to be of dim {3, 2}, while DML expect the output tensor desc
            // dim to be {3, 1, 2}.
            //

            m_outputTensorDescs[0] = CreateTensorDescFromOutput(
                kernelInfo, 
                0, 
                TensorAxis::DoNotCoerce, 
                TensorAxis::W, 
                TensorAxis::RightAligned,
                reducedDims);
        }

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        // Zero the output tensor's memory for ArgMin & ArgMax, which produce INT64 output.
        if (function == DML_REDUCE_FUNCTION_ARGMAX)
        {
            DML_ARGMAX_OPERATOR_DESC argmaxDesc;
            argmaxDesc.AxisDirection = static_cast<DML_AXIS_DIRECTION>(m_selectLastIndex);
            argmaxDesc.InputTensor = inputDescs.data();
            argmaxDesc.OutputTensor = outputDescs.data();
            argmaxDesc.Axes = dmlAxes.data();
            argmaxDesc.AxisCount = gsl::narrow_cast<uint32_t>(dmlAxes.size());

            // If the 64-bit tensors were remapped to 32-bit, then we need to clear the upper 32-bits
            // of each element. If the device directly supports 64-bit elements, then no need.
            DmlOperator::Remap64bitDmlDataTypesTo32bitIfNeeded();
            if (m_outputTensorDescs[0].WasRemapped64bitTo32bit())
            {
                m_zeroOperator = InitializeZeroInt64Tensor(m_outputTensorDescs[0].GetBufferSizeInBytes());
            }
            
            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ARGMAX, &argmaxDesc };
            SetDmlOperatorDesc(opDesc, kernelInfo);
        }
        else if (function == DML_REDUCE_FUNCTION_ARGMIN)
        {
            DML_ARGMIN_OPERATOR_DESC argminDesc;
            argminDesc.AxisDirection = static_cast<DML_AXIS_DIRECTION>(m_selectLastIndex);
            argminDesc.InputTensor = inputDescs.data();
            argminDesc.OutputTensor = outputDescs.data();
            argminDesc.Axes = dmlAxes.data();
            argminDesc.AxisCount = gsl::narrow_cast<uint32_t>(dmlAxes.size());

            // If the 64-bit tensors were remapped to 32-bit, then we need to clear the upper 32-bits
            // of each element. If the device directly supports 64-bit elements, then no need.
            DmlOperator::Remap64bitDmlDataTypesTo32bitIfNeeded();
            if (m_outputTensorDescs[0].WasRemapped64bitTo32bit())
            {
                m_zeroOperator = InitializeZeroInt64Tensor(m_outputTensorDescs[0].GetBufferSizeInBytes());
            }

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ARGMIN, &argminDesc };
            SetDmlOperatorDesc(opDesc, kernelInfo);
        }
        else
        {
            DML_REDUCE_OPERATOR_DESC reduceDesc = {};
            reduceDesc.InputTensor = inputDescs.data();
            reduceDesc.OutputTensor = outputDescs.data();
            reduceDesc.Function = function;
            reduceDesc.Axes = dmlAxes.data();
            reduceDesc.AxisCount = gsl::narrow_cast<uint32_t>(dmlAxes.size());

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_REDUCE, &reduceDesc };
            SetDmlOperatorDesc(opDesc, kernelInfo);
        }
    }

    void Compute(const MLOperatorKernelContext& kernelContext) override
    {
        std::vector<IMLOperatorTensor*> inputTensors = GetInputTensorsForExecute(kernelContext);
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensorsForExecute(kernelContext);

        if (m_zeroOperator)
        {
            ExecuteZeroInt64Tensor(m_zeroOperator.Get(), outputTensors[0]);
        }

        THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
            m_compiledOperator.Get(),
            m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
            gsl::make_span(inputTensors),
            gsl::make_span(outputTensors)));
    }

private:
    ComPtr<IDMLCompiledOperator> m_zeroOperator;
};

// A specific type of operation for registration.
template <DML_REDUCE_FUNCTION Function>
class DmlOperatorReduceTemplate : public DmlOperatorReduce
{
public:
    DmlOperatorReduceTemplate(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperatorReduce(kernelInfo, Function)
    {
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(ReduceSum,       DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_SUM>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceMean,      DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_AVERAGE>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceProd,      DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_MULTIPLY>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceLogSum,    DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_LOG_SUM>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceLogSumExp, DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_LOG_SUM_EXP>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceSumSquare, DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_SUM_SQUARE>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceL1,        DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_L1>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceL2,        DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_L2>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceMax,       DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_MAX>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceMin,       DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_MIN>);
DML_OP_DEFINE_CREATION_FUNCTION(ArgMax,          DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_ARGMAX>);
DML_OP_DEFINE_CREATION_FUNCTION(ArgMin,          DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_ARGMIN>);

} // namespace Dml
