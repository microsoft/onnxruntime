//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#include "precomp.h"

namespace Dml
{

class DmlOperatorResize : public DmlOperator, public ResizeHelper
{
    static DML_INTERPOLATION_MODE StringToResizeMode(const std::string& mode)
    {
        // The ONNX modes are "nearest" and "linear."  Other modes exist for compatibility,
        // since Winml supported them in the past.
        if (mode == "NEAREST" || mode == "nearest" || mode == "nn" || mode == "NN") {
            return DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR;
        }
        else if (mode == "BILINEAR" || mode == "bilinear" || mode == "linear")
        {
            return DML_INTERPOLATION_MODE_LINEAR;
        }
        else
        {
            ML_INVALID_ARGUMENT("Unknown upsampling mode.");
        }
    }

public:
    // Resample a multidimensional image to a new size.
    DmlOperatorResize(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext), 
        ResizeHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 2, "Resize expects 2 input tensors.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "Resize expects 1 output tensor.");

        std::vector<std::optional<uint32_t>> inputIndices = { 0 }; // Use only the first tensor. The second tensor is CPU based and should not be passed to Resize.
        std::vector<std::optional<uint32_t>> outputIndices = { 0 };
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices);

        // If the output tensor dimension count was right-aligned to a larger size,
        // then ensure that scales has the same count as the tensor rank by inserting
        // leading ones, since DirectML requires the scales to have the same count.
        auto paddedScales = m_scales;
        if (m_outputTensorDescs[0].GetDimensionCount() > m_dimCount)
        {
            paddedScales.insert(paddedScales.begin(), m_outputTensorDescs[0].GetDimensionCount() - m_dimCount, 1.0f);
        }

        std::string mode = kernelCreationContext.GetOptionalAttribute<std::string>(AttrName::Mode, "NEAREST");

        // Create the operator with new shape after calling UpdateShape.
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_RESAMPLE_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.InterpolationMode = StringToResizeMode(mode);
        operatorDesc.Scales = paddedScales.data();
        operatorDesc.ScaleCount = gsl::narrow_cast<uint32_t>(paddedScales.size());

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_RESAMPLE, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Resize, DmlOperatorResize);

} // namespace Dml
