//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#include "precomp.h"

namespace Dml
{

class DmlOperatorUpsample2d : public DmlOperator, public Upsample2dHelper
{
    static DML_INTERPOLATION_MODE StringToUpsampleMode(const std::string& mode)
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
    DmlOperatorUpsample2d(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        Upsample2dHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        DmlOperator::Initialize(kernelCreationContext);

        std::string mode = kernelCreationContext.GetOptionalAttribute<std::string>(AttrName::Mode, "NEAREST");
        DML_INTERPOLATION_MODE interpolationMode = StringToUpsampleMode(mode);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        ML_CHECK_VALID_ARGUMENT(inputDescs.size() >= 1);
        ML_CHECK_VALID_ARGUMENT(outputDescs.size() >= 1);

        DML_UPSAMPLE_2D_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.ScaleSize = DML_SIZE_2D{ static_cast<uint32_t>(m_scaleSizeW), static_cast<uint32_t>(m_scaleSizeH) };
        operatorDesc.InterpolationMode = interpolationMode;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_UPSAMPLE_2D, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext); // TODO(jeffbloo): hookup "SetAsFactory" with new DmlOperator/API types
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Upsample, DmlOperatorUpsample2d);

} // namespace Dml
