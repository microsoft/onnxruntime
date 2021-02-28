// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorRegionOfInterestPooling : public DmlOperator, public RoiPoolingHelper
{
public:
    using Self = DmlOperatorRegionOfInterestPooling;

    DmlOperatorRegionOfInterestPooling(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperator(kernelInfo),
        RoiPoolingHelper(kernelInfo, kernelInfo.GetTensorShapeDescription()),
        m_spatialScale(kernelInfo.GetOptionalAttribute<float>(AttrName::SpatialScale, 1.0f))
    {
        DmlOperator::Initialize(kernelInfo);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_ROI_POOLING_OPERATOR_DESC poolingDesc = {};
        poolingDesc.InputTensor = &inputDescs[0];
        poolingDesc.ROITensor = &inputDescs[1];
        poolingDesc.OutputTensor = &outputDescs[0];
        poolingDesc.SpatialScale = m_spatialScale;
        poolingDesc.PooledSize = { m_outputSizeH, m_outputSizeW };

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ROI_POOLING, &poolingDesc };

        SetDmlOperatorDesc(opDesc, kernelInfo);
    }

private:
    float m_spatialScale = 1.0f;
};

DML_OP_DEFINE_CREATION_FUNCTION(MaxRoiPool, DmlOperatorRegionOfInterestPooling);

} // namespace Dml
