//
//  Copyright (c) Microsoft Corporation.  All rights reserved.
//

#pragma once

namespace Windows::AI::MachineLearning::Internal
{
    const UINT IMG_TENSOR_DIMENSION_COUNT_MAX = 4; // NCHW format

    enum IMG_TENSOR_DATA_TYPE
    {
        IMG_TENSOR_DATA_TYPE_FLOAT32,
        IMG_TENSOR_DATA_TYPE_FLOAT16,
        IMG_TENSOR_DATA_TYPE_UINT32,
        IMG_TENSOR_DATA_TYPE_UINT16,
        IMG_TENSOR_DATA_TYPE_UINT8,
        IMG_TENSOR_DATA_TYPE_INT32,
        IMG_TENSOR_DATA_TYPE_INT16,
        IMG_TENSOR_DATA_TYPE_INT8,
        IMG_TENSOR_DATA_TYPE_COUNT
    };

    enum IMG_TENSOR_CHANNEL_TYPE
    {
        IMG_TENSOR_CHANNEL_TYPE_RGB_8,
        IMG_TENSOR_CHANNEL_TYPE_BGR_8,
        IMG_TENSOR_CHANNEL_TYPE_GRAY_8,
        IMG_TENSOR_CHANNEL_TYPE_COUNT
    };

    struct IMG_TENSOR_DESC
    {
        IMG_TENSOR_DATA_TYPE dataType;
        IMG_TENSOR_CHANNEL_TYPE channelType;
        UINT sizes[IMG_TENSOR_DIMENSION_COUNT_MAX];
    };
}