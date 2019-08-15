//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#pragma once

#define ML_CHECK_VALID_ARGUMENT(x, ...)\
  {\
    if ((x) == false) {\
      THROW_HR(E_INVALIDARG);\
    }\
  }

#define ML_INVALID_ARGUMENT(msg)\
      THROW_HR(E_INVALIDARG);\

#define ML_CHECK_HRESULT(hr, ...)\
  {\
    if (FAILED(hr)) {\
      THROW_HR(E_INVALIDARG);\
    }\
  }

namespace OperatorHelper
{
    enum TensorAxis { N, C, H, W, DoNotCoerce = UINT_MAX };
    enum BroadcastMode { NoBroadcast, UnidirectionalBroadcast, MultidirectionalBroadcast };

    using DimensionType = uint32_t;

    static const uint32_t NchwDimensionCount = 4; // Some operators only handle 4 dimensions.
    static const uint32_t NchwSpatialDimensionCount = 2;
    static const uint32_t NcdhwDimensionCount = 5;
    static const uint32_t NcdhwSpatialDimensionCount = 3;

} // namespace OperatorHelper
