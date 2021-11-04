// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ML_CHECK_VALID_ARGUMENT(x, ...)\
  {\
    if ((x) == false) {\
      ORT_THROW_HR(E_INVALIDARG);\
    }\
  }

#define ML_INVALID_ARGUMENT(msg)\
      ORT_THROW_HR(E_INVALIDARG);\

#define ML_CHECK_HRESULT(hr, ...)\
  {\
    if (FAILED(hr)) {\
      ORT_THROW_HR(E_INVALIDARG);\
    }\
  }

namespace OperatorHelper
{
    template<typename T, typename I> T clamp_cast(I input)
    {
      return static_cast<T>(std::clamp<I>(input, std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()));
    }
    enum TensorAxis { N, C, H, W, DoNotCoerce = INT_MAX, LeftAligned = INT_MAX, RightAligned = INT_MIN, NoPlacementAdjustment = 0 };
    enum BroadcastMode { NoBroadcast, UnidirectionalBroadcast, MultidirectionalBroadcast };

    using DimensionType = uint32_t;

    static constexpr uint32_t NchwDimensionCount = 4; // Some operators only handle 4 dimensions.
    static constexpr uint32_t NchwSpatialDimensionCount = 2;
    static constexpr uint32_t NcdhwDimensionCount = 5;
    static constexpr uint32_t NcdhwSpatialDimensionCount = 3;
    static constexpr uint32_t NonspatialDimensionCount = 2; // The batch and channel dimensions of NCW, NCHW, NCDHW....

} // namespace OperatorHelper
