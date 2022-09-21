// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>

#define ML_CHECK_VALID_ARGUMENT(x, ...)\
    {\
        if ((x) == false)\
        {\
            ORT_THROW_HR(E_INVALIDARG);\
        }\
    }

#define ML_INVALID_ARGUMENT(msg)\
    ORT_THROW_HR(E_INVALIDARG);\

#define ML_CHECK_HRESULT(hr, ...)\
    {\
        if (FAILED(hr))\
        {\
            ORT_THROW_HR(E_INVALIDARG);\
        }\
    }

namespace OperatorHelper
{
    // Clamp the input value to the maximum range of the output, before casting to the output type.
    //
    // e.g. int32(300) would yield int8(255) rather than int8(44).
    //      float32(-42) would yield uint32(0) rather than a huge positive number.
    template<typename OutputType, typename InputType> OutputType clamp_cast(InputType input)
    {
        // Determine the larger type to decide which numeric limits to clamp to.
        using InputLimits = std::numeric_limits<InputType>;
        using OutputLimits = std::numeric_limits<OutputType>;
        constexpr int inputMaxDigits = std::max(InputLimits::max_exponent, InputLimits::digits);
        constexpr int outputMaxDigits = std::max(OutputLimits::max_exponent, OutputLimits::digits);
        constexpr bool isEitherTypeUnsigned = std::is_unsigned_v<InputType> || std::is_unsigned_v<OutputType>;
        constexpr bool isOutputTypeLarger = outputMaxDigits > inputMaxDigits;

        InputType lowestValue  = isEitherTypeUnsigned ? static_cast<InputType>(0) :
                                 isOutputTypeLarger ? InputLimits::lowest() :
                                 static_cast<InputType>(OutputLimits::lowest());
        InputType highestValue = isOutputTypeLarger ? InputLimits::max() :
                                 static_cast<InputType>(OutputLimits::max());

        return static_cast<OutputType>(std::clamp<InputType>(input, lowestValue, highestValue));
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
