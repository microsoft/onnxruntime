// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

#include "boost/mp11.hpp"

#include "core/common/type_list.h"
#include "core/framework/float8.h"
#include "core/framework/float16.h"

namespace onnxruntime {

// Contains type lists grouping various ORT element types.
// Element type refers to the element type of a Tensor, Sequence, etc.
namespace element_type_lists {

using AllFixedSizeExceptHalfIR4 =
    TypeList<
        float,
        double,
        int64_t,
        uint64_t,
        int32_t,
        uint32_t,
        int16_t,
        uint16_t,
        int8_t,
        uint8_t,
        bool>;

using AllFixedSizeExceptHalf = AllFixedSizeExceptHalfIR4;

using AllFixedSizeIR4 =
    TypeList<
        float,
        double,
        int64_t,
        uint64_t,
        int32_t,
        uint32_t,
        int16_t,
        uint16_t,
        int8_t,
        uint8_t,
        MLFloat16,
        BFloat16,
        bool>;

using AllFixedSizeIR9 =
    boost::mp11::mp_push_back<
        AllFixedSizeIR4,
        Float8E4M3FN,
        Float8E4M3FNUZ,
        Float8E5M2,
        Float8E5M2FNUZ>;

using AllFixedSize = AllFixedSizeIR4;

using AllIR4 =
    boost::mp11::mp_push_back<
        AllFixedSizeIR4,
        std::string>;

using AllIR9 =
    boost::mp11::mp_push_back<
        AllIR4,
        Float8E4M3FN,
        Float8E4M3FNUZ,
        Float8E5M2,
        Float8E5M2FNUZ>;

using All = AllIR4;

using AllFloat8 =
    TypeList<
        Float8E4M3FN,
        Float8E4M3FNUZ,
        Float8E5M2,
        Float8E5M2FNUZ>;

using AllIeeeFloat = 
    TypeList<
        float,
        double,
        MLFloat16>;

using AllNumericIR4 =
    TypeList<
        float,
        double,
        int64_t,
        uint64_t,
        int32_t,
        uint32_t,
        int16_t,
        uint16_t,
        int8_t,
        uint8_t,
        MLFloat16,
        BFloat16>;

using AllNumeric = AllNumericIR4;

}  // namespace element_type_lists

}  // namespace onnxruntime
