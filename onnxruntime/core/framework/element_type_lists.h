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

using AllFixedSizeExceptHalfIRv4 =
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

using AllFixedSizeExceptHalf = AllFixedSizeExceptHalfIRv4;

using AllFixedSizeIRv4 =
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

#if !defined(DISABLE_FLOAT8_TYPES)
using AllFixedSizeIRv9 =
    boost::mp11::mp_push_back<
        AllFixedSizeIRv4,
        Float8E4M3FN,
        Float8E4M3FNUZ,
        Float8E5M2,
        Float8E5M2FNUZ>;
#else
using AllFixedSizeIRv9 = AllFixedSizeIRv4;
#endif

using AllFixedSize = AllFixedSizeIRv4;

using AllIRv4 =
    boost::mp11::mp_push_back<
        AllFixedSizeIRv4,
        std::string>;

#if !defined(DISABLE_FLOAT8_TYPES)
using AllIRv9 =
    boost::mp11::mp_push_back<
        AllIRv4,
        Float8E4M3FN,
        Float8E4M3FNUZ,
        Float8E5M2,
        Float8E5M2FNUZ>;

#else
using AllIRv9 = AllIRv4;
#endif

using All = AllIRv4;

#if !defined(DISABLE_FLOAT8_TYPES)
using AllFloat8 =
    TypeList<
        Float8E4M3FN,
        Float8E4M3FNUZ,
        Float8E5M2,
        Float8E5M2FNUZ>;
#endif

using AllIeeeFloat =
    TypeList<
        float,
        double,
        MLFloat16>;

using AllNumericIRv4 =
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

using AllNumeric = AllNumericIRv4;

}  // namespace element_type_lists

}  // namespace onnxruntime
