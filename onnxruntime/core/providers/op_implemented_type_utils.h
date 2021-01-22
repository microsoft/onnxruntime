// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "boost/mp11.hpp"

#include "core/common/type_list.h"

namespace onnxruntime {

enum class OpArgType {
  Input,
  Output
};
using OpArgIndex = size_t;

// a tag type to identify the target of the implemented type lists
template <typename Op, OpArgType ArgType, OpArgIndex ArgIndex>
struct OpArgTag {};

// primary templates

// default implemented types - specialize this to specify the default set of implemented types
template <typename OpArg>
struct DefaultImplementedTypes {
  // an empty list of types
  using types = TypeList<>;
};

// overridden implemented types - specialize this to override the set of implemented types
template <typename OpArg>
struct OverriddenImplementedTypes {};

namespace op_implemented_type_utils_internal {

template <typename OpArg>
using GetOverriddenTypes = typename OverriddenImplementedTypes<OpArg>::types;

template <typename OpArg>
using GetDefaultTypes = typename DefaultImplementedTypes<OpArg>::types;

template <typename OpArg>
using GetDefaultAndOverriddenTypes = boost::mp11::mp_set_intersection<GetOverriddenTypes<OpArg>, GetDefaultTypes<OpArg>>;

}  // namespace op_implemented_type_utils_internal

// list of implemented types
template <typename OpArg>
using ImplementedTypeList =
    boost::mp11::mp_eval_or<
        op_implemented_type_utils_internal::GetDefaultTypes<OpArg>,
        op_implemented_type_utils_internal::GetDefaultAndOverriddenTypes, OpArg>;

}  // namespace onnxruntime

#define ORT_IMPLEMENTED_TYPES_OP_TAG_NAME(Op) \
  implemented_types_tag_for_##Op

#define ORT_IMPLEMENTED_TYPES_OP_ARG_TAG(Op, ArgType, ArgIndex) \
  ::onnxruntime::OpArgTag<ORT_IMPLEMENTED_TYPES_OP_TAG_NAME(Op), OpArgType::ArgType, ArgIndex>

// helper macros
// note: need to be careful about which namespace these get called from due to op tag type

#define ORT_DECLARE_OP_ARG_DEFAULT_IMPLEMENTED_TYPES(Op, ArgType, ArgIndex, ...) \
  class ORT_IMPLEMENTED_TYPES_OP_TAG_NAME(Op);                                   \
  template <>                                                                    \
  struct ::onnxruntime::DefaultImplementedTypes<                                 \
      ORT_IMPLEMENTED_TYPES_OP_ARG_TAG(Op, ArgType, ArgIndex)> {                 \
    using types = TypeList<__VA_ARGS__>;                                         \
  };

#define ORT_DECLARE_OP_ARG_OVERRIDDEN_IMPLEMENTED_TYPES(Op, ArgType, ArgIndex, ...) \
  class ORT_IMPLEMENTED_TYPES_OP_TAG_NAME(Op);                                      \
  template <>                                                                       \
  struct ::onnxruntime::OverriddenImplementedTypes<                                 \
      ORT_IMPLEMENTED_TYPES_OP_ARG_TAG(Op, ArgType, ArgIndex)> {                    \
    using types = TypeList<__VA_ARGS__>;                                            \
  };

#define ORT_OP_ARG_IMPLEMENTED_TYPE_LIST(Op, ArgType, ArgIndex) \
  ::onnxruntime::ImplementedTypeList<ORT_IMPLEMENTED_TYPES_OP_ARG_TAG(Op, ArgType, ArgIndex)>
