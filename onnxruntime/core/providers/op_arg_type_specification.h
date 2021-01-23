// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <tuple>

#include "boost/mp11.hpp"

#include "core/common/type_list.h"

namespace onnxruntime {

namespace op_arg_type_specification_internal {

enum class OpArgDirection {
  Input,
  Output
};

using OpArgIndex = size_t;

namespace tags {
// a tag type to identify the target (Op and argument) of the type lists
template <typename Op, OpArgDirection ArgDirection, OpArgIndex ArgIndex>
struct OpArg {};

template <typename OpArgTag>
struct Supported {};

template <typename OpArgTag>
struct Reduced {};

struct GlobalReduced {};
}  // namespace tags

template <typename Tag>
struct OpArgTypes {};

template <typename OpArgTag>
struct EnabledOpArgTypes {
 private:
  template <typename T>
  using GetTypeMember = typename T::type;

  template <typename T>
  using HasTypeMember = boost::mp11::mp_valid<GetTypeMember, T>;

  static_assert(HasTypeMember<OpArgTypes<tags::Supported<OpArgTag>>>::value,
                "OpArgTypes<Supported<OpArgTag>> must have a 'type' data member. "
                "Check that the Op argument supported types are specified.");

  using TypeMembers =
      boost::mp11::mp_append<
          TypeList<
              GetTypeMember<OpArgTypes<tags::Supported<OpArgTag>>>>,
          boost::mp11::mp_transform<
              GetTypeMember,
              boost::mp11::mp_filter<
                  HasTypeMember,
                  TypeList<
                      OpArgTypes<tags::Reduced<OpArgTag>>,
                      OpArgTypes<tags::GlobalReduced>>>>>;

  static_assert(boost::mp11::mp_all_of<TypeMembers, boost::mp11::mp_is_list>::value,
                "All OpArgTypes<Tag> 'type' data members must be type lists.");

  template <typename L>
  using MakeSet =
      boost::mp11::mp_apply<
          boost::mp11::mp_set_push_back,
          boost::mp11::mp_append<TypeList<TypeList<>>, L>>;

  using TypeMemberSets = boost::mp11::mp_transform<MakeSet, TypeMembers>;

 public:
  using type = boost::mp11::mp_apply<boost::mp11::mp_set_intersection, TypeMemberSets>;
};

}  // namespace op_arg_type_specification_internal
}  // namespace onnxruntime

// INTERNAL
// the class name of a tag type identifying an Op for the purposes of Op arg type specification
#define ORT_OP_ARG_TYPE_SPEC_INTERNAL_OP_TAG_CLASS_NAME(Op) \
  op_arg_type_specification_for_##Op##_tag

// INTERNAL
// a tag type identifying an Op and an argument
#define ORT_OP_ARG_TYPE_SPEC_INTERNAL_OP_ARG_TAG(Op, ArgDirection, ArgIndex)           \
  ::onnxruntime::op_arg_type_specification_internal::tags::OpArg<                      \
      ORT_OP_ARG_TYPE_SPEC_INTERNAL_OP_TAG_CLASS_NAME(Op),                             \
      ::onnxruntime::op_arg_type_specification_internal::OpArgDirection::ArgDirection, \
      ArgIndex>

// public macros

/**
 * Specifies the supported types for a given Op argument.
 * This should be specified with the Op implementation.
 *
 * Note: ORT_SPECIFY_OP_ARG_SUPPORTED_TYPES() and ORT_SPECIFY_OP_ARG_REDUCED_TYPES()
 * should be called from the same namespace.
 *
 * @param Op The Op name.
 * @param ArgDirection Direction of the given Op argument - Input or Output.
 * @param ArgIndex Index of the given Op argument.
 * @param ... The types.
 */
#define ORT_SPECIFY_OP_ARG_SUPPORTED_TYPES(Op, ArgDirection, ArgIndex, ...)        \
  class ORT_OP_ARG_TYPE_SPEC_INTERNAL_OP_TAG_CLASS_NAME(Op);                       \
  template <>                                                                      \
  struct ::onnxruntime::op_arg_type_specification_internal::OpArgTypes<            \
      ::onnxruntime::op_arg_type_specification_internal::tags::Supported<          \
          ORT_OP_ARG_TYPE_SPEC_INTERNAL_OP_ARG_TAG(Op, ArgDirection, ArgIndex)>> { \
    using type = ::onnxruntime::TypeList<__VA_ARGS__>;                             \
  };

/**
 * Specifies a reduced set of types for a given Op argument.
 * This can optionally be specified to further limit the enabled types.
 *
 * Note: ORT_SPECIFY_OP_ARG_SUPPORTED_TYPES() and ORT_SPECIFY_OP_ARG_REDUCED_TYPES()
 * should be called from the same namespace.
 *
 * @param Op The Op name.
 * @param ArgDirection Direction of the given Op argument - Input or Output.
 * @param ArgIndex Index of the given Op argument.
 * @param ... The types.
 */
#define ORT_SPECIFY_OP_ARG_REDUCED_TYPES(Op, ArgDirection, ArgIndex, ...)          \
  class ORT_OP_ARG_TYPE_SPEC_INTERNAL_OP_TAG_CLASS_NAME(Op);                       \
  template <>                                                                      \
  struct ::onnxruntime::op_arg_type_specification_internal::OpArgTypes<            \
      ::onnxruntime::op_arg_type_specification_internal::tags::Reduced<            \
          ORT_OP_ARG_TYPE_SPEC_INTERNAL_OP_ARG_TAG(Op, ArgDirection, ArgIndex)>> { \
    using type = ::onnxruntime::TypeList<__VA_ARGS__>;                             \
  };

/**
 * Specifies a reduced set of types globally (applicable to any Op argument).
 * This can optionally be specified to further limit the enabled types.
 * @param ... The types.
 */
#define ORT_SPECIFY_GLOBAL_REDUCED_TYPES(...)                                   \
  template <>                                                                   \
  struct ::onnxruntime::op_arg_type_specification_internal::OpArgTypes<         \
      ::onnxruntime::op_arg_type_specification_internal::tags::GlobalReduced> { \
    using type = ::onnxruntime::TypeList<__VA_ARGS__>;                          \
  };

/**
 * TypeList type with the enabled types for a given Op argument.
 *
 * Note: At the call site, the namespace with the associated
 * ORT_SPECIFY_OP_ARG_SUPPORTED_TYPES() call should be visible.
 *
 * @param Op The Op name.
 * @param ArgDirection Direction of the given Op argument - Input or Output.
 * @param ArgIndex Index of the given Op argument.
 */
#define ORT_OP_ARG_ENABLED_TYPE_LIST(Op, ArgDirection, ArgIndex)        \
  ::onnxruntime::op_arg_type_specification_internal::EnabledOpArgTypes< \
      ORT_OP_ARG_TYPE_SPEC_INTERNAL_OP_ARG_TAG(Op, ArgDirection, ArgIndex)>::type

/**
 * std::tuple type with the enabled types for a given Op argument.
 *
 * Note: At the call site, the namespace with the associated
 * ORT_SPECIFY_OP_ARG_SUPPORTED_TYPES() call should be visible.
 *
 * @param Op The Op name.
 * @param ArgDirection Direction of the given Op argument - Input or Output.
 * @param ArgIndex Index of the given Op argument.
 */
#define ORT_OP_ARG_ENABLED_TYPE_TUPLE(Op, ArgDirection, ArgIndex) \
  boost::mp11::mp_rename<                                         \
      ORT_OP_ARG_ENABLED_TYPE_LIST(Op, ArgDirection, ArgIndex),   \
      std::tuple>
