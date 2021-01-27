// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <tuple>

#include "boost/mp11.hpp"

#include "core/common/type_list.h"

/**
 * These utilities provide a way to control what types are enabled for an Op kernel implementation.
 *
 * At a high level, we have the notion of supported, allowed, and enabled types.
 * - Supported types are the types that the Op kernel implementation supports by default.
 * - Allowed types are the types for which support is requested (for example, by external configuration).
 * - Enabled types are the types that are supported in the actual, compiled implementation. They are obtained from the
 *   intersection of supported and allowed types.
 *
 * The types are associated with an Op kernel argument. It is also possible to specify a global list of allowed types.
 *
 * Use of these utilities is optional. They are useful for cases where one registered Op kernel handles multiple types.
 *
 * See the macros below for usage details.
 */

namespace onnxruntime {
namespace op_kernel_type_control {

enum class OpKernelArgDirection {
  Input,
  Output
};

using OpKernelArgIndex = size_t;

namespace tags {

// a tag that identifies the target (Op kernel argument) of the specified types
template <typename OpKernelTag, OpKernelArgDirection ArgDirection, OpKernelArgIndex ArgIndex>
struct OpKernelArg {};

// a tag that indicates the supported types for a particular Op kernel argument, identified by OpKernelArgTag
template <typename OpKernelArgTag>
struct Supported {};

// a tag that indicates the allowed types for a particular Op kernel argument, identified by OpKernelArgTag
template <typename OpKernelArgTag>
struct Allowed {};

// a tag that indicates the globally allowed types
struct GlobalAllowed {};

}  // namespace tags

// holds specified Op kernel argument types
// if types are defined, the data member 'type' should contain them in a type list
// otherwise, if no types are defined (distinct from an empty list of types), there should be no data member 'type'
// see the tags in onnxruntime::op_kernel_type_control::tags for intended uses
template <typename Tag>
struct OpKernelArgTypes {};

// gives access to the enabled Op kernel argument types via the 'type' data member
template <typename OpKernelArgTag>
struct EnabledOpKernelArgTypes {
 private:
  template <typename T>
  using GetTypeMember = typename T::type;

  // checks whether T has data member 'type'
  template <typename T>
  using HasTypeMember = boost::mp11::mp_valid<GetTypeMember, T>;

  static_assert(HasTypeMember<OpKernelArgTypes<tags::Supported<OpKernelArgTag>>>::value,
                "OpKernelArgTypes<Supported<OpKernelArgTag>> must have a 'type' data member. "
                "Check that the Op kernel argument supported types are specified.");

  // type list of 'type' members to consider
  using TypeMembers =
      boost::mp11::mp_transform<
          GetTypeMember,
          boost::mp11::mp_append<
              // OpKernelArgTypes<Supported<OpKernelArgTag>> should always contain a 'type' member
              TypeList<
                  OpKernelArgTypes<tags::Supported<OpKernelArgTag>>>,
              // OpKernelArgTypes<Reduced<OpKernelArgTag>> and OpKernelArgTypes<GlobalReduced> each may optionally
              // contain a 'type' member, so only include their 'type's if they do
              boost::mp11::mp_filter<
                  HasTypeMember,
                  TypeList<
                      OpKernelArgTypes<tags::Allowed<OpKernelArgTag>>,
                      OpKernelArgTypes<tags::GlobalAllowed>>>>>;

  static_assert(boost::mp11::mp_all_of<TypeMembers, boost::mp11::mp_is_list>::value,
                "All OpKernelArgTypes<Tag> 'type' data members must be type lists.");

  // converts type list L into a type set (type list with unique elements)
  template <typename L>
  using MakeSet =
      boost::mp11::mp_apply<
          boost::mp11::mp_set_push_back,
          boost::mp11::mp_append<TypeList<TypeList<>>, L>>;

  // type list of 'type' members converted to type sets
  using TypeMemberSets = boost::mp11::mp_transform<MakeSet, TypeMembers>;

 public:
  using type = boost::mp11::mp_apply<boost::mp11::mp_set_intersection, TypeMemberSets>;
};

}  // namespace op_kernel_type_control
}  // namespace onnxruntime

// INTERNAL
// the class name of a tag type identifying an Op kernel for the purposes of type control
#define ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_TAG_CLASS_NAME(OpKernel) \
  OpKernelTypeControlFor##OpKernel##Tag

// INTERNAL
// a tag type identifying an Op kernel argument
#define ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_KERNEL_ARG_TAG(OpKernel, ArgDirection, ArgIndex) \
  ::onnxruntime::op_kernel_type_control::tags::OpKernelArg<                                  \
      ::onnxruntime::op_kernel_type_control::                                                \
          ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_TAG_CLASS_NAME(OpKernel),                      \
      ::onnxruntime::op_kernel_type_control::OpKernelArgDirection::ArgDirection,             \
      ArgIndex>

// public macros

/**
 * Specifies the supported types for a given Op kernel argument.
 * This should be specified with the Op kernel implementation.
 *
 * Note: This should be called from the onnxruntime::op_kernel_type_control namespace.
 *
 * @param Op The Op name.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 * @param ... The types.
 */
#define ORT_SPECIFY_OP_KERNEL_ARG_SUPPORTED_TYPES(Op, ArgDirection, ArgIndex, ...)           \
  class ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_TAG_CLASS_NAME(Op);                              \
  template <>                                                                                \
  struct OpKernelArgTypes<                                                                   \
      ::onnxruntime::op_kernel_type_control::tags::Supported<                                \
          ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_KERNEL_ARG_TAG(Op, ArgDirection, ArgIndex)>> { \
    using type = ::onnxruntime::TypeList<__VA_ARGS__>;                                       \
  };

/**
 * Specifies an allowed set of types for a given Op kernel argument.
 * This can optionally be specified to further limit the enabled types.
 *
 * Note: This should be called from the onnxruntime::op_kernel_type_control namespace.
 *
 * @param Op The Op name.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 * @param ... The types.
 */
#define ORT_SPECIFY_OP_KERNEL_ARG_ALLOWED_TYPES(Op, ArgDirection, ArgIndex, ...)             \
  class ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_TAG_CLASS_NAME(Op);                              \
  template <>                                                                                \
  struct OpKernelArgTypes<                                                                   \
      ::onnxruntime::op_kernel_type_control::tags::Allowed<                                  \
          ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_KERNEL_ARG_TAG(Op, ArgDirection, ArgIndex)>> { \
    using type = ::onnxruntime::TypeList<__VA_ARGS__>;                                       \
  };

/**
 * Specifies an allowed set of types globally (applicable to any Op kernel argument).
 * This can optionally be specified to further limit the enabled types.
 *
 * Note: This should be called from the onnxruntime::op_kernel_type_control namespace.
 *
 * @param ... The types.
 */
#define ORT_SPECIFY_OP_KERNEL_GLOBAL_ALLOWED_TYPES(...)             \
  template <>                                                       \
  struct OpKernelArgTypes<                                          \
      ::onnxruntime::op_kernel_type_control::tags::GlobalAllowed> { \
    using type = ::onnxruntime::TypeList<__VA_ARGS__>;              \
  };

/**
 * TypeList type with the enabled types for a given Op kernel argument.
 *
 * @param Op The Op name.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 */
#define ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(Op, ArgDirection, ArgIndex) \
  ::onnxruntime::op_kernel_type_control::EnabledOpKernelArgTypes<       \
      ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_KERNEL_ARG_TAG(               \
          Op, ArgDirection, ArgIndex)>::type

/**
 * std::tuple type with the enabled types for a given Op kernel argument.
 *
 * @param Op The Op name.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 */
#define ORT_OP_KERNEL_ARG_ENABLED_TYPE_TUPLE(Op, ArgDirection, ArgIndex) \
  ::boost::mp11::mp_rename<                                              \
      ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(Op, ArgDirection, ArgIndex),   \
      std::tuple>

#include "core/providers/op_kernel_type_control_overrides.inc"
