// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <tuple>

#include "boost/mp11.hpp"

#include "core/common/type_list.h"
#include "core/framework/data_types.h"

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

enum class OpArgDirection {
  Input,
  Output
};

using OpArgIndex = size_t;

// constant to use for type lists that are valid across all opsets
constexpr int kAllOpSets = -1;

namespace tags {

// a tag that identifies the target (Op argument) of the specified types
template <typename OpTag, OpArgDirection ArgDirection, OpArgIndex ArgIndex>
struct OpArg {};

// a tag that indicates the supported types for a particular Op argument, identified by OpArgTag,
// for a kernel in a particular provider, identified by ProviderTag. as the types may change between opsets,
// the opset must also be specified. if the type list is not opset specific, use kAllOpSets as the value.
template <typename OpArgTag, typename ProviderTag, int OpSet>
struct Supported {};

// a tag that indicates the allowed types for a particular Op argument, identified by OpArgTag
template <typename OpArgTag>
struct Allowed {};

// a tag that indicates the globally allowed types
struct GlobalAllowed {};

}  // namespace tags

// optionally holds a list of types associated with a tag class
// if types are defined, the data member 'types' should contain them in a type list
// otherwise, if no types are defined (distinct from an empty list of types), there should be no data member 'types'
// see the tags in onnxruntime::op_kernel_type_control::tags for intended uses
template <typename Tag>
struct TypesHolder {};

/**
 * Provides a type list of enabled types via the 'types' data member.
 * Enabled types are the set intersection of supported and allowed types.
 *
 * @tparam SupportedTypesHolder A 'TypesHolder' with a list of supported types.
 * @tparam AllowedTypesHolders A list of 'TypesHolder's each with an optional list of allowed types.
 */
template <typename SupportedTypesHolder, typename AllowedTypesHolders>
struct EnabledTypes {
 private:
  static_assert(boost::mp11::mp_is_list<AllowedTypesHolders>::value,
                "AllowedTypesHolders must be a type list.");

  template <typename T>
  using GetTypesMember = typename T::types;

  // checks whether T has data member 'types'
  template <typename T>
  using HasTypesMember = boost::mp11::mp_valid<GetTypesMember, T>;

  static_assert(HasTypesMember<SupportedTypesHolder>::value,
                "SupportedTypesHolder must have a 'types' data member.");

  // the allowed type lists to consider
  // for each element of AllowedTypesHolders, get and include a 'types' data member if present
  using AllowedTypesMembers =
      boost::mp11::mp_transform<
          GetTypesMember,
          boost::mp11::mp_filter<
              HasTypesMember,
              AllowedTypesHolders>>;

  // collect supported and allowed type lists
  using TypeListsToConsider =
      boost::mp11::mp_push_front<AllowedTypesMembers, GetTypesMember<SupportedTypesHolder>>;

  static_assert(boost::mp11::mp_all_of<TypeListsToConsider, boost::mp11::mp_is_list>::value,
                "All 'types' data members must be type lists.");

  // converts type list L into a type set (type list with unique elements)
  template <typename L>
  using MakeSet =
      boost::mp11::mp_apply<
          boost::mp11::mp_set_push_back,
          boost::mp11::mp_append<TypeList<TypeList<>>, L>>;

  // type lists converted to type sets
  using TypeSetsToConsider = boost::mp11::mp_transform<MakeSet, TypeListsToConsider>;

 public:
  using types = boost::mp11::mp_apply<boost::mp11::mp_set_intersection, TypeSetsToConsider>;
};

/**
* Check if the type list contains the specified type.
*/
template <typename TypeList, typename T>
constexpr bool HasType() {
  return std::is_same<boost::mp11::mp_set_contains<TypeList, T>, boost::mp11::mp_true>::value;
}

template <typename T>
using SizeOfT = boost::mp11::mp_size_t<sizeof(T)>;

/**
* Check if the type list contains a type with the same size as T.
* 
* @remarks e.g. will return true if T is int32_t and the list contains any 4 byte type (i.e. sizeof(int32_t)) 
*               such as int32_t, uint32_t or float.
*/
template <typename TypeList, typename T>
constexpr bool HasTypeWithSameSize() {
  using EnabledTypeSizes = boost::mp11::mp_unique<boost::mp11::mp_transform<SizeOfT, TypeList>>;
  return std::is_same<boost::mp11::mp_set_contains<EnabledTypeSizes, SizeOfT<T>>, boost::mp11::mp_true>::value;
}

}  // namespace op_kernel_type_control
}  // namespace onnxruntime

// INTERNAL
// the class name of a tag type identifying an Op for the purposes of type control
#define ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_TAG_CLASS_NAME(OpDomain, OpName) \
  TypeControl_##OpDomain##_##OpName##_OpTag

// INTERNAL
// the class name of a tag type identifying a provider for the purposes of type control
#define ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_PROVIDER_TAG_CLASS_NAME(OpProvider) \
  TypeControl_##OpProvider##_ProviderTag

// INTERNAL
// a tag type identifying an Op argument
#define ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_KERNEL_ARG_TAG(                     \
    OpDomain, OpName, ArgDirection, ArgIndex)                                   \
  ::onnxruntime::op_kernel_type_control::tags::OpArg<                           \
      ::onnxruntime::op_kernel_type_control::                                   \
          ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_TAG_CLASS_NAME(OpDomain, OpName), \
      ::onnxruntime::op_kernel_type_control::OpArgDirection::ArgDirection,      \
      ArgIndex>

// public macros

/** Data types that are used in DataTypeImpl::AllTensorTypes()
*/
#define ORT_OP_KERNEL_TYPE_CTRL_ALL_TENSOR_DATA_TYPES \
  bool,                                               \
      float, double,                                  \
      uint8_t, uint16_t, uint32_t, uint64_t,          \
      int8_t, int16_t, int32_t, int64_t,              \
      MLFloat16, BFloat16,                            \
      std::string

/**
 * Specifies a supported set of types for a given Op kernel argument.
 * This should be specified with the Op kernel implementation.
 *
 * Note: This should be called from the onnxruntime::op_kernel_type_control namespace.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param OpSet The opset that this set of supported types applies to.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 * @param ... The types.
 */
#define ORT_SPECIFY_OP_KERNEL_ARG_SUPPORTED_TYPES(                                                      \
    OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex, ...)                                   \
  class ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_TAG_CLASS_NAME(OpDomain, OpName);                           \
  class ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_PROVIDER_TAG_CLASS_NAME(OpProvider);                           \
  template <>                                                                                           \
  struct TypesHolder<                                                                                   \
      ::onnxruntime::op_kernel_type_control::tags::Supported<                                           \
          ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_KERNEL_ARG_TAG(OpDomain, OpName, ArgDirection, ArgIndex), \
          ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_PROVIDER_TAG_CLASS_NAME(OpProvider),                         \
          OpSet>> {                                                                                     \
    using types = ::onnxruntime::TypeList<__VA_ARGS__>;                                                 \
  };

/**
 * Specifies a supported set of types for a given Op kernel argument that is valid for all opsets.
 * This should be specified with the Op kernel implementation.
 *
 * Note: This should be called from the onnxruntime::op_kernel_type_control namespace.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 * @param ... The types.
 */
#define ORT_SPECIFY_OP_KERNEL_ARG_SUPPORTED_TYPES_ALL_OPSETS(                                  \
    OpProvider, OpDomain, OpName, ArgDirection, ArgIndex, ...)                                 \
  ORT_SPECIFY_OP_KERNEL_ARG_SUPPORTED_TYPES(OpProvider, OpDomain, OpName,                      \
                                            ::onnxruntime::op_kernel_type_control::kAllOpSets, \
                                            ArgDirection, ArgIndex, __VA_ARGS__)

/**
 * TypeList type with the enabled types for a given Op kernel argument.
 * This is created by intersecting the supported types with any type restrictions coming from the allowed or global
 * type lists.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param OpSet The opset to use for the supported types list.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 */
#define ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(                                                                      \
    OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex)                                                  \
  ::onnxruntime::op_kernel_type_control::EnabledTypes<                                                            \
      ::onnxruntime::op_kernel_type_control::TypesHolder<                                                         \
          ::onnxruntime::op_kernel_type_control::tags::Supported<                                                 \
              ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_KERNEL_ARG_TAG(OpDomain, OpName, ArgDirection, ArgIndex),       \
              ::onnxruntime::op_kernel_type_control::                                                             \
                  ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_PROVIDER_TAG_CLASS_NAME(OpProvider),                           \
              OpSet>>,                                                                                            \
      ::onnxruntime::TypeList<                                                                                    \
          ::onnxruntime::op_kernel_type_control::TypesHolder<                                                     \
              ::onnxruntime::op_kernel_type_control::tags::Allowed<                                               \
                  ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_KERNEL_ARG_TAG(OpDomain, OpName, ArgDirection, ArgIndex)>>, \
          ::onnxruntime::op_kernel_type_control::TypesHolder<                                                     \
              ::onnxruntime::op_kernel_type_control::tags::GlobalAllowed>>>::types

/**
 * TypeList type with the enabled types for a given Op kernel argument that is valid for all opsets.
 * This is created by intersecting the supported types with any type restrictions coming from the allowed or global
 * type lists.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 */
#define ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(                                  \
    OpProvider, OpDomain, OpName, ArgDirection, ArgIndex)                                \
  ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(OpProvider, OpDomain, OpName,                      \
                                      ::onnxruntime::op_kernel_type_control::kAllOpSets, \
                                      ArgDirection, ArgIndex)

/**
 * std::tuple type with the enabled types for a given Op kernel argument.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param OpSet The opset to use for the supported types list.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 */
#define ORT_OP_KERNEL_ARG_ENABLED_TYPE_TUPLE(                                              \
    OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex)                           \
  ::boost::mp11::mp_rename<                                                                \
      ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(                                                 \
          OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex, SupportedTypeList), \
      std::tuple>

/**
 * std::tuple type with the enabled types for a given Op kernel argument that is valid for all opsets.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 */
#define ORT_OP_KERNEL_ARG_ENABLED_TYPE_TUPLE_ALL_OPSETS(                                  \
    OpProvider, OpDomain, OpName, ArgDirection, ArgIndex)                                 \
  ORT_OP_KERNEL_ARG_ENABLED_TYPE_TUPLE(OpProvider, OpDomain, OpName,                      \
                                       ::onnxruntime::op_kernel_type_control::kAllOpSets, \
                                       ArgDirection, ArgIndex)
/**
 * Usage example:
 *
 * In MyProvider provider's implementation of MyOp kernel:
 *
 * // specify supported types, i.e., the full set of types that can be enabled
 * ORT_SPECIFY_OP_KERNEL_ARG_SUPPORTED_TYPES(
 *     MyProvider, DomainContainingMyOp, MyOp, Input, 0,
 *     int, float, double);
 *
 * // get enabled types
 * using MyOpFirstInputEnabledTypes =
 *     ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(MyProvider, DomainContainingMyOp, MyOp, Input, 0)
 *
 * ...
 *
 * // in the implementation, we can dispatch to the enabled types
 * utils::MLTypeCallDispatcherFromTypeList<MyOpFirstInputEnabledTypes> dispatcher{firstInputRuntimeType};
 * ...
 */

// all allowed type specifications should be contained in the following file
#include "core/providers/op_kernel_type_control_overrides.inc"
