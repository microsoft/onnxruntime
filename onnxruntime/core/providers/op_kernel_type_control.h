// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <tuple>

#include "boost/mp11.hpp"

#include "core/common/type_list.h"
#include "core/common/type_set_utils.h"

#include "core/framework/data_types.h"

/**
 * These utilities provide a way to control what types are enabled for an Op kernel implementation.
 * At a high level, we have the notion of default, required, allowed, and enabled type sets.
 *
 * 1. Default types are the types that the Op kernel implementation supports by default.
 *    This type set must be provided.
 *
 * 2. Required types are the types that are always supported.
 *    This type set is optional. If not given, the required type set is empty.
 *    They should be a subset of the default types (1).
 *
 * 3. Allowed types are the types for which support is requested (for example, by external configuration).
 *    Zero or more allowed type sets may be given.
 *    The default type set (1) will be limited by set intersection with all allowed type sets.
 *
 * 4. Enabled types are the types that are actually supported in this build.
 *    These are the required types and the default, allowed types.
 *    Defined with set operations:
 *      enabled (4) = union( required (2),
 *                           intersection( default (1)
 *                                         [, allowed_0 (3), allowed_1, ...] ) )
 *
 * These types are usually associated with an Op argument. It is also possible to specify globally allowed types.
 *
 * Use of these utilities is optional. They are useful for cases where one registered Op kernel handles multiple types.
 *
 * See the macros below for usage details. Although this description deals with type sets, lists may be provided which
 * will get converted to sets.
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

// a tag that indicates the default types for a particular Op argument (identified by OpArgTag), provider kernel
// (identified by ProviderTag), and opset (identified by OpSet, use kAllOpSets if applicable to all opsets).
template <typename OpArgTag, typename ProviderTag, int OpSet>
struct Default {};

// a tag that indicates the required types for a particular Op argument (identified by OpArgTag), provider kernel
// (identified by ProviderTag), and opset (identified by OpSet, use kAllOpSets if applicable to all opsets).
template <typename OpArgTag, typename ProviderTag, int OpSet>
struct Required {};

// a tag that indicates the allowed types for a particular Op argument, identified by OpArgTag
template <typename OpArgTag>
struct Allowed {};

// a tag that indicates the globally allowed types
struct GlobalAllowed {};

}  // namespace tags

// optionally holds a list of types associated with a tag class
// if types are defined, the type alias member called 'types' should contain them in a type list
// (e.g., using something like onnxruntime::TypeList, std::tuple, or boost::mp11::mp_list)
// otherwise, if no types are defined (distinct from an empty list of types), there should be no 'types' type alias
// see the tags in onnxruntime::op_kernel_type_control::tags for intended uses
template <typename Tag>
struct TypesHolder {};

/**
 * Provides a type set of enabled types via the 'types' type alias member.
 *
 * @tparam DefaultTypesHolder A 'TypesHolder' with a list of default types.
 * @tparam RequiredTypesHolder A 'TypesHolder' with an optional list of required types.
 *         If no list is provided, this is interpreted as an empty list of required types.
 * @tparam AllowedTypesHolders A list of 'TypesHolder's each with an optional list of allowed types.
 */
template <typename DefaultTypesHolder, typename RequiredTypesHolder, typename AllowedTypesHolders>
struct EnabledTypes {
 private:
  // gets T::types
  template <typename T>
  using GetTypesMember = typename T::types;

  // checks whether T has a type alias member called 'types'
  template <typename T>
  using HasTypesMember = boost::mp11::mp_valid<GetTypesMember, T>;

  template <typename T>
  struct GetTypesMemberAsSetImpl {
    static_assert(boost::mp11::mp_is_list<GetTypesMember<T>>::value, "'types' must be a type list.");
    using type = boost::mp11::mp_unique<GetTypesMember<T>>;
  };

  // gets T::types converted to a type set
  template <typename T>
  using GetTypesMemberAsSet = typename GetTypesMemberAsSetImpl<T>::type;

  template <typename T>
  using GetTypesMemberAsSetOrEmpty =
      // !HasTypesMember<T>::value ? TypeList<> : GetTypesMemberAsSet<T>
      // if !HasTypesMember<T>::value, GetTypesMemberAsSet<T> is not valid
      // mp_eval_if_not does not evaluate it in that case
      boost::mp11::mp_eval_if_not<
          HasTypesMember<T>,
          TypeList<>,
          GetTypesMemberAsSet, T>;

  static_assert(HasTypesMember<DefaultTypesHolder>::value, "Default types must be provided.");

  using DefaultTypeSet = GetTypesMemberAsSet<DefaultTypesHolder>;

  using RequiredTypeSet = GetTypesMemberAsSetOrEmpty<RequiredTypesHolder>;

  static_assert(utils::type_set::IsSubset<RequiredTypeSet, DefaultTypeSet>::value,
                "Required types must be a subset of default types.");

  static_assert(boost::mp11::mp_is_list<AllowedTypesHolders>::value,
                "AllowedTypesHolders must be a type list.");

  using AllowedTypeSets =
      // list of any sets from AllowedTypesHolders
      boost::mp11::mp_transform<
          GetTypesMemberAsSet,
          boost::mp11::mp_filter<
              HasTypesMember,
              AllowedTypesHolders>>;

 public:
  using types =
      boost::mp11::mp_set_union<
          RequiredTypeSet,
          boost::mp11::mp_apply<
              boost::mp11::mp_set_intersection,
              boost::mp11::mp_push_front<AllowedTypeSets, DefaultTypeSet>>>;
};

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
#define ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_ARG_TAG(                            \
    OpDomain, OpName, ArgDirection, ArgIndex)                                   \
  ::onnxruntime::op_kernel_type_control::tags::OpArg<                           \
      ::onnxruntime::op_kernel_type_control::                                   \
          ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_TAG_CLASS_NAME(OpDomain, OpName), \
      ::onnxruntime::op_kernel_type_control::OpArgDirection::ArgDirection,      \
      ArgIndex>

// INTERNAL
// the unqualified TypesHolder type that contains the default types list
#define ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_DEFAULT_TYPES_HOLDER(                                   \
    OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex)                                 \
  TypesHolder<                                                                                   \
      ::onnxruntime::op_kernel_type_control::tags::Default<                                      \
          ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_ARG_TAG(OpDomain, OpName, ArgDirection, ArgIndex), \
          ::onnxruntime::op_kernel_type_control::                                                \
              ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_PROVIDER_TAG_CLASS_NAME(OpProvider),              \
          OpSet>>

// INTERNAL
// the unqualified TypesHolder type that contains the required types list
#define ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_REQUIRED_TYPES_HOLDER(                                  \
    OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex)                                 \
  TypesHolder<                                                                                   \
      ::onnxruntime::op_kernel_type_control::tags::Required<                                     \
          ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_ARG_TAG(OpDomain, OpName, ArgDirection, ArgIndex), \
          ::onnxruntime::op_kernel_type_control::                                                \
              ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_PROVIDER_TAG_CLASS_NAME(OpProvider),              \
          OpSet>>

//
// public macros
//

/**
 * Specifies a default set of types for a given Op kernel argument.
 * Required for Op kernel type control.
 *
 * Note: This should be called from the onnxruntime::op_kernel_type_control namespace.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param OpSet The opset that this set of default types applies to.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 * @param DefaultTypeList The types.
 */
#define ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(                              \
    OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex, DefaultTypeList) \
  class ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_TAG_CLASS_NAME(OpDomain, OpName);     \
  class ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_PROVIDER_TAG_CLASS_NAME(OpProvider);     \
  template <>                                                                     \
  struct ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_DEFAULT_TYPES_HOLDER(                   \
      OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex) {              \
    using types = DefaultTypeList;                                                \
  };

/**
 * Specifies a default set of types for a given Op kernel argument that is valid for all opsets.
 * Required for Op kernel type control.
 *
 * Note: This should be called from the onnxruntime::op_kernel_type_control namespace.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 * @param DefaultTypeList The types.
 */
#define ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(                                  \
    OpProvider, OpDomain, OpName, ArgDirection, ArgIndex, DefaultTypeList)                       \
  ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(OpProvider, OpDomain, OpName,                      \
                                              ::onnxruntime::op_kernel_type_control::kAllOpSets, \
                                              ArgDirection, ArgIndex, DefaultTypeList)

/**
 * Specifies a default set of types for a given Op kernel argument.
 * Required for Op kernel type control.
 *
 * Note: This should be called from the onnxruntime::op_kernel_type_control namespace.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param OpSet The opset that this set of default types applies to.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 * @param ... The types.
 */
#define ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(                      \
    OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex, ...) \
  ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(                        \
      OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex,    \
      ORT_TYPE_LIST(__VA_ARGS__))

/**
 * Specifies a default set of types for a given Op kernel argument that is valid for all opsets.
 * Required for Op kernel type control.
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
#define ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(                                  \
    OpProvider, OpDomain, OpName, ArgDirection, ArgIndex, ...)                               \
  ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(OpProvider, OpDomain, OpName,                      \
                                          ::onnxruntime::op_kernel_type_control::kAllOpSets, \
                                          ArgDirection, ArgIndex, __VA_ARGS__)

/**
 * Specifies a required set of types for a given Op kernel argument.
 * Optional.
 *
 * Note: This should be called from the onnxruntime::op_kernel_type_control namespace.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param OpSet The opset that this set of required types applies to.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 * @param RequiredTypeList The types.
 */
#define ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPE_LIST(                              \
    OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex, RequiredTypeList) \
  class ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_TAG_CLASS_NAME(OpDomain, OpName);      \
  class ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_PROVIDER_TAG_CLASS_NAME(OpProvider);      \
  template <>                                                                      \
  struct ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_REQUIRED_TYPES_HOLDER(                   \
      OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex) {               \
    using types = RequiredTypeList;                                                \
  };

/**
 * Specifies a required set of types for a given Op kernel argument that is valid for all opsets.
 * Optional.
 *
 * Note: This should be called from the onnxruntime::op_kernel_type_control namespace.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 * @param RequiredTypeList The types.
 */
#define ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPE_LIST_ALL_OPSETS(                                  \
    OpProvider, OpDomain, OpName, ArgDirection, ArgIndex, RequiredTypeList)                       \
  ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPE_LIST(OpProvider, OpDomain, OpName,                      \
                                               ::onnxruntime::op_kernel_type_control::kAllOpSets, \
                                               ArgDirection, ArgIndex, RequiredTypeList)

/**
 * Specifies a required set of types for a given Op kernel argument.
 * Optional.
 *
 * Note: This should be called from the onnxruntime::op_kernel_type_control namespace.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param OpSet The opset that this set of required types applies to.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 * @param ... The types.
 */
#define ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(                     \
    OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex, ...) \
  ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPE_LIST(                       \
      OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex,    \
      ORT_TYPE_LIST(__VA_ARGS__))

/**
 * Specifies a required set of types for a given Op kernel argument that is valid for all opsets.
 * Optional.
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
#define ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(                                  \
    OpProvider, OpDomain, OpName, ArgDirection, ArgIndex, ...)                                \
  ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES(OpProvider, OpDomain, OpName,                      \
                                           ::onnxruntime::op_kernel_type_control::kAllOpSets, \
                                           ArgDirection, ArgIndex, __VA_ARGS__)

/**
 * TypeList type with the default types for a given Op kernel argument.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param OpSet The opset to use for the default types list.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 */
#define ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(                     \
    OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex) \
  ::onnxruntime::op_kernel_type_control::                        \
      ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_DEFAULT_TYPES_HOLDER(     \
          OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex)::types

/**
 * TypeList type with the default types for a given Op kernel argument that are valid for all opsets.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 */
#define ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(                                  \
    OpProvider, OpDomain, OpName, ArgDirection, ArgIndex)                                \
  ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(OpProvider, OpDomain, OpName,                      \
                                      ::onnxruntime::op_kernel_type_control::kAllOpSets, \
                                      ArgDirection, ArgIndex)

/**
 * TypeList type with the enabled types for a given Op kernel argument.
 *
 * @param OpProvider The Op provider.
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param OpSet The opset to use for the enabled types list.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 */
#define ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(                                                               \
    OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex)                                           \
  ::onnxruntime::op_kernel_type_control::EnabledTypes<                                                     \
      ::onnxruntime::op_kernel_type_control::ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_DEFAULT_TYPES_HOLDER(        \
          OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex),                                    \
      ::onnxruntime::op_kernel_type_control::ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_REQUIRED_TYPES_HOLDER(       \
          OpProvider, OpDomain, OpName, OpSet, ArgDirection, ArgIndex),                                    \
      ::onnxruntime::TypeList<                                                                             \
          ::onnxruntime::op_kernel_type_control::TypesHolder<                                              \
              ::onnxruntime::op_kernel_type_control::tags::Allowed<                                        \
                  ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_ARG_TAG(OpDomain, OpName, ArgDirection, ArgIndex)>>, \
          ::onnxruntime::op_kernel_type_control::TypesHolder<                                              \
              ::onnxruntime::op_kernel_type_control::tags::GlobalAllowed>>>::types

/**
 * TypeList type with the enabled types for a given Op kernel argument that is valid for all opsets.
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
 * Usage example:
 *
 * In MyProvider provider's implementation of MyOp kernel:
 *
 * namespace onnxruntime {
 * namespace op_kernel_type_control {
 * // specify default types, i.e., the full set of types that can be enabled
 * ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
 *     MyProvider, DomainContainingMyOp, MyOp, Input, 0,
 *     int, float, double);
 * // specify required types, i.e., the set of types that must be enabled
 * ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
 *     MyProvider, DomainContainingMyOp, MyOp, Input, 0,
 *     int);
 * }  // namespace op_kernel_type_control
 * }  // namespace onnxruntime
 *
 * // ...
 *
 * // get enabled types
 * using MyOpFirstInputEnabledTypes =
 *     ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(MyProvider, DomainContainingMyOp, MyOp, Input, 0);
 * // MyOpFirstInputEnabledTypes will contain required type int and may contain other default types
 *
 * // ...
 *
 * // use MLTypeCallDispatcher to dispatch to implementations for enabled types
 * using Dispatcher = onnxruntime::utils::MLTypeCallDispatcherFromTypeList<MyOpFirstInputEnabledTypes>;
 */

// all allowed type specifications should be contained in the following file
#include "core/providers/op_kernel_type_control_overrides.inc"
