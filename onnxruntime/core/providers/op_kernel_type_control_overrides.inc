// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
namespace op_kernel_type_control {

/**
 * Specifies an allowed set of types for a given Op kernel argument.
 * This can optionally be specified to further limit the enabled types.
 *
 * Note: This should be called from the onnxruntime::op_kernel_type_control namespace.
 *
 * @param OpDomain The Op domain.
 * @param OpName The Op name.
 * @param ArgDirection Direction of the given Op kernel argument - Input or Output.
 * @param ArgIndex Index of the given Op kernel argument.
 * @param ... The types.
 */
#define ORT_SPECIFY_OP_KERNEL_ARG_ALLOWED_TYPES(                              \
    OpDomain, OpName, ArgDirection, ArgIndex, ...)                            \
  class ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_TAG_CLASS_NAME(OpDomain, OpName); \
  template <>                                                                 \
  struct TypesHolder<                                                         \
      ::onnxruntime::op_kernel_type_control::tags::Allowed<                   \
          ORT_OP_KERNEL_TYPE_CTRL_INTERNAL_OP_ARG_TAG(                        \
              OpDomain, OpName, ArgDirection, ArgIndex)>> {                   \
    using types = ::onnxruntime::TypeList<__VA_ARGS__>;                       \
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
  struct TypesHolder<                                               \
      ::onnxruntime::op_kernel_type_control::tags::GlobalAllowed> { \
    using types = ::onnxruntime::TypeList<__VA_ARGS__>;             \
  };

// Examples:
// Specify allowed types per Op kernel arg:
//   ORT_SPECIFY_OP_KERNEL_ARG_ALLOWED_TYPES(kOnnxDomain, Cast, Input, 0, float, int64_t);
//   ORT_SPECIFY_OP_KERNEL_ARG_ALLOWED_TYPES(kOnnxDomain, Cast, Output, 0, float, int64_t);
// Specify allowed types globally:
//   ORT_SPECIFY_OP_KERNEL_GLOBAL_ALLOWED_TYPES(float, double, int32_t);

// specify allowed types here

// @@insertion_point_begin(allowed_types)@@
// @@insertion_point_end(allowed_types)@@

#undef ORT_SPECIFY_OP_KERNEL_ARG_ALLOWED_TYPES
#undef ORT_SPECIFY_OP_KERNEL_GLOBAL_ALLOWED_TYPES

}  // namespace op_kernel_type_control
}  // namespace onnxruntime
