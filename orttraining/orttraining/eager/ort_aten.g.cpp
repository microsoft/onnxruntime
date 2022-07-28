// AUTO-GENERATED CODE! - DO NOT EDIT!
// $ python /home/lw/code/onnxruntime/orttraining/orttraining/eager/opgen/opgen.py --output_file /home/lw/code/onnxruntime/orttraining/orttraining/eager/ort_aten.g.cpp.working --ops_module /home/lw/code/onnxruntime/orttraining/orttraining/eager/opgen/opgen/atenops.py --header_file /home/lw/venvs/onnxruntime/lib/python3.8/site-packages/torch/include/ATen/RegistrationDeclarations.h

#include "python/onnxruntime_pybind_state_common.h"

#include <torch/extension.h>
#include <ATen/native/CPUFallback.h>

#include <core/providers/dml/OperatorAuthorHelper/Attributes.h>

#include "ort_tensor.h"
#include "ort_aten.h"
#include "ort_log.h"

namespace torch_ort {
namespace eager {

using namespace at;
using NodeAttributes = onnxruntime::NodeAttributes;

namespace aten {

// aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& abs_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(abs_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Abs(1);
  ort_outputs_0_Abs[0] = ort_input_out;
  
  auto status = invoker.Invoke("Abs", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Abs, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& acos_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(acos_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Acos(1);
  ort_outputs_0_Acos[0] = ort_input_out;
  
  auto status = invoker.Invoke("Acos", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Acos, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor& add_out(
  const Tensor& self, 
  const Tensor& other, 
  // *, 
  const Scalar& alpha, 
  Tensor& out) {
  ORT_LOG_FN(self, other, alpha, out);
  
  auto promoted_type = PromoteScalarTypesWithCategory({other.scalar_type(),self.scalar_type()}, {alpha.type()});
  
  if (
    !IsSupportedType(alpha, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !c10::canCast(*promoted_type, out.scalar_type())) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(add_out)>::call(self, other, alpha, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_alpha = create_ort_value(invoker, alpha);
  if (alpha.type() != *promoted_type){
    ort_input_0_alpha = CastToType(invoker, ort_input_0_alpha, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.scalar_type() != *promoted_type){
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_0_Mul(1);
  
  auto status = invoker.Invoke("Mul", {
    std::move(ort_input_0_alpha),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Mul, nullptr);
  CHECK_STATUS(status);
  
  auto ort_input_1_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type){
    ort_input_1_self = CastToType(invoker, ort_input_1_self, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_1_Add(1);
  if (*promoted_type == out.scalar_type()) {
    ort_outputs_1_Add[0] = ort_input_out;
  }
  
  status = invoker.Invoke("Add", {
    std::move(ort_input_1_self),
    std::move(ort_outputs_0_Mul[0]),
  }, ort_outputs_1_Add, nullptr);
  CHECK_STATUS(status);
  
  if (*promoted_type != out.scalar_type()) {
    CastToType_out(invoker, ort_outputs_1_Add[0], ort_input_out, out.scalar_type());
  }
  return out;
}

// aten::argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor& argmax_out(
  const Tensor& self, 
  c10::optional<int64_t> dim, 
  bool keepdim, 
  // *, 
  Tensor& out);

// aten::acosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& acosh_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(acosh_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Acosh(1);
  ort_outputs_0_Acosh[0] = ort_input_out;
  
  auto status = invoker.Invoke("Acosh", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Acosh, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& asinh_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(asinh_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Asinh(1);
  ort_outputs_0_Asinh[0] = ort_input_out;
  
  auto status = invoker.Invoke("Asinh", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Asinh, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& atanh_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(atanh_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Atanh(1);
  ort_outputs_0_Atanh[0] = ort_input_out;
  
  auto status = invoker.Invoke("Atanh", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Atanh, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)
Tensor as_strided(
  const Tensor& self, 
  IntArrayRef size, 
  IntArrayRef stride, 
  c10::optional<int64_t> storage_offset);

// aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& asin_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(asin_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Asin(1);
  ort_outputs_0_Asin[0] = ort_input_out;
  
  auto status = invoker.Invoke("Asin", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Asin, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& atan_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(atan_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Atan(1);
  ort_outputs_0_Atan[0] = ort_input_out;
  
  auto status = invoker.Invoke("Atan", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Atan, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& ceil_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(ceil_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Ceil(1);
  ort_outputs_0_Ceil[0] = ort_input_out;
  
  auto status = invoker.Invoke("Ceil", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Ceil, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
Tensor& copy_(
  Tensor& self, 
  const Tensor& src, 
  bool non_blocking);

// aten::_copy_from_and_resize(Tensor self, Tensor dst) -> Tensor
Tensor _copy_from_and_resize(
  const Tensor& self, 
  const Tensor& dst);

// aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& cos_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(cos_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Cos(1);
  ort_outputs_0_Cos[0] = ort_input_out;
  
  auto status = invoker.Invoke("Cos", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Cos, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& cosh_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(cosh_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Cosh(1);
  ort_outputs_0_Cosh[0] = ort_input_out;
  
  auto status = invoker.Invoke("Cosh", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Cosh, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor& div_out(
  const Tensor& self, 
  const Tensor& other, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, other, out);
  
  auto promoted_type = PromoteScalarTypesWithCategory({self.scalar_type(),other.scalar_type()}, {});
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !c10::canCast(*promoted_type, out.scalar_type())) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(div_out)>::call(self, other, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type){
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.scalar_type() != *promoted_type){
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_0_Div(1);
  if (*promoted_type == out.scalar_type()) {
    ort_outputs_0_Div[0] = ort_input_out;
  }
  
  auto status = invoker.Invoke("Div", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Div, nullptr);
  CHECK_STATUS(status);
  
  if (*promoted_type != out.scalar_type()) {
    CastToType_out(invoker, ort_outputs_0_Div[0], ort_input_out, out.scalar_type());
  }
  return out;
}

// aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
Tensor empty_memory_format(
  IntArrayRef size, 
  // *, 
  c10::optional<ScalarType> dtype, 
  c10::optional<Layout> layout, 
  c10::optional<Device> device, 
  c10::optional<bool> pin_memory, 
  c10::optional<MemoryFormat> memory_format);

// aten::resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)
const Tensor& resize_(
  const Tensor& self, 
  IntArrayRef size, 
  // *, 
  c10::optional<MemoryFormat> memory_format);

// aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor empty_strided(
  IntArrayRef size, 
  IntArrayRef stride, 
  // *, 
  c10::optional<ScalarType> dtype, 
  c10::optional<Layout> layout, 
  c10::optional<Device> device, 
  c10::optional<bool> pin_memory);

// aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& erf_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(erf_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Erf(1);
  ort_outputs_0_Erf[0] = ort_input_out;
  
  auto status = invoker.Invoke("Erf", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Erf, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& exp_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(exp_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Exp(1);
  ort_outputs_0_Exp[0] = ort_input_out;
  
  auto status = invoker.Invoke("Exp", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Exp, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
Tensor& fill__Scalar(
  Tensor& self, 
  const Scalar& value);

// aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& floor_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(floor_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Floor(1);
  ort_outputs_0_Floor[0] = ort_input_out;
  
  auto status = invoker.Invoke("Floor", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Floor, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::isnan(Tensor self) -> Tensor
Tensor isnan(
  const Tensor& self) {
  ORT_LOG_FN(self);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(isnan)>::call(self);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_IsNaN(1);
  
  auto status = invoker.Invoke("IsNaN", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_IsNaN, nullptr);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_IsNaN[0]),
    tensor_options);
}

// aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& log_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(log_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Log(1);
  ort_outputs_0_Log[0] = ort_input_out;
  
  auto status = invoker.Invoke("Log", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Log, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
Tensor& _log_softmax_out(
  const Tensor& self, 
  int64_t dim, 
  bool half_to_float, 
  // *, 
  Tensor& out);

// aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) out) -> Tensor(a!)
Tensor& _log_softmax_backward_data_out(
  const Tensor& grad_output, 
  const Tensor& output, 
  int64_t dim, 
  ScalarType input_dtype, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(grad_output, output, dim, input_dtype, out);
  
  return native::call_fallback_fn<
    &native::cpu_fallback,
    ATEN_OP(_log_softmax_backward_data_out)>::call(grad_output, output, dim, input_dtype, out);
}

// aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
Tensor& mm_out(
  const Tensor& self, 
  const Tensor& mat2, 
  // *, 
  Tensor& out);

// aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor& mul_out(
  const Tensor& self, 
  const Tensor& other, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, other, out);
  
  auto promoted_type = PromoteScalarTypesWithCategory({self.scalar_type(),other.scalar_type()}, {});
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !c10::canCast(*promoted_type, out.scalar_type())) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(mul_out)>::call(self, other, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type){
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.scalar_type() != *promoted_type){
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_0_Mul(1);
  if (*promoted_type == out.scalar_type()) {
    ort_outputs_0_Mul[0] = ort_input_out;
  }
  
  auto status = invoker.Invoke("Mul", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Mul, nullptr);
  CHECK_STATUS(status);
  
  if (*promoted_type != out.scalar_type()) {
    CastToType_out(invoker, ort_outputs_0_Mul[0], ort_input_out, out.scalar_type());
  }
  return out;
}

// aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& reciprocal_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(reciprocal_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Reciprocal(1);
  ort_outputs_0_Reciprocal[0] = ort_input_out;
  
  auto status = invoker.Invoke("Reciprocal", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Reciprocal, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& neg_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(neg_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Neg(1);
  ort_outputs_0_Neg[0] = ort_input_out;
  
  auto status = invoker.Invoke("Neg", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Neg, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::_reshape_alias(Tensor(a) self, int[] size, int[] stride) -> Tensor(a)
Tensor _reshape_alias(
  const Tensor& self, 
  IntArrayRef size, 
  IntArrayRef stride);

// aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& round_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(round_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Round(1);
  ort_outputs_0_Round[0] = ort_input_out;
  
  auto status = invoker.Invoke("Round", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Round, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::relu(Tensor self) -> Tensor
Tensor relu(
  const Tensor& self) {
  ORT_LOG_FN(self);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(relu)>::call(self);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Relu(1);
  
  auto status = invoker.Invoke("Relu", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Relu, nullptr);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_Relu[0]),
    tensor_options);
}

// aten::relu_(Tensor(a!) self) -> Tensor(a!)
Tensor& relu_(
  Tensor& self) {
  ORT_LOG_FN(self);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(relu_)>::call(self);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Relu(1);
  ort_outputs_0_Relu[0] = ort_input_0_self;
  
  auto status = invoker.Invoke("Relu", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Relu, nullptr);
  CHECK_STATUS(status);
  
  return self;
}

// aten::gelu(Tensor self, *, str approximate='none') -> Tensor
Tensor gelu(
  const Tensor& self, 
  // *, 
  c10::string_view approximate) {
  ORT_LOG_FN(self, approximate);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(gelu)>::call(self, approximate);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Gelu(1);
  
  auto status = invoker.Invoke("Gelu", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Gelu, nullptr, onnxruntime::kMSDomain);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_Gelu[0]),
    tensor_options);
}

// aten::gelu_backward(Tensor grad_output, Tensor self, *, str approximate='none') -> Tensor
Tensor gelu_backward(
  const Tensor& grad_output, 
  const Tensor& self, 
  // *, 
  c10::string_view approximate) {
  ORT_LOG_FN(grad_output, self, approximate);
  
  auto promoted_type = PromoteScalarTypesWithCategory({grad_output.scalar_type(),self.scalar_type()}, {});
  
  if (
    !IsSupportedType(grad_output, {at::kBFloat16,at::kFloat,at::kHalf}) || 
    !IsSupportedType(self, {at::kBFloat16,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(gelu_backward)>::call(grad_output, self, approximate);
  }
  auto& invoker = GetORTInvoker(grad_output.device());
  
  auto ort_input_0_grad_output = create_ort_value(invoker, grad_output);
  if (grad_output.scalar_type() != *promoted_type){
    ort_input_0_grad_output = CastToType(invoker, ort_input_0_grad_output, *promoted_type);
  }
  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type){
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_0_GeluGrad(1);
  
  auto status = invoker.Invoke("GeluGrad", {
    std::move(ort_input_0_grad_output),
    std::move(ort_input_0_self),
  }, ort_outputs_0_GeluGrad, nullptr, onnxruntime::kMSDomain);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = grad_output.options().dtype(*promoted_type);
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_GeluGrad[0]),
    tensor_options);
}

// aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
Tensor hardshrink(
  const Tensor& self, 
  const Scalar& lambd) {
  ORT_LOG_FN(self, lambd);
  
  if (
    !IsSupportedType(self, {at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(hardshrink)>::call(self, lambd);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  NodeAttributes attrs_0(2);
  attrs_0["bias"] = create_ort_attribute(
    "bias", 0, at::ScalarType::Float);
  attrs_0["lambd"] = create_ort_attribute(
    "lambd", lambd, at::ScalarType::Float);
  
  std::vector<OrtValue> ort_outputs_0_Shrink(1);
  
  auto status = invoker.Invoke("Shrink", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Shrink, &attrs_0);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_Shrink[0]),
    tensor_options);
}

// aten::selu(Tensor self) -> Tensor
Tensor selu(
  const Tensor& self) {
  ORT_LOG_FN(self);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(selu)>::call(self);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Selu(1);
  
  auto status = invoker.Invoke("Selu", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Selu, nullptr);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_Selu[0]),
    tensor_options);
}

// aten::selu_(Tensor(a!) self) -> Tensor(a!)
Tensor& selu_(
  Tensor& self) {
  ORT_LOG_FN(self);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(selu_)>::call(self);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Selu(1);
  ort_outputs_0_Selu[0] = ort_input_0_self;
  
  auto status = invoker.Invoke("Selu", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Selu, nullptr);
  CHECK_STATUS(status);
  
  return self;
}

// aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& sigmoid_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(sigmoid_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Sigmoid(1);
  ort_outputs_0_Sigmoid[0] = ort_input_out;
  
  auto status = invoker.Invoke("Sigmoid", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Sigmoid, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& sin_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(sin_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Sin(1);
  ort_outputs_0_Sin[0] = ort_input_out;
  
  auto status = invoker.Invoke("Sin", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Sin, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& sinh_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(sinh_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Sinh(1);
  ort_outputs_0_Sinh[0] = ort_input_out;
  
  auto status = invoker.Invoke("Sinh", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Sinh, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
Tensor slice_Tensor(
  const Tensor& self, 
  int64_t dim, 
  c10::optional<int64_t> start, 
  c10::optional<int64_t> end, 
  int64_t step);

// aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
Tensor _softmax(
  const Tensor& self, 
  int64_t dim, 
  bool half_to_float) {
  ORT_LOG_FN(self, dim, half_to_float);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(_softmax)>::call(self, dim, half_to_float);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  NodeAttributes attrs_0(1);
  attrs_0["axis"] = create_ort_attribute(
    "axis", dim, at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_0_Softmax(1);
  
  auto status = invoker.Invoke("Softmax", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Softmax, &attrs_0);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_Softmax[0]),
    tensor_options);
}

// aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
Tensor sum_dim_IntList(
  const Tensor& self, 
  IntArrayRef dim, 
  bool keepdim, 
  // *, 
  c10::optional<ScalarType> dtype) {
  ORT_LOG_FN(self, dim, keepdim, dtype);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong}) || 
    !IsSupportedType(dim, {at::kLong})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(sum_dim_IntList)>::call(self, dim, keepdim, dtype);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  auto ort_input_0_dim = create_ort_value(invoker, dim);
  
  NodeAttributes attrs_0(1);
  attrs_0["keepdims"] = create_ort_attribute(
    "keepdims", keepdim, at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_0_ReduceSum(1);
  
  auto status = invoker.Invoke("ReduceSum", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_dim),
  }, ort_outputs_0_ReduceSum, &attrs_0);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_ReduceSum[0]),
    tensor_options);
}

// aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& sqrt_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(sqrt_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Sqrt(1);
  ort_outputs_0_Sqrt[0] = ort_input_out;
  
  auto status = invoker.Invoke("Sqrt", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Sqrt, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::t(Tensor(a) self) -> Tensor(a)
Tensor t(
  const Tensor& self) {
  ORT_LOG_FN(self);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(t)>::call(self);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Transpose(1);
  
  auto status = invoker.Invoke("Transpose", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Transpose, nullptr);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_Transpose[0]),
    tensor_options);
}

// aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& tan_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(tan_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Tan(1);
  ort_outputs_0_Tan[0] = ort_input_out;
  
  auto status = invoker.Invoke("Tan", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Tan, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& tanh_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(tanh_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Tanh(1);
  ort_outputs_0_Tanh[0] = ort_input_out;
  
  auto status = invoker.Invoke("Tanh", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Tanh, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor
Tensor threshold_backward(
  const Tensor& grad_output, 
  const Tensor& self, 
  const Scalar& threshold) {
  ORT_LOG_FN(grad_output, self, threshold);
  
  if (
    !IsSupportedType(grad_output, {at::kBFloat16,at::kFloat,at::kHalf}) || 
    !IsSupportedType(self, {at::kBFloat16,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(threshold_backward)>::call(grad_output, self, threshold);
  }
  auto& invoker = GetORTInvoker(grad_output.device());
  
  auto ort_input_0_grad_output = create_ort_value(invoker, grad_output);
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_ReluGrad(1);
  
  auto status = invoker.Invoke("ReluGrad", {
    std::move(ort_input_0_grad_output),
    std::move(ort_input_0_self),
  }, ort_outputs_0_ReluGrad, nullptr, onnxruntime::kMSDomain);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = grad_output.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_ReluGrad[0]),
    tensor_options);
}

// aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
Tensor zeros_like(
  const Tensor& self, 
  // *, 
  c10::optional<ScalarType> dtype, 
  c10::optional<Layout> layout, 
  c10::optional<Device> device, 
  c10::optional<bool> pin_memory, 
  c10::optional<MemoryFormat> memory_format) {
  ORT_LOG_FN(self, dtype, layout, device, pin_memory, memory_format);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(zeros_like)>::call(self, dtype, layout, device, pin_memory, memory_format);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Shape(1);
  
  auto status = invoker.Invoke("Shape", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Shape, nullptr);
  CHECK_STATUS(status);
  
  
  std::vector<OrtValue> ort_outputs_1_ConstantOfShape(1);
  
  status = invoker.Invoke("ConstantOfShape", {
    std::move(ort_outputs_0_Shape[0]),
  }, ort_outputs_1_ConstantOfShape, nullptr);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_1_ConstantOfShape[0]),
    tensor_options);
}

// aten::zero_(Tensor(a!) self) -> Tensor(a!)
Tensor& zero_(
  Tensor& self);

// aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor& sub_out(
  const Tensor& self, 
  const Tensor& other, 
  // *, 
  const Scalar& alpha, 
  Tensor& out) {
  ORT_LOG_FN(self, other, alpha, out);
  
  auto promoted_type = PromoteScalarTypesWithCategory({other.scalar_type(),self.scalar_type()}, {alpha.type()});
  
  if (
    !IsSupportedType(alpha, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !c10::canCast(*promoted_type, out.scalar_type())) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(sub_out)>::call(self, other, alpha, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_alpha = create_ort_value(invoker, alpha);
  if (alpha.type() != *promoted_type){
    ort_input_0_alpha = CastToType(invoker, ort_input_0_alpha, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.scalar_type() != *promoted_type){
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_0_Mul(1);
  
  auto status = invoker.Invoke("Mul", {
    std::move(ort_input_0_alpha),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Mul, nullptr);
  CHECK_STATUS(status);
  
  auto ort_input_1_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type){
    ort_input_1_self = CastToType(invoker, ort_input_1_self, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_1_Sub(1);
  if (*promoted_type == out.scalar_type()) {
    ort_outputs_1_Sub[0] = ort_input_out;
  }
  
  status = invoker.Invoke("Sub", {
    std::move(ort_input_1_self),
    std::move(ort_outputs_0_Mul[0]),
  }, ort_outputs_1_Sub, nullptr);
  CHECK_STATUS(status);
  
  if (*promoted_type != out.scalar_type()) {
    CastToType_out(invoker, ort_outputs_1_Sub[0], ort_input_out, out.scalar_type());
  }
  return out;
}

// aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
Tensor addmm(
  const Tensor& self, 
  const Tensor& mat1, 
  const Tensor& mat2, 
  // *, 
  const Scalar& beta, 
  const Scalar& alpha) {
  ORT_LOG_FN(self, mat1, mat2, beta, alpha);
  
  if (
    !IsSupportedType(mat1, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong}) || 
    !IsSupportedType(mat2, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong}) || 
    !IsSupportedType(self, {at::kBFloat16,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(addmm)>::call(self, mat1, mat2, beta, alpha);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_mat1 = create_ort_value(invoker, mat1);
  auto ort_input_0_mat2 = create_ort_value(invoker, mat2);
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  NodeAttributes attrs_0(2);
  attrs_0["alpha"] = create_ort_attribute(
    "alpha", alpha, at::ScalarType::Float);
  attrs_0["beta"] = create_ort_attribute(
    "beta", beta, at::ScalarType::Float);
  
  std::vector<OrtValue> ort_outputs_0_Gemm(1);
  
  auto status = invoker.Invoke("Gemm", {
    std::move(ort_input_0_mat1),
    std::move(ort_input_0_mat2),
    std::move(ort_input_0_self),
  }, ort_outputs_0_Gemm, &attrs_0);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_Gemm[0]),
    tensor_options);
}

// aten::_local_scalar_dense(Tensor self) -> Scalar
Scalar _local_scalar_dense(
  const Tensor& self) {
  ORT_LOG_FN(self);
  
  return native::call_fallback_fn<
    &native::cpu_fallback,
    ATEN_OP(_local_scalar_dense)>::call(self);
}

// aten::view(Tensor(a) self, int[] size) -> Tensor(a)
Tensor view(
  const Tensor& self, 
  IntArrayRef size);

// aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor& bitwise_and_Tensor_out(
  const Tensor& self, 
  const Tensor& other, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, other, out);
  
  if (
    !IsSupportedType(self, {at::kBool}) || 
    !IsSupportedType(other, {at::kBool})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(bitwise_and_Tensor_out)>::call(self, other, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  auto ort_input_0_other = create_ort_value(invoker, other);
  
  std::vector<OrtValue> ort_outputs_0_And(1);
  ort_outputs_0_And[0] = ort_input_out;
  
  auto status = invoker.Invoke("And", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_other),
  }, ort_outputs_0_And, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
Tensor& ne_Scalar_out(
  const Tensor& self, 
  const Scalar& other, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, other, out);
  
  auto promoted_type = PromoteScalarTypesWithCategory({self.scalar_type()}, {other.type()});
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(ne_Scalar_out)>::call(self, other, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type){
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.type() != *promoted_type){
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_0_Equal(1);
  
  auto status = invoker.Invoke("Equal", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Equal, nullptr);
  CHECK_STATUS(status);
  
  
  std::vector<OrtValue> ort_outputs_1_Not(1);
  
  status = invoker.Invoke("Not", {
    std::move(ort_outputs_0_Equal[0]),
  }, ort_outputs_1_Not, nullptr);
  CHECK_STATUS(status);
  
  
  NodeAttributes attrs_2(1);
  attrs_2["to"] = create_ort_attribute(
    "to", GetONNXTensorProtoDataType(out.scalar_type()), at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_2_Cast(1);
  ort_outputs_2_Cast[0] = ort_input_out;
  
  status = invoker.Invoke("Cast", {
    std::move(ort_outputs_1_Not[0]),
  }, ort_outputs_2_Cast, &attrs_2);
  CHECK_STATUS(status);
  
  return out;
}

// aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor& ne_Tensor_out(
  const Tensor& self, 
  const Tensor& other, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, other, out);
  
  auto promoted_type = PromoteScalarTypesWithCategory({self.scalar_type(),other.scalar_type()}, {});
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(ne_Tensor_out)>::call(self, other, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type){
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.scalar_type() != *promoted_type){
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_0_Equal(1);
  
  auto status = invoker.Invoke("Equal", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Equal, nullptr);
  CHECK_STATUS(status);
  
  
  std::vector<OrtValue> ort_outputs_1_Not(1);
  
  status = invoker.Invoke("Not", {
    std::move(ort_outputs_0_Equal[0]),
  }, ort_outputs_1_Not, nullptr);
  CHECK_STATUS(status);
  
  
  NodeAttributes attrs_2(1);
  attrs_2["to"] = create_ort_attribute(
    "to", GetONNXTensorProtoDataType(out.scalar_type()), at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_2_Cast(1);
  ort_outputs_2_Cast[0] = ort_input_out;
  
  status = invoker.Invoke("Cast", {
    std::move(ort_outputs_1_Not[0]),
  }, ort_outputs_2_Cast, &attrs_2);
  CHECK_STATUS(status);
  
  return out;
}

// aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
Tensor& eq_Scalar_out(
  const Tensor& self, 
  const Scalar& other, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, other, out);
  
  auto promoted_type = PromoteScalarTypesWithCategory({self.scalar_type()}, {other.type()});
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(eq_Scalar_out)>::call(self, other, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type){
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.type() != *promoted_type){
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_0_Equal(1);
  
  auto status = invoker.Invoke("Equal", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Equal, nullptr);
  CHECK_STATUS(status);
  
  
  NodeAttributes attrs_1(1);
  attrs_1["to"] = create_ort_attribute(
    "to", GetONNXTensorProtoDataType(out.scalar_type()), at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_1_Cast(1);
  ort_outputs_1_Cast[0] = ort_input_out;
  
  status = invoker.Invoke("Cast", {
    std::move(ort_outputs_0_Equal[0]),
  }, ort_outputs_1_Cast, &attrs_1);
  CHECK_STATUS(status);
  
  return out;
}

// aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor& eq_Tensor_out(
  const Tensor& self, 
  const Tensor& other, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, other, out);
  
  auto promoted_type = PromoteScalarTypesWithCategory({self.scalar_type(),other.scalar_type()}, {});
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(eq_Tensor_out)>::call(self, other, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type){
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.scalar_type() != *promoted_type){
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_0_Equal(1);
  
  auto status = invoker.Invoke("Equal", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Equal, nullptr);
  CHECK_STATUS(status);
  
  
  NodeAttributes attrs_1(1);
  attrs_1["to"] = create_ort_attribute(
    "to", GetONNXTensorProtoDataType(out.scalar_type()), at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_1_Cast(1);
  ort_outputs_1_Cast[0] = ort_input_out;
  
  status = invoker.Invoke("Cast", {
    std::move(ort_outputs_0_Equal[0]),
  }, ort_outputs_1_Cast, &attrs_1);
  CHECK_STATUS(status);
  
  return out;
}

// aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
Tensor& gt_Scalar_out(
  const Tensor& self, 
  const Scalar& other, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, other, out);
  
  auto promoted_type = PromoteScalarTypesWithCategory({self.scalar_type()}, {other.type()});
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(gt_Scalar_out)>::call(self, other, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type){
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.type() != *promoted_type){
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_0_Greater(1);
  
  auto status = invoker.Invoke("Greater", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Greater, nullptr);
  CHECK_STATUS(status);
  
  
  NodeAttributes attrs_1(1);
  attrs_1["to"] = create_ort_attribute(
    "to", GetONNXTensorProtoDataType(out.scalar_type()), at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_1_Cast(1);
  ort_outputs_1_Cast[0] = ort_input_out;
  
  status = invoker.Invoke("Cast", {
    std::move(ort_outputs_0_Greater[0]),
  }, ort_outputs_1_Cast, &attrs_1);
  CHECK_STATUS(status);
  
  return out;
}

// aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor& gt_Tensor_out(
  const Tensor& self, 
  const Tensor& other, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, other, out);
  
  auto promoted_type = PromoteScalarTypesWithCategory({self.scalar_type(),other.scalar_type()}, {});
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(gt_Tensor_out)>::call(self, other, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type){
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.scalar_type() != *promoted_type){
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_0_Greater(1);
  
  auto status = invoker.Invoke("Greater", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Greater, nullptr);
  CHECK_STATUS(status);
  
  
  NodeAttributes attrs_1(1);
  attrs_1["to"] = create_ort_attribute(
    "to", GetONNXTensorProtoDataType(out.scalar_type()), at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_1_Cast(1);
  ort_outputs_1_Cast[0] = ort_input_out;
  
  status = invoker.Invoke("Cast", {
    std::move(ort_outputs_0_Greater[0]),
  }, ort_outputs_1_Cast, &attrs_1);
  CHECK_STATUS(status);
  
  return out;
}

// aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
Tensor& lt_Scalar_out(
  const Tensor& self, 
  const Scalar& other, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, other, out);
  
  auto promoted_type = PromoteScalarTypesWithCategory({self.scalar_type()}, {other.type()});
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(lt_Scalar_out)>::call(self, other, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type){
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.type() != *promoted_type){
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_0_Less(1);
  
  auto status = invoker.Invoke("Less", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Less, nullptr);
  CHECK_STATUS(status);
  
  
  NodeAttributes attrs_1(1);
  attrs_1["to"] = create_ort_attribute(
    "to", GetONNXTensorProtoDataType(out.scalar_type()), at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_1_Cast(1);
  ort_outputs_1_Cast[0] = ort_input_out;
  
  status = invoker.Invoke("Cast", {
    std::move(ort_outputs_0_Less[0]),
  }, ort_outputs_1_Cast, &attrs_1);
  CHECK_STATUS(status);
  
  return out;
}

// aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor& lt_Tensor_out(
  const Tensor& self, 
  const Tensor& other, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, other, out);
  
  auto promoted_type = PromoteScalarTypesWithCategory({self.scalar_type(),other.scalar_type()}, {});
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(lt_Tensor_out)>::call(self, other, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  if (self.scalar_type() != *promoted_type){
    ort_input_0_self = CastToType(invoker, ort_input_0_self, *promoted_type);
  }
  auto ort_input_0_other = create_ort_value(invoker, other);
  if (other.scalar_type() != *promoted_type){
    ort_input_0_other = CastToType(invoker, ort_input_0_other, *promoted_type);
  }
  
  std::vector<OrtValue> ort_outputs_0_Less(1);
  
  auto status = invoker.Invoke("Less", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Less, nullptr);
  CHECK_STATUS(status);
  
  
  NodeAttributes attrs_1(1);
  attrs_1["to"] = create_ort_attribute(
    "to", GetONNXTensorProtoDataType(out.scalar_type()), at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_1_Cast(1);
  ort_outputs_1_Cast[0] = ort_input_out;
  
  status = invoker.Invoke("Cast", {
    std::move(ort_outputs_0_Less[0]),
  }, ort_outputs_1_Cast, &attrs_1);
  CHECK_STATUS(status);
  
  return out;
}

// aten::masked_select(Tensor self, Tensor mask) -> Tensor
Tensor masked_select(
  const Tensor& self, 
  const Tensor& mask) {
  ORT_LOG_FN(self, mask);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(mask, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(self, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(masked_select)>::call(self, mask);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Shape(1);
  
  auto status = invoker.Invoke("Shape", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Shape, nullptr);
  CHECK_STATUS(status);
  
  auto ort_input_1_mask = create_ort_value(invoker, mask);
  
  std::vector<OrtValue> ort_outputs_1_Expand(1);
  
  status = invoker.Invoke("Expand", {
    std::move(ort_input_1_mask),
    std::move(ort_outputs_0_Shape[0]),
  }, ort_outputs_1_Expand, nullptr);
  CHECK_STATUS(status);
  
  
  std::vector<OrtValue> ort_outputs_2_NonZero(1);
  
  status = invoker.Invoke("NonZero", {
    std::move(ort_outputs_1_Expand[0]),
  }, ort_outputs_2_NonZero, nullptr);
  CHECK_STATUS(status);
  
  
  std::vector<OrtValue> ort_outputs_3_Transpose(1);
  
  status = invoker.Invoke("Transpose", {
    std::move(ort_outputs_2_NonZero[0]),
  }, ort_outputs_3_Transpose, nullptr);
  CHECK_STATUS(status);
  
  auto ort_input_4_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_4_GatherND(1);
  
  status = invoker.Invoke("GatherND", {
    std::move(ort_input_4_self),
    std::move(ort_outputs_3_Transpose[0]),
  }, ort_outputs_4_GatherND, nullptr);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_4_GatherND[0]),
    tensor_options);
}

// aten::nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& nonzero_out(
  const Tensor& self, 
  // *, 
  Tensor& out);

// aten::nonzero(Tensor self) -> Tensor
Tensor nonzero(
  const Tensor& self) {
  ORT_LOG_FN(self);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kBool,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(nonzero)>::call(self);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_NonZero(1);
  
  auto status = invoker.Invoke("NonZero", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_NonZero, nullptr);
  CHECK_STATUS(status);
  
  
  std::vector<OrtValue> ort_outputs_1_Transpose(1);
  
  status = invoker.Invoke("Transpose", {
    std::move(ort_outputs_0_NonZero[0]),
  }, ort_outputs_1_Transpose, nullptr);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options().dtype(at::ScalarType::Long);
  return aten_tensor_from_ort(
    std::move(ort_outputs_1_Transpose[0]),
    tensor_options);
}

// aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& sign_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(sign_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Sign(1);
  ort_outputs_0_Sign[0] = ort_input_out;
  
  auto status = invoker.Invoke("Sign", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Sign, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor
Tensor fmod_Scalar(
  const Tensor& self, 
  const Scalar& other) {
  ORT_LOG_FN(self, other);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(fmod_Scalar)>::call(self, other);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  auto ort_input_0_other = create_ort_value(invoker, other);
  
  NodeAttributes attrs_0(1);
  attrs_0["fmod"] = create_ort_attribute(
    "fmod", 1, at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_0_Mod(1);
  
  auto status = invoker.Invoke("Mod", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Mod, &attrs_0);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_Mod[0]),
    tensor_options);
}

// aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor
Tensor fmod_Tensor(
  const Tensor& self, 
  const Tensor& other) {
  ORT_LOG_FN(self, other);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort}) || 
    !IsSupportedType(other, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(fmod_Tensor)>::call(self, other);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  auto ort_input_0_other = create_ort_value(invoker, other);
  
  NodeAttributes attrs_0(1);
  attrs_0["fmod"] = create_ort_attribute(
    "fmod", 1, at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_0_Mod(1);
  
  auto status = invoker.Invoke("Mod", {
    std::move(ort_input_0_self),
    std::move(ort_input_0_other),
  }, ort_outputs_0_Mod, &attrs_0);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_Mod[0]),
    tensor_options);
}

// aten::min(Tensor self) -> Tensor
Tensor min(
  const Tensor& self) {
  ORT_LOG_FN(self);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(min)>::call(self);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  NodeAttributes attrs_0(1);
  attrs_0["keepdims"] = create_ort_attribute(
    "keepdims", 0, at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_0_ReduceMin(1);
  
  auto status = invoker.Invoke("ReduceMin", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_ReduceMin, &attrs_0);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_ReduceMin[0]),
    tensor_options);
}

// aten::max(Tensor self) -> Tensor
Tensor max(
  const Tensor& self) {
  ORT_LOG_FN(self);
  
  if (
    !IsSupportedType(self, {at::kBFloat16,at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(max)>::call(self);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  NodeAttributes attrs_0(1);
  attrs_0["keepdims"] = create_ort_attribute(
    "keepdims", 0, at::ScalarType::Int);
  
  std::vector<OrtValue> ort_outputs_0_ReduceMax(1);
  
  auto status = invoker.Invoke("ReduceMax", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_ReduceMax, &attrs_0);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_ReduceMax[0]),
    tensor_options);
}

// aten::equal(Tensor self, Tensor other) -> bool
bool equal(
  const Tensor& self, 
  const Tensor& other);

// aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
::std::tuple<Tensor&, Tensor&> nll_loss_forward_output(
  const Tensor& self, 
  const Tensor& target, 
  const c10::optional<Tensor>& weight, 
  int64_t reduction, 
  int64_t ignore_index, 
  // *, 
  Tensor& output, 
  Tensor& total_weight) {
  ORT_LOG_FN(self, target, weight, reduction, ignore_index, output, total_weight);
  
  return native::call_fallback_fn<
    &native::cpu_fallback,
    ATEN_OP(nll_loss_forward_output)>::call(self, target, weight, reduction, ignore_index, output, total_weight);
}

// aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor& nll_loss_backward_grad_input(
  const Tensor& grad_output, 
  const Tensor& self, 
  const Tensor& target, 
  const c10::optional<Tensor>& weight, 
  int64_t reduction, 
  int64_t ignore_index, 
  const Tensor& total_weight, 
  // *, 
  Tensor& grad_input) {
  ORT_LOG_FN(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
  
  return native::call_fallback_fn<
    &native::cpu_fallback,
    ATEN_OP(nll_loss_backward_grad_input)>::call(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
}

// aten::hardsigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& hardsigmoid_out(
  const Tensor& self, 
  // *, 
  Tensor& out) {
  ORT_LOG_FN(self, out);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(hardsigmoid_out)>::call(self, out);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  // resize the output and then create output ort value to be updated.
  resize_output(invoker, dynamic_cast<ORTTensorImpl*>(out.unsafeGetTensorImpl()), self.sizes());
  auto ort_input_out = create_ort_value(invoker, out);
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_HardSigmoid(1);
  ort_outputs_0_HardSigmoid[0] = ort_input_out;
  
  auto status = invoker.Invoke("HardSigmoid", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_HardSigmoid, nullptr);
  CHECK_STATUS(status);
  
  return out;
}

// aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor
Tensor softshrink(
  const Tensor& self, 
  const Scalar& lambd) {
  ORT_LOG_FN(self, lambd);
  
  if (
    !IsSupportedType(self, {at::kByte,at::kDouble,at::kFloat,at::kHalf,at::kInt,at::kLong,at::kShort})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(softshrink)>::call(self, lambd);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  NodeAttributes attrs_0(2);
  attrs_0["bias"] = create_ort_attribute(
    "bias", lambd, at::ScalarType::Float);
  attrs_0["lambd"] = create_ort_attribute(
    "lambd", lambd, at::ScalarType::Float);
  
  std::vector<OrtValue> ort_outputs_0_Shrink(1);
  
  auto status = invoker.Invoke("Shrink", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Shrink, &attrs_0);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_Shrink[0]),
    tensor_options);
}

// aten::isinf(Tensor self) -> Tensor
Tensor isinf(
  const Tensor& self) {
  ORT_LOG_FN(self);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(isinf)>::call(self);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_IsInf(1);
  
  auto status = invoker.Invoke("IsInf", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_IsInf, nullptr);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_IsInf[0]),
    tensor_options);
}

// aten::det(Tensor self) -> Tensor
Tensor det(
  const Tensor& self) {
  ORT_LOG_FN(self);
  
  if (
    !IsSupportedType(self, {at::kDouble,at::kFloat,at::kHalf})) {
    return native::call_fallback_fn<
      &native::cpu_fallback,
      ATEN_OP(det)>::call(self);
  }
  auto& invoker = GetORTInvoker(self.device());
  
  auto ort_input_0_self = create_ort_value(invoker, self);
  
  std::vector<OrtValue> ort_outputs_0_Det(1);
  
  auto status = invoker.Invoke("Det", {
    std::move(ort_input_0_self),
  }, ort_outputs_0_Det, nullptr);
  CHECK_STATUS(status);
  
  at::TensorOptions tensor_options = self.options();
  return aten_tensor_from_ort(
    std::move(ort_outputs_0_Det[0]),
    tensor_options);
}
} // namespace aten

TORCH_LIBRARY_IMPL(aten, ORT, m) {
  m.impl("aten::abs.out", TORCH_FN(aten::abs_out));
  m.impl("aten::acos.out", TORCH_FN(aten::acos_out));
  m.impl("aten::add.out", TORCH_FN(aten::add_out));
  m.impl("aten::argmax.out", TORCH_FN(aten::argmax_out));
  m.impl("aten::acosh.out", TORCH_FN(aten::acosh_out));
  m.impl("aten::asinh.out", TORCH_FN(aten::asinh_out));
  m.impl("aten::atanh.out", TORCH_FN(aten::atanh_out));
  m.impl("aten::as_strided", TORCH_FN(aten::as_strided));
  m.impl("aten::asin.out", TORCH_FN(aten::asin_out));
  m.impl("aten::atan.out", TORCH_FN(aten::atan_out));
  m.impl("aten::ceil.out", TORCH_FN(aten::ceil_out));
  m.impl("aten::copy_", TORCH_FN(aten::copy_));
  m.impl("aten::_copy_from_and_resize", TORCH_FN(aten::_copy_from_and_resize));
  m.impl("aten::cos.out", TORCH_FN(aten::cos_out));
  m.impl("aten::cosh.out", TORCH_FN(aten::cosh_out));
  m.impl("aten::div.out", TORCH_FN(aten::div_out));
  m.impl("aten::empty.memory_format", TORCH_FN(aten::empty_memory_format));
  m.impl("aten::resize_", TORCH_FN(aten::resize_));
  m.impl("aten::empty_strided", TORCH_FN(aten::empty_strided));
  m.impl("aten::erf.out", TORCH_FN(aten::erf_out));
  m.impl("aten::exp.out", TORCH_FN(aten::exp_out));
  m.impl("aten::fill_.Scalar", TORCH_FN(aten::fill__Scalar));
  m.impl("aten::floor.out", TORCH_FN(aten::floor_out));
  m.impl("aten::isnan", TORCH_FN(aten::isnan));
  m.impl("aten::log.out", TORCH_FN(aten::log_out));
  m.impl("aten::_log_softmax.out", TORCH_FN(aten::_log_softmax_out));
  m.impl("aten::_log_softmax_backward_data.out", TORCH_FN(aten::_log_softmax_backward_data_out));
  m.impl("aten::mm.out", TORCH_FN(aten::mm_out));
  m.impl("aten::mul.out", TORCH_FN(aten::mul_out));
  m.impl("aten::reciprocal.out", TORCH_FN(aten::reciprocal_out));
  m.impl("aten::neg.out", TORCH_FN(aten::neg_out));
  m.impl("aten::_reshape_alias", TORCH_FN(aten::_reshape_alias));
  m.impl("aten::round.out", TORCH_FN(aten::round_out));
  m.impl("aten::relu", TORCH_FN(aten::relu));
  m.impl("aten::relu_", TORCH_FN(aten::relu_));
  m.impl("aten::gelu", TORCH_FN(aten::gelu));
  m.impl("aten::gelu_backward", TORCH_FN(aten::gelu_backward));
  m.impl("aten::hardshrink", TORCH_FN(aten::hardshrink));
  m.impl("aten::selu", TORCH_FN(aten::selu));
  m.impl("aten::selu_", TORCH_FN(aten::selu_));
  m.impl("aten::sigmoid.out", TORCH_FN(aten::sigmoid_out));
  m.impl("aten::sin.out", TORCH_FN(aten::sin_out));
  m.impl("aten::sinh.out", TORCH_FN(aten::sinh_out));
  m.impl("aten::slice.Tensor", TORCH_FN(aten::slice_Tensor));
  m.impl("aten::_softmax", TORCH_FN(aten::_softmax));
  m.impl("aten::sum.dim_IntList", TORCH_FN(aten::sum_dim_IntList));
  m.impl("aten::sqrt.out", TORCH_FN(aten::sqrt_out));
  m.impl("aten::t", TORCH_FN(aten::t));
  m.impl("aten::tan.out", TORCH_FN(aten::tan_out));
  m.impl("aten::tanh.out", TORCH_FN(aten::tanh_out));
  m.impl("aten::threshold_backward", TORCH_FN(aten::threshold_backward));
  m.impl("aten::zeros_like", TORCH_FN(aten::zeros_like));
  m.impl("aten::zero_", TORCH_FN(aten::zero_));
  m.impl("aten::sub.out", TORCH_FN(aten::sub_out));
  m.impl("aten::addmm", TORCH_FN(aten::addmm));
  m.impl("aten::_local_scalar_dense", TORCH_FN(aten::_local_scalar_dense));
  m.impl("aten::view", TORCH_FN(aten::view));
  m.impl("aten::bitwise_and.Tensor_out", TORCH_FN(aten::bitwise_and_Tensor_out));
  m.impl("aten::ne.Scalar_out", TORCH_FN(aten::ne_Scalar_out));
  m.impl("aten::ne.Tensor_out", TORCH_FN(aten::ne_Tensor_out));
  m.impl("aten::eq.Scalar_out", TORCH_FN(aten::eq_Scalar_out));
  m.impl("aten::eq.Tensor_out", TORCH_FN(aten::eq_Tensor_out));
  m.impl("aten::gt.Scalar_out", TORCH_FN(aten::gt_Scalar_out));
  m.impl("aten::gt.Tensor_out", TORCH_FN(aten::gt_Tensor_out));
  m.impl("aten::lt.Scalar_out", TORCH_FN(aten::lt_Scalar_out));
  m.impl("aten::lt.Tensor_out", TORCH_FN(aten::lt_Tensor_out));
  m.impl("aten::masked_select", TORCH_FN(aten::masked_select));
  m.impl("aten::nonzero.out", TORCH_FN(aten::nonzero_out));
  m.impl("aten::nonzero", TORCH_FN(aten::nonzero));
  m.impl("aten::sign.out", TORCH_FN(aten::sign_out));
  m.impl("aten::fmod.Scalar", TORCH_FN(aten::fmod_Scalar));
  m.impl("aten::fmod.Tensor", TORCH_FN(aten::fmod_Tensor));
  m.impl("aten::min", TORCH_FN(aten::min));
  m.impl("aten::max", TORCH_FN(aten::max));
  m.impl("aten::equal", TORCH_FN(aten::equal));
  m.impl("aten::nll_loss_forward.output", TORCH_FN(aten::nll_loss_forward_output));
  m.impl("aten::nll_loss_backward.grad_input", TORCH_FN(aten::nll_loss_backward_grad_input));
  m.impl("aten::hardsigmoid.out", TORCH_FN(aten::hardsigmoid_out));
  m.impl("aten::softshrink", TORCH_FN(aten::softshrink));
  m.impl("aten::isinf", TORCH_FN(aten::isinf));
  m.impl("aten::det", TORCH_FN(aten::det));
}

} // namespace eager
} // namespace torch_ort
