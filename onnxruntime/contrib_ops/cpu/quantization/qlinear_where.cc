#include "qlinear_where.h"
#include "qlinear_lookup_table.h"

#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {
namespace {

template <typename T, typename R>
using EnableIfEigenScalar = typename std::enable_if<std::is_arithmetic<T>::value, R>::type;

template <typename T, typename R>
using EnableIfEigenNotScalar = typename std::enable_if<!std::is_arithmetic<T>::value, R>::type;

template <typename T>
ProcessBroadcastSpanFuncs CreateScalarBroadcastFuncs() {
  return ProcessBroadcastSpanFuncs{
      // Scalar Condition + Eigen Input -> Eigen output
      [](BroadcastHelper& per_iter_bh) {
        auto* user_data = static_cast<T*>(per_iter_bh.GetUserData());
        bool target = user_data[0] == 1;
        bool is_copy = user_data[1] == 1;
        bool condition = per_iter_bh.ScalarInput0<bool>();
        auto value = per_iter_bh.EigenInput1<T>();
        auto output = per_iter_bh.OutputEigen<T>();
        if (condition == target) {
          output = value;
        } else {
          output = EigenVectorMap<T>::PlainObject::Constant(value.size(), T{});
        }
        // Transform the output to the correct value from LookupTable
        if (!is_copy) {
          auto* look_up_table = user_data + 2;
          std::transform(value.cbegin(), value.cend(), output.begin(),
                         [condition, target, &look_up_table](const T& value_element) {
                           return condition == target ? look_up_table[value_element] : T{};
                         });
        }
      },
      // Eigen Condition + Scalar Input -> Eigen Output
      [](BroadcastHelper& per_iter_bh) {
        auto* user_data = static_cast<T*>(per_iter_bh.GetUserData());
        bool target = user_data[0] == 1;
        bool is_copy = user_data[1] == 1;
        auto condition = per_iter_bh.EigenInput0<bool>();
        const T& value = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputEigen<T>();
        output = (condition.array() == target)
                     .select(value, EigenVectorMap<T>::PlainObject::Constant(condition.size(), T{}));
        // Transform the output to the correct value from LookupTable
        if (!is_copy) {
          auto* look_up_table = user_data + 2;
          std::transform(condition.cbegin(), condition.cend(), output.begin(),
                         [target, &look_up_table, &value](bool condition_element) {
                           return condition_element == target ? look_up_table[value] : T{};
                         });
        }
      },
      // Eigen Condition + Eigen Input -> Eigen Output
      [](BroadcastHelper& per_iter_bh) {
        auto* user_data = static_cast<T*>(per_iter_bh.GetUserData());
        bool target = user_data[0] == 1;
        bool is_copy = user_data[1] == 1;
        auto condition = per_iter_bh.EigenInput0<bool>();
        auto value = per_iter_bh.EigenInput1<T>();
        auto output = per_iter_bh.OutputEigen<T>();
        output = (condition.array() == target)
                     .select(value, EigenVectorMap<T>::PlainObject::Constant(condition.size(), T{}));
        // Transform the output to the correct value from LookupTable
        if (!is_copy) {
          auto* look_up_table = user_data + 2;
          std::transform(condition.cbegin(), condition.cend(), value.cbegin(), output.begin(),
                         [target, &look_up_table](bool condition_element, const T& value_element) {
                           return (condition_element == target) ? look_up_table[value_element] : T{};
                         });
        }
      }};
}

template <typename T>
ProcessBroadcastSpanFuncs CreateNonScalarBroadcastFuncs() {
  return ProcessBroadcastSpanFuncs{
      [](BroadcastHelper& per_iter_bh) {
        auto* user_data = static_cast<T*>(per_iter_bh.GetUserData());
        bool target = user_data[0] == 1;
        bool is_copy = user_data[1] == 1;
        auto* look_up_table = user_data + 2;
        bool condition = per_iter_bh.ScalarInput0<bool>();
        auto value = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();
        if (condition == target) {
          // Transform the output to the correct value from LookupTable
          std::transform(value.cbegin(), value.cend(), output.begin(),
                         [condition, target, &look_up_table,is_copy](const T& value_element) {
                           return is_copy ? value_element : look_up_table[value_element];
                         });
        } else {
          std::fill(output.begin(), output.end(), T{});
        }
      },
      [](BroadcastHelper& per_iter_bh) {
        auto* user_data = static_cast<T*>(per_iter_bh.GetUserData());
        bool target = user_data[0] == 1;
        bool is_copy = user_data[1] == 1;
        auto* look_up_table = user_data + 2;
        auto condition = per_iter_bh.SpanInput0<bool>();
        const T& value = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();
        // Transform the output to the correct value from LookupTable
        std::transform(condition.begin(), condition.end(), output.begin(),
                       [target, &value,&look_up_table,is_copy](bool condition_element) {
                         return condition_element == target ? is_copy ? value : look_up_table[value] : T{};
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        auto* user_data = static_cast<T*>(per_iter_bh.GetUserData());
        bool target = user_data[0] == 1;
        bool is_copy = user_data[1] == 1;
        auto* look_up_table = user_data + 2;
        auto condition = per_iter_bh.SpanInput0<bool>();
        auto value = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();
        // Transform the output to the correct value from LookupTable
        std::transform(condition.begin(), condition.end(), value.cbegin(), output.begin(),
                       [target,&look_up_table,is_copy](bool condition_element, const T& value_element) {
                         return condition_element == target ? is_copy ? value_element : look_up_table[value_element] : T{};
                       });
      }};
}

template <typename T>
EnableIfEigenScalar<T, ProcessBroadcastSpanFuncs> SelectBroadcastFuncs() {
  // NOTE: Workaround a VS2017 bug by calling a separate function to create the broadcast funcs.
  // If we create them directly here it doesn't bring in the definitions of the Eigen classes leading to
  // a 'class has no constructors' error
  return CreateScalarBroadcastFuncs<T>();
}

template <typename T>
EnableIfEigenNotScalar<T, ProcessBroadcastSpanFuncs> SelectBroadcastFuncs() {
  return CreateNonScalarBroadcastFuncs<T>();
}
// function pointer to create typed tensor from type agnostic code whilst avoiding the overhead of std::function
using AllocTensorFunc = std::unique_ptr<Tensor> (*)(const TensorAllocator& allocator, const TensorShape& shape);

static std::unique_ptr<Tensor> UntypedSelect(OpKernelContext& context, std::vector<uint8_t>& user_data, const ProcessBroadcastSpanFuncs& functors, const TensorAllocator& allocator, AllocTensorFunc allocate_tensor) {
  const auto& condition = *context.Input<Tensor>(0);
  // select the X input (input 1) for 'true', and Y input (input 2) for 'false'
  bool target = user_data[0] == 1;
  const auto& values = *context.Input<Tensor>(target ? 1 : 4);

  InputBroadcaster input_broadcaster(condition, values);
  std::unique_ptr<Tensor> selection_tensor = allocate_tensor(allocator, input_broadcaster.GetOutputShape());
  OutputBroadcaster output_broadcaster(input_broadcaster.GetSpanSize(), *selection_tensor);
  // store value of 'target' directly in void* for user_data so it's accessible in the state-less functors
  BroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster, reinterpret_cast<void*>(user_data.data()));
  BroadcastLooper(broadcast_helper, functors);
  return selection_tensor;
}

// Merging Functions
template <typename T>
void MergeScalarAndVector(EigenVectorMap<T> output, const T& scalar_value, ConstEigenVectorMap<T> vector_value) {
  if (scalar_value != T{}) {
    output = EigenVectorMap<T>::PlainObject::Constant(vector_value.size(), scalar_value);
  } else {
    output = vector_value;
  }
};

template <typename T>
EnableIfEigenScalar<T, ProcessBroadcastSpanFuncs> MergeBroadcastFuncs() {
  return ProcessBroadcastSpanFuncs{
      [](BroadcastHelper& per_iter_bh) {
        MergeScalarAndVector(per_iter_bh.OutputEigen<T>(),
                             per_iter_bh.ScalarInput0<T>(),  // X selection
                             per_iter_bh.EigenInput1<T>());  // Y selection
      },
      [](BroadcastHelper& per_iter_bh) {
        MergeScalarAndVector(per_iter_bh.OutputEigen<T>(),
                             per_iter_bh.ScalarInput1<T>(),  // Y selection
                             per_iter_bh.EigenInput0<T>());  // X selection
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X_selection = per_iter_bh.EigenInput0<T>();
        auto Y_selection = per_iter_bh.EigenInput1<T>();
        per_iter_bh.OutputEigen<T>() = X_selection.binaryExpr(Y_selection,
                                                              [](T x, T y) -> T {
                                                                return x != T{} ? x : y;
                                                              });
      }};
}

template <typename T>
void MergeScalarAndVector(gsl::span<T> output, const T& scalar_value, gsl::span<const T> vector_value) {
  if (!scalar_value.empty()) {
    std::fill(output.begin(), output.end(), scalar_value);
  } else {
    std::copy(vector_value.cbegin(), vector_value.cend(), output.begin());
  }
};

template <typename T>

EnableIfEigenNotScalar<T, ProcessBroadcastSpanFuncs> MergeBroadcastFuncs() {
  return ProcessBroadcastSpanFuncs{
      [](BroadcastHelper& per_iter_bh) {
        MergeScalarAndVector(per_iter_bh.OutputSpan<T>(),
                             per_iter_bh.ScalarInput0<T>(),  // X selection
                             per_iter_bh.SpanInput1<T>());   // Y selection
      },
      [](BroadcastHelper& per_iter_bh) {
        MergeScalarAndVector(per_iter_bh.OutputSpan<T>(),
                             per_iter_bh.ScalarInput1<T>(),  // Y selection
                             per_iter_bh.SpanInput0<T>());   // X selection
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X_selection = per_iter_bh.SpanInput0<T>();
        auto Y_selection = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();
        std::transform(X_selection.cbegin(), X_selection.cend(), Y_selection.cbegin(), output.begin(),
                       [](const T& x, const T& y) { return !x.empty() ? x : y; });
      }};
}
static void UntypedMerge(OpKernelContext& context,
                         const Tensor& X_selection_tensor, const Tensor& Y_selection_tensor,
                         const ProcessBroadcastSpanFuncs& functors) {
  InputBroadcaster merge_broadcaster{X_selection_tensor, Y_selection_tensor};
  Tensor& output = *context.Output(0, merge_broadcaster.GetOutputShape());

  OutputBroadcaster output_broadcaster{merge_broadcaster.GetSpanSize(), output};
  BroadcastHelper broadcast_helper(merge_broadcaster, output_broadcaster);

  BroadcastLooper(broadcast_helper, functors);
}
}  // namespace

namespace contrib {

QLinearWhere::QLinearWhere(const OpKernelInfo& info) : OpKernel(info) {
  size_t input_def_count = info.node().InputDefs().size();
  ORT_ENFORCE(input_def_count == kExpected_input_count,
              "There must be ", kExpected_input_count, " inputs! (condition, x, x_scale, x_zero_point, y, y_scale, y_zero_point, z_scale, z_zero_point)");
  const Tensor* tensor_x_scale = nullptr;
  const Tensor* tensor_x_zero_point = nullptr;
  const Tensor* tensor_y_scale = nullptr;
  const Tensor* tensor_y_zero_point = nullptr;
  const Tensor* tensor_z_scale = nullptr;
  const Tensor* tensor_z_zero_point = nullptr;

  bool get_x_scale = info.TryGetConstantInput(2, &tensor_x_scale);
  bool get_x_zero_point = info.TryGetConstantInput(3, &tensor_x_zero_point);
  bool get_y_scale = info.TryGetConstantInput(5, &tensor_y_scale);
  bool get_y_zero_point = info.TryGetConstantInput(6, &tensor_y_zero_point);
  bool get_z_scale = info.TryGetConstantInput(7, &tensor_z_scale);
  bool get_z_zero_point = info.TryGetConstantInput(8, &tensor_z_zero_point);
  if (!get_z_scale || !get_z_zero_point) {
    // Can not build fix lookup table
    return;
  }
  ORT_ENFORCE(
      tensor_x_zero_point->GetElementType() == tensor_y_zero_point->GetElementType() &&
          tensor_x_zero_point->GetElementType() == tensor_z_zero_point->GetElementType() &&
          tensor_y_zero_point->GetElementType() == tensor_z_zero_point->GetElementType(),
      "Wrong input type encountered for zero point input def of x, y, z");
  bool is_signed_int8 = tensor_z_zero_point->IsDataType<int8_t>();
  const auto identity_float = [](float v) -> float { return v; };

  if (get_x_scale && get_x_zero_point) {
    // Build fix lookup table for x
    is_x_fixed_copy_ = has_same_scale(tensor_x_scale, tensor_z_scale) &&
                       has_same_zero_point(is_signed_int8, tensor_x_zero_point, tensor_z_zero_point);
    if (!is_x_fixed_copy_) {
      x_fixed_lookup_table_.resize(256);
      if (is_signed_int8) {
        QlinearBuildLookupTable<int8_t>(
            x_fixed_lookup_table_.data(), tensor_x_scale, tensor_x_zero_point,
            tensor_z_scale, tensor_z_zero_point, identity_float);
      } else {
        QlinearBuildLookupTable<uint8_t>(
            x_fixed_lookup_table_.data(), tensor_x_scale, tensor_x_zero_point,
            tensor_z_scale, tensor_z_zero_point, identity_float);
      }
    }
    is_x_dynamic_ = false;
  }

  if (get_y_scale && get_y_zero_point) {
    // Build fix lookup table for y
    is_y_fixed_copy_ = has_same_scale(tensor_y_scale, tensor_z_scale) &&
                       has_same_zero_point(is_signed_int8, tensor_y_zero_point, tensor_z_zero_point);
    if (!is_y_fixed_copy_) {
      y_fixed_lookup_table_.resize(256);
      if (is_signed_int8) {
        QlinearBuildLookupTable<int8_t>(
            y_fixed_lookup_table_.data(), tensor_y_scale, tensor_y_zero_point,
            tensor_z_scale, tensor_z_zero_point, identity_float);
      } else {
        QlinearBuildLookupTable<uint8_t>(
            y_fixed_lookup_table_.data(), tensor_y_scale, tensor_y_zero_point,
            tensor_z_scale, tensor_z_zero_point, identity_float);
      }
    }
    is_y_dynamic_ = false;
  }
}

Status QLinearWhere::Compute(OpKernelContext* ctx) const {
//  const auto* tensor_condition = ctx->Input<Tensor>(0);
//  const auto* tensor_x_input = ctx->Input<Tensor>(1);
  const auto* tensor_x_scale = ctx->Input<Tensor>(2);
  const auto* tensor_x_zero_point = ctx->Input<Tensor>(3);
//  const auto* tensor_y_input = ctx->Input<Tensor>(4);
  const auto* tensor_y_scale = ctx->Input<Tensor>(5);
  const auto* tensor_y_zero_point = ctx->Input<Tensor>(6);
  const auto* tensor_z_scale = ctx->Input<Tensor>(7);
  const auto* tensor_z_zero_point = ctx->Input<Tensor>(8);
//  auto* tensor_output = ctx->Output(0, tensor_condition->Shape());
  ORT_ENFORCE(tensor_x_scale->IsDataType<float>(), "Input scale is not float for quantized input x @ 2");
  ORT_ENFORCE(tensor_y_scale->IsDataType<float>(), "Input scale is not float for quantized input y @ 5");
  ORT_ENFORCE(tensor_z_scale->IsDataType<float>(), "Input scale is not float for quantized output z @ 7");
  ORT_ENFORCE(tensor_x_zero_point->GetElementType() == tensor_y_zero_point->GetElementType() &&
                  tensor_x_zero_point->GetElementType() == tensor_z_zero_point->GetElementType() &&
                  tensor_y_zero_point->GetElementType() == tensor_z_zero_point->GetElementType(),
              "Wrong input type encountered for zero point of quantized input @", 3, 6, 8);
  bool is_signed_int8 = tensor_z_zero_point->IsDataType<int8_t>();
  const auto identity_float = [](float v) -> float { return v; };

  std::vector<uint8_t> x_dynamic_lookup_table;
  bool is_x_copy = !is_x_dynamic_ ? is_x_fixed_copy_ : has_same_scale(tensor_x_scale, tensor_z_scale) && has_same_zero_point(is_signed_int8, tensor_x_zero_point, tensor_z_zero_point);
  if (is_x_dynamic_ && !is_x_copy) {
    x_dynamic_lookup_table.resize(256);
    if (is_signed_int8) {
      QlinearBuildLookupTable<int8_t>(
          x_dynamic_lookup_table.data(), tensor_x_scale, tensor_x_zero_point,
          tensor_z_scale, tensor_z_zero_point, identity_float);
    } else {
      QlinearBuildLookupTable<uint8_t>(
          x_dynamic_lookup_table.data(), tensor_x_scale, tensor_x_zero_point,
          tensor_z_scale, tensor_z_zero_point, identity_float);
    }
  }

  // Build dynamic lookup table for y
  std::vector<uint8_t> y_dynamic_lookup_table;
  bool is_y_copy = !is_y_dynamic_ ? is_y_fixed_copy_ : has_same_scale(tensor_y_scale, tensor_z_scale) && has_same_zero_point(is_signed_int8, tensor_y_zero_point, tensor_z_zero_point);
  if (is_y_dynamic_ && !is_y_copy) {
    y_dynamic_lookup_table.resize(256);
    if (is_signed_int8) {
      QlinearBuildLookupTable<int8_t>(
          y_dynamic_lookup_table.data(), tensor_y_scale, tensor_y_zero_point,
          tensor_z_scale, tensor_z_zero_point, identity_float);
    } else {
      QlinearBuildLookupTable<uint8_t>(
          y_dynamic_lookup_table.data(), tensor_y_scale, tensor_y_zero_point,
          tensor_z_scale, tensor_z_zero_point, identity_float);
    }
  }
  // Each lookup table is 256 bytes
  const auto& x_lookup_table = is_x_dynamic_ ? x_dynamic_lookup_table : x_fixed_lookup_table_;
  const auto& y_lookup_table = is_y_dynamic_ ? y_dynamic_lookup_table : y_fixed_lookup_table_;

  // Each user_data will sized at 2 + 256 bytes, and contain {is_x/y, is_x/y_copy, x/y_lookup_table} respectively
  std::vector<uint8_t> x_user_data(258);
  std::vector<uint8_t> y_user_data(258);
  x_user_data[0] = 1;
  y_user_data[0] = 0;
  x_user_data[1] = is_x_copy ? 1 : 0;
  y_user_data[1] = is_y_copy ? 1 : 0;
  if (!is_x_copy) {
    std::copy(x_lookup_table.begin(), x_lookup_table.end(), x_user_data.begin() + 2);
  }
  if (!is_y_copy) {
    std::copy(y_lookup_table.begin(), y_lookup_table.end(), y_user_data.begin() + 2);
  }
  //Allocator, Allocation, and SelectBroadcastFuncs are the same implementation from where_op.cc
  const auto typed_tensor_allocation = [](const TensorAllocator& allocator,
                                          const TensorShape& shape) {
    return allocator.Allocate<uint8_t>(shape);
  };
  TensorAllocator tensor_allocator{*ctx};
  ProcessBroadcastSpanFuncs funcs = SelectBroadcastFuncs<uint8_t>();
  // UntypedSelect is MODIFIED from where_op.cc
  auto X_selection_tensor = UntypedSelect(*ctx, x_user_data, funcs, tensor_allocator, typed_tensor_allocation);
  auto Y_selection_tensor = UntypedSelect(*ctx, y_user_data, funcs, tensor_allocator, typed_tensor_allocation);
  // UntypedMerge is the same as from where_op.cc
  UntypedMerge(*ctx, *X_selection_tensor, *Y_selection_tensor, MergeBroadcastFuncs<uint8_t>());

  return Status();
}

ONNX_CPU_OPERATOR_MS_KERNEL(
    QLinearWhere,
    1,
    KernelDefBuilder().TypeConstraint(
        "T",
        {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()}),
    QLinearWhere)
}  // namespace contrib
}  // namespace onnxruntime