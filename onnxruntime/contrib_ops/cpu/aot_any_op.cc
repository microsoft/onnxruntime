// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cctype>
#include <cstdint>
#include <charconv>

#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/common/status.h"
#include "core/framework/tensorprotoutils.h"
#include "core/common/string_utils.h"

#include "aot_any_op.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace contrib {

template < typename T>
AOTanyOp<T>::AOTanyOp(const OpKernelInfo& info) : OpKernel(info) {
  std::string lib_path;
  if (info.GetAttr<std::string>("lib_path", &lib_path).IsOK() == false) {
    ORT_THROW("AOT: lib_path is not specified");
  }

  std::vector<std::string> output_shapes;
  if (info.GetAttrs<std::string>("output_shapes", output_shapes).IsOK() == false) {
    ORT_THROW("AOT: output_shapes is not specified");
  }
  for (auto shape_str : output_shapes) {
    TensorShapeVector shape;
    const auto stop_ops = utils::SplitString(shape_str, ",");
    for (size_t i = 0; i < stop_ops.size(); i++) {
      const auto stop_op = stop_ops[i];
      if (std::isdigit(stop_op[0])) {
        int i3;
        auto result = std::from_chars(stop_op.begin(), stop_op.end(), i3);
        if (result.ec == std::errc::invalid_argument || result.ec == std::errc::result_out_of_range) {
          ORT_THROW("AOT: output_shapes is not valid");
        }
        shape.push_back(i3);
      } else {
        shape.push_back(-1);
      }
    }
    output_shapes_.push_back(shape);
  }

  // dynamic_shape
  std::vector<int64_t> dynamic_shape;
  if (info.GetAttrs<int64_t>("dynamic_shape", dynamic_shape).IsOK() == false) {
    ORT_THROW("AOT: dynamic_shape is not specified");
  }
  for (auto d_sp : dynamic_shape) {
    auto* data = reinterpret_cast<const int32_t*>(&d_sp);
    dynamic_dims_.emplace_back(data[0], data[1]);
  }

  std::vector<int64_t> match_pairs;
  if (info.GetAttrs<int64_t>("match_pairs", match_pairs).IsOK() == false) {
    ORT_THROW("AOT: match_pairs is not specified");
  }
  for (const auto& match_pair : match_pairs) {
    auto pairs = reinterpret_cast<const uint16_t*>(&match_pair);
    dynamic_dims_pair_.emplace_back(std::vector<uint16_t>{pairs[0], pairs[1], pairs[2]});
  }

  void* library_handle = nullptr;
  auto path_str = ToPathString(lib_path);
  ORT_THROW_IF_ERROR(Env::Default().LoadDynamicLibrary(path_str, false, &library_handle));
  if (!library_handle) {
    ORT_THROW(" Failed to load library");
  }

  if (info.GetAttr<int64_t>("func_type", &func_type_).IsOK() == false) {
    ORT_THROW(" func_type is not specified");
  }

  if (info.GetAttr<std::string>("func_name", &func_name_).IsOK() == false) {
    ORT_THROW(" func_name is not specified");
  }

  ORT_THROW_IF_ERROR(Env::Default().GetSymbolFromLibrary(library_handle, func_name_.c_str(),
                                                         (void**)&func_));
}

template < typename T>
Status AOTanyOp<T>::Compute(OpKernelContext* context) const {
  auto input_cnt = context->InputCount();
  auto output_cnt = context->OutputCount();
  ORT_RETURN_IF_NOT(input_cnt == func_type_, "AOTanyOp: input count is not matched");

  InlinedVector<const Tensor*> inputs(input_cnt);
  InlinedVector<Tensor*> outputs(output_cnt);
  InlinedVector<const void*> input_args(input_cnt);
  InlinedVector<void*> output_args(output_cnt);

  TensorShape broadcast_input_shape(output_shapes_[0]);

  for (int i = 0; i < input_cnt; i++) {
    inputs[i] = context->Input<Tensor>(i);
    input_args[i] = inputs[i]->DataRaw();
    for (size_t j = 0; inputs[i]->Shape().NumDimensions() == broadcast_input_shape.NumDimensions() &&
                       j < broadcast_input_shape.NumDimensions(); j++) {
      broadcast_input_shape[j] = std::max(broadcast_input_shape[j], inputs[i]->Shape()[j]);
    }
  }

  std::vector<int64_t> dynamic_dims(dynamic_dims_.size());
  for (size_t i = 0; i < dynamic_dims_.size(); i++) {
    dynamic_dims[i] = inputs[dynamic_dims_[i].first]->Shape()[dynamic_dims_[i].second];
  }

  ORT_RETURN_IF_NOT(broadcast_input_shape.NumDimensions() >= output_shapes_[0].size(), "AOTanyOp: input shape is not valid");

  // bool hit = Node().Name() == "/mobilebert/encoder/layer.9/attention/self/Div--->/mobilebert/encoder/layer.9/attention/self/Add";
  TensorShape loop_shape(broadcast_input_shape);
  for (size_t i = 0, shape_pair_offset = 0; i < size_t(output_cnt); i++) {
    TensorShape output_shape(output_shapes_[i]);
    for (; shape_pair_offset< dynamic_dims_pair_.size() && dynamic_dims_pair_[shape_pair_offset][0] == i; shape_pair_offset++) {
      auto output_inner_idx = dynamic_dims_pair_[shape_pair_offset][1],idx_in_dynamic = dynamic_dims_pair_[shape_pair_offset][2];
      ORT_ENFORCE(output_shape[output_inner_idx] == -1, "Shape match error: this should be -1, but got ", output_shape[output_inner_idx]);
      output_shape[output_inner_idx] = dynamic_dims[idx_in_dynamic];
    }

    outputs[i] = context->Output(i, output_shape);
    output_args[i] = outputs[i]->MutableDataRaw();
  }
  auto task_count = loop_shape.NumDimensions()>1? loop_shape[0] * loop_shape[1]: loop_shape[0];

  // printf("function: %s, task_count:%ld, dynamic_dims_[0]:%ld, dynamic_dims_[1]:%ld\n", func_name_.c_str(), task_count, loop_shape[dynamic_dims_[0]],
  //        loop_shape[dynamic_dims_[1]]);
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  concurrency::ThreadPool::TryParallelFor(
      tp, static_cast<int32_t>(task_count),
      3,
      [&](ptrdiff_t task_start, ptrdiff_t task_end) {
        func_((const void**)input_args.data(), task_start, task_end,
              dynamic_dims.data(), (void**)output_args.data());
      });

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    AOTanyOp,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
.TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),DataTypeImpl::GetTensorType<int64_t>(),DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<bool>()})
    .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),DataTypeImpl::GetTensorType<int64_t>(),DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<bool>()}),
    AOTanyOp<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    AOTanyOp,
    kMSDomain,
    1,
    int64_t,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
    .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<int64_t>(), DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<bool>()})
    .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),DataTypeImpl::GetTensorType<int64_t>(),DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<bool>()}),
    AOTanyOp<int64_t>);

}  // namespace contrib
}  // namespace onnxruntime
