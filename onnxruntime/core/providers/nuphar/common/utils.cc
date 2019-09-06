// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/common/utils.h"

#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace nuphar {

bool NodeArgShapeUnknownOnAxis(const NodeArg* def, int64_t axis) {
  auto shape = def->Shape();
  axis = HandleNegativeAxis(axis, shape->dim_size());
  ORT_ENFORCE(axis < shape->dim_size());
  auto dim = shape->dim(axis);
  return dim.has_dim_param() || (!dim.has_dim_param() && !dim.has_dim_value());
}

bool HasUnknownShapeOnAxis(const ConstPointerContainer<std::vector<NodeArg*>>& defs, int64_t axis) {
  for (const NodeArg* def : defs) {
    if (NodeArgShapeUnknownOnAxis(def, axis)) {
      return true;
    }
  }
  return false;
}

bool HasUnknownShapeOnAxes(const NodeArg* def, std::vector<int64_t>& axes) {
  for (auto axis : axes) {
    if (NodeArgShapeUnknownOnAxis(def, axis)) {
      return true;
    }
  }
  return false;
}

Status GetSliceAxesFromTensorProto(std::vector<int64_t>& axes,
                                   const ONNX_NAMESPACE::TensorProto& axes_tp) {
  size_t tp_sz_in_bytes;
  ORT_RETURN_IF_ERROR(utils::GetSizeInBytesFromTensorProto<0>(axes_tp, &tp_sz_in_bytes));
  OrtValue ort_value;
  std::unique_ptr<char[]> data(new char[tp_sz_in_bytes]);

#define UNPACK_TENSOR(T)                                                \
   T* p = reinterpret_cast<T*>(data.get());                             \
   ORT_RETURN_IF_ERROR(utils::UnpackTensor<T>(                          \
       axes_tp,                                                         \
       axes_tp.raw_data().size() ? axes_tp.raw_data().data() : nullptr, \
       axes_tp.raw_data().size(),                                       \
       p,                                                               \
       tp_sz_in_bytes / sizeof(T)));                                    \
   std::vector<T> tmp_axes(p, p + tp_sz_in_bytes / sizeof(T));

  switch (axes_tp.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      UNPACK_TENSOR(int32_t);
      for (auto axis : tmp_axes) {
        axes.push_back(static_cast<int64_t>(axis));
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      UNPACK_TENSOR(int64_t);
      axes.insert(axes.end(), tmp_axes.begin(), tmp_axes.end());
      break;
    }
    default:
      ORT_NOT_IMPLEMENTED("Unimplemented type: ", axes_tp.data_type());
  }

  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
