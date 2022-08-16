// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/common/safeint.h"

#include "shaper.h"
#include "helper.h"

namespace onnxruntime {
namespace nnapi {

#define SHAPER_FUNC(FUNC, ...)                  \
  ORT_RETURN_IF_ERROR(FUNC##Impl(__VA_ARGS__)); \
  shape_ops_.push_back(                         \
      [__VA_ARGS__](Shaper& shaper) {           \
        return shaper.FUNC##Impl(__VA_ARGS__);  \
      });                                       \
  return Status::OK();

Status Shaper::Eltwise(const std::string& input1_name,
                       const std::string& input2_name,
                       const std::string& output_name) {
  SHAPER_FUNC(Eltwise, input1_name, input2_name, output_name);
}

#undef SHAPER_FUNC

Status Shaper::EltwiseImpl(const std::string& input1_name,
                           const std::string& input2_name,
                           const std::string& output_name) {
  const Shape& shape1 = shape_map_.at(input1_name);
  const Shape& shape2 = shape_map_.at(input2_name);

  // broadcasting support
  bool shape1IsBigger = shape1.size() >= shape2.size();
  auto max_shape = shape1IsBigger ? shape1 : shape2;
  auto min_shape = shape1IsBigger ? shape2 : shape1;
  for (int i = (int)max_shape.size() - 1,
           j = (int)min_shape.size() - 1;
       i >= 0 && j >= 0;
       i--, j--) {
    int dim_max_shape = max_shape[i];
    int dim_min_shape = min_shape[j];
    if (dim_max_shape != dim_min_shape) {
      ORT_RETURN_IF_NOT(dim_max_shape == 1 || dim_min_shape == 1,
                        "Dimensions are not compatible, dim1: ", std::to_string(dim_max_shape),
                        "dim2: ", std::to_string(dim_min_shape));
    }

    if (dim_max_shape == 0 || dim_min_shape == 0) {
      max_shape[i] = 0;
    } else if (dim_max_shape < dim_min_shape) {
      max_shape[i] = dim_min_shape;
    }
  }

  shape_map_[output_name] = max_shape;
  return Status::OK();
}

void Shaper::AddShape(const std::string& name, const Shape& shape) {
  shape_map_[name] = shape;
}

Status Shaper::UpdateShape(const std::string& name, const Shape& new_shape) {
  const Shape& old_shape = shape_map_.at(name);
  if (old_shape != new_shape) {
    ORT_RETURN_IF_NOT(Product(old_shape) == 0 || !old_shape.empty(),
                      "The shape should be same size or old shape has size 0 (dynamic shape)");

    shape_map_[name] = new_shape;
  }

  return Status::OK();
}

Status Shaper::UpdateDynamicDimensions() {
  for (auto& shape_op : shape_ops_)
    ORT_RETURN_IF_ERROR(shape_op(*this));

  return Status::OK();
}

void Shaper::Clear() {
  shape_map_.clear();
  shape_ops_.clear();
}

}  // namespace nnapi
}  // namespace onnxruntime
