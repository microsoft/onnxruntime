// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/node_arg.h"

#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"

namespace ONNX_NAMESPACE {
class TensorShapeProto;
std::ostream& operator<<(std::ostream& out, const TensorShapeProto& shape_proto);
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

using namespace ONNX_NAMESPACE;
using namespace ONNX_NAMESPACE::Utils;
using namespace ::onnxruntime::common;

// there are some known invalid usages of dim_param and dim_value. remove them from the TypeProto so that
// they don't affect shape inferencing or the allocation planner
static void RemoveInvalidValues(ONNX_NAMESPACE::TypeProto& type) {
  if (utils::HasTensorType(type) && utils::HasShape(type.tensor_type())) {
    auto* shape = type.mutable_tensor_type()->mutable_shape();
    for (int i = 0, end = shape->dim_size(); i < end; ++i) {
      auto& dim = *shape->mutable_dim(i);
      if (utils::HasDimParam(dim)) {
        if (dim.dim_param().empty()) {
          dim.clear_dim_param();
        }
      } else if (utils::HasDimValue(dim)) {
        if (dim.dim_value() < 0) {
          dim.clear_dim_value();
        }
      }
    }
  }
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
NodeArg::NodeArg(const std::string& name, const TypeProto* p_node_arg_type) {
  node_arg_info_.set_name(name);
  // If the name is empty, it means the arg does not exist.
  exists_ = !(name.empty());
  if (nullptr != p_node_arg_type) {
    (*node_arg_info_.mutable_type()) = *p_node_arg_type;
#if !defined(ORT_MINIMAL_BUILD)
    // should not be possible to have invalid values in the ORT format model, so we don't need this
    // in a minimal build
    RemoveInvalidValues(*node_arg_info_.mutable_type());
#endif
    type_ = DataTypeUtils::ToType(node_arg_info_.type());
  } else {
    type_ = nullptr;
  }
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

NodeArg::NodeArg(NodeArgInfo&& node_arg_info) {
  node_arg_info_ = std::move(node_arg_info);

  exists_ = !node_arg_info_.name().empty();
  if (node_arg_info_.has_type())
    type_ = DataTypeUtils::ToType(node_arg_info_.type());
  else {
    type_ = nullptr;
  }
}

const std::string& NodeArg::Name() const noexcept {
  return node_arg_info_.name();
}

DataType NodeArg::Type() const noexcept {
  return type_;
}

const TypeProto* NodeArg::TypeAsProto() const noexcept {
  if (utils::HasType(node_arg_info_))
    return &node_arg_info_.type();

  return nullptr;
}

const TensorShapeProto* NodeArg::Shape() const {
  const TypeProto* type = TypeAsProto();
  if (type == nullptr) return nullptr;
  const auto typeCase = type->value_case();
  switch (typeCase) {
    case TypeProto::kTensorType: {
      if (utils::HasShape(type->tensor_type())) {
        return &(type->tensor_type().shape());
      }
      return nullptr;
    }
#if !defined(DISABLE_SPARSE_TENSORS)
    case TypeProto::kSparseTensorType: {
      if (utils::HasShape(type->sparse_tensor_type())) {
        return &(type->sparse_tensor_type().shape());
      }
      return nullptr;
    }
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)
    case TypeProto::kOptionalType: {
      // Shape is applicable only for optional tensor type
      if (utils::HasOptionalTensorType(*type) &&
          utils::HasShape(utils::GetOptionalTypeProto(*type).tensor_type())) {
        return &(utils::GetOptionalTypeProto(*type).tensor_type().shape());
      }
      return nullptr;
    }
#endif

    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::kOpaqueType:
    case TypeProto::VALUE_NOT_SET:
    default:
      return nullptr;
  }
}

bool NodeArg::HasTensorOrScalarShape() const {
  const TypeProto* type = TypeAsProto();
  if (!type) return false;
  const auto type_case = type->value_case();
  switch (type_case) {
    case TypeProto::kTensorType:
#if !defined(DISABLE_SPARSE_TENSORS)
    case TypeProto::kSparseTensorType:
#endif
      // Standard tensor has a valid shape field while
      // scalar's shape is empty. Thus, we don't need to
      // check shape here.
      return true;
    case TypeProto::kSequenceType:
    case TypeProto::kOptionalType:
    case TypeProto::kMapType:
    case TypeProto::kOpaqueType:
    case TypeProto::VALUE_NOT_SET:
    default:
      return false;
  }
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
void NodeArg::SetShape(const TensorShapeProto& shape) {
  const auto type_case = node_arg_info_.type().value_case();
  switch (type_case) {
    case TypeProto::kTensorType:
      *(node_arg_info_.mutable_type()->mutable_tensor_type()->mutable_shape()) = shape;
      break;
#if !defined(DISABLE_SPARSE_TENSORS)
    case TypeProto::kSparseTensorType:
      *(node_arg_info_.mutable_type()->mutable_sparse_tensor_type()->mutable_shape()) = shape;
      break;
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)
    case TypeProto::kOptionalType:
      // Set shape only for optional tensors
      if (utils::HasOptionalTensorType(node_arg_info_.type())) {
        *(utils::GetMutableOptionalTypeProto(*(node_arg_info_.mutable_type()))
              ->mutable_tensor_type()
              ->mutable_shape()) = shape;
      }
      break;
#endif
    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::kOpaqueType:
    case TypeProto::VALUE_NOT_SET:
    default:
      return;
  }
}

void NodeArg::ClearShape() {
  const auto type_case = node_arg_info_.type().value_case();
  switch (type_case) {
    case TypeProto::kTensorType:
      node_arg_info_.mutable_type()->mutable_tensor_type()->clear_shape();
      break;
#if !defined(DISABLE_SPARSE_TENSORS)
    case TypeProto::kSparseTensorType:
      node_arg_info_.mutable_type()->mutable_sparse_tensor_type()->clear_shape();
      break;
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)
    case TypeProto::kOptionalType:
      // Clear shape only for optional tensors
      if (utils::HasOptionalTensorType(node_arg_info_.type())) {
        utils::GetMutableOptionalTypeProto(*(node_arg_info_.mutable_type()))
            ->mutable_tensor_type()
            ->clear_shape();
      }
      break;
#endif

    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::kOpaqueType:
    case TypeProto::VALUE_NOT_SET:
    default:
      return;
  }
}

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)

common::Status NodeArg::OverrideTypesHelper(const ONNX_NAMESPACE::TypeProto& input_type,
                                            int32_t input_tensor_elem_type,
                                            int32_t current_tensor_elem_type,
                                            bool override_types) {
  if (input_tensor_elem_type != current_tensor_elem_type) {
    if (override_types) {
      DataType inferred_type = DataTypeUtils::ToType(input_type);
      // The "SetType" call will override the shape information to empty.
      // If the original tensor has shape information, need to set it back.
      if (Shape()) {
        auto old_shape = *Shape();
        SetType(inferred_type);
        SetShape(old_shape);
      } else {
        SetType(inferred_type);
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Tensor element type mismatch. ",
                             static_cast<TensorProto_DataType>(input_tensor_elem_type), " != ",
                             static_cast<TensorProto_DataType>(current_tensor_elem_type));
    }
  }

  return Status::OK();
}

common::Status NodeArg::UpdateTypeAndShape(const ONNX_NAMESPACE::TypeProto& input_type, bool strict,
                                           bool override_types, const logging::Logger& logger) {
  if (!utils::HasType(node_arg_info_)) {
    SetType(input_type);
    return Status::OK();
  }

  auto& current_type = *node_arg_info_.mutable_type();
  const auto current_type_case = current_type.value_case();
  const auto input_type_case = input_type.value_case();

  if (current_type_case != input_type_case)
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type mismatch. Current=",
                           current_type_case, " Input=", input_type_case);

  switch (input_type_case) {
    case TypeProto::kTensorType: {
      const auto& input_tensor_type = input_type.tensor_type();
      const auto& input_tensor_elem_type = input_tensor_type.elem_type();
      const auto& current_tensor_elem_type = current_type.tensor_type().elem_type();

      ORT_RETURN_IF_ERROR(OverrideTypesHelper(input_type, input_tensor_elem_type, current_tensor_elem_type, override_types));

      if (utils::HasShape(input_tensor_type)) {
        if (utils::HasShape(current_type)) {
          ORT_RETURN_IF_ERROR(graph_utils::MergeShapeInfo(Name(), input_type, current_type, strict, logger));
        } else {
          *current_type.mutable_tensor_type() = input_tensor_type;
        }
      }

      break;
    }

#if !defined(DISABLE_SPARSE_TENSORS)
    case TypeProto::kSparseTensorType: {
      const auto& input_tensor_type = input_type.sparse_tensor_type();
      const auto input_tensor_elem_type = input_tensor_type.elem_type();
      const auto current_tensor_elem_type = current_type.sparse_tensor_type().elem_type();

      ORT_RETURN_IF_ERROR(OverrideTypesHelper(input_type, input_tensor_elem_type, current_tensor_elem_type, override_types));

      if (utils::HasShape(input_tensor_type)) {
        if (utils::HasShape(current_type)) {
          ORT_RETURN_IF_ERROR(graph_utils::MergeShapeInfo(Name(), input_type, current_type, strict, logger));
        } else {
          *current_type.mutable_sparse_tensor_type() = input_tensor_type;
        }
      }
      break;
    }
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)
    case TypeProto::kOptionalType: {
      bool is_input_type_optional_tensor_type = utils::HasOptionalTensorType(input_type);
      bool is_current_type_optional_tensor_type = utils::HasOptionalTensorType(current_type);

      // Check for homogeneity within optional type
      if (is_input_type_optional_tensor_type != is_current_type_optional_tensor_type) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Optional Type mismatch. Expected: ", ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(current_type),
                               " . Got: ", ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(input_type));
      }

      // Updating element type and shape is only applicable for optional tensors
      if (is_input_type_optional_tensor_type) {
        const auto& optional_input_type = utils::GetOptionalTypeProto(input_type);
        auto& optional_current_type = *utils::GetMutableOptionalTypeProto(current_type);

        const auto& input_tensor_type = optional_input_type.tensor_type();
        const auto& input_tensor_elem_type = input_tensor_type.elem_type();
        const auto& current_tensor_elem_type = optional_current_type.tensor_type().elem_type();

        ORT_RETURN_IF_ERROR(OverrideTypesHelper(input_type, input_tensor_elem_type, current_tensor_elem_type, override_types));

        if (utils::HasShape(optional_input_type.tensor_type())) {
          if (utils::HasShape(optional_current_type.tensor_type())) {
            ORT_RETURN_IF_ERROR(graph_utils::MergeShapeInfo(Name(), optional_input_type, optional_current_type, strict, logger));
          } else {
            *optional_current_type.mutable_tensor_type() = input_tensor_type;
          }
        }
      } else {
        // TODO: What is the plan for optional sequence tensors ?
        // Currently, we don't do anything for the generic sequence type
        // as seen below. This section needs an update if we choose to
        // change that in the future.
      }

      break;
    }
#endif

    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::kOpaqueType:
    case TypeProto::VALUE_NOT_SET:
    default:
      break;
  }

  return Status::OK();
}

common::Status NodeArg::UpdateTypeAndShape(const NodeArg& node_arg, bool strict, bool override_types,
                                           const logging::Logger& logger) {
  auto status = Status::OK();

  if (utils::HasType(node_arg.node_arg_info_))
    status = UpdateTypeAndShape(node_arg.node_arg_info_.type(), strict, override_types, logger);

  return status;
}

void NodeArg::SetType(DataType p_type) {
  if (nullptr == p_type) {
    return;
  }

  type_ = p_type;
  *(node_arg_info_.mutable_type()) = DataTypeUtils::ToTypeProto(p_type);
}

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

void NodeArg::SetType(const TypeProto& type_proto) {
  type_ = DataTypeUtils::ToType(type_proto);
  *(node_arg_info_.mutable_type()) = type_proto;
}

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

bool NodeArg::Exists() const noexcept {
  return exists_;
}
}  // namespace onnxruntime
