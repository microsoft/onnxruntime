// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/providers/cpu/ml/tree_ensemble_helper.h"
#include "core/common/common.h"
#include "onnx/defs/tensor_proto_util.h"

using namespace ::onnxruntime::common;
using namespace std;
namespace onnxruntime {
namespace ml {

Status GetNumberOfElementsAttrsOrDefault(const OpKernelInfo& info, const std::string& name,
                                         ONNX_NAMESPACE::TensorProto_DataType proto_type,
                                         size_t& n_elements, ONNX_NAMESPACE::TensorProto& proto) {
  auto status = info.GetAttr(name, &proto);
  if (!status.IsOK()) {
    // Attribute is missing, n_elements is set to 0.
    n_elements = 0;
    return Status::OK();
  }
  auto n_dims = proto.dims_size();
  if (n_dims == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Attribute:'", name, "' is specified but is empty.");
  }
  ORT_ENFORCE(n_dims == 1, "Attribute '", name, "' must be a vector.");
  ORT_ENFORCE(proto.data_type() == proto_type,
              "Unexpected type (", proto.data_type(), "(for attribute '", name, "'.");

  n_elements = onnxruntime::narrow<size_t>(proto.dims()[0]);
  ORT_ENFORCE(n_elements > 0, "Attribute '", name, "' has one dimension but is empty.");
  return Status::OK();
}

template <typename TH>
Status GetVectorAttrsOrDefault(const OpKernelInfo& info, const std::string& name,
                               ONNX_NAMESPACE::TensorProto_DataType proto_type, std::vector<TH>& data) {
  if (proto_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE) {
    ORT_ENFORCE((std::is_same<double, TH>::value));
  } else if (proto_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
    ORT_ENFORCE((std::is_same<float, TH>::value));
  } else {
    ORT_NOT_IMPLEMENTED("GetVectorAttrsOrDefault not implemented for type ", proto_type);
  }

  ONNX_NAMESPACE::TensorProto proto;
  size_t n_elements;
  data.clear();
  ORT_THROW_IF_ERROR(GetNumberOfElementsAttrsOrDefault(info, name, proto_type, n_elements, proto));
  if (n_elements == 0) {
    return Status::OK();
  }
  data = ONNX_NAMESPACE::ParseData<TH>(&proto);
  return Status::OK();
}

Status GetVectorAttrsOrDefault(const OpKernelInfo& info, const std::string& name, std::vector<double>& data) {
  return GetVectorAttrsOrDefault(info, name, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE, data);
}

Status GetVectorAttrsOrDefault(const OpKernelInfo& info, const std::string& name, std::vector<float>& data) {
  return GetVectorAttrsOrDefault(info, name, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT, data);
}

}  // namespace ml
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
