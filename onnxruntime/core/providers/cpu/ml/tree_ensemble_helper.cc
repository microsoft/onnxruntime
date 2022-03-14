// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/tree_ensemble_helper.h"
#include "core/common/common.h"

using namespace ::onnxruntime::common;
using namespace std;
namespace onnxruntime {
namespace ml {

template<typename TH>
Status GetVectorAttrsOrDefault(const OpKernelInfo& info, const std::string& name,
                               ONNX_NAMESPACE::TensorProto_DataType proto_type, std::vector<TH>& data) {
  ONNX_NAMESPACE::TensorProto proto;
  data.clear();
  if (!info.GetAttr(name, &proto).IsOK()) {
    return Status::OK();
  }
  auto n_dims = proto.dims_size();
  if (n_dims == 0) {
    return Status::OK();
  }
  ORT_ENFORCE(n_dims == 1, "Attribute '", name, "' must be a vector.");
  ORT_ENFORCE(proto.data_type() == proto_type,
              "Unexpected type (", proto.data_type(), "(for attribute '", name, "'.");
  auto n_elements = proto.dims()[0];
  ORT_ENFORCE(n_elements > 0, "Attribute '", name, "' has one dimension but is empty.");

  if (proto_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE) {
    ORT_ENFORCE((std::is_same<double, TH>::value));
  } else if (proto_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
    ORT_ENFORCE((std::is_same<float, TH>::value));
  } else {
    ORT_ENFORCE(false, "Not implemented for type ", proto_type);
  }

  data.reserve(n_elements);
  for (int i = 0; i < static_cast<int>(n_elements); ++i) {
    if (proto_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE) {
      data.push_back(static_cast<TH>(proto.double_data(i)));
    } else if (proto_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
      data.push_back(static_cast<TH>(proto.float_data(i)));
    }
  }
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
