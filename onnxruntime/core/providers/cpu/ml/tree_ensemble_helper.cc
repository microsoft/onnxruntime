// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/tree_ensemble_helper.h"
#include "core/common/common.h"

using namespace ::onnxruntime::common;
using namespace std;
namespace onnxruntime {
namespace ml {

template<typename TH>
std::vector<TH> GetVectorAttrsOrDefault(const OpKernelInfo& info, const std::string& name, const std::vector<TH> default_value, 
                                        ONNX_NAMESPACE::TensorProto_DataType proto_type) {
  ONNX_NAMESPACE::TensorProto proto;
  if (!info.GetAttr(name, &proto).IsOK()) {
    return default_value;
  }
  auto n_dims = proto.dims_size();
  if (n_dims == 0) {
    return default_value;
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

  std::vector<TH> data(n_elements);
  for (int i = 0; i < static_cast<int>(data.size()); ++i) {
    if (proto_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE) {
      data[i] = static_cast<TH>(proto.double_data(i));
    } else if (proto_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
      data[i] = static_cast<TH>(proto.float_data(i));
    }
  }
  return data;
}

std::vector<double> GetVectorAttrsOrDefault(const OpKernelInfo& info, const std::string& name, const std::vector<double>& default_value) {
  return GetVectorAttrsOrDefault(info, name, default_value, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE);
}

std::vector<float> GetVectorAttrsOrDefault(const OpKernelInfo& info, const std::string& name, const std::vector<float>& default_value) {
  return GetVectorAttrsOrDefault(info, name, default_value, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT);
}

}  // namespace ml
}  // namespace onnxruntime
