// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_HOSTING_EXECUTOR_H
#define ONNXRUNTIME_HOSTING_EXECUTOR_H

#include <google/protobuf/stubs/status.h>

#include "core/framework/data_types.h"

#include "environment.h"
#include "predict.pb.h"

namespace onnxruntime {
namespace hosting {

class Executor {
 public:
  explicit Executor(HostingEnvironment& hosting_env) : env_(hosting_env) {}

  // Prediction method
  google::protobuf::util::Status predict(const std::string& name, const std::string& version, const std::string& request_id,
                                         onnxruntime::hosting::PredictRequest& request,
                                         /* out */ onnxruntime::hosting::PredictResponse& response);

 private:
  onnx::TensorProto_DataType MLDataTypeToTensorProtoDataType(const onnxruntime::DataTypeImpl* cpp_type);

  // Convert MLValue to TensorProto. Some fields are ignored:
  //   * name field: could not get from MLValue
  //   * doc_string: could not get from MLValue
  //   * segment field: we do not expect very large tensors in the prediction output
  //   * external_data field: we do not expect very large tensors in the prediction output
  common::Status MLValue2TensorProto(onnxruntime::MLValue& ml_value, bool using_raw_data,
                                     /* out */ onnx::TensorProto& tensor_proto);

 private:
  HostingEnvironment& env_;
};
}  // namespace hosting
}  // namespace onnxruntime

#endif  //ONNXRUNTIME_HOSTING_EXECUTOR_H
