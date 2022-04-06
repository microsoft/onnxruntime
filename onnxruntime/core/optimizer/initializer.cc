// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)
#include "core/optimizer/initializer.h"

#include "gsl/gsl"

#include "core/common/path.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_external_data_info.h"
#include "core/platform/env.h"

namespace onnxruntime {
Status Initializer::ReadExternalRawData(
    const ONNX_NAMESPACE::TensorProto& tensor_proto, const Path& model_path, std::vector<char>& raw_data) {
  ORT_RETURN_IF_NOT(
      tensor_proto.data_type() != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED &&
          tensor_proto.data_type() != ONNX_NAMESPACE::TensorProto_DataType_STRING,
      "External data type must not be UNDEFINED or STRING.");

  ORT_RETURN_IF(
      model_path.IsEmpty(),
      "model_path must not be empty. Ensure that a path is provided when the model is created or loaded.");

  std::unique_ptr<ExternalDataInfo> external_data{};
  ORT_RETURN_IF_ERROR(ExternalDataInfo::Create(tensor_proto.external_data(), external_data));

  size_t actual_tensor_data_length;
  ORT_RETURN_IF_ERROR(utils::GetSizeInBytesFromTensorProto<0>(
      tensor_proto, &actual_tensor_data_length));
  const size_t external_data_length = external_data->GetLength();

  ORT_RETURN_IF_NOT(
      external_data_length == 0 ||
          external_data_length == actual_tensor_data_length,
      "TensorProto external data size mismatch. ",
      "Computed size: ", actual_tensor_data_length,
      ", external_data.length: ", external_data_length);

  Path external_data_relative_path{};
  ORT_RETURN_IF_ERROR(Path::Parse(
      external_data->GetRelPath(), external_data_relative_path));

  std::vector<char> buffer(actual_tensor_data_length);

  ORT_RETURN_IF_ERROR(Env::Default().ReadFileIntoBuffer(
      (model_path.ParentPath() / external_data_relative_path).ToPathString().c_str(),
      external_data->GetOffset(),
      actual_tensor_data_length,
      gsl::make_span(buffer)));

  raw_data = std::move(buffer);

  return Status::OK();
}
}  // namespace onnxruntime

#endif  // !(ORT_MINIMAL_BUILD)
