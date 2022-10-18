// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#include "curl/curl.h"
#include "aml_op.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace cloud {

common::Status AzureMLOp::Compute(OpKernelContext* context) const {
  std::string input_string;
  for (int i = 0; i < context->InputCount(); i++) {
    const auto* input_value = context->GetInputOrtValue(i);
    ORT_ENFORCE(input_value->IsTensor());  // todo - suppport other types
    const auto& input_tensor = input_value->Get<Tensor>();
    ONNX_NAMESPACE::TensorProto input_proto = onnxruntime::utils::TensorToTensorProto(input_tensor, this->Node().InputDefs()[i]->Name());
    input_proto.AppendToString(&input_string);
  }
  try {
    Data response = invoker_->Send({const_cast<char*>(input_string.c_str()), input_string.size()});
    //todo - try avoid additional copy on response data
    std::string output_string{static_cast<const char*>(response.content), response.size_in_byte};
    std::stringstream in_stream{output_string};
    ONNX_NAMESPACE::TensorProto output_proto;
    int output_index{0};
    while (output_proto.ParsePartialFromIstream(&in_stream)) {
      TensorShape shape{output_proto.dims()};
      auto* output_tensor = context->Output(output_index++, shape);
      memcpy(output_tensor->MutableDataRaw(), output_proto.raw_data().c_str(), output_proto.ByteSizeLong());
    }
    //todo - release response data more securely
    free(response.content);
    return Status::OK();
  } catch (...) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get response from ", uri_);
  }
}

} // namespace cloud
}  // namespace onnxruntime