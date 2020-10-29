// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/zipmap.h"
#include "core/util/math_cpuonly.h"
/**
https://github.com/onnx/onnx/blob/master/onnx/defs/traditionalml/defs.cc
ONNX_OPERATOR_SCHEMA(ZipMap)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
Makes a map from the input and the attributes.
Assumes input 0 are the values, and the keys are specified by the attributes.
Must provide keys in either classlabels_strings or classlabels_int64s (but not both).
Input 0 may have a batch size larger than 1,
but each input in the batch must be the size of the keys specified by the attributes.
The order of the input and attributes determines the key-value mapping.
)DOC")
.Input(0, "X", "The input values", "tensor(float)")
.Output(0, "Z", "The output map", "T")
.TypeConstraint(
"T",
{ "seq(map(string, float))", "seq(map(int64, float))" },
" allowed types.")
.Attr("classlabels_strings", "keys if using string keys", AttributeProto::STRINGS, OPTIONAL)
.Attr("classlabels_int64s", "keys if using int keys", AttributeProto::INTS, OPTIONAL);
*/
using namespace ::onnxruntime::common;
using namespace std;
namespace onnxruntime {
namespace ml {
ONNX_CPU_OPERATOR_ML_KERNEL(
    ZipMap,
    1,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetType<std::vector<std::map<std::string, float>>>(),
                                            DataTypeImpl::GetType<std::vector<std::map<std::int64_t, float>>>()}),
    ZipMapOp);

ZipMapOp::ZipMapOp(const OpKernelInfo& info)
    : OpKernel(info),
      classlabels_int64s_(info.GetAttrsOrDefault<int64_t>("classlabels_int64s")),
      classlabels_strings_(info.GetAttrsOrDefault<std::string>("classlabels_strings")) {
  ORT_ENFORCE(classlabels_strings_.empty() ^ classlabels_int64s_.empty(),
              "Must provide classlabels_strings or classlabels_int64s but not both.");
  using_strings_ = !classlabels_strings_.empty();
}

common::Status ZipMapOp::Compute(OpKernelContext* context) const {
  const auto* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *tensor_pointer;
  const std::vector<int64_t>& x_dims = X.Shape().GetDims();

  if (x_dims.empty()) {
    return Status(ONNXRUNTIME,
                  INVALID_ARGUMENT,
                  "Zipmap does not support empty dim count");
  }

  int64_t batch_size = x_dims.size() > 1 ? x_dims[0] : 1;
  int64_t features_per_batch = x_dims[x_dims.size() - 1];

  if (x_dims.size() > 2) {
    return Status(ONNXRUNTIME,
                  INVALID_ARGUMENT,
                  "Zipmap only supports 1D or 2D input tensors");
  }

  const auto* x_data = X.template Data<float>();

  if (using_strings_) {
    if (features_per_batch != static_cast<int64_t>(classlabels_strings_.size())) {
      return Status(ONNXRUNTIME,
                    INVALID_ARGUMENT,
                    "Input features_per_batch[" + std::to_string(features_per_batch) +
                        "] != number of classlabels[" + std::to_string(classlabels_strings_.size()) + "]");
    }
    auto* y_data = context->Output<std::vector<std::map<std::string, float>>>(0);
    if (y_data == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");

    //auto* y_data = Y->template MutableData<std::vector<std::map<std::string, float>>>();
    y_data->resize(batch_size);
    int64_t current_weight_0 = 0;
    for (int64_t n = 0; n < batch_size; n++) {
      std::map<std::string, float> map1;
      for (int64_t j = 0; j < features_per_batch; j++) {
        map1[classlabels_strings_[j]] = x_data[current_weight_0 + j];
      }
      current_weight_0 += features_per_batch;
      (*y_data)[n] = std::move(map1);
    }
  } else {
    if (features_per_batch != static_cast<int64_t>(classlabels_int64s_.size())) {
      return Status(ONNXRUNTIME,
                    INVALID_ARGUMENT,
                    "Input features_per_batch[" + std::to_string(features_per_batch) +
                        "] != number of classlabels[" + std::to_string(classlabels_int64s_.size()) + "]");
    }
    auto* y_data = context->Output<std::vector<std::map<std::int64_t, float>>>(0);
    if (y_data == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
    //auto* y_data = Y->template MutableData<std::vector<std::map<int64_t, float>>>();
    y_data->resize(batch_size);
    int64_t current_weight_0 = 0;
    for (int n = 0; n < batch_size; n++) {
      std::map<int64_t, float> map2;
      for (int j = 0; j < features_per_batch; j++) {
        map2[classlabels_int64s_[j]] = x_data[current_weight_0 + j];
      }
      current_weight_0 += features_per_batch;
      (*y_data)[n] = std::move(map2);
    }
  }
  return common::Status::OK();
}
}  // namespace ml
}  // namespace onnxruntime
