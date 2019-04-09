// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <utility>
#include <vector>
#include <string>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/ml_value.h"
#include "core/framework/framework_common.h"
#include "core/framework/path_lib.h"
#include "test/training/runner/training_util.h"

namespace onnxruntime {
namespace training {

typedef std::basic_string<PATH_CHAR_TYPE> PATH_STRING_TYPE;

/** A class for loading training/test data from protobuf files.
    One sample contains multiple protobuf files (.pb), named by <filename>_input_<count>.pb.
    "count" starts from 0.
    (For test set only: the one with largest "count" is label, which is passed to error function during evaluation.)
    For example, given a folder containing below files:
     - xxx_input_0.pb
     - xxx_input_1.pb
     - xxx_input_2.pb
     - yyy_input_0.pb
     - yyy_input_1.pb
     - yyy_input_2.pb
    2 training samples (aka. xxx and yyy) are generated, xxx_input_2.pb and yyy_input_2.pb are the labels.
*/

class DataLoader {
 public:
  common::Status Load(const PATH_STRING_TYPE& dir);
  ~DataLoader();

  DataSet& MutableDataSet() {
    return *data_set_;
  }

 private:
  common::Status AddData(const std::vector<ONNX_NAMESPACE::TensorProto>& inputs);

  std::unique_ptr<DataSet> data_set_;
  std::vector<std::unique_ptr<char[]>> buffer_for_mlvalues_;
  std::vector<OrtCallback> deleter_for_mlvalues_;
};
}  // namespace training
}  // namespace onnxruntime
