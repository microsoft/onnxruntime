// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "orttraining/core/graph/graph_augmenter.h"

namespace onnxruntime {
namespace test {
using ONNX_NAMESPACE::AttributeProto;

typedef std::vector<std::vector<float>> TwoDArray;

class OpFunctionTester : public OpTester {
 public:
  OpFunctionTester(const char* op, int opset_version = 9, const char* domain = onnxruntime::kOnnxDomain)
      : OpTester(op, opset_version, domain) {}
  TwoDArray RunFunctionBodyGraphOnCPU();
  virtual ~OpFunctionTester();
};

template <class T>
std::unique_ptr<T> CreateOpTester(const onnxruntime::training::OpDef& op_def,
                                  const TwoDArray& input_data,
                                  const std::vector<std::vector<int64_t>>& input_dims,
                                  const TwoDArray& expected_output_data,
                                  const std::vector<std::vector<int64_t>>& output_dims,
                                  const std::vector<AttributeProto>& attributes,
                                  int opset_version);

void CompareResults(const onnxruntime::training::OpDef& op_def,
                    const TwoDArray& input_data,
                    const std::vector<std::vector<int64_t>>& input_dims,
                    const std::vector<std::vector<int64_t>>& output_dims,
                    const std::vector<AttributeProto>& attributes,
                    int opset_version = 7);

TwoDArray CreateEmpty2DArray(const std::vector<std::vector<int64_t>>& dims);

}  // namespace test
}  // namespace onnxruntime
