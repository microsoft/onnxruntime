// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"
#include "test/util/include/test_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace {

using onnxruntime::Node;
using onnxruntime::NodeArg;
using onnxruntime::ProviderOptions;
using onnxruntime::test::AddQDQNodePair;
using onnxruntime::test::AddQDQNodePairWithOutputAsGraphOutput;
using onnxruntime::test::BuildOpTestCase;
using onnxruntime::test::ExpectedEPNodeAssignment;
using onnxruntime::test::GetTestInputQuantParams;
using onnxruntime::test::GetTestQDQModelFn;
using onnxruntime::test::MakeTestInput;
using onnxruntime::test::ModelTestBuilder;
using onnxruntime::test::QDQTolerance;
using onnxruntime::test::QuantParams;
using onnxruntime::test::RunQnnModelTest;
using onnxruntime::test::TestInputDef;
using onnxruntime::test::TestQDQModelAccuracy;
using onnxruntime::utils::MakeAttribute;

constexpr char kEinsumOp[] = "Einsum";
constexpr char kEinsumEquation[] = "equation";
constexpr char kQnnBackendType[] = "backend_type";
constexpr char kQnnBackendTypeCpu[] = "cpu";
constexpr char kQnnBackendTypeHtp[] = "htp";
constexpr char kOffloadGraphIoQuantization[] = "offload_graph_io_quantization";
constexpr char kOffloadGraphIoQuantizationDisable[] = "0";

template <typename DataType>
static void RunQnnEinsum(
    const std::string& backend,
    const TestInputDef<DataType>& in0,
    const TestInputDef<DataType>& in1,
    const std::string& equation,
    const float tolerance) {
  ProviderOptions provider_options;
  provider_options[kQnnBackendType] = backend;
  provider_options[kOffloadGraphIoQuantization] = kOffloadGraphIoQuantizationDisable;
  RunQnnModelTest(
      /*build_test_case=*/BuildOpTestCase<DataType, DataType>(
          /*op_type=*/kEinsumOp,
          /*input_defs_1=*/{in0, in1},
          /*input_defs_2=*/{},
          /*attrs=*/{MakeAttribute(kEinsumEquation, equation)}),
      /*provider_options=*/provider_options,
      /*opset_version=*/12,
      /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
      /*tolerance=*/tolerance);
}

template <typename InputAQType, typename InputBQType>
GetTestQDQModelFn<InputAQType> BuildTestCaseQdq(const std::vector<TestInputDef<float>>& input_defs,
                                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                bool use_contrib_qdq = false) {
  return [input_defs, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                              std::vector<QuantParams<InputAQType>>& output_qparams) {
    const size_t num_inputs = input_defs.size();

    std::vector<NodeArg*> op_inputs;
    op_inputs.reserve(num_inputs);

    // Process input 0
    NodeArg* input0 = MakeTestInput<float>(builder, input_defs[0]);
    QuantParams<InputAQType> input0_qparams = GetTestInputQuantParams<InputAQType>(input_defs[0]);
    NodeArg* input0_after_qdq = AddQDQNodePair<InputAQType>(builder, input0, input0_qparams.scale,
                                                            input0_qparams.zero_point, use_contrib_qdq);
    op_inputs.push_back(input0_after_qdq);

    // Process input 1
    NodeArg* input1 = MakeTestInput<float>(builder, input_defs[1]);
    QuantParams<InputBQType> input1_qparams = GetTestInputQuantParams<InputBQType>(input_defs[1]);
    NodeArg* input1_after_qdq = AddQDQNodePair<InputBQType>(builder, input1, input1_qparams.scale,
                                                            input1_qparams.zero_point, use_contrib_qdq);
    op_inputs.push_back(input1_after_qdq);

    // Op -> op_output
    auto* output = builder.MakeIntermediate();
    Node& node = builder.AddNode(kEinsumOp, op_inputs, {output});
    for (const auto& attr : attrs) {
      node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<InputAQType>(builder, output, output_qparams[0].scale,
                                                       output_qparams[0].zero_point, use_contrib_qdq);
  };
}

template <typename InputAQType, typename InputBQType>
static void RunQnnHtpQdqEinsum(const TestInputDef<float>& in0,
                               const TestInputDef<float>& in1,
                               const std::string& equation,
                               QDQTolerance tolerance) {
  ProviderOptions provider_options;
  provider_options[kQnnBackendType] = kQnnBackendTypeHtp;
  provider_options[kOffloadGraphIoQuantization] = kOffloadGraphIoQuantizationDisable;
  std::vector<ONNX_NAMESPACE::AttributeProto> attrs{MakeAttribute(kEinsumEquation, equation)};
  auto f32_model_builder = BuildOpTestCase<float, float>(
      /*op_type=*/kEinsumOp,
      /*input_defs_1=*/{in0, in1},
      /*input_defs_2=*/{},
      /*attrs=*/attrs);
  auto qdq_model_builder = BuildTestCaseQdq<InputAQType, InputBQType>(
      /*input_defs=*/{in0, in1}, /*attrs=*/attrs, /*use_contrib_qdq=*/false);
  TestQDQModelAccuracy<InputAQType>(/*f32_model_fn=*/f32_model_builder,
                                    /*qdq_model_fn=*/qdq_model_builder,
                                    /*qnn_options=*/provider_options,
                                    /*opset_version=*/12,
                                    /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                                    /*tolerance=*/tolerance);
}

}  // namespace

namespace onnxruntime {
namespace test {

//
// QNN CPU
//

TEST_F(QnnCPUBackendTests, EinsumRank2) {
  const std::vector<int64_t> shape0{2, 3};
  const std::vector<int64_t> shape1{3, 4};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/kQnnBackendTypeCpu,
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"ab,bc->ac",
      /*tolerance=*/1e-4f);
}

TEST_F(QnnCPUBackendTests, EinsumRank4MatMul) {
  const std::vector<int64_t> shape0{3, 4, 5, 6};
  const std::vector<int64_t> shape1{3, 4, 6, 5};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/kQnnBackendTypeCpu,
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bhij,bhjd->bhid",
      /*tolerance=*/1e-4f);
}

TEST_F(QnnCPUBackendTests, EinsumRank4MatMulTransposeY) {
  const std::vector<int64_t> shape0{2, 3, 4, 6};
  const std::vector<int64_t> shape1{2, 3, 5, 6};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/kQnnBackendTypeCpu,
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bhid,bhjd->bhij",
      /*tolerance=*/1e-4f);
}

TEST_F(QnnCPUBackendTests, EinsumRank4MatMulTransposeAll1) {
  const std::vector<int64_t> shape0{1, 9, 1, 7};
  const std::vector<int64_t> shape1{1, 7, 1, 9};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/kQnnBackendTypeCpu,
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bchq,bkhc->bkhq",
      /*tolerance=*/1e-4f);
}

TEST_F(QnnCPUBackendTests, EinsumRank4MatMulTransposeAll2) {
  const std::vector<int64_t> shape0{1, 7, 1, 7};
  const std::vector<int64_t> shape1{1, 9, 1, 7};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/kQnnBackendTypeCpu,
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bkhq,bchk->bchq",
      /*tolerance=*/1e-4f);
}

//
// QNN HTP F16
//

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, EinsumF16Rank2MatMul) {
  const std::vector<int64_t> shape0{2, 3};
  const std::vector<int64_t> shape1{3, 4};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/kQnnBackendTypeHtp,
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"ij,jk->ik",
      /*tolerance=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, EinsumF16Rank4MatMul) {
  const std::vector<int64_t> shape0{3, 1, 5, 2};
  const std::vector<int64_t> shape1{3, 1, 2, 5};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/kQnnBackendTypeHtp,
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bhij,bhjd->bhid",
      /*tolerance=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, EinsumF16Rank4MatMulTransposeY) {
  const std::vector<int64_t> shape0{2, 3, 4, 2};
  const std::vector<int64_t> shape1{2, 3, 5, 2};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/kQnnBackendTypeHtp,
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bhid,bhjd->bhij",
      /*tolerance=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, EinsumF16Rank4MatMulTransposeAll1) {
  const std::vector<int64_t> shape0{1, 3, 1, 7};
  const std::vector<int64_t> shape1{1, 7, 1, 3};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/kQnnBackendTypeHtp,
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bchq,bkhc->bkhq",
      /*tolerance=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, EinsumF16Rank4MatMulTransposeAll2) {
  const std::vector<int64_t> shape0{1, 4, 1, 4};
  const std::vector<int64_t> shape1{1, 9, 1, 4};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/kQnnBackendTypeHtp,
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bkhq,bchk->bchq",
      /*tolerance=*/1e-2f);
}

//
// QNN HTP QDQ
//

TEST_F(QnnHTPBackendTests, EinsumQdqRank2MatMul) {
  const std::vector<int64_t> shape0{2, 3};
  const std::vector<int64_t> shape1{3, 4};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnHtpQdqEinsum<uint8_t, uint8_t>(
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"ij,jk->ik",
      /*tolerance=*/QDQTolerance());
}

TEST_F(QnnHTPBackendTests, EinsumQdqRank4MatMul) {
  const std::vector<int64_t> shape0{3, 1, 5, 2};
  const std::vector<int64_t> shape1{3, 1, 2, 5};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnHtpQdqEinsum<uint8_t, uint8_t>(
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bhij,bhjd->bhid",
      /*tolerance=*/QDQTolerance());
}

TEST_F(QnnHTPBackendTests, EinsumQdqRank4MatMulTransposeY) {
  const std::vector<int64_t> shape0{2, 3, 4, 2};
  const std::vector<int64_t> shape1{2, 3, 5, 2};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnHtpQdqEinsum<uint8_t, uint8_t>(
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bhid,bhjd->bhij",
      /*tolerance=*/QDQTolerance());
}

TEST_F(QnnHTPBackendTests, EinsumQdqRank4MatMulTransposeAll1) {
  const std::vector<int64_t> shape0{1, 3, 1, 7};
  const std::vector<int64_t> shape1{1, 7, 1, 3};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnHtpQdqEinsum<uint8_t, uint8_t>(
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bchq,bkhc->bkhq",
      /*tolerance=*/QDQTolerance());
}

TEST_F(QnnHTPBackendTests, EinsumQdqRank4MatMulTransposeAll2) {
  const std::vector<int64_t> shape0{1, 4, 1, 4};
  const std::vector<int64_t> shape1{1, 9, 1, 4};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnHtpQdqEinsum<uint8_t, uint8_t>(
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bkhq,bchk->bchq",
      /*tolerance=*/QDQTolerance());
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
