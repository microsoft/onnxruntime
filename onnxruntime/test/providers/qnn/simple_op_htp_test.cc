// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <filesystem>
#include <variant>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a non-QDQ model on the QNN CPU backend and compares output to CPU EP.
template <typename InputType = float>
static void RunOpTestOnCPU(const std::string& op_type,
                           const std::vector<TestInputDef<InputType>>& input_defs,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment,
                           const std::string& op_domain = kOnnxDomain) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildOpTestCase<InputType>(op_type, input_defs, {}, attrs, op_domain),
                  provider_options,
                  opset_version,
                  expected_ep_assignment);
}

// Test float DepthToSpace on the QNN CPU backend.
// TODO: Flaky test tails often.
// Value of: expected_tensor.DataAsSpan<float>()
// Expected: contains 16 values, where each value and its corresponding value in 16-byte object
// <10-00 00-00 00-00 00-00 40-00 23-D1 82-02 00-00> are an almost-equal pair
// Actual: 16-byte object <10-00 00-00 00-00 00-00 40-00 12-D1 82-02 00-00>, where the value pair (2, 0.1) at
// index #2 don't match, which is -1.9 from 2
//
// If/when fixed, enable QNN EP in cpu test TensorOpTest.SpaceToDepthTest_1
TEST_F(QnnCPUBackendTests, DISABLED_SpaceToDepth_Flaky) {
  std::vector<float> X =
      {0.0f, 0.1f, 0.2f, 0.3f,
       1.0f, 1.1f, 1.2f, 1.3f,

       2.0f, 2.1f, 2.2f, 2.3f,
       3.0f, 3.1f, 3.2f, 3.3f};

  for (size_t i = 0; i < 4; i++) {
    RunOpTestOnCPU("SpaceToDepth",
                   {TestInputDef<float>({1, 2, 2, 4}, false, X)},
                   {utils::MakeAttribute("blocksize", static_cast<int64_t>(2))},
                   7,
                   ExpectedEPNodeAssignment::All);
  }
}

// Value of: expected_tensor.DataAsSpan<float>()
// Expected: contains 108 values, where each value and its corresponding value in 16-byte object
// <6C-00 00-00 00-00 00-00 40-00 23-BB 0E-02 00-00> are an almost-equal pair
// Actual: 16-byte object <6C-00 00-00 00-00 00-00 40-00 12-BB 0E-02 00-00>, where the value pair (18, 1)
// at index #2 don't match, which is -17 from 18
//
// If/when fixed, enable QNN EP in cpu test TensorOpTest.SpaceToDepthTest_2
TEST_F(QnnCPUBackendTests, DISABLED_SpaceToDepth_Flaky2) {
  const std::vector<float> X = {
      0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
      11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.,
      22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
      33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43.,
      44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54.,
      55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65.,
      66., 67., 68., 69., 70., 71., 72., 73., 74., 75., 76.,
      77., 78., 79., 80., 81., 82., 83., 84., 85., 86., 87.,
      88., 89., 90., 91., 92., 93., 94., 95., 96., 97., 98.,
      99., 100., 101., 102., 103., 104., 105., 106., 107.};

  for (size_t i = 0; i < 4; i++) {
    RunOpTestOnCPU("SpaceToDepth",
                   {TestInputDef<float>({2, 3, 3, 6}, false, X)},
                   {utils::MakeAttribute("blocksize", static_cast<int64_t>(3))},
                   7,
                   ExpectedEPNodeAssignment::All);
  }
}

// Test f32 Relu on the CPU backend.
// TODO: When this is fixed, enable ActivationOpTest.Relu test in cpu/activation/activation_op_test tests.
// Disabled because QNN SDK 2.17 Relu treats inf as FLT_MAX.
// Log: the value pair (inf, 3.40282347e+38) at index #12 don't match
TEST_F(QnnCPUBackendTests, DISABLED_UnaryOp_Relu) {
  std::vector<float> input_data{-1.0f, 0, 1.0f,
                                100.0f, -100.0f, 1000.0f, -1000.0f,
                                FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10,
                                FLT_MAX, -FLT_MAX, std::numeric_limits<float>::infinity()};
  RunOpTestOnCPU("Relu",
                 {TestInputDef<float>({13}, false, input_data)},
                 {},
                 14,
                 ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Tests the accuracy of a QDQ model on QNN EP by comparing to CPU EP, which runs both the fp32 model
// and the QDQ model.
template <typename InputQType = uint8_t>
static void RunQDQOpTest(const std::string& op_type,
                         const std::vector<TestInputDef<float>>& input_defs,
                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                         int opset_version,
                         ExpectedEPNodeAssignment expected_ep_assignment,
                         const std::string& op_domain = kOnnxDomain,
                         bool use_contrib_qdq = false,
                         QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type, input_defs, {}, attrs, op_domain),
                       BuildQDQOpTestCase<InputQType>(op_type, input_defs, {}, attrs, op_domain, use_contrib_qdq),
                       provider_options,
                       opset_version,
                       expected_ep_assignment,
                       tolerance);
}

// Runs a non-QDQ model on HTP and compares output to CPU EP.
template <typename InputType = float>
static void RunOpTest(const std::string& op_type,
                      const std::vector<TestInputDef<InputType>>& input_defs,
                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                      int opset_version,
                      ExpectedEPNodeAssignment expected_ep_assignment,
                      const std::string& op_domain = kOnnxDomain,
                      float fp32_abs_err = 1e-5f,
                      bool enable_htp_fp16_precision = false) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  if (enable_htp_fp16_precision) {
    provider_options["enable_htp_fp16_precision"] = "1";
  }

  // Runs model with a Q/DQ binary op and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildOpTestCase<InputType>(op_type, input_defs, {}, attrs, op_domain),
                  provider_options,
                  opset_version,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Runs an FP16 model on the QNN HTP backend and compares QNN EP's accuracy to CPU EP.
static void RunFP16OpTest(const std::string& op_type,
                          const std::vector<TestInputDef<float>>& input_defs,
                          const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                          int opset_version,
                          ExpectedEPNodeAssignment expected_ep_assignment,
                          const std::string& op_domain = kOnnxDomain,
                          float tolerance = 0.004f) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::vector<TestInputDef<MLFloat16>> input_fp16_defs;
  input_fp16_defs.reserve(input_defs.size());

  for (size_t i = 0; i < input_defs.size(); i++) {
    input_fp16_defs.push_back(ConvertToFP16InputDef(input_defs[i]));
  }

  auto model_fp32_fn = BuildOpTestCase<float>(op_type, input_defs, {}, attrs, op_domain);
  auto model_fp16_fn = BuildOpTestCase<MLFloat16>(op_type, input_fp16_defs, {}, attrs, op_domain);

  TestFp16ModelAccuracy(model_fp32_fn,
                        model_fp16_fn,
                        provider_options,
                        opset_version,
                        expected_ep_assignment,
                        tolerance);
}

// Test the accuracy of QDQ Sigmoid.
TEST_F(QnnHTPBackendTests, UnaryOp_Sigmoid) {
  RunQDQOpTest<uint8_t>("Sigmoid",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Tests accuracy of 16-bit QDQ Sigmoid.
TEST_F(QnnHTPBackendTests, UnaryOp_Sigmoid_U16) {
  RunQDQOpTest<uint16_t>("Sigmoid",
                         {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                         {},
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);  // Use MS domain Q/DQ ops
}

// Test the accuracy of QDQ Tanh.
TEST_F(QnnHTPBackendTests, UnaryOp_Tanh) {
  RunQDQOpTest<uint8_t>("Tanh",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// disabled for QNN 2.28.0.241029 backendValidateOpConfig failed
// still fails on QNN 2.28.2.
// QnnDsp <E> [4294967295] has incorrect Value -32768, expected equal to 0.
// QnnDsp <V> validateNativeOps node_token_6:qti.aisw:Tanh htp op validator failed 3110
// QnnDsp <V> registered validator failed => 3110
// QnnDsp <E> QnnBackend_validateOpConfig failed 3110
// QnnDsp <V> Wake up free backend (id: 1)'s thread(s)
// QnnDsp <E> Failed to validate op node_token_6 with error 0xc26
// Tests accuracy of 16-bit QDQ Tanh.
TEST_F(QnnHTPBackendTests, DISABLED_UnaryOp_Tanh_U16) {
  RunQDQOpTest<uint16_t>("Tanh",
                         {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                         {},
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);  // Use MS domain Q/DQ ops
}

// Check that QNN compiles DQ -> Gelu -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Gelu) {
  RunQDQOpTest<uint8_t>("Gelu",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        11,
                        ExpectedEPNodeAssignment::All,
                        kMSDomain);  // GeLu is a contrib op.
}

// Tests accuracy of 16-bit QDQ GeLu.
// TODO(adrianlizarraga): Inaccuracy detected for output 'output', element 5.
// Output quant params: scale=0.00015259021893143654, zero_point=0.
// Expected val: 10
// QNN QDQ val: 9.997406005859375 (err 0.002593994140625)
// CPU QDQ val: 9.999847412109375 (err 0.000152587890625)
TEST_F(QnnHTPBackendTests, UnaryOp_Gelu_U16) {
  const std::vector<float> input_data = {-10.0f, -8.4f, 0.0f, 4.3f, 7.1f, 10.0f};
  RunQDQOpTest<uint16_t>("Gelu",
                         {TestInputDef<float>({1, 2, 3}, false, input_data)},
                         {},
                         11,
                         ExpectedEPNodeAssignment::All,
                         kMSDomain,  // GeLu is a contrib op.
                         true);      // Use MS domain Q/DQ ops.
}

// Check that QNN compiles DQ -> Elu -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Elu) {
  RunQDQOpTest<uint8_t>("Elu",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        11,
                        ExpectedEPNodeAssignment::All);
}

// Tests accuracy of 16-bit QDQ Elu.
// TODO(adrianlizarraga): Re-enable. This works on QNN SDK 2.14.1!
// Inaccuracy detected for output 'output', element 1.
// Output quant params: scale=0.00011093531065853313, zero_point=8992.
// Expected val: -0.99751651287078857
// QNN QDQ val: 6.2726154327392578 (err 7.2701320648193359)
// CPU QDQ val: -0.99753034114837646 (err 1.3828277587890625e-05)
TEST_F(QnnHTPBackendTests, DISABLE_UnaryOp_Elu_U16) {
  RunQDQOpTest<uint16_t>("Elu",
                         {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                         {},
                         11,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);
}

// Tests accuracy of QDQ Relu
// TODO: Relu does not set negative values to zero!
// Could be due to ORT's ReluQuantFusion!
//
// Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.039215687662363052, zero_point=0.
// Expected val: 0
// QNN QDQ val: -10 (err 10)
// CPU QDQ val: 0 (err 0)
TEST_F(QnnHTPBackendTests, UnaryOp_Relu) {
  RunQDQOpTest<uint8_t>("Relu",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        14,
                        ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> HardSwish -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_HardSwish) {
  RunQDQOpTest<uint8_t>("HardSwish",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        14,
                        ExpectedEPNodeAssignment::All);
}

// Tests accuracy of 16-bit QDQ HardSwish
TEST_F(QnnHTPBackendTests, UnaryOp_HardSwish_U16) {
  const std::vector<float> input_data = {-10.0f, -8.4f, 0.0f, 4.3f, 7.1f, 10.0f};
  RunQDQOpTest<uint16_t>("HardSwish",
                         {TestInputDef<float>({1, 2, 3}, false, input_data)},
                         {},
                         14,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);
}

// Check that QNN compiles DQ -> Atan -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Atan) {
  RunQDQOpTest<uint8_t>("Atan",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        14,
                        ExpectedEPNodeAssignment::All);
}

// Tests accuracy of 16-bit QDQ Atan
// TODO(adrianlizarraga): Inaccuracy detected for output 'output', element 1.
// Output quant params: scale=4.4895936298416927e-05, zero_point=32768.
// Expected val: -1.4219063520431519
// QNN QDQ val: -1.4220787286758423 (err 0.00017237663269042969)
// CPU QDQ val: -1.4218991994857788 (err 7.152557373046875e-06)
TEST_F(QnnHTPBackendTests, UnaryOp_Atan_U16) {
  const std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunQDQOpTest<uint16_t>("Atan",
                         {TestInputDef<float>({1, 2, 3}, false, input_data)},
                         {},
                         14,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,  // Atan domain
                         true);        // Q/DQ op domain is com.microsoft
}

// Check that QNN compiles DQ -> Asin -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Asin) {
  RunQDQOpTest<uint8_t>("Asin",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-0.5, 0.5, 6))},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Sign -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Sign) {
  RunQDQOpTest<uint8_t>("Sign",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Tests accuracy of 16-bit QDQ Sign
TEST_F(QnnHTPBackendTests, UnaryOp_Sign_U16) {
  const std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunQDQOpTest<uint16_t>("Sign",
                         {TestInputDef<float>({1, 2, 3}, false, input_data)},
                         {},
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,  // Sign op domain
                         true);        // Use com.microsoft Q/DQ op domains
}

// Check that QNN compiles DQ -> Sin -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Sin) {
  RunQDQOpTest<uint8_t>("Sin",
                        {TestInputDef<float>({1, 2, 3}, false, -3.14159f, 3.14159f)},
                        {},
                        11,
                        ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Cos -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Cos) {
  RunQDQOpTest<uint8_t>("Cos",
                        {TestInputDef<float>({1, 2, 3}, false, {-3.14159f, -1.5f, -0.5f, 0.0f, 1.5, 3.14159f})},
                        {},
                        11,
                        ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Cos -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Cos_InaccurateFixed) {
  RunQDQOpTest<uint8_t>("Cos",
                        {TestInputDef<float>({1, 2, 3}, false, {-3.14159f, -1.88436f, -0.542863f, 0.0f, 1.05622f, 3.14159f})},
                        {},
                        11,
                        ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Log -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, UnaryOp_Log) {
  RunQDQOpTest<uint8_t>("Log",
                        {TestInputDef<float>({1, 2, 3}, false, {3.14159f, 100.88436f, 10.542863f, 9.1f, 1.05622f, 3.14159f})},
                        {},
                        11, ExpectedEPNodeAssignment::All);
}

// Test accuracy of 8-bit QDQ Exp
TEST_F(QnnHTPBackendTests, UnaryOp_Exp) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunQDQOpTest<uint8_t>("Exp",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Test accuracy of 8-bit QDQ Sqrt
TEST_F(QnnHTPBackendTests, UnaryOp_Sqrt) {
  std::vector<float> input_data = GetFloatDataInRange(0.0f, 20.0f, 9);
  RunQDQOpTest<uint8_t>("Sqrt",
                        {TestInputDef<float>({1, 3, 3}, false, input_data)},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Test accuracy of 8-bit QDQ Neg
TEST_F(QnnHTPBackendTests, UnaryOp_Neg) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunQDQOpTest<uint8_t>("Neg",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Test Not operator on HTP backend.
TEST_F(QnnHTPBackendTests, UnaryOp_Not) {
  RunOpTest<bool>("Not",
                  {TestInputDef<bool>({1, 4}, false, {false, false, true, true})},
                  {},
                  17,
                  ExpectedEPNodeAssignment::All);
}

// Test accuracy of 8-bit QDQ Round
TEST_F(QnnHTPBackendTests, UnaryOp_Round) {
  std::vector<float> input_data = GetFloatDataInRange(-9.0f, 9.0f, 6);
  RunQDQOpTest<uint8_t>("Round",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},
                        11,
                        ExpectedEPNodeAssignment::All);
}

// Tests accuracy of 16-bit QDQ Log
TEST_F(QnnHTPBackendTests, UnaryOp_Log_U16) {
  const std::vector<float> input_data = GetFloatDataInRange(1.0f, 128.0f, 6);
  RunQDQOpTest<uint16_t>("Log",
                         {TestInputDef<float>({1, 2, 3}, false, input_data)},
                         {},
                         11,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,  // Log op domain
                         true);        // Use com.microsoft domain for Q/DQ ops
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that the default axis (-1) for SoftMax opset 13 works.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_DefaultAxis) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},  // Uses default axis of -1 for opset 13
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Tests accuracy of 16-bit QDQ Softmax (opset 13) with default axis
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_U16_DefaultAxis) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint16_t>("Softmax",
                         {TestInputDef<float>({1, 2, 3}, false, input_data)},
                         {},  // Uses default axis of -1 for opset 13
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,  // Sofmax's domain
                         true);        // Use com.microsoft domain for Q/DQ ops
}

// Test that 8-bit QDQ Softmax (opset 13) with axis != -1 is supported by QNN EP.
// QNN EP will wrap the operator with transposes.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_NonLastAxis) {
  const std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f, 100.0f, 110.0f, 120.0f,
                                         1.0856307f, 0.99734545f, 0.2829785f, 1.5062947f, 0.5786002f, 1.6514366f,
                                         2.4266791f, 0.42891264f, 1.2659363f};
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 2, 3, 3}, false, input_data)},
                        {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Test that 8-bit QDQ Softmax (opset 13) with axis != -1 is supported by QNN EP.
// QNN EP will wrap the operator with transposes.
// This is a configuration used in one of our partner's models.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_NonLastAxis_LargeInput) {
  const std::vector<float> input_data = GetFloatDataInRange(-50.0f, 50.0f, 124);
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 124, 1}, false, input_data)},
                        {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Test that 16-bit QDQ Softmax (opset 13) with axis != -1 is supported by QNN EP.
// QNN EP will wrap the operator with transposes.
// This is a configuration used in one of our partner's models.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax13_U16_NonLastAxis_LargeInput) {
  const std::vector<float> input_data = GetFloatDataInRange(-50.0f, 50.0f, 124);
  RunQDQOpTest<uint16_t>("Softmax",
                         {TestInputDef<float>({1, 124, 1}, false, input_data)},
                         {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that the default axis (1) for SoftMax opset < 13 does not work.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax11_DefaultAxisFails) {
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f)},
                        {},  // Uses default axis of 1 for opset < 13.
                        11,
                        ExpectedEPNodeAssignment::None);
}

// Check that QNN compiles DQ -> Softmax -> Q as a single unit.
// Test that setting an axis value of -1 works for Softmax opset < 13.
TEST_F(QnnHTPBackendTests, UnaryOp_Softmax11_SetValidAxis) {
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f)},
                        {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                        11,
                        ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> LogSoftmax -> Q as a single unit.
// Test that the default axis (-1) for LogSoftmax opset 13 works.
TEST_F(QnnHTPBackendTests, UnaryOp_LogSoftmax13_DefaultAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("LogSoftmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},  // Uses default axis of -1 for opset 13
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Test that 8-bit QDQ LogSoftmax (opset 13) with axis != -1 is supported by QNN EP.
// QNN EP will wrap the operator with transposes.
TEST_F(QnnHTPBackendTests, UnaryOp_LogSoftmax13_NonLastAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("LogSoftmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> LogSoftmax -> Q as a single unit.
// Test that the default axis (1) for LogSoftmax opset < 13 does not work.
TEST_F(QnnHTPBackendTests, UnaryOp_LogSoftmax11_DefaultAxisFails) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("LogSoftmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},  // Uses default axis of 1 for opset < 13.
                        11,
                        ExpectedEPNodeAssignment::None);
}

// Check that QNN compiles DQ -> LogSoftmax -> Q as a single unit.
// Test that setting an axis value of -1 works for LogSoftmax opset < 13.
TEST_F(QnnHTPBackendTests, UnaryOp_LogSoftmax11_SetValidAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("LogSoftmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                        11,
                        ExpectedEPNodeAssignment::All);
}

// Test accuracy of QDQ Abs op.
TEST_F(QnnHTPBackendTests, UnaryOp_Abs) {
  RunQDQOpTest<uint8_t>("Abs",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Test accuracy of 16-bit QDQ Abs op.
TEST_F(QnnHTPBackendTests, UnaryOp_Abs_U16) {
  const std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  RunQDQOpTest<uint16_t>("Abs",
                         {TestInputDef<float>({1, 2, 3}, false, input_data)},
                         {},
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,  // Abs op's domain
                         true);        // Use com.microsoft domain for Q/DQ ops
}

// Test accuracy of QDQ Ceil op.
TEST_F(QnnHTPBackendTests, UnaryOp_Ceil) {
  const std::vector<float> input_data = GetFloatDataInRange(-12.0f, 12.0f, 6);
  RunQDQOpTest<uint8_t>("Ceil",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Test accuracy of 16-bit QDQ Ceil op.
// TODO(adrianlizarraga): Fails in QNN SDK 2.21 (linux). On Windows ARM64, fails in QNN SDK 2.19.
//
// input: [-12.0, -7.199, -2.399, 2.4, 7.2, 12.0]
// CPU EP f32 model output: [-12.0, -7.0, -2.0, 3.0, 8.0, 12.0]
// CPU EP qdq model output: [-12.0, -6.99, -1.99, 3.0, 8.0, 11.99]
// QNN EP qdq model output: [-11.0 (WRONG), -7.0, -2.0, 2.99, 8.0, 11.99]
TEST_F(QnnHTPBackendTests, DISABLED_UnaryOp_Ceil_U16) {
  const std::vector<float> input_data = GetFloatDataInRange(-12.0f, 12.0f, 6);
  RunQDQOpTest<uint16_t>("Ceil",
                         {TestInputDef<float>({1, 2, 3}, false, input_data)},
                         {},
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,  // Ceil op's domain
                         true);        // Use com.microsoft domain for Q/DQ ops
}

// Test QDQ Floor op.
TEST_F(QnnHTPBackendTests, UnaryOp_Floor) {
  const std::vector<float> input_data = GetFloatDataInRange(-12.0f, 12.0f, 6);
  RunQDQOpTest<uint8_t>("Floor",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Test QDQ DepthToSpace.
TEST_F(QnnHTPBackendTests, DepthToSpaceOp_CRD) {
  const std::vector<float> X = {0., 1., 2.,
                                3., 4., 5.,
                                9., 10., 11.,
                                12., 13., 14.,
                                18., 19., 20.,
                                21., 22., 23.,
                                27., 28., 29.,
                                30., 31., 32.};
  RunQDQOpTest<uint8_t>("DepthToSpace",
                        {TestInputDef<float>({1, 4, 2, 3}, false, X)},
                        {utils::MakeAttribute("blocksize", static_cast<int64_t>(2)),
                         utils::MakeAttribute("mode", "CRD")},
                        11,
                        ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ DepthToSpace.
TEST_F(QnnHTPBackendTests, DepthToSpaceOp_U16_CRD) {
  const std::vector<float> X = {0., 1., 2.,
                                3., 4., 5.,
                                9., 10., 11.,
                                12., 13., 14.,
                                18., 19., 20.,
                                21., 22., 23.,
                                27., 28., 29.,
                                30., 31., 32.};
  RunQDQOpTest<uint16_t>("DepthToSpace",
                         {TestInputDef<float>({1, 4, 2, 3}, false, X)},
                         {utils::MakeAttribute("blocksize", static_cast<int64_t>(2)),
                          utils::MakeAttribute("mode", "CRD")},
                         11,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,  // Op's domain
                         true);        // Use com.microsoft domain for Q/DQ ops
}

// Test QDQ DepthToSpace.
TEST_F(QnnHTPBackendTests, DepthToSpaceOp_DCR) {
  const std::vector<float> X = {0., 1., 2.,
                                3., 4., 5.,
                                9., 10., 11.,
                                12., 13., 14.,
                                18., 19., 20.,
                                21., 22., 23.,
                                27., 28., 29.,
                                30., 31., 32.};
  RunQDQOpTest<uint8_t>("DepthToSpace",
                        {TestInputDef<float>({1, 4, 2, 3}, false, X)},
                        {utils::MakeAttribute("blocksize", static_cast<int64_t>(2)),
                         utils::MakeAttribute("mode", "DCR")},
                        11,
                        ExpectedEPNodeAssignment::All);
}

// Test QDQ SpaceToDepth.
TEST_F(QnnHTPBackendTests, SpaceToDepthOp) {
  const std::vector<float> X = {0.0f, 0.1f, 0.2f, 0.3f,
                                1.0f, 1.1f, 1.2f, 1.3f,

                                2.0f, 2.1f, 2.2f, 2.3f,
                                3.0f, 3.1f, 3.2f, 3.3f};
  RunQDQOpTest<uint8_t>("SpaceToDepth",
                        {TestInputDef<float>({1, 2, 2, 4}, false, X)},
                        {utils::MakeAttribute("blocksize", static_cast<int64_t>(2))},
                        11,
                        ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ SpaceToDepth.
TEST_F(QnnHTPBackendTests, SpaceToDepthOp_U16) {
  const std::vector<float> X = {0.0f, 0.1f, 0.2f, 0.3f,
                                1.0f, 1.1f, 1.2f, 1.3f,

                                2.0f, 2.1f, 2.2f, 2.3f,
                                3.0f, 3.1f, 3.2f, 3.3f};
  RunQDQOpTest<uint16_t>("SpaceToDepth",
                         {TestInputDef<float>({1, 2, 2, 4}, false, X)},
                         {utils::MakeAttribute("blocksize", static_cast<int64_t>(2))},
                         11,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,  // Op's domain
                         true);        // Use com.microsoft domain for Q/DQ ops
}

TEST_F(QnnHTPBackendTests, QuantAccuracyTest) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Note: a graph input -> Q -> DQ -> is optimized by Qnn to have a perfectly accurate output.
  // ORT's CPU EP, on the otherhand, actually quantizes and dequantizes the input, which leads to different outputs.
  auto builder_func = [](ModelTestBuilder& builder) {
    const TestInputDef<float> input0_def({1, 2, 3}, false, {1.0f, 2.0f, 10.0f, 20.0f, 100.0f, 200.0f});

    // input -> Q -> Transpose -> DQ -> output
    NodeArg* input0 = MakeTestInput(builder, input0_def);
    QuantParams<uint8_t> qparams = GetTestInputQuantParams<uint8_t>(input0_def);

    auto* quant_input = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(input0, qparams.scale, qparams.zero_point, quant_input);

    auto* op_output = builder.MakeIntermediate();
    builder.AddNode("Transpose", {quant_input}, {op_output});

    NodeArg* output = builder.MakeOutput();
    builder.AddDequantizeLinearNode<uint8_t>(op_output, qparams.scale, qparams.zero_point, output);
  };

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Qnn context cache binary file
  RunQnnModelTest(builder_func,
                  provider_options,
                  13,
                  ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ Add
TEST_F(QnnHTPBackendTests, BinaryOp_Add4D) {
  RunQDQOpTest<uint8_t>("Add",
                        {TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f),
                         TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f)},
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Add
TEST_F(QnnHTPBackendTests, BinaryOp_Add4D_U16) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunQDQOpTest<uint16_t>("Add",
                         {TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                          TestInputDef<float>({1, 2, 2, 2}, false, input_data)},
                         {},
                         17,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);  // Use com.microsoft Q/DQ ops
}

// Test 8-bit QDQ Sub
TEST_F(QnnHTPBackendTests, BinaryOp_Sub4D) {
  RunQDQOpTest<uint8_t>("Sub",
                        {TestInputDef<float>({1, 3, 8, 8}, false, -10.0f, 10.0f),
                         TestInputDef<float>({1, 3, 8, 8}, false, -10.0f, 10.0f)},
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Sub
TEST_F(QnnHTPBackendTests, BinaryOp_Sub4D_U16) {
  std::vector<float> input0_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  std::vector<float> input1_data = GetFloatDataInRange(0.0f, 20.0f, 8);
  RunQDQOpTest<uint16_t>("Sub",
                         {TestInputDef<float>({1, 2, 2, 2}, false, input0_data),
                          TestInputDef<float>({1, 2, 2, 2}, false, input1_data)},
                         {},
                         17,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);  // Use com.microsoft Q/DQ ops
}

TEST_F(QnnHTPBackendTests, BinaryOp_Sub4D_LargeInputs) {
  RunQDQOpTest<uint8_t>("Sub",
                        {TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                         TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f)},
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Sub4D_Broadcast) {
  RunQDQOpTest<uint8_t>("Sub",
                        {TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                         TestInputDef<float>({3, 1, 1}, true, {1.0f, 0.5f, -0.3f})},
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test accuracy of QDQ Pow
#if defined(__linux__)
// TODO: This fails on Linux (HTP emulation). Works on Windows ARM64.
// Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.051073111593723297, zero_point=2.
// Expected val: 0.0099999997764825821
// QNN QDQ val: 12.921497344970703 (err 12.911497116088867)
// CPU QDQ val: -0.10214622318744659 (err 0.11214622110128403)
TEST_F(QnnHTPBackendTests, DISABLED_BinaryOp_Pow) {
#else
TEST_F(QnnHTPBackendTests, BinaryOp_Pow) {
#endif
  std::vector<float> bases_input = {-10.0f, -8.0f, -6.0f, 1.0f, 2.0f, 3.0f, 5.5f, 10.0f};
  std::vector<float> exponents_input = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 1.5f, 0.2f};
  RunQDQOpTest<uint8_t>("Pow",
                        {TestInputDef<float>({1, 2, 2, 2}, false, bases_input),
                         TestInputDef<float>({1, 2, 2, 2}, false, exponents_input)},
                        {},
                        15,
                        ExpectedEPNodeAssignment::All);
}

// Test accuracy of QDQ PRelu with dynamic slopes.
TEST_F(QnnHTPBackendTests, BinaryOp_PRelu_DynamicSlopes) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  std::vector<float> slopes_data = GetFloatDataInRange(-1.0f, 1.0f, 8);
  RunQDQOpTest<uint8_t>("PRelu",
                        {TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                         TestInputDef<float>({1, 2, 2, 2}, false, slopes_data)},
                        {},
                        16,
                        ExpectedEPNodeAssignment::All);
}

// Test accuracy of QDQ PRelu with static slope weights.
TEST_F(QnnHTPBackendTests, BinaryOp_PRelu_StaticSlopes) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  std::vector<float> slopes_data = GetFloatDataInRange(-1.0f, 1.0f, 8);
  RunQDQOpTest<uint8_t>("PRelu",
                        {TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                         TestInputDef<float>({1, 2, 2, 2}, true, slopes_data)},
                        {},
                        16,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Div4D_SmallInputs) {
  std::vector<float> input0_data = {-10.0f, -8.0f, -1.0f, 0.0f, 1.0f, 2.1f, 8.0f, 10.0f};
  std::vector<float> input1_data = {5.0f, 4.0f, 1.0f, 1.0f, 1.0f, 4.0f, 4.0f, 5.0f};
  RunQDQOpTest<uint8_t>("Div",
                        {TestInputDef<float>({1, 2, 2, 2}, false, input0_data),
                         TestInputDef<float>({1, 2, 2, 2}, false, input1_data)},
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Sub with small input values.
TEST_F(QnnHTPBackendTests, BinaryOp_Div4D_U16_SmallInputs) {
  std::vector<float> input0_data = {-10.0f, -8.0f, -1.0f, 0.0f, 1.0f, 2.1f, 8.0f, 10.0f};
  std::vector<float> input1_data = {5.0f, 4.0f, 1.0f, 1.0f, 1.0f, 4.0f, 4.0f, 5.0f};
  RunQDQOpTest<uint16_t>("Div",
                         {TestInputDef<float>({1, 2, 2, 2}, false, input0_data),
                          TestInputDef<float>({1, 2, 2, 2}, false, input1_data)},
                         {},
                         17,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);  // Use com.microsoft Q/DQ ops
}

// TODO: Enable when this is fixed.
// QNN v2.13: Inaccuracy detected for output 'output', element 2551923.
// Output quant params: scale=4100.92626953125, zero_point=126.
// Expected val: -277957.3125
// QNN QDQ val: 0 (err 277957.3125)
// CPU QDQ val: -516716.71875 (err 238759.40625)
TEST_F(QnnHTPBackendTests, DISABLED_BinaryOp_Div4D_LargeInputs) {
  RunQDQOpTest<uint8_t>("Div",
                        {TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                         TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f)},
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BinaryOp_Div4D_Broadcast) {
  RunQDQOpTest<uint8_t>("Div",
                        {TestInputDef<float>({1, 3, 768, 1152}, false, -1.0f, 1.0f),
                         TestInputDef<float>({3, 1, 1}, true, {1.0f, 0.5f, -0.3f})},
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ Mul
TEST_F(QnnHTPBackendTests, BinaryOp_Mul4D) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0, 10.0f, 8);
  RunQDQOpTest<uint8_t>("Mul",
                        {TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                         TestInputDef<float>({1, 2, 2, 2}, false, input_data)},
                        {},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Mul
TEST_F(QnnHTPBackendTests, BinaryOp_Mul4D_U16) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunQDQOpTest<uint16_t>("Mul",
                         {TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                          TestInputDef<float>({1, 2, 2, 2}, false, input_data)},
                         {},
                         17,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);  // Use com.microsoft Q/DQ ops
}

// Test And
TEST_F(QnnHTPBackendTests, BinaryOp_And4D) {
  RunOpTest<bool>("And",
                  {TestInputDef<bool>({1, 4}, false, {false, false, true, true}),
                   TestInputDef<bool>({1, 4}, false, {false, true, false, true})},
                  {},
                  17,
                  ExpectedEPNodeAssignment::All);
}

// Test that Or is not yet supported on CPU backend.
TEST_F(QnnHTPBackendTests, BinaryOp_HTP_Or_Unsupported) {
  RunOpTest<bool>("Or",
                  {TestInputDef<bool>({1, 4}, false, {false, false, true, true}),
                   TestInputDef<bool>({1, 4}, false, {false, true, false, true})},
                  {},
                  17,
                  ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ GridSample with bilinear
TEST_F(QnnHTPBackendTests, GridSample_Bilinear) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                         TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                        {utils::MakeAttribute("align_corners", static_cast<int64_t>(0)),
                         utils::MakeAttribute("mode", "bilinear"),
                         utils::MakeAttribute("padding_mode", "zeros")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ GridSample with bilinear
TEST_F(QnnHTPBackendTests, GridSample_U16_Bilinear) {
  RunQDQOpTest<uint16_t>("GridSample",
                         {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                          TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                         {utils::MakeAttribute("align_corners", static_cast<int64_t>(0)),
                          utils::MakeAttribute("mode", "bilinear"),
                          utils::MakeAttribute("padding_mode", "zeros")},
                         17,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);  // Use com.microsoft Q/DQ ops
}

// Test 8-bit QDQ GridSample with align corners
TEST_F(QnnHTPBackendTests, GridSample_AlignCorners) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                         TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                        {utils::MakeAttribute("align_corners", static_cast<int64_t>(1)),
                         utils::MakeAttribute("mode", "bilinear"),
                         utils::MakeAttribute("padding_mode", "zeros")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ GridSample with align corners
TEST_F(QnnHTPBackendTests, GridSample_U16_AlignCorners) {
  RunQDQOpTest<uint16_t>("GridSample",
                         {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                          TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                         {utils::MakeAttribute("align_corners", static_cast<int64_t>(1)),
                          utils::MakeAttribute("mode", "bilinear"),
                          utils::MakeAttribute("padding_mode", "zeros")},
                         17,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);  // Use com.microsoft Q/DQ ops
}

// Test QDQ GridSample with padding mode: border
// Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.046370312571525574, zero_point=129.
// Expected val: 3.3620510101318359
// QNN QDQ val: 3.2922921180725098 (err 0.069758892059326172)
// CPU QDQ val: 3.3850328922271729 (err 0.022981882095336914)
TEST_F(QnnHTPBackendTests, DISABLED_GridSample_BorderPadding) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({1, 1, 3, 2}, false, -10.0f, 10.0f),
                         TestInputDef<float>({1, 2, 4, 2}, false, -10.0f, 10.0f)},
                        {utils::MakeAttribute("mode", "bilinear"),
                         utils::MakeAttribute("padding_mode", "border")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ GridSample with nearest mode
TEST_F(QnnHTPBackendTests, GridSample_Nearest) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                         TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                        {utils::MakeAttribute("mode", "nearest")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ GridSample with nearest mode
TEST_F(QnnHTPBackendTests, GridSample_U16_Nearest) {
  RunQDQOpTest<uint16_t>("GridSample",
                         {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                          TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                         {utils::MakeAttribute("mode", "nearest")},
                         17,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);
}

// Test QDQ GridSample with reflection padding mode
// Inaccuracy detected for output 'output', element 2.
// Output quant params: scale=0.024269860237836838, zero_point=0.
// Expected val: 3.212885856628418
// QNN QDQ val: 3.1308119297027588 (err 0.08207392692565918)
// CPU QDQ val: 3.2036216259002686 (err 0.0092642307281494141)
TEST_F(QnnHTPBackendTests, DISABLED_GridSample_ReflectionPaddingMode) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({1, 1, 3, 2}, false, -10.0f, 10.0f),
                         TestInputDef<float>({1, 2, 4, 2}, false, -10.0f, 10.0f)},
                        {utils::MakeAttribute("padding_mode", "reflection")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test QDQ Concat: 3 inputs concatenated at the last axis.
TEST_F(QnnHTPBackendTests, VariadicOp_Concat_3Inputs_LastAxis) {
  RunQDQOpTest<uint8_t>("Concat",
                        {TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f),
                         TestInputDef<float>({1, 2, 2, 3}, false, -1.0f, 1.0f),
                         TestInputDef<float>({1, 2, 2, 1}, false, -2.0f, 2.0f)},
                        {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Test QDQ Concat: 2 inputs concatenated at the second axis.
TEST_F(QnnHTPBackendTests, VariadicOp_Concat_2Inputs_2ndAxis) {
  RunQDQOpTest<uint8_t>("Concat",
                        {TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f),
                         TestInputDef<float>({1, 3, 2, 2}, false, -2.0f, 2.0f)},
                        {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                        13,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LpNormalization_u8_rank4) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunQDQOpTest<uint8_t>("LpNormalization",
                        {TestInputDef<float>({1, 2, 2, 2}, false, input_data)},
                        {utils::MakeAttribute("axis", static_cast<int64_t>(-1)),  // Last axis
                         utils::MakeAttribute("p", static_cast<int64_t>(2))},     // Order 2 to map to QNN's L2Norm operator
                        13,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LpNormalization_u16_rank4) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunQDQOpTest<uint16_t>("LpNormalization",
                         {TestInputDef<float>({1, 2, 2, 2}, false, input_data)},
                         {utils::MakeAttribute("axis", static_cast<int64_t>(-1)),  // Last axis
                          utils::MakeAttribute("p", static_cast<int64_t>(2))},     // Order 2 to map to QNN's L2Norm operator
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);
}

static GetTestQDQModelFn<uint16_t> BuildQDQConvertAddTestCase(const TestInputDef<float>& input0_def,
                                                              const TestInputDef<float>& input1_def) {
  return [input0_def, input1_def](ModelTestBuilder& builder, std::vector<QuantParams<uint16_t>>& output_qparams) {
    constexpr bool use_contrib_qdq = true;

    // Input0 -> Quantize(u8) -> Dequantize(u8 to float) -> input0_after_qdq
    NodeArg* input0 = MakeTestInput<float>(builder, input0_def);
    QuantParams<uint8_t> input0_u8_qparams = GetTestInputQuantParams<uint8_t>(input0_def);
    NodeArg* input0_after_qdq = AddQDQNodePair<uint8_t>(builder, input0, input0_u8_qparams.scale,
                                                        input0_u8_qparams.zero_point, use_contrib_qdq);

    // input0_after_qdq -> Quantize(u16) -> Dequantize(u16 to float)
    QuantParams<uint16_t> input0_u16_qparams = GetTestInputQuantParams<uint16_t>(input0_def);
    NodeArg* input0_after_convert = AddQDQNodePair<uint16_t>(builder, input0_after_qdq, input0_u16_qparams.scale,
                                                             input0_u16_qparams.zero_point, use_contrib_qdq);

    // Input1 -> Quantize(u16) -> Dequantize(u16 to float) -> input1_after_qdq
    NodeArg* input1 = MakeTestInput<float>(builder, input1_def);
    QuantParams<uint16_t> input1_qparams = GetTestInputQuantParams<uint16_t>(input1_def);
    NodeArg* input1_after_qdq = AddQDQNodePair<uint16_t>(builder, input1, input1_qparams.scale,
                                                         input1_qparams.zero_point, use_contrib_qdq);

    // Add op -> op_output
    auto* op_output = builder.MakeIntermediate();
    builder.AddNode("Add", {input0_after_convert, input1_after_qdq}, {op_output});

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, op_output, output_qparams[0].scale,
                                                    output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// Test quantization type conversion (mixed precision) with Add.
// First input is converted from uint8_t to uint16_t.
TEST_F(QnnHTPBackendTests, Add_U8_U16_Convert) {
  std::vector<float> input0_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  std::vector<float> input1_data = GetFloatDataInRange(-20.0f, 20.0f, 8);
  TestInputDef<float> input0_def({1, 2, 2, 2}, false, input0_data);
  TestInputDef<float> input1_def({1, 2, 2, 2}, false, input1_data);

  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(BuildOpTestCase<float>("Add", {input0_def, input1_def}, {}, {}, kOnnxDomain),
                       BuildQDQConvertAddTestCase(input0_def, input1_def),
                       provider_options,
                       18,
                       ExpectedEPNodeAssignment::All);
}

// Builds a graph where a (DQ -> Q) sequence at the graph's output is fuse into a QNN Convert operator.
// ONNX Graph: DQ -> Add -> Q -> DQ -> Q -> graph_output
// QNN Graph:  DQ -> Add -> Q -> Convert -> graph_output
template <typename InQuantType, typename OutQuantType>
static GetTestModelFn BuildDQQConvertAtOutputTestCase(const TestInputDef<float>& input0_def,
                                                      const TestInputDef<float>& input1_def,
                                                      const QuantParams<OutQuantType>& output_qparams) {
  return [input0_def, input1_def, output_qparams](ModelTestBuilder& builder) {
    // Input0 -> Quantize(InQuantType) -> Dequantize(InQuantType to float) -> input0_after_qdq
    NodeArg* input0 = MakeTestInput<float>(builder, input0_def);
    QuantParams<InQuantType> input0_qparams = GetTestInputQuantParams<InQuantType>(input0_def);
    NodeArg* input0_after_qdq = AddQDQNodePair<InQuantType>(builder, input0, input0_qparams.scale,
                                                            input0_qparams.zero_point);

    // Input1 -> Quantize(InQuantType) -> Dequantize(InQuantType to float) -> input1_after_qdq
    NodeArg* input1 = MakeTestInput<float>(builder, input1_def);
    QuantParams<InQuantType> input1_qparams = GetTestInputQuantParams<InQuantType>(input1_def);
    NodeArg* input1_after_qdq = AddQDQNodePair<InQuantType>(builder, input1, input1_qparams.scale,
                                                            input1_qparams.zero_point);

    // Add op -> op_output
    auto* op_output = builder.MakeIntermediate();
    builder.AddNode("Add", {input0_after_qdq, input1_after_qdq}, {op_output});

    // op_output -> Quantize(InQuantType) -> add_out_q
    QuantParams<InQuantType> add_out_qparams = ConvertQuantParams<OutQuantType, InQuantType>(output_qparams);
    add_out_qparams.scale *= 1.01f;  // Make qparams slightly different so DQ->Q are not optimized out.
    NodeArg* add_out_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<InQuantType>(op_output, add_out_qparams.scale,
                                               add_out_qparams.zero_point, add_out_q);

    // Add DQ
    NodeArg* add_out_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<InQuantType>(add_out_q, add_out_qparams.scale,
                                                 add_out_qparams.zero_point, add_out_dq);

    // Add a Q to quantize to OutQuantType
    // The previous DQ and this Q will be fused into a QNN Convert.
    NodeArg* q_conv_out = builder.MakeOutput();
    builder.AddQuantizeLinearNode<OutQuantType>(add_out_dq, output_qparams.scale, output_qparams.zero_point,
                                                q_conv_out);
  };
}

// Test fusion of (DQ -> Q) into QNN's Convert op using the same quant type.
TEST_F(QnnHTPBackendTests, DQ_Q_ConvertFusion_SameType) {
  std::vector<float> input0_data = {-8.0f, -6.0, -2.0f, 0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
  std::vector<float> input1_data = {-8.0f, -6.0, -2.0f, 0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
  TestInputDef<float> input0_def({1, 2, 2, 2}, false, input0_data);
  TestInputDef<float> input1_def({1, 2, 2, 2}, false, input1_data);

  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  QuantParams<uint8_t> out_qparams_u8 = {1.0f, 128};
  QuantParams<uint16_t> out_qparams_u16 = {1.0f, 32768};

  // QNN Convert op converts uint8 to uint8 at the graph output. Slightly different scale values.
  RunQnnModelTest(BuildDQQConvertAtOutputTestCase<uint8_t, uint8_t>(input0_def, input1_def, out_qparams_u8),
                  provider_options,
                  21,
                  ExpectedEPNodeAssignment::All);

  // QNN Convert op converts uint16 to uint16 at the graph output. Slightly different scale values.
  RunQnnModelTest(BuildDQQConvertAtOutputTestCase<uint16_t, uint16_t>(input0_def, input1_def, out_qparams_u16),
                  provider_options,
                  21,
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_QU8) {
  RunQDQOpTest<uint8_t>("HardSigmoid",
                        {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                        {utils::MakeAttribute("alpha", 0.1f),
                         utils::MakeAttribute("beta", 0.4f)},
                        21,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_QU16) {
  RunQDQOpTest<uint16_t>("HardSigmoid",
                         {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 6))},
                         {},
                         21,
                         ExpectedEPNodeAssignment::All);
}

// Test that QDQ HardSigmoid is supported by QNN EP.
TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_QDQ_Supported) {
  RunQDQOpTest<uint8_t>("HardSigmoid",
                        {TestInputDef<float>({1, 2, 2, 2}, false, -10.0f, 10.0f)},
                        {},
                        19,
                        ExpectedEPNodeAssignment::All);
}

// Check that QNN EP can support float32 HardSigmoid on HTP.
// Enables running f32 ops using fp16 precision.
TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_F32_as_FP16) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 16);

  RunOpTest<float>("HardSigmoid",
                   {TestInputDef<float>({1, 2, 8}, false, input_data)},
                   {},
                   21,
                   ExpectedEPNodeAssignment::All,
                   kOnnxDomain,
                   0.004f,  // Tolerance. Comparing fp16 (QNN) with fp32 (CPU EP), so expect to need larger tolerance.
                   true);   // enable_htp_fp16_precision

  // Rank 4, non-default alpha and beta
  RunOpTest<float>("HardSigmoid",
                   {TestInputDef<float>({1, 2, 2, 4}, false, input_data)},
                   {utils::MakeAttribute("alpha", 0.1f),
                    utils::MakeAttribute("beta", 0.4f)},
                   21,
                   ExpectedEPNodeAssignment::All,
                   kOnnxDomain,
                   0.004f,  // Tolerance. Comparing fp16 (QNN) with fp32 (CPU EP), so expect to need larger tolerance.
                   true);   // enable_htp_fp16_precision
}

// Check that QNN EP can support float16 HardSigmoid on HTP
TEST_F(QnnHTPBackendTests, UnaryOp_HardSigmoid_FP16) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 16);

  RunFP16OpTest("HardSigmoid",
                {TestInputDef<float>({1, 2, 8}, false, input_data)},
                {},
                21,
                ExpectedEPNodeAssignment::All,
                kOnnxDomain);

  // Rank 4, non-default alpha and beta
  RunFP16OpTest("HardSigmoid",
                {TestInputDef<float>({1, 2, 2, 4}, false, input_data)},
                {utils::MakeAttribute("alpha", 0.1f),
                 utils::MakeAttribute("beta", 0.4f)},
                21,
                ExpectedEPNodeAssignment::All,
                kOnnxDomain);
}

// Returns a function that creates the model `X * HardSigmoid(X)`, which can be potentially fused
// into a single HardSwish(X) operator.
template <typename FloatType>
static GetTestModelFn BuildHardSigmoidFusionTestCase(TestInputDef<FloatType>& input_def,
                                                     std::optional<float> alpha,
                                                     std::optional<float> beta) {
  return [input_def, alpha, beta](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput<FloatType>(builder, input_def);

    // input -> HardSigmoid<alpha, beta> -> hs_output
    NodeArg* hs_output = builder.MakeIntermediate();
    Node& hs_node = builder.AddNode("HardSigmoid", {input}, {hs_output});

    if (alpha.has_value()) {
      hs_node.AddAttribute("alpha", alpha.value());
    }

    if (beta.has_value()) {
      hs_node.AddAttribute("beta", beta.value());
    }

    // hs_output -> Mul -> output
    //               ^
    //               |
    // input --------+
    auto* output = builder.MakeOutput();
    builder.AddNode("Mul", {hs_output, input}, {output});
  };
}

// Test FP32 fusion of HardSigmoid into HardSwish on the HTP backend with the enable_htp_fp16_precision option enabled
// to run it with fp16 precision.
TEST_F(QnnHTPBackendTests, HardSigmoidFusedIntoHardSwish_FP32_as_FP16) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  provider_options["enable_htp_fp16_precision"] = "1";

  std::vector<float> input_data = {-8.0f, -2.0f, 0.0f, 0.5f, 0.9f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, 0.0f, 0.2f, 0.4f, 0.8f, 2.1f, 4.3f, 7.0f};

  auto input_def = TestInputDef<float>({2, 2, 2, 2}, false, input_data);
  constexpr float alpha = 1.0f / 6.0f;
  constexpr float beta = 0.5f;
  auto model_fn = BuildHardSigmoidFusionTestCase<float>(input_def, alpha, beta);

  RunQnnModelTest(model_fn,
                  provider_options,
                  18,  // opset
                  ExpectedEPNodeAssignment::All,
                  0.01f);  // abs err. Comparing fp16 (QNN) vs fp32 (CPU EP) so can't expect too much.
}

// Test FP16 fusion of HardSigmoid into HardSwish on the HTP backend.
TEST_F(QnnHTPBackendTests, HardSigmoidFusedIntoHardSwish_FP16) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::vector<float> input_data = {-8.0f, -2.0f, 0.0f, 0.5f, 0.9f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, 0.0f, 0.2f, 0.4f, 0.8f, 2.1f, 4.3f, 7.0f};

  auto input_def = TestInputDef<float>({2, 2, 2, 2}, false, input_data);
  auto input_fp16_def = ConvertToFP16InputDef(input_def);

  constexpr float alpha = 1.0f / 6.0f;
  constexpr float beta = 0.5f;
  auto model_fp32_fn = BuildHardSigmoidFusionTestCase<float>(input_def, alpha, beta);
  auto model_fp16_fn = BuildHardSigmoidFusionTestCase<MLFloat16>(input_fp16_def, alpha, beta);

  TestFp16ModelAccuracy(model_fp32_fn,
                        model_fp16_fn,
                        provider_options,
                        18,  // opset
                        ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
