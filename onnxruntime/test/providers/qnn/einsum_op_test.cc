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

using onnxruntime::ProviderOptions;
using onnxruntime::test::BuildOpTestCase;
using onnxruntime::test::ExpectedEPNodeAssignment;
using onnxruntime::test::RunQnnModelTest;
using onnxruntime::test::TestInputDef;
using onnxruntime::utils::MakeAttribute;

template <typename DataType>
static void RunQnnEinsum(
    const std::string& backend,
    const TestInputDef<DataType>& in0,
    const TestInputDef<DataType>& in1,
    const std::string& equation,
    const float f32_abs_err = 1e-4f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend;
  provider_options["offload_graph_io_quantization"] = "0";
  RunQnnModelTest(
      /*build_test_case=*/BuildOpTestCase<DataType, DataType>(
          /*op_type=*/"Einsum",
          /*input_defs_1=*/{in0, in1},
          /*input_defs_2=*/{},
          /*attrs=*/{MakeAttribute("equation", equation)}),
      /*provider_options=*/provider_options,
      /*opset_version=*/13,
      /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
      /*f32_abs_err=*/f32_abs_err);
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
      /*backend=*/"cpu",
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"ab,bc->ac");
}

TEST_F(QnnCPUBackendTests, EinsumRank4MatMul) {
  const std::vector<int64_t> shape0{3, 4, 5, 6};
  const std::vector<int64_t> shape1{3, 4, 6, 5};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/"cpu",
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bhij,bhjd->bhid");
}

TEST_F(QnnCPUBackendTests, EinsumRank4MatMulTransposeY) {
  const std::vector<int64_t> shape0{2, 3, 4, 6};
  const std::vector<int64_t> shape1{2, 3, 5, 6};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/"cpu",
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bhid,bhjd->bhij");
}

TEST_F(QnnCPUBackendTests, EinsumRank4MatMulTransposeAll1) {
  const std::vector<int64_t> shape0{1, 9, 1, 7};
  const std::vector<int64_t> shape1{1, 7, 1, 9};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/"cpu",
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bchq,bkhc->bkhq");
}

TEST_F(QnnCPUBackendTests, EinsumRank4MatMulTransposeAll2) {
  const std::vector<int64_t> shape0{1, 7, 1, 7};
  const std::vector<int64_t> shape1{1, 9, 1, 7};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/"cpu",
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bkhq,bchk->bchq");
}

//
// QNN HTP
//

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, EinsumRank2MatMul) {
  const std::vector<int64_t> shape0{2, 3};
  const std::vector<int64_t> shape1{3, 4};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/"htp",
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"ij,jk->ik",
      /*f32_abs_err=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, EinsumRank4MatMul) {
  const std::vector<int64_t> shape0{3, 1, 5, 2};
  const std::vector<int64_t> shape1{3, 1, 2, 5};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/"htp",
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bhij,bhjd->bhid",
      /*f32_abs_err=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, EinsumRank4MatMulTransposeY) {
  const std::vector<int64_t> shape0{2, 3, 4, 2};
  const std::vector<int64_t> shape1{2, 3, 5, 2};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/"htp",
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bhid,bhjd->bhij",
      /*f32_abs_err=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, EinsumRank4MatMulTransposeAll1) {
  const std::vector<int64_t> shape0{1, 3, 1, 7};
  const std::vector<int64_t> shape1{1, 7, 1, 3};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/"htp",
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bchq,bkhc->bkhq",
      /*f32_abs_err=*/1e-2f);
}

TEST_F(QnnHTPBackendTests, EinsumRank4MatMulTransposeAll2) {
  const std::vector<int64_t> shape0{1, 4, 1, 4};
  const std::vector<int64_t> shape1{1, 9, 1, 4};
  const std::vector<float> data0 = GetSequentialFloatData(shape0, /*start=*/-0.1f, /*step=*/0.05f);
  const std::vector<float> data1 = GetSequentialFloatData(shape1, /*start=*/-0.1f, /*step=*/0.05f);
  RunQnnEinsum<float>(
      /*backend=*/"htp",
      /*in0=*/TestInputDef<float>(shape0, /*is_initializer=*/false, std::move(data0)),
      /*in1=*/TestInputDef<float>(shape1, /*is_initializer=*/false, std::move(data1)),
      /*equation=*/"bkhq,bchk->bchq",
      /*f32_abs_err=*/1e-2f);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
