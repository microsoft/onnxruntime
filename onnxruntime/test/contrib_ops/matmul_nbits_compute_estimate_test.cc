// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/matmul_nbits_compute_estimate.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/framework/runtime_workload_collector.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#endif

namespace onnxruntime {
namespace test {
namespace {

using contrib::ComputeEstimateUnavailableReason;
using contrib::EstimateMatMulNBitsBias;
using contrib::EstimateMatMulNBitsCompute;
using contrib::MatMulNBitsBiasEstimate;
using contrib::MatMulNBitsBiasEstimateResult;
using contrib::MatMulNBitsComputeEstimate;
using contrib::MatMulNBitsComputeEstimateResult;

void ExpectBiasCount(const MatMulNBitsBiasEstimateResult& result, uint64_t expected) {
  const auto* estimate = std::get_if<MatMulNBitsBiasEstimate>(&result);
  ASSERT_NE(estimate, nullptr);
  EXPECT_EQ(estimate->floating_point_ops, expected);
}

void ExpectBiasUnavailable(const MatMulNBitsBiasEstimateResult& result, ComputeEstimateUnavailableReason expected) {
  const auto* reason = std::get_if<ComputeEstimateUnavailableReason>(&result);
  ASSERT_NE(reason, nullptr);
  EXPECT_EQ(*reason, expected);
}

TEST(MatMulNBitsBiasEstimateTest, MatrixBatchedAndRankOneCounts) {
  ExpectBiasCount(EstimateMatMulNBitsBias(std::array<int64_t, 2>{3, 64}, 64, 32, true), 96);
  ExpectBiasCount(EstimateMatMulNBitsBias(std::array<int64_t, 3>{2, 3, 64}, 64, 32, true), 192);
  ExpectBiasCount(EstimateMatMulNBitsBias(std::array<int64_t, 1>{64}, 64, 32, true), 32);
}

TEST(MatMulNBitsBiasEstimateTest, AbsentBiasDoesNotRequireGeometry) {
  ExpectBiasCount(EstimateMatMulNBitsBias({}, 64, 32, false), 0);
  ExpectBiasCount(EstimateMatMulNBitsBias(std::array<int64_t, 2>{-1, 64}, 64, 32, false), 0);
}

TEST(MatMulNBitsBiasEstimateTest, UnknownPresenceIsNotZero) {
  ExpectBiasUnavailable(EstimateMatMulNBitsBias(std::array<int64_t, 2>{0, 64}, 64, 32, std::nullopt),
                        ComputeEstimateUnavailableReason::MissingSemanticValue);
}

TEST(MatMulNBitsBiasEstimateTest, UnknownGeometryIsNotZero) {
  ExpectBiasUnavailable(EstimateMatMulNBitsBias({}, 64, 32, true),
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  for (const auto& shape : {std::array<int64_t, 3>{2, -1, 64}, std::array<int64_t, 3>{0, -1, 64},
                            std::array<int64_t, 3>{2, 3, -1}}) {
    ExpectBiasUnavailable(EstimateMatMulNBitsBias(shape, 64, 32, true),
                          ComputeEstimateUnavailableReason::UnknownShape);
  }
}

TEST(MatMulNBitsBiasEstimateTest, InvalidGeometry) {
  ExpectBiasUnavailable(EstimateMatMulNBitsBias(std::array<int64_t, 2>{-2, 64}, 64, 32, true),
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectBiasUnavailable(EstimateMatMulNBitsBias(std::array<int64_t, 2>{3, 63}, 64, 32, true),
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectBiasUnavailable(EstimateMatMulNBitsBias(std::array<int64_t, 2>{3, 64}, -1, 32, true),
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectBiasUnavailable(EstimateMatMulNBitsBias(std::array<int64_t, 2>{3, 64}, 64, -1, true),
                        ComputeEstimateUnavailableReason::InvalidMetadata);
}

TEST(MatMulNBitsBiasEstimateTest, EmptyOutputAndZeroReduction) {
  ExpectBiasCount(EstimateMatMulNBitsBias(std::array<int64_t, 2>{0, 64}, 64, 32, true), 0);
  ExpectBiasCount(EstimateMatMulNBitsBias(std::array<int64_t, 2>{3, 64}, 64, 0, true), 0);
  ExpectBiasCount(EstimateMatMulNBitsBias(std::array<int64_t, 2>{3, 0}, 0, 32, true), 96);
}

TEST(MatMulNBitsBiasEstimateTest, ZeroPrecedesOverflow) {
  constexpr auto max = std::numeric_limits<int64_t>::max();
  ExpectBiasCount(EstimateMatMulNBitsBias(std::array<int64_t, 4>{max, max, 0, 64}, 64, max, true), 0);
}

TEST(MatMulNBitsBiasEstimateTest, OverflowAndIndependentCoreCount) {
  constexpr auto max = std::numeric_limits<int64_t>::max();
  ExpectBiasCount(EstimateMatMulNBitsBias(std::array<int64_t, 2>{max, 64}, 64, 2, true),
                  std::numeric_limits<uint64_t>::max() - 1);
  ExpectBiasUnavailable(EstimateMatMulNBitsBias(std::array<int64_t, 2>{max, 64}, 64, 3, true),
                        ComputeEstimateUnavailableReason::Overflow);
  ExpectBiasCount(EstimateMatMulNBitsBias(std::array<int64_t, 1>{max}, max, 32, true), 32);
}

void ExpectCount(const MatMulNBitsComputeEstimateResult& result, uint64_t expected) {
  const auto* estimate = std::get_if<MatMulNBitsComputeEstimate>(&result);
  ASSERT_NE(estimate, nullptr);
  EXPECT_EQ(estimate->dense_equivalent_ops, expected);
  EXPECT_EQ(estimate->kConventionVersion, 1u);
}

void ExpectUnavailable(const MatMulNBitsComputeEstimateResult& result,
                       ComputeEstimateUnavailableReason expected) {
  const auto* reason = std::get_if<ComputeEstimateUnavailableReason>(&result);
  ASSERT_NE(reason, nullptr);
  EXPECT_EQ(*reason, expected);
}

TEST(MatMulNBitsComputeEstimateTest, MatrixAndBatchedCounts) {
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, 64}, 64, 32), 12288);
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 3>{2, 3, 64}, 64, 32), 24576);
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 4>{2, 3, 4, 64}, 64, 32), 98304);
}

TEST(MatMulNBitsComputeEstimateTest, RankOneInput) {
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 1>{64}, 64, 32), 4096);
}

TEST(MatMulNBitsComputeEstimateTest, EmptyWorkloads) {
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{0, 64}, 64, 32), 0);
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, 64}, 64, 0), 0);
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, 0}, 0, 32), 0);
}

TEST(MatMulNBitsComputeEstimateTest, ZeroPrecedesProductOverflow) {
  constexpr auto max = std::numeric_limits<int64_t>::max();
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 4>{max, max, 0, 64}, 64, max), 0);
}

TEST(MatMulNBitsComputeEstimateTest, UnknownDimensionsAreNotZero) {
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{-1, 64}, 64, 32),
                    ComputeEstimateUnavailableReason::UnknownShape);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, -1}, 64, 32),
                    ComputeEstimateUnavailableReason::UnknownShape);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 3>{0, -1, 64}, 64, 32),
                    ComputeEstimateUnavailableReason::UnknownShape);
}

TEST(MatMulNBitsComputeEstimateTest, InvalidMetadata) {
  ExpectUnavailable(EstimateMatMulNBitsCompute({}, 64, 32), ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, 63}, 64, 32),
                    ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{-2, 64}, 64, 32),
                    ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, 64}, -1, 32),
                    ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{3, 64}, 64, -1),
                    ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 3>{0, -2, 64}, 64, 32),
                    ComputeEstimateUnavailableReason::InvalidMetadata);
}

TEST(MatMulNBitsComputeEstimateTest, CountsUseUnsigned64BitRange) {
  constexpr auto max = std::numeric_limits<int64_t>::max();
  ExpectCount(EstimateMatMulNBitsCompute(std::array<int64_t, 1>{max}, max, 1),
              std::numeric_limits<uint64_t>::max() - 1);
}

TEST(MatMulNBitsComputeEstimateTest, Overflow) {
  constexpr auto max = std::numeric_limits<int64_t>::max();
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 2>{2, max}, max, 1),
                    ComputeEstimateUnavailableReason::Overflow);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 1>{2}, 2, max),
                    ComputeEstimateUnavailableReason::Overflow);
  ExpectUnavailable(EstimateMatMulNBitsCompute(std::array<int64_t, 3>{max, max, 1}, 1, 1),
                    ComputeEstimateUnavailableReason::Overflow);
}

#if !defined(ORT_MINIMAL_BUILD)
MatMulNBitsRuntimeMetadata CpuFlopsMetadata(MatMulNBitsRuntimeRoute route, MatMulNBitsRuntimeComputeMode mode) {
  MatMulNBitsRuntimeMetadata metadata;
  metadata.input_shape = {2, 3, 64};
  metadata.output_shape = {2, 3, 32};
  metadata.k = 64;
  metadata.n = 32;
  metadata.bits = 4;
  metadata.block_size = 32;
  metadata.bias_present = true;
  metadata.input_element_type = ONNX_NAMESPACE::TensorProto::FLOAT;
  metadata.route = route;
  metadata.compute_mode = mode;
  return metadata;
}

TEST(MatMulNBitsCpuFlopsEstimateTest, FloatingCoreAndSingleReconstructionIncludeBatchesOnce) {
  auto metadata = CpuFlopsMetadata(MatMulNBitsRuntimeRoute::Unpacked, MatMulNBitsRuntimeComputeMode::Float32Gemm);
  for (const auto type : {ONNX_NAMESPACE::TensorProto::FLOAT, ONNX_NAMESPACE::TensorProto::FLOAT16}) {
    metadata.input_element_type = type;
    for (int bits : {2, 4, 8}) {
      metadata.bits = static_cast<uint32_t>(bits);
      const auto result = contrib::EstimateMatMulNBitsCpuFlops(metadata);
      EXPECT_EQ(result.model, contrib::MatMulNBitsCpuFlopsModel::UnpackedFloat);
      ExpectBiasCount(result.core, 24576);
      ExpectBiasCount(result.bias, 192);
      ExpectBiasCount(result.dequantization, 4096);
    }
  }
  metadata.input_shape = {64};
  metadata.m = 999;
  metadata.batch_count = 999;
  const auto result = contrib::EstimateMatMulNBitsCpuFlops(metadata);
  ExpectBiasCount(result.core, 4096);
  ExpectBiasCount(result.bias, 32);
  ExpectBiasCount(result.dequantization, 4096);
}

TEST(MatMulNBitsCpuFlopsEstimateTest, PackedFloatingModelsUsePaddedReconstructionNotPaddedCore) {
  auto metadata = CpuFlopsMetadata(MatMulNBitsRuntimeRoute::PackedQNBit, MatMulNBitsRuntimeComputeMode::SQNBitFp32);
  metadata.input_shape = {2, 3, 65};
  metadata.k = 65;
  for (const auto mode : {MatMulNBitsRuntimeComputeMode::SQNBitFp32, MatMulNBitsRuntimeComputeMode::HQNBitFp16}) {
    metadata.compute_mode = mode;
    metadata.input_element_type = ONNX_NAMESPACE::TensorProto::FLOAT16;
    const auto result = contrib::EstimateMatMulNBitsCpuFlops(metadata);
    EXPECT_EQ(result.model, contrib::MatMulNBitsCpuFlopsModel::PackedFloat);
    ExpectBiasCount(result.core, 24960);
    ExpectBiasCount(result.bias, 192);
    ExpectBiasCount(result.dequantization, 6144);
  }
}

TEST(MatMulNBitsCpuFlopsEstimateTest, IntegerCoreIsZeroButBlockScalingDependsOnOutputAndBlockCount) {
  auto metadata = CpuFlopsMetadata(MatMulNBitsRuntimeRoute::PackedQNBit, MatMulNBitsRuntimeComputeMode::SQNBitInt8);
  for (const auto mode : {MatMulNBitsRuntimeComputeMode::SQNBitInt8, MatMulNBitsRuntimeComputeMode::HQNBitInt8}) {
    metadata.compute_mode = mode;
    metadata.input_element_type = ONNX_NAMESPACE::TensorProto::FLOAT16;
    const auto result = contrib::EstimateMatMulNBitsCpuFlops(metadata);
    EXPECT_EQ(result.model, contrib::MatMulNBitsCpuFlopsModel::PackedInt8);
    ExpectBiasCount(result.core, 0);
    ExpectBiasCount(result.bias, 192);
    ExpectBiasCount(result.dequantization, 1920);
  }
  metadata.input_shape = {2, 3, 65};
  metadata.k = 65;
  ExpectBiasCount(contrib::EstimateMatMulNBitsCpuFlops(metadata).dequantization, 2880);
  metadata.block_size = 64;
  ExpectBiasCount(contrib::EstimateMatMulNBitsCpuFlops(metadata).dequantization, 1920);
}

TEST(MatMulNBitsCpuFlopsEstimateTest, MissingRouteAndLutKeepBiasIndependent) {
  auto metadata = CpuFlopsMetadata(MatMulNBitsRuntimeRoute::PackedLut, MatMulNBitsRuntimeComputeMode::Lut);
  auto result = contrib::EstimateMatMulNBitsCpuFlops(metadata);
  EXPECT_FALSE(result.model);
  ExpectBiasUnavailable(result.core, ComputeEstimateUnavailableReason::UnsupportedRoute);
  ExpectBiasUnavailable(result.dequantization, ComputeEstimateUnavailableReason::UnsupportedRoute);
  ExpectBiasCount(result.bias, 192);
  metadata.route.reset();
  result = contrib::EstimateMatMulNBitsCpuFlops(metadata);
  ExpectBiasUnavailable(result.core, ComputeEstimateUnavailableReason::MissingSemanticValue);
  metadata.route = MatMulNBitsRuntimeRoute::Unpacked;
  metadata.compute_mode.reset();
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCpuFlops(metadata).dequantization,
                        ComputeEstimateUnavailableReason::MissingSemanticValue);
}

TEST(MatMulNBitsCpuFlopsEstimateTest, UnknownAndInvalidGeometryAreNotZero) {
  auto metadata = CpuFlopsMetadata(MatMulNBitsRuntimeRoute::Unpacked, MatMulNBitsRuntimeComputeMode::Float32Gemm);
  metadata.input_shape = {0, -1, 64};
  auto result = contrib::EstimateMatMulNBitsCpuFlops(metadata);
  ExpectBiasUnavailable(result.core, ComputeEstimateUnavailableReason::UnknownShape);
  ExpectBiasUnavailable(result.dequantization, ComputeEstimateUnavailableReason::UnknownShape);
  metadata.input_shape = {2, 3, 63};
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCpuFlops(metadata).core,
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  metadata.input_shape.clear();
  metadata.bias_present = false;
  result = contrib::EstimateMatMulNBitsCpuFlops(metadata);
  ExpectBiasCount(result.bias, 0);
  ExpectBiasUnavailable(result.dequantization, ComputeEstimateUnavailableReason::InvalidMetadata);
}

TEST(MatMulNBitsCpuFlopsEstimateTest, InvalidQuantizationAndModeCombinationsAreExplicit) {
  auto metadata = CpuFlopsMetadata(MatMulNBitsRuntimeRoute::PackedQNBit, MatMulNBitsRuntimeComputeMode::HQNBitFp16);
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCpuFlops(metadata).core,
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  metadata.compute_mode = MatMulNBitsRuntimeComputeMode::Float32Gemm;
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCpuFlops(metadata).core,
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  metadata.compute_mode = MatMulNBitsRuntimeComputeMode::SQNBitFp32;
  for (uint32_t block_size : {0, 8, 24}) {
    metadata.block_size = block_size;
    ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCpuFlops(metadata).dequantization,
                          ComputeEstimateUnavailableReason::InvalidMetadata);
  }
  metadata.block_size = 512;
  metadata.bits = 2;
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCpuFlops(metadata).core,
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  metadata.bits = 3;
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCpuFlops(metadata).core,
                        ComputeEstimateUnavailableReason::InvalidMetadata);
}

TEST(MatMulNBitsCpuFlopsEstimateTest, UnsupportedTypeAndBf16ModeAreNotAdvertised) {
  auto metadata = CpuFlopsMetadata(MatMulNBitsRuntimeRoute::PackedQNBit, MatMulNBitsRuntimeComputeMode::SQNBitFp32);
  metadata.input_element_type = ONNX_NAMESPACE::TensorProto::BFLOAT16;
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCpuFlops(metadata).core,
                        ComputeEstimateUnavailableReason::UnsupportedRoute);
  metadata.input_element_type = 0;
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCpuFlops(metadata).core,
                        ComputeEstimateUnavailableReason::MissingSemanticValue);
  metadata.input_element_type = ONNX_NAMESPACE::TensorProto::FLOAT;
  metadata.compute_mode = MatMulNBitsRuntimeComputeMode::BHQNBitBf16;
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCpuFlops(metadata).core,
                        ComputeEstimateUnavailableReason::UnsupportedRoute);
}

TEST(MatMulNBitsCpuFlopsEstimateTest, EmptyOutputRequiresConsistentNonexecutionAndZeroReductionKeepsBias) {
  auto metadata = CpuFlopsMetadata(MatMulNBitsRuntimeRoute::EmptyOutput, MatMulNBitsRuntimeComputeMode::NotExecuted);
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCpuFlops(metadata).core,
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  metadata.input_shape = {2, 0, 64};
  auto result = contrib::EstimateMatMulNBitsCpuFlops(metadata);
  EXPECT_EQ(result.model, contrib::MatMulNBitsCpuFlopsModel::NotExecuted);
  ExpectBiasCount(result.core, 0);
  ExpectBiasCount(result.bias, 0);
  ExpectBiasCount(result.dequantization, 0);
  metadata = CpuFlopsMetadata(MatMulNBitsRuntimeRoute::PackedQNBit, MatMulNBitsRuntimeComputeMode::SQNBitInt8);
  metadata.input_shape = {2, 3, 0};
  metadata.k = 0;
  result = contrib::EstimateMatMulNBitsCpuFlops(metadata);
  ExpectBiasCount(result.core, 0);
  ExpectBiasCount(result.bias, 192);
  ExpectBiasCount(result.dequantization, 0);
}

TEST(MatMulNBitsCpuFlopsEstimateTest, ComponentOverflowIsIndependentAndCountsUseUint64) {
  const int64_t max = std::numeric_limits<int64_t>::max();
  auto metadata = CpuFlopsMetadata(MatMulNBitsRuntimeRoute::Unpacked, MatMulNBitsRuntimeComputeMode::Float32Gemm);
  metadata.input_shape = {2, 1};
  metadata.k = 1;
  metadata.n = max;
  auto result = contrib::EstimateMatMulNBitsCpuFlops(metadata);
  ExpectBiasUnavailable(result.core, ComputeEstimateUnavailableReason::Overflow);
  ExpectBiasCount(result.bias, std::numeric_limits<uint64_t>::max() - 1);
  ExpectBiasCount(result.dequantization, std::numeric_limits<uint64_t>::max() - 1);
  metadata.route = MatMulNBitsRuntimeRoute::PackedQNBit;
  metadata.compute_mode = MatMulNBitsRuntimeComputeMode::SQNBitFp32;
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCpuFlops(metadata).dequantization,
                        ComputeEstimateUnavailableReason::Overflow);
  metadata.compute_mode = MatMulNBitsRuntimeComputeMode::SQNBitInt8;
  result = contrib::EstimateMatMulNBitsCpuFlops(metadata);
  ExpectBiasCount(result.core, 0);
  ExpectBiasUnavailable(result.dequantization, ComputeEstimateUnavailableReason::Overflow);
  metadata.input_shape = {max, max, 0};
  metadata.k = 0;
  result = contrib::EstimateMatMulNBitsCpuFlops(metadata);
  ExpectBiasCount(result.core, 0);
  ExpectBiasCount(result.dequantization, 0);
  ExpectBiasUnavailable(result.bias, ComputeEstimateUnavailableReason::Overflow);
}

MatMulNBitsRuntimeMetadata CudaFlopsMetadata(MatMulNBitsRuntimeRoute route) {
  auto metadata = CpuFlopsMetadata(route, MatMulNBitsRuntimeComputeMode::CudaFloatingPoint);
  metadata.input_element_type = ONNX_NAMESPACE::TensorProto::FLOAT16;
  metadata.reconstruction_k = metadata.k;
  return metadata;
}

TEST(MatMulNBitsCudaFlopsEstimateTest, FusedAndMixedWeightFamiliesKeepTheSameFloatingConvention) {
  for (const auto route : {MatMulNBitsRuntimeRoute::CudaFused, MatMulNBitsRuntimeRoute::CudaFpAIntBGemv,
                           MatMulNBitsRuntimeRoute::CudaFpAIntBGemm, MatMulNBitsRuntimeRoute::CudaMixedFpAIntB}) {
    auto metadata = CudaFlopsMetadata(route);
    for (const auto type : {ONNX_NAMESPACE::TensorProto::FLOAT16, ONNX_NAMESPACE::TensorProto::BFLOAT16}) {
      metadata.input_element_type = type;
      for (uint32_t bits : {2, 4, 8}) {
        metadata.bits = bits;
        const auto result = contrib::EstimateMatMulNBitsCudaFlops(metadata);
        ASSERT_TRUE(result.model);
        ExpectBiasCount(result.core, 24576);
        ExpectBiasCount(result.bias, 192);
        ExpectBiasCount(result.dequantization, 4096);
      }
    }
  }
}

TEST(MatMulNBitsCudaFlopsEstimateTest, MaterializedPaddingDoesNotExpandTheCoreOrMultiplyByBatchCount) {
  auto metadata = CudaFlopsMetadata(MatMulNBitsRuntimeRoute::CudaDequantizeGemm);
  metadata.input_element_type = ONNX_NAMESPACE::TensorProto::FLOAT;
  metadata.input_shape = {2, 3, 65};
  metadata.k = 65;
  metadata.reconstruction_k = 96;
  metadata.batch_count = 17;
  auto result = contrib::EstimateMatMulNBitsCudaFlops(metadata);
  EXPECT_EQ(result.model, contrib::MatMulNBitsCudaFlopsModel::Materialized);
  ExpectBiasCount(result.core, 24960);
  ExpectBiasCount(result.bias, 192);
  ExpectBiasCount(result.dequantization, 6144);
  metadata.reconstruction_k = 65;
  ExpectBiasCount(contrib::EstimateMatMulNBitsCudaFlops(metadata).dequantization, 4160);
}

TEST(MatMulNBitsCudaFlopsEstimateTest, MissingAndInvalidProcessedGeometryKeepKnownCoreAndBias) {
  auto metadata = CudaFlopsMetadata(MatMulNBitsRuntimeRoute::CudaDequantizeGemm);
  metadata.reconstruction_k.reset();
  auto result = contrib::EstimateMatMulNBitsCudaFlops(metadata);
  ExpectBiasCount(result.core, 24576);
  ExpectBiasCount(result.bias, 192);
  ExpectBiasUnavailable(result.dequantization, ComputeEstimateUnavailableReason::MissingSemanticValue);
  for (int64_t k : {-1, 63, 65, 128}) {
    metadata.reconstruction_k = k;
    result = contrib::EstimateMatMulNBitsCudaFlops(metadata);
    ExpectBiasCount(result.core, 24576);
    ExpectBiasUnavailable(result.dequantization, ComputeEstimateUnavailableReason::InvalidMetadata);
  }
  metadata.route = MatMulNBitsRuntimeRoute::CudaFused;
  metadata.input_shape = {2, 3, 65};
  metadata.k = 65;
  metadata.reconstruction_k = 96;
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCudaFlops(metadata).dequantization,
                        ComputeEstimateUnavailableReason::InvalidMetadata);
}

TEST(MatMulNBitsCudaFlopsEstimateTest, MissingOrCpuRoutesAreNotMisclassifiedAsCuda) {
  auto metadata = CudaFlopsMetadata(MatMulNBitsRuntimeRoute::CudaFused);
  metadata.route.reset();
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCudaFlops(metadata).core,
                        ComputeEstimateUnavailableReason::MissingSemanticValue);
  metadata.route = MatMulNBitsRuntimeRoute::CudaFused;
  metadata.compute_mode.reset();
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCudaFlops(metadata).dequantization,
                        ComputeEstimateUnavailableReason::MissingSemanticValue);
  for (const auto route : {MatMulNBitsRuntimeRoute::Unpacked, MatMulNBitsRuntimeRoute::PackedQNBit,
                           MatMulNBitsRuntimeRoute::PackedLut}) {
    metadata.route = route;
    metadata.compute_mode = MatMulNBitsRuntimeComputeMode::CudaFloatingPoint;
    const auto result = contrib::EstimateMatMulNBitsCudaFlops(metadata);
    ExpectBiasUnavailable(result.core, ComputeEstimateUnavailableReason::UnsupportedRoute);
    ExpectBiasCount(result.bias, 192);
  }
}

TEST(MatMulNBitsCudaFlopsEstimateTest, UnknownInvalidAndInconsistentMetadataAreExplicit) {
  auto metadata = CudaFlopsMetadata(MatMulNBitsRuntimeRoute::CudaFused);
  metadata.input_shape = {0, -1, 64};
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCudaFlops(metadata).core,
                        ComputeEstimateUnavailableReason::UnknownShape);
  metadata.input_shape = {2, 3, 63};
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCudaFlops(metadata).core,
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  metadata.input_shape = {2, 3, 64};
  metadata.compute_mode = MatMulNBitsRuntimeComputeMode::SQNBitInt8;
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCudaFlops(metadata).core,
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  metadata.compute_mode = MatMulNBitsRuntimeComputeMode::CudaFloatingPoint;
  metadata.block_size = 24;
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCudaFlops(metadata).core,
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  metadata.block_size = 32;
  metadata.bits = 3;
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCudaFlops(metadata).core,
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  metadata.bits = 4;
  metadata.input_element_type = ONNX_NAMESPACE::TensorProto::INT8;
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCudaFlops(metadata).core,
                        ComputeEstimateUnavailableReason::UnsupportedRoute);
  metadata.input_element_type = ONNX_NAMESPACE::TensorProto::FLOAT;
  metadata.route = MatMulNBitsRuntimeRoute::CudaFpAIntBGemm;
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCudaFlops(metadata).core,
                        ComputeEstimateUnavailableReason::InvalidMetadata);
}

TEST(MatMulNBitsCudaFlopsEstimateTest, EmptyRankOneAndZeroReductionPreserveBiasSemantics) {
  auto metadata = CudaFlopsMetadata(MatMulNBitsRuntimeRoute::EmptyOutput);
  metadata.compute_mode = MatMulNBitsRuntimeComputeMode::NotExecuted;
  metadata.reconstruction_k.reset();
  ExpectBiasUnavailable(contrib::EstimateMatMulNBitsCudaFlops(metadata).core,
                        ComputeEstimateUnavailableReason::InvalidMetadata);
  metadata.input_shape = {2, 0, 64};
  auto result = contrib::EstimateMatMulNBitsCudaFlops(metadata);
  EXPECT_EQ(result.model, contrib::MatMulNBitsCudaFlopsModel::NotExecuted);
  ExpectBiasCount(result.core, 0);
  ExpectBiasCount(result.bias, 0);
  ExpectBiasCount(result.dequantization, 0);
  metadata = CudaFlopsMetadata(MatMulNBitsRuntimeRoute::CudaFused);
  metadata.input_shape = {64};
  metadata.bias_present = false;
  result = contrib::EstimateMatMulNBitsCudaFlops(metadata);
  ExpectBiasCount(result.core, 4096);
  ExpectBiasCount(result.bias, 0);
  ExpectBiasCount(result.dequantization, 4096);
  metadata.input_shape = {2, 3, 0};
  metadata.k = 0;
  metadata.reconstruction_k = 0;
  metadata.bias_present = true;
  result = contrib::EstimateMatMulNBitsCudaFlops(metadata);
  ExpectBiasCount(result.core, 0);
  ExpectBiasCount(result.bias, 192);
  ExpectBiasCount(result.dequantization, 0);
}

TEST(MatMulNBitsCudaFlopsEstimateTest, CoreAndDequantizationOverflowRemainIndependent) {
  auto metadata = CudaFlopsMetadata(MatMulNBitsRuntimeRoute::CudaFused);
  metadata.input_shape = {2, 1};
  metadata.k = 1;
  metadata.reconstruction_k = 1;
  metadata.n = std::numeric_limits<int64_t>::max();
  auto result = contrib::EstimateMatMulNBitsCudaFlops(metadata);
  ExpectBiasUnavailable(result.core, ComputeEstimateUnavailableReason::Overflow);
  ExpectBiasCount(result.bias, std::numeric_limits<uint64_t>::max() - 1);
  ExpectBiasCount(result.dequantization, std::numeric_limits<uint64_t>::max() - 1);
  metadata.input_shape = {1};
  metadata.route = MatMulNBitsRuntimeRoute::CudaDequantizeGemm;
  metadata.reconstruction_k = 32;
  result = contrib::EstimateMatMulNBitsCudaFlops(metadata);
  ExpectBiasCount(result.core, std::numeric_limits<uint64_t>::max() - 1);
  ExpectBiasUnavailable(result.dequantization, ComputeEstimateUnavailableReason::Overflow);
}

class MatMulNBitsComputeEstimateGraphTest : public ::testing::Test {
 protected:
  MatMulNBitsComputeEstimateGraphTest()
      : model_("compute_estimate", false, ModelMetaData(), PathString(),
               IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 17}, {kMSDomain, 1}}, {},
               DefaultLoggingManager().DefaultLogger()),
        builder_(model_.MainGraph()) {}

  template <typename T = float>
  Node& AddMatMul(std::optional<std::vector<int64_t>> shape = std::vector<int64_t>{2, 3, 64},
                  int64_t bits = 4, bool bias = false, bool explicit_absent_bias = false) {
    NodeArg* a = builder_.MakeInput<T>(shape);
    NodeArg* b = builder_.MakeInput<uint8_t>(std::vector<int64_t>{32, 2, 32 * bits / 8});
    NodeArg* scales = builder_.MakeInput<T>(std::vector<int64_t>{32, 2});
    std::vector<NodeArg*> inputs{a, b, scales};
    if (bias || explicit_absent_bias) {
      auto& missing = builder_.graph_.GetOrCreateNodeArg("", nullptr);
      inputs.push_back(&missing);
      inputs.push_back(&missing);
      inputs.push_back(bias ? builder_.MakeInput<T>(std::vector<int64_t>{32}) : &missing);
    }
    auto* output = builder_.MakeOutput<T>(std::nullopt);
    auto& node = builder_.AddNode("MatMulNBits", inputs, {output}, kMSDomain);
    node.AddAttribute("K", int64_t{64});
    node.AddAttribute("N", int64_t{32});
    node.AddAttribute("block_size", int64_t{32});
    node.AddAttribute("bits", bits);
    builder_.SetGraphOutputs();
    return node;
  }

  Model model_;
  ModelTestBuilder builder_;
};

TEST_F(MatMulNBitsComputeEstimateGraphTest, ResolvedGraph) {
  auto& node = AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectCount(EstimateMatMulNBitsCompute(node), 24576);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, BiasPresentAndCoreUnchanged) {
  auto& node = AddMatMul(std::vector<int64_t>{2, 3, 64}, 4, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectBiasCount(EstimateMatMulNBitsBias(node), 192);
  ExpectCount(EstimateMatMulNBitsCompute(node), 24576);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, BiasAbsentWithUnknownRank) {
  auto& node = AddMatMul(std::nullopt);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectBiasCount(EstimateMatMulNBitsBias(node), 0);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, ExplicitlyOmittedBiasWithUnknownRank) {
  auto& node = AddMatMul(std::nullopt, 4, false, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectBiasCount(EstimateMatMulNBitsBias(node), 0);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, BiasPresentWithUnknownRank) {
  auto& node = AddMatMul(std::nullopt, 4, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectBiasUnavailable(EstimateMatMulNBitsBias(node), ComputeEstimateUnavailableReason::UnknownShape);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, BiasPresentWithSymbolicShape) {
  auto& node = AddMatMul(std::vector<int64_t>{-1, 64}, 4, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectBiasUnavailable(EstimateMatMulNBitsBias(node), ComputeEstimateUnavailableReason::UnknownShape);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, BiasHalfPrecisionTwoBit) {
  auto& node = AddMatMul<MLFloat16>(std::vector<int64_t>{2, 3, 64}, 2, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectBiasCount(EstimateMatMulNBitsBias(node), 192);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, RankOneGraph) {
  auto& node = AddMatMul(std::vector<int64_t>{64});
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectCount(EstimateMatMulNBitsCompute(node), 4096);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, TwoBitWithBias) {
  auto& node = AddMatMul(std::vector<int64_t>{2, 3, 64}, 2, true);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectCount(EstimateMatMulNBitsCompute(node), 24576);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, EightBitHalfPrecision) {
  auto& node = AddMatMul<MLFloat16>(std::vector<int64_t>{2, 3, 64}, 8);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectCount(EstimateMatMulNBitsCompute(node), 24576);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, BFloat16) {
  auto& node = AddMatMul<BFloat16>();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectCount(EstimateMatMulNBitsCompute(node), 24576);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, UnknownRank) {
  auto& node = AddMatMul(std::nullopt);
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::UnknownShape);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, PartiallyKnownShape) {
  auto& node = AddMatMul(std::vector<int64_t>{-1, 64});
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::UnknownShape);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, MissingAttribute) {
  auto& node = AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  node.ClearAttribute("K");
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::InvalidMetadata);
  ExpectBiasUnavailable(EstimateMatMulNBitsBias(node), ComputeEstimateUnavailableReason::InvalidMetadata);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, WrongAttributeType) {
  auto& node = AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  node.AddAttribute("N", std::string("32"));
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::InvalidMetadata);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, UnresolvedVersion) {
  auto& node = AddMatMul();
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::UnsupportedVersion);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, UnsupportedVersion) {
  auto& node = AddMatMul();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  node.SetSinceVersion(2);
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::UnsupportedVersion);
  ExpectBiasUnavailable(EstimateMatMulNBitsBias(node), ComputeEstimateUnavailableReason::UnsupportedVersion);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, EmptyGraphWorkload) {
  auto& node = AddMatMul(std::vector<int64_t>{0, 64});
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectCount(EstimateMatMulNBitsCompute(node), 0);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, UnsupportedDomain) {
  auto* input = builder_.MakeInput<float>(std::vector<int64_t>{2, 64});
  auto* output = builder_.MakeOutput<float>(std::nullopt);
  auto& node = builder_.AddNode("MatMulNBits", {input}, {output}, "other.domain");
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::UnsupportedOperator);
  ExpectBiasUnavailable(EstimateMatMulNBitsBias(node), ComputeEstimateUnavailableReason::UnsupportedOperator);
}

TEST_F(MatMulNBitsComputeEstimateGraphTest, UnsupportedOperator) {
  auto* input = builder_.MakeInput<float>(std::vector<int64_t>{2, 3});
  auto* output = builder_.MakeOutput<float>(std::nullopt);
  auto& node = builder_.AddNode("Identity", {input}, {output});
  builder_.SetGraphOutputs();
  ASSERT_STATUS_OK(model_.MainGraph().Resolve());
  ExpectUnavailable(EstimateMatMulNBitsCompute(node), ComputeEstimateUnavailableReason::UnsupportedOperator);
}
#endif

}  // namespace
}  // namespace test
}  // namespace onnxruntime
