// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/providers/cuda/math/einsum_utils/einsum_fast_path.h"

namespace onnxruntime {
namespace cuda {
namespace test {
namespace {

struct PlannedEinsum {
  EinsumFastPathKind kind;
  EinsumFastPathPlan plan;
};

PlannedEinsum Plan(std::string_view equation,
                   std::initializer_list<TensorShapeVector> input_shapes) {
  AllocatorPtr allocator = std::make_shared<CPUAllocator>();
  uint8_t dummy_data = 0;
  std::vector<std::unique_ptr<Tensor>> owned_inputs;
  std::vector<const Tensor*> inputs;
  owned_inputs.reserve(input_shapes.size());
  inputs.reserve(input_shapes.size());
  for (const auto& shape : input_shapes) {
    owned_inputs.push_back(
        std::make_unique<Tensor>(
            DataTypeImpl::GetType<float>(), TensorShape(shape), &dummy_data, allocator->Info()));
    inputs.push_back(owned_inputs.back().get());
  }

  EinsumEquationPreprocessor equation_preprocessor{std::string(equation)};
  EinsumComputePreprocessor compute_preprocessor(equation_preprocessor, inputs, allocator, nullptr);
  ORT_THROW_IF_ERROR(compute_preprocessor.Analyze());

  EinsumFastPathPlan plan;
  ORT_THROW_IF_ERROR(CreateEinsumFastPathPlan(compute_preprocessor, plan));
  return {plan.kind, std::move(plan)};
}

TEST(EinsumFastPathPlannerTest, ClassifiesSingleInputPaths) {
  EXPECT_EQ(Plan("ij->ij", {{2, 3}}).kind, EinsumFastPathKind::Copy);
  EXPECT_EQ(Plan("...ji->...ij", {{2, 3, 4}}).kind, EinsumFastPathKind::Transpose);
  EXPECT_EQ(Plan("abc->ac", {{2, 3, 4}}).kind, EinsumFastPathKind::ReduceSum);
  EXPECT_EQ(Plan("iij->ij", {{3, 3, 4}}).kind, EinsumFastPathKind::Diagonal);
  EXPECT_EQ(Plan("...ii->...", {{2, 4, 4}}).kind, EinsumFastPathKind::Trace);
  EXPECT_EQ(Plan("iii->i", {{5, 5, 5}}).kind, EinsumFastPathKind::Diagonal);
}

TEST(EinsumFastPathPlannerTest, ClassifiesTwoInputPaths) {
  EXPECT_EQ(Plan("ij,ij->ij", {{3, 4}, {3, 4}}).kind, EinsumFastPathKind::Multiply);
  EXPECT_EQ(Plan("i,j->ji", {{3}, {4}}).kind, EinsumFastPathKind::Multiply);
  EXPECT_EQ(Plan("bij,bj->bij", {{2, 3, 4}, {2, 4}}).kind, EinsumFastPathKind::Multiply);
  EXPECT_EQ(Plan("ij,jk->ik", {{3, 4}, {4, 5}}).kind, EinsumFastPathKind::MatMul);
  EXPECT_EQ(Plan("bhmd,bhnd->bhmn", {{2, 8, 16, 64}, {2, 8, 32, 64}}).kind,
            EinsumFastPathKind::MatMul);
  EXPECT_EQ(Plan("abc,cde->abde", {{2, 3, 4}, {4, 5, 6}}).kind,
            EinsumFastPathKind::MatMul);
  EXPECT_EQ(Plan("i,i->", {{64}, {64}}).kind, EinsumFastPathKind::MatMul);
}

TEST(EinsumFastPathPlannerTest, ConservativelyRejectsUnsupportedLayouts) {
  EXPECT_EQ(Plan("ik,kj->ij", {{3, 1}, {4, 5}}).kind, EinsumFastPathKind::None);
  EXPECT_EQ(Plan("ij,jk->ki", {{3, 4}, {4, 5}}).kind, EinsumFastPathKind::None);
  EXPECT_EQ(Plan("bthd,bshd->bhts", {{2, 16, 8, 64}, {2, 32, 8, 64}}).kind,
            EinsumFastPathKind::None);
  EXPECT_EQ(Plan("ij,jk,kl->il", {{2, 3}, {3, 4}, {4, 5}}).kind,
            EinsumFastPathKind::None);
  EXPECT_EQ(Plan("iij,jk->ik", {{3, 3, 4}, {4, 5}}).kind,
            EinsumFastPathKind::None);
  EXPECT_EQ(Plan("i,j->ij", {{65537}, {65536}}).kind,
            EinsumFastPathKind::None);
}

}  // namespace
}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
