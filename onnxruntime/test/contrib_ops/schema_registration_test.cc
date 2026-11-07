// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace contrib {

using namespace ONNX_NAMESPACE;

#ifndef NDEBUG
// Capture the counter around each macro in the same translation unit so the
// regression is covered even when ONNX static registration is disabled.
static const size_t count_before_ms_schema = DbgOperatorSetTracker::Instance().GetCount();
#endif

ONNX_MS_OPERATOR_SET_SCHEMA(SchemaCounterMicrosoftTest, 1,
                            OpSchema()
                                .Input(0, "X", "Input", "tensor(float)")
                                .Output(0, "Y", "Output", "tensor(float)"));

#ifndef NDEBUG
static const size_t count_after_ms_schema = DbgOperatorSetTracker::Instance().GetCount();
#endif

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(SchemaCounterContribTest, 1,
                                 OpSchema()
                                     .Input(0, "X", "Input", "tensor(float)")
                                     .Output(0, "Y", "Output", "tensor(float)"));

#ifndef NDEBUG
static const size_t count_after_contrib_schema = DbgOperatorSetTracker::Instance().GetCount();
#endif

TEST(ContribSchemaRegistrationTest, MicrosoftSchemaDoesNotCountAsOnnxStaticSchema) {
#ifndef NDEBUG
  EXPECT_EQ(count_before_ms_schema, count_after_ms_schema);
#endif
  const auto schema = GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SchemaCounterMicrosoftTest)>();
  EXPECT_EQ(schema.Name(), "SchemaCounterMicrosoftTest");
  EXPECT_EQ(schema.domain(), kMSDomain);
  EXPECT_EQ(schema.SinceVersion(), 1);
}

TEST(ContribSchemaRegistrationTest, ContribSchemaDoesNotCountAsOnnxStaticSchema) {
#ifndef NDEBUG
  EXPECT_EQ(count_after_ms_schema, count_after_contrib_schema);
#endif
  const auto schema = GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, SchemaCounterContribTest)>();
  EXPECT_EQ(schema.Name(), "SchemaCounterContribTest");
  EXPECT_EQ(schema.domain(), kOnnxDomain);
  EXPECT_EQ(schema.SinceVersion(), 1);
}

}  // namespace contrib
}  // namespace onnxruntime
