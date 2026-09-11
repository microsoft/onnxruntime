// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the SafeInt-guarded working-buffer/size products in MlasConvPrepare.
// These cases construct attacker-controlled shape/count combinations whose
// products overflow size_t and verify that MlasConvPrepare throws
// onnxruntime::OnnxRuntimeException (raised by SafeInt's overflow handler)
// rather than silently producing an under-sized working buffer or wrapped
// tensor size.

#include "gtest/gtest.h"

#include "mlas.h"

// These tests rely on the ORT-internal SafeInt overflow handler
// (onnxruntime::OnnxRuntimeException) and on ORT exception support, so they
// are skipped in standalone MLAS builds and in no-exception configurations.
#if !defined(ORT_NO_EXCEPTIONS) && !defined(BUILD_MLAS_NO_ONNXRUNTIME)

#include "core/common/exceptions.h"

#include <cstdint>
#include <cstddef>

namespace {

// A value whose square overflows size_t on every supported pointer width
// (64-bit: 2^33, 32-bit: 2^17). Each individual value still fits in size_t
// and in int64_t shape entries, so the loop accumulators reach the SafeInt
// multiplication before the per-tensor value itself is invalid.
constexpr size_t kHalfShift = static_cast<size_t>(1) << ((sizeof(size_t) * 8 / 2) + 1);

// Identity activation, shared across all tests because MlasConvPrepare
// dereferences but does not invoke it.
MLAS_ACTIVATION MakeIdentityActivation() {
  MLAS_ACTIVATION activation{};
  activation.ActivationKind = MlasIdentityActivation;
  return activation;
}

// A baseline-valid 3D conv setup. Callers mutate one field per test to drive
// the specific SafeInt-guarded product into overflow. 3D is chosen so the
// KleidiAI MlasConvPrepareOverride (which only handles 2D) is bypassed on
// every platform.
struct ConvPrepareInputs {
  size_t Dimensions = 3;
  size_t BatchCount = 1;
  size_t GroupCount = 1;
  size_t InputChannels = 1;
  size_t FilterCount = 1;
  int64_t InputShape[3] = {1, 1, 1};
  int64_t KernelShape[3] = {1, 1, 1};
  int64_t DilationShape[3] = {1, 1, 1};
  // Stride[1] = 2 (rather than all-ones) keeps execution out of the pointwise
  // "direct GEMM" early-return so the SafeInt-guarded code paths are reached.
  int64_t StrideShape[3] = {1, 1, 2};
  int64_t Padding[6] = {0, 0, 0, 0, 0, 0};
  int64_t OutputShape[3] = {1, 1, 1};
  float Beta = 0.0f;
  bool ChannelsLast = false;
};

void RunConvPrepare(const ConvPrepareInputs& in, size_t* working_buffer_size_out = nullptr) {
  MLAS_CONV_PARAMETERS parameters{};
  MLAS_ACTIVATION activation = MakeIdentityActivation();
  size_t working_buffer_size = 0;

  MlasConvPrepare(&parameters,
                  in.Dimensions,
                  in.BatchCount,
                  in.GroupCount,
                  in.InputChannels,
                  in.InputShape,
                  in.KernelShape,
                  in.DilationShape,
                  in.Padding,
                  in.StrideShape,
                  in.OutputShape,
                  in.FilterCount,
                  &activation,
                  &working_buffer_size,
                  in.ChannelsLast,
                  in.Beta,
                  /*ThreadPool=*/nullptr);

  if (working_buffer_size_out != nullptr) {
    *working_buffer_size_out = working_buffer_size;
  }
}

}  // namespace

// Sanity check: the SafeInt instrumentation must not regress the happy path.
TEST(MlasConvPrepareSafeIntTest, SmallShapeDoesNotThrow) {
  ConvPrepareInputs in;
  in.InputShape[0] = 4;
  in.InputShape[1] = 4;
  in.InputShape[2] = 4;
  in.KernelShape[0] = 1;
  in.KernelShape[1] = 1;
  in.KernelShape[2] = 1;
  in.OutputShape[0] = 2;
  in.OutputShape[1] = 2;
  in.OutputShape[2] = 2;
  in.InputChannels = 2;
  in.FilterCount = 2;

  size_t working_buffer_size = 0;
  EXPECT_NO_THROW(RunConvPrepare(in, &working_buffer_size));
}

// SafeInputSize *= Parameters->InputShape[dim] must trip on overflow.
TEST(MlasConvPrepareSafeIntTest, InputSizeProductOverflowsThrows) {
  ConvPrepareInputs in;
  in.InputShape[0] = static_cast<int64_t>(kHalfShift);
  in.InputShape[1] = static_cast<int64_t>(kHalfShift);
  in.InputShape[2] = 1;

  EXPECT_THROW(RunConvPrepare(in), onnxruntime::OnnxRuntimeException);
}

// SafeOutputSize *= Parameters->OutputShape[dim] must trip on overflow.
TEST(MlasConvPrepareSafeIntTest, OutputSizeProductOverflowsThrows) {
  ConvPrepareInputs in;
  in.OutputShape[0] = static_cast<int64_t>(kHalfShift);
  in.OutputShape[1] = static_cast<int64_t>(kHalfShift);
  in.OutputShape[2] = 1;

  EXPECT_THROW(RunConvPrepare(in), onnxruntime::OnnxRuntimeException);
}

// SafeK is seeded with InputChannels and then folded against the kernel shape;
// growing the kernel dimensions until the running product overflows must throw.
TEST(MlasConvPrepareSafeIntTest, KernelProductOverflowsThrows) {
  ConvPrepareInputs in;
  in.InputChannels = kHalfShift;
  in.KernelShape[0] = static_cast<int64_t>(kHalfShift);
  in.KernelShape[1] = 1;
  in.KernelShape[2] = 1;

  EXPECT_THROW(RunConvPrepare(in), onnxruntime::OnnxRuntimeException);
}

// In the ExpandThenGemm path *WorkingBufferSize = SafeInt<size_t>(OutputSize) * K.
// Individual values fit, but the cross-tensor product overflows and must throw
// rather than under-sizing the im2col buffer.
TEST(MlasConvPrepareSafeIntTest, ExpandThenGemmWorkingBufferOverflowsThrows) {
  ConvPrepareInputs in;
  // OutputSize = kHalfShift (only one non-unit output dim so the running
  // SafeOutputSize accumulation stays in range).
  in.OutputShape[0] = static_cast<int64_t>(kHalfShift);
  in.OutputShape[1] = 1;
  in.OutputShape[2] = 1;
  // K = InputChannels * prod(KernelShape) = kHalfShift, again in range.
  in.InputChannels = kHalfShift;
  in.KernelShape[0] = 1;
  in.KernelShape[1] = 1;
  in.KernelShape[2] = 1;
  // FilterCount > OutputSize selects MlasConvAlgorithmExpandThenGemm.
  in.FilterCount = kHalfShift + 1;
  // Non-trivial stride keeps us out of the pointwise GemmDirect early return
  // even though AllPaddingIsZero remains true.
  in.StrideShape[0] = 1;
  in.StrideShape[1] = 1;
  in.StrideShape[2] = 2;

  EXPECT_THROW(RunConvPrepare(in), onnxruntime::OnnxRuntimeException);
}

// The MlasConvAlgorithmExpandThenGemmSegmented path multiplies BatchCount and
// GroupCount inside SafeInt before clamping TargetThreadCount; that product
// must throw on overflow rather than wrapping silently.
TEST(MlasConvPrepareSafeIntTest, BatchGroupProductOverflowsThrows) {
  ConvPrepareInputs in;
  in.BatchCount = kHalfShift;
  in.GroupCount = kHalfShift;
  // Keep every per-tensor accumulator small so the only SafeInt product that
  // can fail is the BatchCount * GroupCount one inside MlasConvPrepare.
  in.InputChannels = 1;
  in.FilterCount = 1;  // FilterCount <= OutputSize -> reaches the segmented branch.
  in.InputShape[0] = 1;
  in.InputShape[1] = 1;
  in.InputShape[2] = 1;
  in.KernelShape[0] = 1;
  in.KernelShape[1] = 1;
  in.KernelShape[2] = 1;
  in.OutputShape[0] = 1;
  in.OutputShape[1] = 1;
  in.OutputShape[2] = 1;
  // Non-trivial stride keeps us out of the pointwise GemmDirect early return.
  in.StrideShape[0] = 1;
  in.StrideShape[1] = 1;
  in.StrideShape[2] = 2;

  EXPECT_THROW(RunConvPrepare(in), onnxruntime::OnnxRuntimeException);
}

#endif  // !defined(ORT_NO_EXCEPTIONS) && !defined(BUILD_MLAS_NO_ONNXRUNTIME)
