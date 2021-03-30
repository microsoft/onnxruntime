// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * IMPORTANT NOTE AT THE TOP OF THE FILE
 *
 * This file contains tests which verify expected kernel def hashes.
 * It is important for these to remain stable so that ORT format models are
 * backward compatible.
 *
 * If you are seeing a test failure from one of these tests, it is likely that
 * some kernel definition changed in a way that updated its hash value.
 * This is what we want to catch! Please update the kernel definition.
 * If adding more supported types to an existing kernel definition, consider
 * using KernelDefBuilder::FixedTypeConstraintForHash().
 *
 * For example:
 * Say we have a kernel definition like this, which supports types int and
 * double:
 *     KernelDefBuilder{}
 *         .TypeConstraint(
 *             "T", BuildKernelDefConstraints<int, double>())
 * If we want to update the kernel definition to add support for float, we can
 * change it to something like this:
 *     KernelDefBuilder{}
 *         .TypeConstraint(
 *             "T", BuildKernelDefConstraints<int, double, float>())
 *         .FixedTypeConstraintForHash(
 *             "T", BuildKernelDefConstraints<int, double>())
 * In the updated kernel definition, the original types are specified with
 * FixedTypeConstraintForHash().
 *
 * New kernel definitions should not use FixedTypeConstraintForHash().
 * It is a way to keep the hash stable as kernel definitions change.
 *
 * It is also possible that you have added a new kernel definition and are
 * seeing a message from one of these tests about updating the expected data.
 * Please do that if appropriate.
 *
 * In the unlikely event that we need to make a change to the kernel def
 * hashing that breaks backward compatibility, the expected values may need to
 * be updated.
 * The expected value files are in this directory:
 *     onnxruntime/test/testdata/kernel_def_hashes
 */

#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <iostream>

#include "gtest/gtest.h"

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 28020)
#endif
#include "nlohmann/json.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif

#include "asserts.h"
#include "core/common/common.h"
#include "core/common/path_string.h"
#include "core/framework/kernel_registry.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"

using json = nlohmann::json;

namespace onnxruntime {
namespace test {

namespace {
static constexpr const char* kRunKernelDefHashTestOrFailEnvVar =
    "ORT_TEST_RUN_KERNEL_DEF_HASH_TEST_OR_FAIL";

std::string DumpKernelDefHashes(const onnxruntime::KernelDefHashes& kernel_def_hashes) {
  const json j(kernel_def_hashes);
  return j.dump(/* indent */ 4);
}

KernelDefHashes ParseKernelDefHashes(std::istream& in) {
  KernelDefHashes kernel_def_hashes{};
  const json j = json::parse(in);
  j.get_to<onnxruntime::KernelDefHashes>(kernel_def_hashes);
  return kernel_def_hashes;
}

KernelDefHashes ReadKernelDefHashesFromFile(const PathString& path) {
  std::ifstream in{path};
  ORT_ENFORCE(in, "Failed to open file: ", ToMBString(path));
  const auto kernel_def_hashes = ParseKernelDefHashes(in);
  return kernel_def_hashes;
}

void CheckKernelDefHashes(const KernelDefHashes& actual, const KernelDefHashes& expected) {
  ASSERT_TRUE(std::is_sorted(actual.begin(), actual.end()));
  ASSERT_TRUE(std::is_sorted(expected.begin(), expected.end()));

  constexpr const char* kNoteReference = "Note: Please read the note at the top of this file: " __FILE__;

  KernelDefHashes expected_minus_actual{};
  std::set_difference(expected.begin(), expected.end(), actual.begin(), actual.end(),
                      std::back_inserter(expected_minus_actual));
  EXPECT_TRUE(expected_minus_actual.empty())
      << "Some expected kernel def hashes were not found.\n"
      << kNoteReference << "\n"
      << DumpKernelDefHashes(expected_minus_actual);

  KernelDefHashes actual_minus_expected{};
  std::set_difference(actual.begin(), actual.end(), expected.begin(), expected.end(),
                      std::back_inserter(actual_minus_expected));
  EXPECT_TRUE(actual_minus_expected.empty())
      << "Unexpected kernel def hashes were found, please update the expected values as needed "
         "(see the output below).\n"
      << kNoteReference << "\n"
      << DumpKernelDefHashes(actual_minus_expected);
}
}  // namespace

TEST(KernelDefHashTest, DISABLED_PrintCpuKernelDefHashes) {
  KernelRegistry kernel_registry{};
  ASSERT_STATUS_OK(RegisterCPUKernels(kernel_registry));
  const auto cpu_kernel_def_hashes = kernel_registry.ExportKernelDefHashes();
  std::cout << DumpKernelDefHashes(cpu_kernel_def_hashes) << "\n";
}

TEST(KernelDefHashTest, ExpectedCpuKernelDefHashes) {
  // this test should only run in a build containing exactly the set of CPU
  // kernels that can be used in ORT format models
  const bool is_enabled = []() {
#if !defined(DISABLE_CONTRIB_OPS) &&       \
    !defined(DISABLE_ML_OPS) &&            \
    !defined(ML_FEATURIZERS) &&            \
    !defined(BUILD_MS_EXPERIMENTAL_OPS) && \
    !defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_OPS)
    return MlasNchwcGetBlockSize() > 1;
#else
    return false;
#endif
  }();

  if (!is_enabled) {
    std::cout << "This build might not have the expected CPU kernels, skipping test...\n";
    if (ParseEnvironmentVariableWithDefault<bool>(kRunKernelDefHashTestOrFailEnvVar, false)) {
      FAIL() << "Skipped test is treated as a failure.";
    }
    return;
  }

  KernelRegistry kernel_registry{};
  ASSERT_STATUS_OK(RegisterCPUKernels(kernel_registry));
  auto cpu_kernel_def_hashes = kernel_registry.ExportKernelDefHashes();
  const auto expected_cpu_kernel_def_hashes =
      ReadKernelDefHashesFromFile(ORT_TSTR("testdata/kernel_def_hashes/cpu.json"));
  CheckKernelDefHashes(cpu_kernel_def_hashes, expected_cpu_kernel_def_hashes);
}

}  // namespace test
}  // namespace onnxruntime
