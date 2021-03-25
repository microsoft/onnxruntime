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

#include "core/common/common.h"
#include "core/common/path_string.h"
#include "core/framework/kernel_registry.h"
#include "core/mlas/inc/mlas.h"
#include "default_providers.h"

using json = nlohmann::json;

namespace onnxruntime {
namespace test {

namespace {
std::ostream& operator<<(std::ostream& out, const KernelDefHashes& kernel_def_hashes) {
  const json j(kernel_def_hashes);
  out << j.dump(/* indent */ 4);
  return out;
}

std::istream& operator>>(std::istream& in, KernelDefHashes& kernel_def_hashes) {
  const json j = json::parse(in);
  j.get_to<KernelDefHashes>(kernel_def_hashes);
  return in;
}

void AppendKernelDefHashesFromFile(const PathString& path, KernelDefHashes& kernel_def_hashes) {
  KernelDefHashes hashes_to_append{};
  {
    std::ifstream in{path};
    in >> hashes_to_append;
    ASSERT_TRUE(in) << "Failed to get kernel def hashes from file: " << path;
  }
  kernel_def_hashes.insert(kernel_def_hashes.end(), hashes_to_append.begin(), hashes_to_append.end());
}

void CheckKernelDefHashes(const KernelDefHashes& actual, const KernelDefHashes& expected) {
  ASSERT_TRUE(std::is_sorted(actual.begin(), actual.end()));
  ASSERT_TRUE(std::is_sorted(expected.begin(), expected.end()));

  constexpr const char* kNoteReference = "Note: Please read the note at the top of this file: " __FILE__;

  KernelDefHashes expected_minus_actual{};
  std::set_difference(expected.begin(), expected.end(), actual.begin(), actual.end(),
                      std::back_inserter(expected_minus_actual));
  EXPECT_EQ(expected_minus_actual, KernelDefHashes{})
      << "Some expected kernel def hashes were not found.\n"
      << kNoteReference;

  KernelDefHashes actual_minus_expected{};
  std::set_difference(actual.begin(), actual.end(), expected.begin(), expected.end(),
                      std::back_inserter(actual_minus_expected));
  if (!actual_minus_expected.empty()) {
    std::cerr << "Extra actual kernel def hashes were found, please update the expected values as needed "
                 "(see the output below).\n"
              << kNoteReference << "\n"
              << actual_minus_expected << "\n";
  }
}

const KernelDefHashes& GetExpectedCpuKernelDefHashes() {
  static const KernelDefHashes expected_cpu_kernel_def_hashes = []() {
    KernelDefHashes result{};
    AppendKernelDefHashesFromFile(ORT_TSTR("testdata/kernel_def_hashes/onnx.cpu.json"), result);
#ifndef DISABLE_ML_OPS
    AppendKernelDefHashesFromFile(ORT_TSTR("testdata/kernel_def_hashes/onnx.ml.cpu.json"), result);
#endif  // !DISABLE_ML_OPS
#ifndef DISABLE_CONTRIB_OPS
    AppendKernelDefHashesFromFile(ORT_TSTR("testdata/kernel_def_hashes/contrib.cpu.json"), result);
    if (MlasNchwcGetBlockSize() > 1) {
      AppendKernelDefHashesFromFile(ORT_TSTR("testdata/kernel_def_hashes/contrib.nchwc.cpu.json"), result);
    }
#endif  // !DISABLE_CONTRIB_OPS
#ifdef ENABLE_TRAINING_OPS
    AppendKernelDefHashesFromFile(ORT_TSTR("testdata/kernel_def_hashes/training_ops.cpu.json"), result);
#endif  // ENABLE_TRAINING_OPS
    std::sort(result.begin(), result.end());

    return result;
  }();

  return expected_cpu_kernel_def_hashes;
}
}  // namespace

TEST(KernelDefHashTest, DISABLED_PrintCpuKernelDefHashes) {
  auto cpu_ep = DefaultCpuExecutionProvider();
  auto kernel_registry = cpu_ep->GetKernelRegistry();
  const auto cpu_kernel_def_hashes = kernel_registry->ExportKernelDefHashes();
  std::cout << cpu_kernel_def_hashes << "\n";
}

TEST(KernelDefHashTest, ExpectedCpuKernelDefHashes) {
  auto cpu_ep = DefaultCpuExecutionProvider();
  auto kernel_registry = cpu_ep->GetKernelRegistry();
  auto cpu_kernel_def_hashes = kernel_registry->ExportKernelDefHashes();
  const auto& expected_cpu_kernel_def_hashes = GetExpectedCpuKernelDefHashes();

  // remove kernel def hash added by another test
  cpu_kernel_def_hashes.erase(
      std::remove_if(
          cpu_kernel_def_hashes.begin(), cpu_kernel_def_hashes.end(),
          [](const KernelDefHashes::value_type& key_and_hash) {
            return key_and_hash.first == "OpaqueCApiTestKernel com.microsoft.mlfeaturizers CPUExecutionProvider";
          }),
      cpu_kernel_def_hashes.end());

  CheckKernelDefHashes(cpu_kernel_def_hashes, expected_cpu_kernel_def_hashes);
}

}  // namespace test
}  // namespace onnxruntime
