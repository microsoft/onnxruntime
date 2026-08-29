// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "EpContextDataReadPolicy.h"
#include "LoadModelArgumentPolicy.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& what) {
  if (!condition) {
    ++g_failures;
    std::printf("FAILED: %s\n", what.c_str());
  }
}

void expectRejected(double raw, const std::string& what) {
  const auto result = onnxruntimejsi::parseMaxDataSize(raw);
  check(!result.ok, what);
  check(!result.error.empty(), what + " (reports an error)");
  check(result.value == 0, what + " (leaves the value zeroed)");
}

void expectAccepted(double raw, size_t expected, const std::string& what) {
  const auto result = onnxruntimejsi::parseMaxDataSize(raw);
  check(result.ok, what);
  check(result.error.empty(), what + " (reports no error)");
  check(result.value == expected, what + " (parses the value)");
}

// Kept out of line so a 32-bit build does not constant-fold a truncating cast in the branch that
// only ever runs on 64-bit targets.
size_t toSizeT(double value) { return static_cast<size_t>(value); }

void testParseMaxDataSize() {
  expectAccepted(1.0, 1, "accepts the smallest positive limit");
  expectAccepted(1024.0, 1024, "accepts a typical limit");
  expectAccepted(2147483648.0, toSizeT(2147483648.0),
                 "accepts a limit above INT32_MAX");

  expectRejected(0.0, "rejects zero");
  expectRejected(-0.0, "rejects negative zero");
  expectRejected(-1.0, "rejects a negative limit");
  expectRejected(1.5, "rejects a fractional limit");
  expectRejected(std::numeric_limits<double>::infinity(), "rejects infinity");
  expectRejected(-std::numeric_limits<double>::infinity(),
                 "rejects negative infinity");
  expectRejected(std::numeric_limits<double>::quiet_NaN(), "rejects NaN");
  expectRejected(onnxruntimejsi::kMaxSafeInteger + 2.0,
                 "rejects a value beyond Number.MAX_SAFE_INTEGER");

  // size_t is narrower than the safe-integer range on 32-bit targets, so the same JavaScript
  // number is accepted on 64-bit and rejected on 32-bit.
  if (sizeof(size_t) < sizeof(uint64_t)) {
    expectRejected(onnxruntimejsi::kMaxSafeInteger,
                   "rejects a limit that size_t cannot represent");
  } else {
    expectAccepted(onnxruntimejsi::kMaxSafeInteger,
                   toSizeT(onnxruntimejsi::kMaxSafeInteger),
                   "accepts Number.MAX_SAFE_INTEGER on 64-bit targets");
  }
}

void testCheckDataSize() {
  const auto underLimit = onnxruntimejsi::checkDataSize(10, 16, "ctx");
  check(underLimit.ok, "accepts a payload below the limit");
  check(underLimit.error.empty(), "accepts a payload below the limit quietly");

  const auto atLimit = onnxruntimejsi::checkDataSize(16, 16, "ctx");
  check(atLimit.ok, "accepts a payload exactly at the limit");

  const auto empty = onnxruntimejsi::checkDataSize(0, 1, "ctx");
  check(empty.ok, "accepts an empty payload");

  const auto overLimit = onnxruntimejsi::checkDataSize(17, 16, "ctx");
  check(!overLimit.ok, "rejects a payload above the limit");
  check(overLimit.error.find("ctx") != std::string::npos,
        "names the data in the limit error");
  check(overLimit.error.find("17") != std::string::npos,
        "reports the actual size in the limit error");
  check(overLimit.error.find("16") != std::string::npos,
        "reports the configured limit in the limit error");
}

void testLoadModelArgumentLayout() {
  using Type = onnxruntimejsi::LoadModelArgumentType;

  constexpr std::array pathArguments{Type::String, Type::Object};
  const auto pathLayout = onnxruntimejsi::resolveLoadModelArgumentLayout(
      pathArguments.data(), pathArguments.size());
  check(pathLayout.valid, "accepts the path overload");
  check(!pathLayout.usesModelBuffer, "identifies the path overload");
  check(pathLayout.optionsIndex == 1,
        "selects path options from argument 1");

  constexpr std::array bufferArguments{
      Type::ArrayBuffer, Type::Number, Type::Number, Type::Object};
  const auto bufferLayout = onnxruntimejsi::resolveLoadModelArgumentLayout(
      bufferArguments.data(), bufferArguments.size());
  check(bufferLayout.valid, "accepts the buffer overload");
  check(bufferLayout.usesModelBuffer, "identifies the buffer overload");
  check(bufferLayout.optionsIndex == 3,
        "selects buffer options from argument 3");

  constexpr std::array missingBufferOptions{
      Type::ArrayBuffer, Type::Number, Type::Number};
  check(!onnxruntimejsi::resolveLoadModelArgumentLayout(
             missingBufferOptions.data(), missingBufferOptions.size())
             .valid,
        "rejects a buffer overload without options");

  constexpr std::array wrongBufferOffset{
      Type::ArrayBuffer, Type::Object, Type::Number, Type::Object};
  check(!onnxruntimejsi::resolveLoadModelArgumentLayout(
             wrongBufferOffset.data(), wrongBufferOffset.size())
             .valid,
        "rejects a non-numeric buffer offset");

  constexpr std::array extraPathArgument{
      Type::String, Type::Object, Type::Other};
  check(!onnxruntimejsi::resolveLoadModelArgumentLayout(
             extraPathArgument.data(), extraPathArgument.size())
             .valid,
        "rejects extra path arguments");
}

}  // namespace

int main() {
  testParseMaxDataSize();
  testCheckDataSize();
  testLoadModelArgumentLayout();

  if (g_failures != 0) {
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
