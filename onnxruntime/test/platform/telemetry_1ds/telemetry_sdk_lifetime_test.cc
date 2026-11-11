// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#include "Version.hpp"
#include "api/AuthTokensController.hpp"
#include "pal/PAL.hpp"

namespace {

namespace sdk = Microsoft::Applications::Events;
namespace pal = Microsoft::Applications::Events::PlatformAbstraction;

bool test_version = false;
bool late_access = false;
int version_accesses = 0;
int mutex_accesses = 0;
const void* initial_address = nullptr;
std::string initial_version;

[[noreturn]] void Fail(const char* message) {
  std::fprintf(stderr, "FAIL: %s\n", message);
  std::_Exit(21);
}

void AccessSdkObject() {
  if (test_version) {
    const std::string_view version = pal::getSdkVersion();
    if (!late_access) {
      initial_address = version.data();
      initial_version = version;
      const std::string_view suffix = "-" BUILD_VERSION_STR;
      if (version.size() < suffix.size() ||
          std::string_view(version).substr(version.size() - suffix.size()) != suffix) {
        Fail("SDK version did not contain the configured build version");
      }
    } else if (version.data() != initial_address || version != initial_version) {
      Fail("late SDK-version access did not preserve its address and value");
    }
  } else {
    auto& mutex = sdk::AuthTokensController::GetMutex();
    std::lock_guard lock(mutex);
    if (!late_access) {
      initial_address = &mutex;
    } else if (&mutex != initial_address) {
      Fail("late auth-mutex access did not preserve its address");
    }
  }
}

void VerifyExit() {
  if (!initial_address || (test_version ? version_accesses : mutex_accesses) != 1) {
    Fail("SDK object was not initialized after registering the exit owner");
  }
  late_access = true;
  // Model an owner's late worker without initializing the SDK upload machinery.
  std::thread late_worker(AccessSdkObject);
  late_worker.join();
  if ((test_version ? version_accesses : mutex_accesses) != 2) {
    Fail("late worker did not exercise the instrumented SDK accessor");
  }
  std::fputs("PASS: SDK object survived until the late exit worker joined\n", stderr);
}

}  // namespace

extern "C" void OrtTelemetrySdkVersionDestroyed() {
  Fail("SDK version destroyed before the late exit owner");
}

extern "C" void OrtTelemetryAuthMutexDestroyed() {
  Fail("auth mutex destroyed before the late exit owner");
}

extern "C" void OrtTelemetrySdkVersionAccess() {
  ++version_accesses;
}

extern "C" void OrtTelemetryAuthMutexAccess() {
  ++mutex_accesses;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    Fail("expected an SDK lifetime test mode");
  }
  const std::string_view mode = argv[1];
  if (mode != "sdk-version" && mode != "sdk-version-threaded" &&
      mode != "auth-mutex" && mode != "auth-mutex-threaded") {
    Fail("unknown SDK lifetime test mode");
  }
  test_version = mode == "sdk-version" || mode == "sdk-version-threaded";
  if (std::atexit(VerifyExit) != 0) {
    Fail("could not register the late exit owner");
  }
  if (mode == "sdk-version-threaded" || mode == "auth-mutex-threaded") {
    std::thread initializer(AccessSdkObject);
    initializer.join();
  } else {
    AccessSdkObject();
  }
  return 0;
}
