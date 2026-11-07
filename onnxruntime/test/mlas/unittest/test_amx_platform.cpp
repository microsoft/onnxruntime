// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"

#if defined(__linux__) && defined(MLAS_TARGET_AMD64) && defined(MLAS_ENABLE_TEST_HOOKS)

#include <pthread.h>
#include <sched.h>
#include <thread>

namespace {

constexpr unsigned long kTilePermissionMask = (1ul << 17) | (1ul << 18);

struct ProbeContext {
  bool PreserveData;
  int Calls = 0;
};

bool TestTileDataRoundTrip(int cpu, void* arg) {
  auto& context = *static_cast<ProbeContext*>(arg);
  ++context.Calls;
  const int current = sched_getcpu();
  EXPECT_GE(current, 0);
  EXPECT_NE(current, cpu);
  cpu_set_t target;
  CPU_ZERO(&target);
  CPU_SET(cpu, &target);
  const int rc = sched_setaffinity(0, sizeof(target), &target);
  EXPECT_EQ(rc, 0);
  EXPECT_EQ(sched_getcpu(), cpu);
  return rc == 0 && context.PreserveData;
}

void TestProbeResult(bool preserve_data, unsigned long bitmask = kTilePermissionMask) {
  cpu_set_t before;
  if (sched_getaffinity(0, sizeof(before), &before) != 0) {
    GTEST_SKIP() << "CPU affinity is unavailable";
  }
  if (CPU_COUNT(&before) < 2) {
    GTEST_SKIP() << "Migration probe requires two allowed CPUs";
  }
  if (sched_setaffinity(0, sizeof(before), &before) != 0) {
    ProbeContext context{true};
    EXPECT_FALSE(MlasAMXCheckOSSupportForTesting(bitmask, TestTileDataRoundTrip, &context));
    EXPECT_EQ(context.Calls, 0);
    GTEST_SKIP() << "CPU affinity changes are unavailable";
  }
  ProbeContext context{preserve_data};
  EXPECT_EQ(MlasAMXCheckOSSupportForTesting(bitmask, TestTileDataRoundTrip, &context), preserve_data);
  EXPECT_EQ(context.Calls, 1);
  cpu_set_t after;
  ASSERT_EQ(sched_getaffinity(0, sizeof(after), &after), 0);
  EXPECT_TRUE(CPU_EQUAL(&before, &after));
}

}  // namespace

TEST(AmxPlatform, PreservedTileData) {
  TestProbeResult(true);
}

TEST(AmxPlatform, LostTileData) {
  TestProbeResult(false);
}

TEST(AmxPlatform, TileDataPermissionRequired) {
  for (unsigned long permission : {0ul, 1ul << 17}) {
    ProbeContext context{true};
    EXPECT_FALSE(MlasAMXCheckOSSupportForTesting(permission, TestTileDataRoundTrip, &context));
    EXPECT_EQ(context.Calls, 0);
  }
}

TEST(AmxPlatform, OlderKernelTileDataPermission) {
  TestProbeResult(true, 1ul << 18);
}

TEST(AmxPlatform, CancellationStateUnchanged) {
  for (int state : {PTHREAD_CANCEL_ENABLE, PTHREAD_CANCEL_DISABLE}) {
    std::thread worker([state] {
      int original;
      ASSERT_EQ(pthread_setcancelstate(state, &original), 0);
      ProbeContext context{false};
      EXPECT_FALSE(MlasAMXCheckOSSupportForTesting(kTilePermissionMask, TestTileDataRoundTrip, &context));
      int previous;
      ASSERT_EQ(pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &previous), 0);
      EXPECT_EQ(previous, state);
      ASSERT_EQ(pthread_setcancelstate(original, nullptr), 0);
    });
    worker.join();
  }
}

#if defined(__GLIBC__)
#if __GLIBC_PREREQ(2, 18)
TEST(AmxPlatform, DetachedThreadDefaults) {
  pthread_attr_t original;
  ASSERT_EQ(pthread_getattr_default_np(&original), 0);
  pthread_attr_t attributes;
  ASSERT_EQ(pthread_attr_init(&attributes), 0);
  ASSERT_EQ(pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED), 0);
  const int rc = pthread_setattr_default_np(&attributes);
  EXPECT_EQ(pthread_attr_destroy(&attributes), 0);
  ASSERT_EQ(rc, 0);
  TestProbeResult(true);
  EXPECT_EQ(pthread_setattr_default_np(&original), 0);
  EXPECT_EQ(pthread_attr_destroy(&original), 0);
}
#endif
#endif

TEST(AmxPlatform, UnverifiableMigrationDisablesAmx) {
  cpu_set_t before;
  if (sched_getaffinity(0, sizeof(before), &before) != 0) {
    GTEST_SKIP() << "CPU affinity is unavailable";
  }
  int cpu = 0;
  while (cpu < CPU_SETSIZE && !CPU_ISSET(cpu, &before)) {
    ++cpu;
  }
  ASSERT_LT(cpu, CPU_SETSIZE);
  std::thread worker([cpu] {
    cpu_set_t target;
    CPU_ZERO(&target);
    CPU_SET(cpu, &target);
    if (sched_setaffinity(0, sizeof(target), &target) != 0) {
      GTEST_SKIP() << "CPU affinity changes are unavailable";
    }
    ProbeContext context{true};
    EXPECT_FALSE(MlasAMXCheckOSSupportForTesting(kTilePermissionMask, TestTileDataRoundTrip, &context));
    EXPECT_EQ(context.Calls, 0);
#if !defined(ORT_MINIMAL_BUILD)
    MLAS_PLATFORM platform;
    EXPECT_NE(platform.GemmU8S8Dispatch, &MlasGemmU8S8DispatchAmx);
#endif
  });
  worker.join();
  cpu_set_t after;
  ASSERT_EQ(sched_getaffinity(0, sizeof(after), &after), 0);
  EXPECT_TRUE(CPU_EQUAL(&before, &after));
}

#if !defined(ORT_MINIMAL_BUILD)
TEST(AmxPlatform, DispatchRequiresPreservedTileData) {
  unsigned int eax, ebx, ecx, edx;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  constexpr unsigned int amx_features = (1u << 24) | (1u << 25);
  if ((edx & amx_features) != amx_features) {
    GTEST_SKIP() << "AMX-TILE and AMX-INT8 are required";
  }
  if (!GetMlasPlatform().Avx2Supported_) {
    GTEST_SKIP() << "AMX dispatch requires AVX2/FMA3";
  }
  __cpuid(1, eax, ebx, ecx, edx);
  if ((ecx & (1u << 27)) == 0) {
    GTEST_SKIP() << "OSXSAVE is required";
  }
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  constexpr unsigned int tile_state = (1u << 17) | (1u << 18);
  if ((eax & tile_state) != tile_state) {
    GTEST_SKIP() << "AMX state is not enabled";
  }
  const bool supported = MlasInitAMX();
  MLAS_PLATFORM platform;
  EXPECT_EQ(platform.GemmU8S8Dispatch == &MlasGemmU8S8DispatchAmx, supported);
  EXPECT_EQ(GetMlasPlatform().GemmU8S8Dispatch == &MlasGemmU8S8DispatchAmx, supported);
}
#endif

#endif
