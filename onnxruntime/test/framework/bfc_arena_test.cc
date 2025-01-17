// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <absl/base/config.h>
#include "core/framework/bfc_arena.h"
#include "core/framework/allocator_utils.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <cstdlib>
#include "core/framework/stream_handles.h"

namespace onnxruntime {
namespace test {
static void CheckStats(BFCArena* a, int64_t num_allocs, int64_t bytes_in_use,
                       int64_t max_bytes_in_use, int64_t max_alloc_size) {
  AllocatorStats stats;
  a->GetStats(&stats);
  EXPECT_EQ(stats.bytes_in_use, bytes_in_use);
  EXPECT_EQ(stats.max_bytes_in_use, max_bytes_in_use);
  EXPECT_EQ(stats.num_allocs, num_allocs);
  EXPECT_EQ(stats.max_alloc_size, max_alloc_size);
}

TEST(BFCArenaTest, NoDups) {
  BFCArena a(std::unique_ptr<IAllocator>(new CPUAllocator()), 1 << 30);
  CheckStats(&a, 0, 0, 0, 0);

  // Allocate a lot of raw pointers
  std::vector<void*> ptrs;
  for (int s = 1; s < 1024; s++) {
    void* raw = a.Alloc(s);
    ptrs.push_back(raw);
  }
  CheckStats(&a, 1023, 654336, 654336, 1024);

  std::sort(ptrs.begin(), ptrs.end());

  // Make sure none of them are equal, and that none of them overlap.
  for (size_t i = 1; i < ptrs.size(); i++) {
    ASSERT_NE(ptrs[i], ptrs[i - 1]);  // No dups
    size_t req_size = a.RequestedSize(ptrs[i - 1]);
    ASSERT_GT(req_size, 0u);
    ASSERT_GE(static_cast<size_t>(static_cast<char*>(ptrs[i]) - static_cast<char*>(ptrs[i - 1])),
              req_size);
  }

  for (size_t i = 0; i < ptrs.size(); i++) {
    a.Free(ptrs[i]);
  }
  CheckStats(&a, 1023, 0, 654336, 1024);
}

TEST(BFCArenaTest, AllocationsAndDeallocations) {
  BFCArena a(std::unique_ptr<IAllocator>(new CPUAllocator()), 1 << 30);
  // Allocate 256 raw pointers of sizes between 100 bytes and about a meg
  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  std::vector<void*> initial_ptrs;
  for (int s = 1; s < 256; s++) {
    size_t size = std::min<size_t>(
        std::max<size_t>(rand() % 1048576, 100), 1048576);
    void* raw = a.Alloc(size);

    initial_ptrs.push_back(raw);
  }

  // Deallocate half of the memory, and keep track of the others.
  std::vector<void*> existing_ptrs;
  for (size_t i = 0; i < initial_ptrs.size(); i++) {
    if (i % 2 == 1) {
      a.Free(initial_ptrs[i]);
    } else {
      existing_ptrs.push_back(initial_ptrs[i]);
    }
  }

  // Ensure out of memory errors work and do not prevent future allocations from
  // working.

  EXPECT_THROW(a.Alloc((1 << 30) + 1), OnnxRuntimeException);

  // Allocate a lot of raw pointers
  for (int s = 1; s < 256; s++) {
    size_t size = std::min<size_t>(
        std::max<size_t>(rand() % 1048576, 100), 1048576);
    void* raw = a.Alloc(size);
    existing_ptrs.push_back(raw);
  }

  std::sort(existing_ptrs.begin(), existing_ptrs.end());
  // Make sure none of them are equal
  for (size_t i = 1; i < existing_ptrs.size(); i++) {
    EXPECT_NE(existing_ptrs[i], existing_ptrs[i - 1]);  // No dups

    size_t req_size = a.RequestedSize(existing_ptrs[i - 1]);
    ASSERT_GT(req_size, 0u);

    // Check that they don't overlap.
    ASSERT_GE(static_cast<size_t>(static_cast<char*>(existing_ptrs[i]) -
                                  static_cast<char*>(existing_ptrs[i - 1])),
              req_size);
  }

  for (size_t i = 0; i < existing_ptrs.size(); i++) {
    a.Free(existing_ptrs[i]);
  }
}

TEST(BFCArenaTest, ExerciseCoalescing) {
  BFCArena a(std::unique_ptr<IAllocator>(new CPUAllocator()), 1 << 30);
  CheckStats(&a, 0, 0, 0, 0);

  void* first_ptr = a.Alloc(4096);
  a.Free(first_ptr);
  CheckStats(&a, 1, 0, 4096, 4096);
  for (int i = 0; i < 1024; ++i) {
    // Allocate several buffers of different sizes, and then clean them
    // all up.  We should be able to repeat this endlessly without
    // causing fragmentation and growth.
    void* t1 = a.Alloc(4096);

    void* t2 = a.Alloc(1048576 * sizeof(int64_t));
    void* t3 = a.Alloc(2048 * sizeof(double));
    void* t4 = a.Alloc(1048576 * sizeof(int64_t) * sizeof(float));

    a.Free(t1);
    a.Free(t2);
    a.Free(t3);
    a.Free(t4);
  }
  CheckStats(&a, 4097, 0,
             1024 * sizeof(float) + 1048576 * sizeof(int64_t) +
                 2048 * sizeof(double) + 1048576 * sizeof(int64_t) * sizeof(float),
             1048576 * sizeof(int64_t) * sizeof(float));

  // At the end, we should have coalesced all memory into one region
  // starting at the beginning, so validate that allocating a pointer
  // starts from this region.
  void* first_ptr_after = a.Alloc(1024 * sizeof(float));
  EXPECT_EQ(first_ptr, first_ptr_after);
  a.Free(first_ptr_after);
}

TEST(BFCArenaTest, AllocateZeroBufSize) {
  BFCArena a(std::unique_ptr<IAllocator>(new CPUAllocator()), 1 << 30);
  void* ptr = a.Alloc(0);
  EXPECT_EQ(nullptr, ptr);
}

TEST(BFCArenaTest, AllocatedVsRequested) {
  BFCArena a(std::unique_ptr<IAllocator>(new CPUAllocator()), 1 << 30);
  void* t1 = a.Alloc(4);
  EXPECT_EQ(4u, a.RequestedSize(t1));
  EXPECT_EQ(256u, a.AllocatedSize(t1));
  a.Free(t1);
}

void TestCustomMemoryLimit_ProcessException(const OnnxRuntimeException& ex) {
#ifdef GTEST_USES_POSIX_RE
  EXPECT_THAT(ex.what(),
              testing::ContainsRegex("Available memory of [0-9]+ is smaller than requested bytes of [0-9]+"));
#else
  EXPECT_THAT(ex.what(),
              testing::ContainsRegex("Available memory of \\d+ is smaller than requested bytes of \\d+"));
#endif  // #ifdef GTEST_USES_POSIX_RE
}

// Address Sanitizer would report allocation-size-too-big if we don't disable this test.
#ifndef ABSL_HAVE_ADDRESS_SANITIZER
TEST(BFCArenaTest, TestCustomMemoryLimit) {
  {
    // Configure a 1MiB byte limit
    BFCArena a(std::unique_ptr<IAllocator>(new CPUAllocator()), 1 << 20);

    void* first_ptr = a.Alloc(sizeof(float) * (1 << 6));
    EXPECT_NE(nullptr, first_ptr);

    // test allocation of more than available memory throws
    ORT_TRY {
      a.Alloc(sizeof(float) * (1 << 20));
      FAIL() << "Allocation should have thrown";
    }
    ORT_CATCH(const OnnxRuntimeException& ex) {
      ORT_HANDLE_EXCEPTION([&ex]() {
        TestCustomMemoryLimit_ProcessException(ex);
      });
    }
    ORT_CATCH(...) {
      FAIL() << "Allocation should have thrown OnnxRuntimeException";
    }
    a.Free(first_ptr);
  }

  {
    // allow for the maximum amount of memory less 5MiB
    constexpr size_t available = std::numeric_limits<size_t>::max() - (5 * 1024 * 1024);
    BFCArena b(std::unique_ptr<IAllocator>(new CPUAllocator()), available,
               ArenaExtendStrategy::kSameAsRequested);  // need this strategy. kNextPowerOfTwo would overflow size_t

    void* first_ptr = b.Alloc(sizeof(float) * (1 << 6));
    EXPECT_NE(nullptr, first_ptr);

    // test allocation that is less than available memory, but more than what could reasonably be expected to exist.
    // first alloc creates a 1MB block so allow for that not being available.
    ORT_TRY {
      b.Alloc(available - (3 * 1024 * 1024));
      FAIL() << "Allocation should have thrown";
    }
    ORT_CATCH(const OnnxRuntimeException& ex) {
      ORT_HANDLE_EXCEPTION([&ex]() {
        EXPECT_THAT(ex.what(), testing::HasSubstr("Failed to allocate memory for requested buffer of size"));
      });
    }
    ORT_CATCH(...) {
      FAIL() << "Allocation should have thrown OnnxRuntimeException";
    }
    b.Free(first_ptr);
  }
}
#endif

TEST(BFCArenaTest, AllocationsAndDeallocationsWithGrowth) {
  // Max of 2GiB, but starts out small.
  BFCArena a(std::unique_ptr<IAllocator>(new CPUAllocator()), 1LL << 31);

  // Allocate 10 raw pointers of sizes between 100 bytes and about
  // 64 megs.
  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  constexpr int32_t max_mem = 1 << 27;

  std::vector<void*> initial_ptrs;
  for (int s = 1; s < 10; s++) {
    size_t size = std::min<size_t>(
        std::max<size_t>(rand() % max_mem, 100), max_mem);
    void* raw = a.Alloc(size);

    initial_ptrs.push_back(raw);
  }

  // Deallocate half of the memory, and keep track of the others.
  std::vector<void*> existing_ptrs;
  for (size_t i = 0; i < initial_ptrs.size(); i++) {
    if (i % 2 == 1) {
      a.Free(initial_ptrs[i]);
    } else {
      existing_ptrs.push_back(initial_ptrs[i]);
    }
  }

  constexpr int32_t max_mem_2 = 1 << 26;
  // Allocate a lot of raw pointers between 100 bytes and 64 megs.
  for (int s = 1; s < 10; s++) {
    size_t size = std::min<size_t>(
        std::max<size_t>(rand() % max_mem_2, 100), max_mem_2);
    void* raw = a.Alloc(size);
    existing_ptrs.push_back(raw);
  }

  std::sort(existing_ptrs.begin(), existing_ptrs.end());
  // Make sure none of them are equal
  for (size_t i = 1; i < existing_ptrs.size(); i++) {
    EXPECT_NE(existing_ptrs[i], existing_ptrs[i - 1]);  // No dups

    size_t req_size = a.RequestedSize(existing_ptrs[i - 1]);
    ASSERT_GT(req_size, 0u);

    // Check that they don't overlap.
    ASSERT_GE(static_cast<size_t>(
                  static_cast<char*>(existing_ptrs[i]) -
                  static_cast<char*>(existing_ptrs[i - 1])),
              req_size);
  }

  for (size_t i = 0; i < existing_ptrs.size(); i++) {
    a.Free(existing_ptrs[i]);
  }
}

TEST(BFCArenaTest, TestReserve) {
  // Configure a 1GiB byte limit
  BFCArena a(std::unique_ptr<IAllocator>(new CPUAllocator()), 1 << 30);

  void* first_ptr = a.Alloc(sizeof(float) * (1 << 6));
  void* second_ptr = a.Reserve(sizeof(float) * (1 << 20));
  a.Free(first_ptr);
  a.Free(second_ptr);

  AllocatorStats stats;
  a.GetStats(&stats);
  EXPECT_EQ(stats.total_allocated_bytes, 1048576);
}

TEST(BFCArenaTest, TestShrink) {
  AllocatorStats stats;
  BFCArena a(std::unique_ptr<IAllocator>(new CPUAllocator()), 1 << 30, ArenaExtendStrategy::kSameAsRequested);
  void* p1k = a.Alloc(1024);
  /* void* p10M =*/a.Alloc(10 * 1024 * 1024);
  a.GetStats(&stats);
  EXPECT_EQ(stats.num_arena_extensions, 2) << "Expect 2 regions but got " << stats.num_arena_extensions << " region";
  a.Free(p1k);

  EXPECT_EQ(a.Shrink(), Status::OK());
  a.GetStats(&stats);
  EXPECT_EQ(stats.num_arena_extensions, 1) << "1 region left as p10M is still in use";
  EXPECT_EQ(stats.num_arena_shrinkages, 1) << "shrink only once as only p1k is freed";
  EXPECT_EQ(stats.total_allocated_bytes, 10 * 1024 * 1024) << "Expect 10M bytes but actually " << stats.total_allocated_bytes << " bytes";
}

class BadAllocator : public IAllocator {
 public:
  BadAllocator() : IAllocator(OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator)) {}

  void* Alloc(size_t /*size*/) override { throw std::bad_alloc(); }
  void Free(void* /*p*/) override {}
};

TEST(BFCArenaTest, TestBackoffDoesntHang) {
  // test that if there are allocation failures the backoff logic doesn't hang. See comments in BFCArena::Extend
  BFCArena a(std::unique_ptr<IAllocator>(new BadAllocator()), 10 * 1024 * 1024);
  EXPECT_THROW(a.Alloc(1024), OnnxRuntimeException) << "Arena should be unable to allocate memory";
}

struct NotificationMock : public synchronize::Notification {
 public:
  NotificationMock(Stream& s) : Notification(s) {}
  void Activate() override {}
};

struct StreamMock : public Stream {
 public:
  StreamMock(const OrtDevice& device) : Stream(nullptr, device) {}
  std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) override {
    return std::make_unique<NotificationMock>(*this);
  }
  void Flush() override {}
  Status CleanUpOnRunEnd() override { return Status::OK(); }
};

#ifdef ORT_ENABLE_STREAM
TEST(StreamAwareArenaTest, TwoStreamAllocation) {
  StreamAwareArena a(std::unique_ptr<IAllocator>(new CPUAllocator()), 1 << 30, false);
  CheckStats(&a, 0, 0, 0, 0);

  OrtDevice tmp;

  StreamMock stream1(tmp), stream2(tmp);

  auto* stream1_chunk_a = a.AllocOnStream(4096, &stream1, nullptr);
  auto* stream2_chunk_a = a.AllocOnStream(4096, &stream2, nullptr);
  a.Free(stream1_chunk_a);
  auto* stream2_chunk_b = a.AllocOnStream(4096, &stream2, nullptr);
  // stream2 can't reuse stream1's chunk
  EXPECT_NE(stream2_chunk_b, stream1_chunk_a);
  a.Free(stream2_chunk_a);
  auto* stream1_chunk_c = a.AllocOnStream(4096, &stream1, nullptr);
  // it should pick the first chunk
  EXPECT_EQ(stream1_chunk_c, stream1_chunk_a);

  auto* stream1_chunk_d = a.AllocOnStream(4096, &stream1, nullptr);
  // it shouldn't pick stream2_chunk_a's buffer
  EXPECT_NE(stream1_chunk_d, stream2_chunk_a);
  a.Free(stream2_chunk_b);
  // test clean stream2
  a.ReleaseStreamBuffers(&stream2);
  auto stream1_chunk_e = a.AllocOnStream(8192, &stream1, nullptr);
  // now it should pick the stream2_chunk_a's buffer
  EXPECT_EQ(stream1_chunk_e, stream2_chunk_a);
  a.Free(stream1_chunk_c);
  a.Free(stream1_chunk_d);
  // add stream2 to stream 1 depenency
  auto stream1_notification_a = stream1.CreateNotification(1);
  stream1_notification_a->ActivateAndUpdate();
  stream2.UpdateStreamClock(stream1_notification_a->GetStreamSyncTable());
  auto* stream2_chunk_c = a.AllocOnStream(4096, &stream2, nullptr);
  // it should pick the first chunk
  EXPECT_EQ(stream2_chunk_c, stream1_chunk_c);
  auto* stream2_chunk_d = a.AllocOnStream(4096, &stream2, nullptr);
  // it should pick the third slot
  EXPECT_EQ(stream2_chunk_d, stream1_chunk_d);
  // continue allocate on stream1
  auto* stream1_chunk_f = a.AllocOnStream(4096, &stream1, nullptr);
  a.Free(stream1_chunk_f);
  auto* stream2_chunk_e = a.AllocOnStream(4096, &stream2, nullptr);
  EXPECT_NE(stream2_chunk_e, stream1_chunk_f);
  a.Free(stream1_chunk_e);
  // test clean stream1
  a.ReleaseStreamBuffers(&stream1);
  auto* stream2_chunk_f = a.AllocOnStream(8192, &stream2, nullptr);
  // now it should pick stream1_chunk_e
  EXPECT_EQ(stream2_chunk_f, stream1_chunk_e);

  // cleanup
  a.Free(stream2_chunk_d);
  a.Free(stream2_chunk_e);
  a.Free(stream2_chunk_f);
}

TEST(StreamAwareArenaTest, TestSecureTheChunk) {
  StreamAwareArena a(std::unique_ptr<IAllocator>(new CPUAllocator()), 1 << 30, true);
  OrtDevice tmp;
  StreamMock stream1(tmp), stream2(tmp);

  void* p1 = a.AllocOnStream(BFCArena::DEFAULT_INITIAL_CHUNK_SIZE_BYTES, &stream1, nullptr);
  a.Free(p1);

  bool waitFunctionInvoked = false;
  void* p2 = a.AllocOnStream(BFCArena::DEFAULT_INITIAL_CHUNK_SIZE_BYTES, &stream2,
                             [&waitFunctionInvoked](Stream&, synchronize::Notification&) { waitFunctionInvoked = true; });

  std::unordered_map<Stream*, uint64_t> syncTable;
  stream2.CloneCurrentStreamSyncTable(syncTable);
  EXPECT_EQ(syncTable.size(), 1u) << "stream2 has been updated with stream1's nofitication on the clock";
  EXPECT_TRUE(waitFunctionInvoked) << "wait function should be invoked";
  a.Free(p2);
}
#endif

TEST(BFCArenaTest, TestExtendStrategy) {
  int64_t extend_delta_bytes = 0;
  {
    // Use kNextPowerOfTwo strategy with default extension limit: 1GB.
    BFCArena a(std::unique_ptr<IAllocator>(new CPUAllocator()), 1UL << 30, ArenaExtendStrategy::kNextPowerOfTwo);
    size_t block_size = 1 << 20;  // 1MB
    a.Alloc(block_size);
    AllocatorStats stats;
    a.GetStats(&stats);
    int64_t prev_allocated_bytes = stats.total_allocated_bytes;
    extend_delta_bytes = stats.total_allocated_bytes;
    ASSERT_EQ(extend_delta_bytes, static_cast<int64_t>(block_size));
    for (int i = 1; i < 256; ++i) {
      a.Alloc(block_size);
      a.GetStats(&stats);
      if (stats.total_allocated_bytes != prev_allocated_bytes) {
        int64_t new_delta_bytes = stats.total_allocated_bytes - prev_allocated_bytes;
        ASSERT_EQ(new_delta_bytes, 2 * extend_delta_bytes);
        extend_delta_bytes = new_delta_bytes;
        prev_allocated_bytes = stats.total_allocated_bytes;
      }
    }
  }
  int64_t extend_limit = 1 << 25;  // 32MB
  ASSERT_GT(extend_delta_bytes, extend_limit);
  extend_delta_bytes = 0;
  {
    // Use kNextPowerOfTwo strategy with much smaller extension limit: 32MB.
    OrtArenaCfg config(0, 0, -1, -1, -1, extend_limit);
    AllocatorCreationInfo device_info{
        [](OrtDevice::DeviceId) { return std::make_unique<CPUAllocator>(); },
        0, true, config};
    auto allocator = CreateAllocator(device_info);
    size_t block_size = 1 << 20;  // 1MB
    BFCArena& a = *static_cast<BFCArena*>(allocator.get());
    a.Alloc(block_size);
    AllocatorStats stats;
    a.GetStats(&stats);
    int64_t prev_allocated_bytes = stats.total_allocated_bytes;
    extend_delta_bytes = stats.total_allocated_bytes;
    ASSERT_EQ(extend_delta_bytes, static_cast<int64_t>(block_size));
    int reach_limit_count = 0;
    for (int i = 1; i < 256; ++i) {
      a.Alloc(block_size);
      a.GetStats(&stats);
      if (stats.total_allocated_bytes != prev_allocated_bytes) {
        int64_t new_delta_bytes = stats.total_allocated_bytes - prev_allocated_bytes;
        if (new_delta_bytes < extend_limit) {
          ASSERT_EQ(new_delta_bytes, 2 * extend_delta_bytes) << "index:" << i;
        } else {
          // The increasing of new chunk reaches the limit.
          ++reach_limit_count;
          ASSERT_EQ(new_delta_bytes, extend_limit);
        }
        extend_delta_bytes = new_delta_bytes;
        prev_allocated_bytes = stats.total_allocated_bytes;
      }
    }
    ASSERT_GT(reach_limit_count, 2);
    // It is OK to allocate more than extend_limit.
    ASSERT_NE(a.Alloc(block_size * 64), nullptr);
  }
  ASSERT_EQ(extend_delta_bytes, extend_limit);
}

}  // namespace test
}  // namespace onnxruntime
