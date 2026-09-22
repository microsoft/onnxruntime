// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ORT_CUDA_GDS_AVAILABLE)

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>

#include <fcntl.h>
#include <unistd.h>

#include "core/common/inlined_containers.h"
#include "core/providers/cuda/cuda_external_data_loader_gds.h"
#include "core/providers/cuda/cuda_external_data_loader_gds_api.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
namespace {

class CudaGdsIoTest : public ::testing::Test {
 protected:
  static constexpr size_t kAlignment = cuda::kGdsIoAlignment;
  static constexpr size_t kBufferSize = cuda::kGdsBufferSize;
  static constexpr uint8_t kUntouched = 0xff;

  struct Read {
    size_t length;
    off_t file_offset;
  };

  void SetUp() override {
    file_.reset(std::tmpfile());
    ASSERT_NE(file_, nullptr);
    original_flags_ = fcntl(fileno(file_.get()), F_GETFL);
    ASSERT_GE(original_flags_, 0);

    api_.handle_register = [this](CUfileHandle_t* handle, CUfileDescr_t* descriptor) {
      calls_.push_back("register");
      EXPECT_EQ(descriptor->type, CU_FILE_HANDLE_TYPE_OPAQUE_FD);
      registered_fd_ = descriptor->handle.fd;
      EXPECT_NE(registered_fd_, fileno(file_.get()));
      EXPECT_EQ(fcntl(registered_fd_, F_GETFL), original_flags_ | O_DIRECT);
      if (register_error_ == CU_FILE_SUCCESS) {
        *handle = this;
      }
      return CUfileError_t{register_error_, CUDA_SUCCESS};
    };
    api_.handle_deregister = [this](CUfileHandle_t handle) {
      calls_.push_back("deregister");
      EXPECT_EQ(handle, this);
      EXPECT_EQ(fcntl(registered_fd_, F_GETFL), original_flags_ | O_DIRECT);
    };
    api_.read = [this](CUfileHandle_t handle, void* buffer, size_t length,
                       off_t file_offset, off_t buffer_offset) -> ssize_t {
      calls_.push_back("read");
      EXPECT_EQ(handle, this);
      EXPECT_EQ(buffer, staging_.data());
      EXPECT_EQ(buffer_offset, 0);
      reads_.push_back({length, file_offset});
      if (reads_.size() == fail_read_) {
        errno = EIO;
        return read_result_;
      }
      std::memset(buffer, ReadValue(file_offset), length);
      return static_cast<ssize_t>(length);
    };
    api_.copy = [this](void* destination, const void* source, size_t length, cudaMemcpyKind kind) {
      calls_.push_back("copy");
      EXPECT_EQ(kind, cudaMemcpyDeviceToDevice);
      EXPECT_EQ(source, staging_.data());
      EXPECT_EQ(destination, destination_.data() + kAlignment + copied_);
      EXPECT_EQ(length, reads_.back().length);
      if (fail_copy_) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Injected device copy failure");
      }
      std::memcpy(destination, source, length);
      copied_ += length;
      return Status::OK();
    };
    api_.synchronize = [this](cudaStream_t stream) {
      calls_.push_back("synchronize");
      EXPECT_EQ(stream, nullptr);
      if (fail_synchronize_) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Injected device synchronization failure");
      }
      return Status::OK();
    };
  }

  Status Load(size_t length = kAlignment) {
    destination_.assign(length + 2 * kAlignment, kUntouched);
    return cuda::LoadGdsFile(api_, staging_.data(), fileno(file_.get()), kAlignment,
                             length, destination_.data() + kAlignment);
  }

  void ExpectCleanup(bool registered = true) {
    ASSERT_GE(registered_fd_, 0);
    EXPECT_EQ(std::count(calls_.begin(), calls_.end(), "deregister"), registered ? 1 : 0);
    EXPECT_EQ(fcntl(registered_fd_, F_GETFD), -1);
    EXPECT_EQ(errno, EBADF);
    EXPECT_EQ(fcntl(fileno(file_.get()), F_GETFL), original_flags_);
    ExpectBytes(0, kAlignment, kUntouched);
    ExpectBytes(destination_.size() - kAlignment, kAlignment, kUntouched);
  }

  void ExpectBytes(size_t offset, size_t length, uint8_t value) {
    ASSERT_LE(offset + length, destination_.size());
    EXPECT_TRUE(std::all_of(destination_.begin() + offset, destination_.begin() + offset + length,
                            [value](uint8_t byte) { return byte == value; }))
        << "offset=" << offset << ", length=" << length;
  }

  void ExpectError(const Status& status, std::string_view message) {
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(status.ErrorMessage().find(message), std::string::npos) << status;
  }

  static uint8_t ReadValue(off_t file_offset) {
    return static_cast<uint8_t>((file_offset / kAlignment) % 251);
  }

  std::unique_ptr<std::FILE, int (*)(std::FILE*)> file_{nullptr, std::fclose};
  int original_flags_{-1};
  int registered_fd_{-1};
  cuda::GdsReadApi api_;
  InlinedVector<uint8_t> staging_ = InlinedVector<uint8_t>(kBufferSize);
  InlinedVector<uint8_t> destination_;
  InlinedVector<Read> reads_;
  InlinedVector<std::string_view> calls_;
  CUfileOpError register_error_{CU_FILE_SUCCESS};
  size_t fail_read_{0};
  ssize_t read_result_{0};
  size_t copied_{0};
  bool fail_copy_{false};
  bool fail_synchronize_{false};
};

TEST_F(CudaGdsIoTest, ReadsMultipleChunksAndTailBeforeReusingStagingBuffer) {
  const size_t length = 2 * kBufferSize + kAlignment;
  const auto status = Load(length);
  ASSERT_TRUE(status.IsOK()) << status;

  ASSERT_EQ(reads_.size(), 3U);
  EXPECT_EQ(copied_, length);
  for (size_t i = 0; i < reads_.size(); ++i) {
    const size_t chunk_length = i == 2 ? kAlignment : kBufferSize;
    const auto file_offset = static_cast<off_t>(kAlignment + i * kBufferSize);
    EXPECT_EQ(reads_[i].length, chunk_length);
    EXPECT_EQ(reads_[i].file_offset, file_offset);
    ExpectBytes(kAlignment + i * kBufferSize, chunk_length, ReadValue(file_offset));
  }
  EXPECT_EQ(calls_, (InlinedVector<std::string_view>{
                        "register", "read", "copy", "synchronize",
                        "read", "copy", "synchronize", "read", "copy", "synchronize", "deregister"}));
  ExpectCleanup();
}

TEST_F(CudaGdsIoTest, ShortReadDoesNotCopyIncompleteData) {
  fail_read_ = 1;
  read_result_ = kAlignment - 1;
  ExpectError(Load(), "cuFileRead returned 4095 bytes; expected 4096.");
  EXPECT_EQ(calls_, (InlinedVector<std::string_view>{"register", "read", "deregister"}));
  ExpectBytes(kAlignment, kAlignment, kUntouched);
  ExpectCleanup();
}

TEST_F(CudaGdsIoTest, EndOfFileDoesNotCopyStaleData) {
  fail_read_ = 1;
  read_result_ = 0;
  ExpectError(Load(), "cuFileRead returned 0 bytes; expected 4096.");
  EXPECT_EQ(calls_, (InlinedVector<std::string_view>{"register", "read", "deregister"}));
  ExpectBytes(kAlignment, kAlignment, kUntouched);
  ExpectCleanup();
}

TEST_F(CudaGdsIoTest, DecodesPosixReadError) {
  fail_read_ = 1;
  read_result_ = -1;
  ExpectError(Load(), std::string("cuFileRead failed: ") + std::strerror(EIO));
  EXPECT_EQ(calls_, (InlinedVector<std::string_view>{"register", "read", "deregister"}));
  ExpectBytes(kAlignment, kAlignment, kUntouched);
  ExpectCleanup();
}

TEST_F(CudaGdsIoTest, DecodesCuFileReadError) {
  fail_read_ = 1;
  read_result_ = -CU_FILE_IO_NOT_SUPPORTED;
  const auto status = Load();
  ExpectError(status, std::string("cuFileRead failed: ") + cufileop_status_error(CU_FILE_IO_NOT_SUPPORTED));
  ExpectError(status, "(" + std::to_string(CU_FILE_IO_NOT_SUPPORTED) + ")");
  EXPECT_EQ(calls_, (InlinedVector<std::string_view>{"register", "read", "deregister"}));
  ExpectBytes(kAlignment, kAlignment, kUntouched);
  ExpectCleanup();
}

TEST_F(CudaGdsIoTest, RegistrationFailureClosesDescriptorAndRestoresFlags) {
  register_error_ = CU_FILE_INVALID_VALUE;
  ExpectError(Load(), std::string("cuFileHandleRegister failed: ") + cufileop_status_error(register_error_));
  EXPECT_EQ(calls_, (InlinedVector<std::string_view>{"register"}));
  ExpectBytes(kAlignment, kAlignment, kUntouched);
  ExpectCleanup(false);
}

TEST_F(CudaGdsIoTest, CopyFailureStopsBeforeSynchronizationAndNextRead) {
  fail_copy_ = true;
  ExpectError(Load(kBufferSize + kAlignment), "Injected device copy failure");
  EXPECT_EQ(calls_, (InlinedVector<std::string_view>{"register", "read", "copy", "deregister"}));
  ExpectBytes(kAlignment, kBufferSize + kAlignment, kUntouched);
  ExpectCleanup();
}

TEST_F(CudaGdsIoTest, SynchronizationFailureStopsBeforeStagingBufferReuse) {
  fail_synchronize_ = true;
  ExpectError(Load(kBufferSize + kAlignment), "Injected device synchronization failure");
  EXPECT_EQ(calls_, (InlinedVector<std::string_view>{"register", "read", "copy", "synchronize", "deregister"}));
  ExpectBytes(kAlignment, kBufferSize, ReadValue(kAlignment));
  ExpectBytes(kAlignment + kBufferSize, kAlignment, kUntouched);
  ExpectCleanup();
}

TEST_F(CudaGdsIoTest, LaterReadFailureLeavesRemainingDestinationUntouched) {
  fail_read_ = 2;
  read_result_ = -1;
  ExpectError(Load(2 * kBufferSize + kAlignment), std::strerror(EIO));
  ASSERT_EQ(reads_.size(), 2U);
  EXPECT_EQ(reads_[1].file_offset, static_cast<off_t>(kAlignment + kBufferSize));
  EXPECT_EQ(calls_, (InlinedVector<std::string_view>{
                        "register", "read", "copy", "synchronize", "read", "deregister"}));
  ExpectBytes(kAlignment, kBufferSize, ReadValue(kAlignment));
  ExpectBytes(kAlignment + kBufferSize, kBufferSize + kAlignment, kUntouched);
  ExpectCleanup();
}

TEST_F(CudaGdsIoTest, InvalidDescriptorDoesNotCallCuFile) {
  ExpectError(cuda::LoadGdsFile(api_, staging_.data(), -1, 0, kAlignment, staging_.data()),
              "requires an open POSIX file descriptor");
  EXPECT_TRUE(calls_.empty());
}

TEST_F(CudaGdsIoTest, ClosedDescriptorReportsDuplicationFailure) {
  const int descriptor = dup(fileno(file_.get()));
  ASSERT_GE(descriptor, 0);
  ASSERT_EQ(close(descriptor), 0);
  ExpectError(cuda::LoadGdsFile(api_, staging_.data(), descriptor, 0, kAlignment, staging_.data()),
              "Failed to duplicate external-data file descriptor");
  EXPECT_TRUE(calls_.empty());
  EXPECT_EQ(fcntl(fileno(file_.get()), F_GETFL), original_flags_);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif
