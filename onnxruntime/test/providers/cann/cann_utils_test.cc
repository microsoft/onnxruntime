// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <gsl/util>
#include <iterator>
#include <string>

#include "core/providers/cann/cann_utils.h"
#include "gtest/gtest.h"
#include "test/util/include/asserts.h"
#include "test/util/include/temp_dir.h"

namespace onnxruntime::test {

namespace fs = std::filesystem;

class CannUtilsTest : public testing::Test {
 protected:
  CannUtilsTest() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CannUtilsTest);

  void SetUp() override { fs::current_path(temp_dir_.Path()); }
  void TearDown() override { fs::current_path(original_dir_); }

  void ExpectContents(const std::string& file_name, const std::string& expected) {
    const auto matched_file = cann::MatchFile(file_name);
    ASSERT_FALSE(matched_file.empty());

    std::ifstream input(matched_file, std::ios::binary);
    ASSERT_TRUE(input);

    const std::string contents{
        std::istreambuf_iterator<char>{input},
        std::istreambuf_iterator<char>{}};
    EXPECT_EQ(contents, expected);
  }

 private:
  const fs::path original_dir_ = fs::current_path();
  TemporaryDirectory temp_dir_{ORT_TSTR("cann_atomic_save_test")};
};

TEST_F(CannUtilsTest, SaveFileAtomicallyDoesNotPublishAnIncompleteFile) {
  const std::string file_name = "dummy_model";

  std::promise<void> partial_write, finish_write;
  auto partial_ready = partial_write.get_future();
  auto finish_ready = finish_write.get_future();

  auto status = std::async(std::launch::async, [&] {
    return cann::detail::SaveFileAtomically(file_name, [&](const std::string& filename) {
      std::ofstream output(filename + ".om", std::ios::binary);

      output << "partial" << std::flush;
      partial_write.set_value();
      finish_ready.wait();
      output << "-finish";

      return Status::OK();
    });
  });

  {
    auto unblock = gsl::finally([&] { finish_write.set_value(); });
    ASSERT_EQ(partial_ready.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_TRUE(cann::MatchFile(file_name).empty());
  }

  ASSERT_STATUS_OK(status.get());
  ExpectContents(file_name, "partial-finish");
}

TEST_F(CannUtilsTest, SaveFileAtomicallyPublishesCompleteFilesFromMultipleWriters) {
  const std::string file_name = "dummy_model";

  const int num_writers = 3;
  std::array<std::promise<void>, num_writers> partial_write, finish_write;
  std::array<std::future<Status>, num_writers> status;

  auto unblock = gsl::finally([&] {
    for (int i = 0; i < num_writers; ++i) {
      try {
        finish_write[i].set_value();
      } catch (const std::future_error&) {
        /* ignore */
      }
    }
  });

  for (int i = 0; i < num_writers; ++i) {
    status[i] = std::async(std::launch::async, [&, i] {
      return cann::detail::SaveFileAtomically(file_name, [&](const std::string& filename) {
        std::ofstream output(filename + ".om", std::ios::binary);

        output << "writer-" << i << std::flush;
        partial_write[i].set_value();
        finish_write[i].get_future().wait();
        output << "-finish";

        return Status::OK();
      });
    });
  }

  for (auto& partial : partial_write) {
    ASSERT_EQ(partial.get_future().wait_for(std::chrono::seconds(10)), std::future_status::ready);
  }

  EXPECT_TRUE(cann::MatchFile(file_name).empty());

  for (int i = 0; i < num_writers; ++i) {
    finish_write[i].set_value();
    ASSERT_STATUS_OK(status[i].get());
    ExpectContents(file_name, "writer-" + std::to_string(i) + "-finish");
  }
}

TEST_F(CannUtilsTest, SaveFileAtomicallyDoesNotPublishAfterSaveFailure) {
  const std::string file_name = "dummy_model";

  auto status = cann::detail::SaveFileAtomically(file_name, [](const std::string& filename) {
    std::ofstream output(filename + ".om", std::ios::binary);
    output << "partial" << std::flush;
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Injected save failure");
  });

  EXPECT_FALSE(status.IsOK());
  EXPECT_TRUE(cann::MatchFile(file_name).empty());
}

}  // namespace onnxruntime::test
