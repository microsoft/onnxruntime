// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/capture.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/logging/sinks/composite_sink.h"
#include "core/common/logging/sinks/file_sink.h"

#include "test/common/logging/helpers.h"

using namespace ::onnxruntime::logging;
using InstanceType = LoggingManager::InstanceType;

namespace {
void CheckStringInFile(const std::string& filename, const std::string& look_for) {
  std::ifstream ifs{filename};
  std::string content(std::istreambuf_iterator<char>{ifs},
                      std::istreambuf_iterator<char>{});

  EXPECT_NE(content.find(look_for), std::string::npos);
}

void DeleteFile(const std::string& filename) {
  int result = std::remove(filename.c_str());
  EXPECT_EQ(result, 0);
}
}  // namespace

/// <summary>
/// Tests that the std::clog sink produces the expected output.
/// </summary>
TEST(LoggingTests, TestCLogSink) {
  const std::string filename{"TestCLogSink.out"};
  const std::string logid{"CLogSink"};
  const std::string message{"Test clog message"};
  const Severity min_log_level = Severity::kWARNING;

  // redirect clog to a file so we can check the output
  std::ofstream ofs(filename);

  auto old_rdbuf = std::clog.rdbuf();
  std::clog.rdbuf(ofs.rdbuf());

  // create scoped manager so sink gets destroyed once done
  {
    LoggingManager manager{std::unique_ptr<ISink>{new CLogSink{}}, min_log_level, false,
                           InstanceType::Temporal};

    auto logger = manager.CreateLogger(logid);

    LOGS(*logger, WARNING) << message;
  }

  // check message was flushed to file before we close ofs.
  CheckStringInFile(filename, message);

  // revert redirection
  std::clog.rdbuf(old_rdbuf);
  ofs.close();

  DeleteFile(filename);
}

/// <summary>
/// Tests that the std::cerr sink produces the expected output.
/// </summary>
TEST(LoggingTests, TestCErrSink) {
  const std::string filename{"TestCErrSink.out"};
  const std::string logid{"CErrSink"};
  const std::string message{"Test cerr message"};
  const Severity min_log_level = Severity::kWARNING;

  // redirect clog to a file so we can check the output
  std::ofstream ofs(filename);
  ofs << std::unitbuf;  // turn off buffering so we replicate how std::cerr behaves.

  auto old_rdbuf = std::cerr.rdbuf();
  std::cerr.rdbuf(ofs.rdbuf());

  // create scoped manager so sink gets destroyed once done
  {
    LoggingManager manager{std::unique_ptr<ISink>{new CErrSink{}}, min_log_level, false,
                           InstanceType::Temporal};

    auto logger = manager.CreateLogger(logid);

    LOGS(*logger, WARNING) << message;
  }

  // check message was flushed to file before we close ofs.
  CheckStringInFile(filename, message);

  // revert redirection
  std::cerr.rdbuf(old_rdbuf);
  ofs.close();

  DeleteFile(filename);
}

/// <summary>
/// Tests that the file_sink produces the expected output.
/// </summary>
TEST(LoggingTests, TestFileSink) {
  const std::string filename{"TestFileSink.out"};
  const std::string logid{"FileSink"};
  const std::string message{"Test message"};
  const Severity min_log_level = Severity::kWARNING;

  // create scoped manager so sink gets destroyed once done
  {
    LoggingManager manager{std::unique_ptr<ISink>{new FileSink{filename, false, false}},
                           min_log_level, false, InstanceType::Temporal};

    auto logger = manager.CreateLogger(logid);

    LOGS(*logger, WARNING) << message;
  }

  CheckStringInFile(filename, message);
  DeleteFile(filename);
}
//TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26400)
#endif
/// <summary>
/// Tests that a composite_sink works correctly.
/// </summary>
TEST(LoggingTests, TestCompositeSink) {
  const std::string logid{"TestCompositeSink"};
  const Severity min_log_level = Severity::kWARNING;

  MockSink* sink_ptr1 = new MockSink();
  MockSink* sink_ptr2 = new MockSink();

  // both should be called for a single log statement
  EXPECT_CALL(*sink_ptr1, SendImpl(testing::_, testing::_, testing::_)).Times(1);
  EXPECT_CALL(*sink_ptr2, SendImpl(testing::_, testing::_, testing::_)).Times(1);

  CompositeSink* sink = new CompositeSink();
  sink->AddSink(std::unique_ptr<ISink>{sink_ptr1}).AddSink(std::unique_ptr<ISink>{sink_ptr2});
  LoggingManager manager{std::unique_ptr<ISink>(sink), min_log_level, false, InstanceType::Temporal};

  auto logger = manager.CreateLogger(logid);

  LOGS_CATEGORY(*logger, WARNING, "ArbitraryCategory") << "Warning";
}
