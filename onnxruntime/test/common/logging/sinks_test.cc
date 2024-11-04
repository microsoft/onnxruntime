// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/common/common.h"
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

#ifdef _WIN32
#define ONNXRUNTIME_CLOG_STREAM std::wclog
#define ONNXRUNTIME_CERR_STREAM std::wcerr
#else
#define ONNXRUNTIME_CLOG_STREAM std::clog
#define ONNXRUNTIME_CERR_STREAM std::cerr
#endif

/// <summary>
/// Tests that the std::clog sink produces the expected output.
/// </summary>
TEST(LoggingTests, TestCLogSink) {
  const std::string filename{"TestCLogSink.out"};
  const std::string logid{"CLogSink"};
  const std::string message{"Test clog message"};
  const Severity min_log_level = Severity::kWARNING;

  // redirect clog to a file so we can check the output
#ifdef _WIN32
  std::wofstream ofs(filename);
#else
  std::ofstream ofs(filename);
#endif

  auto old_rdbuf = ONNXRUNTIME_CLOG_STREAM.rdbuf();
  ONNXRUNTIME_CLOG_STREAM.rdbuf(ofs.rdbuf());

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
  ONNXRUNTIME_CLOG_STREAM.rdbuf(old_rdbuf);
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
#ifdef _WIN32
  std::wofstream ofs(filename);
#else
  std::ofstream ofs(filename);
#endif
  ofs << std::unitbuf;  // turn off buffering so we replicate how std::cerr behaves.

  auto old_rdbuf = ONNXRUNTIME_CERR_STREAM.rdbuf();
  ONNXRUNTIME_CERR_STREAM.rdbuf(ofs.rdbuf());

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
  ONNXRUNTIME_CERR_STREAM.rdbuf(old_rdbuf);
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
#ifdef _WIN32
    LoggingManager manager{std::make_unique<FileSink>(onnxruntime::ToWideString(filename), false, false),
                           min_log_level, false, InstanceType::Temporal};
#else
    LoggingManager manager{std::unique_ptr<ISink>{new FileSink{filename, false, false}},
                           min_log_level, false, InstanceType::Temporal};
#endif
    auto logger = manager.CreateLogger(logid);

    LOGS(*logger, WARNING) << message;
  }

  CheckStringInFile(filename, message);
  DeleteFile(filename);
}
// TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26400)
#endif
/// <summary>
/// Tests that a composite_sink works correctly.
/// </summary>
TEST(LoggingTests, TestCompositeSinkBasic) {
  const std::string logid{"TestCompositeSinkBasic"};
  const Severity min_log_level = Severity::kWARNING;

  MockSink* sink_ptr1 = new MockSink();
  MockSink* sink_ptr2 = new MockSink();

  // both should be called for a single log statement
  EXPECT_CALL(*sink_ptr1, SendImpl(testing::_, testing::_, testing::_)).Times(1);
  EXPECT_CALL(*sink_ptr2, SendImpl(testing::_, testing::_, testing::_)).Times(1);

  CompositeSink* sink = new CompositeSink();
  sink->AddSink(std::unique_ptr<ISink>{sink_ptr1}, min_log_level).AddSink(std::unique_ptr<ISink>{sink_ptr2}, min_log_level);
  LoggingManager manager{std::unique_ptr<ISink>(sink), min_log_level, false, InstanceType::Temporal};

  auto logger = manager.CreateLogger(logid);

  LOGS_CATEGORY(*logger, WARNING, "ArbitraryCategory") << "Warning";
}

/// <summary>
/// Tests that removing a sink of a specific type correctly updates the composite sink.
/// </summary>
TEST(LoggingTests, TestRemoveSink) {
  CompositeSink sink;
  MockSink* mock_sink1 = new MockSink();
  MockEtwSink* mock_sink2 = new MockEtwSink();
  sink.AddSink(std::unique_ptr<ISink>(mock_sink1), Severity::kWARNING);
  sink.AddSink(std::unique_ptr<ISink>(mock_sink2), Severity::kERROR);

  // Set expectations that no SendImpl will be called on the removed sink
  EXPECT_CALL(*mock_sink1, SendImpl(testing::_, testing::_, testing::_)).Times(0);

  // Remove the sink and check severity update
  auto new_severity = sink.RemoveSink(SinkType::EtwSink);
  EXPECT_EQ(new_severity, Severity::kWARNING);  // assuming mock_sink2 had SpecificType and was removed

  // Verify that sink2 is still in the composite
  EXPECT_TRUE(sink.HasType(SinkType::BaseSink));
}

/// <summary>
/// Tests the HasOnlyOneSink method to ensure it correctly identifies when one sink is left.
/// </summary>
TEST(LoggingTests, TestHasOnlyOneSink) {
  CompositeSink sink;
  sink.AddSink(std::unique_ptr<ISink>(new MockEtwSink()), Severity::kWARNING);
  sink.AddSink(std::unique_ptr<ISink>(new MockSink()), Severity::kERROR);

  EXPECT_FALSE(sink.HasOnlyOneSink());

  sink.RemoveSink(SinkType::EtwSink);
  EXPECT_TRUE(sink.HasOnlyOneSink());

  sink.RemoveSink(SinkType::BaseSink);  // Remove the last one
  EXPECT_FALSE(sink.HasOnlyOneSink());
}

/// <summary>
/// Tests the GetRemoveSingleSink method to ensure it returns the last sink and empties the composite sink.
/// </summary>
TEST(LoggingTests, TestGetRemoveSingleSink) {
  CompositeSink sink;
  auto* single_mock_sink = new MockSink();
  sink.AddSink(std::unique_ptr<ISink>(single_mock_sink), Severity::kWARNING);

  // Check we have one sink
  EXPECT_TRUE(sink.HasOnlyOneSink());

  // Get and remove the single sink
  auto removed_sink = sink.GetRemoveSingleSink();
  EXPECT_EQ(removed_sink.get(), single_mock_sink);  // Check it's the same sink
  EXPECT_FALSE(sink.HasOnlyOneSink());              // Should be empty now
}
