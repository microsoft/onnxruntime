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

/// <summary>
/// Tests that CompositeSink with an ETW sink honors each originating logger's configured severity on BaseSink
/// while keeping independent ETW filtering, and preserves per-sink thresholds when no ETW sink is present.
/// </summary>
TEST(LoggingTests, TestCompositeSinkPerLoggerAndEtwSeverity) {
#ifndef ORT_MINIMAL_BUILD
  {
    std::vector<std::pair<std::string, Severity>> base_records;
    std::vector<std::pair<std::string, Severity>> etw_records;

    auto* base_sink = new MockSink();
    auto* etw_sink = new MockEtwSink();
    ON_CALL(*base_sink, SendImpl(testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([&](const Timestamp&, const std::string& id, const Capture& msg) {
          base_records.emplace_back(id, msg.Severity());
        }));
    ON_CALL(*etw_sink, SendImpl(testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([&](const Timestamp&, const std::string& id, const Capture& msg) {
          etw_records.emplace_back(id, msg.Severity());
        }));
    EXPECT_CALL(*base_sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());
    EXPECT_CALL(*etw_sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());

    auto composite = std::make_unique<CompositeSink>();
    composite->AddSink(std::unique_ptr<ISink>(base_sink), Severity::kWARNING)
        .AddSink(std::unique_ptr<ISink>(etw_sink), Severity::kINFO);

    LoggingManager manager{std::move(composite), std::min(Severity::kWARNING, Severity::kINFO), false,
                           InstanceType::Temporal};
    auto default_logger = manager.CreateLogger("default_logger");
    auto verbose_logger = manager.CreateLogger("verbose_logger", Severity::kVERBOSE, false);
    auto fatal_logger = manager.CreateLogger("fatal_logger", Severity::kFATAL, false);

    LOGS(*default_logger, VERBOSE) << "v";
    LOGS(*default_logger, INFO) << "i";
    LOGS(*default_logger, WARNING) << "w";

    LOGS(*verbose_logger, VERBOSE) << "v";
    LOGS(*verbose_logger, INFO) << "i";
    LOGS(*verbose_logger, WARNING) << "w";

    LOGS(*fatal_logger, INFO) << "i";
    LOGS(*fatal_logger, WARNING) << "w";

    EXPECT_EQ(base_records,
              (std::vector<std::pair<std::string, Severity>>{
                  {"default_logger", Severity::kWARNING},
                  {"verbose_logger", Severity::kVERBOSE},
                  {"verbose_logger", Severity::kINFO},
                  {"verbose_logger", Severity::kWARNING},
              }));
    EXPECT_EQ(etw_records,
              (std::vector<std::pair<std::string, Severity>>{
                  {"default_logger", Severity::kINFO},
                  {"default_logger", Severity::kWARNING},
                  {"verbose_logger", Severity::kINFO},
                  {"verbose_logger", Severity::kWARNING},
                  {"fatal_logger", Severity::kINFO},
                  {"fatal_logger", Severity::kWARNING},
              }));
  }
#endif  // !ORT_MINIMAL_BUILD

  // Standalone CompositeSink without ETW preserves per-sink thresholds.
  {
    std::vector<Severity> warn_sink_records;
    std::vector<Severity> error_sink_records;

    auto* warn_sink = new MockSink();
    auto* error_sink = new MockSink();
    ON_CALL(*warn_sink, SendImpl(testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([&](const Timestamp&, const std::string&, const Capture& msg) {
          warn_sink_records.push_back(msg.Severity());
        }));
    ON_CALL(*error_sink, SendImpl(testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([&](const Timestamp&, const std::string&, const Capture& msg) {
          error_sink_records.push_back(msg.Severity());
        }));
    EXPECT_CALL(*warn_sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());
    EXPECT_CALL(*error_sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());

    auto composite = std::make_unique<CompositeSink>();
    composite->AddSink(std::unique_ptr<ISink>(warn_sink), Severity::kWARNING)
        .AddSink(std::unique_ptr<ISink>(error_sink), Severity::kERROR);

    LoggingManager manager{std::move(composite), Severity::kWARNING, false, InstanceType::Temporal};
    auto logger = manager.CreateLogger("standalone");
    LOGS(*logger, WARNING) << "w";
    LOGS(*logger, ERROR) << "e";

    EXPECT_EQ(warn_sink_records, (std::vector<Severity>{Severity::kWARNING, Severity::kERROR}));
    EXPECT_EQ(error_sink_records, (std::vector<Severity>{Severity::kERROR}));
  }
}

#ifndef ORT_MINIMAL_BUILD
/// <summary>
/// Tests that SetDefaultLoggerSeverity updates the default logger without clobbering existing/explicit loggers
/// while preserving independent ETW filtering.
/// </summary>
TEST(LoggingTests, TestDefaultLoggerSeverityWithEtwSink) {
  auto* default_manager = LoggingManager::GetDefaultInstance();
  ASSERT_NE(default_manager, nullptr);

  const Severity orig_severity = LoggingManager::DefaultLogger().GetSeverity();
  auto cleanup = gsl::finally([&]() {
    default_manager->RemoveSink(SinkType::EtwSink);
    LoggingManager::SetDefaultLoggerSeverity(orig_severity);
  });

  LoggingManager::SetDefaultLoggerSeverity(Severity::kWARNING);
  auto existing_warning_logger = default_manager->CreateLogger("existing_warning");
  EXPECT_EQ(existing_warning_logger->GetSeverity(), Severity::kWARNING);

  std::vector<Severity> etw_severities;
  ASSERT_TRUE(default_manager->AddSinkOfType(
      SinkType::EtwSink,
      [&]() -> std::unique_ptr<ISink> {
        auto sink = std::make_unique<MockEtwSink>();
        ON_CALL(*sink, SendImpl(testing::_, testing::_, testing::_))
            .WillByDefault(testing::Invoke([&](const Timestamp&, const std::string&, const Capture& msg) {
              etw_severities.push_back(msg.Severity());
            }));
        EXPECT_CALL(*sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());
        return sink;
      },
      Severity::kERROR));

  LoggingManager::SetDefaultLoggerSeverity(Severity::kVERBOSE);
  EXPECT_EQ(LoggingManager::DefaultLogger().GetSeverity(), Severity::kVERBOSE);
  LOGS_DEFAULT(VERBOSE) << "verbose";
  LOGS_DEFAULT(INFO) << "info";
  LOGS_DEFAULT(WARNING) << "warning";
  LOGS_DEFAULT(ERROR) << "error";
  EXPECT_EQ(etw_severities, (std::vector<Severity>{Severity::kERROR}));

  default_manager->RemoveSink(SinkType::EtwSink);
  etw_severities.clear();
  ASSERT_TRUE(default_manager->AddSinkOfType(
      SinkType::EtwSink,
      [&]() -> std::unique_ptr<ISink> {
        auto sink = std::make_unique<MockEtwSink>();
        ON_CALL(*sink, SendImpl(testing::_, testing::_, testing::_))
            .WillByDefault(testing::Invoke([&](const Timestamp&, const std::string&, const Capture& msg) {
              etw_severities.push_back(msg.Severity());
            }));
        EXPECT_CALL(*sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());
        return sink;
      },
      Severity::kVERBOSE));

  LoggingManager::SetDefaultLoggerSeverity(Severity::kERROR);
  EXPECT_EQ(LoggingManager::DefaultLogger().GetSeverity(), Severity::kERROR);
  EXPECT_TRUE(LoggingManager::DefaultLogger().OutputIsEnabled(Severity::kVERBOSE, DataType::SYSTEM));
  EXPECT_EQ(existing_warning_logger->GetSeverity(), Severity::kWARNING);
  LOGS_DEFAULT(VERBOSE) << "verbose";
  LOGS_DEFAULT(INFO) << "info";
  LOGS_DEFAULT(WARNING) << "warning";
  LOGS_DEFAULT(ERROR) << "error";
  EXPECT_EQ(etw_severities,
            (std::vector<Severity>{Severity::kVERBOSE, Severity::kINFO, Severity::kWARNING, Severity::kERROR}));
}
#endif  // !ORT_MINIMAL_BUILD
