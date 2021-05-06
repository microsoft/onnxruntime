// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <exception>
#include <functional>
#include <string>

#include "core/common/logging/isink.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"

#include "test/common/logging/helpers.h"

// if we pull in the whole 'testing' namespace we get warnings from date.h as both use '_' in places.
// to avoid that we explicitly pull in the pieces we are using
using testing::Eq;
using testing::Field;
using testing::Ge;
using testing::HasSubstr;
using testing::Property;

namespace onnxruntime {
using namespace logging;
using InstanceType = LoggingManager::InstanceType;

namespace test {

static std::string default_logger_id{"TestFixtureDefaultLogger"};

// class to provide single default instance of LoggingManager for use with macros involving 'DEFAULT'
class LoggingTestsFixture : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    // logger uses kWARNING so we can test filtering of kVERBOSE output,
    // and filters user data so that can also be tested
#if !defined(SKIP_DEFAULT_LOGGER_TESTS)
    const bool filter_user_data = false;
    default_logging_manager_ = std::make_unique<LoggingManager>(
        std::unique_ptr<ISink>{new CLogSink {}}, Severity::kWARNING, filter_user_data,
        InstanceType::Default, &default_logger_id, /*default_max_vlog_level*/ -1);
#endif
  }

  static void TearDownTestCase() {
  }

  // Objects declared here can be used by all tests in the test case for Foo.
  static std::unique_ptr<LoggingManager> default_logging_manager_;
};

std::unique_ptr<LoggingManager> LoggingTestsFixture::default_logging_manager_;

/// <summary>
/// Tests that the ORT_WHERE macro populates all fields correctly.
/// </summary>
TEST_F(LoggingTestsFixture, TestWhereMacro) {
  const std::string logid{"TestWhereMacro"};
  const std::string message{"Testing the WHERE macro."};
  const Severity min_log_level = Severity::kVERBOSE;

  const std::string file = __FILE__;
  const std::string function = __FUNCTION__;
  int log_line = 0;

  std::cout << function << std::endl;

  MockSink* sink_ptr = new MockSink();

  EXPECT_CALL(*sink_ptr, SendImpl(testing::_, HasSubstr(logid),
                                  Property(&Capture::Location,
                                           AllOf(Field(&CodeLocation::line_num, Eq(std::ref(log_line))),
                                                 Field(&CodeLocation::file_and_path, HasSubstr("onnxruntime")),      // path
                                                 Field(&CodeLocation::file_and_path, HasSubstr("logging_test.cc")),  // filename
                                                 Field(&CodeLocation::function, HasSubstr(function))))))
      .WillRepeatedly(PrintArgs());

  LoggingManager manager{std::unique_ptr<ISink>(sink_ptr), min_log_level, false, InstanceType::Temporal};

  std::unique_ptr<Logger> logger = manager.CreateLogger(logid);

  log_line = __LINE__ + 1;
  LOGS(*logger, ERROR) << message;
}

/// <summary>
/// Tests that the logging manager filters based on severity and user data correctly.
/// </summary>
TEST_F(LoggingTestsFixture, TestDefaultFiltering) {
  const std::string logid{"TestDefaultFiltering"};
  const Severity min_log_level = Severity::kWARNING;
  const bool filter_user_data = true;

  MockSink* sink_ptr = new MockSink();

  EXPECT_CALL(*sink_ptr, SendImpl(testing::_, HasSubstr(logid), testing::_))  // Property(&Capture::Severity, Ge(min_log_level))))
      .Times(1)
      .WillRepeatedly(PrintArgs());

  LoggingManager manager{std::unique_ptr<ISink>(sink_ptr), min_log_level, filter_user_data,
                         InstanceType::Temporal};

  auto logger = manager.CreateLogger(logid);

  LOGS(*logger, VERBOSE) << "Filtered by severity";
  LOGS_USER(*logger, ERROR) << "Filtered user data";
  LOGF(*logger, ERROR, "%s", "hello");  // not filtered
  LOGF_USER(*logger, ERROR, "Filtered %s", "user data");

  LOGS_DEFAULT(WARNING) << "Warning";  // not filtered
  LOGS_USER_DEFAULT(ERROR) << "Default logger doesn't filter user data";
  LOGF_DEFAULT(VERBOSE, "Filtered by severity");
  LOGF_USER_DEFAULT(WARNING, "Default logger doesn't filter user data");
}

/// <summary>
/// Tests that the logger filter overrides work correctly.
/// </summary>
TEST_F(LoggingTestsFixture, TestLoggerFiltering) {
  const std::string logid{"TestLoggerFiltering"};
  const bool default_filter_user_data = true;
  const int default_max_vlog_level = -1;

  MockSink* sink_ptr = new MockSink();

  int num_expected_calls = 2;
  if (logging::vlog_enabled) {
    ++num_expected_calls;
  }
  EXPECT_CALL(*sink_ptr, SendImpl(testing::_, HasSubstr(logid), testing::_))  // Property(&Capture::Severity, Ge(min_log_level))))
      .Times(num_expected_calls)
      .WillRepeatedly(PrintArgs());

  LoggingManager manager{std::unique_ptr<ISink>(sink_ptr), Severity::kERROR, default_filter_user_data,
                         InstanceType::Temporal, nullptr, default_max_vlog_level};

  bool filter_user_data = false;
  int max_vlog_level = 2;
  auto logger = manager.CreateLogger(logid, Severity::kVERBOSE, filter_user_data, max_vlog_level);

  LOGS(*logger, VERBOSE) << "VERBOSE enabled in this logger";
  LOGS_USER(*logger, ERROR) << "USER data not filtered in this logger";
  VLOGS(*logger, 2) << "VLOG enabled up to " << max_vlog_level;
}

/// <summary>
/// Tests that the logging manager constructor validates its usage correctly.
/// </summary>
#if !defined(ORT_NO_EXCEPTIONS)
TEST_F(LoggingTestsFixture, TestLoggingManagerCtor) {
  // throw if sink is null
  EXPECT_THROW((LoggingManager{std::unique_ptr<ISink>{nullptr}, Severity::kINFO, false,
                               InstanceType::Temporal}),
               ::onnxruntime::OnnxRuntimeException);

  // can't have two logging managers with InstanceType of Default.
  // this should clash with LoggingTestsFixture::default_logging_manager_
  EXPECT_THROW((LoggingManager{std::unique_ptr<ISink>{new MockSink{}}, Severity::kINFO, false,
                               InstanceType::Default}),
               ::onnxruntime::OnnxRuntimeException);
}
#endif

/// <summary>
/// Tests that the conditional logging macros work correctly.
/// </summary>
TEST_F(LoggingTestsFixture, TestConditionalMacros) {
  const std::string logger_id{"TestConditionalMacros"};
  const Severity min_log_level = Severity::kVERBOSE;
  const bool filter_user_data = false;

  MockSink* sink_ptr = new MockSink();

  // two logging calls that are true using default logger which won't hit our MockSink

  // two logging calls that are true using non-default logger
  EXPECT_CALL(*sink_ptr, SendImpl(testing::_, HasSubstr(logger_id), testing::_))
      .Times(2)
      .WillRepeatedly(PrintArgs());

  LoggingManager manager{std::unique_ptr<ISink>(sink_ptr), min_log_level, filter_user_data,
                         InstanceType::Temporal};

  auto logger = manager.CreateLogger(logger_id);

  // macros to use local logger
  LOGS_IF(logger != nullptr, *logger, INFO) << "Valid logger";                   // true
  LOGF_USER_IF(logger != nullptr, *logger, INFO, "Logger is %p", logger.get());  // true

  // macros to test LoggingTestsFixture::default_logging_manager_
  LOGS_DEFAULT_IF(logger == nullptr, INFO) << "Null logger";                    // false
  LOGS_USER_DEFAULT_IF(logger != nullptr, INFO) << "Valid logger";              // true
  LOGF_DEFAULT_IF(logger == nullptr, INFO, "Logger is %p", logger.get());       // false
  LOGF_USER_DEFAULT_IF(logger != nullptr, INFO, "Logger is %p", logger.get());  // true
}

/// <summary>
/// Tests that the VLOG* macros produce the expected output.
/// Disabled in Release build, so should be no calls to SendImpl in that case.
/// </summary>
TEST_F(LoggingTestsFixture, TestVLog) {
  const std::string logid{"TestVLog"};

  MockSink* sink_ptr = new MockSink();

  // we only get the non-default calls from below in this sink
  EXPECT_CALL(*sink_ptr, SendImpl(testing::_, HasSubstr(logid), testing::_))
#ifndef NDEBUG
      .Times(2)
      .WillRepeatedly(PrintArgs());
#else
      .Times(0);
#endif

  const bool filter_user_data = false;
  LoggingManager manager{std::unique_ptr<ISink>(sink_ptr), Severity::kVERBOSE, filter_user_data, InstanceType::Temporal};

  int max_vlog_level = 2;
  auto logger = manager.CreateLogger(logid, Severity::kVERBOSE, filter_user_data, max_vlog_level);

  // test local logger
  VLOGS(*logger, max_vlog_level) << "Stream";              // logged
  VLOGF(*logger, max_vlog_level + 1, "Printf %d", 1);      // ignored due to level
  VLOGS_USER(*logger, max_vlog_level + 1) << "User data";  // ignored due to level
  VLOGF_USER(*logger, 0, "User Id %d", 1);                 // logged

  // test default logger - just using macros to check they compile as we can't
  // automatically validate the output
  VLOGS_DEFAULT(0) << "Stream";            // ignored due to level
  VLOGF_DEFAULT(10, "Printf %d", 1);       // ignored due to level
  VLOGS_USER_DEFAULT(0) << "User data";    // ignored due to level
  VLOGF_USER_DEFAULT(0, "User Id %d", 1);  // ignored due to level

#ifndef NDEBUG
  // test we can globally disable
  logging::vlog_enabled = false;
  VLOGS(*logger, 0) << "Should be ignored.";  // ignored as disabled
#endif
}

class CTestSink : public OStreamSink {
 public:
  CTestSink(std::ostringstream& stream) : OStreamSink(stream, /*flush*/ true) {
  }
};

TEST_F(LoggingTestsFixture, TestTruncation) {
  const std::string logger_id{"TestTruncation"};
  const Severity min_log_level = Severity::kVERBOSE;
  const bool filter_user_data = false;

  std::ostringstream out;
  auto* sink_ptr = new CTestSink{out};

  LoggingManager manager{std::unique_ptr<ISink>(sink_ptr), min_log_level, filter_user_data,
                         InstanceType::Temporal};

  auto logger = manager.CreateLogger(logger_id);

  // attempt to print string longer than hard-coded 2K buffer limit
  LOGF(*logger, ERROR, "%s", std::string(4096, 'a').c_str());

  EXPECT_THAT(out.str(), HasSubstr("[...truncated...]"));
}

TEST_F(LoggingTestsFixture, TestStreamMacroFromConditionalWithoutCompoundStatement) {
  constexpr const char* logger_id = "TestStreamMacroFromConditionalWithoutCompoundStatement";
  constexpr Severity min_log_level = Severity::kVERBOSE;
  constexpr bool filter_user_data = false;
  constexpr const char* true_message = "true";
  constexpr const char* false_message = "false";

  auto sink = std::make_unique<MockSink>();
  {
    testing::InSequence s{};
    EXPECT_CALL(*sink, SendImpl(testing::_,
                                HasSubstr(logger_id),
                                testing::Property(&Capture::Message, Eq(true_message))))
        .WillOnce(PrintArgs());
    EXPECT_CALL(*sink, SendImpl(testing::_,
                                HasSubstr(logger_id),
                                testing::Property(&Capture::Message, Eq(false_message))))
        .WillOnce(PrintArgs());
  }

  LoggingManager manager{std::move(sink), min_log_level, filter_user_data, InstanceType::Temporal};

  auto logger = manager.CreateLogger(logger_id, min_log_level, filter_user_data);

  auto log_from_conditional_without_compound_statement = [&logger](bool condition) {
    if (condition)
      LOGS(*logger, VERBOSE) << true_message;
    else
      LOGS(*logger, VERBOSE) << false_message;
  };

  log_from_conditional_without_compound_statement(true);
  log_from_conditional_without_compound_statement(false);
}

}  // namespace test
}  // namespace onnxruntime
