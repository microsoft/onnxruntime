// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef USE_ONNXRUNTIME_DLL
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <google/protobuf/message_lite.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <absl/base/config.h>
#include "gtest/gtest.h"

// Manually initialize the Ort API object for every test.
// This is needed to allow some tests to mock the C API object.
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/composite_sink.h"
#include "test/common/logging/helpers.h"

/**
 * Creates a mock sink that expects a configurable number of logging calls. Validates the inputs and outputs
 * of each call.
 *
 * \param log_id The ID of the logger being tested.
 * \param num_calls The number of expected logging calls.
 * \param file The filepath from which all logs are made. Typically __FILE__.
 * \param func The name of the function from which all logs are made. Typically __FUNCTION__.
 * \param line A reference to the variable that will contain each call's line number. The caller is expected to
 *             update this value with the correct line number before each logging call.
 * \return A unique_ptr containing the mock sink.
 */
static std::unique_ptr<onnxruntime::logging::ISink> SetupMockSinkCalls(const std::string& log_id,
                                                                       int num_calls, const char* file,
                                                                       const char* func, const int& line) {
  using namespace onnxruntime;
  auto mock_sink = std::make_unique<MockSink>();

  // Matchers that check the input arguments to the MockSink::SendImpl() method.
  const auto ignore_timestamp = testing::_;
  const auto log_id_matcher = testing::HasSubstr(log_id);
  const auto line_num_matcher = testing::Field(&CodeLocation::line_num, testing::Eq(std::ref(line)));
  const auto file_path_matcher = testing::Field(&CodeLocation::file_and_path, testing::HasSubstr(file));
  const auto func_name_matcher = testing::Field(&CodeLocation::function, testing::HasSubstr(func));
  const auto source_location_matcher = testing::Property(&logging::Capture::Location,
                                                         testing::AllOf(line_num_matcher,
                                                                        file_path_matcher,
                                                                        func_name_matcher));

  // Expect calls to the mock sink's SendImpl() method.
  EXPECT_CALL(*mock_sink, SendImpl(ignore_timestamp, log_id_matcher, source_location_matcher))
      .Times(num_calls)              // Called at most num_calls times
      .WillRepeatedly(PrintArgs());  // The PrintArgs() action will generate the expected stdout output for each call.
                                     // See test/common/logging/helpers.h for the PrintArgs() implementation.

  return mock_sink;
}

// Test fixture that initializes the actual global Ort API object before testing.
class RealCAPITestsFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    Ort::InitApi();
  }

  void TearDown() override {
    Ort::InitApi(nullptr);
  }
};

// Mock version of OrtStatus that is used when mocking C APIs.
struct MockOrtStatus {
  OrtErrorCode code;
  const char* msg;
};

// Test fixture that does not initialize the global Ort API object. Used by tests
// that mock the C API.
class MockCAPITestsFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clear the Ort API object.
    // Individual tests must create a mock API and set it via Ort::InitApi(&mock_ort_api)
    Ort::InitApi(nullptr);
  }

  void TearDown() override {
    Ort::InitApi(nullptr);
  }

 public:
  static MockOrtStatus mock_status_;  // Used by mocked C APIs that need to return a status.
};

MockOrtStatus MockCAPITestsFixture::mock_status_{OrtErrorCode::ORT_FAIL, "Mock failure"};

TEST_F(RealCAPITestsFixture, CCppApiLoggerGetLoggingSeverityLevel) {
  // Tests the output of the C and C++ Logger_GetLoggingSeverityLevel APIs.
  // Creates a logger with a minimum severity set to WARNING and uses the C and C++ APIs
  // to get the expected logger severity.

  const std::string log_id = "aaabbbccc";
  const onnxruntime::logging::Severity min_severity = onnxruntime::logging::Severity::kWARNING;

  onnxruntime::logging::LoggingManager manager{std::make_unique<MockSink>(), min_severity, false,
                                               onnxruntime::logging::LoggingManager::InstanceType::Temporal};

  std::unique_ptr<onnxruntime::logging::Logger> logger = manager.CreateLogger(log_id);
  const OrtLogger* c_ort_logger = reinterpret_cast<const OrtLogger*>(logger.get());

  // Test OrtApi::Logger_GetLoggingSeverityLevel
  {
    OrtLoggingLevel logger_severity{};
    Ort::ThrowOnError(Ort::GetApi().Logger_GetLoggingSeverityLevel(c_ort_logger, &logger_severity));
    ASSERT_EQ(logger_severity, static_cast<OrtLoggingLevel>(min_severity));
  }

  // Test Ort::Logger::GetLoggingSeverityLevel
  {
    Ort::Logger cpp_logger{c_ort_logger};
    OrtLoggingLevel logger_severity = cpp_logger.GetLoggingSeverityLevel();
    ASSERT_EQ(logger_severity, static_cast<OrtLoggingLevel>(min_severity));
  }
}

TEST_F(RealCAPITestsFixture, CApiLoggerLogMessage) {
  // Tests the output and filtering of the OrtApi::Logger_LogMessage API.
  // The C API OrtApi::Logger_LogMessage is called three times.
  // The first two calls go through, but the last call is filtered out due to an insufficient severity.

  const std::string log_id = "0xdeadbeef";
  int line_num = 0;
  const onnxruntime::logging::Severity min_severity = onnxruntime::logging::Severity::kWARNING;

  // Setup a mock sink that expects 2 log calls.
  auto mock_sink = SetupMockSinkCalls(log_id, 2, __FILE__, __FUNCTION__, line_num);

  onnxruntime::logging::LoggingManager manager{std::move(mock_sink), min_severity, false,
                                               onnxruntime::logging::LoggingManager::InstanceType::Temporal};

  std::unique_ptr<onnxruntime::logging::Logger> logger = manager.CreateLogger(log_id);
  const OrtLogger* c_ort_logger = reinterpret_cast<const OrtLogger*>(logger.get());

  // Test OrtApi::Logger_GetLoggingSeverityLevel
  OrtLoggingLevel logger_severity{};
  Ort::ThrowOnError(Ort::GetApi().Logger_GetLoggingSeverityLevel(c_ort_logger, &logger_severity));
  ASSERT_EQ(logger_severity, static_cast<OrtLoggingLevel>(min_severity));

  // Test 2 calls to OrtApi::Logger_LogMessage
  line_num = __LINE__ + 1;
  Ort::ThrowOnError(Ort::GetApi().Logger_LogMessage(c_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Hello",
                                                    ORT_FILE, line_num, static_cast<const char*>(__FUNCTION__)));

  line_num = __LINE__ + 1;
  Ort::ThrowOnError(Ort::GetApi().Logger_LogMessage(c_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Hello2",
                                                    ORT_FILE, line_num, static_cast<const char*>(__FUNCTION__)));

  // The following call to OrtApi::Logger_LogMessage should be filtered out due to insufficient severity.
  line_num = __LINE__ + 1;
  Ort::ThrowOnError(Ort::GetApi().Logger_LogMessage(c_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "Ignored",
                                                    ORT_FILE, line_num, static_cast<const char*>(__FUNCTION__)));
}

// The code below where it tests for formatting error generates an out-of-bound memory access. Therefore we disable it
// when memory sanitizer is enabled.
#ifdef ABSL_HAVE_ADDRESS_SANITIZER
TEST_F(RealCAPITestsFixture, DISABLED_CppApiORTCXXLOG) {
#else
TEST_F(RealCAPITestsFixture, CppApiORTCXXLOG) {
#endif
  // Tests the output and filtering of the ORT_CXX_LOG and ORT_CXX_LOG_NOEXCEPT macros in the C++ API.
  // The first two calls go through, but the last two calls are filtered out due to an insufficient severity.

  const std::string log_id = "ORT_CXX_LOG";
  int line_num = 0;
  const onnxruntime::logging::Severity min_severity = onnxruntime::logging::Severity::kWARNING;

  // Setup a mock sink that expects 2 log calls.
  auto mock_sink = SetupMockSinkCalls(log_id, 2, __FILE__, __FUNCTION__, line_num);

  onnxruntime::logging::LoggingManager manager{std::move(mock_sink), min_severity, false,
                                               onnxruntime::logging::LoggingManager::InstanceType::Temporal};

  std::unique_ptr<onnxruntime::logging::Logger> logger = manager.CreateLogger(log_id);
  Ort::Logger cpp_ort_logger{reinterpret_cast<const OrtLogger*>(logger.get())};

  // Test Ort::Logger::GetLoggingSeverityLevel
  OrtLoggingLevel logger_severity = cpp_ort_logger.GetLoggingSeverityLevel();
  ASSERT_EQ(logger_severity, static_cast<OrtLoggingLevel>(min_severity));

  // Test calls to ORT_CXX_LOG and ORT_CXX_LOG_NOEXCEPT
  line_num = __LINE__ + 1;
  ORT_CXX_LOG(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Hello");

  line_num = __LINE__ + 1;
  ORT_CXX_LOG_NOEXCEPT(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Hello2");

  // The following calls should be filtered out due to insufficient severity.
  line_num = __LINE__ + 1;
  ORT_CXX_LOG(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "Ignored");

  line_num = __LINE__ + 1;
  ORT_CXX_LOG_NOEXCEPT(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "Ignored2");
}

#ifdef ABSL_HAVE_ADDRESS_SANITIZER
TEST_F(RealCAPITestsFixture, DISABLED_CppApiORTCXXLOGF) {
#else
TEST_F(RealCAPITestsFixture, CppApiORTCXXLOGF) {
#endif
  // Tests the output and filtering of the ORT_CXX_LOGF and ORT_CXX_LOGF_NOEXCEPT macros in the C++ API.
  // The first set of calls go through. The next set of calls are filtered out due to an insufficient severity.
  // The last calls have a formatting error and we expect an exception depending on which macro is used.

  const std::string log_id = "ORT_CXX_LOGF";
  int line_num = 0;
  const onnxruntime::logging::Severity min_severity = onnxruntime::logging::Severity::kWARNING;

  // Setup a mock sink that expects 5 log calls.
  auto mock_sink = SetupMockSinkCalls(log_id, 5, __FILE__, __FUNCTION__, line_num);

  onnxruntime::logging::LoggingManager manager{std::move(mock_sink), min_severity, false,
                                               onnxruntime::logging::LoggingManager::InstanceType::Temporal};

  std::unique_ptr<onnxruntime::logging::Logger> logger = manager.CreateLogger(log_id);
  Ort::Logger cpp_ort_logger{reinterpret_cast<const OrtLogger*>(logger.get())};

  // Test Ort::Logger::GetLoggingSeverityLevel
  OrtLoggingLevel logger_severity = cpp_ort_logger.GetLoggingSeverityLevel();
  ASSERT_EQ(logger_severity, static_cast<OrtLoggingLevel>(min_severity));

  //
  // Test successful calls to ORT_CXX_LOGF and ORT_CXX_LOGF_NOEXCEPT.
  //

  // No variadic args
  line_num = __LINE__ + 1;
  ORT_CXX_LOGF(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Hello");

  // Two variadic lvalue args
  line_num = __LINE__ + 1;
  ORT_CXX_LOGF(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Hello %s %d", log_id.c_str(), line_num);

  // No variadic args
  line_num = __LINE__ + 1;
  ORT_CXX_LOGF_NOEXCEPT(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Hello");

  // Two literal variadic args
  line_num = __LINE__ + 1;
  ORT_CXX_LOGF_NOEXCEPT(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Hello %s %f", "world", 1.2f);

  // Test long message (exercise different control flow path that allocates heap memory).
  std::string long_str(2048, 'a');
  line_num = __LINE__ + 1;
  ORT_CXX_LOGF(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Long string: %s", long_str.c_str());

  //
  // The following calls should be filtered out due to insufficient severity.
  //

  line_num = __LINE__ + 1;
  ORT_CXX_LOGF(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "Ignored");

  line_num = __LINE__ + 1;
  ORT_CXX_LOGF_NOEXCEPT(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "Ignored %d", line_num);
}

TEST_F(MockCAPITestsFixture, CppLogMacroBypassCApiCall) {
  // Tests the ability of the ORT_CXX_LOG* macros to completely bypass calling the C logging APIs.
  // We do this by creating a mock OrtApi struct that is hardcoded to return a specific error when Logger_LogMessage
  // is called.

  const std::string log_id = "MOCK_API";
  const onnxruntime::logging::Severity min_severity = onnxruntime::logging::Severity::kWARNING;

  onnxruntime::logging::LoggingManager manager{std::make_unique<MockSink>(), min_severity, false,
                                               onnxruntime::logging::LoggingManager::InstanceType::Temporal};
  std::unique_ptr<onnxruntime::logging::Logger> logger = manager.CreateLogger(log_id);

  // Create a mock OrtApi.
  OrtApi mock_ort_api{};

  // Set OrtApi::GetErrorCode to get MockOrtStatus::code.
  mock_ort_api.GetErrorCode = [](const OrtStatus* status) noexcept -> OrtErrorCode {
    const auto* actual_status = reinterpret_cast<const MockOrtStatus*>(status);
    return actual_status->code;
  };

  // Set OrtApi::GetErrorCode to get MockOrtStatus::msg.
  mock_ort_api.GetErrorMessage = [](const OrtStatus* status) noexcept -> const char* {
    const auto* actual_status = reinterpret_cast<const MockOrtStatus*>(status);
    return actual_status->msg;
  };

  // OrtApi::ReleaseStatus is a no-op in this mocking environment.
  mock_ort_api.ReleaseStatus = [](OrtStatus* /* status */) noexcept -> void {
    // Do nothing. We're always using a reinterpreted pointer to MockCAPITestsFixture::mock_status_.
  };

  // OrtApi::Logger_GetLoggingSeverityLevel needs to return the logger's severity.
  mock_ort_api.Logger_GetLoggingSeverityLevel = [](const OrtLogger* logger,
                                                   OrtLoggingLevel* out) noexcept -> OrtStatus* {
    const auto& actual_logger = *reinterpret_cast<const onnxruntime::logging::Logger*>(logger);
    *out = static_cast<OrtLoggingLevel>(actual_logger.GetSeverity());
    return nullptr;
  };

  // Hardcode OrtApi::Logger_LogMessage to always return a "Mock failure" failure status when called.
  // This will allows to detect if the C API is actually called or bypassed.
  mock_ort_api.Logger_LogMessage = [](const OrtLogger* /* logger */, OrtLoggingLevel /* log_severity_level */,
                                      const char* /* message */, const ORTCHAR_T* /* file_path */,
                                      int /* line_number */, const char* /* func_name */) noexcept -> OrtStatus* {
    return reinterpret_cast<OrtStatus*>(&MockCAPITestsFixture::mock_status_);
  };

  // Set the mock OrtApi object for use in the C++ API.
  Ort::InitApi(&mock_ort_api);

  Ort::Logger cpp_ort_logger{logger.get()->ToExternal()};

  // The ORT_CXX_LOG* macros will bypass calling the C API if the cached severity level in the Ort::Logger exceeds the
  // message's severity level. Thus, the following two macro calls will not call the mock C API, and nothing
  // will happen.
  ORT_CXX_LOG(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "Should not call mock C API!");
  ORT_CXX_LOGF(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "Should not call mock C API!");

  // Messages with the ERROR severity level exceed the cached severity value in Ort::Logger. Therefore, the C API
  // will be called, which will throw a "Mock failure" exception.
  try {
    ORT_CXX_LOG(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Should call mock C API, which fails");
    FAIL();
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), MockCAPITestsFixture::mock_status_.code);
    ASSERT_THAT(excpt.what(), testing::HasSubstr(MockCAPITestsFixture::mock_status_.msg));
  }

  // Same as above, but with ORT_CXX_LOGF.
  try {
    ORT_CXX_LOGF(cpp_ort_logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Should call mock C API, which fails");
    FAIL();
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), MockCAPITestsFixture::mock_status_.code);
    ASSERT_THAT(excpt.what(), testing::HasSubstr(MockCAPITestsFixture::mock_status_.msg));
  }
}

#ifndef ORT_MINIMAL_BUILD
TEST(DefaultLoggerCompositeSinkTests, PreexistingEtwPreservesCallbackAndPlatformThresholds) {
  using namespace onnxruntime::logging;
  const std::string log_id = "Default";

  // Case 1: ETW enabled at WARNING before environment creation (matching EnhanceSinkWithEtw),
  // followed by lowering default logger severity to VERBOSE (as done by set_default_logger_callback(..., severity=0)).
  {
    std::vector<Severity> base_severities;
    std::vector<Severity> etw_severities;

    auto base_sink = std::make_unique<MockSink>();
    auto etw_sink = std::make_unique<MockEtwSink>();
    ON_CALL(*base_sink, SendImpl(testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([&](const Timestamp&, const std::string&, const Capture& msg) {
          base_severities.push_back(msg.Severity());
        }));
    ON_CALL(*etw_sink, SendImpl(testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([&](const Timestamp&, const std::string&, const Capture& msg) {
          etw_severities.push_back(msg.Severity());
        }));
    EXPECT_CALL(*base_sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());
    EXPECT_CALL(*etw_sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());

    auto composite = std::make_unique<CompositeSink>();
    composite->AddSink(std::move(base_sink), Severity::kWARNING)
        .AddSink(std::move(etw_sink), Severity::kWARNING);

    LoggingManager manager{std::move(composite), Severity::kWARNING, false,
                           LoggingManager::InstanceType::Default, &log_id};

    LoggingManager::SetDefaultLoggerSeverity(Severity::kVERBOSE);
    LOGS_DEFAULT(VERBOSE) << "v";
    LOGS_DEFAULT(INFO) << "i";
    LOGS_DEFAULT(WARNING) << "w";

    EXPECT_EQ(base_severities, (std::vector<Severity>{Severity::kVERBOSE, Severity::kINFO, Severity::kWARNING}));
    EXPECT_EQ(etw_severities, (std::vector<Severity>{Severity::kWARNING}));
  }

  // Case 2: ETW enabled at VERBOSE before environment creation. Platform sink stays at WARNING before registration,
  // and raising default logger severity to ERROR keeps ETW receiving VERBOSE while gating BaseSink at ERROR.
  {
    std::vector<Severity> base_severities;
    std::vector<Severity> etw_severities;

    auto base_sink = std::make_unique<MockSink>();
    auto etw_sink = std::make_unique<MockEtwSink>();
    ON_CALL(*base_sink, SendImpl(testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([&](const Timestamp&, const std::string&, const Capture& msg) {
          base_severities.push_back(msg.Severity());
        }));
    ON_CALL(*etw_sink, SendImpl(testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([&](const Timestamp&, const std::string&, const Capture& msg) {
          etw_severities.push_back(msg.Severity());
        }));
    EXPECT_CALL(*base_sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());
    EXPECT_CALL(*etw_sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());

    auto composite = std::make_unique<CompositeSink>();
    composite->AddSink(std::move(base_sink), Severity::kWARNING)
        .AddSink(std::move(etw_sink), Severity::kVERBOSE);

    LoggingManager manager{std::move(composite), std::min(Severity::kWARNING, Severity::kVERBOSE), false,
                           LoggingManager::InstanceType::Default, &log_id};

    LOGS_DEFAULT(VERBOSE) << "v";
    LOGS_DEFAULT(INFO) << "i";
    LOGS_DEFAULT(WARNING) << "w";
    EXPECT_EQ(base_severities, (std::vector<Severity>{Severity::kWARNING}));
    EXPECT_EQ(etw_severities, (std::vector<Severity>{Severity::kVERBOSE, Severity::kINFO, Severity::kWARNING}));

    base_severities.clear();
    etw_severities.clear();
    LoggingManager::SetDefaultLoggerSeverity(Severity::kERROR);
    LOGS_DEFAULT(VERBOSE) << "v";
    LOGS_DEFAULT(INFO) << "i";
    LOGS_DEFAULT(WARNING) << "w";
    LOGS_DEFAULT(ERROR) << "e";
    EXPECT_EQ(base_severities, (std::vector<Severity>{Severity::kERROR}));
    EXPECT_EQ(etw_severities,
              (std::vector<Severity>{Severity::kVERBOSE, Severity::kINFO, Severity::kWARNING, Severity::kERROR}));
  }
}

TEST(DefaultLoggerCompositeSinkTests, PreexistingEtwPreservesExistingAndExplicitLoggerSeverities) {
  using namespace onnxruntime::logging;
  const std::string log_id = "Default";

  // Verify across ETW severities that raising the default logger severity (as set_default_logger_callback(..., 4)
  // does) does not suppress records accepted by existing session loggers or explicit session/run severity overrides,
  // and delivers the exact same BaseSink records with ETW present as after removing ETW.
  for (const Severity etw_severity : {Severity::kVERBOSE, Severity::kWARNING, Severity::kERROR}) {
    std::vector<std::pair<std::string, Severity>> base_records_with_etw;
    std::vector<std::pair<std::string, Severity>> base_records_without_etw;
    std::vector<std::pair<std::string, Severity>> etw_records;

    bool collecting_with_etw = true;
    auto base_sink = std::make_unique<MockSink>();
    auto etw_sink = std::make_unique<MockEtwSink>();
    ON_CALL(*base_sink, SendImpl(testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([&](const Timestamp&, const std::string& id, const Capture& msg) {
          auto& target = collecting_with_etw ? base_records_with_etw : base_records_without_etw;
          target.emplace_back(id, msg.Severity());
        }));
    ON_CALL(*etw_sink, SendImpl(testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([&](const Timestamp&, const std::string& id, const Capture& msg) {
          etw_records.emplace_back(id, msg.Severity());
        }));
    EXPECT_CALL(*base_sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());
    EXPECT_CALL(*etw_sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());

    auto composite = std::make_unique<CompositeSink>();
    composite->AddSink(std::move(base_sink), Severity::kWARNING)
        .AddSink(std::move(etw_sink), etw_severity);

    LoggingManager manager{std::move(composite), std::min(Severity::kWARNING, etw_severity), false,
                           LoggingManager::InstanceType::Default, &log_id};

    // Existing session loggers created before default severity changes to FATAL:
    // 1) manager default overload, and 2) InferenceSession snapshot of DefaultLogger().GetSeverity().
    auto existing_manager_logger = manager.CreateLogger("existing_manager");
    auto existing_session_logger =
        manager.CreateLogger("existing_session", LoggingManager::DefaultLogger().GetSeverity(), false);
    EXPECT_EQ(existing_manager_logger->GetSeverity(), Severity::kWARNING);
    EXPECT_EQ(existing_session_logger->GetSeverity(), Severity::kWARNING);

    // Raise default logger severity to FATAL (matching set_default_logger_callback(..., severity=4)).
    LoggingManager::SetDefaultLoggerSeverity(Severity::kFATAL);
    EXPECT_EQ(LoggingManager::DefaultLogger().GetSeverity(), Severity::kFATAL);

    // Post-update loggers: default-configured sessions get FATAL, explicit overrides keep their requested level.
    auto post_default_manager_logger = manager.CreateLogger("post_default_manager");
    auto post_default_session_logger =
        manager.CreateLogger("post_default_session", LoggingManager::DefaultLogger().GetSeverity(), false);
    auto explicit_verbose_session_logger = manager.CreateLogger("explicit_verbose_session", Severity::kVERBOSE, false);

    const auto emit_all = [&]() {
      LOGS(*existing_manager_logger, INFO) << "i";
      LOGS(*existing_manager_logger, WARNING) << "w";

      LOGS(*existing_session_logger, INFO) << "i";
      LOGS(*existing_session_logger, WARNING) << "w";

      LOGS_DEFAULT(WARNING) << "w";
      LOGS(*post_default_manager_logger, WARNING) << "w";
      LOGS(*post_default_session_logger, WARNING) << "w";

      LOGS(*explicit_verbose_session_logger, VERBOSE) << "v";
      LOGS(*explicit_verbose_session_logger, INFO) << "i";
      LOGS(*explicit_verbose_session_logger, WARNING) << "w";
    };

    emit_all();

    const std::vector<std::pair<std::string, Severity>> expected_base_records{
        {"existing_manager", Severity::kWARNING},
        {"existing_session", Severity::kWARNING},
        {"explicit_verbose_session", Severity::kVERBOSE},
        {"explicit_verbose_session", Severity::kINFO},
        {"explicit_verbose_session", Severity::kWARNING},
    };
    EXPECT_EQ(base_records_with_etw, expected_base_records);

    for (const auto& [id, sev] : etw_records) {
      (void)id;
      EXPECT_GE(sev, etw_severity);
    }
    if (etw_severity <= Severity::kWARNING) {
      EXPECT_FALSE(etw_records.empty());
    } else {
      EXPECT_TRUE(etw_records.empty());
    }

    // Removing ETW must leave BaseSink delivery for every logger unchanged.
    collecting_with_etw = false;
    manager.RemoveSink(SinkType::EtwSink);
    emit_all();
    EXPECT_EQ(base_records_without_etw, expected_base_records);
  }
}
#endif  // !ORT_MINIMAL_BUILD

TEST(DefaultLoggerCompositeSinkTests, ConcurrentSetDefaultLoggerSeverityAndCreateLogger) {
  using namespace onnxruntime::logging;
  const std::string log_id = "Default";

  std::mutex records_mutex;
  int base_count = 0;

  auto base_sink = std::make_unique<MockSink>();
  auto etw_sink = std::make_unique<MockEtwSink>();
  ON_CALL(*base_sink, SendImpl(testing::_, testing::_, testing::_))
      .WillByDefault(testing::Invoke([&](const Timestamp&, const std::string&, const Capture&) {
        std::lock_guard<std::mutex> lock(records_mutex);
        ++base_count;
      }));
  ON_CALL(*etw_sink, SendImpl(testing::_, testing::_, testing::_))
      .WillByDefault(testing::Invoke([](const Timestamp&, const std::string&, const Capture&) {}));
  EXPECT_CALL(*base_sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());
  EXPECT_CALL(*etw_sink, SendImpl(testing::_, testing::_, testing::_)).Times(testing::AnyNumber());

  auto composite = std::make_unique<CompositeSink>();
  composite->AddSink(std::move(base_sink), Severity::kVERBOSE)
      .AddSink(std::move(etw_sink), Severity::kWARNING);

  LoggingManager manager{std::move(composite), Severity::kVERBOSE, false,
                         LoggingManager::InstanceType::Default, &log_id};

  constexpr int kNumWriters = 2;
  constexpr int kNumReaders = 4;
  constexpr int kIterations = 1000;
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> invalid_severity_observed{false};

  std::vector<std::thread> threads;
  threads.reserve(kNumWriters + kNumReaders);

  for (int w = 0; w < kNumWriters; ++w) {
    threads.emplace_back([w, &ready, &start]() {
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kIterations; ++i) {
        const Severity sev = ((i + w) % 2 == 0) ? Severity::kVERBOSE : Severity::kFATAL;
        LoggingManager::SetDefaultLoggerSeverity(sev);
      }
    });
  }

  for (int r = 0; r < kNumReaders; ++r) {
    threads.emplace_back([r, &manager, &ready, &start, &invalid_severity_observed]() {
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kIterations; ++i) {
        auto logger = manager.CreateLogger("reader_" + std::to_string(r));
        const Severity sev = logger->GetSeverity();
        if (sev != Severity::kVERBOSE && sev != Severity::kFATAL) {
          invalid_severity_observed.store(true, std::memory_order_relaxed);
        }
        LOGS(*logger, INFO) << "probe";
        LOGS_DEFAULT(INFO) << "default_probe";
      }
    });
  }

  while (ready.load(std::memory_order_acquire) < kNumWriters + kNumReaders) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_FALSE(invalid_severity_observed.load(std::memory_order_relaxed));
  EXPECT_GE(base_count, 0);
}

#define TEST_MAIN main

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_SIMULATOR || TARGET_OS_IOS

#undef TEST_MAIN
#define TEST_MAIN main_no_link_  // there is a UI test app for iOS.

// iOS tests require ortenv_setup() and ortenv_teardown() to be defined.
// See onnxruntime/test/xctest/xcgtest.mm
extern "C" void ortenv_setup() {
  // Do nothing. These logging tests do not require an env to be setup initially.
}

extern "C" void ortenv_teardown() {
  // Do nothing.
}

#endif  // TARGET_OS_SIMULATOR || TARGET_OS_IOS
#endif  // defined(__APPLE__)

int TEST_MAIN(int argc, char** argv) {
  int status = 0;
  ORT_TRY {
    ::testing::InitGoogleTest(&argc, argv);
    status = RUN_ALL_TESTS();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      std::cerr << ex.what();
      status = -1;
    });
  }

#ifndef USE_ONNXRUNTIME_DLL
  // make memory leak checker happy
  ::google::protobuf::ShutdownProtobufLibrary();
#endif
  return status;
}
