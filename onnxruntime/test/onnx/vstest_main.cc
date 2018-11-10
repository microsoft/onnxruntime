// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <CppUnitTest.h>
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>
#include <core/platform/env.h>
#include "core/session/onnxruntime_cxx_api.h"
#include "runner.h"
#include "test_allocator.h"
#include <experimental/filesystem>
#include <filesystem>
//#include "vstest_logger.h"
using onnxruntime::SessionOptionsWrapper;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using std::experimental::filesystem::v1::path;
namespace Microsoft {
namespace VisualStudio {
namespace CppUnitTestFramework {
template <>
std::wstring ToString<>(const EXECUTE_RESULT& q) {
  switch (q) {
    case EXECUTE_RESULT::SUCCESS:
      return L"SUCCESS";
    case EXECUTE_RESULT::UNKNOWN_ERROR:
      return L"UNKNOWN_ERROR";
    case EXECUTE_RESULT::WITH_EXCEPTION:
      return L"WITH_EXCEPTION";
    case EXECUTE_RESULT::RESULT_DIFFERS:
      return L"RESULT_DIFFERS";
    case EXECUTE_RESULT::SHAPE_MISMATCH:
      return L"SHAPE_MISMATCH";
    case EXECUTE_RESULT::TYPE_MISMATCH:
      return L"TYPE_MISMATCH";
    case EXECUTE_RESULT::NOT_SUPPORT:
      return L"NOT_SUPPORT";
    case EXECUTE_RESULT::LOAD_MODEL_FAILED:
      return L"LOAD_MODEL_FAILED";
    case EXECUTE_RESULT::INVALID_GRAPH:
      return L"INVALID_GRAPH";
    case EXECUTE_RESULT::INVALID_ARGUMENT:
      return L"INVALID_ARGUMENT";
    case EXECUTE_RESULT::MODEL_SHAPE_MISMATCH:
      return L"MODEL_SHAPE_MISMATCH";
    case EXECUTE_RESULT::MODEL_TYPE_MISMATCH:
      return L"MODEL_TYPE_MISMATCH";
  }
  return L"UNKNOWN_RETURN_CODE";
}
}  // namespace CppUnitTestFramework
}  // namespace VisualStudio
}  // namespace Microsoft

static void run(ONNXEnv* env, SessionOptionsWrapper& sf, const wchar_t* test_folder) {
  char buf[1024];
  std::vector<EXECUTE_RESULT> res;
  {
    //Current working directory is the one who contains 'onnx_test_runner_vstest.dll'
    //We want debug build and release build share the same test data files, so it should
    //be one level up.
    std::wstring test_folder_full_path(L"..\\models\\");
    test_folder_full_path.append(test_folder);
    path p1(test_folder_full_path);
    std::unique_ptr<ONNXRuntimeAllocator> default_allocator(MockedONNXRuntimeAllocator::Create());
    std::vector<ITestCase*> tests = LoadTests({p1.c_str()}, {}, default_allocator.get());
    Assert::AreEqual((size_t)1, tests.size());
    int p_models = ::onnxruntime::Env::Default().GetNumCpuCores();
    if (tests[0]->GetTestCaseName() == "coreml_FNS-Candy_ImageNet") {
      p_models = 2;
    }
    snprintf(buf, sizeof(buf), "running test %s with %d cores", tests[0]->GetTestCaseName().c_str(), p_models);
    Logger::WriteMessage(buf);
    ONNXRUNTIME_EVENT finish_event;
    ::onnxruntime::Status status = CreateOnnxRuntimeEvent(&finish_event);
    Assert::IsTrue(status.IsOK());
    Assert::IsNotNull(finish_event);
    RunSingleTestCase(tests[0], sf, p_models, 1, GetDefaultThreadPool(::onnxruntime::Env::Default()), nullptr, [finish_event, &res](std::shared_ptr<TestCaseResult> result, PTP_CALLBACK_INSTANCE pci) {
      res = result->GetExcutionResult();
      return OnnxRuntimeSetEventWhenCallbackReturns(pci, finish_event);
    });
    status = WaitAndCloseEvent(finish_event);
    Assert::IsTrue(status.IsOK());
    Assert::AreEqual(tests[0]->GetDataCount(), res.size());
    delete tests[0];
  }
  for (EXECUTE_RESULT r : res) {
    Assert::AreEqual(EXECUTE_RESULT::SUCCESS, r);
  }
}

static ONNXEnv* env;

TEST_MODULE_INITIALIZE(ModuleInitialize) {
  ONNXStatusPtr ost = ONNXRuntimeInitialize(ONNXRUNTIME_LOGGING_LEVEL_kWARNING, "Default", &env);
  if (ost != nullptr) {
    Logger::WriteMessage(ONNXRuntimeGetErrorMessage(ost));
    ReleaseONNXStatus(ost);
    Assert::Fail(L"create onnx env failed");
  }
}

TEST_MODULE_CLEANUP(ModuleCleanup) {
  ReleaseONNXEnv(env);
}
// clang-format off
TEST_CLASS(ONNX_TEST){
public :    
#include "vsts/tests.inc"
};
