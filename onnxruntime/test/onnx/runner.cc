// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "runner.h"

#include <fstream>
#include <cmath>

#include <core/common/logging/logging.h>
#include <core/graph/constants.h>
#include <core/platform/env.h>
#include <core/framework/tensorprotoutils.h>
#include "test_allocator.h"
#include <core/platform/path_lib.h>
#ifdef _WIN32
#include <Windows.h>
#else
#include <pthread.h>
#endif

#include <test/compare_ortvalue.h>
#include "TestCase.h"
#include "heap_buffer.h"
#include "OrtValueList.h"
#include "FixedCountFinishCallback.h"

using namespace onnxruntime;
using ::onnxruntime::common::Status;

void ORT_CALLBACK RunTestCase(ORT_CALLBACK_INSTANCE pci, void* context, ORT_WORK work) {
  OnnxRuntimeCloseThreadpoolWork(work);
  assert(context != nullptr);
  TestCaseTask* task(static_cast<TestCaseTask*>(context));
  ITestCase& info = *task->env.tests[task->task_id];
  std::shared_ptr<TestCaseResult> ret;
  try {
    RunSingleTestCase(info, task->env.env, task->env.sf, task->concurrent_runs, task->repeat_count, task->pool, pci,
                      [task](std::shared_ptr<TestCaseResult> result, ORT_CALLBACK_INSTANCE pci) {
                        return OnTestCaseFinished(pci, task, result);
                      });
  } catch (std::exception& ex) {
    LOGF_DEFAULT(ERROR, "Test %s failed:%s", info.GetTestCaseName().c_str(), ex.what());

    ret = std::make_shared<TestCaseResult>(info.GetDataCount(), EXECUTE_RESULT::WITH_EXCEPTION, info.GetNodeName());
    auto status = OnTestCaseFinished(pci, task, ret);
    if (!status.IsOK()) {
      ORT_THROW("OnTestCaseFinished failed: ", status.ErrorMessage());
    }
  }
}

void PTestRunner::Start(ORT_CALLBACK_INSTANCE pci, size_t concurrent_runs) noexcept {
  bool atleast_one_run_scheduled = false;
  try {
    concurrent_runs = std::min<size_t>(std::max<size_t>(1, concurrent_runs), c_.GetDataCount());
    next_test_to_run = 0;
    for (size_t i = 0; i < concurrent_runs; ++i) {
      if (!ScheduleNew()) {
        if (i == 0) {
          // This should be a rare case.
          // Log an error. While exiting Finish will be called which will take care of the cleanup
          LOGF_DEFAULT(ERROR, "No task scheduled for test case: %s\n", c_.GetTestCaseName().c_str());
        } 
        // In cases when concurrent_runs == number of test cases, before i reaches "concurrent_runs" 
        // an older tasks may finish and schedule a new task from OnTaskFinished method so no task is left to be scheduled 
        // from here.
        break;
      }
      atleast_one_run_scheduled = true;
    }
  } catch (const std::exception& ex) {
    LOGF_DEFAULT(ERROR, "Cannot schedule tasks for test %s. Failing with error :%s\n", c_.GetTestCaseName().c_str(), ex.what());
  } catch (...) {
    LOGF_DEFAULT(ERROR, "Cannot schedule tasks for test %s.\n", c_.GetTestCaseName().c_str());
  }

  // If no task was scheduled then call Finish to perform cleanup
  // Otherwise the last task to complete will call Finish
  if (!atleast_one_run_scheduled) {
    Finish(pci);
  }
}

bool PTestRunner::ScheduleNew() {
  size_t next_test = next_test_to_run++;
  if (next_test >= c_.GetDataCount()) return false;
  DataTask* t = new DataTask{this, next_test};
  Status st = CreateAndSubmitThreadpoolWork(RunSingleDataItem, t, tpool_);
  if (!st.IsOK()) {
    delete t;
    LOGF_DEFAULT(ERROR, "schedule test task failed: %s\n", st.ErrorMessage().c_str());
    return false;
  }
  return true;
}

void PTestRunner::OnTaskFinished(size_t, EXECUTE_RESULT, ORT_CALLBACK_INSTANCE pci) noexcept {
  try {
    ScheduleNew();
    if (++finished == c_.GetDataCount()) {
      //For each test case, only one DataTask can reach here
      Finish(pci);
    }
  } catch (std::exception& ex) {
    LOGF_DEFAULT(ERROR, "%s:unrecoverable error:%s,exit...\n", c_.GetTestCaseName().c_str(), ex.what());
    abort();
  } catch (...) {
    LOGF_DEFAULT(ERROR, "%s:unrecoverable error,exit...\n", c_.GetTestCaseName().c_str());
    abort();
  }
}

PTestRunner::PTestRunner(OrtSession* session1,
                         const ITestCase& c, PThreadPool tpool,
                         TestCaseCallBack on_finished1)
    : DataRunner(session1, c.GetTestCaseName(), c, on_finished1), next_test_to_run(0), finished(0), tpool_(tpool) {
}

void ORT_CALLBACK RunSingleDataItem(ORT_CALLBACK_INSTANCE instance, void* context, ORT_WORK work) {
  OnnxRuntimeCloseThreadpoolWork(work);
  DataTask* task(static_cast<DataTask*>(context));
  PTestRunner* env = task->env;
  const size_t task_id = task->task_id;
  delete task;
  env->RunTask(task_id, instance);
}

Status OnTestCaseFinished(ORT_CALLBACK_INSTANCE pci, TestCaseTask* task, std::shared_ptr<TestCaseResult> result) {
  auto task_id = task->task_id;
  bool failed = false;

  std::unique_ptr<TestCaseTask> delete_finished_task(task);

  TestEnv& env = task->env;
  int next_test = env.next_test_to_run++;
  if (static_cast<size_t>(next_test) < env.tests.size()) {
    //schedule the next TestCase
    std::unique_ptr<TestCaseTask> t(new TestCaseTask{env, next_test, task->concurrent_runs, task->repeat_count,
                                                     task->pool});
    Status st = CreateAndSubmitThreadpoolWork(RunTestCase, t.get(), task->pool);
    if (st.IsOK()) {
      t.release();
    } else {
      return st;
    }
  }

  return failed ? env.finished->Fail(pci) : env.finished->OnFinished(task_id, result, pci);
}

//Do not run this function in the thread pool passed in
static Status ParallelRunTests(TestEnv& env, int p_models, size_t current_runs, size_t repeat_count,
                               PThreadPool pool) {
  p_models = static_cast<int>(std::min<size_t>(p_models, env.tests.size()));
  LOGF_DEFAULT(ERROR, "Running tests in parallel: at most %d models at any time", p_models);
  env.next_test_to_run = p_models;
  for (int i = 0; i != p_models; ++i) {
    TestCaseTask* t(new TestCaseTask{env, i, current_runs, repeat_count, pool});
    try {
      auto st = CreateAndSubmitThreadpoolWork(RunTestCase, t, pool);
      if (!st.IsOK()) {
        delete t;
        return st;
      }
    } catch (std::exception&) {
      delete t;
      throw;
    }
  }
  bool ret = env.finished->Wait();
  if (!ret) {
    return Status(::onnxruntime::common::ONNXRUNTIME, ::onnxruntime::common::FAIL, "ParallelRunTests failed");
  }
  LOGF_DEFAULT(ERROR, "Running tests finished. Generating report");
  return Status::OK();
}

Status RunTests(TestEnv& env, int p_models, int concurrent_runs, size_t repeat_count, PThreadPool tpool) {
  TestResultStat& stat = env.stat;
  stat.total_model_count = env.tests.size();
  stat.total_test_case_count = std::accumulate(env.tests.begin(), env.tests.end(), static_cast<size_t>(0),
                                               [](size_t v, const ITestCase* info) {
                                                 return info->GetDataCount() + v;
                                               });
  std::vector<std::shared_ptr<TestCaseResult>> results;

  if (p_models > 1 && env.tests.size() > 1) {
    ORT_RETURN_IF_ERROR(ParallelRunTests(env, p_models, concurrent_runs, repeat_count, tpool));
    results = env.finished->getResults();
  } else {
    //run models one by one
    for (size_t i = 0; i != env.tests.size(); ++i) {
      const char* test_case_name = env.tests[i]->GetTestCaseName().c_str();
      ORT_EVENT ev;
      ORT_RETURN_IF_ERROR(CreateOnnxRuntimeEvent(&ev));
      try {
        RunSingleTestCase(*env.tests[i], env.env, env.sf, concurrent_runs, repeat_count, tpool, nullptr,
                          [&results, ev](std::shared_ptr<TestCaseResult> result, ORT_CALLBACK_INSTANCE pci) {
                            results.push_back(result);
                            return OnnxRuntimeSetEventWhenCallbackReturns(pci, ev);
                          });

        ORT_RETURN_IF_ERROR(WaitAndCloseEvent(ev));
      } catch (std::exception& ex) {
        LOGF_DEFAULT(ERROR, "Test %s failed:%s", test_case_name, ex.what());
        std::string node_name = env.tests[i]->GetNodeName();
        results.push_back(
            std::make_shared<TestCaseResult>(env.tests[i]->GetDataCount(), EXECUTE_RESULT::WITH_EXCEPTION, node_name));
        OrtCloseEvent(ev);
      }
    }
  }
  for (size_t i = 0; i != env.tests.size(); ++i) {
    if (!results[i]) {
      stat.AddFailedTest(std::pair<std::string, std::string>(env.tests[i]->GetTestCaseName(),
                                                             env.tests[i]->GetTestCaseVersion()));
      continue;
    }
    const TestCaseResult& r = *results[i];
    for (const EXECUTE_RESULT res : r.GetExcutionResult()) {
      if (res != EXECUTE_RESULT::SUCCESS && res != EXECUTE_RESULT::NOT_SUPPORT) {
        stat.AddFailedTest(std::pair<std::string, std::string>(env.tests[i]->GetTestCaseName(),
                                                               env.tests[i]->GetTestCaseVersion()));
      }
      switch (res) {
        case EXECUTE_RESULT::SUCCESS:
          stat.succeeded++;
          break;
        case EXECUTE_RESULT::INVALID_ARGUMENT:
        case EXECUTE_RESULT::UNKNOWN_ERROR:
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::INVALID_GRAPH:
          stat.invalid_graph++;
          break;
        case EXECUTE_RESULT::WITH_EXCEPTION:
          stat.throwed_exception++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::RESULT_DIFFERS:
          stat.result_differs++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::MODEL_SHAPE_MISMATCH:
        case EXECUTE_RESULT::SHAPE_MISMATCH:
        case EXECUTE_RESULT::MODEL_TYPE_MISMATCH:
        case EXECUTE_RESULT::TYPE_MISMATCH:
          stat.result_differs++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::NOT_SUPPORT:
          stat.not_implemented++;
          if (!r.node_name.empty()) stat.AddNotImplementedKernels(r.node_name);
          break;
        case EXECUTE_RESULT::LOAD_MODEL_FAILED:
          stat.load_model_failed++;
          if (!r.node_name.empty()) stat.AddFailedKernels(r.node_name);
          break;
        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "unknown result");
      }
    }
  }
  return common::Status::OK();
}

void LoadTests(const std::vector<std::basic_string<PATH_CHAR_TYPE>>& input_paths,
               const std::vector<std::basic_string<PATH_CHAR_TYPE>>& whitelisted_test_cases,
               double default_per_sample_tolerance, double default_relative_per_sample_tolerance,
               const std::unordered_set<std::basic_string<ORTCHAR_T>>& disabled_tests,
               const std::function<void(std::unique_ptr<ITestCase>)>& process_function) {
  std::vector<std::basic_string<PATH_CHAR_TYPE>> paths(input_paths);
  while (!paths.empty()) {
    std::basic_string<PATH_CHAR_TYPE> node_data_root_path = paths.back();
    paths.pop_back();
    std::basic_string<PATH_CHAR_TYPE> my_dir_name = GetLastComponent(node_data_root_path);
    LoopDir(node_data_root_path, [&](const PATH_CHAR_TYPE* filename, OrtFileType f_type) -> bool {
      if (filename[0] == '.') return true;
      if (f_type == OrtFileType::TYPE_DIR) {
        std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(node_data_root_path, filename);
        paths.push_back(p);
        return true;
      }
      std::basic_string<PATH_CHAR_TYPE> filename_str = filename;
      if (!HasExtensionOf(filename_str, ORT_TSTR("onnx"))) return true;

      std::basic_string<PATH_CHAR_TYPE> test_case_name = my_dir_name;
      if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0) test_case_name = test_case_name.substr(5);

      if (!whitelisted_test_cases.empty() && std::find(whitelisted_test_cases.begin(), whitelisted_test_cases.end(),
                                                       test_case_name) == whitelisted_test_cases.end()) {
        return true;
      }
      if (disabled_tests.find(test_case_name) != disabled_tests.end()) return true;

      std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(node_data_root_path, filename_str);

      std::unique_ptr<TestModelInfo> model_info(TestModelInfo::LoadOnnxModel(p.c_str()));
      std::unique_ptr<ITestCase> l = CreateOnnxTestCase(ToMBString(test_case_name), std::move(model_info),
                                                        default_per_sample_tolerance,
                                                        default_relative_per_sample_tolerance);
      process_function(std::move(l));
      return true;
    });
  }
}

SeqTestRunner::SeqTestRunner(OrtSession* session1,
                             const ITestCase& c, size_t repeat_count,
                             TestCaseCallBack on_finished1)
    : DataRunner(session1, c.GetTestCaseName(), c, on_finished1), repeat_count_(repeat_count) {
}

DataRunner::DataRunner(OrtSession* session1, const std::string& test_case_name1, const ITestCase& c,
                       TestCaseCallBack on_finished1)
    : test_case_name_(test_case_name1),
      c_(c),
      session(session1),
      on_finished(on_finished1),
      default_allocator(onnxruntime::make_unique<MockedOrtAllocator>()) {
  std::string s = c_.GetNodeName();
  result = std::make_shared<TestCaseResult>(c_.GetDataCount(), EXECUTE_RESULT::NOT_SET, s);
  SetTimeSpecToZero(&spent_time_);
}

DataRunner::~DataRunner() {
  Ort::GetApi().ReleaseSession(session);
}

void DataRunner::RunTask(size_t task_id, ORT_CALLBACK_INSTANCE pci) {
  EXECUTE_RESULT res = EXECUTE_RESULT::UNKNOWN_ERROR;
  try {
    res = RunTaskImpl(task_id);
  } catch (const std::exception& ex) {
    res = EXECUTE_RESULT::WITH_EXCEPTION;
    LOGS_DEFAULT(ERROR) << c_.GetTestCaseName() << ":" << ex.what();
  }

  result->SetResult(task_id, res);
  OnTaskFinished(task_id, res, pci);
}

EXECUTE_RESULT DataRunner::RunTaskImpl(size_t task_id) {
  onnxruntime::test::HeapBuffer holder;
  std::unordered_map<std::string, OrtValue*> feeds;
  c_.LoadTestData(task_id, holder, feeds, true);

  // Create output feed
  size_t output_count = 0;
  Ort::ThrowOnError(Ort::GetApi().SessionGetOutputCount(session, &output_count));
  std::vector<std::string> output_names(output_count);
  for (size_t i = 0; i != output_count; ++i) {
    char* output_name = nullptr;
    Ort::ThrowOnError(Ort::GetApi().SessionGetOutputName(session, i, default_allocator.get(), &output_name));
    assert(output_name != nullptr);
    output_names[i] = output_name;
    default_allocator->Free(output_name);
  }
  if (feeds.size() > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
    ORT_THROW("length overflow");
  }
  std::vector<const char*> input_names(feeds.size());
  OrtValueArray input_values(static_cast<int>(feeds.size()));
  size_t input_index = 0;
  for (auto& kvp : feeds) {
    input_names[input_index] = kvp.first.c_str();
    input_values.Set(input_index, kvp.second);
    ++input_index;
  }

  TIME_SPEC start_time;
  TIME_SPEC end_time;
  OrtValueArray output_values(static_cast<int>(output_count));
  {
    std::vector<const char*> output_names_raw_ptr(output_count);
    for (size_t i = 0; i != output_count; ++i) {
      output_names_raw_ptr[i] = output_names[i].c_str();
    }
    GetMonotonicTimeCounter(&start_time);
    Ort::ThrowOnError(Ort::GetApi().Run(session, nullptr, input_names.data(), input_values.Data(),
                                        static_cast<size_t>(input_values.Length()), output_names_raw_ptr.data(),
                                        output_count, output_values.Data()));
  }
  GetMonotonicTimeCounter(&end_time);
  AccumulateTimeSpec(&spent_time_, &start_time, &end_time);

  double per_sample_tolerance;
  double relative_per_sample_tolerance;
  bool post_procesing;
  Status status;
  if (!(status = c_.GetPerSampleTolerance(&per_sample_tolerance)).IsOK()) {
    LOGS_DEFAULT(ERROR) << status.ErrorMessage() << "\n";
    return StatusCodeToExecuteResult(status.Code());
  }
  if (!(status = c_.GetRelativePerSampleTolerance(&relative_per_sample_tolerance)).IsOK()) {
    LOGS_DEFAULT(ERROR) << status.ErrorMessage() << "\n";
    return StatusCodeToExecuteResult(status.Code());
  }
  if (!(status = c_.GetPostProcessing(&post_procesing)).IsOK()) {
    LOGS_DEFAULT(ERROR) << status.ErrorMessage() << "\n";
    return StatusCodeToExecuteResult(status.Code());
  }

  //TODO: if there are no output value files, just skip the validation
  std::unordered_map<std::string, OrtValue*> expected_output_values;
  c_.LoadTestData(task_id, holder, expected_output_values, false);

  std::unordered_map<std::string, OrtValue*> name_fetch_output_map;
  std::unordered_map<std::string, const ONNX_NAMESPACE::ValueInfoProto*> name_output_value_info_proto;
  size_t i = 0;
  for (auto& output_name : output_names) {
    // p_fetches is filled in the order of output_names.
    name_fetch_output_map[output_name] = output_values.Get(i);
    const ONNX_NAMESPACE::ValueInfoProto* infoProto = c_.GetOutputInfoFromModel(i);
    if (infoProto != nullptr) name_output_value_info_proto.insert(std::make_pair(infoProto->name(), infoProto));
    i++;
  }

  EXECUTE_RESULT res = EXECUTE_RESULT::SUCCESS;
  for (auto& output : expected_output_values) {
    OrtValue* expected_output_value = output.second;
    const std::string& output_name = output.first;
    auto iter = name_fetch_output_map.find(output_name);
    if (iter == name_fetch_output_map.end()) {
      res = EXECUTE_RESULT::INVALID_GRAPH;
      LOGF_DEFAULT(ERROR, "cannot find %s in the outputs", output_name.c_str());
      break;
    }
    OrtValue* actual_output_value = iter->second;
    std::pair<COMPARE_RESULT, std::string> ret =
        CompareOrtValue(*actual_output_value, *expected_output_value, per_sample_tolerance,
                        relative_per_sample_tolerance, post_procesing);
    COMPARE_RESULT compare_result = ret.first;
    if (compare_result == COMPARE_RESULT::SUCCESS) {
      const ONNX_NAMESPACE::ValueInfoProto* v = name_output_value_info_proto[output_name];
      if (v == nullptr) continue;
      ret = VerifyValueInfo(*v, Ort::Unowned<Ort::Value>{actual_output_value});
      compare_result = ret.first;
      if (compare_result != COMPARE_RESULT::SUCCESS) {
        switch (compare_result) {
          case COMPARE_RESULT::NOT_SUPPORT:
            res = EXECUTE_RESULT::NOT_SUPPORT;
            break;
          case COMPARE_RESULT::SHAPE_MISMATCH:
            res = EXECUTE_RESULT::MODEL_SHAPE_MISMATCH;
            break;
          case COMPARE_RESULT::TYPE_MISMATCH:
            res = EXECUTE_RESULT::MODEL_TYPE_MISMATCH;
            break;
          default:
            res = EXECUTE_RESULT::UNKNOWN_ERROR;
        }
      }
    } else {
      switch (compare_result) {
        case COMPARE_RESULT::NOT_SUPPORT:
          res = EXECUTE_RESULT::NOT_SUPPORT;
          break;
        case COMPARE_RESULT::RESULT_DIFFERS:
          res = EXECUTE_RESULT::RESULT_DIFFERS;
          break;
        case COMPARE_RESULT::SHAPE_MISMATCH:
          res = EXECUTE_RESULT::SHAPE_MISMATCH;
          break;
        case COMPARE_RESULT::TYPE_MISMATCH:
          res = EXECUTE_RESULT::TYPE_MISMATCH;
          break;
        default:
          res = EXECUTE_RESULT::UNKNOWN_ERROR;
      }
    }

    if (compare_result != COMPARE_RESULT::SUCCESS && !ret.second.empty()) {
      LOGS_DEFAULT(ERROR) << test_case_name_ << ":output=" << output_name << ":" << ret.second;
    }
    if (compare_result != COMPARE_RESULT::SUCCESS) {
      break;
    }
  }
  for (auto& kvp : expected_output_values) {
    Ort::GetApi().ReleaseValue(kvp.second);
  }
  return res;
}

void SeqTestRunner::Start(ORT_CALLBACK_INSTANCE pci, size_t) noexcept {
  try {
    const size_t data_count = c_.GetDataCount();
    for (size_t idx_repeat = 0; idx_repeat != repeat_count_; ++idx_repeat) {
      for (size_t idx_data = 0; idx_data != data_count; ++idx_data) {
        RunTask(idx_data, nullptr);
      }
    }
  } catch (...) {
    LOGS_DEFAULT(ERROR) << "SeqTestRunner::Start - Encountred exception with running tests";
  }

  Finish(pci);
}

void RunSingleTestCase(const ITestCase& info, Ort::Env& env, const Ort::SessionOptions& sf,
                       size_t concurrent_runs, size_t repeat_count, PThreadPool tpool,
                       ORT_CALLBACK_INSTANCE pci, TestCaseCallBack on_finished) {
  std::shared_ptr<TestCaseResult> ret;
  size_t data_count = info.GetDataCount();
  try {
    std::string node_name = info.GetNodeName();
    auto sf2 = sf.Clone();
    sf2.SetLogId(info.GetTestCaseName().c_str());
    Ort::Session session_object{env, info.GetModelUrl(), sf2};
    LOGF_DEFAULT(INFO, "testing %s\n", info.GetTestCaseName().c_str());
    //temp hack. Because we have no resource control. We may not have enough memory to run this test in parallel
    if (info.GetTestCaseName() == "coreml_FNS-Candy_ImageNet") {
      concurrent_runs = 1;
    }

    // DataRunner owns itself. In case of success or failures DataRunner will delete itself.
    // In case of SeqTestRunner SeqTestRunner::Start will call DataRunner::Finish at the end and will delete itself.
    // In case of PTestRunner on success the last task will call Finish and in cases of failure Start will itself call DataRunner::Finish
    // before exiting. Start does not throw so we do not need to worry about this case.
    DataRunner* r;
    if (concurrent_runs > 1 && data_count > 1) {
      r = new PTestRunner(session_object.release(), info, tpool, on_finished);
    } else {
      r = new SeqTestRunner(session_object.release(), info, repeat_count, on_finished);
    }
    r->Start(pci, concurrent_runs);

    // both PTestRunner and SeqTestRunner call DataRunner::Finish which will delete itself and call on_finished callback
    // at this point we know everything has started without any exceptions so simply return.
    return;

  } catch (const Ort::Exception& ex) {
    LOGF_DEFAULT(ERROR, "Test %s failed:%s", info.GetTestCaseName().c_str(), ex.what());
    std::string node_name;
    ret = std::make_shared<TestCaseResult>(data_count, EXECUTE_RESULT::NOT_SUPPORT, "");
  }

  // we will land here in case of session creation failures.
  // in all other cases DataRunner::Finish will call this.
  on_finished(ret, pci);
}

EXECUTE_RESULT StatusCodeToExecuteResult(int input) {
  switch (input) {
    case common::NOT_IMPLEMENTED:
      return EXECUTE_RESULT::NOT_SUPPORT;
    case common::INVALID_GRAPH:
      return EXECUTE_RESULT::INVALID_GRAPH;
    case common::INVALID_ARGUMENT:
      return EXECUTE_RESULT::INVALID_ARGUMENT;
    default:
      return EXECUTE_RESULT::UNKNOWN_ERROR;
  }
}
