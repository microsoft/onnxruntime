// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testenv.h"
#include "callables.h"
#include "heap_buffer.h"
#include "TestCaseResult.h"
#include "FixedCountFinishCallback.h"
#include <core/session/onnxruntime_cxx_api.h>

// Class that allows to run a single TestCase either sync or async.
namespace onnxruntime {
namespace test {

// This runs a single DataTask on a threadpool and
// invokes a callback to TestCaseRequestContext that orchestrates
// the DataTasks related to the model
class DataTaskRequestContext {
 public:
  // This is a callback that will be invoked by the individual task
   // when it completes
  using Callback = Callable<void, size_t, EXECUTE_RESULT>;
  // static void Run(size_t task_id);

  DataTaskRequestContext(const DataTaskRequestContext&) = delete;
  DataTaskRequestContext& operator=(const DataTaskRequestContext&) = delete;

private:

  DataTaskRequestContext(const Callback& cb, const ITestCase& c, Ort::Session& session, size_t task_id) 
    : cb_(cb),
       session_(session),
      c_(c),
      task_id_(task_id) {
  }

  ~DataTaskRequestContext() {}

  void RunImpl();

  Callback cb_;
  Ort::Session& session_;
  const ITestCase& c_;
  size_t task_id_;
};

//void DataTaskRequestContext::Run(size_t task_id) {
//  Callback empty_cb;
//  DataTaskRequestContext ctx(empty_cb, task_id);
//}

void DataTaskRequestContext::RunImpl() {
  onnxruntime::test::HeapBuffer holder;
  std::unordered_map<std::string, OrtValue*> feeds;
  c_.LoadTestData(task_id_, holder, feeds, true);

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
  // return res;
}


// This runs data tasks related to a single model in parallel
class TestCaseRequestContext {
 public:

  // Run sync
  static void RunTestCase(const ITestCase& info, Ort::Env& env, const Ort::SessionOptions& sf, size_t repeat_count);
  // Run async
  static void RequestRunTestCase(Callback cb, const ITestCase& info, Ort::Env& env, const Ort::SessionOptions& sf,
                                 size_t concurrent_runs, size_t repeat_count, PThreadPool tpool);

 private:
  // Something we call in case of async execution
  Callback cb_;
};

}  // namespace test
}  // namespace onnxruntime

using onnxruntime::Status;
TestEnv::TestEnv(const std::vector<ITestCase*>& tests1, TestResultStat& stat1, Ort::Env& env1,
                 Ort::SessionOptions& sf1, PThreadPool tp)
    : tests(tests1),
      next_test_to_run(0),
      stat(stat1),
      // finished(new FixedCountFinishCallback<TestCaseResult>(static_cast<int>(tests1.size()))),
      env(env1),
      sf(sf1),
      tp_(tp) {
}

TestEnv::~TestEnv() {
  // need dtor in .cc so 'finished' can be cleaned up as TestCaseResult only has a forward declaration in the header.
}


