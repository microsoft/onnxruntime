// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/allocator_impl.h"
#include "core/session/inference_session_utils.h"
#include "core/session/IOBinding.h"
#include "core/framework/allocator.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/framework/utils.h"
#include <cassert>
#include <cstring>
#include <functional>
#include <sstream>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/status.h"
#include "core/common/safeint.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/ml_value.h"
#include "core/session/environment.h"
#include "core/framework/callback.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/session/inference_session.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/framework/data_types.h"
#include "abi_session_options_impl.h"
#include "core/framework/TensorSeq.h"
#include "core/platform/ort_mutex.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif

#include <comdef.h>
#include "runner/training_runner.h"
#include "runner/orttraining_c_api.h"
#include "runner/orttraining_apis.h"

#include "core/framework/bfc_arena.h"
#include "cxxopts.hpp"
#include <condition_variable>
#include <mutex>
#include <tuple>
#include <map>
#include <sstream>
#include <iostream>

using namespace onnxruntime::logging;
using onnxruntime::BFloat16;
using onnxruntime::DataTypeImpl;
using onnxruntime::Environment;
using onnxruntime::IAllocator;
using onnxruntime::InputDefList;
using onnxruntime::MLFloat16;
using onnxruntime::OutputDefList;
using onnxruntime::Tensor;
using onnxruntime::ToOrtStatus;
using onnxruntime::common::Status;

using namespace onnxruntime;
using namespace training;

namespace onnxruntime {
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(OrtDevice::DeviceId device_id,
                                                                               OrtCudnnConvAlgoSearch cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE,
                                                                               size_t cuda_mem_limit = std::numeric_limits<size_t>::max(),
                                                                               onnxruntime::ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo,
                                                                               bool do_copy_in_default_stream = true);
}

#define MAX_BUFFER 1024

#ifndef ORT_STATUS_PTR
#ifdef _WIN32
#define ORT_STATUS_PTR _Check_return_ _Ret_maybenull_ OrtStatusPtr
#else
#define ORT_STATUS_PTR OrtStatus*
#endif
#endif

// Return the OrtStatus if it indicates an error
#define ORT_API_RETURN_IF_ERROR(expr) \
  do {                                \
    auto _status = (expr);            \
    if (_status)                      \
      return _status;                 \
  } while (0)

// Convert internal onnxruntime::Status to OrtStatus and return if there's an error
#define ORT_API_RETURN_IF_STATUS_NOT_OK(expr) \
  do {                                        \
    auto _status = (expr);                    \
    if (!_status.IsOK())                      \
      return ToOrtStatus(_status);            \
  } while (0)

#define TENSOR_READ_API_BEGIN                          \
  API_IMPL_BEGIN                                       \
  auto v = reinterpret_cast<const ::OrtValue*>(value); \
  auto& tensor = v->Get<onnxruntime::Tensor>();

#define TENSOR_READWRITE_API_BEGIN \
  API_IMPL_BEGIN                   \
  auto v = (value);                \
  auto tensor = v->GetMutable<onnxruntime::Tensor>();


ORT_API_STATUS_IMPL(OrtTrainingApis::CreateTrainingParameters, OrtTrainingParameters** out) {
  API_IMPL_BEGIN
  *out = new OrtTrainingParameters();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtTrainingApis::ReleaseTrainingParameters, _Frees_ptr_opt_ OrtTrainingParameters* ptr) {
  delete ptr;
}

ORT_API_STATUS_IMPL(OrtTrainingApis::CloneTrainingParameters, const OrtTrainingParameters* input, OrtTrainingParameters** out) {
  API_IMPL_BEGIN
  *out = new OrtTrainingParameters(*input);
  return nullptr;
  API_IMPL_END
}

static char* StrDup(const wchar_t* val) {
  _bstr_t b(val);
  return _strdup((char*)b);
}

ORT_API_STATUS_IMPL(OrtTrainingApis::SetTrainingParameter_string, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingStringParameter key, _In_ const ORTCHAR_T* value) {
  API_IMPL_BEGIN
  Status status;

  switch (key) 
  {
    case OrtTrainingStringParameter::ORT_TRAINING_MODEL_PATH:
      pParam->m_param.model_name = _bstr_t(value);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_LOG_PATH:
      pParam->m_param.log_dir = ToPathString(value);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_INPUT_LABELS:
      pParam->m_strInputLabels = _bstr_t(value);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_OUTPUT_PREDICTIONS:
      pParam->m_strOutputPredictions = _bstr_t(value);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_OUTPUT_LOSS:
      pParam->m_strOutputLoss = _bstr_t(value);
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

static char* StrDup(const std::string& str, _Inout_ OrtAllocator* allocator) {
  char* output_string = reinterpret_cast<char*>(allocator->Alloc(allocator, str.size() + 1));
  memcpy(output_string, str.c_str(), str.size());
  output_string[str.size()] = '\0';
  return output_string;
}

static char* StrDup(const wchar_t* str, _Inout_ OrtAllocator* allocator) {
  _bstr_t b(str);
  return StrDup((char*)b, allocator);
}

static char* StrDup(const onnxruntime::PathString& str, _Inout_ OrtAllocator* allocator) {
  return StrDup(str.c_str(), allocator);
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetTrainingParameter_string, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingStringParameter key, _Inout_ OrtAllocator* allocator, _Outptr_ char** ppvalue) {
  API_IMPL_BEGIN
  Status status;

  switch (key) {
    case OrtTrainingStringParameter::ORT_TRAINING_MODEL_PATH:
      *ppvalue = StrDup(pParam->m_param.model_name.c_str(), allocator);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_LOG_PATH:
      *ppvalue = StrDup(pParam->m_param.log_dir.c_str(), allocator);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_INPUT_LABELS:
      *ppvalue = StrDup(pParam->m_strInputLabels, allocator);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_OUTPUT_PREDICTIONS:
      *ppvalue = StrDup(pParam->m_strOutputPredictions, allocator);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_OUTPUT_LOSS:
      *ppvalue = StrDup(pParam->m_strOutputLoss, allocator);
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::SetTrainingParameter_bool, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingBooleanParameter key, _In_ const bool value) {
  API_IMPL_BEGIN
  Status status;

  switch (key) {
    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_CUDA:
      pParam->m_bUseCuda = value;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_GIST:
      pParam->m_param.use_gist = value;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_PROFILER:
      pParam->m_param.use_profiler = value;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_TENSORBOARD:
      pParam->m_bUseTensorboard = value;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_IS_PERFTEST:
      pParam->m_param.is_perf_test = value;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_SHUFFLE_DATA:
      pParam->m_param.shuffle_data = value;
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}
ORT_API_STATUS_IMPL(OrtTrainingApis::GetTrainingParameter_bool, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingBooleanParameter key, _Out_ bool* pvalue) {
  API_IMPL_BEGIN
  Status status;

  switch (key) {
    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_CUDA:
      *pvalue = pParam->m_bUseCuda;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_GIST:
      *pvalue = pParam->m_param.use_gist;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_PROFILER:
      *pvalue = pParam->m_param.use_profiler;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_TENSORBOARD:
      *pvalue = pParam->m_bUseTensorboard;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_IS_PERFTEST:
      *pvalue = pParam->m_param.is_perf_test;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_SHUFFLE_DATA:
      *pvalue = pParam->m_param.shuffle_data;
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}


ORT_API_STATUS_IMPL(OrtTrainingApis::SetTrainingParameter_long, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingLongParameter key, _In_ const long value) {
  API_IMPL_BEGIN
  Status status;

  switch (key) {
    case OrtTrainingLongParameter::ORT_TRAINING_EVAL_BATCH_SIZE:
      pParam->m_param.eval_batch_size = (size_t)value;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_EVAL_PERIOD:
      pParam->m_param.evaluation_period = (size_t)value;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_NUM_TRAIN_STEPS:
      pParam->m_param.num_train_steps = (size_t)value;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_TRAIN_BATCH_SIZE:
      pParam->m_param.batch_size = (size_t)value;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_DISPLAY_LOSS_STEPS:
      pParam->m_param.display_loss_steps = (size_t)value;
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetTrainingParameter_long, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingLongParameter key, _Out_ long* pvalue) {
  API_IMPL_BEGIN
  Status status;

  switch (key) {
    case OrtTrainingLongParameter::ORT_TRAINING_EVAL_BATCH_SIZE:
      *pvalue = (long)pParam->m_param.eval_batch_size;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_EVAL_PERIOD:
      *pvalue = (long)pParam->m_param.evaluation_period;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_NUM_TRAIN_STEPS:
      *pvalue = (long)pParam->m_param.num_train_steps;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_TRAIN_BATCH_SIZE:
      *pvalue = (long)pParam->m_param.batch_size;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_DISPLAY_LOSS_STEPS:
      *pvalue = (long)pParam->m_param.display_loss_steps;
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}


ORT_API_STATUS_IMPL(OrtTrainingApis::SetTrainingParameter_double, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingNumericParameter key, _In_ const double value) {
  API_IMPL_BEGIN
  Status status;

  switch (key) {
    case OrtTrainingNumericParameter::ORT_TRAINING_LEARNING_RATE:
      pParam->m_param.lr_params.initial_lr = (float)value;
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetTrainingParameter_double, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingNumericParameter key, _Inout_ OrtAllocator* allocator, _Outptr_ char** ppvalue) {
  API_IMPL_BEGIN
  Status status;
  char szBuffer[MAX_BUFFER];

  switch (key) {
    case OrtTrainingNumericParameter::ORT_TRAINING_LEARNING_RATE:
      snprintf(szBuffer, MAX_BUFFER - 1, "%f", pParam->m_param.lr_params.initial_lr);
      *ppvalue = StrDup(szBuffer, allocator); 
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}


ORT_API_STATUS_IMPL(OrtTrainingApis::SetTrainingOptimizer, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingOptimizer opt) {
  API_IMPL_BEGIN
  Status status;

  switch (opt) {
    case OrtTrainingOptimizer::ORT_TRAINING_OPTIMIZER_SGD:
      pParam->m_param.training_optimizer_name = "SGDOptimizer";
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetTrainingOptimizer, _In_ OrtTrainingParameters* pParam,
                    _Out_ OrtTrainingOptimizer* popt) {
  API_IMPL_BEGIN
  Status status;

  if (pParam->m_param.training_optimizer_name == "SGDOptimizer")
    *popt = OrtTrainingOptimizer::ORT_TRAINING_OPTIMIZER_SGD;

  else
    status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");

  return ToOrtStatus(status);
  API_IMPL_END
}


ORT_API_STATUS_IMPL(OrtTrainingApis::SetTrainingLossFunction, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingLossFunction loss) {
  API_IMPL_BEGIN
  Status status;

  switch (loss) {
    case OrtTrainingLossFunction::ORT_TRAINING_LOSS_FUNCTION_SOFTMAXCROSSENTROPY:
      pParam->m_strLossFunction = "SoftmaxCrossEntropy";
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetTrainingLossFunction, _In_ OrtTrainingParameters* pParam,
                    _Out_ OrtTrainingLossFunction* ploss) {
  API_IMPL_BEGIN
  Status status;

  if (pParam->m_strLossFunction == "SoftmaxCrossEntropy")
    *ploss = OrtTrainingLossFunction::ORT_TRAINING_LOSS_FUNCTION_SOFTMAXCROSSENTROPY;

  else
    status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");

  return ToOrtStatus(status);
  API_IMPL_END
}


// Currently these are only used to show training with C# interaction,  
// A more long term change is to add to the training_runner a uniqe key passed in 
// to the RunWithUpdate training_runner function.
std::map<std::string, std::tuple<OrtErrorFunctionCallback, OrtEvaluationFunctionCallback>> m_fnFunctionMap;
std::string m_strLastMapUsed; 

static void error_function_callback(const std::vector<std::string>& feed_names,
                           const std::vector<OrtValue>& feeds,
                           const std::vector<std::string>& fetch_names,
                           const std::vector<OrtValue>& fetches,
                           size_t /*step*/) {
  const OrtValue* label_o = &feeds[1];
  const std::string label = feed_names[1];
  const OrtValue* predict_o = &fetches[0];
  const std::string predict = fetch_names[0];
  const OrtValue* loss_o = &fetches[1];
  const std::string loss = fetch_names[1];
  const int queue_id = 0;

  std::string strLabel = feed_names[1];
  m_strLastMapUsed = strLabel;
  OrtErrorFunctionCallback errorFn = std::get<0>(m_fnFunctionMap[strLabel]);
  OrtValueCollection col(3);

  col.m_rgValues[0] = (OrtValue*)label_o;
  col.m_rgNames[0] = label;
  col.m_rgValues[1] = (OrtValue*)predict_o;
  col.m_rgNames[1] = predict;
  col.m_rgValues[2] = (OrtValue*)loss_o;
  col.m_rgNames[2] = loss;
  col.BeforeUsingAsInput(queue_id);

  errorFn(col.Count(), &col);
}

static void evaluation_function_callback(size_t num_samples, size_t step, const std::string str) {
  OrtEvaluationFunctionCallback evalFn = std::get<1>(m_fnFunctionMap[m_strLastMapUsed]);

  evalFn(num_samples, step);
}

ORT_API_STATUS_IMPL(OrtTrainingApis::SetupTrainingParameters, _In_ OrtTrainingParameters* pParam,
                                        OrtErrorFunctionCallback errorFn,
                                        OrtEvaluationFunctionCallback evalFn) {
  API_IMPL_BEGIN
  Status status;

  pParam->m_param.model_path = ToPathString(pParam->m_param.model_name) + ORT_TSTR(".onnx");
  pParam->m_param.model_with_loss_func_path = ToPathString(pParam->m_param.model_name) + ORT_TSTR("_with_cost.onnx");
  pParam->m_param.model_with_training_graph_path = ToPathString(pParam->m_param.model_name) + ORT_TSTR("_bw.onnx");
  pParam->m_param.model_actual_running_graph_path = ToPathString(pParam->m_param.model_name) + ORT_TSTR("_bw_running.onnx");
  pParam->m_param.output_dir = ORT_TSTR(".");

  // Gist encode
  pParam->m_param.model_gist_encode_path = ToPathString(pParam->m_param.model_name) + ORT_TSTR("_encode_gist.onnx");

  if (pParam->m_strLossFunction == "SoftmaxCrossEntropy") {
    pParam->m_param.loss_func_info = LossFunctionInfo(OpDef("SoftmaxCrossEntropy", kMSDomain, 1),
                                                      pParam->m_strOutputLoss,
                                                      {pParam->m_strOutputPredictions, pParam->m_strInputLabels});
  } 
  else {
    status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
    return ToOrtStatus(status);
  }

  pParam->m_param.fetch_names = {pParam->m_strOutputPredictions, pParam->m_strOutputLoss};
  pParam->m_param.error_function = error_function_callback;
  pParam->m_param.post_evaluation_callback = evaluation_function_callback;
  m_fnFunctionMap.insert(std::make_pair(pParam->m_strInputLabels, std::make_pair(errorFn, evalFn)));

  // Setup CUDA
  if (pParam->m_bUseCuda) {
    // Use local rank as device ID of the associated CUDA EP.
    OrtDevice::DeviceId device_id = static_cast<OrtDevice::DeviceId>(MPIContext::GetInstance().GetLocalRank());
    pParam->m_param.providers.emplace(kCudaExecutionProvider, CreateExecutionProviderFactory_CUDA(device_id));
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

class DataSetEx : public DataSet {
 public:
  DataSetEx(OrtTrainingParameters* pParam, OrtDataUse dataUse) 
      : DataSet(pParam->m_rgstrDataFeedNames) 
  {
    m_pParam = pParam;
    m_dataUse = dataUse;
  }

  virtual size_t NumSamples() const 
  { 
      return m_pParam->m_param.batch_size;
  }

  virtual std::vector<OrtValue> GetKthBatch(size_t batch_size, size_t k_th, AllocatorPtr allocator = nullptr) const
  {
    OrtDataGetBatchCallback fnGetData;
    OrtValueCollection* pData;

    if (m_dataUse == ORT_DATAUSE_TRAINING) {
      fnGetData = m_pParam->m_fnTrainingDataGetBatch;
      pData = m_pParam->m_pTrainingData;
    } 
    else {
      fnGetData = m_pParam->m_fnTestingDataGetBatch;
      pData = m_pParam->m_pTestingData;
    }

    fnGetData(m_pParam->m_param.batch_size, pData->Capacity(), pData);

    std::vector<OrtValue*> dataPtr = pData->ValuePtrs();
    std::vector<OrtValue> result;

    for (int i = 0; i < dataPtr.size(); i++) {
      result.push_back(*dataPtr[i]);
    }

    return result;
  }

  private:
    OrtTrainingParameters* m_pParam;
    OrtDataUse m_dataUse;
};

ORT_API_STATUS_IMPL(OrtTrainingApis::SetupTrainingData, _In_ OrtTrainingParameters* pParam,
                    _In_ OrtDataGetBatchCallback trainingdataqueryFn, 
                    _In_ OrtDataGetBatchCallback testingdataqueryFn,
                    _In_ const ORTCHAR_T* szFeedNames) {
  API_IMPL_BEGIN
  Status status;

  pParam->m_fnTrainingDataGetBatch = trainingdataqueryFn;
  pParam->m_fnTestingDataGetBatch = testingdataqueryFn;

  pParam->m_pszInitFeedNames = StrDup(szFeedNames);
  istringstream fstrm(pParam->m_pszInitFeedNames);
  std::string strName;

  while (getline(fstrm, strName, ';')) {
    pParam->m_rgstrDataFeedNames.push_back(strName);
  }

  int nInputCount = 2;

  pParam->m_pTrainingData = new OrtValueCollection(nInputCount);
  pParam->m_pTestingData = new OrtValueCollection(nInputCount);

  auto device_count = MPIContext::GetInstance().GetWorldSize();
  auto trainingData = std::make_shared<DataSetEx>(pParam, OrtDataUse::ORT_DATAUSE_TRAINING);
  auto testingData = std::make_shared<DataSetEx>(pParam, OrtDataUse::ORT_DATAUSE_TESTING);

  pParam->m_pTrainingDataLoader = new SingleDataLoader(std::dynamic_pointer_cast<DataSet>(trainingData), pParam->m_rgstrDataFeedNames);
  pParam->m_pTestingDataLoader = new SingleDataLoader(std::dynamic_pointer_cast<DataSet>(testingData), pParam->m_rgstrDataFeedNames);

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::InitializeTraining, _In_ OrtEnv* pEnv, _In_ OrtTrainingParameters* pParam) {
  API_IMPL_BEGIN
  Status status;

  pParam->m_pTrainingRunner = new TrainingRunner(pParam->m_param, pEnv->GetEnvironment());    
  status = pParam->m_pTrainingRunner->Initialize();

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::RunTraining, _In_ OrtTrainingParameters* pParam) {
  API_IMPL_BEGIN
  Status status;

  status = pParam->m_pTrainingRunner->Run(pParam->m_pTrainingDataLoader, pParam->m_pTestingDataLoader);

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::EndTraining, _In_ OrtTrainingParameters* pParam) {
  API_IMPL_BEGIN
  Status status;

  status = pParam->m_pTrainingRunner->EndTraining(pParam->m_pTestingDataLoader);

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetCount, _In_ OrtValueCollection* pCol, _Out_ size_t* pnCount) {
  API_IMPL_BEGIN
  Status status;

  *pnCount = pCol->Count();

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetCapacity, _In_ OrtValueCollection* pCol, _Out_ size_t* pnCount) {
  API_IMPL_BEGIN
  Status status;

  *pnCount = pCol->Capacity();

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetAt, _In_ OrtValueCollection* pCol, _In_ size_t nIdx, _Outptr_ OrtValue** output, _Inout_ OrtAllocator* allocator, _Outptr_ char** ppName) {
  API_IMPL_BEGIN
  Status status;

  if (nIdx < pCol->m_rgValues.size()) {
    *output = pCol->m_rgValues[nIdx];
    *ppName = StrDup(pCol->m_rgNames[nIdx], allocator); 

  } 
  else {
    status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::SetAt, _In_ OrtValueCollection* pCol, _In_ size_t nIdx, _In_ OrtValue* input, _In_ const ORTCHAR_T* pszName) {
  API_IMPL_BEGIN
  Status status;

  if (nIdx < pCol->m_rgValues.size()) {
    pCol->m_rgValues[nIdx] = input;
    
    if (pszName != nullptr)
      pCol->m_rgNames[nIdx] = _bstr_t(pszName);
    else
      pCol->m_rgNames[nIdx] = "";
  } 
  else {
    status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

static constexpr OrtTrainingApiBase ort_training_api_base = {
    &OrtTrainingApis::GetApi,
    &OrtTrainingApis::GetVersionString,
};

/* Rules on how to add a new Ort API version

In general, NEVER remove or rearrange the members in this structure unless a new version is being created. The
goal is for newer shared libraries of the Onnx Runtime to work with binaries targeting the previous versions.
In order to do that we need to ensure older binaries get the older interfaces they are expecting.

See \onnxruntime\core\session\onnxruntime_c_api.cc notes for OrtApi table.

*/
static constexpr OrtTrainingApi ort_training_api_1_to_6 = {
    // NOTE: The ordering of these fields MUST not change after that version has shipped since existing binaries depend on this ordering.
    &OrtTrainingApis::CreateTrainingParameters,
    &OrtTrainingApis::CloneTrainingParameters,

    &OrtTrainingApis::SetTrainingParameter_string,
    &OrtTrainingApis::GetTrainingParameter_string,

    &OrtTrainingApis::SetTrainingParameter_bool,
    &OrtTrainingApis::GetTrainingParameter_bool,

    &OrtTrainingApis::SetTrainingParameter_long,
    &OrtTrainingApis::GetTrainingParameter_long,

    &OrtTrainingApis::SetTrainingParameter_double,
    &OrtTrainingApis::GetTrainingParameter_double,

    &OrtTrainingApis::SetTrainingOptimizer,
    &OrtTrainingApis::GetTrainingOptimizer,

    &OrtTrainingApis::SetTrainingLossFunction,
    &OrtTrainingApis::GetTrainingLossFunction,

    &OrtTrainingApis::SetupTrainingParameters,
    &OrtTrainingApis::SetupTrainingData,

    &OrtTrainingApis::InitializeTraining,
    &OrtTrainingApis::RunTraining,
    &OrtTrainingApis::EndTraining,

    &OrtTrainingApis::GetCount,
    &OrtTrainingApis::GetCapacity,
    &OrtTrainingApis::GetAt,
    &OrtTrainingApis::SetAt,

    &OrtTrainingApis::ReleaseTrainingParameters,
};


ORT_API(const OrtTrainingApi*, OrtTrainingApis::GetApi, uint32_t version) {
  if (version >= 1 && version <= 6)
    return &ort_training_api_1_to_6;

  return nullptr;  // Unsupported version
}

ORT_API(const char*, OrtTrainingApis::GetVersionString) {
  return ORT_VERSION;
}

const OrtTrainingApiBase* ORT_API_CALL OrtGetTrainingApiBase(void) NO_EXCEPTION {
  return &ort_training_api_base;
}
