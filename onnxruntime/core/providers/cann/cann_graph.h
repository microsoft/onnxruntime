// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <utility>
#include <iomanip>
#include <string>
#include <exception>
#include <future>
#include <thread>

#include "core/providers/cann/cann_common.h"
#include "core/providers/cann/cann_inc.h"
#include "core/providers/cann/cann_utils.h"
#include "core/providers/cann/cann_execution_provider_info.h"

namespace onnxruntime {
namespace cann {

struct GeState {
  GeState()
      : future_init(promise_init.get_future().share()), ex_ptr_init(nullptr), future_final(promise_final.get_future()), ex_ptr_final(nullptr) {}

  std::thread thread;

  std::promise<void> promise_init;
  std::shared_future<void> future_init;
  std::exception_ptr ex_ptr_init;

  std::promise<void> promise_final;
  std::future<void> future_final;
  std::exception_ptr ex_ptr_final;
};

struct CannModelPreparation {
  explicit CannModelPreparation(uint32_t modelID) {
    modelDesc_ = aclmdlCreateDesc();
    CANN_CALL_THROW(aclmdlGetDesc(modelDesc_, modelID));

    inputSet_ = aclmdlCreateDataset();
    outputSet_ = aclmdlCreateDataset();
  }

  virtual ~CannModelPreparation() {
    CANN_CALL_THROW(aclmdlDestroyDesc(modelDesc_));

    CANN_CALL_THROW(aclmdlDestroyDataset(inputSet_));
    CANN_CALL_THROW(aclmdlDestroyDataset(outputSet_));

    for (auto buf : inputBuffers_) {
      CANN_CALL_THROW(aclDestroyDataBuffer(buf));
    }

    for (auto buf : outputBuffers_) {
      CANN_CALL_THROW(aclDestroyDataBuffer(buf));
    }
  }

  std::vector<aclDataBuffer*> inputBuffers_;
  std::vector<aclDataBuffer*> outputBuffers_;
  aclmdlDataset* inputSet_;
  aclmdlDataset* outputSet_;
  aclmdlDesc* modelDesc_;
};

#define CANN_MODEL_PREPARE_INPUTBUFFER(var, ...)    \
  do {                                              \
    auto _rPtr = aclCreateDataBuffer(__VA_ARGS__);  \
    if (_rPtr == nullptr) {                         \
      ORT_THROW("aclCreateDataBuffer run failed");  \
    } else {                                        \
      var.inputBuffers_.push_back(_rPtr);           \
      aclmdlAddDatasetBuffer(var.inputSet_, _rPtr); \
    }                                               \
  } while (0)

#define CANN_MODEL_PREPARE_OUTPUTBUFFER(var, ...)    \
  do {                                               \
    auto _rPtr = aclCreateDataBuffer(__VA_ARGS__);   \
    if (_rPtr == nullptr) {                          \
      ORT_THROW("aclCreateDataBuffer run failed");   \
    } else {                                         \
      var.outputBuffers_.push_back(_rPtr);           \
      aclmdlAddDatasetBuffer(var.outputSet_, _rPtr); \
    }                                                \
  } while (0)

std::vector<NodeIndex> SupportONNXModel(const GraphViewer& graph_viewer);
Status ParserONNXModel(std::string string_model, ge::Graph& graph);
Status BuildONNXModel(ge::Graph& graph, std::string input_shape, const char* soc_name, std::string file_name,
                      CANNExecutionProviderInfo& info, ge::ModelBufferData& model);

}  // namespace cann
}  // namespace onnxruntime
