#pragma once

#include "Env.h"
#include "JsiHelper.h"
#include <jsi/jsi.h>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <vector>

using namespace facebook::jsi;

namespace onnxruntimejsi {

class InferenceSessionHostObject
    : public HostObjectHelper,
      public std::enable_shared_from_this<InferenceSessionHostObject> {
 public:
  InferenceSessionHostObject(std::shared_ptr<Env> env) : HostObjectHelper({
                                                                              METHOD_INFO(InferenceSessionHostObject, loadModel, 2),
                                                                              METHOD_INFO(InferenceSessionHostObject, run, 2),
                                                                              METHOD_INFO(InferenceSessionHostObject, dispose, 0),
                                                                              METHOD_INFO(InferenceSessionHostObject, endProfiling, 0),
                                                                          },
                                                                          {
                                                                              GETTER_INFO(InferenceSessionHostObject, inputMetadata),
                                                                              GETTER_INFO(InferenceSessionHostObject, outputMetadata),
                                                                          }),
                                                         env_(env) {}

  static inline facebook::jsi::Value
  constructor(std::shared_ptr<Env> env, facebook::jsi::Runtime& runtime,
              const facebook::jsi::Value& thisValue,
              const facebook::jsi::Value* arguments, size_t count) {
    return facebook::jsi::Object::createFromHostObject(
        runtime, std::make_shared<InferenceSessionHostObject>(env));
  }

 private:
  std::shared_ptr<Env> env_;
  std::shared_ptr<Ort::Session> session_;

  class LoadModelAsyncWorker;
  class RunAsyncWorker;

  DEFINE_METHOD(loadModel);
  DEFINE_METHOD(run);
  DEFINE_METHOD(dispose);
  DEFINE_METHOD(endProfiling);

  DEFINE_GETTER(inputMetadata);
  DEFINE_GETTER(outputMetadata);
};

}  // namespace onnxruntimejsi
