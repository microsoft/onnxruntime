#pragma once

#include <ReactCommon/CallInvoker.h>
#include <algorithm>
#include <functional>
#include <jsi/jsi.h>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <vector>

namespace onnxruntimejsi {

class Env : public std::enable_shared_from_this<Env> {
 public:
  Env(std::shared_ptr<facebook::react::CallInvoker> jsInvoker)
      : jsInvoker_(jsInvoker) {}

  ~Env() {}

  inline void initOrtEnv(OrtLoggingLevel logLevel, const char* logid) {
    if (ortEnv_) {
      return;
    }
    ortEnv_ = std::make_shared<Ort::Env>(logLevel, logid);
  }

  inline void setTensorConstructor(
      std::shared_ptr<facebook::jsi::WeakObject> tensorConstructor) {
    tensorConstructor_ = tensorConstructor;
  }

  inline facebook::react::CallInvoker* getJsInvoker() const {
    return jsInvoker_.get();
  }
  inline facebook::jsi::Value
  getTensorConstructor(facebook::jsi::Runtime& runtime) const {
    return tensorConstructor_->lock(runtime);
  }

  inline Ort::Env& getOrtEnv() const { return *ortEnv_; }

 private:
  std::shared_ptr<facebook::react::CallInvoker> jsInvoker_;
  std::shared_ptr<facebook::jsi::WeakObject> tensorConstructor_;
  std::shared_ptr<Ort::Env> ortEnv_;
};

}  // namespace onnxruntimejsi
