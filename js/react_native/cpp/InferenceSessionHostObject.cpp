#include "InferenceSessionHostObject.h"
#include "AsyncWorker.h"
#include "JsiUtils.h"
#include "SessionUtils.h"
#include "TensorUtils.h"
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

using namespace facebook::jsi;

namespace onnxruntimejsi {

class InferenceSessionHostObject::LoadModelAsyncWorker : public AsyncWorker {
 public:
  LoadModelAsyncWorker(Runtime& runtime, const Value* arguments, size_t count,
                       std::shared_ptr<InferenceSessionHostObject> session)
      : AsyncWorker(runtime, session->env_), session_(session) {
    if (count < 1)
      throw JSError(runtime, "loadModel requires at least 1 argument");
    if (arguments[0].isString()) {
      modelPath_ = arguments[0].asString(runtime).utf8(runtime);
      if (modelPath_.find("file:/") == 0) {
        modelPath_ = modelPath_.substr(5);
        if (modelPath_.find("//") == 0) {
          modelPath_ = modelPath_.substr(2);
        }
      }
    } else if (arguments[0].isObject() &&
               arguments[0].asObject(runtime).isArrayBuffer(runtime)) {
      auto arrayBufferObj = arguments[0].asObject(runtime);
      auto arrayBuffer = arrayBufferObj.getArrayBuffer(runtime);
      modelData_ = arrayBuffer.data(runtime);
      modelDataLength_ = arrayBuffer.size(runtime);
    } else {
      throw JSError(runtime, "Model path or buffer is required");
    }
    keepValue(runtime, arguments[0]);
    if (count > 1) {
      parseSessionOptions(runtime, arguments[1], sessionOptions_, session->env_,
                          epContextDataRead_);
    }
    abortEpContextDataRead_ = epContextDataRead_;
    // The state stays private to this worker until Ort::Session construction succeeds. Nothing is
    // published to the host object here, so a rejected load releases it deterministically in
    // onReject() instead of leaving it retained until a later dispose or garbage collection.
  }

  ~LoadModelAsyncWorker() override { abortAndJoin(); }

 protected:
  void execute() {
    if (modelPath_.empty()) {
      loadedSession_ = std::make_shared<Ort::Session>(
          session_->env_->getOrtEnv(), modelData_, modelDataLength_,
          sessionOptions_);
    } else {
      loadedSession_ = std::make_shared<Ort::Session>(
          session_->env_->getOrtEnv(), modelPath_.c_str(), sessionOptions_);
    }
  }

  Value onResolve(Runtime& rt) {
    // Publish only on the JS thread so dispose(), reload, and successful construction cannot race
    // while reading or replacing the host object's shared_ptr members.
    session_->session_.reset();
    previousEpContextDataRead_ = std::move(session_->epContextDataRead_);
    session_->epContextDataRead_ = std::move(epContextDataRead_);
    session_->session_ = std::move(loadedSession_);

    // The retired state can no longer be reached by any session; release it here so the JS
    // function is dropped on the JS thread.
    releaseState(previousEpContextDataRead_);
    abortEpContextDataRead_.reset();
    return Value::undefined();
  }

  Value onReject(Runtime& rt, const std::string& err) {
    // Session construction failed, so the state was never published. Release it on the JS thread
    // rather than waiting for this worker to be collected.
    releaseState(epContextDataRead_);
    abortEpContextDataRead_.reset();
    return AsyncWorker::onReject(rt, err);
  }

  void onAbort() override {
    if (abortEpContextDataRead_) {
      abortEpContextDataRead_->invalidate();
    }
  }

 private:
  // Must run on the JS thread: dropping the last reference releases the JS callback function.
  static void
  releaseState(std::shared_ptr<EpContextDataReadCallback>& state) noexcept {
    if (state) {
      state->invalidate();
      state.reset();
    }
  }

  std::string error_;
  std::string modelPath_;
  void* modelData_;
  size_t modelDataLength_;
  std::shared_ptr<InferenceSessionHostObject> session_;
  Ort::SessionOptions sessionOptions_;
  // Strong snapshot of the state referenced by sessionOptions_, owned solely by this worker until
  // the session it belongs to exists.
  std::shared_ptr<EpContextDataReadCallback> epContextDataRead_;
  // Immutable snapshot used by onAbort() to unblock a worker waiting for the JS thread. execute()
  // may move epContextDataRead_ into the host concurrently, so the abort path must not access it.
  std::shared_ptr<EpContextDataReadCallback> abortEpContextDataRead_;
  // State displaced from the host object by a successful reload, kept alive until the JS thread
  // can release it.
  std::shared_ptr<EpContextDataReadCallback> previousEpContextDataRead_;
  // Declared after every callback-state reference so a worker-local session is destroyed first.
  std::shared_ptr<Ort::Session> loadedSession_;
  std::shared_ptr<WeakObject> weakResolve_;
  std::shared_ptr<WeakObject> weakReject_;
  std::thread thread_;
};

DEFINE_METHOD(InferenceSessionHostObject::loadModel) {
  auto self = shared_from_this();
  auto worker =
      std::make_shared<LoadModelAsyncWorker>(runtime, arguments, count, self);
  return worker->toPromise(runtime);
}

class InferenceSessionHostObject::RunAsyncWorker : public AsyncWorker {
 public:
  RunAsyncWorker(Runtime& runtime, const Value* arguments, size_t count,
                 std::shared_ptr<InferenceSessionHostObject> session)
      : AsyncWorker(runtime, session->env_),
        env_(session->env_),
        session_(session->session_),
        memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
    if (count < 1)
      throw JSError(runtime, "run requires at least 1 argument");
    if (count > 2 && !arguments[2].isUndefined()) {
      parseRunOptions(runtime, arguments[2], runOptions_);
    }
    forEach(runtime, arguments[0].asObject(runtime),
            [&](const std::string& key, const Value& value, size_t index) {
              inputNames_.push_back(key);
              inputValues_.push_back(TensorUtils::createOrtValueFromJSTensor(
                  runtime, value.asObject(runtime), memoryInfo_));
              keepValue(runtime, value);
            });
    forEach(runtime, arguments[1].asObject(runtime),
            [&](const std::string& key, const Value& value, size_t index) {
              outputNames_.push_back(key);
              if (value.isObject() &&
                  TensorUtils::isTensor(runtime, value.asObject(runtime))) {
                outputValues_.push_back(TensorUtils::createOrtValueFromJSTensor(
                    runtime, value.asObject(runtime), memoryInfo_));
                jsOutputValues_.push_back(std::make_shared<WeakObject>(
                    runtime, value.asObject(runtime)));
                keepValue(runtime, value);
              } else {
                outputValues_.push_back(Ort::Value());
                jsOutputValues_.push_back(nullptr);
              }
            });
  }

  ~RunAsyncWorker() override { abortAndJoin(); }

 protected:
  void execute() {
    auto inputNames = std::vector<const char*>(inputNames_.size());
    std::transform(inputNames_.begin(), inputNames_.end(), inputNames.begin(),
                   [](const std::string& name) { return name.c_str(); });
    auto outputNames = std::vector<const char*>(outputNames_.size());
    std::transform(outputNames_.begin(), outputNames_.end(),
                   outputNames.begin(),
                   [](const std::string& name) { return name.c_str(); });
    auto session = session_.lock();
    if (!session) {
      throw std::runtime_error("Session is released");
    }
    session->Run(runOptions_, inputNames.data(), inputValues_.data(),
                 inputValues_.size(), outputNames.data(),
                 outputValues_.data(), outputValues_.size());
  }

  Value onResolve(Runtime& rt) {
    auto resultObject = Object(rt);
    auto tensorConstructor =
        env_->getTensorConstructor(rt).asObject(rt);
    for (size_t i = 0; i < outputValues_.size(); ++i) {
      if (jsOutputValues_[i] != nullptr && outputValues_[i].IsTensor()) {
        resultObject.setProperty(rt, outputNames_[i].c_str(),
                                 jsOutputValues_[i]->lock(rt));
      } else {
        auto tensorObj = TensorUtils::createJSTensorFromOrtValue(
            rt, outputValues_[i], tensorConstructor);
        resultObject.setProperty(rt, outputNames_[i].c_str(),
                                 Value(rt, tensorObj));
      }
    }
    return Value(rt, resultObject);
  }

  void onAbort() override {
    runOptions_.SetTerminate();
  }

 private:
  std::shared_ptr<Env> env_;
  std::weak_ptr<Ort::Session> session_;
  Ort::MemoryInfo memoryInfo_;
  Ort::RunOptions runOptions_;
  std::vector<std::string> inputNames_;
  std::vector<Ort::Value> inputValues_;
  std::vector<std::string> outputNames_;
  std::vector<Ort::Value> outputValues_;
  std::vector<std::shared_ptr<WeakObject>> jsOutputValues_;
};

DEFINE_METHOD(InferenceSessionHostObject::run) {
  auto self = shared_from_this();
  auto worker =
      std::make_shared<RunAsyncWorker>(runtime, arguments, count, self);
  return worker->toPromise(runtime);
}

DEFINE_METHOD(InferenceSessionHostObject::dispose) {
  // Release the session first: ONNX Runtime may still call the read callback while it exists.
  session_.reset();
  if (epContextDataRead_) {
    epContextDataRead_->invalidate();
    epContextDataRead_.reset();
  }
  return Value::undefined();
}

DEFINE_METHOD(InferenceSessionHostObject::endProfiling) {
  try {
    Ort::AllocatorWithDefaultOptions allocator;
    auto filename = session_->EndProfilingAllocated(allocator);
    return String::createFromUtf8(runtime, std::string(filename.get()));
  } catch (const std::exception& e) {
    throw JSError(runtime, std::string(e.what()));
  }
}

DEFINE_GETTER(InferenceSessionHostObject::inputMetadata) {
  if (!session_) {
    return Array(runtime, 0);
  }
  try {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputs = session_->GetInputCount();
    auto array = Array(runtime, numInputs);

    for (size_t i = 0; i < numInputs; i++) {
      auto item = Object(runtime);
      auto inputName = session_->GetInputNameAllocated(i, allocator);
      item.setProperty(
          runtime, "name",
          String::createFromUtf8(runtime, std::string(inputName.get())));

      try {
        auto typeInfo = session_->GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

        // Get data type
        auto dataType = tensorInfo.GetElementType();
        item.setProperty(runtime, "type", static_cast<double>(dataType));

        // Get shape
        auto shape = tensorInfo.GetShape();
        auto shapeArray = Array(runtime, shape.size());
        for (size_t j = 0; j < shape.size(); j++) {
          shapeArray.setValueAtIndex(runtime, j,
                                     Value(static_cast<double>(shape[j])));
        }
        item.setProperty(runtime, "shape", shapeArray);

        item.setProperty(runtime, "isTensor", Value(true));

        // symbolicDimensions
        auto symbolicDimensions = tensorInfo.GetSymbolicDimensions();
        auto symbolicDimensionsArray =
            Array(runtime, symbolicDimensions.size());
        for (size_t j = 0; j < symbolicDimensions.size(); j++) {
          symbolicDimensionsArray.setValueAtIndex(
              runtime, j,
              String::createFromUtf8(runtime, symbolicDimensions[j]));
        }
        item.setProperty(runtime, "symbolicDimensions",
                         symbolicDimensionsArray);
      } catch (const std::exception&) {
        // Fallback for unknown types
        item.setProperty(runtime, "type",
                         String::createFromUtf8(runtime, "unknown"));
        item.setProperty(runtime, "shape", Array(runtime, 0));
        item.setProperty(runtime, "isTensor", Value(false));
      }

      array.setValueAtIndex(runtime, i, Value(runtime, item));
    }

    return Value(runtime, array);
  } catch (const Ort::Exception& e) {
    throw JSError(runtime, std::string(e.what()));
  }
}

DEFINE_GETTER(InferenceSessionHostObject::outputMetadata) {
  if (!session_) {
    return Array(runtime, 0);
  }
  try {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numOutputs = session_->GetOutputCount();
    auto array = Array(runtime, numOutputs);

    for (size_t i = 0; i < numOutputs; i++) {
      auto item = Object(runtime);
      auto outputName = session_->GetOutputNameAllocated(i, allocator);
      item.setProperty(
          runtime, "name",
          String::createFromUtf8(runtime, std::string(outputName.get())));

      try {
        auto typeInfo = session_->GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

        // Get data type
        auto dataType = tensorInfo.GetElementType();
        item.setProperty(runtime, "type", static_cast<double>(dataType));

        // Get shape
        auto shape = tensorInfo.GetShape();
        auto shapeArray = Array(runtime, shape.size());
        for (size_t j = 0; j < shape.size(); j++) {
          shapeArray.setValueAtIndex(runtime, j,
                                     Value(static_cast<double>(shape[j])));
        }
        item.setProperty(runtime, "shape", shapeArray);

        item.setProperty(runtime, "isTensor", Value(true));

        // symbolicDimensions
        auto symbolicDimensions = tensorInfo.GetSymbolicDimensions();
        auto symbolicDimensionsArray =
            Array(runtime, symbolicDimensions.size());
        for (size_t j = 0; j < symbolicDimensions.size(); j++) {
          symbolicDimensionsArray.setValueAtIndex(
              runtime, j,
              String::createFromUtf8(runtime, symbolicDimensions[j]));
        }
        item.setProperty(runtime, "symbolicDimensions",
                         symbolicDimensionsArray);
      } catch (const std::exception&) {
        // Fallback for unknown types
        item.setProperty(runtime, "type",
                         String::createFromUtf8(runtime, "unknown"));
        item.setProperty(runtime, "shape", Array(runtime, 0));
        item.setProperty(runtime, "isTensor", Value(false));
      }

      array.setValueAtIndex(runtime, i, Value(runtime, item));
    }

    return Value(runtime, array);
  } catch (const Ort::Exception& e) {
    throw JSError(runtime, std::string(e.what()));
  }
}

}  // namespace onnxruntimejsi
