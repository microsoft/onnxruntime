#pragma once

#include "Env.h"
#include "log.h"
#include <atomic>
#include <jsi/jsi.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace facebook::jsi;

namespace onnxruntimejsi {

/**
 * @brief AsyncWorker is a helper class to run a function asynchronously and
 * return a promise.
 *
 * @param rt The runtime to use.
 * @param env The environment to use.
 */
class AsyncWorker : public std::enable_shared_from_this<AsyncWorker> {
 public:
  AsyncWorker(Runtime& rt, std::shared_ptr<Env> env) : rt_(rt), env_(env) {}

  ~AsyncWorker() {
    if (worker_.joinable()) {
      if (worker_.get_id() == std::this_thread::get_id()) {
        worker_.detach();
      } else {
        worker_.join();
      }
    }
  }

  /**
   * @brief Make sure the value won't be garbage collected during the async
   * operation.
   *
   * @param rt The runtime to use.
   * @param value The value to keep.
   */
  void keepValue(Runtime& rt, const Value& value) {
    keptValues_.push_back(std::make_shared<Value>(rt, value));
  }

  /**
   * @brief Create a promise to be used in the async operation.
   *
   * @param rt The runtime to use.
   * @return The promise.
   */
  Value toPromise(Runtime& rt) {
    auto promiseCtor = rt.global().getPropertyAsFunction(rt, "Promise");
    auto self = shared_from_this();

    return promiseCtor.callAsConstructor(
        rt, Function::createFromHostFunction(
                rt, PropNameID::forAscii(rt, "executor"), 2,
                [self](Runtime& rt, const Value& thisVal, const Value* args,
                       size_t count) -> Value {
                  self->resolveFunc_ = std::make_shared<Value>(rt, args[0]);
                  self->rejectFunc_ = std::make_shared<Value>(rt, args[1]);
                  self->worker_ = std::thread([self]() {
                    try {
                      self->execute();
                      self->dispatchResolve();
                    } catch (const std::exception& e) {
                      self->dispatchReject(e.what());
                    }
                  });
                  return Value::undefined();
                }));
  }

 protected:
  virtual void execute() = 0;

  virtual Value onResolve(Runtime& rt) = 0;
  virtual Value onReject(Runtime& rt, const std::string& err) {
    return String::createFromUtf8(rt, err);
  }

 private:
  void dispatchResolve() {
    auto self = shared_from_this();
    env_->getJsInvoker()->invokeAsync([self]() {
      auto resVal = self->onResolve(self->rt_);
      self->resolveFunc_->asObject(self->rt_)
          .asFunction(self->rt_)
          .call(self->rt_, resVal);
      self->clearKeeps();
    });
  }

  void dispatchReject(const std::string& err) {
    auto self = shared_from_this();
    env_->getJsInvoker()->invokeAsync([self, err]() {
      auto resVal = self->onReject(self->rt_, err);
      self->rejectFunc_->asObject(self->rt_)
          .asFunction(self->rt_)
          .call(self->rt_, resVal);
      self->clearKeeps();
    });
  }

  void clearKeeps() {
    keptValues_.clear();
    resolveFunc_.reset();
    rejectFunc_.reset();
  }

  Runtime& rt_;
  std::shared_ptr<Env> env_;
  std::vector<std::shared_ptr<Value>> keptValues_;
  std::shared_ptr<Value> resolveFunc_;
  std::shared_ptr<Value> rejectFunc_;
  std::thread worker_;
};

}  // namespace onnxruntimejsi
