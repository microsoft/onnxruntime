#include "JsiMain.h"
#include "InferenceSessionHostObject.h"
#include "JsiHelper.hpp"
#include "SessionUtils.h"
#include <memory>

using namespace facebook::jsi;

namespace onnxruntimejsi {

std::shared_ptr<Env>
install(Runtime& runtime,
        std::shared_ptr<facebook::react::CallInvoker> jsInvoker) {
  auto env = std::make_shared<Env>(jsInvoker);
  try {
    auto ortApi = Object(runtime);

    auto initOrtOnceMethod = Function::createFromHostFunction(
        runtime, PropNameID::forAscii(runtime, "initOrtOnce"), 2,
        [env](Runtime& runtime, const Value& thisValue, const Value* arguments,
              size_t count) -> Value {
          try {
            OrtLoggingLevel logLevel = ORT_LOGGING_LEVEL_WARNING;
            if (count > 0 && arguments[0].isNumber()) {
              int level = static_cast<int>(arguments[0].asNumber());
              switch (level) {
                case 0:
                  logLevel = ORT_LOGGING_LEVEL_VERBOSE;
                  break;
                case 1:
                  logLevel = ORT_LOGGING_LEVEL_INFO;
                  break;
                case 2:
                  logLevel = ORT_LOGGING_LEVEL_WARNING;
                  break;
                case 3:
                  logLevel = ORT_LOGGING_LEVEL_ERROR;
                  break;
                case 4:
                  logLevel = ORT_LOGGING_LEVEL_FATAL;
                  break;
                default:
                  logLevel = ORT_LOGGING_LEVEL_WARNING;
                  break;
              }
            }
            env->setTensorConstructor(std::make_shared<WeakObject>(
                runtime, arguments[1].asObject(runtime)));
            env->initOrtEnv(logLevel, "onnxruntime-react-native-jsi");
            return Value::undefined();
          } catch (const std::exception& e) {
            throw JSError(runtime, "Failed to initialize ONNX Runtime: " +
                                       std::string(e.what()));
          }
        });

    ortApi.setProperty(runtime, "initOrtOnce", initOrtOnceMethod);

    auto createInferenceSessionMethod = Function::createFromHostFunction(
        runtime, PropNameID::forAscii(runtime, "createInferenceSession"), 0,
        std::bind(InferenceSessionHostObject::constructor, env,
                  std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4));
    ortApi.setProperty(runtime, "createInferenceSession",
                       createInferenceSessionMethod);

    auto listSupportedBackendsMethod = Function::createFromHostFunction(
        runtime, PropNameID::forAscii(runtime, "listSupportedBackends"), 0,
        [](Runtime& runtime, const Value& thisValue, const Value* arguments,
           size_t count) -> Value {
          auto backends = Array(runtime, supportedBackends.size());
          for (size_t i = 0; i < supportedBackends.size(); i++) {
            auto backend = Object(runtime);
            backend.setProperty(
                runtime, "name",
                String::createFromUtf8(runtime, supportedBackends[i]));
            backends.setValueAtIndex(runtime, i, Value(runtime, backend));
          }
          return Value(runtime, backends);
        });

    ortApi.setProperty(runtime, "listSupportedBackends",
                       listSupportedBackendsMethod);

    ortApi.setProperty(
        runtime, "version",
        String::createFromUtf8(runtime, OrtGetApiBase()->GetVersionString()));

    runtime.global().setProperty(runtime, "OrtApi", ortApi);
  } catch (const std::exception& e) {
    throw JSError(runtime, "Failed to install ONNX Runtime JSI bindings: " +
                               std::string(e.what()));
  }

  return env;
}

}  // namespace onnxruntimejsi
