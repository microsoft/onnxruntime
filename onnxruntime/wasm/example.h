#pragma once

#include <emscripten.h>
#include <emscripten/bind.h>

namespace onnxruntime {
namespace logging {
  class LoggingManager;
}
class CPUExecutionProvider;
class Model;
class Node;
}

class Example {
  public:
    Example();
    ~Example() = default;

    bool Load(const emscripten::val& model_data);
    bool Run();

  private:
    Example(const Example&) = delete;
    Example& operator=(const Example&) = delete;
    Example(Example&&) = delete;
    Example& operator=(Example&&) = delete;

    std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager_;
    std::shared_ptr<onnxruntime::Model> model_;
    std::unique_ptr<onnxruntime::CPUExecutionProvider> cpu_execution_provider_;
    std::vector<const onnxruntime::Node*> nodes_;
};

EMSCRIPTEN_BINDINGS(Example) {
  emscripten::class_<Example>("Example")
    .constructor()
      .function("Load", &Example::Load)
      .function("Run", &Example::Run);
}
