#pragma once

#if !defined(BUILD_NATIVE)
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

namespace Ort {
  struct Env;
  struct Session;
}

class Example {
  public:
    Example() = default;
    ~Example() = default;

#if defined(BUILD_NATIVE)
    bool Load(const std::string& model_path);
#else
    bool Load(const emscripten::val& model_data);
#endif
    bool Run();

  private:
    Example(const Example&) = delete;
    Example& operator=(const Example&) = delete;
    Example(Example&&) = delete;
    Example& operator=(Example&&) = delete;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};

#if !defined(BUILD_NATIVE)
EMSCRIPTEN_BINDINGS(Example) {
  emscripten::class_<Example>("Example")
    .constructor()
      .function("Load", &Example::Load)
      .function("Run", &Example::Run);
}
#endif