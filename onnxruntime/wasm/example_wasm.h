#pragma once

#include <emscripten.h>
#include <emscripten/bind.h>

namespace Ort {
  struct Env;
  struct Session;
}

class Example {
  public:
    Example() = default;
    ~Example() = default;

    bool Load(const emscripten::val& model_data);
    bool Run();

  private:
    Example(const Example&) = delete;
    Example& operator=(const Example&) = delete;
    Example(Example&&) = delete;
    Example& operator=(Example&&) = delete;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};

EMSCRIPTEN_BINDINGS(Example) {
  emscripten::class_<Example>("Example")
    .constructor()
      .function("Load", &Example::Load)
      .function("Run", &Example::Run);
}
