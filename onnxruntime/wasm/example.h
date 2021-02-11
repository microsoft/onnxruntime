#pragma once

#include <emscripten.h>
#include <emscripten/bind.h>

class Example {
  public:
    void Run(const emscripten::val& model_data);
};

EMSCRIPTEN_BINDINGS(Example) {
  emscripten::class_<Example>("Example")
    .constructor()
      .function("Run", &Example::Run);
}
