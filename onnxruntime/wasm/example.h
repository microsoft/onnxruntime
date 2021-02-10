#pragma once

#include <emscripten.h>
#include <emscripten/bind.h>

class Example {
  public:
    void Run(int multiplier);
};

EMSCRIPTEN_BINDINGS(Example) {
  emscripten::class_<Example>("Example")
    .constructor()
      .function("Run", &Example::Run);
}
