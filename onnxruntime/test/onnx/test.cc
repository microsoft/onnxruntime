#include "callables.h"
#include <memory>

struct A {
  void g(size_t) {
  }
};

using Callback = onnxruntime::test::Callable<void, size_t>;

void f() {
  std::unique_ptr<A> p(new A());

  onnxruntime::test::CallableFactory<A, void, size_t> f(p.get());
  auto cb = f.GetCallable<&A::g>();

  cb.Invoke(13U);
}