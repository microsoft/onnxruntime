// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

namespace onnxruntime {
/// <summary>
/// This class serves as a callable entity that can be used
/// either in a threadpool or as a callback for async operations. It supports both
/// plain functions as well as member functions and provides type
/// erasure functionality. Unlike std::function, it
/// does NOT do any memory allocations and it is cheap to copy.
/// </summary>
///
/// <example>
/// Plain function:
///   int g(void* context, float);
///   Callable<int, float> cb(nullptr, g);
///   int result = cb.Invoke(1.2f);
///
/// Class method:
/// class A {
///    int f(float);
/// };
///
/// A a;
/// CallableFactory<A, int, float> f(&a);
/// - We get back Callable<int, float> as above but call instance method
/// auto cb = f.GetCallable<&A::f>();
/// int result = cb.Invoke(1.2f);
/// </example>
///
/// <typeparam name="Result">Call back result type</typeparam>
/// <typeparam name="...Args">Types of args for the callable</typeparam>
template <typename Result,
          typename... Args>
class Callable {
 public:
  using ContextType = void;

  using ResultType = Result;

  using FunctionType = ResultType (*)(ContextType*, Args...);

  // You can use this directly if you have a simple
  // function like C-function or a stateless lambda
  // For methods we want to make use of AsyncMethodCallback
  // And covert it to a Callable
  Callable(ContextType* context,
           FunctionType func) noexcept : context_(context),
                                         func_(func) {
  }

  Callable() noexcept = default;

  Callable(const Callable&) noexcept = default;
  Callable& operator=(const Callable&) noexcept = default;

  Callable(Callable&& o) noexcept : context_(o.context_), func_(o.func_) {
    o.Clear();
  }

  Callable& operator=(Callable&& o) noexcept {
    context_ = o.context_;
    func_ = o.func_;
    o.Clear();
    return *this;
  }

  void Clear() noexcept {
    context_ = nullptr;
    func_ = nullptr;
  }

  operator bool() const noexcept { return func_ != nullptr; }

  bool Valid() const noexcept { return func_ != nullptr; }

  ContextType* GetContext() const noexcept { return context_; }

  FunctionType GetFunctionPtr() const noexcept { return func_; }

  // Differentiate instantiation on the presence
  // of return value. The below two overloads
  // must be mutually exclusive
  template <typename T = ResultType>
  typename std::enable_if<!std::is_same<void, T>::value, T>::type
  Invoke(Args... args) const {
    return func_(context_, std::forward<Args>(args)...);
  }

  template <typename T = ResultType>
  typename std::enable_if<std::is_same<void, T>::value, void>::type
  Invoke(Args... args) const {
    func_(context_, std::forward<Args>(args)...);
  }

 private:
  ContextType* context_ = nullptr;
  FunctionType func_ = nullptr;
};

template <typename ObjectType,
          typename ResultType,
          typename... Args>
class CallableFactory {
 public:
  using CallableType = Callable<ResultType, Args...>;

  explicit CallableFactory(ObjectType* obj) noexcept : obj_(obj) {
  }

  template <typename ResultType2,
            ResultType2 (ObjectType::*MethodPtr)(Args...)>
  struct Binder {
    static ResultType2 InvokeMethod(void* obj, Args... args) {
      return (static_cast<ObjectType*>(obj)->*MethodPtr)(
          std::forward<Args>(args)...);
    }
  };

  // UnaVOIDable void specialization
  template <ResultType (ObjectType::*MethodPtr)(Args...)>
  struct Binder<void, MethodPtr> {
    static void InvokeMethod(void* obj, Args... args) {
      (static_cast<ObjectType*>(obj)->*MethodPtr)(
          std::forward<Args>(args)...);
    }
  };

  template <ResultType (ObjectType::*MethodPtr)(Args...)>
  CallableType GetCallable() const noexcept {
    CallableType result(obj_, &Binder<ResultType, MethodPtr>::InvokeMethod);
    return result;
  }

 private:
  ObjectType* obj_;
};

}  // namespace onnxruntime
