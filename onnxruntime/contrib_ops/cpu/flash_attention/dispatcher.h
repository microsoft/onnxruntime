#pragma once
#include <atomic>
#include <utility>

namespace onnxruntime {

enum class CPUCapability {
  DEFAULT = 0,
  AVX2 = 1,
  AVX512 = 2,
  NUM_OPTIONS
};

CPUCapability get_cpu_capability();

template <typename FnPtr, typename T>
struct DispatchStub;

struct DispatchStubImpl {
  void* get_call_ptr(
      void* DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
      ,
      void* AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      ,
      void* AVX2
#endif
  );

  /**
   * The CPU Dispatch actual method is chosen in decreasing order of preference by
   * DispatchStubImpl::choose_cpu_impl() in case none is found by
   * DispatchStubImpl::get_call_ptr() in cpu_dispatch_ptr.
   */
  void* choose_cpu_impl(
      void* DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
      ,
      void* AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      ,
      void* AVX2
#endif
  );

  // Fixing dispatch error in Windows debug builds.
  // See https://github.com/pytorch/pytorch/issues/22681 for more details.
#if defined(_MSC_VER) && defined(_DEBUG)
  std::atomic<void*> cpu_dispatch_ptr;
  void* cuda_dispatch_ptr;
  void* hip_dispatch_ptr;
  void* mps_dispatch_ptr;
  void* privateuse1_dispatch_ptr;
#else
  std::atomic<void*> cpu_dispatch_ptr{nullptr};
  void* cuda_dispatch_ptr = nullptr;
  void* hip_dispatch_ptr = nullptr;
  void* mps_dispatch_ptr = nullptr;
  void* privateuse1_dispatch_ptr = nullptr;
#endif
};

template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*)(Args...);

  DispatchStub() = default;
  DispatchStub(const DispatchStub&) = delete;
  DispatchStub& operator=(const DispatchStub&) = delete;

 private:
  FnPtr get_call_ptr(c10::DeviceType device_type) {
    return reinterpret_cast<FnPtr>(
        impl.get_call_ptr(device_type, reinterpret_cast<void*>(DEFAULT)
#ifdef HAVE_AVX512_CPU_DEFINITION
                                           ,
                          reinterpret_cast<void*>(AVX512)
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
                              ,
                          reinterpret_cast<void*>(AVX2)
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
                              ,
                          reinterpret_cast<void*>(VSX)
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
                              ,
                          reinterpret_cast<void*>(ZVECTOR)
#endif
                              ));
  }

 public:
  template <typename... ArgTypes>
  rT operator()(c10::DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...);
  }

  void set_cuda_dispatch_ptr(FnPtr fn_ptr) {
    impl.cuda_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_hip_dispatch_ptr(FnPtr fn_ptr) {
    impl.hip_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_mps_dispatch_ptr(FnPtr fn_ptr) {
    impl.mps_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_privateuse1_dispatch_ptr(FnPtr fn_ptr) {
    impl.privateuse1_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  static TORCH_API FnPtr DEFAULT;
#ifdef HAVE_AVX512_CPU_DEFINITION
  static TORCH_API FnPtr AVX512;
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  static TORCH_API FnPtr AVX2;
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  static TORCH_API FnPtr VSX;
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  static TORCH_API FnPtr ZVECTOR;
#endif
 private:
  DispatchStubImpl impl;
};

// Compiler will complain if you put things like std::tuple<Tensor, Tensor> in
// the `fn` argument of DECLARE_DISPATCH. Some possible workarounds, e.g.,
// adding parentheses and using helper struct to get rid of the parentheses, do
// not work with MSVC. So do a `using`-declaration if you need to pass in such
// `fn`, e.g., grid_sampler_2d_backward_cpu_kernel in GridSampleKernel.h.
#define DECLARE_DISPATCH(fn, name)                                                         \
  struct name##_DECLARE_DISPATCH_type : DispatchStub<fn, name##_DECLARE_DISPATCH_type> {   \
    name##_DECLARE_DISPATCH_type() = default;                                              \
    name##_DECLARE_DISPATCH_type(const name##_DECLARE_DISPATCH_type&) = delete;            \
    name##_DECLARE_DISPATCH_type& operator=(const name##_DECLARE_DISPATCH_type&) = delete; \
  };                                                                                       \
  extern TORCH_API struct name##_DECLARE_DISPATCH_type name;

#define DEFINE_DISPATCH(name) struct name##_DECLARE_DISPATCH_type name

#define REGISTER_ARCH_DISPATCH(name, arch, fn) \
  template <>                                  \
  name##_DECLARE_DISPATCH_type::FnPtr TORCH_API DispatchStub<name##_DECLARE_DISPATCH_type::FnPtr, struct name##_DECLARE_DISPATCH_type>::arch = fn;

#ifdef HAVE_AVX512_CPU_DEFINITION
#define REGISTER_AVX512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, AVX512, fn)
#else
#define REGISTER_AVX512_DISPATCH(name, fn)
#endif

#ifdef HAVE_AVX2_CPU_DEFINITION
#define REGISTER_AVX2_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, AVX2, fn)
#else
#define REGISTER_AVX2_DISPATCH(name, fn)
#endif

// Macro to register the same kernel for all CPU arch types. This is useful
// if a kernel does not benefit from being recompiled across different arch types.
#define REGISTER_ALL_CPU_DISPATCH(name, fn) \
  REGISTER_ARCH_DISPATCH(name, DEFAULT, fn) \
  REGISTER_AVX512_DISPATCH(name, fn)        \
  REGISTER_AVX2_DISPATCH(name, fn)

#if defined(CPU_CAPABILITY)
// REGISTER_DISPATCH now dispatches an AVX512 kernel to nullptr but registers other dispatches.
// ALSO_REGISTER_AVX512_DISPATCH should be used for ensuring AVX512 dispatch, among others.
#ifdef CPU_CAPABILITY_AVX512
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, nullptr)
#else
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif

#define ALSO_REGISTER_AVX512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)

#endif

}  // namespace onnxruntime
