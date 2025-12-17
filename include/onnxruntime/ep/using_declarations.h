//
// EP specific using declarations
//

#define EP_SPECIFIC_USING_DECLARATIONS                                             \
  using FuncManager = onnxruntime::ep::detail::FuncManager;                        \
  using KernelCreatePtrFn = onnxruntime::ep::detail::KernelCreatePtrFn;            \
  using KernelDefBuilder = onnxruntime::ep::detail::KernelDefBuilder;              \
  using KernelRegistry = onnxruntime::ep::detail::KernelRegistry;                  \
  using KernelCreateInfo = onnxruntime::ep::detail::KernelCreateInfo;              \
  using BuildKernelCreateInfoFn = onnxruntime::ep::detail::KernelCreateInfo (*)(); \
  using OpKernelInfo = onnxruntime::ep::detail::OpKernelInfo;                      \
  using OpKernelContext = onnxruntime::ep::detail::OpKernelContext;                \
  using OpKernel = onnxruntime::ep::detail::OpKernel;                              \
  using DataTransferManager = onnxruntime::ep::detail::DataTransferManager;
