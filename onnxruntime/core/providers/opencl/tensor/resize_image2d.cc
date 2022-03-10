#include "resize.h"

#include "core/providers/cpu/tensor/upsample.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME resize_kernel_src
#include "opencl_generated/tensor/kernels/resize_image2d.cl.inc"
}  // namespace

class Resize : public OpenCLKernel, UpsampleBase {
 public:
  explicit Resize(const OpKernelInfo& info)
      : OpenCLKernel(info), UpsampleBase(info) {
    VLOGS_DEFAULT(0) << "Init Resize (OpenCLKernel)";
    LoadProgram(resize_kernel_src, resize_kernel_src_len);
    LoadKernel("ResizeBilinear2D");
  };

  Status Compute(OpKernelContext* context) const override {
    ZoneScopedN("Resize::Compute");
    VLOG_CL_NODE();
    ORT_RETURN_IF(mode_ != UpsampleMode::LINEAR, "only supports linear interpolation");

    const auto* X = context->Input<Tensor>(0);
    const auto& X_shape = X->Shape();
    ORT_RETURN_IF(X_shape.NumDimensions() != 4, "only support 4D NCHW input");
    TensorShapeVector Y_shape(X->Shape().GetDims().size());
    if (scales_cached_) {
      ComputeOutputShape(scales_, X_shape.GetDims(), Y_shape);
      const auto* Y = context->Output(0, Y_shape);
      return LaunchResize2D(X, Y, scales_[3], scales_[2]);
    }

    const auto* scales_data = context->Input<Tensor>(scales_input_idx_);
    const auto* sizes = context->Input<Tensor>(sizes_input_idx_);
    // Get scales data
    std::vector<float> scales_array(X->Shape().GetDims().size());
    if (scales_data != nullptr && scales_data->Shape().Size() != 0) {
      ORT_ENFORCE(sizes == nullptr, "Only one of scales or sizes must be provided as input.");
      ParseScalesData(scales_data, scales_array);
      ComputeOutputShape(scales_array, X_shape.GetDims(), Y_shape);
    } else {
      ORT_ENFORCE(sizes != nullptr && sizes->Shape().Size() != 0, "Either scales or sizes MUST be provided as input.");

      // When sizes input is available directly populate it into the output_dims array.
      memcpy(Y_shape.data(), sizes->Data<int64_t>(), sizes->Shape().Size() * sizeof(int64_t));

      ORT_ENFORCE(X->Shape().GetDims().size() == Y_shape.size(),
                  "Resize: input tensor's rank does not match the output tensor's rank.");
      ParseScalesDataFromOutputSize(Y_shape, X->Shape().GetDims(), scales_array);
    }
    const auto* Y = context->Output(0, Y_shape);
    return LaunchResize2D(X, Y, scales_array[3], scales_array[2]);
  }

  Status LaunchResize2D(const Tensor* X, const Tensor* Y, float scale_x, float scale_y) const {
    VLOG_CL_IMAGE2D("Input", X);
    VLOG_CL_IMAGE2D("Output", Y);

    const auto& X_shape = X->Shape();
    const auto& Y_shape = Y->Shape();

    auto desc = Image2DDesc::PackFromTensorNCHW(Y->Shape());
    ZoneNamedN(_tracy_Resize, "Resize (kernel launch)", true);
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("ResizeBilinear2D")}
            .setInt2(desc.Width(), desc.Height())
            .setImage2Ds(*X, *Y)
            .setInt2(X_shape[3], X_shape[2])
            .setInt2(Y_shape[3], Y_shape[2])
            .setArg<cl_float>(1.0 / scale_x)
            .setArg<cl_float>(1.0 / scale_y)
            .setArg<cl_int>(coordinate_transform_mode_)
            .Launch(*exec_, desc.AsNDRange()));

    return Status::OK();
  }
};

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Resize,
    kOnnxDomain,
    11, 12,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, {1, 2, 3}),
    Resize)

ONNX_OPENCL_OPERATOR_KERNEL(
    Resize,
    13,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, {1, 2, 3}),
    Resize)

}  // namespace opencl
}  // namespace onnxruntime
