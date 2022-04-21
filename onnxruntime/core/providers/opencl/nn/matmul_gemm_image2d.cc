// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <CL/cl.h>
#include "matmul_gemm.h"

#include <sstream>

#include "core/framework/tensorprotoutils.h"
#include "core/providers/opencl/opencl_allocator.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_data_transfer.h"
#include "core/providers/opencl/opencl_execution_provider.h"
#include "core/providers/opencl/opencl_program_manager.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "contrib_ops/cpu/fused_activation.h"
namespace {

namespace kernel_name {
constexpr const char* MatMul_KR = "MatMul_KR";
constexpr const char* MatMul_NOKR = "MatMul_NOKR";
constexpr const char* MatMulU8 = "MatMulU8";
}  // namespace kernel_name

}  // namespace

namespace onnxruntime {
namespace opencl {
    
class MatMulCreater {
 public:
  static std::string GenerateCodeAlongKsize(int64_t K_size) {
    std::ostringstream oss;

    oss << R"_CODE(
    // matrix_a format: image:(K, batch_a M/4,)
    // matrix_b format: image:(N/4, batch_b K, )
    // matrix_c format: image:(N, batch_c * M/4)
    )_CODE";
    oss << "\n#define DEAL_NON_UNIFORM_DIM2(input1, input2) "
        << "if (input1 >= global_size_dim0 || input2 >= global_size_dim1) {return;}\n";
    oss << R"_CODE(__kernel void )_CODE";
    oss << (K_size%4>0 ? "MatMul_KR":"MatMul_NOKR")
        << R"_CODE((__private const int global_size_dim0,
                  __private const int global_size_dim1,
                  __read_only image2d_t matrix_a,
                  __read_only image2d_t matrix_b,
                  __write_only image2d_t matrix_c,
                  __private const int M_size,
                  __private const int K_size,
                  __private const int N_size,
                  __private const int m_blocks,
                  __private const int batch_a,
                  __private const int batch_b) {
      const int image_col_block = get_global_id(0);
      const int image_row_block = get_global_id(1);
      DEAL_NON_UNIFORM_DIM2(image_col_block, image_row_block);
      const int out_image_col_block = image_col_block * 4;
      int batch_c_idx = image_row_block / m_blocks;
      int m_idx = image_row_block % m_blocks;
      int batch_a_idx = select(0, batch_c_idx, batch_c_idx < batch_a);
      int batch_b_idx = select(0, batch_c_idx, batch_c_idx < batch_b);
      FLOAT4 matrix_a_data_0, matrix_a_data_1, matrix_a_data_2, matrix_a_data_3;
      FLOAT4 matrix_b_data_0, matrix_b_data_1, matrix_b_data_2, matrix_b_data_3;
      FLOAT4 sum0 = (FLOAT4)0,sum1 = (FLOAT4)0,sum2 = (FLOAT4)0,sum3 = (FLOAT4)0;
    int matrix_a_y_idx = batch_a_idx * m_blocks + m_idx;
    int matrix_b_y_offset = batch_b_idx * K_size;
    int i=0;
    for (; i <= K_size-4; i+=4) {
        matrix_a_data_0 = RI_F(matrix_a, (int2)(i, matrix_a_y_idx));
        matrix_a_data_1 = RI_F(matrix_a, (int2)(i+1, matrix_a_y_idx));
        matrix_a_data_2 = RI_F(matrix_a, (int2)(i+2, matrix_a_y_idx));
        matrix_a_data_3 = RI_F(matrix_a, (int2)(i+3, matrix_a_y_idx));

        const int matrix_b_y_offset_i= matrix_b_y_offset+i;
        matrix_b_data_0 = RI_F(matrix_b, (int2)(image_col_block, matrix_b_y_offset_i));
        matrix_b_data_1 = RI_F(matrix_b, (int2)(image_col_block, matrix_b_y_offset_i + 1));
        matrix_b_data_2 = RI_F(matrix_b, (int2)(image_col_block, matrix_b_y_offset_i + 2));
        matrix_b_data_3 = RI_F(matrix_b, (int2)(image_col_block, matrix_b_y_offset_i + 3));

        sum0 = mad(matrix_a_data_0.x, matrix_b_data_0, sum0);
        sum1 = mad(matrix_a_data_0.y, matrix_b_data_0, sum1);
        sum2 = mad(matrix_a_data_0.z, matrix_b_data_0, sum2);
        sum3 = mad(matrix_a_data_0.w, matrix_b_data_0, sum3);

        sum0 = mad(matrix_a_data_1.x, matrix_b_data_1, sum0);
        sum1 = mad(matrix_a_data_1.y, matrix_b_data_1, sum1);
        sum2 = mad(matrix_a_data_1.z, matrix_b_data_1, sum2);
        sum3 = mad(matrix_a_data_1.w, matrix_b_data_1, sum3);

        sum0 = mad(matrix_a_data_2.x, matrix_b_data_2, sum0);
        sum1 = mad(matrix_a_data_2.y, matrix_b_data_2, sum1);
        sum2 = mad(matrix_a_data_2.z, matrix_b_data_2, sum2);
        sum3 = mad(matrix_a_data_2.w, matrix_b_data_2, sum3);

        sum0 = mad(matrix_a_data_3.x, matrix_b_data_3, sum0);
        sum1 = mad(matrix_a_data_3.y, matrix_b_data_3, sum1);
        sum2 = mad(matrix_a_data_3.z, matrix_b_data_3, sum2);
        sum3 = mad(matrix_a_data_3.w, matrix_b_data_3, sum3);
    }
    )_CODE";
    if (K_size % 4) {
      oss << R"_CODE(
        int k_remains=K_size%4;
        matrix_b_y_offset+=i;
        matrix_a_data_0 = RI_F(matrix_a, (int2)(i, matrix_a_y_idx));
        matrix_a_data_1 = RI_F(matrix_a, (int2)(i+1, matrix_a_y_idx));
        matrix_a_data_2 = RI_F(matrix_a, (int2)(i+2, matrix_a_y_idx));

        matrix_b_data_0 = k_remains>0?RI_F(matrix_b, (int2)(image_col_block, matrix_b_y_offset)):(FLOAT4)0;
        matrix_b_data_1 = k_remains>1?RI_F(matrix_b, (int2)(image_col_block, matrix_b_y_offset + 1)):(FLOAT4)0;
        matrix_b_data_2 = k_remains>2?RI_F(matrix_b, (int2)(image_col_block, matrix_b_y_offset + 2)):(FLOAT4)0;

        sum0 = mad(matrix_a_data_0.x, matrix_b_data_0, sum0);
        sum1 = mad(matrix_a_data_0.y, matrix_b_data_0, sum1);
        sum2 = mad(matrix_a_data_0.z, matrix_b_data_0, sum2);
        sum3 = mad(matrix_a_data_0.w, matrix_b_data_0, sum3);

        sum0 = mad(matrix_a_data_1.x, matrix_b_data_1, sum0);
        sum1 = mad(matrix_a_data_1.y, matrix_b_data_1, sum1);
        sum2 = mad(matrix_a_data_1.z, matrix_b_data_1, sum2);
        sum3 = mad(matrix_a_data_1.w, matrix_b_data_1, sum3);

        sum0 = mad(matrix_a_data_2.x, matrix_b_data_2, sum0);
        sum1 = mad(matrix_a_data_2.y, matrix_b_data_2, sum1);
        sum2 = mad(matrix_a_data_2.z, matrix_b_data_2, sum2);
        sum3 = mad(matrix_a_data_2.w, matrix_b_data_2, sum3);
      )_CODE";
    }
    oss << R"_CODE(
    matrix_a_data_0 = (FLOAT4)(sum0.x, sum1.x, sum2.x, sum3.x);
    matrix_a_data_1 = (FLOAT4)(sum0.y, sum1.y, sum2.y, sum3.y);
    matrix_a_data_2 = (FLOAT4)(sum0.z, sum1.z, sum2.z, sum3.z);
    matrix_a_data_3 = (FLOAT4)(sum0.w, sum1.w, sum2.w, sum3.w);

    WI_F(matrix_c, (int2)(out_image_col_block,   image_row_block), matrix_a_data_0);
    if (out_image_col_block+1<N_size){
        WI_F(matrix_c, (int2)(out_image_col_block+1, image_row_block), matrix_a_data_1);
    }
    if (out_image_col_block+2<N_size){
        WI_F(matrix_c, (int2)(out_image_col_block+2, image_row_block), matrix_a_data_2);
    }
    if (out_image_col_block+3<N_size){
        WI_F(matrix_c, (int2)(out_image_col_block+3, image_row_block), matrix_a_data_3);
    }
    }
    )_CODE";
    return oss.str();
    };
};
// TODO: This is shared across C++ code and opencl kernel code
// unify them in a shared header
typedef MLAS_ACTIVATION_KIND ActivationKind;

class MatMul : public OpenCLKernel {
 public:
  explicit MatMul(const OpKernelInfo& info) : OpenCLKernel(info) {
    ORT_ENFORCE(GetFusedActivationAttr(info, act_info_).IsOK());
    //we might want to support Matmul_transpose or tranpose_matmul,
  };

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr /*alloc*/,
                 bool& is_packed, PrePackedWeights* /*prepacked_weights*/) override {
    is_packed = false;
    // only kernel weight is PrePack-ed
    if (input_idx == 1) {
      ORT_RETURN_IF_ERROR(PackMatMulWeight(tensor));
      // only kernel weight is PrePack-ed
      W_shape_ = tensor.Shape();
      size_t k_size = W_shape_.NumDimensions() == 3 ? W_shape_[1] : W_shape_[0];
      std::string code = MatMulCreater::GenerateCodeAlongKsize(k_size);
      LoadProgram(code.c_str(), code.size());
      if(k_size%4>0){
        LoadKernel(kernel_name::MatMul_KR);
      } else {
        LoadKernel(kernel_name::MatMul_NOKR);
      }
      is_packed = true;
    }
    return Status::OK();
  }

  Status Compute(OpKernelContext* context) const override {
    ZoneScopedN("MatMul::Compute");
    VLOG_CL_NODE();
    //size_t num_inputs = OpKernel::Node().InputDefs().size();
    const Tensor* X = context->Input<Tensor>(0);
    int64_t dim_x = X->Shape().NumDimensions();
    ORT_ENFORCE(dim_x == 3 || dim_x == 2);

    //const Tensor* B = num_inputs >= 3 ? context->Input<Tensor>(2) : nullptr;
    //const Tensor* Sum = num_inputs >= 4 ? context->Input<Tensor>(3) : nullptr;

    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(helper.Compute(X->Shape(), W_shape_));
    Tensor* Y = context->Output(0, helper.OutputShape());

    // Bail out early if the output is going to be empty
    if (Y->Shape().Size() == 0)
      return Status::OK();
    auto mat_k = X->Shape()[--dim_x];
    auto mat_m = X->Shape()[--dim_x];
    auto batch_x = (--dim_x) < 0 ? 1 : X->Shape()[0];
    auto batch_w = W_shape_.NumDimensions() == 2 ? 1 : W_shape_[0];
    auto batch_out = Y->Shape().NumDimensions() == 2 ? 1 : Y->Shape()[0];
    auto mat_n = Y->Shape()[Y->Shape().NumDimensions() - 1];
    
    uint32_t gsx = gsl::narrow_cast<uint32_t>(CeilDiv(mat_n, 4));
    uint32_t gsy = gsl::narrow_cast<uint32_t>(batch_out * CeilDiv(mat_m, 4));
    {
      ZoneScopedN("MatMul (kernel launch)");
      const char* clkernel_name = kernel_name::MatMul_KR;
      if (mat_k % 4 == 0) {
        clkernel_name = kernel_name::MatMul_NOKR;
      }
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel(clkernel_name)}
              .SetArg<cl_int>(gsx)
              .SetArg<cl_int>(gsy)
              .SetImage2Ds(*X, static_cast<cl_mem>(packed_weight_.get()), *Y)
              .SetArg<cl_int>(mat_m)  // m
              .SetArg<cl_int>(mat_k)  // k
              .SetArg<cl_int>(mat_n)  // n
              .SetArg<cl_int>(CeilDiv(mat_m, 4))
              .SetArg<cl_int>(batch_x)
              .SetArg<cl_int>(batch_w)
              .Launch(*exec_, {gsx, gsy}));
    }
    return Status::OK();
  }

 private:
  Status PackMatMulWeight(const Tensor& src) {
    ZoneScopedN("PackMatMulWeight");
    auto shape = src.Shape();
    auto desc = Image2DDesc::PackFromMatMulWeight(shape);
    packed_weight_ = exec_->GetScratchImage2D(desc);
    CL_CHECK_MEM_OBJECT_IS_IMAGE_2D(GetPackedWeight());
    VLOGF_DEFAULT(V_COPY, "Copy    host(%p) --> Image2D(%p)", src.DataRaw(), GetPackedWeight());
    std::shared_ptr<float> out;
    auto tmp = exec_->GetScratchBuffer(src.SizeInBytes());
    {
      int64_t dim = shape.NumDimensions();
      ORT_ENFORCE(dim == 3 || dim == 2);
      dim--;
      size_t mat_n = shape[dim--];
      size_t k_dim = shape[dim--];
      size_t batch = dim < 0 ? 1 : shape[dim];
      const float* from = src.Data<float>();
      size_t linear = 0;
      size_t mat_n_ceil_4 = CeilDiv(mat_n, 4);
      out = std::shared_ptr<float>(new float[batch * mat_n_ceil_4 * 4 * k_dim], [](float* p) { delete[] p; });
      for (size_t bz = 0; bz < batch; ++bz) {
        size_t batch_offset = bz * (mat_n_ceil_4 * 4) * k_dim;
        for (size_t i = 0; i < k_dim; ++i) {
          for (size_t j = 0; j < mat_n_ceil_4; j++) {
            for (size_t k = 0; k < 4; ++k) {
              linear = batch_offset + i * (mat_n_ceil_4 * 4) + j * 4 + k;
              size_t ox = j * 4 + k;
              size_t oy = i;
              if (ox >= mat_n) {
                out.get()[linear] = 0;
                continue;
              }
              out.get()[linear] = from[batch_offset+oy * mat_n + ox];
            }
          }
        }
      }
    }
    size_t origin[3] = {0, 0, 0}, region[3];
    region[0] = desc.UWidth();
    region[1] = desc.UHeight();
    region[2] = 1;
    // TODO: refactor out clEnqueueWriteImage, backend api exposed
    ORT_RETURN_IF_CL_ERROR(clEnqueueWriteImage(
        exec_->GetCommandQueue(), static_cast<cl_mem>(GetPackedWeight()), CL_TRUE, origin, region, 0, 0,  out.get(), 0, nullptr, nullptr));
    // Do sync copy, since we cannot extend the lifetime of src or tmp
    ORT_RETURN_IF_CL_ERROR(clFinish(exec_->GetCommandQueue()));
    return Status::OK();
  }


  MLAS_ACTIVATION act_info_;
  TensorShape W_shape_;
  IAllocatorUniquePtrToClMem packed_weight_;

  cl_mem GetPackedWeight() const {
    return packed_weight_.get();
  }
};

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    1, 10,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),  // conv kernel weight will be handled via PrePack
    MatMul);
ONNX_OPENCL_OPERATOR_KERNEL(
    MatMul,
    11,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),  // conv kernel weight will be handled via PrePack
    MatMul);
ONNX_OPENCL_OPERATOR_KERNEL(
    Gemm,
    11,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),  // conv kernel weight will be handled via PrePack
    MatMul);

ONNX_OPERATOR_KERNEL_EX(
    FusedMatMul,
    kMSDomain,
    1,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),  // conv kernel weight will be handled via PrePack
    MatMul                                        // register the Conv OpKernel as the FusedConv impl
);

}  // namespace opencl
}  // namespace onnxruntime
