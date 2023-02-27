// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ft_vit_int8_custom_op.h"
#include "stdio.h"
#include "stdlib.h"
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>

using namespace fastertransformer;

static fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);

template<typename T>
FTViTINT8CustomKernel<T>::FTViTINT8CustomKernel(const OrtKernelInfo* info,
                                                void* compute_stream,
                                                int batch_size,
                                                int img_size,
                                                int patch_size,
                                                int embed_dim,
                                                int head_num,
                                                int layer_num,
                                                int has_cls_token,
                                                int int8_mode):
batch_size_(batch_size), img_size_(img_size), patch_size_(patch_size), embed_dim_(embed_dim), has_cls_token_(has_cls_token) 
{
    checkCUDNN(cudnnCreate(&cudnn_handle_));
    checkCUDNN(cudnnSetStream(cudnn_handle_, stream_));
    check_cuda_error(cublasCreate(&cublas_handle_));
    check_cuda_error(cublasSetStream(cublas_handle_, stream_));
    check_cuda_error(cublasLtCreate(&cublaslt_handle_));

    cublas_algo_map_ = new cublasAlgoMap("igemm_config.in");

    int sm = getSMVersion();

    cublas_wrapper_mutex_ = new std::mutex();

    bool use_ORDER_COL32_2R_4R4 = false;
#if (CUDART_VERSION >= 11000)
    if (sm >= 80) {
        use_ORDER_COL32_2R_4R4 = true;
    }
#endif

    cublas_wrapper_ = new cublasINT8MMWrapper(
        cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_, cublas_wrapper_mutex_, use_ORDER_COL32_2R_4R4);

    if (std::is_same<T, half>::value) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper_->setFP32GemmConfig();
    }

    const int  in_chans       = 3;
    const int  inter_size     = embed_dim * 4;
    const int  head_dim       = embed_dim / head_num;
    const bool with_cls_token = has_cls_token > 0;
    const int  seq_len        = (img_size / patch_size) * (img_size / patch_size) + (with_cls_token ? 1 : 0);

    params_ = ViTINT8Weight<T>(embed_dim, inter_size, layer_num, img_size, patch_size, in_chans, with_cls_token);

    FT_LOG_INFO("batch_size: %d, img_size : %d,\n"
                "patch_size: %d, embed_dim: %d,\n"
                "head_num  : %d, head_dim : %d,\n"
                "layer_num : %d, seq_len  : %d,\n"
                "inter_size:%d\n",
                batch_size,
                img_size,
                patch_size,
                embed_dim,
                head_num,
                head_dim,
                layer_num,
                seq_len,
                inter_size);

    attention_type_ = getAttentionType<T>(head_dim, getSMVersion(), true, seq_len);
    printf("attention_type: %d\n", int(attention_type_));

    int max_batch = batch_size;
    vit_ = new ViTTransformerINT8<T>(max_batch,
                                         img_size,
                                         in_chans,
                                         patch_size,
                                         embed_dim,
                                         head_num,
                                         inter_size,
                                         layer_num,
                                         with_cls_token,
                                         getSMVersion(),
                                         1.0f,
                                         int8_mode,
                                         stream_,
                                         cudnn_handle_,
                                         cublas_wrapper_,
                                         &allocator,
                                         false,
                                         attention_type_);
}

template<typename T>
void FTViTINT8CustomKernel<T>::Compute(OrtKernelContext* context) {

    const int  in_chans       = 3;
    const bool with_cls_token = has_cls_token_ > 0;
    const int seq_len       = (img_size_ / patch_size_) * (img_size_ / patch_size_) + (with_cls_token ? 1 : 0);

    Ort::KernelContext kcontext(context);

    Ort::ConstValue ort_val = kcontext.GetInput(0);
    const void* p_input_data = ort_val.GetTensorData<void>();

    std::vector<int64_t> output_shape{(int64_t)batch_size_, (int64_t)seq_len, (int64_t)embed_dim_};
    Ort::UnownedValue ort_val_output = kcontext.GetOutput(0, output_shape);
    const void* p_output_data = ort_val_output.GetTensorData<void>();

    std::vector<fastertransformer::Tensor> input_tensors = std::vector<fastertransformer::Tensor>{
        fastertransformer::Tensor{MEMORY_GPU,
               getTensorType<T>(),
               std::vector<size_t>{(size_t)batch_size_, (size_t)in_chans, (size_t)img_size_, (size_t)img_size_},
               (const T*)p_input_data}};

    std::vector<fastertransformer::Tensor> output_tensors =
        std::vector<fastertransformer::Tensor>{fastertransformer::Tensor{MEMORY_GPU,
                                   getTensorType<T>(),
                                   std::vector<size_t>{(size_t)batch_size_, (size_t)seq_len, (size_t)embed_dim_},
                                   (T*)p_output_data}};

    //CudaTimer cuda_timer(stream_);
    //cuda_timer.start();

    vit_->forward(&output_tensors, &input_tensors, &params_);

    //float total_time = cuda_timer.stop();
    //printf("vit forward time:%.2f ms\n", total_time);
}

template<typename T>
FTViTINT8CustomKernel<T>::~FTViTINT8CustomKernel() {
    delete vit_;
    delete cublas_algo_map_;
    delete cublas_wrapper_mutex_;

    sync_check_cuda_error();

    // free data
    check_cuda_error(cublasDestroy(cublas_handle_));
    check_cuda_error(cublasLtDestroy(cublaslt_handle_));
    checkCUDNN(cudnnDestroy(cudnn_handle_));

    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
}

//
// FTViTINT8CustomOp 
//
FTViTINT8CustomOp::FTViTINT8CustomOp(const char* provider, void* compute_stream) {
    provider_ = provider; 
    compute_stream_ = compute_stream;
}

void* FTViTINT8CustomOp::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { 
    // default values
    int batch_size     = 1;
    int img_size       = 224;
    int patch_size     = 16;
    int embed_dim      = 768;
    int num_heads      = 12;
    int layer_num      = 12;
    int with_cls_token = 1;
    int is_fp16        = 0;
    int int8_mode      = 2;

    Ort::ConstKernelInfo kinfo(info);

    // extract batch size and image size from kernel info 
    Ort::TypeInfo type_info = kinfo.GetInputTypeInfo(0); // first input has the dims of (batch, channel num, img size, img size)
    Ort::ConstTypeInfo const_type_info = type_info.GetConst(); 
    Ort::ConstTensorTypeAndShapeInfo type_shape_info = const_type_info.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> input_shape = type_shape_info.GetShape();
    batch_size = static_cast<int>(input_shape[0]);
    img_size = static_cast<int>(input_shape[2]);

    // extract is fp16 from kernel info
    ONNXTensorElementDataType onnx_type = type_shape_info.GetElementType();
    is_fp16 = (onnx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)? 1 : 0;

    // extract others from kernel info
    patch_size = static_cast<int>(kinfo.GetAttribute<int64_t>("patch_size"));
    embed_dim = static_cast<int>(kinfo.GetAttribute<int64_t>("embed_dim"));
    num_heads = static_cast<int>(kinfo.GetAttribute<int64_t>("num_heads"));
    layer_num = static_cast<int>(kinfo.GetAttribute<int64_t>("layer_num"));
    with_cls_token = static_cast<int>(kinfo.GetAttribute<int64_t>("with_cls_token"));
    int8_mode = static_cast<int>(kinfo.GetAttribute<int64_t>("int8_mode"));

    if (is_fp16) {
        return new FTViTINT8CustomKernel<half>(info,
                                                compute_stream_,
                                                batch_size,
                                                img_size,
                                                patch_size,
                                                embed_dim,
                                                num_heads,
                                                layer_num,
                                                with_cls_token,
                                                int8_mode); 
    } else {
        return new FTViTINT8CustomKernel<float>(info,
                                                compute_stream_,
                                                batch_size,
                                                img_size,
                                                patch_size,
                                                embed_dim,
                                                num_heads,
                                                layer_num,
                                                with_cls_token,
                                                int8_mode); 

    }
}

