// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "ft_vit_int8_custom_op.h"
#include "core/framework/provider_options.h"
#include "stdio.h"
#include "stdlib.h"
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>

using namespace onnxruntime;
using namespace fastertransformer;

FTViTINT8CustomKernel::FTViTINT8CustomKernel(const OrtKernelInfo* info, void* compute_stream) {

    const int batch_size    = 1;
    const int img_size      = 224;
    const int patch_size    = 16;
    const int embed_dim     = 768;
    const int head_num      = 12;
    const int layer_num     = 12;
    const int has_cls_token = 1;
    const int is_fp16       = 0;
    const int int8_mode     = 2;


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

    //if (std::is_same<T, half>::value) {
        //cublas_wrapper->setFP16GemmConfig();
    //}
    //else if (std::is_same<T, float>::value) {
        //cublas_wrapper->setFP32GemmConfig();
    //}
    cublas_wrapper_->setFP32GemmConfig();

    const int  in_chans       = 3;
    const int  inter_size     = embed_dim * 4;
    const int  head_dim       = embed_dim / head_num;
    const bool with_cls_token = has_cls_token > 0;
    const int  seq_len        = (img_size / patch_size) * (img_size / patch_size) + (with_cls_token ? 1 : 0);

    //ViTINT8Weight<T> params =
        //ViTINT8Weight<T>(embed_dim, inter_size, layer_num, img_size, patch_size, in_chans, with_cls_token);
    params_ = ViTINT8Weight<float>(embed_dim, inter_size, layer_num, img_size, patch_size, in_chans, with_cls_token);

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

    //AttentionType attention_type = getAttentionType<T>(head_dim, getSMVersion(), true, seq_len);
    attention_type_ = getAttentionType<float>(head_dim, getSMVersion(), true, seq_len);
    printf("attention_type: %d\n", int(attention_type_));

    fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);
    int max_batch = batch_size;
    printf("before ViTTransformerINT8\n");
    vit_ = new ViTTransformerINT8<float>(max_batch,
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
    printf("after ViTTransformerINT8\n");
}

void FTViTINT8CustomKernel::Compute(OrtKernelContext* context) {

    const int batch_size    = 1;
    const int img_size      = 224;
    const int patch_size    = 16;
    const int embed_dim     = 768;
    const int has_cls_token = 1;
    const int  in_chans       = 3;
    const bool with_cls_token = has_cls_token > 0;
    const int seq_len       = (img_size / patch_size) * (img_size / patch_size) + (with_cls_token ? 1 : 0);

    FT_LOG_INFO("batch_size: %d, img_size : %d,\n"
                "patch_size: %d, embed_dim: %d,\n"
                "seq_len  : %d,\n",
                batch_size,
                img_size,
                patch_size,
                embed_dim,
                seq_len);

    //float *input_d, *output_d;
    //deviceMalloc(&input_d, batch_size * img_size * img_size * in_chans, false);
    //deviceMalloc(&output_d, batch_size * seq_len * embed_dim, false);

    Ort::KernelContext kcontext(context);

    const size_t num_inputs = kcontext.GetInputCount();
    printf("number of input %ld\n", num_inputs);
    Ort::ConstValue ort_val = kcontext.GetInput(0);
    const void* p_input_data = ort_val.GetTensorData<void>();
    //const size_t num_outputs = kcontext.GetOutputCount();

    std::vector<int64_t> output_shape{(int64_t)batch_size, (int64_t)seq_len, (int64_t)embed_dim};
    Ort::UnownedValue ort_val_output = kcontext.GetOutput(0, output_shape);
    const void* p_output_data = ort_val_output.GetTensorData<void>();

    std::vector<fastertransformer::Tensor> input_tensors = std::vector<fastertransformer::Tensor>{
        fastertransformer::Tensor{MEMORY_GPU,
               getTensorType<float>(),
               std::vector<size_t>{(size_t)batch_size, (size_t)in_chans, (size_t)img_size, (size_t)img_size},
               p_input_data}};
               //input_d}};

    std::vector<fastertransformer::Tensor> output_tensors =
        std::vector<fastertransformer::Tensor>{fastertransformer::Tensor{MEMORY_GPU,
                                   getTensorType<float>(),
                                   std::vector<size_t>{(size_t)batch_size, (size_t)seq_len, (size_t)embed_dim},
                                   p_output_data}};
                                   //output_d}};

    printf("before forward\n");
    vit_->forward(&output_tensors, &input_tensors, &params_);
    printf("after forward\n");
}

void FTViTINT8CustomKernel::Release() {
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
