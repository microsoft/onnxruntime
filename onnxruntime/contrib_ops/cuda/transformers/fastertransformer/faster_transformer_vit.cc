// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "faster_transformer_vit.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <vit/ViT.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include <iostream> //slx

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace fastertransformer;
template<typename T>
int vitExample(int batch_size, int img_size, int patch_size, int embed_dim, int head_num, int layer_num, int token_classifier)
{
    cudnnHandle_t    cudnn_handle;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t     stream = 0;
    checkCUDNN(cudnnCreate(&cudnn_handle));
    checkCUDNN(cudnnSetStream(cudnn_handle, stream));
    check_cuda_error(cublasCreate(&cublas_handle));
    check_cuda_error(cublasSetStream(cublas_handle, stream));
    check_cuda_error(cublasLtCreate(&cublaslt_handle));

    fastertransformer::cublasAlgoMap* cublas_algo_map = new fastertransformer::cublasAlgoMap("gemm_config.in");

    std::mutex* cublas_wrapper_mutex = new std::mutex();

    fastertransformer::cublasMMWrapper* cublas_wrapper =
        new fastertransformer::cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, nullptr);

    if (std::is_same<T, half>::value) {
        cublas_wrapper->setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }
    const int  in_chans       = 3;
    const bool with_cls_token = token_classifier > 0;
    const int  inter_size     = embed_dim * 4;
    const int  head_dim       = embed_dim / head_num;
    const int  seq_len        = (img_size / patch_size) * (img_size / patch_size) + (with_cls_token ? 1 : 0);

    fastertransformer::ViTWeight<T> params =
        fastertransformer::ViTWeight<T>(embed_dim, inter_size, layer_num, img_size, patch_size, in_chans, with_cls_token);

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

    AttentionType attention_type = getAttentionType<T>(head_dim, getSMVersion(), true, seq_len);
    printf("Attention Type: %d\n", int(attention_type));
    fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);
    int                                               max_batch = batch_size;
    fastertransformer::ViTTransformer<T>*                                vit       = new fastertransformer::ViTTransformer<T>(max_batch,
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
                                                   stream,
                                                   cudnn_handle,
                                                   cublas_wrapper,
                                                   &allocator,
                                                   false,
                                                   attention_type);

    T *input_d, *output_d;
    deviceMalloc(&input_d, batch_size * img_size * img_size * in_chans, false);
    deviceMalloc(&output_d, batch_size * seq_len * embed_dim, false);

    std::vector<fastertransformer::Tensor> input_tensors = std::vector<fastertransformer::Tensor>{
        fastertransformer::Tensor{MEMORY_GPU,
               fastertransformer::getTensorType<T>(),
               std::vector<size_t>{(size_t)batch_size, (size_t)in_chans, (size_t)img_size, (size_t)img_size},
               input_d}};

    std::vector<fastertransformer::Tensor> output_tensors =
        std::vector<fastertransformer::Tensor>{fastertransformer::Tensor{MEMORY_GPU,
                                   getTensorType<T>(),
                                   std::vector<size_t>{(size_t)batch_size, (size_t)seq_len, (size_t)embed_dim},
                                   output_d}};

    // warmup
    for (int i = 0; i < 10; i++) {
        vit->forward(&output_tensors, &input_tensors, &params);
    }

    int       ite = 100;
    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    for (int i = 0; i < ite; i++) {
        vit->forward(&output_tensors, &input_tensors, &params);
    }
    float total_time = cuda_timer.stop();

    FT_LOG_INFO("batch_size: %d, img_size : %d,\n"
                "patch_size: %d, embed_dim: %d,\n"
                "head_num  : %d, head_dim : %d,\n"
                "layer_num : %d, is_fp16  : %d,\n"
                "FT-CPP-time %.2f ms (%d iterations) ",
                batch_size,
                img_size,
                patch_size,
                embed_dim,
                head_num,
                head_dim,
                layer_num,
                std::is_same<T, half>::value,
                total_time / ite,
                ite);

    delete vit;
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;

    // free data
    check_cuda_error(cudaFree(output_d));
    check_cuda_error(cudaFree(input_d));
    check_cuda_error(cublasDestroy(cublas_handle));
    check_cuda_error(cublasLtDestroy(cublaslt_handle));
    checkCUDNN(cudnnDestroy(cudnn_handle));

    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    return 0;
}
ONNX_OPERATOR_KERNEL_EX(
    FasterTransformerVit,
    kMSDomain,///kOnnxDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FasterTransformerVit);

Status FasterTransformerVit::ComputeInternal(OpKernelContext* /*context*/) const {

  ///fastertransformer::BertWeight<float> bert_weights(2, 2, 2);
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
/*
    if (argc != 9) {
        printf(
            "[ERROR] vit_example batch_size img_size patch_size embed_dim head_number layer_num with_cls_token is_fp16\n");
        printf("e.g. ./bin/vit_example 1 224 16 768 12 12 1 0 \n");
        return 0;
    }
*/
    const int batch_size       = 1; //atoi(argv[1]);
    const int img_size         = 224; //atoi(argv[2]);
    const int patch_size       = 16; //atoi(argv[3]);
    const int embed_dim        = 768; //atoi(argv[4]);
    const int head_num         = 12; //atoi(argv[5]);
    const int layer_num        = 12; //atoi(argv[6]);
    const int token_classifier = 1; //atoi(argv[7]);
    const int is_fp16          = 0; //atoi(argv[8]);

    int res;
    if (is_fp16) {
        res = vitExample<half>(batch_size, img_size, patch_size, embed_dim, head_num, layer_num, token_classifier);
    }
    else {
        res = vitExample<float>(batch_size, img_size, patch_size, embed_dim, head_num, layer_num, token_classifier);
    }
	std::cout << "vitExample return: " << res << std::endl;
  
  return Status::OK();

}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
