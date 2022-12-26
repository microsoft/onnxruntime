// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "faster_transformer_bert.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <bert/Bert.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include <iostream> //slx

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    FasterTransformerBert,
    kMSDomain,///kOnnxDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FasterTransformerBert);

FasterTransformerBert::FasterTransformerBert(const OpKernelInfo& info) : CudaKernel(info) {//????? or like /transformer/beam_search.cc,
///FasterTransformerVit::FasterTransformerVit(const OpKernelInfo& op_kernel_info) : onnxruntime::contrib::transformers::FasterTransformerVit(op_kernel_info) {
  ///ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
  ///ORT_ENFORCE(epsilon_ >= 0);
}


using namespace fastertransformer;
template<typename T>
int bertExample(size_t batch_size,
                size_t num_layers,
                size_t seq_len,
                size_t head_num,
                size_t size_per_head,
                bool   is_remove_padding)
{
    ///printf("[INFO] Device: %s \n", getDeviceName().c_str());
    ///print_mem_usage("Before loading model");
    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size   = 4 * hidden_units;

    cudaStream_t     stream;
    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparselt_handle;
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle));
#endif
    cublasSetStream(cublas_handle, stream);
    fastertransformer::cublasAlgoMap* cublas_algo_map = new fastertransformer::cublasAlgoMap("gemm_config.in", "");

    Allocator<AllocatorType::CUDA> allocator(getDevice());

    std::mutex* cublas_wrapper_mutex = new std::mutex();
#ifdef SPARSITY_ENABLED
    fastertransformer::cublasMMWrapper cublas_wrapper = fastertransformer::cublasMMWrapper(
        cublas_handle, cublaslt_handle, cusparselt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
#else
    fastertransformer::cublasMMWrapper cublas_wrapper =
        fastertransformer::cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
#endif
    if (std::is_same<T, half>::value) {
        cublas_wrapper.setFP16GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper.setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    fastertransformer::BertWeight<T> bert_weights(hidden_units, inter_size, num_layers);

    AttentionType attention_type = getAttentionType<T>(size_per_head, getSMVersion(), is_remove_padding, seq_len);

    fastertransformer::Bert<T> bert = fastertransformer::Bert<T>(0,  // max_batch_size_, deprecated
                           0,  // max_seq_len_, deprecated
                           head_num,
                           size_per_head,
                           inter_size,
                           num_layers,
                           getSMVersion(),
                           1.0f,
                           stream,
                           &cublas_wrapper,
                           &allocator,
                           false,
                           attention_type,
                           false,
                           ActivationType::Gelu,
                           LayerNormType::post_layernorm);

    T* out_tensor;
    T* from_tensor;
    deviceMalloc(&out_tensor, batch_size * seq_len * head_num * size_per_head, false);
    deviceMalloc(&from_tensor, batch_size * seq_len * head_num * size_per_head, false);

    int*         h_sequence_lengths = new int[batch_size];
    unsigned int seed               = 0;
    for (uint i = 0; i < batch_size; i++) {
        h_sequence_lengths[i] = rand_r(&seed) % seq_len;
    }
    int* d_sequence_lengths;
    deviceMalloc(&d_sequence_lengths, batch_size, false);
    cudaH2Dcpy(d_sequence_lengths, h_sequence_lengths, batch_size);
    delete[] h_sequence_lengths;

    std::vector<fastertransformer::Tensor> input_tensors =
        std::vector<fastertransformer::Tensor>{fastertransformer::Tensor{MEMORY_GPU,
                                   fastertransformer::getTensorType<T>(),
                                   std::vector<size_t>{batch_size, seq_len, (size_t)(head_num * size_per_head)},
                                   from_tensor},
                            fastertransformer::Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size}, d_sequence_lengths}};

    std::vector<fastertransformer::Tensor> output_tensors =
        std::vector<fastertransformer::Tensor>{fastertransformer::Tensor{MEMORY_GPU,
                                   fastertransformer::getTensorType<T>(),
                                   std::vector<size_t>{batch_size, seq_len, (size_t)(head_num * size_per_head)},
                                   out_tensor}};
    print_mem_usage("After loading model");

    // warmup
    for (int i = 0; i < 10; i++) {
        bert.forward(&output_tensors, &input_tensors, &bert_weights);
    }
    print_mem_usage("After inference");

    // profile time
    const int ite = 10;
    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    for (int i = 0; i < ite; i++) {
        bert.forward(&output_tensors, &input_tensors, &bert_weights);
    }
    float total_time = cuda_timer.stop();

    FT_LOG_INFO("batch_size %ld seq_len %ld layer %ld "
                "FT-CPP-time %.2f ms (%d iterations) ",
                batch_size,
                seq_len,
                num_layers,
                total_time / ite,
                ite);

#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparselt_handle);
#endif
    deviceFree(d_sequence_lengths);
    deviceFree(from_tensor);
    deviceFree(out_tensor);
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;
    return 0;
}


Status FasterTransformerBert::ComputeInternal(OpKernelContext* /*context*/) const {

  ///fastertransformer::BertWeight<float> bert_weights(2, 2, 2);
    int                  batch_size        = 1;//atoi(argv[1]);
    int                  num_layers        = 12;//atoi(argv[2]);
    int                  seq_len           = 32;//atoi(argv[3]);
    int                  head_num          = 12;//atoi(argv[4]);
    int                  size_per_head     = 64;//atoi(argv[5]);
    bool                 is_remove_padding = false; //static_cast<bool>(atoi(argv[7]));
    const CublasDataType data_type         = static_cast<CublasDataType>(0);//static_cast<CublasDataType>(atoi(argv[6]));  // 0 FP32, 1 FP16, 2 BF 16

    int res;
    if (data_type == FLOAT_DATATYPE) {
        res = bertExample<float>(
            batch_size, num_layers, seq_len, head_num, size_per_head, is_remove_padding);
    }
    else if (data_type == HALF_DATATYPE) {
        res = bertExample<half>(
            batch_size, num_layers, seq_len, head_num, size_per_head, is_remove_padding);
    }
#ifdef ENABLE_BF16
    else if (data_type == BFLOAT16_DATATYPE) {
        res = bertExample<__nv_bfloat16>(
            batch_size, num_layers, seq_len, head_num, size_per_head, is_remove_padding);
    }
#endif
    else {
        throw std::runtime_error(std::string("[FT][ERROR] is_fp16 should be 0 (use float)"
                                             "or 1 (use half). \n "));
    }
	std::cout << "bertExample return: " << res << std::endl;
  
  return Status::OK();

}

Status FasterTransformerBert::Compute(OpKernelContext* context) const {//??????
  auto s = ComputeInternal(context);

  if (s.IsOK()) {
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA error ", cudaGetErrorName(err), ":", cudaGetErrorString(err));
    }
  }

  return s;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
