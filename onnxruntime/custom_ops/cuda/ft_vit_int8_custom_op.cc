// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ft_vit_int8_custom_op.h"
#include "stdio.h"
#include "stdlib.h"
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>

using namespace fastertransformer;

#define L_ROOT "transformer.encoder.layer.%d"
#define ATT_Q "attn.query"
#define ATT_K "attn.key"
#define ATT_V "attn.value"
#define ATT_OUT "attn.out"
#define ATT_NORM "attention_norm"
#define FFN_NORM "ffn_norm"
#define FFN_IN "ffn.fc1"
#define FFN_OUT "ffn.fc2"

const std::vector<const char*> layer_weight_names = {L_ROOT "." ATT_NORM ".weight",
                                                     L_ROOT "." ATT_NORM ".bias",
                                                     L_ROOT "." ATT_Q ".weight",
                                                     L_ROOT "." ATT_Q ".bias",
                                                     L_ROOT "." ATT_K ".weight",
                                                     L_ROOT "." ATT_K ".bias",
                                                     L_ROOT "." ATT_V ".weight",
                                                     L_ROOT "." ATT_V ".bias",
                                                     L_ROOT "." ATT_OUT ".weight",
                                                     L_ROOT "." ATT_OUT ".bias",
                                                     L_ROOT "." FFN_NORM ".weight",
                                                     L_ROOT "." FFN_NORM ".bias",
                                                     L_ROOT "." FFN_IN ".weight",
                                                     L_ROOT "." FFN_IN ".bias",
                                                     L_ROOT "." FFN_OUT ".weight",
                                                     L_ROOT "." FFN_OUT ".bias",
                                                     L_ROOT ".amaxList",
                                                     L_ROOT ".h_amaxList"};

const std::vector<std::string> pre_layer_weight_names  = {"transformer.embeddings.patch_embeddings.weight",
                                                          "transformer.embeddings.patch_embeddings.bias",
                                                          "transformer.embeddings.cls_token",
                                                          "transformer.embeddings.position_embeddings"};
const std::vector<std::string> post_layer_weight_names = {"transformer.encoder.encoder_norm.weight",
                                                          "transformer.encoder.encoder_norm.bias"};

template<typename T>
void loadWeightsPtrINT8(const OrtKernelInfo* info, std::vector<const T*>& w, int layer_num, bool with_cls_token = true)
{
    Ort::ConstKernelInfo kinfo(info);
    std::unordered_map<std::string, Ort::ConstValue> weights_map = {};

    // Get weights (constant inputs) from kernel info
    for (size_t i = 0; i < kinfo.GetInputCount(); i++) {
        std::string name = kinfo.GetInputName(i);
        int is_constant = 0;
        Ort::ConstValue value = kinfo.GetTensorConstantInput(i, &is_constant);
        if (is_constant) {
            weights_map[name] = value; 
        }
    }

    // Load weights to weight address vector 
    long unsigned int idx = 0;
    for (auto& name : pre_layer_weight_names) {
        if (!with_cls_token && name == "transformer.embeddings.cls_token") {
            continue;
        }

        auto iter = weights_map.find(name);
        if (iter != weights_map.end()) {
            Ort::ConstValue ort_val = iter->second; 
            w[idx++] = ort_val.GetTensorData<T>();
        }
    }

    for (int i = 0; i < layer_num; i++) {
        for (auto& name : layer_weight_names) {
            char str_buf[1024];
            sprintf(str_buf, name, i);
            std::string string_buf = str_buf;

            auto iter = weights_map.find(string_buf);
            if (iter != weights_map.end()) {
                Ort::ConstValue ort_val = iter->second; 
                w[idx++] = ort_val.GetTensorData<T>();
            }
        }
    }

    for (auto& name : post_layer_weight_names) {
        auto iter = weights_map.find(name);
        if (iter != weights_map.end()) {
            Ort::ConstValue ort_val = iter->second; 
            w[idx++] = ort_val.GetTensorData<T>();
        }
    }

    FT_CHECK(idx == w.size());
}

FTViTINT8CustomKernel::FTViTINT8CustomKernel(const OrtKernelInfo* info,
                                             void* compute_stream,
                                             int batch_size,
                                             int img_size,
                                             int patch_size,
                                             int embed_dim,
                                             int head_num,
                                             int layer_num,
                                             int has_cls_token,
                                             int is_fp16,
                                             int int8_mode):
batch_size_(batch_size), img_size_(img_size), embed_dim_(embed_dim), is_fp16_(is_fp16)
{
    checkCUDNN(cudnnCreate(&cudnn_handle_));
    checkCUDNN(cudnnSetStream(cudnn_handle_, stream_));
    check_cuda_error(cublasCreate(&cublas_handle_));
    check_cuda_error(cublasSetStream(cublas_handle_, stream_));
    check_cuda_error(cublasLtCreate(&cublaslt_handle_));

    cublas_algo_map_ = new cublasAlgoMap("igemm_config.in");

    int sm = getSMVersion();

    cublas_wrapper_mutex_ = new std::mutex();
    allocator_ = new fastertransformer::Allocator<AllocatorType::CUDA>(getDevice());

    bool use_ORDER_COL32_2R_4R4 = false;
#if (CUDART_VERSION >= 11000)
    if (sm >= 80) {
        use_ORDER_COL32_2R_4R4 = true;
    }
#endif

    cublas_wrapper_ = new cublasINT8MMWrapper(
        cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_, cublas_wrapper_mutex_, use_ORDER_COL32_2R_4R4);

    if (is_fp16_) {
        cublas_wrapper_->setFP16GemmConfig();
    }
    else {
        cublas_wrapper_->setFP32GemmConfig();
    }

    int  max_batch      = batch_size;
    int  in_chans       = 3;
    int  inter_size     = embed_dim * 4;
    int  head_dim       = embed_dim / head_num;
    bool with_cls_token = has_cls_token > 0;
    int  seq_len        = (img_size / patch_size) * (img_size / patch_size) + (with_cls_token ? 1 : 0);
    size_t weights_num  = pre_layer_weight_names.size() + post_layer_weight_names.size() + layer_num * layer_weight_names.size();
    seq_len_ = seq_len;
    in_chans_ = in_chans;
    weights_num_ = weights_num;

    if (is_fp16_) {
        params_fp16_ = ViTINT8Weight<half>(embed_dim, inter_size, layer_num, img_size, patch_size, in_chans, with_cls_token);
        std::vector<const half*> w_fp16;
        w_fp16.resize(weights_num);
        loadWeightsPtrINT8<half>(info, w_fp16, layer_num, with_cls_token); 
        const half* const* pp_buf = &w_fp16[0];
        params_fp16_.CopyWeightsFromHostBuffers(pp_buf);
        attention_type_ = getAttentionType<half>(head_dim, getSMVersion(), true, seq_len);
        vit_fp16_ = new ViTTransformerINT8<half>(max_batch,
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
                                            allocator_,
                                            false,
                                            attention_type_);

    }
    else {
        params_fp32_ = ViTINT8Weight<float>(embed_dim, inter_size, layer_num, img_size, patch_size, in_chans, with_cls_token);
        std::vector<const float*> w_fp32;
        w_fp32.resize(weights_num);
        loadWeightsPtrINT8<float>(info, w_fp32, layer_num, with_cls_token); 
        const float* const* pp_buf = &w_fp32[0];
        params_fp32_.CopyWeightsFromHostBuffers(pp_buf);
        attention_type_ = getAttentionType<float>(head_dim, getSMVersion(), true, seq_len);
        vit_fp32_ = new ViTTransformerINT8<float>(max_batch,
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
                                             allocator_,
                                             false,
                                             attention_type_);
    }

    FT_LOG_INFO("batch_size: %d, img_size : %d,\n"
                "patch_size: %d, embed_dim: %d,\n"
                "head_num  : %d, head_dim : %d,\n"
                "layer_num : %d, seq_len  : %d,\n"
                "inter_size :%d, attention_type : %d\n",
                batch_size,
                img_size,
                patch_size,
                embed_dim,
                head_num,
                head_dim,
                layer_num,
                seq_len,
                inter_size,
                int(attention_type_));
}

void FTViTINT8CustomKernel::Compute(OrtKernelContext* context) {
    Ort::KernelContext kcontext(context);
    Ort::ConstValue ort_val = kcontext.GetInput(0);

    std::vector<int64_t> output_shape{(int64_t)batch_size_, (int64_t)seq_len_, (int64_t)embed_dim_};
    Ort::UnownedValue ort_val_output = kcontext.GetOutput(0, output_shape);

    if (is_fp16_) {
        const half* p_input_data = ort_val.GetTensorData<half>();
        half* p_output_data = ort_val_output.GetTensorMutableData<half>();

        std::vector<fastertransformer::Tensor> input_tensors = std::vector<fastertransformer::Tensor>{
            fastertransformer::Tensor{MEMORY_GPU,
                                      getTensorType<half>(),
                                      std::vector<size_t>{(size_t)batch_size_, (size_t)in_chans_, (size_t)img_size_, (size_t)img_size_},
                                      p_input_data}};

        std::vector<fastertransformer::Tensor> output_tensors = std::vector<fastertransformer::Tensor>{
            fastertransformer::Tensor{MEMORY_GPU,
                                      getTensorType<half>(),
                                      std::vector<size_t>{(size_t)batch_size_, (size_t)seq_len_, (size_t)embed_dim_},
                                      p_output_data}};

        vit_fp16_->forward(&output_tensors, &input_tensors, &params_fp16_);
    }
    else {
        const float* p_input_data = ort_val.GetTensorData<float>();
        float* p_output_data = ort_val_output.GetTensorMutableData<float>();

        std::vector<fastertransformer::Tensor> input_tensors = std::vector<fastertransformer::Tensor>{
            fastertransformer::Tensor{MEMORY_GPU,
                                      getTensorType<float>(),
                                      std::vector<size_t>{(size_t)batch_size_, (size_t)in_chans_, (size_t)img_size_, (size_t)img_size_},
                                      p_input_data}};

        std::vector<fastertransformer::Tensor> output_tensors = std::vector<fastertransformer::Tensor>{
            fastertransformer::Tensor{MEMORY_GPU,
                                      getTensorType<float>(),
                                      std::vector<size_t>{(size_t)batch_size_, (size_t)seq_len_, (size_t)embed_dim_},
                                      p_output_data}};
        vit_fp32_->forward(&output_tensors, &input_tensors, &params_fp32_);
    }
}

FTViTINT8CustomKernel::~FTViTINT8CustomKernel() {
    if (is_fp16_) {
        delete vit_fp16_;
    }
    else {
        delete vit_fp32_;
    }
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

OrtMemType FTViTINT8CustomOp::GetInputMemoryType(size_t index) const {
    if (index == 0) {
        return OrtMemTypeDefault;
    } else {
        // Second input and the rest of them are weights and we want them to be placed in CPU memory first,
        // so that we can leverage FT weight's CopyWeightsFromHostBuffers() to prepare the weights before ViT forward.
        // If we don't explicitly make weights stay in CPU memory, ORT will place them in GPU memory by default and we
        // might need to modify FT source code in order to get the weights on GPU memory. We want to avoid modifying FT code.
        return OrtMemTypeCPUInput;
    }
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

    return new FTViTINT8CustomKernel(info,
                                     compute_stream_,
                                     batch_size,
                                     img_size,
                                     patch_size,
                                     embed_dim,
                                     num_heads,
                                     layer_num,
                                     with_cls_token,
                                     is_fp16,
                                     int8_mode);
}
