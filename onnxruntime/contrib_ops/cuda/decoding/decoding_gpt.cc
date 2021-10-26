// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "decoding_gpt.h"

#include "contrib_ops/cuda/fastertransformer/open_decoder.h"
#include "contrib_ops/cuda/fastertransformer/gpt.h"
#include "contrib_ops/cuda/fastertransformer/utils/common.h"
#include "contrib_ops/cuda/fastertransformer/utils/arguments.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      DecodingGpt,                                                \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DecodingGpt2<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

//#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
//REGISTER_KERNEL_TYPED(BFloat16)
//#endif

using namespace ONNX_NAMESPACE;
using namespace fastertransformer;

template <typename T>
DecodingGpt2<T>::DecodingGpt2(const OpKernelInfo& op_kernel_info) : DecodingBase(op_kernel_info) {
  //const TransformerOptions* options = TransformerOptions::GetInstance();
  int64_t value = 0;
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("batch_size", &value).IsOK());
  batch_size_ = static_cast<int>(value);
  ORT_ENFORCE(batch_size_ >= 1);


  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("candidate_num", &value).IsOK());
  candidate_num_ = static_cast<int>(value);
  ORT_ENFORCE(candidate_num_ >= 1);

  op_kernel_info.GetAttrOrDefault<float>("probability_threshold", &probability_threshold_, 0.0f);
  
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("max_seq_len", &value).IsOK());
  max_seq_len_ = static_cast<int>(value);

  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("head_num", &value).IsOK());
  head_num_ = static_cast<int>(value);

  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("size_per_head", &value).IsOK());
  size_per_head_ = static_cast<int>(value);

  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("num_layer", &value).IsOK());
  num_layer_ = static_cast<int>(value);

  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("start_id", &value).IsOK());
  start_id_ = static_cast<int>(value);

  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("end_id", &value).IsOK());
  end_id_ = static_cast<int>(value);

  op_kernel_info.GetAttrOrDefault<float>("temperature", &temperature_, 1.0f);

  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("is_fuse_qkv", &value).IsOK());
  is_fuse_qkv_ = value > 0;

#ifndef NDEBUG
        srand(0); // Fixing the random seed
#else
        srand(static_cast<unsigned int>(time(NULL)));
#endif
}

template <typename T>
Status DecodingGpt2<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  int vocab_size = static_cast<int>(context->Input<Tensor>(18)->Shape().GetDims()[0]);

  // TODO: shall we get handles from CUDA EP instance like Einsum?
  auto cublas_handle = const_cast<DecodingGpt2<T>*>(this)->get_cublas_handler();
  auto cublaslt_handle = const_cast<DecodingGpt2<T>*>(this)->get_cublaslt_handler();

  DecodingInitParam<DataType_> decoding_params;
  decoding_params.cublas_handle = cublas_handle;
  decoding_params.cublaslt_handle = cublaslt_handle;
  
  std::vector<int64_t> output_shape(2);
  //output_shape[0] = batch_size_;
  //output_shape[1] = max_seq_len_;
  output_shape[0] = max_seq_len_;
  output_shape[1] = batch_size_;
  Tensor* output_ids = context->Output(0, output_shape);

  decoding_params.output_ids = reinterpret_cast<int*>(output_ids->template MutableData<int32_t>()),
  
  check_cuda_error(cudaMemset(decoding_params.output_ids, 0, sizeof(int) * max_seq_len_ * batch_size_));

  typedef DecoderTransformerTraits<traits_::OpType> DecodingTraits_;
  DecodingGpt<DecodingTraits_::OpType> *decoding_handler;
  const cudaStream_t &stream = Stream();
  decoding_params.stream = stream;


// Get temp space allocator - we will use this to allocate memory for intermediate tensors
  AllocatorPtr allocator;
  auto status = context->GetTempSpaceAllocator(&allocator);
  if (!status.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION,
                           "There was a problem acquiring temporary memory allocator in DecodingGpt op");
  }

  fastertransformer::Allocator<AllocatorType::ORT> allocator_(allocator, stream);
  ORT_TRY
  {
      decoding_handler = new DecodingGpt<DecodingTraits_::OpType>(
          allocator_, batch_size_, 
          max_seq_len_, head_num_, size_per_head_,
          vocab_size, num_layer_,
          start_id_, end_id_,
          candidate_num_, probability_threshold_, temperature_,
          1, 1, is_fuse_qkv_);
  }
  ORT_CATCH (std::runtime_error &e)
  {
      ORT_THROW(e.what());
  }

  DecoderInitParam<DataType_> *params = new DecoderInitParam<DataType_>[num_layer_];
  const int hidden_unit = size_per_head_ * head_num_;
  for (int i = 0; i < num_layer_; i++)
  {
      //params[i].request_max_mem_seq_len = -1;
      params[i].request_batch_size = batch_size_;
      params[i].stream = stream;
      params[i].cublas_handle = cublas_handle;
      params[i].cublaslt_handle = cublaslt_handle;
      check_cuda_error(cublasSetStream(params[i].cublas_handle, params[i].stream));

      this->get_tensor(context, 0, &params[i].self_layernorm.beta, i * hidden_unit);
      this->get_tensor(context, 1, &params[i].self_layernorm.gamma, i * hidden_unit);

      if (is_fuse_qkv_)
      {
          this->get_tensor(context, 2, &params[i].self_attention.query_weight.kernel, i * hidden_unit * hidden_unit * 3);
          this->get_tensor(context, 3, &params[i].self_attention.query_weight.bias, i * hidden_unit * 3);
      }
      else
      {
          this->get_tensor(context, 2, &params[i].self_attention.query_weight.kernel, i * hidden_unit * hidden_unit);
          this->get_tensor(context, 3, &params[i].self_attention.query_weight.bias, i * hidden_unit);
          this->get_tensor(context, 4, &params[i].self_attention.key_weight.kernel, i * hidden_unit * hidden_unit);
          this->get_tensor(context, 5, &params[i].self_attention.key_weight.bias, i * hidden_unit);
          this->get_tensor(context, 6, &params[i].self_attention.value_weight.kernel, i * hidden_unit * hidden_unit);
          this->get_tensor(context, 7, &params[i].self_attention.value_weight.bias, i * hidden_unit);
      }

      this->get_tensor(context, 8, &params[i].self_attention.attention_output_weight.kernel, i * hidden_unit * hidden_unit);
      this->get_tensor(context, 9, &params[i].self_attention.attention_output_weight.bias, i * hidden_unit);

      this->get_tensor(context, 10, &params[i].ffn_layernorm.beta, i * hidden_unit);
      this->get_tensor(context, 11, &params[i].ffn_layernorm.gamma, i * hidden_unit);
      this->get_tensor(context, 12, &params[i].ffn.intermediate_weight.kernel, i * hidden_unit * hidden_unit * 4);
      this->get_tensor(context, 13, &params[i].ffn.intermediate_weight.bias, i * hidden_unit * 4);
      this->get_tensor(context, 14, &params[i].ffn.output_weight.kernel, i * hidden_unit * hidden_unit * 4);
      this->get_tensor(context, 15, &params[i].ffn.output_weight.bias, i * hidden_unit);
  }

  this->get_tensor(context, 16, &decoding_params.layernorm.beta);
  this->get_tensor(context, 17, &decoding_params.layernorm.gamma);
  this->get_tensor(context, 18, &decoding_params.embedding_table);
  this->get_tensor(context, 19, &decoding_params.embedding_kernel);
  this->get_tensor(context, 20, &decoding_params.position_encoding_table);

  const DataType_* d_attn_mask = nullptr;
  this->get_tensor(context, 21, &d_attn_mask);

  const int* d_start_ids = reinterpret_cast<const int *>(context->Input<Tensor>(22)->template Data<int32_t>());
  ORT_ENFORCE(d_start_ids != nullptr);
  decoding_params.d_start_ids = d_start_ids;  

  const int* d_min_start_length = reinterpret_cast<const int *>(context->Input<Tensor>(23)->template Data<int32_t>());
  ORT_ENFORCE(d_min_start_length != nullptr);

  const int* d_max_start_length = reinterpret_cast<const int *>(context->Input<Tensor>(24)->template Data<int32_t>());
  ORT_ENFORCE(d_max_start_length != nullptr);

  int min_start_length = -1;
  int max_start_length = -1;
  cudaMemcpyAsync(&min_start_length, d_min_start_length, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&max_start_length, d_max_start_length, sizeof(int), cudaMemcpyDeviceToHost, stream);
  ORT_ENFORCE(min_start_length != -1);
  ORT_ENFORCE(max_start_length != -1);

  const int* d_start_lengths = reinterpret_cast<const int *>(context->Input<Tensor>(25)->template Data<int32_t>());
  ORT_ENFORCE(d_start_lengths != nullptr);

  TensorParallelParam tensor_parallel_param;
  tensor_parallel_param.rank = 0;
  tensor_parallel_param.world_size = 1;
  tensor_parallel_param.local_head_num_ = head_num_;
  tensor_parallel_param.local_hidden_units_ = hidden_unit;

  LayerParallelParam layer_parallel_param;
  layer_parallel_param.rank = 0;
  layer_parallel_param.world_size = 1;
  layer_parallel_param.layers_per_group = num_layer_;
  layer_parallel_param.local_batch_size = batch_size_;
  
  decoding_params.d_attn_mask = d_attn_mask;
  decoding_params.d_start_lengths = d_start_lengths;

  decoding_handler->set_tensor_parallel_param(tensor_parallel_param);
  decoding_handler->set_layer_parallel_param(layer_parallel_param);

  decoding_params.request_batch_size = batch_size_;
  decoding_params.max_input_len = max_start_length;
  for(int i = 0; i < decoding_handler->get_num_layer(); i++)
  {
      params[i].request_batch_size = batch_size_;
  }
  decoding_params.request_input_len = min_start_length;

  decoding_params.request_output_len = 32;
  //decoding_params.parent_ids = nullptr
  //decoding_params.sequence_length = nullptr

  ORT_TRY
  {
      decoding_handler->forward_context(params, decoding_params);
      decoding_handler->forward(params, decoding_params);
  }
  ORT_CATCH (std::runtime_error &e)
  {
      ORT_THROW(e.what());
  }
  ORT_CATCH (...)
  {
      ORT_RETHROW
  }

  delete decoding_handler;
  delete[] params;

  CUDA_CALL(cudaGetLastError());
  return Status::OK();
}

}  //namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
