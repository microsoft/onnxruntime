// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/attnlstm/multi_scale_deformable_attention.h"

#include "core/framework/op_kernel.h"

#include <array>
#include <bitset>
#include <cstdlib>
#include <memory>

namespace onnxruntime {
namespace contrib {

MultiScaleDeformableAttention::MultiScaleDeformableAttention(const OpKernelInfo& info) : OpKernel(info) {
#ifdef _WIN32
  // Windows
  std::array<int, 4> cpui;
  __cpuid(cpui.data(), 0);
  int nIds = cpui[0];

  if(nIds < 7)
  {
      return;
  }

  __cpuidex(cpui.data(), 7, 0);
  std::bitset<32> ebx;
  std::bitset<32> ecx;

  ebx = cpui[1];
  ecx = cpui[2];

  bool AVX512F = ebx[16];
  bool AVX512DQ = ebx[17];

  if(AVX512F && AVX512DQ){
    route = ImplementationRoute::AVX512;
  }
  else{
    route = ImplementationRoute::Generic;
  }

#elif defined(__linux__)
  // Linux
  __builtin_cpu_init();
  if(__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")){
    route = ImplementationRoute::AVX512;
  }
  else{
    route = ImplementationRoute::Generic;
  }
#else
  route = ImplementationRoute::Generic;
#endif
}

Status MultiScaleDeformableAttention::Compute(_Inout_ OpKernelContext* context) const {
  const auto* value = context->Input<Tensor>(0);  // Shape: [1, S, M, D]
  const auto* value_spatial_shapes = context->Input<Tensor>(1); // Shape: [L, 2]
  const auto* reference_points = context->Input<Tensor>(2); // Shape: [1, L, Q, 2]
  const auto* sampling_locations = context->Input<Tensor>(3); // Shape: [1, L, Q, M, P, 2]
  const auto* attention_weights = context->Input<Tensor>(4);  // Shape: [1, L, Q, M, P]

  const auto& value_input_shape = value->Shape();
  const auto& value_spatial_shapes_input_shape = value_spatial_shapes->Shape();
  const auto& attention_weights_input_shape = attention_weights->Shape();

  const int64_t M = value_input_shape[2];
  const int64_t D = value_input_shape[3];
  const int64_t L = value_spatial_shapes_input_shape[0];
  const int64_t P = attention_weights_input_shape[4];
  const int64_t Q = attention_weights_input_shape[2];

  auto* output = context->Output(0, { 1, Q, M*D }); // Shape: [1, Q, M*D]
  float * output_ptr = output->MutableData<float>();

  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  bool useGeneric;
  {
    char * env;
    #ifdef _WIN32
    _dupenv_s(&env, nullptr, "ORT_USE_GENERIC");
    #else
    env = std::getenv("ORT_USE_GENERIC");
    #endif
    if(env == nullptr || std::atoi(env) == 0){
      useGeneric = false;
    }
    else{
      useGeneric = true;
    }
    #ifdef _WIN32
    free(env);
    #endif
  }

  if(route == ImplementationRoute::AVX512 && D == 16 && M == 8 && P == 4 && !useGeneric) {
    // AVX512 implementation
    ComputeAVX512(
      value->Data<float>(),
      value_spatial_shapes->Data<int64_t>(),
      reference_points->Data<float>(),
      sampling_locations->Data<float>(),
      attention_weights->Data<float>(),
      output_ptr,
      M,
      L,
      P,
      D,
      Q,
      thread_pool,
      alloc);
  } else {
    // Generic implementation
    ComputeGeneric(
      value->Data<float>(),
      value_spatial_shapes->Data<int64_t>(),
      reference_points->Data<float>(),
      sampling_locations->Data<float>(),
      attention_weights->Data<float>(),
      output_ptr,
      M,
      L,
      P,
      D,
      Q,
      thread_pool,
      alloc);
  }

  return Status::OK();
}

void MultiScaleDeformableAttention::ComputeGeneric(
  const float* value,
  const int64_t* value_spatial_shapes,
  const float* reference_points,
  const float* sampling_locations,
  const float* attention_weights,
  float* output,
  int64_t M,
  int64_t L,
  int64_t P,
  int64_t D,
  int64_t Q,
  concurrency::ThreadPool* thread_pool,
  AllocatorPtr alloc) const {
  memset(output, 0, Q * M * D * sizeof(float));

  uint32_t threadCount = static_cast<uint32_t>(concurrency::ThreadPool::DegreeOfParallelism(thread_pool));
  if(Q <= 32){
      threadCount = 1;
  }

  const size_t perThreadWorkSize = D * sizeof(float);
  const size_t totalWorkSize = threadCount * perThreadWorkSize;

  char * buffer = static_cast<char *>(alloc->AllocArray(totalWorkSize, sizeof(char)));

  auto worker_lambda = [&](std::ptrdiff_t thread_id) -> void {
    auto task_info = concurrency::ThreadPool::PartitionWork(thread_id, threadCount, Q);
    float *thread_buffer = reinterpret_cast<float *>(buffer + thread_id * perThreadWorkSize);
    const float *value_begin = value;
    for(int64_t source_level = 0; source_level < L; ++source_level){
      int64_t feature_map_height = value_spatial_shapes[source_level * 2];
      int64_t feature_map_width = value_spatial_shapes[source_level * 2 + 1];
      for(std::ptrdiff_t iq = task_info.start; iq < task_info.end; ++iq){
        float reference_point_h = reference_points[(source_level * Q + iq) * 2 + 1];
        float reference_point_w = reference_points[(source_level * Q + iq) * 2];
        for(int im = 0; im < M; ++im){
          for(int ip = 0; ip < P; ++ip){
            float sampling_location_h = sampling_locations[((source_level * Q + iq) * M + im) * P * 2 + ip * 2 + 1];
            float sampling_location_w = sampling_locations[((source_level * Q + iq) * M + im) * P * 2 + ip * 2];
            sampling_location_h += reference_point_h;
            sampling_location_w += reference_point_w;
            sampling_location_h -= 0.5f;  // align_corner = False
            sampling_location_w -= 0.5f;  // align_corner = False
            float h_floor = std::floor(sampling_location_h);
            float h_weight_high = sampling_location_h - h_floor;
            float h_weight_low = 1.0f - h_weight_high;
            int h_low = static_cast<int>(h_floor);
            int h_high = h_low + 1;
            float w_floor = std::floor(sampling_location_w);
            float w_weight_high = sampling_location_w - w_floor;
            float w_weight_low = 1.0f - w_weight_high;
            int w_low = static_cast<int>(w_floor);
            int w_high = w_low + 1;

            memset(thread_buffer, 0, D * sizeof(float));
            auto inbound = [&](int h, int w) -> bool {
              return h >= 0 && h < feature_map_height && w >= 0 && w < feature_map_width;
            };
            auto access = [&](int h, int w, int d) -> float {
              return value_begin[((h * feature_map_width + w) * M + im) * D + d];
            };

            if(inbound(h_low, w_low)){
              for(int id = 0; id < D; ++id){
                thread_buffer[id] += h_weight_low * w_weight_low * access(h_low, w_low, id);
              }
            }
            if(inbound(h_low, w_high)){
              for(int id = 0; id < D; ++id){
                thread_buffer[id] += h_weight_low * w_weight_high * access(h_low, w_high, id);
              }
            }
            if(inbound(h_high, w_low)){
              for(int id = 0; id < D; ++id){
                thread_buffer[id] += h_weight_high * w_weight_low * access(h_high, w_low, id);
              }
            }
            if(inbound(h_high, w_high)){
              for(int id = 0; id < D; ++id){
                thread_buffer[id] += h_weight_high * w_weight_high * access(h_high, w_high, id);
              }
            }

            float weight = attention_weights[((source_level * Q + iq) * M + im) * P + ip];
            float *output_ptr = output + (iq * M + im) * D;
            for(int id = 0; id < D; ++id){
              output_ptr[id] += weight * thread_buffer[id];
            }
          }
        }
      }
      value_begin += feature_map_height * feature_map_width * M * D;
    }
  };
  concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, threadCount, worker_lambda);
}

ONNX_CPU_OPERATOR_MS_KERNEL(
    MultiScaleDeformableAttention,
    1,
    KernelDefBuilder()
      .TypeConstraint(
        "T1",
        {DataTypeImpl::GetTensorType<float>()})
      .TypeConstraint(
        "T2",
        {DataTypeImpl::GetTensorType<int64_t>()}),
    MultiScaleDeformableAttention)

}  // namespace contrib
}  // namespace onnxruntime
