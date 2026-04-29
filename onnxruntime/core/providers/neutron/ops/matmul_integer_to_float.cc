// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#include "core/providers/neutron/ops/matmul_integer_to_float.h"
#include "core/providers/neutron/ops/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/neutron/neutron_fwd.h"
#include "core/util/math_cpuonly.h"
#include <algorithm>

#if NEUTRON_AARCH64
#include "neutron/NeutronDriver.h"
#endif

namespace onnxruntime {
namespace neutron {

extern std::shared_ptr<NeutronStackAllocator> neutronAlloc;

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulIntegerToFloat,
    kMSDomain,
    1,
    uint8_t,
    kNeutronExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(),
                               DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),
    MatMulIntegerToFloat);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulIntegerToFloat,
    kMSDomain,
    1,
    int8_t,
    kNeutronExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),
    MatMulIntegerToFloat);

Status MatMulIntegerToFloat::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                     /*out*/ bool& is_packed,
                                     /*out*/ PrePackedWeights* prepacked_weights) {
  try {
    switch (input_idx) {
      case IN_A:
        break;
      case IN_B: {
        m_b_rows = tensor.Shape()[1];
        m_b_cols = tensor.Shape()[0];

        if ((m_b_rows % 16) || (m_b_rows * 16 >= 1024 * 1024))
          throw std::invalid_argument("NeutronEP:MatMulIntegerToFloat invalid argument(s)");

        auto [channelDensity, numNeutrons, divisions] = TilingSolver(m_b_cols, -1, 4, 8, false, false);

        m_handle = neutronAlloc->getMemoryHandle();
        m_header = (uint32_t*)neutronAlloc->Alloc(16 * sizeof(uint32_t), m_handle);
        m_b_neutron = (int8_t*)neutronAlloc->Alloc(m_b_rows * m_b_cols, m_handle);

        const int8_t* b_data = static_cast<const int8_t*>(tensor.DataRaw());
        OrganizeWeightsData(b_data, m_b_neutron, m_b_rows,
                            m_b_cols, channelDensity, numNeutrons, 8, 16, true);
        clean_cache(m_b_neutron, m_b_rows * m_b_cols);

        // Neutron expects the bias as a parameter anyway.
        m_b_bias = (int32_t*)neutronAlloc->Alloc(m_b_rows * sizeof(int32_t), m_handle);
        for (uint32_t i = 0; i < m_b_rows; i++) {
          int32_t row_sum = 0;
          for (uint32_t j = 0; j < m_b_cols; j++) {
            row_sum += *(b_data + j * m_b_rows + i);
          }
          m_b_bias[i] = row_sum;
        }

        m_b_factors = (uint32_t*)neutronAlloc->Alloc(m_b_rows * sizeof(uint32_t), m_handle);
        uint32_t scaler = ScaleToNeutron(1.0);
        for (uint32_t i = 0; i < m_b_rows; i++) {
          m_b_factors[i] = scaler;
        }
        clean_cache(m_b_factors, m_b_rows * sizeof(uint32_t));
      } break;
      case IN_A_SCALE: {
        m_dynamic_scale = false;
        m_a_scale_data = *tensor.Data<float>();
      } break;
      case IN_B_SCALE: {
        // support scale per tensor and per channel
        if (!m_dynamic_scale) {
          uint32_t scale_size = tensor.Shape().NumDimensions() ? tensor.Shape()[0] : 1;
          m_out_scale.resize(scale_size);
          for (size_t i = 0; i < m_out_scale.size(); i++) {
            m_out_scale[i] = tensor.Data<float>()[i] * m_a_scale_data;
          }
        }
      } break;
      case IN_A_ZERO_POINT: {
        m_dynamic_bias = false;
        m_a_zp = *(static_cast<const uint8_t*>(tensor.DataRaw()));
        for (uint32_t i = 0; i < m_b_rows; i++) {
          m_b_bias[i] = (int32_t)(-m_b_bias[i] * m_a_zp);
        }
        clean_cache(m_b_bias, m_b_rows * sizeof(int32_t));
      } break;
      case IN_B_ZERO_POINT:
        // we assume B has ZP equal to 0
        // todo: implement a check
        break;
      case IN_BIAS: {
        m_output_bias = tensor.Data<float>();
      } break;
    }
  } catch (const std::exception& e) {
    // Do not delegate this instance if out of memory
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }
  return Status::OK();
}

Status MatMulIntegerToFloat::Compute(OpKernelContext* ctx) const {
  struct timespec t1, t2, t3, t4, t5;

  const Tensor* a = ctx->Input<Tensor>(IN_A);
  const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  if (!m_header || !m_b_neutron || !m_b_factors) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "NeutronEP:MatMulInteger falied to init.");
  } else {
    clock_gettime(CLOCK_REALTIME, &t1);

    neutronAlloc->pushMemoryState(m_handle);

    // non-transposed b
    uint32_t neutron_a_rows = a->Shape()[1];
    uint32_t neutron_a_cols = a->Shape()[2];
    uint32_t neutron_b_rows = b ? b->Shape()[1] : m_b_rows;
    uint32_t neutron_b_cols = b ? b->Shape()[0] : m_b_cols;
    if (neutron_a_cols != neutron_b_cols) {
      LOGS_DEFAULT(WARNING) << "Neutron dimenssions do not match!";
    }

    const float* out_scale_data = NULL;
    bool scale_per_tensor = false;
    std::vector<float> dyn_out_scale;

    if (!m_dynamic_scale) {
      out_scale_data = m_out_scale.data();
      scale_per_tensor = (m_out_scale.size() == 1);
    } else {
      float a_scale_data = *(static_cast<const float*>(ctx->Input<Tensor>(IN_A_SCALE)->DataRaw()));

      const Tensor* b_scale = ctx->Input<Tensor>(IN_B_SCALE);
      uint32_t scale_size = b_scale->Shape().NumDimensions() ? b_scale->Shape()[0] : 1;

      dyn_out_scale.resize(scale_size);
      for (size_t i = 0; i < dyn_out_scale.size(); i++) {
        dyn_out_scale[i] = a_scale_data * (static_cast<const float*>(b_scale->DataRaw()))[i];
      }
      out_scale_data = dyn_out_scale.data();
      scale_per_tensor = (dyn_out_scale.size() == 1);
    }

    if (m_dynamic_bias) {
      uint8_t a_zp = *(static_cast<const uint8_t*>(ctx->Input<Tensor>(IN_A_ZERO_POINT)->DataRaw()));
      for (uint32_t i = 0; i < m_b_rows; i++) {
        m_b_bias[i] = (int32_t)(-m_b_bias[i] * a_zp);
      }
      clean_cache(m_b_bias, m_b_rows * sizeof(int32_t));
    }

    clock_gettime(CLOCK_REALTIME, &t2);

    uint32_t a_size = neutron_a_rows * neutron_a_cols * sizeof(uint8_t);
    uint8_t* a_neutron = (uint8_t*)neutronAlloc->AllocReserved(a_size, m_handle);
    auto a_data = static_cast<const uint8_t*>(a->DataRaw());
    memcpy(a_neutron, a_data, a_size);

    uint32_t y_size = neutron_a_rows * neutron_b_rows * sizeof(int32_t);
    int32_t* y_neutron = (int32_t*)neutronAlloc->AllocReserved(y_size, m_handle);

    m_header[0] = GetMatmulTypeFlag(true, a->IsDataType<int8_t>());
    m_header[1] = 0;
    m_header[2] = neutron_a_rows;
    m_header[3] = neutron_a_cols;
    m_header[4] = neutron_b_rows;
    m_header[5] = (uint8_t*)a_neutron - (uint8_t*)m_header;
    m_header[6] = (uint8_t*)m_b_neutron - (uint8_t*)m_header;
    m_header[7] = (uint8_t*)m_b_bias - (uint8_t*)m_header;
    m_header[8] = (uint8_t*)m_b_factors - (uint8_t*)m_header;
    m_header[9] = (uint8_t*)y_neutron - (uint8_t*)m_header;
    m_header[10] = 0;   // m_y_zp;
    m_header[11] = 4;   // result num bytes
    m_header[12] = 8;   // Weight Bits
    m_header[13] = -1;  // Group Size equal to negative means no group size
    m_header[14] = 0;
    m_header[15] = 0;

    clock_gettime(CLOCK_REALTIME, &t3);

    NeutronError ret = ENONE;
    ret = matmul((const void*)m_header, 16 * sizeof(uint32_t),
                 (const void*)a_neutron, a_size,
                 (const void*)y_neutron, y_size, m_handle);
    if (ret != ENONE) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "matmul() error");
    }

    clock_gettime(CLOCK_REALTIME, &t4);

    Tensor* y = ctx->Output(OUT_Y, {1, neutron_a_rows, neutron_b_rows});
    float* y_data = static_cast<float*>(y->MutableDataRaw());

    int32_t* input = y_neutron;
    auto* output = y_data;
    for (uint32_t i = 0; i < static_cast<uint32_t>(neutron_a_rows); i++) {
      for (uint32_t j = 0; j < static_cast<uint32_t>(neutron_b_rows); j++) {
        uint32_t scale_idx = scale_per_tensor ? 0 : j;
        if (m_output_bias) {
          float bias_val = static_cast<float>(m_output_bias[j]);
          float input_val = static_cast<float>(static_cast<int32_t>(*input++));
          float scale_val = static_cast<float>(out_scale_data[scale_idx]);
          *output++ = bias_val + input_val * scale_val;
        } else {
          float input_val = static_cast<float>(static_cast<int32_t>(*input++));
          *output++ = input_val * static_cast<float>(out_scale_data[scale_idx]);
        }
      }
    }

    neutronAlloc->popMemoryState(m_handle);
    clock_gettime(CLOCK_REALTIME, &t5);
  }
  return Status::OK();
}

}  // namespace neutron
}  // namespace onnxruntime
