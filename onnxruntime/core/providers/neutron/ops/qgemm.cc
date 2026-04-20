// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#include "core/providers/neutron/ops/qgemm.h"
#include "core/providers/neutron/ops/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/neutron/neutron_fwd.h"

#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cpu/quantization/matmul_integer_base.h"
#include "core/util/math_cpuonly.h"

#if NEUTRON_AARCH64
#include "neutron/NeutronDriver.h"
#endif
#include "core/providers/neutron/neutron_allocator.h"

namespace onnxruntime {
namespace neutron {

extern std::shared_ptr<NeutronStackAllocator> neutronAlloc;

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QGemm,
    kMSDomain,
    1,
    uint8_t,
    kNeutronExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("TA", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("TB", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("TC", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("TYZ", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("TY", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<uint8_t>()}),
    QGemm);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QGemm,
    kMSDomain,
    1,
    int8_t,
    kNeutronExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("TA", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("TB", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("TC", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("TYZ", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("TY", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<int8_t>()}),
    QGemm);

Status QGemm::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                      /*out*/ bool& is_packed,
                      /*out*/ PrePackedWeights* prepacked_weights) {
  try {
    switch (input_idx) {
      case IN_A:
        break;
      case IN_A_SCALE:
        m_a_scale_data = *(tensor.Data<float>());
        break;
      case IN_A_ZERO_POINT:
        m_a_zp = *(static_cast<const uint8_t*>(tensor.DataRaw()));
        break;
      case IN_B: {
        size_t num_dims = tensor.Shape().NumDimensions();
        m_b_rows = tensor.Shape()[num_dims - 1];
        m_b_cols = ((num_dims == 1) ? 1 : tensor.Shape()[num_dims - 2]);

        if ((m_b_rows % 16) || (m_b_rows * 16 >= 1024 * 1024))
          throw std::invalid_argument("NeutronEP:QLinearMatMul invalid argument(s)");

        auto [channelDensity, numNeutrons, divisions] = TilingSolver(m_b_cols, -1, 1, 8, false, false);

        m_b_neutron = (int8_t*)neutronAlloc->Alloc(m_b_rows * m_b_cols, m_handle);
        const int8_t* b_data = static_cast<const int8_t*>(tensor.DataRaw());
        OrganizeWeightsData(b_data, m_b_neutron, m_b_rows,
                            m_b_cols, channelDensity, numNeutrons, 8, 16, true);
        clean_cache(m_b_neutron, m_b_rows * m_b_cols);

        m_b_bias = (int32_t*)neutronAlloc->Alloc(m_b_rows * sizeof(int32_t), m_handle);
        for (uint32_t i = 0; i < m_b_rows; i++) {
          int32_t row_sum = 0;
          for (uint32_t j = 0; j < m_b_cols; j++) {
            row_sum += *(b_data + j * m_b_rows + i);
          }
          m_b_bias[i] = row_sum;
        }
      } break;
      case IN_B_SCALE: {
        auto data = tensor.Data<float>();
        if (IsScalarOr1ElementVector(&tensor)) {
          m_b_scales.assign(m_b_rows, *data);
        } else {
          m_b_scales.assign(data, data + m_b_rows);
        }
      } break;
      case IN_B_ZERO_POINT:
        // we assume B has ZP equal to 0
        // todo: implement a check
        break;
      case IN_C: {
        m_c_data = tensor.Data<int32_t>();
      } break;
      case IN_Y_SCALE: {
        m_y_scale_data = *(tensor.Data<float>());

        const int64_t output_scale_size = m_b_rows;
        for (int64_t i = 0; i < output_scale_size; i++) {
          m_output_scales.push_back(m_a_scale_data * m_b_scales[i] / m_y_scale_data);
        }

        m_b_factors = (uint32_t*)neutronAlloc->Alloc(m_b_rows * sizeof(uint32_t), m_handle);

        for (uint32_t i = 0; i < m_b_rows; i++) {
          m_b_factors[i] = ScaleToNeutron(m_output_scales[i]);
        }
        clean_cache(m_b_factors, m_b_rows * sizeof(uint32_t));
      } break;
      case IN_Y_ZERO_POINT: {
        m_y_zp = *(static_cast<const uint8_t*>(tensor.DataRaw()));
        for (uint32_t i = 0; i < m_b_rows; i++) {
          m_b_bias[i] = (int32_t)(m_y_zp / m_output_scales[i] + m_c_data[i] - m_b_bias[i] * m_a_zp);
        }
        clean_cache(m_b_bias, m_b_rows * sizeof(int32_t));
      } break;
    }
  } catch (const std::exception& e) {
    // Do not delegate this instance if out of memory
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }
  return Status::OK();
}

Status QGemm::Compute(OpKernelContext* ctx) const {
  const auto* a = ctx->Input<Tensor>(IN_A);
  const auto* b = ctx->Input<Tensor>(IN_B);

  // validate offsets
  const auto* a_offset = ctx->Input<Tensor>(IN_A_ZERO_POINT);
  const auto* b_offset = ctx->Input<Tensor>(IN_B_ZERO_POINT);
  const auto* y_offset = ctx->Input<Tensor>(IN_Y_ZERO_POINT);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_offset),
              "QLinearMatmul : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsBQuantParamSupported(b_offset->Shape(), b->Shape()),
              "QLinearMatmul : weight zero point must be a scalar, 1D tensor of size 1, or last to second dimension is 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_offset),
              "QLinearMatmul : result zero point must be a scalar or 1D tensor of size 1");

  // validate scale
  const auto* a_scale = ctx->Input<Tensor>(IN_A_SCALE);
  const auto* b_scale = ctx->Input<Tensor>(IN_B_SCALE);
  const auto* y_scale = ctx->Input<Tensor>(IN_Y_SCALE);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale),
              "QLinearMatmul : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsBQuantParamSupported(b_scale->Shape(), b->Shape()),
              "QLinearMatmul : weight scale must be a scalar, 1D tensor of size 1, or last to second dimension is 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_scale),
              "QLinearMatmul : result scale must be a scalar or 1D tensor of size 1");

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape(), &b_scale->Shape(), &b_offset->Shape()));

  // non-transposed b
  uint32_t neutron_a_rows = static_cast<uint32_t>(helper.M());
  uint32_t neutron_a_cols = static_cast<uint32_t>(helper.K());
  uint32_t neutron_b_rows = static_cast<uint32_t>(helper.N());
  auto num_matmuls = helper.OutputOffsets().size();

  for (size_t batch = 0; batch < num_matmuls; batch++) {
    neutronAlloc->pushMemoryState(m_handle);

    Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());
    // Bail out early if the output is going to be empty
    if (y->Shape().Size() == 0)
      return Status::OK();

    uint32_t a_size = neutron_a_rows * neutron_a_cols;
    int8_t* a_neutron = (int8_t*)neutronAlloc->AllocReserved(a_size * sizeof(uint8_t), m_handle);
    auto a_data = static_cast<const int8_t*>(a->DataRaw());
    memcpy(a_neutron, a_data + helper.LeftOffsets()[batch], a_size);
    clean_cache(a_neutron, a_size);

    uint32_t y_size = neutron_a_rows * neutron_b_rows;
    int8_t* y_neutron = (int8_t*)neutronAlloc->AllocReserved(y_size * sizeof(uint8_t), m_handle);

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
    m_header[10] = m_y_zp;
    m_header[11] = 1;   // result num bytes
    m_header[12] = 8;   // Weight Bits
    m_header[13] = -1;  // Group Size equal to negative means no group size
    m_header[14] = 0;
    m_header[15] = 0;

    NeutronError ret = ENONE;
    ret = matmul((const void*)m_header, 16 * sizeof(uint32_t), (const void*)a_neutron,
                 a_size, (const void*)y_neutron, y_size, m_handle);
    if (ret != ENONE) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "matmul() error");
    }

    memcpy(static_cast<int8_t*>(y->MutableDataRaw()) + helper.OutputOffsets()[batch], y_neutron, y_size);

    neutronAlloc->popMemoryState(m_handle);
  }  // batch
  return Status::OK();
}
}  // namespace neutron
}  // namespace onnxruntime
