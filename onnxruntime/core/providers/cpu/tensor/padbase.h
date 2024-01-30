// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"

namespace onnxruntime {

enum class Mode : int {
  Constant = 0,
  Reflect,
  Edge,
  Wrap
};

class PadBase {
 public:
  // Pads and slices are usually about twice the shapes involved
  using PadsVector = InlinedVector<int64_t, kTensorShapeSmallBufferElementsSize * 2>;

  // The following several functions are shared among the providers

  /// <summary>
  /// Handle the case when the input shape has zero dim values.
  /// Depending on the mode, the input dim with zero value must match the output dim value.
  ///
  /// </summary>
  /// <param name="mode">Padding mode enum value</param>
  /// <param name="input_shape">actual input shape</param>
  /// <param name="output_shape">output_shape</param>
  /// <returns>Error if current mode padding can not be achieved with zero dim values</returns>
  static Status HandleDimValueZero(const Mode& mode, const TensorShape& input_shape, const TensorShape& output_shape);

  /// <summary>
  /// Compute Pads by applying axes if specified otherwise copy the supplied pads.
  ///
  /// The function queries optional axes input (since version 18) and if present,
  /// applies it as a mask to the pads. If axes is not present, the pads are copied as is.
  /// If axes are present, they  are used as a mask over pads, so only those axes are being padded.
  /// </summary>
  /// <param name="ctx">kernel context to query axes input</param>
  /// <param name="data_rank">input rank</param>
  /// <param name="pads_data">pads data from pads input</param>
  /// <param name="pads">resulting pads</param>
  static void ComputePads(OpKernelContext& ctx, size_t data_rank, gsl::span<const int64_t> pads_data,
                          PadsVector& pads);

  /// <summary>
  /// Separates negative pad values to slices and zeros them out in original pads.
  /// Leaving the rest of slices values as zero.
  ///
  /// This function is used inline in the Pad CUDA implementation and is not exposed via a provider
  /// interfaces.
  /// </summary>
  /// <param name="pads">pad values</param>
  /// <param name="slices">slices output</param>
  static void SeparateNegativeToSlices(gsl::span<int64_t> pads, PadsVector& slices) {
    slices.assign(pads.size(), 0);
    for (size_t index = 0, lim = pads.size(); index < lim; index++) {
      if (pads[index] < 0) {
        slices[index] = pads[index];
        pads[index] = 0;
      }
    }
  }

  // End provider shared

  /// <summary>
  /// Flatten no padding inner most Axis, so one memcpy cover multiple Axis.
  /// For example, for a shape of [1,224,224,3] with padding [0,3,3,0,0,3,3,0], can be flatten as
  /// [1,224,224*3] with padding [0,3,3*3,0,3,3*3].
  ///
  /// This is a helper function pads are expected to be twice the rank
  /// </summary>
  /// <param name="input_dims">original input dims</param>
  /// <param name="pads">pad values</param>
  /// <param name="slices">slices</param>
  /// <param name="reshaped_dims">result dims</param>
  static void FlattenInnerShape(gsl::span<const int64_t> input_dims, gsl::span<const int64_t> pads,
                                gsl::span<const int64_t> slices, TensorShapeVector& reshaped_dims);

  /// <summary>
  /// Used after the inner shape is flattened, so we can apply this function to pads and slices
  /// to reshape them as well.
  /// </summary>
  /// <param name="src_pad">pads</param>
  /// <param name="src_dim_count">original dim count</param>
  /// <param name="new_dim_count">expected flattended dim count</param>
  /// <param name="inner_no_pad_size">is the left most dimension that was flattened.
  ///  In the example above, that would be 224, reverse computed from 224*3</param>
  /// <param name="reshaped_pad">resulting reshaped pads or slices</param>
  static void ReshapePads(gsl::span<const int64_t> src_pad, size_t src_dim_count, size_t new_dim_count,
                          size_t inner_no_pad_size, PadsVector& reshaped_pad);

 protected:
  PadBase(const OpKernelInfo& info) : value_(info.GetAttrOrDefault("value", 0.f)) {
    std::string mode;
    if (info.GetAttr("mode", &mode).IsOK()) {
      if (mode == "constant")
        mode_ = Mode::Constant;
      else if (mode == "reflect")
        mode_ = Mode::Reflect;
      else if (mode == "edge")
        mode_ = Mode::Edge;
      else if (mode == "wrap")
        mode_ = Mode::Wrap;
      else
        ORT_THROW("Invalid 'mode' attribute value");
    }

    const auto& kernel_def = info.GetKernelDef();

    int start_ver, end_ver;
    kernel_def.SinceVersion(&start_ver, &end_ver);

    // kMSDomain contrib kernel AND OnnxDomain start version >= 11 => DynamicPad
    if (start_ver >= 11 || kernel_def.Domain() == kMSDomain) {
      is_dynamic_ = true;
    }

    if (!is_dynamic_) {
      gsl::span<const int64_t> pads_span;
      if (!info.GetAttrsAsSpan("pads", pads_span).IsOK())
        ORT_THROW("Invalid 'pads' attribute value");
      pads_.assign(pads_span.begin(), pads_span.end());
      // Separate out any negative pads_ into the slices_ array
      slices_.resize(pads_.size(), 0);
      for (size_t index = 0; index < pads_.size(); index++) {
        if (pads_[index] < 0) {
          slices_[index] = pads_[index];
          pads_[index] = 0;
        }
      }
    }
  }

  ~PadBase() = default;

  Mode mode_{Mode::Constant};
  PadsVector pads_;    // After construction, only >=0 values are in here
  PadsVector slices_;  // All of the negative padding values are separated out into slices_
  const float value_;  // will always be float (when 'value' parsed from attribute - opset 10 and below)

  // flag used to differentiate the cases where some input values to the op are
  // to be obtained from (is_dynamic_ = false) attributes vs (is_dynamic_ = true) inputs
  bool is_dynamic_ = false;
};

}  // namespace onnxruntime
