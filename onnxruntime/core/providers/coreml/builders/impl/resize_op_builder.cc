// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>

#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/utils.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class ResizeOpBuilder : public BaseOpBuilder {
 public:
  // allow roi and scales potentially being empty inputs that are ignored during processing
  ResizeOpBuilder() : BaseOpBuilder(/*allow empty inputs*/ true) {}

 private:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  // Resize opset 10- is very different than Resize opset 11+, with many key attributes missing
  // We only support Resize opset 11+ here
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 11; }

  bool SupportsMLProgram() const override { return true; }
};

namespace {
std::vector<int64_t> GetAxes(const NodeAttrHelper& helper, size_t input_rank) {
  auto axes = helper.Get("axes", std::vector<int64_t>{});
  if (axes.empty()) {
    axes.resize(input_rank);
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    for (auto& value : axes) {
      if (value < 0) {
        value = HandleNegativeAxis(value, input_rank);
      }
    }
  }

  return axes;
}

bool GetValidatedResizeScales(const GraphViewer& graph_viewer,
                              const Node& node,
                              const std::vector<int64_t>& input_shape,
                              const std::vector<int64_t>& axes,
                              std::vector<float>& scales,
                              const logging::Logger& logger) {
  const auto& input_defs = node.InputDefs();
  int64_t input_rank = input_shape.size();

  if (input_shape[input_rank - 2] == -1 || input_shape[input_rank - 1] == -1) {
    LOGS(logger, VERBOSE) << "Resize with 'scales' requires the H and W dimensions to have fixed values";
    return false;
  }

  const auto* scales_tensor = graph_viewer.GetConstantInitializer(input_defs[2]->Name());
  if (!scales_tensor) {
    LOGS(logger, VERBOSE) << "Resize 'scales' input must be a constant initializer";
    return false;
  }

  const auto& graph = graph_viewer.GetGraph();
  Initializer unpacked_tensor(graph, *scales_tensor, graph.ModelPath());
  auto scales_data = unpacked_tensor.DataAsSpan<float>();
  scales.assign(scales_data.begin(), scales_data.end());

  for (size_t idx = 0, end = axes.size(); idx < end; ++idx) {
    auto axis = axes[idx];
    auto scale = scales[idx];
    if (axis < (input_rank - 2) && scale != 1.0f) {
      LOGS(logger, VERBOSE) << "Resize only supports resizing the last two axes. Scale of axis " << axis << " is "
                            << scale;
      return false;
    }
  }

  return true;
}

bool GetValidatedResizeSizes(const GraphViewer& graph_viewer,
                             const Node& node,
                             const std::vector<int64_t>& input_shape,
                             const std::vector<int64_t>& axes,
                             std::vector<int64_t>& sizes, const logging::Logger& logger) {
  const auto& input_defs = node.InputDefs();
  int64_t input_rank = input_shape.size();

  const auto* sizes_tensor = graph_viewer.GetConstantInitializer(input_defs[3]->Name());
  if (!sizes_tensor) {
    LOGS(logger, VERBOSE) << "Resize 'sizes' input must be a constant initializer";
    return false;
  }

  Initializer unpacked_tensor(graph_viewer.GetGraph(), *sizes_tensor, graph_viewer.ModelPath());
  auto sizes_data = unpacked_tensor.DataAsSpan<int64_t>();
  sizes.assign(sizes_data.begin(), sizes_data.end());

  for (size_t idx = 0, end = axes.size(); idx < end; ++idx) {
    auto axis = axes[idx];
    auto cur_size = input_shape[idx];
    auto new_size = sizes[idx];
    if (axis < (input_rank - 2) && cur_size != new_size) {
      LOGS(logger, VERBOSE) << "Resize only supports resizing the last two axes. Input rank: " << input_rank
                            << " Change to size of axis " << axis << " from " << cur_size << " to " << new_size;
      return false;
    }
  }

  return true;
}
}  // namespace

void ResizeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& input_defs = node.InputDefs();

  // In Resize-11 both roi and scales were required even if you were using sizes.
  // https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Resize-11
  // From Resize-13 on they're all optional.
  //
  // We don't support roi so would never take a node with meaningful roi input. The roi input can however be provided
  // and is ignored unless coordinate_transformation_mode is set to 'tf_crop_and_resize'.
  // e.g. our unit tests tend to always provide an empty tensor as roi input instead of as a missing optional input.
  // Due to this we always call AddInputToSkip on the roi input.
  //
  // We require the sizes or scales input to be a constant initializers to take the node (i.e. they won't be an input
  // to the CoreML model for the partition, so calling AddInputToSkip isn't relevant).
  // Individual values from scales and sizes are added directly to the layer, so we won't use the initializer.
  //
  // That leaves an edge case for Resize-11 where scales could have been provided as an empty input tensor but
  // we're using a constant initializer for sizes. In this case AddInputToSkip needs to be called for the scales input.

  model_builder.AddInitializerToSkip(input_defs[1]->Name());  // roi
  model_builder.AddInputToSkip(input_defs[1]->Name());

  if (input_defs[2]->Exists()) {
    model_builder.AddInitializerToSkip(input_defs[2]->Name());  // scales
  }

  if (input_defs.size() > 3 && input_defs[3]->Exists()) {
    model_builder.AddInitializerToSkip(input_defs[3]->Name());  // sizes

    if (node.SinceVersion() < 13) {
      model_builder.AddInputToSkip(input_defs[2]->Name());  // skip the unused scales input
    }
  }
}

Status ResizeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                              const logging::Logger& logger) const {
  const auto input_defs = node.InputDefs();
  const auto output_defs = node.OutputDefs();
  const auto& graph_viewer = model_builder.GetGraphViewer();

  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Error getting input shape");
  size_t input_rank = input_shape.size();

  // we know we have either a scales or sizes input so this is safe.
  // check for sizes first. this handles Resize-11 where scales was a required input but sizes were used if provided.
  bool using_sizes = input_defs.size() >= 4 && input_defs[3]->Exists();
  bool using_scales = !using_sizes;

  NodeAttrHelper helper(node);
  const auto& mode = helper.Get("mode", "nearest");
  bool is_nearest = mode == "nearest";
  bool is_linear = !is_nearest;

  auto axes = GetAxes(helper, input_rank);
  std::vector<float> output_scales;
  std::vector<int64_t> output_sizes;
  size_t num_scales = 0;
  size_t num_sizes = 0;

  if (using_scales) {
    ORT_RETURN_IF_NOT(GetValidatedResizeScales(graph_viewer, node, input_shape, axes, output_scales, logger),
                      "Error getting validated scales");
    num_scales = output_scales.size();

    // special case linear downsample.
    // the CoreML implementation seems to be flaky and gives different outputs on different OS versions.
    // use bilinear_resize instead. we check in IsOpSupportedImpl that the downsample input is evenly
    // divisible by the output size so there's no rounding involved.
    if (is_linear && (output_scales[num_scales - 1] < 1.f || output_scales[num_scales - 2] < 1.f)) {
      using_scales = false;
      using_sizes = true;
      num_sizes = num_scales;
      output_sizes = input_shape;
      // only the last two dims have their size changed
      output_sizes[input_rank - 2] = static_cast<int64_t>(input_shape[input_rank - 2] * output_scales[num_scales - 2]);
      output_sizes[input_rank - 1] = static_cast<int64_t>(input_shape[input_rank - 1] * output_scales[num_scales - 1]);
    }
  } else {
    ORT_RETURN_IF_NOT(GetValidatedResizeSizes(graph_viewer, node, input_shape, axes, output_sizes, logger),
                      "Error getting validated sizes");
    num_sizes = output_sizes.size();
  }

  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;  // NOLINT

    std::string_view coreml_op_type;
    if (using_scales) {
      // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.image_resizing.upsample_bilinear
      // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.image_resizing.upsample_nearest_neighbor
      coreml_op_type = is_linear ? "upsample_bilinear" : "upsample_nearest_neighbor";
    } else {
      // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.image_resizing.resize_bilinear
      // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.image_resizing.resize_nearest_neighbor
      coreml_op_type = is_linear ? "resize_bilinear" : "resize_nearest_neighbor";
    }

    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, coreml_op_type);
    AddOperationInput(*op, "x", input_defs[0]->Name());

    std::string coord_trans_mode = helper.Get("coordinate_transformation_mode", "half_pixel");

    if (using_scales) {
      float scale_height = output_scales[num_scales - 2];
      float scale_width = output_scales[num_scales - 1];
      AddOperationInput(*op, "scale_factor_height",
                        model_builder.AddScalarConstant(coreml_op_type, "scale_factor_height", scale_height));
      AddOperationInput(*op, "scale_factor_width",
                        model_builder.AddScalarConstant(coreml_op_type, "scale_factor_width", scale_width));

      if (is_linear) {
        // we only allow these coord modes in the 'is supported' check,
        //   - half_pixel or pytorch_half_pixel with output size > 1 -> align_corners = false
        //   - align_corners -> align_corners = true
        bool align_corners = coord_trans_mode == "align_corners";
        AddOperationInput(*op, "align_corners",
                          model_builder.AddScalarConstant(coreml_op_type, "align_corners", align_corners));
      }
    } else {
      assert(using_sizes);
      int64_t target_height = output_sizes[num_sizes - 2];
      int64_t target_width = output_sizes[num_sizes - 1];

      AddOperationInput(*op, "target_size_height",
                        model_builder.AddScalarConstant(coreml_op_type, "target_size_height", target_height));
      AddOperationInput(*op, "target_size_width",
                        model_builder.AddScalarConstant(coreml_op_type, "target_size_width", target_width));

      if (is_linear) {
        // we only allow these coord modes in the 'is supported' check,
        //   - half_pixel or pytorch_half_pixel with output size > 1 -> UNALIGN_CORNERS
        //   - align_corners -> STRICT_ALIGN_CORNERS
        //   - asymmetric -> DEFAULT
        std::string sampling_mode_value;
        if (coord_trans_mode == "asymmetric") {
          sampling_mode_value = "DEFAULT";
        } else if (coord_trans_mode == "align_corners") {
          sampling_mode_value = "STRICT_ALIGN_CORNERS";
        } else {
          sampling_mode_value = "UNALIGN_CORNERS";
        }

        AddOperationInput(*op, "sampling_mode",
                          model_builder.AddScalarConstant(coreml_op_type, "sampling_mode", sampling_mode_value));
      }
    }

    AddOperationOutput(*op, *output_defs[0]);
    model_builder.AddOperation(std::move(op));
  } else {
    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);

    auto* coreml_upsample = layer->mutable_upsample();

    // we already checked the mode must be NN or Bilinear in IsOpSupportedImpl
    if (is_linear) {
      coreml_upsample->set_mode(COREML_SPEC::UpsampleLayerParams_InterpolationMode_BILINEAR);
    } else {
      coreml_upsample->set_mode(COREML_SPEC::UpsampleLayerParams_InterpolationMode_NN);
    }

    if (using_scales) {
      coreml_upsample->add_scalingfactor(static_cast<int64_t>(output_scales[num_scales - 2]));
      coreml_upsample->add_scalingfactor(static_cast<int64_t>(output_scales[num_scales - 1]));
    } else {
      auto scale_height = output_sizes[num_sizes - 2] / input_shape[input_rank - 2];
      auto scale_width = output_sizes[num_sizes - 1] / input_shape[input_rank - 1];
      coreml_upsample->add_scalingfactor(static_cast<int64_t>(scale_height));
      coreml_upsample->add_scalingfactor(static_cast<int64_t>(scale_width));
    }

    *layer->mutable_input()->Add() = input_defs[0]->Name();
    *layer->mutable_output()->Add() = output_defs[0]->Name();

    model_builder.AddLayer(std::move(layer));
  }

  return Status::OK();
}

bool ResizeOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                        const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, VERBOSE) << "Resize: input shape was not known";
    return false;
  }

  // as we allow empty shapes in the checks done by BaseOpBuilder::HasSupportedInputs we explicitly check for an empty
  // an empty input here to be consistent.
  // this should never happen in a real model though as a dim with value 0 (i.e. no input data) would typically be a
  // dynamic dimension where a previous step had no output (e.g. Loop of zero interations, NonZero with no matches,
  // NonMaxSupression with no boxes).
  if (DoesShapeSpecifyZeroElements(input_shape)) {
    LOGS(logger, VERBOSE) << "Resize input shape has with dimension values of 0 which is not supported.";
    return false;
  }

  const auto input_rank = input_shape.size();
  if (input_params.create_mlprogram) {
    if (input_rank < 3 || input_rank > 5) {
      LOGS(logger, VERBOSE) << "Resize only supports 3D to 5D input. Got: " << input_rank << "D";
      return false;
    }
  } else {
    if (input_rank != 4) {
      LOGS(logger, VERBOSE) << "Resize only support 4d shape. Got: " << input_rank << "D";
      return false;
    }
  }

  // check attributes
  NodeAttrHelper helper(node);

  if (helper.Get("antialias", 0) != 0) {
    LOGS(logger, VERBOSE) << "Resize does not support antialias";
    return false;
  }

  const auto& mode = helper.Get("mode", "nearest");
  bool is_linear = mode == "linear";
  bool is_nearest = mode == "nearest";
  if (!is_linear && !is_nearest) {
    LOGS(logger, VERBOSE) << "Resize unsupported input mode: " << mode;
    return false;
  }

  if (is_nearest) {
    const auto nearest_mode = helper.Get("nearest_mode", "round_prefer_floor");
    if (nearest_mode != "floor") {
      LOGS(logger, VERBOSE) << "Resize only supports 'floor' nearest_mode. Got: " << nearest_mode;
      return false;
    }
  }

  if (helper.Get("exclude_outside", 0) != 0) {
    LOGS(logger, VERBOSE) << "Resize does not support 'exclude_outside'";
    return false;
  }

  const auto keep_aspect_ratio_policy = helper.Get("keep_aspect_ratio_policy", "stretch");
  if (keep_aspect_ratio_policy != "stretch") {
    LOGS(logger, VERBOSE) << "Resize only supports keep_aspect_ratio_policy of 'stretch'. Got "
                          << keep_aspect_ratio_policy;
    return false;
  }

  // check for sizes first. this handles Resize-11 where scales was a required input but sizes were used if provided.
  bool using_sizes = input_defs.size() >= 4 && input_defs[3]->Exists();
  bool using_scales = !using_sizes && input_defs.size() >= 3 && input_defs[2]->Exists();

  if (!using_scales && !using_sizes) {
    LOGS(logger, VERBOSE) << "Resize requires 'scales' or 'sizes' input";
    return false;
  }

  // 'axes' is from opset 18 on and allows scales or sizes to have entries for the subset of axes.
  // we fill with default values if necessary so that the processing is consistent across all supported opsets.
  auto axes = GetAxes(helper, input_rank);
  std::vector<float> output_scales;
  std::vector<int64_t> output_sizes;

  // make sure scales/sizes are constant initializers, and are only modifying the last two dimensions of the input.
  if (using_scales) {
    if (!GetValidatedResizeScales(input_params.graph_viewer, node, input_shape, axes, output_scales, logger)) {
      return false;
    }

    size_t num_scales = output_scales.size();
    float scale_h = output_scales[num_scales - 2];
    float scale_w = output_scales[num_scales - 1];

    // NeuralNetwork supports upsample only with round numbers.
    //
    // ML Program results seem to match if round numbers are involved. When downsampling the scaling value should be
    // 1 / <factor of input size>. e.g. if input size is 8, scaling factor could be 1/8, 1/4 or 1/2.
    if (scale_h >= 1.f && scale_w >= 1.f) {
      // upsample (or no-op with both == 1.f that we won't bother special-casing)
      if (roundf(scale_h) != scale_h) {
        LOGS(logger, VERBOSE) << "Resize: scale_h: " << scale_h << " is not a whole number";
        return false;
      }

      if (roundf(scale_w) != scale_w) {
        LOGS(logger, VERBOSE) << "Resize: scale_w: " << scale_w << " is not a whole number";
        return false;
      }
    } else if (scale_h <= 1.f && scale_w <= 1.f) {
      // downsample
      if (input_params.create_mlprogram) {
        auto h_in = input_shape[input_rank - 2];
        auto w_in = input_shape[input_rank - 1];

        if (!utils::ReciprocalIsAFactorOfN(h_in, scale_h)) {
          LOGS(logger, VERBOSE) << "Resize: downsampling scale " << scale_h
                                << " is not a factor of input height: " << h_in;
          return false;
        }

        if (!utils::ReciprocalIsAFactorOfN(w_in, scale_w)) {
          LOGS(logger, VERBOSE) << "Resize: downsampling scale " << scale_w
                                << " is not a factor of input width: " << w_in;
          return false;
        }

      } else {
        LOGS(logger, VERBOSE) << "Resize: downsampling is not supported.";
        return false;
      }
    } else {
      LOGS(logger, VERBOSE) << "Resize: scale_h: " << scale_h << " and scale_w: " << scale_w
                            << " must both be >= 1 or <= 1";
      return false;
    }
  } else {
    assert(using_sizes);

    if (!GetValidatedResizeSizes(input_params.graph_viewer, node, input_shape, axes, output_sizes, logger)) {
      return false;
    }

    if (input_params.create_mlprogram) {
      // no additional requirements
    } else {
      if (!IsStaticShape(input_shape)) {
        // need to convert from sizes to scales when creating the NN layer, so the input H and W are required
        LOGS(logger, VERBOSE) << "Resize input shape with dynamic dimensions is not supported.";
        return false;
      }

      // For now we only support upsample, so the output_size_h and output_size_w should be an integer >= 1
      // TODO support ResizeBilinear
      auto input_size_h = input_shape[input_rank - 2];
      auto input_size_w = input_shape[input_rank - 1];

      auto num_sizes = output_sizes.size();  // could be smaller than input_rank if axes was used
      auto output_size_h = output_sizes[num_sizes - 2];
      auto output_size_w = output_sizes[num_sizes - 1];

      // Onnx spec requires output sizes to be a positive integer, so we are not checking that here
      if (output_size_h % input_size_h != 0) {
        LOGS(logger, VERBOSE) << "Resize: output_size_h: " << output_size_h
                              << " is not a multiple of input_size_h: " << input_size_h;
        return false;
      }

      if (output_size_w % input_size_w != 0) {
        LOGS(logger, VERBOSE) << "Resize: output_size_w: " << output_size_w
                              << " is not a multiple of input_size_w: " << input_size_w;
        return false;
      }
    }
  }

  std::string coord_trans_mode = helper.Get("coordinate_transformation_mode", "half_pixel");
  bool using_asymmetric = coord_trans_mode == "asymmetric";

  if (input_params.create_mlprogram) {
    if (is_nearest) {
      // Potential CoreML operators we could map to:
      //
      // image_resizing.upsample_nearest_neighbor
      // - mode: nearest
      // - coordinate_transformation_mode: asymmetric
      // - 'scales' input
      //
      // image_resizing.resize_nearest_neighbor
      // - mode: nearest
      // - coordinate_transformation_mode: asymmetric
      // - 'sizes' input
      if (!using_asymmetric) {
        LOGS(logger, VERBOSE) << "Resize with 'mode' of 'nearest' requires 'coordinate_transformation_mode' of "
                                 "'asymmetric' . Got: "
                              << coord_trans_mode;
        return false;
      }
    } else {
      assert(is_linear);
      // Potential CoreML operators we could map to:
      //
      // image_resizing.upsample_bilinear
      // - mode: linear
      // - 'scales' input
      // - coordinate_transformation_mode
      //   - half_pixel -> align_corners = false
      //   - align_corners -> align_corners = true
      //
      // image_resizing.resize_bilinear
      // - mode: linear
      // - 'sizes' input
      // - coordinate_transformation_mode -> sampling_mode
      //   - half_pixel -> UNALIGN_CORNERS
      //   - align_corners -> STRICT_ALIGN_CORNERS
      //   - asymmetric -> DEFAULT
      //

      // if output size != 1, coordinate_transformation_mode of pytorch_half_pixel is the same as half_pixel
      if (coord_trans_mode == "pytorch_half_pixel") {
        int64_t h_out{0}, w_out{0};
        if (using_scales) {
          size_t num_scales = output_scales.size();
          h_out = std::llround(input_shape[input_rank - 2] * output_scales[num_scales - 2]);
          w_out = std::llround(input_shape[input_rank - 1] * output_scales[num_scales - 1]);
        } else {
          size_t num_sizes = output_sizes.size();
          h_out = output_sizes[num_sizes - 2];
          w_out = output_sizes[num_sizes - 1];
        }

        if (h_out > 1 && w_out > 1) {
          coord_trans_mode = "half_pixel";
        }
      }

      if (coord_trans_mode == "half_pixel" ||
          coord_trans_mode == "align_corners" ||
          (using_sizes && coord_trans_mode == "asymmetric")) {
        // supported

        // FWIW we could calculate (if shape inferencing didn't already) the output sizes and convert a node with
        // `scales` and co-ord mode of `asymmetric` to having `sizes` input so it's supported.
      } else {
        LOGS(logger, VERBOSE) << "Resize with 'mode' of 'linear' requires 'coordinate_transformation_mode' of "
                                 "'half_pixel', or 'align_corners', or 'asymmetric' with 'sizes' input. Got: "
                              << coord_trans_mode;

        return false;
      }
    }
  } else {
    // NeuralNetwork checks
    if (!using_asymmetric) {
      // align_corners and half_pixel could be supported in ResizeBilinear but as NeuralNetwork is deprecated
      // there's no known value to adding that.
      LOGS(logger, VERBOSE) << "Resize only supports 'asymmetric' coordinate_transformation_mode. Got: "
                            << coord_trans_mode;
      return false;
    }
  }

  return true;
}

void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ResizeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
