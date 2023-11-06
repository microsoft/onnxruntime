// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace AttrName
{
    static constexpr const char* AcrossChannels = "across_channels";
    static constexpr const char* ActivationAlpha = "activation_alpha";
    static constexpr const char* ActivationBeta = "activation_beta";
    static constexpr const char* Activations = "activations";
    static constexpr const char* AllowZero = "allowzero";
    static constexpr const char* Alpha = "alpha";
    static constexpr const char* AlignCorners = "align_corners";
    static constexpr const char* AutoPad = "auto_pad";
    static constexpr const char* Axes = "axes";
    static constexpr const char* Axis = "axis";
    static constexpr const char* AxisW = "axis_w";
    static constexpr const char* BatchAxis = "batch_axis";
    static constexpr const char* BatchDimensions = "batch_dims";
    static constexpr const char* Beta = "beta";
    static constexpr const char* Bias = "bias";
    static constexpr const char* BlockSize = "blocksize";
    static constexpr const char* Border = "border";
    static constexpr const char* Broadcast = "broadcast";
    static constexpr const char* CeilMode = "ceil_mode";
    static constexpr const char* ChannelsLast = "channels_last";
    static constexpr const char* Clip = "clip";
    static constexpr const char* CoordinateTransformationMode = "coordinate_transformation_mode";
    static constexpr const char* CountIncludePad = "count_include_pad";
    static constexpr const char* CubicCoefficientA = "cubic_coeff_a";
    static constexpr const char* DetectPositive = "detect_positive";
    static constexpr const char* DetectNegative = "detect_negative";
    static constexpr const char* Dilations = "dilations";
    static constexpr const char* Direction = "direction";
    static constexpr const char* Dtype = "dtype";
    static constexpr const char* End = "end";
    static constexpr const char* Ends = "ends";
    static constexpr const char* Epsilon = "epsilon";
    static constexpr const char* Equation = "equation";
    static constexpr const char* ExcludeOutside = "exclude_outside";
    static constexpr const char* Exclusive = "exclusive";
    static constexpr const char* Exponent = "exponent";
    static constexpr const char* ExtrapolationValue = "extrapolation_value";
    static constexpr const char* Fmod = "fmod";
    static constexpr const char* Gamma = "gamma";
    static constexpr const char* Group = "group";
    static constexpr const char* HeightScale = "height_scale";
    static constexpr const char* HiddenSize = "hidden_size";
    static constexpr const char* High = "high";
    static constexpr const char* InputForget = "input_forget";
    static constexpr const char* K = "k";
    static constexpr const char* KeepDims = "keepdims";
    static constexpr const char* KernelShape = "kernel_shape";
    static constexpr const char* LinearBeforeReset = "linear_before_reset";
    static constexpr const char* Lambda = "lambd"; // Deliberate typo to match ONNX spec.
    static constexpr const char* Largest = "largest";
    static constexpr const char* Layout = "layout";
    static constexpr const char* Low = "low";
    static constexpr const char* Max = "max";
    static constexpr const char* Mean = "mean";
    static constexpr const char* Min = "min";
    static constexpr const char* Mode = "mode";
    static constexpr const char* NearestMode = "nearest_mode";
    static constexpr const char* NewAxis = "new_axis";
    static constexpr const char* NoopWithEmptyAxes = "noop_with_empty_axes";
    static constexpr const char* NormalizeVariance = "normalize_variance";
    static constexpr const char* NumOutputs = "num_outputs";
    static constexpr const char* P = "p";
    static constexpr const char* PaddingMode = "padding_mode";
    static constexpr const char* OutputHeight = "output_height";
    static constexpr const char* OutputShape = "output_shape";
    static constexpr const char* OutputPadding = "output_padding";
    static constexpr const char* OutputWidth = "output_width";
    static constexpr const char* Pads = "pads";
    static constexpr const char* PooledShape = "pooled_shape";
    static constexpr const char* Reduction = "reduction";
    static constexpr const char* Reverse = "reverse";
    static constexpr const char* SampleSize = "sample_size";
    static constexpr const char* SamplingRatio = "sampling_ratio";
    static constexpr const char* Scale = "scale";
    static constexpr const char* Scales = "scales";
    static constexpr const char* Seed = "seed";
    static constexpr const char* SelectLastIndex = "select_last_index";
    static constexpr const char* Shape = "shape";
    static constexpr const char* Size = "size";
    static constexpr const char* Sorted = "sorted";
    static constexpr const char* Spatial = "spatial";
    static constexpr const char* SpatialScale = "spatial_scale";
    static constexpr const char* Split = "split";
    static constexpr const char* Start = "start";
    static constexpr const char* Starts = "starts";
    static constexpr const char* Steepness = "steepness";
    static constexpr const char* StorageOrder = "storage_order";
    static constexpr const char* Strides = "strides";
    static constexpr const char* Tiles = "tiles";
    static constexpr const char* TimeAxis = "time_axis";
    static constexpr const char* To = "to";
    static constexpr const char* TrainingMode = "training_mode";
    static constexpr const char* TransA = "transA";
    static constexpr const char* TransBatchA = "transBatchA";
    static constexpr const char* TransB = "transB";
    static constexpr const char* TransBatchB = "transBatchB";
    static constexpr const char* Upper = "upper";
    static constexpr const char* Value = "value";
    static constexpr const char* WidthScale = "width_scale";
    static constexpr const char* QkvHiddenSizes = "qkv_hidden_sizes";
    static constexpr const char* Unidirectional = "unidirectional";
    static constexpr const char* NumHeads = "num_heads";

    static constexpr const char* FusedActivation = "fused_activation";
    static constexpr const char* FusedActivationDomain = "fused_activation_domain";
    static constexpr const char* FusedActivationSinceVersion = "fused_activation_since_version";
    static constexpr const char* FusedAlpha = "fused_alpha";
    static constexpr const char* FusedBeta = "fused_beta";
    static constexpr const char* FusedGamma = "fused_gamma";
    static constexpr const char* FusedRatio = "fused_ratio";
    static constexpr const char* MaskFilterValue = "mask_filter_value";
    static constexpr const char* DoRotary = "do_rotary";
    static constexpr const char* Activation = "activation";
    static constexpr const char* Groups = "groups";

    static constexpr const char* GraphFusedActivation = "activation";
    static constexpr const char* GraphFusedAxis = "activation_axis";

} // namespace AttrName

namespace AttrValue
{
    static constexpr const char* ActivationRelu = "Relu";
    static constexpr const char* ActivationLeakyRelu = "LeakyRelu";
    static constexpr const char* ActivationThresholdedRelu = "ThresholdedRelu";
    static constexpr const char* ActivationTanh = "Tanh";
    static constexpr const char* ActivationScaledTanh = "ScaledTanh";
    static constexpr const char* ActivationSigmoid = "Sigmoid";
    static constexpr const char* ActivationSigmoidHard = "HardSigmoid";
    static constexpr const char* ActivationElu = "Elu";
    static constexpr const char* ActivationSoftsign = "Softsign";
    static constexpr const char* ActivationSoftplus = "Softplus";
    static constexpr const char* Bilinear = "BILINEAR";
    static constexpr const char* Constant = "constant";
    static constexpr const char* DirectionBidirectional = "bidirectional";
    static constexpr const char* DirectionForward = "forward";
    static constexpr const char* DirectionReverse = "reverse";
    static constexpr const char* Edge = "edge";
    static constexpr const char* NCHW = "NCHW";
    static constexpr const char* NearestNeighbor = "NN";
    static constexpr const char* NotSet = "NOTSET";
    static constexpr const char* Reflect = "reflect";

} // namespace AttrValue
