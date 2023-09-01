//! Module defining image manipulation models available to download.
//!
//! See [https://github.com/onnx/models#image_manipulation](https://github.com/onnx/models#image_manipulation)

use crate::download::{vision::Vision, AvailableOnnxModel, ModelUrl};

/// Image Manipulation
///
/// > Image manipulation models use neural networks to transform input images to modified output images. Some
/// > popular models in this category involve style transfer or enhancing images by increasing resolution.
///
/// Source: [https://github.com/onnx/models#image_manipulation](https://github.com/onnx/models#image_manipulation)
#[derive(Debug, Clone)]
pub enum ImageManipulation {
    /// Super Resolution
    ///
    /// > The Super Resolution machine learning model sharpens and upscales the input image to refine the
    /// > details and improve quality.
    ///
    /// Source: [https://github.com/onnx/models/tree/main/vision/super_resolution/sub_pixel_cnn_2016](https://github.com/onnx/models/tree/main/vision/super_resolution/sub_pixel_cnn_2016)
    ///
    /// Variant downloaded: ONNX Version 1.5 with Opset Version 10.
    SuperResolution,
    /// Fast Neural Style Transfer
    ///
    /// > This artistic style transfer model mixes the content of an image with the style of another image.
    /// > Examples of the styles can be seen
    /// > [in this PyTorch example](https://github.com/pytorch/examples/tree/main/fast_neural_style#models).
    ///
    /// Source: [https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style](https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style)
    FastNeuralStyleTransfer(FastNeuralStyleTransferStyle),
}

/// Fast Neural Style Transfer Style
///
/// Source: [https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style](https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style)
///
/// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
#[derive(Debug, Clone)]
pub enum FastNeuralStyleTransferStyle {
    /// Mosaic style
    Mosaic,
    /// Candy style
    Candy,
    /// RainPrincess style
    RainPrincess,
    /// Udnie style
    Udnie,
    /// Pointilism style
    Pointilism,
}

impl ModelUrl for ImageManipulation {
    fn fetch_url(&self) -> &'static str {
        match self {
            ImageManipulation::SuperResolution => "https://github.com/onnx/models/raw/main/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx",
            ImageManipulation::FastNeuralStyleTransfer(style) => style.fetch_url(),
        }
    }
}

impl ModelUrl for FastNeuralStyleTransferStyle {
    fn fetch_url(&self) -> &'static str {
        match self {
            FastNeuralStyleTransferStyle::Mosaic => "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx",
            FastNeuralStyleTransferStyle::Candy => "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/candy-9.onnx",
            FastNeuralStyleTransferStyle::RainPrincess => "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx",
            FastNeuralStyleTransferStyle::Udnie => "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/udnie-9.onnx",
            FastNeuralStyleTransferStyle::Pointilism => "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/pointilism-9.onnx",
        }
    }
}

impl From<ImageManipulation> for AvailableOnnxModel {
    fn from(model: ImageManipulation) -> Self {
        AvailableOnnxModel::Vision(Vision::ImageManipulation(model))
    }
}

impl From<FastNeuralStyleTransferStyle> for AvailableOnnxModel {
    fn from(style: FastNeuralStyleTransferStyle) -> Self {
        AvailableOnnxModel::Vision(Vision::ImageManipulation(
            ImageManipulation::FastNeuralStyleTransfer(style),
        ))
    }
}
