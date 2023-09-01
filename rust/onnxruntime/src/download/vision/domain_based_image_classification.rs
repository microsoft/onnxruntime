//! Module defining domain-based image classification models available to download.
//!
//! See [https://github.com/onnx/models#domain-based-image-classification-](https://github.com/onnx/models#domain-based-image-classification-)

use crate::download::{vision::Vision, AvailableOnnxModel, ModelUrl};

/// Image classification model
#[derive(Debug, Clone)]
pub enum DomainBasedImageClassification {
    /// Handwritten digits prediction using CNN
    ///
    /// Source: [https://github.com/onnx/models/tree/main/vision/classification/mnist](https://github.com/onnx/models/tree/main/vision/classification/mnist)
    ///
    /// Variant downloaded: ONNX Version 1.3 with Opset Version 8.
    Mnist,
}

impl ModelUrl for DomainBasedImageClassification {
    fn fetch_url(&self) -> &'static str {
        match self {
            DomainBasedImageClassification::Mnist => "https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-8.onnx",
        }
    }
}

impl From<DomainBasedImageClassification> for AvailableOnnxModel {
    fn from(model: DomainBasedImageClassification) -> Self {
        AvailableOnnxModel::Vision(Vision::DomainBasedImageClassification(model))
    }
}
