//! Module defining body, face and gesture analysis models available to download.
//!
//! See [https://github.com/onnx/models#body_analysis](https://github.com/onnx/models#body_analysis)

use crate::download::{vision::Vision, AvailableOnnxModel, ModelUrl};

/// Body, Face & Gesture Analysis
///
/// > Face detection models identify and/or recognize human faces and emotions in given images. Body and Gesture
/// > Analysis models identify gender and age in given image.
///
/// Source: [https://github.com/onnx/models#body_analysis](https://github.com/onnx/models#body_analysis)
#[derive(Debug, Clone)]
pub enum BodyFaceGestureAnalysis {
    /// A CNN based model for face recognition which learns discriminative features of faces and produces
    /// embeddings for input face images.
    ///
    /// Source: [https://github.com/onnx/models/tree/main/vision/body_analysis/arcface](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface)
    ///
    /// Variant downloaded: ONNX Version 1.3 with Opset Version 8.
    ArcFace,
    /// Deep CNN for emotion recognition trained on images of faces.
    ///
    /// Source: [https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus](https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus)
    ///
    /// Variant downloaded: ONNX Version 1.3 with Opset Version 8.
    EmotionFerPlus,
}

impl ModelUrl for BodyFaceGestureAnalysis {
    fn fetch_url(&self) -> &'static str {
        match self {
            BodyFaceGestureAnalysis::ArcFace => "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
            BodyFaceGestureAnalysis::EmotionFerPlus => "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        }
    }
}

impl From<BodyFaceGestureAnalysis> for AvailableOnnxModel {
    fn from(model: BodyFaceGestureAnalysis) -> Self {
        AvailableOnnxModel::Vision(Vision::BodyFaceGestureAnalysis(model))
    }
}
