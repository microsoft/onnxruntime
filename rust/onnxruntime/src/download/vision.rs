//! Module defining computer vision models available to download.
//!
//! See [https://github.com/onnx/models#image_classification](https://github.com/onnx/models#image_classification)

use super::ModelUrl;

pub mod body_face_gesture_analysis;
pub mod domain_based_image_classification;
pub mod image_classification;
pub mod image_manipulation;
pub mod object_detection_image_segmentation;

// Re-exports
pub use body_face_gesture_analysis::BodyFaceGestureAnalysis;
pub use domain_based_image_classification::DomainBasedImageClassification;
pub use image_classification::ImageClassification;
pub use image_manipulation::ImageManipulation;
pub use object_detection_image_segmentation::ObjectDetectionImageSegmentation;

/// Computer vision model
#[derive(Debug, Clone)]
pub enum Vision {
    /// Domain-based Image Classification
    DomainBasedImageClassification(DomainBasedImageClassification),
    /// Image classification model
    ImageClassification(ImageClassification),
    /// Object Detection & Image Segmentation
    ObjectDetectionImageSegmentation(ObjectDetectionImageSegmentation),
    /// Body, Face & Gesture Analysis
    BodyFaceGestureAnalysis(BodyFaceGestureAnalysis),
    /// Image Manipulation
    ImageManipulation(ImageManipulation),
}

impl ModelUrl for Vision {
    fn fetch_url(&self) -> &'static str {
        match self {
            Vision::DomainBasedImageClassification(variant) => variant.fetch_url(),
            Vision::ImageClassification(variant) => variant.fetch_url(),
            Vision::ObjectDetectionImageSegmentation(variant) => variant.fetch_url(),
            Vision::BodyFaceGestureAnalysis(variant) => variant.fetch_url(),
            Vision::ImageManipulation(variant) => variant.fetch_url(),
        }
    }
}
