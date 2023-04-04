//! Models for computer vision.

pub mod body_face_gesture_analysis;
pub mod domain_based_image_classification;
pub mod image_classification;
pub mod image_manipulation;
pub mod object_detection_image_segmentation;

pub use body_face_gesture_analysis::BodyFaceGestureAnalysis;
pub use domain_based_image_classification::DomainBasedImageClassification;
pub use image_classification::{ImageClassification, InceptionVersion, ResNet, ResNetV1, ResNetV2, ShuffleNetVersion, Vgg};
pub use image_manipulation::{FastNeuralStyleTransferStyle, ImageManipulation};
pub use object_detection_image_segmentation::ObjectDetectionImageSegmentation;
