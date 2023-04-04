//! Models for body, face, & gesture analysis.

use crate::download::ModelUrl;

/// Models for body, face, & gesture analysis.
#[derive(Debug, Clone)]
pub enum BodyFaceGestureAnalysis {
	/// A CNN based model for face recognition which learns discriminative features of faces and produces embeddings for
	/// input face images.
	ArcFace,
	/// Deep CNN for emotion recognition trained on images of faces.
	EmotionFerPlus
}

impl ModelUrl for BodyFaceGestureAnalysis {
	fn fetch_url(&self) -> &'static str {
		match self {
			BodyFaceGestureAnalysis::ArcFace => "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
			BodyFaceGestureAnalysis::EmotionFerPlus => {
				"https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
			}
		}
	}
}
