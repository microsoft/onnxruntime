pub mod language;
pub mod vision;

/// Represents a type that returns an ONNX model URL.
pub trait ModelUrl {
	/// Returns the model URL associated with this model.
	fn fetch_url(&self) -> &'static str;
}
