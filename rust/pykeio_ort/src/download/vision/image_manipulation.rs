use crate::download::ModelUrl;

/// Image Manipulation
///
/// > Image manipulation models use neural networks to transform input images to modified output images. Some
/// > popular models in this category involve style transfer or enhancing images by increasing resolution.
#[derive(Debug, Clone)]
pub enum ImageManipulation {
	/// Super Resolution
	///
	/// > The Super Resolution machine learning model sharpens and upscales the input image to refine the
	/// > details and improve quality.
	SuperResolution,
	/// Fast Neural Style Transfer
	///
	/// > This artistic style transfer model mixes the content of an image with the style of another image.
	/// > Examples of the styles can be seen
	/// > [in this PyTorch example](https://github.com/pytorch/examples/tree/master/fast_neural_style#models).
	FastNeuralStyleTransfer(FastNeuralStyleTransferStyle)
}

#[derive(Debug, Clone)]
pub enum FastNeuralStyleTransferStyle {
	Mosaic,
	Candy,
	RainPrincess,
	Udnie,
	Pointilism
}

impl ModelUrl for ImageManipulation {
	fn fetch_url(&self) -> &'static str {
		match self {
			ImageManipulation::SuperResolution => {
				"https://github.com/onnx/models/raw/main/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx"
			}
			ImageManipulation::FastNeuralStyleTransfer(style) => style.fetch_url()
		}
	}
}

impl ModelUrl for FastNeuralStyleTransferStyle {
	fn fetch_url(&self) -> &'static str {
		match self {
			FastNeuralStyleTransferStyle::Mosaic => "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx",
			FastNeuralStyleTransferStyle::Candy => "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/candy-9.onnx",
			FastNeuralStyleTransferStyle::RainPrincess => {
				"https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx"
			}
			FastNeuralStyleTransferStyle::Udnie => "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/udnie-9.onnx",
			FastNeuralStyleTransferStyle::Pointilism => {
				"https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/pointilism-9.onnx"
			}
		}
	}
}
