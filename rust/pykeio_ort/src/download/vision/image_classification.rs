//! Models for image classification.

#![allow(clippy::upper_case_acronyms)]

use crate::download::ModelUrl;

/// Models for image classification.
#[derive(Debug, Clone)]
pub enum ImageClassification {
	/// Image classification aimed for mobile targets.
	///
	/// > MobileNet models perform image classification - they take images as input and classify the major
	/// > object in the image into a set of pre-defined classes. They are trained on ImageNet dataset which
	/// > contains images from 1000 classes. MobileNet models are also very efficient in terms of speed and
	/// > size and hence are ideal for embedded and mobile applications.
	MobileNet,
	/// Image classification, trained on ImageNet with 1000 classes.
	///
	/// > ResNet models provide very high accuracies with affordable model sizes. They are ideal for cases when
	/// > high accuracy of classification is required.
	ResNet(ResNet),
	/// A small CNN with AlexNet level accuracy on ImageNet with 50x fewer parameters.
	///
	/// > SqueezeNet is a small CNN which achieves AlexNet level accuracy on ImageNet with 50x fewer parameters.
	/// > SqueezeNet requires less communication across servers during distributed training, less bandwidth to
	/// > export a new model from the cloud to an autonomous car and more feasible to deploy on FPGAs and other
	/// > hardware with limited memory.
	SqueezeNet,
	/// Image classification, trained on ImageNet with 1000 classes.
	///
	/// > VGG models provide very high accuracies but at the cost of increased model sizes. They are ideal for
	/// > cases when high accuracy of classification is essential and there are limited constraints on model sizes.
	Vgg(Vgg),
	/// Convolutional neural network for classification, which competed in the ImageNet Large Scale Visual Recognition
	/// Challenge in 2012.
	AlexNet,
	/// Convolutional neural network for classification, which competed in the ImageNet Large Scale Visual Recognition
	/// Challenge in 2014.
	GoogleNet,
	/// Variant of AlexNet, it's the name of a convolutional neural network for classification, which competed in the
	/// ImageNet Large Scale Visual Recognition Challenge in 2012.
	CaffeNet,
	/// Convolutional neural network for detection.
	///
	/// > This model was made by transplanting the R-CNN SVM classifiers into a fc-rcnn classification layer.
	RcnnIlsvrc13,
	/// Convolutional neural network for classification.
	DenseNet121,
	/// Google's Inception
	Inception(InceptionVersion),
	/// Computationally efficient CNN architecture designed specifically for mobile devices with very limited computing
	/// power.
	ShuffleNet(ShuffleNetVersion),
	/// Deep convolutional networks for classification.
	///
	/// > This model's 4th layer has 512 maps instead of 1024 maps mentioned in the paper.
	ZFNet512,
	/// Image classification model that achieves state-of-the-art accuracy.
	///
	/// > It is designed to run on mobile CPU, GPU, and EdgeTPU devices, allowing for applications on mobile and loT,
	/// where computational resources are limited.
	EfficientNetLite4
}

#[derive(Debug, Clone)]
pub enum InceptionVersion {
	V1,
	V2
}

#[derive(Debug, Clone)]
pub enum ResNet {
	V1(ResNetV1),
	V2(ResNetV2)
}

#[derive(Debug, Clone)]
pub enum ResNetV1 {
	/// ResNet v1 with 18 layers.
	ResNet18,
	/// ResNet v1 with 34 layers.
	ResNet34,
	/// ResNet v1 with 50 layers.
	ResNet50,
	/// ResNet v1 with 101 layers.
	ResNet101,
	/// ResNet v1 with 152 layers.
	ResNet152
}

#[derive(Debug, Clone)]
pub enum ResNetV2 {
	ResNet18,
	ResNet34,
	ResNet50,
	ResNet101,
	ResNet152
}

#[derive(Debug, Clone)]
pub enum Vgg {
	/// VGG with 16 convolutional layers
	Vgg16,
	/// VGG with 16 convolutional layers, with batch normalization applied after each convolutional layer.
	///
	/// The batch normalization leads to better convergence and slightly better accuracies.
	Vgg16Bn,
	/// VGG with 19 convolutional layers
	Vgg19,
	/// VGG with 19 convolutional layers, with batch normalization applied after each convolutional layer.
	///
	/// The batch normalization leads to better convergence and slightly better accuracies.
	Vgg19Bn
}

/// Computationally efficient CNN architecture designed specifically for mobile devices with very limited computing
/// power.
#[derive(Debug, Clone)]
pub enum ShuffleNetVersion {
	/// The original ShuffleNet.
	V1,
	/// ShuffleNetV2 is an improved architecture that is the state-of-the-art in terms of speed and accuracy tradeoff
	/// used for image classification.
	V2
}

impl ModelUrl for ImageClassification {
	fn fetch_url(&self) -> &'static str {
		match self {
			ImageClassification::MobileNet => "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
			ImageClassification::SqueezeNet => "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx",
			ImageClassification::Inception(version) => version.fetch_url(),
			ImageClassification::ResNet(version) => version.fetch_url(),
			ImageClassification::Vgg(variant) => variant.fetch_url(),
			ImageClassification::AlexNet => "https://github.com/onnx/models/raw/main/vision/classification/alexnet/model/bvlcalexnet-9.onnx",
			ImageClassification::GoogleNet => {
				"https://github.com/onnx/models/raw/main/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx"
			}
			ImageClassification::CaffeNet => "https://github.com/onnx/models/raw/main/vision/classification/caffenet/model/caffenet-9.onnx",
			ImageClassification::RcnnIlsvrc13 => "https://github.com/onnx/models/raw/main/vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.onnx",
			ImageClassification::DenseNet121 => "https://github.com/onnx/models/raw/main/vision/classification/densenet-121/model/densenet-9.onnx",
			ImageClassification::ShuffleNet(version) => version.fetch_url(),
			ImageClassification::ZFNet512 => "https://github.com/onnx/models/raw/main/vision/classification/zfnet-512/model/zfnet512-9.onnx",
			ImageClassification::EfficientNetLite4 => {
				"https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4.onnx"
			}
		}
	}
}

impl ModelUrl for InceptionVersion {
	fn fetch_url(&self) -> &'static str {
		match self {
			InceptionVersion::V1 => {
				"https://github.com/onnx/models/raw/main/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.onnx"
			}
			InceptionVersion::V2 => {
				"https://github.com/onnx/models/raw/main/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx"
			}
		}
	}
}

impl ModelUrl for ResNet {
	fn fetch_url(&self) -> &'static str {
		match self {
			ResNet::V1(variant) => variant.fetch_url(),
			ResNet::V2(variant) => variant.fetch_url()
		}
	}
}

impl ModelUrl for ResNetV1 {
	fn fetch_url(&self) -> &'static str {
		match self {
			ResNetV1::ResNet18 => "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v1-7.onnx",
			ResNetV1::ResNet34 => "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet34-v1-7.onnx",
			ResNetV1::ResNet50 => "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx",
			ResNetV1::ResNet101 => "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet101-v1-7.onnx",
			ResNetV1::ResNet152 => "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet152-v1-7.onnx"
		}
	}
}

impl ModelUrl for ResNetV2 {
	fn fetch_url(&self) -> &'static str {
		match self {
			ResNetV2::ResNet18 => "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v2-7.onnx",
			ResNetV2::ResNet34 => "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet34-v2-7.onnx",
			ResNetV2::ResNet50 => "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx",
			ResNetV2::ResNet101 => "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet101-v2-7.onnx",
			ResNetV2::ResNet152 => "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet152-v2-7.onnx"
		}
	}
}

impl ModelUrl for Vgg {
	fn fetch_url(&self) -> &'static str {
		match self {
			Vgg::Vgg16 => "https://github.com/onnx/models/raw/main/vision/classification/vgg/model/vgg16-7.onnx",
			Vgg::Vgg16Bn => "https://github.com/onnx/models/raw/main/vision/classification/vgg/model/vgg16-bn-7.onnx",
			Vgg::Vgg19 => "https://github.com/onnx/models/raw/main/vision/classification/vgg/model/vgg19-7.onnx",
			Vgg::Vgg19Bn => "https://github.com/onnx/models/raw/main/vision/classification/vgg/model/vgg19-bn-7.onnx"
		}
	}
}

impl ModelUrl for ShuffleNetVersion {
	fn fetch_url(&self) -> &'static str {
		match self {
			ShuffleNetVersion::V1 => "https://github.com/onnx/models/raw/main/vision/classification/shufflenet/model/shufflenet-9.onnx",
			ShuffleNetVersion::V2 => "https://github.com/onnx/models/raw/main/vision/classification/shufflenet/model/shufflenet-v2-10.onnx"
		}
	}
}
