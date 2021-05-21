Pod::Spec.new do |spec|
  spec.name         = "OnnxRuntimeBase"
  spec.version      = "1.7.0"
  spec.summary      = "Onnx Runtime C/C++ Package"
  spec.description  = <<-DESC
  Onnx Runtime C/C++ framework pod.
                   DESC

  spec.homepage     = "https://github.com/microsoft/onnxruntime"
  spec.license = { :type => 'MIT' }
  spec.authors      = { "ONNX Runtime" => "onnxruntime@microsoft.com" }
  spec.platform     = :ios
  spec.platform     = :ios, "11.0"
  spec.source       = { :http => "https://0.0.0.0:3456/onnxruntime.zip" }
  spec.vendored_frameworks = 'onnxruntime.framework'
end
