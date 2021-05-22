Pod::Spec.new do |spec|
  spec.name         = "OnnxRuntimeBase"
  spec.version      = "${ORT_VERSION}"
  spec.summary      = "Onnx Runtime C/C++ Package"
  spec.description  = <<-DESC
  Onnx Runtime C/C++ framework pod.
                   DESC

  spec.homepage     = "https://github.com/microsoft/onnxruntime"
  spec.license      = { :type => 'MIT' }
  spec.authors      = { "ONNX Runtime" => "onnxruntime@microsoft.com" }
  spec.platform     = :ios
  spec.platform     = :ios, '13.0'
  spec.source       = { :http => 'file:' + '${ORT_BASE_FRAMEWORK_FILE}' }
  spec.vendored_frameworks = 'onnxruntime.framework'
end
