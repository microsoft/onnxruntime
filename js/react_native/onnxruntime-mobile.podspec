require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |spec|
  spec.name                 = "onnxruntime-mobile"
  spec.version              = package["version"]
  spec.summary              = "ONNX Runtime C/C++ Package"
  spec.description          = <<-DESC
  ONNX Runtime C/C++ framework pod.
                   DESC

  spec.homepage             = "https://github.com/microsoft/onnxruntime"
  spec.license              = { :type => 'MIT' }
  spec.authors              = { "ONNX Runtime" => "onnxruntime@microsoft.com" }
  spec.platform             = :ios, '13.0'
  # if you are going to use a file as the spec.source, add 'file:' before your file path
  spec.source               = { :http => 'file:' + __dir__ + '/local_pods/onnxruntime-mobile.zip' }
  spec.vendored_frameworks  = 'onnxruntime.framework'
  spec.source_files         = 'onnxruntime.framework/Headers/*.h'
end
