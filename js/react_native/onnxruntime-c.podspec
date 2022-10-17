require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |spec|
  spec.name                 = "onnxruntime-c"
  spec.version              = package["version"]
  spec.summary              = "ONNX Runtime C/C++ Pod"
  spec.description          = <<-DESC
  A pod for the ONNX Runtime C/C++ library.
                                DESC

  spec.homepage             = "https://github.com/microsoft/onnxruntime"
  spec.license              = { :type => "MIT", :file => "LICENSE" }
  spec.authors              = { "ONNX Runtime" => "onnxruntime@microsoft.com" }
  spec.platform             = :ios, '11.0'
  # if you are going to use a file as the spec.source, add 'file:' before your file path
  spec.source               = { :http => 'file:' + __dir__ + '/local_pods/onnxruntime-c.zip' }
  spec.vendored_frameworks  = "onnxruntime.xcframework"
  spec.static_framework     = true
  spec.weak_framework       = [ "CoreML" ]
  spec.source_files         = "Headers/*.h"
  spec.preserve_paths       = [ "LICENSE" ]
  spec.library              = "c++"
  spec.pod_target_xcconfig  = {
    "OTHER_CPLUSPLUSFLAGS" => "-fvisibility=hidden -fvisibility-inlines-hidden",
  }
end
