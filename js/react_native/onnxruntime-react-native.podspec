require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |spec|
  spec.static_framework = true

  spec.name                 = "onnxruntime-react-native"
  spec.version              = package["version"]
  spec.summary              = package["description"]
  spec.homepage             = package["homepage"]
  spec.license              = package["license"]
  spec.authors              = package["author"]

  spec.platforms            = { :ios => "11.0" }
  spec.source               = { :git => "https://github.com/Microsoft/onnxruntime.git", :tag => "rel-#{spec.version}" }

  spec.source_files         = "ios/*.{h,mm}"

  spec.dependency "React-Core"
  spec.dependency "onnxruntime-mobile-c"
end
