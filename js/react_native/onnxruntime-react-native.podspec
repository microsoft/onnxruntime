require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

# Expect to return the absolute path of the react native root project dir
current_dir = File.dirname(__FILE__)
root_dir =  File.dirname(File.dirname(current_dir))

Pod::Spec.new do |spec|
  spec.static_framework = true

  spec.name                 = "onnxruntime-react-native"
  spec.version              = package["version"]
  spec.summary              = package["description"]
  spec.homepage             = package["homepage"]
  spec.license              = package["license"]
  spec.authors              = package["author"]

  spec.platforms            = { :ios => "12.4" }
  spec.source               = { :git => "https://github.com/Microsoft/onnxruntime.git", :tag => "rel-#{spec.version}" }

  spec.source_files         = "ios/*.{h,mm}"

  spec.dependency "React-Core"
  spec.dependency "onnxruntime-c"

  # Read the ort package name field in react native root directory
  if (File.exist?(File.join(root_dir, 'package.json')))
    if (root_package["ortPackageName"] == "onnxruntime-ext")
      spec.dependency "onnxruntime-extensions-c"
    end
  end
end
