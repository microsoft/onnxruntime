require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

# Expect to return the absolute path of the react native root project dir
root_dir =  File.dirname(File.dirname(__dir__))

common_cpp_flags = '-Wall -Wextra -DUSE_COREML'

Pod::Spec.new do |spec|
  spec.static_framework = true

  spec.name                 = "onnxruntime-react-native"
  spec.version              = package["version"]
  spec.summary              = package["description"]
  spec.homepage             = package["homepage"]
  spec.license              = package["license"]
  spec.authors              = package["author"]

  spec.platforms            = { :ios => "15.1" }
  spec.source               = { :git => "https://github.com/Microsoft/onnxruntime.git", :tag => "rel-#{spec.version}" }

  spec.source_files         = "ios/*.{h,mm}", "cpp/*.{h,cpp}"

  spec.dependency "React-Core"
  spec.dependency "React-callinvoker"
  spec.dependency "onnxruntime-c"

  spec.xcconfig = {
    'OTHER_CPLUSPLUSFLAGS' => common_cpp_flags,
  }

  if (File.exist?(File.join(root_dir, 'package.json')))
    # Read the react native root project directory package.json file
    root_package = JSON.parse(File.read(File.join(root_dir, 'package.json')))
    if (root_package["onnxruntimeExtensionsEnabled"] == 'true')
      spec.dependency "onnxruntime-extensions-c"
      spec.xcconfig = {
        'OTHER_CPLUSPLUSFLAGS' => common_cpp_flags + ' -DORT_ENABLE_EXTENSIONS=1',
      }
    end
  else
    puts "Could not find package.json file in the expected directory: #{root_dir}. ONNX Runtime Extensions will not be enabled."
  end

end
