require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

# Expect to return the absolute path of the react native root project dir
root_dir =  File.dirname(File.dirname(__dir__))
root_package_path = File.join(root_dir, 'package.json')
root_package = File.exist?(root_package_path) ? JSON.parse(File.read(root_package_path)) : {}

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
  spec.private_header_files = "ios/*.h", "cpp/*.h"

  if root_package["onnxruntimeVersion"]
    spec.dependency "onnxruntime-c", root_package["onnxruntimeVersion"]
  else
    spec.dependency "onnxruntime-c"
  end

  if respond_to?(:install_modules_dependencies, true)
    install_modules_dependencies(spec)
  else
    spec.dependency "React-Core"
    spec.dependency "React-callinvoker"
  end

  spec.xcconfig = {
    'OTHER_CPLUSPLUSFLAGS' => common_cpp_flags,
  }

  if File.exist?(root_package_path)
    if (root_package["onnxruntimeExtensionsEnabled"] == 'true')
      if root_package["onnxruntimeExtensionsVersion"]
        spec.dependency "onnxruntime-extensions-c", root_package["onnxruntimeExtensionsVersion"]
      else
        spec.dependency "onnxruntime-extensions-c"
      end
      spec.xcconfig = {
        'OTHER_CPLUSPLUSFLAGS' => common_cpp_flags + ' -DORT_ENABLE_EXTENSIONS=1',
      }
    end
  else
    puts "Could not find package.json file in the expected directory: #{root_dir}. ONNX Runtime Extensions will not be enabled."
  end

end
