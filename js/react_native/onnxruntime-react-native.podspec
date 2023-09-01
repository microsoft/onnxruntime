require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

# Expect to return the absolute path of the react native root project dir
root_dir =  File.dirname(File.dirname(__dir__))

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

  spec.xcconfig = {
    'OTHER_CPLUSPLUSFLAGS' => '-Wall -Wextra',
  }

  if (File.exist?(File.join(root_dir, 'package.json')))
    # Read the react native root project directory package.json file
    root_package = JSON.parse(File.read(File.join(root_dir, 'package.json')))
    if (root_package["onnxruntimeExtensionsEnabled"] == 'true')
      spec.dependency "onnxruntime-extensions-c"
      spec.xcconfig = {
        'OTHER_CPLUSPLUSFLAGS' => '-DORT_ENABLE_EXTENSIONS=1 -Wall -Wextra',
      }
    end
  else
    puts "Could not find package.json file in the expected directory: #{root_dir}. ONNX Runtime Extensions will not be enabled."
  end

end
