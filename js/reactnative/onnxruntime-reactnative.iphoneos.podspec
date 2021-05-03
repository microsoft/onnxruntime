require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "onnxruntime-reactnative.iphoneos"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => "10.0" }
  s.source       = { :git => "https://github.com/Microsoft/onnxruntime.git", :tag => "#{s.version}" }

  
  s.source_files = "ios/*.{h,mm}", "ios/Libraries/**/*.h"

  s.dependency "React-Core"

  s.subspec 'Onnxruntime' do |onnxruntime|
    onnxruntime.preserve_paths = 'ios/Libraries/onnxruntime/include/*.h'
    onnxruntime.vendored_libraries = 'ios/Libraries/onnxruntime/lib/iphoneos/libonnxruntime.1.6.0.dylib'
    onnxruntime.xcconfig = { 'HEADER_SEARCH_PATHS' => "${PODS_ROOT}/#{s.name}/Libraries/onnxruntime/include/**", 'LIBRARY_SEARCH_PATHS' => "${PODS_ROOT)/#{s.name}/Libraries/onnxruntime/lib/**" }
  end

end
