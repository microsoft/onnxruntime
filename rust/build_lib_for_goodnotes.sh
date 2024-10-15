rustup default nightly-2022-08-03 && cargo build -Z build-std=panic_abort,std --target aarch64-apple-ios-macabi --features generate-bindings
rustup default nightly-2022-08-03 && cargo build -Z build-std=panic_abort,std --target x86_64-apple-ios-macabi --features generate-bindings

#Paco uses stable-aarch64-apple-darwin unchanged - rustc 1.67.0 (fc594f156 2023-01-24)
rustup default stable && cargo build --target x86_64-apple-ios --features generate-bindings
rustup default stable && cargo build --target aarch64-apple-ios-sim --features generate-bindings
rustup default stable && cargo build --target aarch64-apple-ios --features generate-bindings


#rustup target add x86_64-apple-darwin
rustup default stable && cargo build --target x86_64-apple-darwin --features generate-bindings
rustup default stable && cargo build --target aarch64-apple-darwin --features generate-bindings


rustup default nightly-2022-08-03 && ORT_LIB_LOCATION=/Users/goodnotesci/goodnotes/gn_onnx/onnx_binaries/onnxruntime-osx-universal2-1.14.1 ORT_STRATEGY=system cargo test -Z build-std=panic_abort,std --target x86_64-apple-ios-macabi --features generate-bindings


rustup default stable && cargo test --target aarch64-apple-ios-sim

rustup default stable && cargo build --target aarch64-apple-ios-sim

python tools/ci_build/github/apple/build_macabi_framework.py --config Release --build_dir /Users/goodnotesci/goodnotes/handwriting_synthesis_rust/paco_ios_release --include_ops_by_config tools/ci_build/github/apple/hws_mobile_package.required_operators.config --path_to_protoc_exe /usr/local/bin/protoc-3.21.12.0 tools/ci_build/github/apple/default_full_macabi_framework_build_settings.json