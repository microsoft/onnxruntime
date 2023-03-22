rustup default nightly-2022-08-03 && cargo build -Z build-std=panic_abort,std --target aarch64-apple-ios-macabi --features generate-bindings
rustup default nightly-2022-08-03 && cargo build -Z build-std=panic_abort,std --target x86_64-apple-ios-macabi --features generate-bindings

#Paco uses stable-aarch64-apple-darwin unchanged - rustc 1.67.0 (fc594f156 2023-01-24)
rustup default stable && cargo build --target x86_64-apple-ios --features generate-bindings
rustup default stable && cargo build --target aarch64-apple-ios-sim --features generate-bindings
rustup default stable && cargo build --target aarch64-apple-ios --features generate-bindings


#rustup target add x86_64-apple-darwin
rustup default stable && cargo build --target x86_64-apple-darwin --features generate-bindings
rustup default stable && cargo build --target aarch64-apple-darwin --features generate-bindings
