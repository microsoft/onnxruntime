#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bindings/linux/x86_64/bindings.rs"));

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bindings/linux/aarch64/bindings.rs"));

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bindings/macos/x86_64/bindings.rs"));

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bindings/macos/aarch64/bindings.rs"));

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bindings/windows/x86_64/bindings.rs"));

#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bindings/windows/aarch64/bindings.rs"));

#[cfg(all(target_os = "ios", target_arch = "x86_64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bindings/ios/x86_64/bindings.rs"));

#[cfg(all(target_os = "ios", target_arch = "aarch64"))]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bindings/ios/aarch64/bindings.rs"));
