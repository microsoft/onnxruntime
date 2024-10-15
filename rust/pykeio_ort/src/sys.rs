#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// Disable clippy and `u128` not being FFI-safe
#![allow(clippy::all)]
#![allow(improper_ctypes)]
// bindgen-rs generates test code that dereferences null pointers
#![allow(deref_nullptr)]

include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bindings/bindings.rs"));

#[cfg(target_os = "windows")]
pub type OnnxEnumInt = i32;
#[cfg(not(target_os = "windows"))]
pub type OnnxEnumInt = u32;
