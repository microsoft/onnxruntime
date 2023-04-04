// #![allow(non_upper_case_globals)]
// #![allow(non_camel_case_types)]
// #![allow(non_snake_case)]
// // Disable clippy and `u128` not being FFI-safe (see #1)
// #![allow(clippy::all)]
// #![allow(improper_ctypes)]

// #[allow(clippy::all)]

// include!(concat!(
//     env!("CARGO_MANIFEST_DIR"),
//     "/src/generated/bindings.rs"
// ));

// #[cfg(target_os = "windows")]
// pub type OnnxEnumInt = i32;
// #[cfg(not(target_os = "windows"))]
// pub type OnnxEnumInt = u32;


#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// Disable clippy and `u128` not being FFI-safe (see #1)
#![allow(clippy::all)]
#![allow(improper_ctypes)]

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/generated/bindings.rs"
));

// include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(target_os = "windows")]
pub type OnnxEnumInt = i32;
#[cfg(not(target_os = "windows"))]
pub type OnnxEnumInt = u32;

pub use libloading::library_filename;

// use std::process;
// println!("{}", env!("OUT_DIR"));
// std::process::exit(-1);