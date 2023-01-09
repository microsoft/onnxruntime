#![warn(missing_docs)]

//! ONNX Runtime
//!
//! This crate is a (safe) wrapper around Microsoft's [ONNX Runtime](https://github.com/microsoft/onnxruntime/)
//! through its C API.
//!
//! From its [GitHub page](https://github.com/microsoft/onnxruntime/):
//!
//! > ONNX Runtime is a cross-platform, high performance ML inferencing and training accelerator.
//!
//! The (highly) unsafe [C API](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_c_api.h)
//! is wrapped using bindgen as [`onnxruntime-sys`](https://crates.io/crates/onnxruntime-sys).
//!
//! The unsafe bindings are wrapped in this crate to expose a safe API.
//!
//! For now, efforts are concentrated on the inference API. Training is _not_ supported.
//!
//! # Example
//!
//! The C++ example that uses the C API
//! ([`C_Api_Sample.cpp`](https://github.com/microsoft/onnxruntime/blob/v1.3.1/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp))
//! was ported to
//! [`onnxruntime`](https://github.com/nbigaouette/onnxruntime-rs/blob/main/onnxruntime/examples/sample.rs).
//!
//! First, an environment must be created using and [`EnvBuilder`](environment/struct.EnvBuilder.html):
//!
//! ```no_run
//! # use std::error::Error;
//! # use onnxruntime::{environment::Environment, LoggingLevel};
//! # fn main() -> Result<(), Box<dyn Error>> {
//! let environment = Environment::builder()
//!     .with_name("test")
//!     .with_log_level(LoggingLevel::Verbose)
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! Then a [`Session`](session/struct.Session.html) is created from the environment, some options and an ONNX model file:
//!
//! ```no_run
//! # use std::error::Error;
//! # use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel};
//! # fn main() -> Result<(), Box<dyn Error>> {
//! # let environment = Environment::builder()
//! #     .with_name("test")
//! #     .with_log_level(LoggingLevel::Verbose)
//! #     .build()?;
//! let mut session = environment
//!     .new_session_builder()?
//!     .with_graph_optimization_level(GraphOptimizationLevel::Basic)?
//!     .with_intra_op_num_threads(1)?
//!     .with_model_from_file("squeezenet.onnx")?;
//! # Ok(())
//! # }
//! ```
//!
#![cfg_attr(
    feature = "model-fetching",
    doc = r##"
Instead of loading a model from file using [`with_model_from_file()`](session/struct.SessionBuilder.html#method.with_model_from_file),
a model can be fetched directly from the [ONNX Model Zoo](https://github.com/onnx/models) using
[`with_model_downloaded()`](session/struct.SessionBuilder.html#method.with_model_downloaded) method
(requires the `model-fetching` feature).

```no_run
# use std::error::Error;
# use onnxruntime::{environment::Environment, download::vision::ImageClassification, LoggingLevel, GraphOptimizationLevel};
# fn main() -> Result<(), Box<dyn Error>> {
# let environment = Environment::builder()
#     .with_name("test")
#     .with_log_level(LoggingLevel::Verbose)
#     .build()?;
let mut session = environment
    .new_session_builder()?
    .with_graph_optimization_level(GraphOptimizationLevel::Basic)?
    .with_intra_op_num_threads(1)?
    .with_model_downloaded(ImageClassification::SqueezeNet)?;
# Ok(())
# }
```

See [`AvailableOnnxModel`](download/enum.AvailableOnnxModel.html) for the different models available
to download.
"##
)]
//!
//! Inference will be run on data passed as an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html).
//!
//! ```no_run
//! # use std::error::Error;
//! # use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel, tensor::construct::ConstructTensor};
//! # fn main() -> Result<(), Box<dyn Error>> {
//! # let environment = Environment::builder()
//! #     .with_name("test")
//! #     .with_log_level(LoggingLevel::Verbose)
//! #     .build()?;
//! # let mut session = environment
//! #     .new_session_builder()?
//! #     .with_graph_optimization_level(GraphOptimizationLevel::Basic)?
//! #     .with_intra_op_num_threads(1)?
//! #     .with_model_from_file("squeezenet.onnx")?;
//! let array = ndarray::Array::linspace(0.0_f32, 1.0, 100);
//! // Multiple inputs and outputs are possible
//! let input_tensor = vec![array.into()];
//! let outputs = session.run(input_tensor)?;
//! # Ok(())
//! # }
//! ```
//!
//! The outputs are of type [`OrtOwnedTensor`](tensor/ort_owned_tensor/struct.OrtOwnedTensor.html)s inside a vector,
//! with the same length as the inputs.
//!
//! See the [`sample.rs`](https://github.com/nbigaouette/onnxruntime-rs/blob/main/onnxruntime/examples/sample.rs)
//! example for more details.

use onnxruntime_sys as sys;

// Make functions `extern "stdcall"` for Windows 32bit.
// This behaviors like `extern "system"`.
#[cfg(all(target_os = "windows", target_arch = "x86"))]
macro_rules! extern_system_fn {
    ($(#[$meta:meta])* fn $($tt:tt)*) => ($(#[$meta])* extern "stdcall" fn $($tt)*);
    ($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => ($(#[$meta])* $vis extern "stdcall" fn $($tt)*);
    ($(#[$meta:meta])* unsafe fn $($tt:tt)*) => ($(#[$meta])* unsafe extern "stdcall" fn $($tt)*);
    ($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => ($(#[$meta])* $vis unsafe extern "stdcall" fn $($tt)*);
}

// Make functions `extern "C"` for normal targets.
// This behaviors like `extern "system"`.
#[cfg(not(all(target_os = "windows", target_arch = "x86")))]
macro_rules! extern_system_fn {
    ($(#[$meta:meta])* fn $($tt:tt)*) => ($(#[$meta])* extern "C" fn $($tt)*);
    ($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => ($(#[$meta])* $vis extern "C" fn $($tt)*);
    ($(#[$meta:meta])* unsafe fn $($tt:tt)*) => ($(#[$meta])* unsafe extern "C" fn $($tt)*);
    ($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => ($(#[$meta])* $vis unsafe extern "C" fn $($tt)*);
}

pub mod download;
pub mod environment;
pub mod error;
mod memory;
pub mod session;
pub mod tensor;

// Re-export
pub use error::{OrtApiError, OrtError, Result};
use sys::OnnxEnumInt;

// Re-export ndarray as it's part of the public API anyway
pub use ndarray;

fn g_ort() -> sys::OrtApi {
    let base = unsafe { sys::OrtGetApiBase() };

    let api_version = sys::ORT_API_VERSION;

    let api_ptr_mut = unsafe { (*base).GetApi.unwrap()(api_version) };

    unsafe { *api_ptr_mut }
}

fn char_p_to_string(raw: *const i8) -> Result<String> {
    let c_string = unsafe { std::ffi::CStr::from_ptr(raw as *mut i8).to_owned() };

    match c_string.into_string() {
        Ok(string) => Ok(string),
        Err(e) => Err(OrtApiError::IntoStringError(e)),
    }
    .map_err(OrtError::StringConversion)
}

mod onnxruntime {
    //! Module containing a custom logger, used to catch the runtime's own logging and send it
    //! to Rust's tracing logging instead.

    use std::ffi::CStr;
    use tracing::{debug, error, info, span, trace, warn, Level};

    use onnxruntime_sys as sys;

    /// Runtime's logging sends the code location where the log happened, will be parsed to this struct.
    #[derive(Debug)]
    struct CodeLocation<'a> {
        file: &'a str,
        line_number: &'a str,
        function: &'a str,
    }

    impl<'a> From<&'a str> for CodeLocation<'a> {
        fn from(code_location: &'a str) -> Self {
            let mut splitter = code_location.split(' ');
            let file_and_line_number = splitter.next().unwrap_or("<unknown file:line>");
            let function = splitter.next().unwrap_or("<unknown module>");
            let mut file_and_line_number_splitter = file_and_line_number.split(':');
            let file = file_and_line_number_splitter
                .next()
                .unwrap_or("<unknown file>");
            let line_number = file_and_line_number_splitter
                .next()
                .unwrap_or("<unknown line number>");

            CodeLocation {
                file,
                line_number,
                function,
            }
        }
    }

    extern_system_fn! {
        /// Callback from C that will handle the logging, forwarding the runtime's logs to the tracing crate.
        pub(crate) fn custom_logger(
            _params: *mut std::ffi::c_void,
            severity: sys::OrtLoggingLevel,
            category: *const i8,
            logid: *const i8,
            code_location: *const i8,
            message: *const i8,
        ) {
            let log_level = match severity {
                sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE => Level::TRACE,
                sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO => Level::DEBUG,
                sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING => Level::INFO,
                sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR => Level::WARN,
                sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL => Level::ERROR,
            };

            assert_ne!(category, std::ptr::null());
            let category = unsafe { CStr::from_ptr(category) };
            assert_ne!(code_location, std::ptr::null());
            let code_location = unsafe { CStr::from_ptr(code_location) }
                .to_str()
                .unwrap_or("unknown");
            assert_ne!(message, std::ptr::null());
            let message = unsafe { CStr::from_ptr(message) };

            assert_ne!(logid, std::ptr::null());
            let logid = unsafe { CStr::from_ptr(logid) };

            // Parse the code location
            let code_location: CodeLocation = code_location.into();

            let span = span!(
                Level::TRACE,
                "onnxruntime",
                category = category.to_str().unwrap_or("<unknown>"),
                file = code_location.file,
                line_number = code_location.line_number,
                function = code_location.function,
                logid = logid.to_str().unwrap_or("<unknown>"),
            );
            let _enter = span.enter();

            match log_level {
                Level::TRACE => trace!("{:?}", message),
                Level::DEBUG => debug!("{:?}", message),
                Level::INFO => info!("{:?}", message),
                Level::WARN => warn!("{:?}", message),
                Level::ERROR => error!("{:?}", message),
            }
        }
    }
}

/// Logging level of the ONNX Runtime C API
#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum LoggingLevel {
    /// Verbose log level
    Verbose = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE as OnnxEnumInt,
    /// Info log level
    Info = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO as OnnxEnumInt,
    /// Warning log level
    Warning = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING as OnnxEnumInt,
    /// Error log level
    Error = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR as OnnxEnumInt,
    /// Fatal log level
    Fatal = sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL as OnnxEnumInt,
}

impl From<LoggingLevel> for sys::OrtLoggingLevel {
    fn from(val: LoggingLevel) -> Self {
        match val {
            LoggingLevel::Verbose => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
            LoggingLevel::Info => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
            LoggingLevel::Warning => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            LoggingLevel::Error => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
            LoggingLevel::Fatal => sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL,
        }
    }
}

/// Optimization level performed by ONNX Runtime of the loaded graph
///
/// See the [official documentation](https://github.com/microsoft/onnxruntime/blob/main/docs/ONNX_Runtime_Graph_Optimizations.md)
/// for more information on the different optimization levels.
#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum GraphOptimizationLevel {
    /// Disable optimization
    DisableAll = sys::GraphOptimizationLevel::ORT_DISABLE_ALL as OnnxEnumInt,
    /// Basic optimization
    Basic = sys::GraphOptimizationLevel::ORT_ENABLE_BASIC as OnnxEnumInt,
    /// Extended optimization
    Extended = sys::GraphOptimizationLevel::ORT_ENABLE_EXTENDED as OnnxEnumInt,
    /// Add optimization
    All = sys::GraphOptimizationLevel::ORT_ENABLE_ALL as OnnxEnumInt,
}

impl From<GraphOptimizationLevel> for sys::GraphOptimizationLevel {
    fn from(val: GraphOptimizationLevel) -> Self {
        use GraphOptimizationLevel::{All, Basic, DisableAll, Extended};
        match val {
            DisableAll => sys::GraphOptimizationLevel::ORT_DISABLE_ALL,
            Basic => sys::GraphOptimizationLevel::ORT_ENABLE_BASIC,
            Extended => sys::GraphOptimizationLevel::ORT_ENABLE_EXTENDED,
            All => sys::GraphOptimizationLevel::ORT_ENABLE_ALL,
        }
    }
}

// FIXME: Use https://docs.rs/bindgen/0.54.1/bindgen/struct.Builder.html#method.rustified_enum
// FIXME: Add tests to cover the commented out types
/// Enum mapping ONNX Runtime's supported tensor types
#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum TensorElementDataType {
    /// 32-bit floating point, equivalent to Rust's `f32`
    Float = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT as OnnxEnumInt,
    /// Unsigned 8-bit int, equivalent to Rust's `u8`
    Uint8 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 as OnnxEnumInt,
    /// Signed 8-bit int, equivalent to Rust's `i8`
    Int8 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 as OnnxEnumInt,
    /// Unsigned 16-bit int, equivalent to Rust's `u16`
    Uint16 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 as OnnxEnumInt,
    /// Signed 16-bit int, equivalent to Rust's `i16`
    Int16 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 as OnnxEnumInt,
    /// Signed 32-bit int, equivalent to Rust's `i32`
    Int32 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 as OnnxEnumInt,
    /// Signed 64-bit int, equivalent to Rust's `i64`
    Int64 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 as OnnxEnumInt,
    /// String, equivalent to Rust's `String`
    String = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING as OnnxEnumInt,
    // /// Boolean, equivalent to Rust's `bool`
    // Bool = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL as OnnxEnumInt,
    // /// 16-bit floating point, equivalent to Rust's `f16`
    // Float16 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 as OnnxEnumInt,
    /// 64-bit floating point, equivalent to Rust's `f64`
    Double = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE as OnnxEnumInt,
    /// Unsigned 32-bit int, equivalent to Rust's `u32`
    Uint32 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 as OnnxEnumInt,
    /// Unsigned 64-bit int, equivalent to Rust's `u64`
    Uint64 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 as OnnxEnumInt,
    // /// Complex 64-bit floating point, equivalent to Rust's `???`
    // Complex64 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 as OnnxEnumInt,
    // /// Complex 128-bit floating point, equivalent to Rust's `???`
    // Complex128 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 as OnnxEnumInt,
    // /// Brain 16-bit floating point
    // Bfloat16 = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 as OnnxEnumInt,
}

impl From<TensorElementDataType> for sys::ONNXTensorElementDataType {
    fn from(val: TensorElementDataType) -> Self {
        use TensorElementDataType::{
            Double, Float, Int16, Int32, Int64, Int8, String, Uint16, Uint32, Uint64, Uint8,
        };
        match val {
            Float => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            Uint8 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
            Int8 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
            Uint16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
            Int16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
            Int32 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            Int64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            String => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
            // Bool => {
            //     sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
            // }
            // Float16 => {
            //     sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
            // }
            Double => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
            Uint32 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
            Uint64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
            // Complex64 => {
            //     sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64
            // }
            // Complex128 => {
            //     sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128
            // }
            // Bfloat16 => {
            //     sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
            // }
        }
    }
}

/// Trait used to map Rust types (for example `f32`) to ONNX types (for example `Float`)
pub trait TypeToTensorElementDataType {
    /// Return the ONNX type for a Rust type
    fn tensor_element_data_type() -> TensorElementDataType;

    /// If the type is `String`, returns `Some` with utf8 contents, else `None`.
    fn try_utf8_bytes(&self) -> Option<&[u8]>;
}

macro_rules! impl_type_trait {
    ($type_:ty, $variant:ident) => {
        impl TypeToTensorElementDataType for $type_ {
            fn tensor_element_data_type() -> TensorElementDataType {
                // unsafe { std::mem::transmute(TensorElementDataType::$variant) }
                TensorElementDataType::$variant
            }

            fn try_utf8_bytes(&self) -> Option<&[u8]> {
                None
            }
        }
    };
}

impl_type_trait!(f32, Float);
impl_type_trait!(u8, Uint8);
impl_type_trait!(i8, Int8);
impl_type_trait!(u16, Uint16);
impl_type_trait!(i16, Int16);
impl_type_trait!(i32, Int32);
impl_type_trait!(i64, Int64);
// impl_type_trait!(bool, Bool);
// impl_type_trait!(f16, Float16);
impl_type_trait!(f64, Double);
impl_type_trait!(u32, Uint32);
impl_type_trait!(u64, Uint64);
// impl_type_trait!(, Complex64);
// impl_type_trait!(, Complex128);
// impl_type_trait!(, Bfloat16);

/// Adapter for common Rust string types to Onnx strings.
///
/// It should be easy to use both `String` and `&str` as [`TensorElementDataType::String`] data, but
/// we can't define an automatic implementation for anything that implements `AsRef<str>` as it
/// would conflict with the implementations of [`TypeToTensorElementDataType`] for primitive numeric
/// types (which might implement `AsRef<str>` at some point in the future).
pub trait Utf8Data {
    /// Returns the utf8 contents.
    fn utf8_bytes(&self) -> &[u8];
}

impl Utf8Data for String {
    fn utf8_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<'a> Utf8Data for &'a str {
    fn utf8_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<T: Utf8Data> TypeToTensorElementDataType for T {
    fn tensor_element_data_type() -> TensorElementDataType {
        TensorElementDataType::String
    }

    fn try_utf8_bytes(&self) -> Option<&[u8]> {
        Some(self.utf8_bytes())
    }
}

/// Allocator type
#[derive(Debug, Clone)]
#[repr(i32)]
pub enum AllocatorType {
    // Invalid = sys::OrtAllocatorType::Invalid as i32,
    /// Device allocator
    Device = sys::OrtAllocatorType::OrtDeviceAllocator as i32,
    /// Arena allocator
    Arena = sys::OrtAllocatorType::OrtArenaAllocator as i32,
}

impl From<AllocatorType> for sys::OrtAllocatorType {
    fn from(val: AllocatorType) -> Self {
        use AllocatorType::{Arena, Device};
        match val {
            // Invalid => sys::OrtAllocatorType::Invalid,
            Device => sys::OrtAllocatorType::OrtDeviceAllocator,
            Arena => sys::OrtAllocatorType::OrtArenaAllocator,
        }
    }
}

/// Memory type
///
/// Only support ONNX's default type for now.
#[derive(Debug, Clone)]
#[repr(i32)]
pub enum MemType {
    // FIXME: C API's `OrtMemType_OrtMemTypeCPU` defines it equal to `OrtMemType_OrtMemTypeCPUOutput`. How to handle this??
    // CPUInput = sys::OrtMemType::OrtMemTypeCPUInput as i32,
    // CPUOutput = sys::OrtMemType::OrtMemTypeCPUOutput as i32,
    // CPU = sys::OrtMemType::OrtMemTypeCPU as i32,
    /// Default memory type
    Default = sys::OrtMemType::OrtMemTypeDefault as i32,
}

impl From<MemType> for sys::OrtMemType {
    fn from(val: MemType) -> Self {
        use MemType::Default;
        match val {
            // CPUInput => sys::OrtMemType::OrtMemTypeCPUInput,
            // CPUOutput => sys::OrtMemType::OrtMemTypeCPUOutput,
            // CPU => sys::OrtMemType::OrtMemTypeCPU,
            Default => sys::OrtMemType::OrtMemTypeDefault,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_char_p_to_string() {
        let s = std::ffi::CString::new("foo").unwrap();
        let ptr = s.as_c_str().as_ptr();
        assert_eq!("foo", char_p_to_string(ptr).unwrap());
    }
}
