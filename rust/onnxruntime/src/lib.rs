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
//! # use std::env::var;
//! # use onnxruntime::{environment::Environment, LoggingLevel};
//! # fn main() -> Result<(), Box<dyn Error>> {
//! # let path = var("RUST_ONNXRUNTIME_LIBRARY_PATH").ok();
//!
//! let builder = Environment::builder()
//!     .with_name("test")
//!     .with_log_level(LoggingLevel::Warning);
//!
//!  let builder = if let Some(path) = path {
//!     builder.with_library_path(path)
//!  } else {
//!     builder
//!  };
//!  let environment = builder.build()?;
//!  Ok(())
//!  }
//! ```
//!
//! Then a [`Session`](session/struct.Session.html) is created from the environment, some options and an ONNX model file:
//!
//! ```no_run
//! # use std::error::Error;
//! # use std::env::var;
//! # use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel};
//! # fn main() -> Result<(), Box<dyn Error>> {
//! # let path = var("RUST_ONNXRUNTIME_LIBRARY_PATH").ok();
//! #
//! # let builder = Environment::builder()
//! #    .with_name("test")
//! #    .with_log_level(LoggingLevel::Warning);
//! #
//! # let builder = if let Some(path) = path {
//! #    builder.with_library_path(path)
//! # } else {
//! #    builder
//! # };
//! # let environment = builder.build()?;
//! let mut session = environment
//!     .new_session_builder()?
//!     .with_graph_optimization_level(GraphOptimizationLevel::Basic)?
//!     .with_intra_op_num_threads(1)?
//!     .with_model_from_file("squeezenet.onnx")?;
//! # Ok(())
//! # }
//! ```
#![cfg_attr(
    feature = "model-fetching",
    doc = r##"
Instead of loading a model from file using [`with_model_from_file()`](session/struct.SessionBuilder.html#method.with_model_from_file),
a model can be fetched directly from the [ONNX Model Zoo](https://github.com/onnx/models) using
[`with_model_downloaded()`](session/struct.SessionBuilder.html#method.with_model_downloaded) method
(requires the `model-fetching` feature).

```no_run
# use std::error::Error;
# use std::env::var;
# use onnxruntime::{environment::Environment, download::vision::ImageClassification, LoggingLevel, GraphOptimizationLevel};
# fn main() -> Result<(), Box<dyn Error>> {
# let path = var("RUST_ONNXRUNTIME_LIBRARY_PATH").ok();
#
# let builder = Environment::builder()
#    .with_name("test")
#    .with_log_level(LoggingLevel::Warning);
#
# let builder = if let Some(path) = path {
#    builder.with_library_path(path)
# } else {
#    builder
# };
# let environment = builder.build()?;

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
//! # use std::env::var;
//! # use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel, tensor::construct::ConstructTensor};
//! # fn main() -> Result<(), Box<dyn Error>> {
//! # let path = var("RUST_ONNXRUNTIME_LIBRARY_PATH").ok();
//! #
//! # let builder = Environment::builder()
//! #    .with_name("test")
//! #    .with_log_level(LoggingLevel::Warning);
//! #
//! # let builder = if let Some(path) = path {
//! #    builder.with_library_path(path)
//! # } else {
//! #    builder
//! # };
//! # let environment = builder.build()?;
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

use std::ffi::CString;

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

pub mod allocator;

pub mod download;
pub mod environment;
pub use environment::Environment;
pub(crate) use environment::ort_api;

pub mod error;
pub use error::{OrtApiError, OrtError, OrtResult};

pub mod session;
pub use session::Session;

pub mod memory_info;
pub use memory_info::MemoryInfo;

pub mod metadata;
pub use metadata::Metadata;

pub mod tensor_type_and_shape_info;
pub use tensor_type_and_shape_info::TensorTypeAndShapeInfo;

pub mod value;
pub use value::{OrtValue, MutableOrtValue, MutableOrtValueTyped, NdArrayOrtValue, AsOrtValue};

pub mod io_binding;
pub use io_binding::IoBinding;

// Re-export
use sys::OnnxEnumInt;

// Re-export ndarray as it's part of the public API anyway
pub use ndarray;

fn char_ptr_to_string(raw: *const i8) -> OrtResult<String> {
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
#[derive(Debug, Clone, Copy)]
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

#[derive(Clone, Debug)]
/// DeviceName for MemoryInfo location
pub enum DeviceName {
    /// Cpu
    Cpu,
    /// Cuda
    Cuda,
    /// CudaPinned
    CudaPinned,
    /// Cann
    Cann,
    /// CannPinned
    CannPinned,
    /// Dml
    Dml,
    /// Hip
    Hip,
    /// HipPinned
    HipPinned,
    /// OpenVinoCpu
    OpenVinoCpu,
    /// OpenVinoGpu
    OpenVinoGpu,
}

impl From<DeviceName> for CString {
    fn from(val: DeviceName) -> Self {
        match val {
            DeviceName::Cpu => CString::new("Cpu").unwrap(),
            DeviceName::Cuda => CString::new("Cuda").unwrap(),
            DeviceName::CudaPinned => CString::new("CudaPinned").unwrap(),
            DeviceName::Cann => CString::new("Cann").unwrap(),
            DeviceName::CannPinned => CString::new("CannPinned").unwrap(),
            DeviceName::Dml => CString::new("DML").unwrap(),
            DeviceName::Hip => CString::new("Hip").unwrap(),
            DeviceName::HipPinned => CString::new("HipPinned").unwrap(),
            DeviceName::OpenVinoCpu => CString::new("OpenVINO_CPU").unwrap(),
            DeviceName::OpenVinoGpu => CString::new("OpenVINO_GPU").unwrap(),
        }
    }
}

impl From<&str> for DeviceName {
    fn from(val: &str) -> Self {
        match val {
            "Cpu" => DeviceName::Cpu,
            // not sure why this value exists
            "CUDA_CPU" => DeviceName::Cpu,
            "Cuda" => DeviceName::Cuda,
            "CudaPinned" => DeviceName::CudaPinned,
            "Cann" => DeviceName::Cuda,
            "CannPinned" => DeviceName::CudaPinned,
            "Dml" => DeviceName::Dml,
            "Hip" => DeviceName::Hip,
            "HipPinned" => DeviceName::HipPinned,
            "OpenVINO_CPU" => DeviceName::OpenVinoCpu,
            "OpenVINO_GPU" => DeviceName::OpenVinoGpu,
            other => unimplemented!("{other:?} not implemented"),
        }
    }
}

/// Enum mapping ONNX Runtime's supported tensor types
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorElementDataType {
    /// 32-bit floating point, equivalent to Rust's `f32`
    Float,
    /// Unsigned 8-bit int, equivalent to Rust's `u8`
    Uint8,
    /// Signed 8-bit int, equivalent to Rust's `i8`
    Int8,
    /// Unsigned 16-bit int, equivalent to Rust's `u16`
    Uint16,
    /// Signed 16-bit int, equivalent to Rust's `i16`
    Int16,
    /// Signed 32-bit int, equivalent to Rust's `i32`
    Int32,
    /// Signed 64-bit int, equivalent to Rust's `i64`
    Int64,
    /// String, equivalent to Rust's `String`
    String,
    /// Boolean, equivalent to Rust's `bool`
    Bool,
    /// 16-bit floating point, equivalent to Rust's `f16`
    Float16,
    /// 64-bit floating point, equivalent to Rust's `f64`
    Double,
    /// Unsigned 32-bit int, equivalent to Rust's `u32`
    Uint32,
    /// Unsigned 64-bit int, equivalent to Rust's `u64`
    Uint64,
    /// Undefined
    Undefined,
}

impl From<TensorElementDataType> for sys::ONNXTensorElementDataType {
    fn from(val: TensorElementDataType) -> Self {
        use TensorElementDataType::*;
        match val {
            Float => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            Uint8 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
            Int8 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
            Uint16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
            Int16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
            Int32 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            Int64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            String => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
            Bool => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
            Float16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
            Double => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
            Uint32 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
            Uint64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
            Undefined => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
        }
    }
}

impl From<sys::ONNXTensorElementDataType> for TensorElementDataType {
    fn from(val: sys::ONNXTensorElementDataType) -> Self {
        match val {
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED => {
                TensorElementDataType::Undefined
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
                TensorElementDataType::Float
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => {
                TensorElementDataType::Uint8
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => {
                TensorElementDataType::Int8
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => {
                TensorElementDataType::Uint16
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => {
                TensorElementDataType::Int16
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
                TensorElementDataType::Int32
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
                TensorElementDataType::Int64
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => {
                TensorElementDataType::String
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => {
                TensorElementDataType::Bool
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => {
                TensorElementDataType::Float16
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => {
                TensorElementDataType::Double
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => {
                TensorElementDataType::Uint32
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => {
                TensorElementDataType::Uint64
            }
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 => todo!(),
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 => todo!(),
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => todo!(),
        }
    }
}


/// Trait used to map Rust types (for example `f32`) to ONNX types (for example `Float`)
pub trait TypeToTensorElementDataType {
    /// Return the ONNX type for a Rust type
    fn tensor_element_data_type() -> TensorElementDataType;

    /// If the type is `String`, returns `Some` with utf8 contents,else `None`.
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

//impl_type_trait!(f16, Float16);   // Requires half crate
impl_type_trait!(f32, Float);
impl_type_trait!(f64, Double);
impl_type_trait!(i8, Int8);
impl_type_trait!(i16, Int16);
impl_type_trait!(i32, Int32);
impl_type_trait!(i64, Int64);
impl_type_trait!(u8, Uint8);
impl_type_trait!(u16, Uint16);
impl_type_trait!(u32, Uint32);
impl_type_trait!(u64, Uint64);


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

/// Memory type
///
/// Only support ONNX's default type for now.
#[derive(Debug, Clone)]
#[repr(i32)]
pub enum MemType {
    /// CPUInput
    CPUInput,
    /// CPUOutput
    CPUOutput,
    /// Default
    Default,
}

impl From<MemType> for sys::OrtMemType {
    fn from(val: MemType) -> Self {
        match val {
            MemType::CPUInput => sys::OrtMemType::OrtMemTypeCPUInput,
            MemType::CPUOutput => sys::OrtMemType::OrtMemTypeCPUOutput,
            MemType::Default => sys::OrtMemType::OrtMemTypeDefault,
        }
    }
}

impl From<sys::OrtMemType> for MemType {
    fn from(val: sys::OrtMemType) -> Self {
        match val {
            sys::OrtMemType::OrtMemTypeCPUInput => MemType::CPUInput,
            sys::OrtMemType::OrtMemTypeCPUOutput => MemType::CPUOutput,
            sys::OrtMemType::OrtMemTypeDefault => MemType::Default,
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
        assert_eq!("foo", char_ptr_to_string(ptr).unwrap());
    }
}
