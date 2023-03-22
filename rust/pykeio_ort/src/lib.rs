#![doc = include_str!("../README.md")]

pub mod download;
pub mod environment;
pub mod error;
pub mod execution_providers;
pub mod memory;
pub mod metadata;
pub mod session;
pub mod sys;
pub mod tensor;

use std::{
	ffi::{self, CStr},
	os::raw::c_char,
	ptr,
	sync::{atomic::AtomicPtr, Arc, Mutex}
};

pub use environment::Environment;
pub use error::{OrtApiError, OrtError, OrtResult};
pub use execution_providers::ExecutionProvider;
use lazy_static::lazy_static;
pub use session::{Session, SessionBuilder};

use self::sys::OnnxEnumInt;

macro_rules! extern_system_fn {
	($(#[$meta:meta])* fn $($tt:tt)*) => ($(#[$meta])* extern "C" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => ($(#[$meta])* $vis extern "C" fn $($tt)*);
	($(#[$meta:meta])* unsafe fn $($tt:tt)*) => ($(#[$meta])* unsafe extern "C" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => ($(#[$meta])* $vis unsafe extern "C" fn $($tt)*);
}

pub(crate) use extern_system_fn;

lazy_static! {
	pub(crate) static ref G_ORT_API: Arc<Mutex<AtomicPtr<sys::OrtApi>>> = {
		let base: *const sys::OrtApiBase = unsafe { sys::OrtGetApiBase() };
		assert_ne!(base, ptr::null());
		let get_api: extern_system_fn! { unsafe fn(u32) -> *const sys::OrtApi } = unsafe { (*base).GetApi.unwrap() };
		let api: *const sys::OrtApi = unsafe { get_api(sys::ORT_API_VERSION) };
		Arc::new(Mutex::new(AtomicPtr::new(api as *mut sys::OrtApi)))
	};
}

/// Attempts to acquire the global OrtApi object.
///
/// # Panics
///
/// Panics if another thread panicked while holding the API lock, or if the ONNX Runtime API could not be initialized.
pub fn ort() -> sys::OrtApi {
	let mut api_ref = G_ORT_API.lock().expect("failed to acquire OrtApi lock; another thread panicked?");
	let api_ref_mut: &mut *mut sys::OrtApi = api_ref.get_mut();
	let api_ptr_mut: *mut sys::OrtApi = *api_ref_mut;

	assert_ne!(api_ptr_mut, ptr::null_mut());

	unsafe { *api_ptr_mut }
}

macro_rules! ortsys {
	($method:tt) => {
		$crate::ort().$method.unwrap()
	};
	(unsafe $method:tt) => {
		unsafe { $crate::ort().$method.unwrap() }
	};
	($method:tt($($n:expr),+ $(,)?)) => {
		$crate::ort().$method.unwrap()($($n),+)
	};
	(unsafe $method:tt($($n:expr),+ $(,)?)) => {
		unsafe { $crate::ort().$method.unwrap()($($n),+) }
	};
	($method:tt($($n:expr),+ $(,)?); nonNull($($check:expr),+ $(,)?)$(;)?) => {
		$crate::ort().$method.unwrap()($($n),+);
		$($crate::error::assert_non_null_pointer($check, stringify!($method))?;)+
	};
	(unsafe $method:tt($($n:expr),+ $(,)?); nonNull($($check:expr),+ $(,)?)$(;)?) => {
		unsafe { $crate::ort().$method.unwrap()($($n),+) };
		$($crate::error::assert_non_null_pointer($check, stringify!($method))?;)+
	};
	($method:tt($($n:expr),+ $(,)?) -> $err:expr$(;)?) => {
		$crate::error::status_to_result($crate::ort().$method.unwrap()($($n),+)).map_err($err)?;
	};
	(unsafe $method:tt($($n:expr),+ $(,)?) -> $err:expr$(;)?) => {
		$crate::error::status_to_result(unsafe { $crate::ort().$method.unwrap()($($n),+) }).map_err($err)?;
	};
	($method:tt($($n:expr),+ $(,)?) -> $err:expr; nonNull($($check:expr),+ $(,)?)$(;)?) => {
		$crate::error::status_to_result($crate::ort().$method.unwrap()($($n),+)).map_err($err)?;
		$($crate::error::assert_non_null_pointer($check, stringify!($method))?;)+
	};
	(unsafe $method:tt($($n:expr),+ $(,)?) -> $err:expr; nonNull($($check:expr),+ $(,)?)$(;)?) => {
		$crate::error::status_to_result(unsafe { $crate::ort().$method.unwrap()($($n),+) }).map_err($err)?;
		$($crate::error::assert_non_null_pointer($check, stringify!($method))?;)+
	};
}

macro_rules! ortfree {
	(unsafe $allocator_ptr:expr, $ptr:tt) => {
		unsafe { (*$allocator_ptr).Free.unwrap()($allocator_ptr, $ptr as *mut std::ffi::c_void) }
	};
	($allocator_ptr:expr, $ptr:tt) => {
		(*$allocator_ptr).Free.unwrap()($allocator_ptr, $ptr as *mut std::ffi::c_void)
	};
}

pub(crate) use ortfree;
pub(crate) use ortsys;

pub(crate) fn char_p_to_string(raw: *const c_char) -> OrtResult<String> {
	let c_string = unsafe { CStr::from_ptr(raw as *mut c_char).to_owned() };
	match c_string.into_string() {
		Ok(string) => Ok(string),
		Err(e) => Err(OrtApiError::IntoStringError(e))
	}
	.map_err(OrtError::FfiStringConversion)
}

/// ONNX's logger sends the code location where the log occurred, which will be parsed into this struct.
#[derive(Debug)]
struct CodeLocation<'a> {
	file: &'a str,
	line: &'a str,
	function: &'a str
}

impl<'a> From<&'a str> for CodeLocation<'a> {
	fn from(code_location: &'a str) -> Self {
		let mut splitter = code_location.split(' ');
		let file_and_line = splitter.next().unwrap_or("<unknown file>:<unknown line>");
		let function = splitter.next().unwrap_or("<unknown function>");
		let mut file_and_line_splitter = file_and_line.split(':');
		let file = file_and_line_splitter.next().unwrap_or("<unknown file>");
		let line = file_and_line_splitter.next().unwrap_or("<unknown line>");

		CodeLocation { file, line, function }
	}
}

extern_system_fn! {
	/// Callback from C that will handle ONNX logging, forwarding ONNX's logs to the `tracing` crate.
	pub(crate) fn custom_logger(_params: *mut ffi::c_void, severity: sys::OrtLoggingLevel, category: *const c_char, log_id: *const c_char, code_location: *const c_char, message: *const c_char) {
		use tracing::{span, Level, trace, debug, warn, info, error};

		let log_level = match severity {
			sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE => Level::TRACE,
			sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO => Level::DEBUG,
			sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING => Level::INFO,
			sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR => Level::WARN,
			sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_FATAL => Level::ERROR,
			_ => Level::TRACE
		};

		assert_ne!(category, ptr::null());
		let category = unsafe { CStr::from_ptr(category) };
		assert_ne!(code_location, ptr::null());
		let code_location = unsafe { CStr::from_ptr(code_location) }.to_str().unwrap_or("unknown");
		assert_ne!(message, ptr::null());
		let message = unsafe { CStr::from_ptr(message) };
		assert_ne!(log_id, ptr::null());
		let log_id = unsafe { CStr::from_ptr(log_id) };

		let code_location = CodeLocation::from(code_location);
		let span = span!(
			Level::TRACE,
			"ort",
			category = category.to_str().unwrap_or("<unknown>"),
			file = code_location.file,
			line = code_location.line,
			function = code_location.function,
			log_id = log_id.to_str().unwrap_or("<unknown>")
		);
		let _enter = span.enter();

		match log_level {
			Level::TRACE => trace!("{:?}", message),
			Level::DEBUG => debug!("{:?}", message),
			Level::INFO => info!("{:?}", message),
			Level::WARN => warn!("{:?}", message),
			Level::ERROR => error!("{:?}", message)
		}
	}
}

/// The minimum logging level. Logs will be handled by the `tracing` crate.
#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum LoggingLevel {
	/// Verbose logging level. This will log *a lot* of messages!
	Verbose = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE as OnnxEnumInt,
	/// Info logging level.
	Info = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO as OnnxEnumInt,
	/// Warning logging level. Recommended to receive potentially important warnings.
	Warning = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING as OnnxEnumInt,
	/// Error logging level.
	Error = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR as OnnxEnumInt,
	/// Fatal logging level.
	Fatal = sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_FATAL as OnnxEnumInt
}

impl From<LoggingLevel> for sys::OrtLoggingLevel {
	fn from(logging_level: LoggingLevel) -> Self {
		match logging_level {
			LoggingLevel::Verbose => sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
			LoggingLevel::Info => sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_INFO,
			LoggingLevel::Warning => sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_WARNING,
			LoggingLevel::Error => sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_ERROR,
			LoggingLevel::Fatal => sys::OrtLoggingLevel_ORT_LOGGING_LEVEL_FATAL
		}
	}
}

/// ONNX Runtime provides various graph optimizations to improve performance. Graph optimizations are essentially
/// graph-level transformations, ranging from small graph simplifications and node eliminations to more complex node
/// fusions and layout optimizations.
///
/// Graph optimizations are divided in several categories (or levels) based on their complexity and functionality. They
/// can be performed either online or offline. In online mode, the optimizations are done before performing the
/// inference, while in offline mode, the runtime saves the optimized graph to disk (most commonly used when converting
/// an ONNX model to an ONNX Runtime model).
///
/// The optimizations belonging to one level are performed after the optimizations of the previous level have been
/// applied (e.g., extended optimizations are applied after basic optimizations have been applied).
///
/// **All optimizations (i.e. [`GraphOptimizationLevel::Level3`]) are enabled by default.**
///
/// # Online/offline mode
/// All optimizations can be performed either online or offline. In online mode, when initializing an inference session,
/// we also apply all enabled graph optimizations before performing model inference. Applying all optimizations each
/// time we initiate a session can add overhead to the model startup time (especially for complex models), which can be
/// critical in production scenarios. This is where the offline mode can bring a lot of benefit. In offline mode, after
/// performing graph optimizations, ONNX Runtime serializes the resulting model to disk. Subsequently, we can reduce
/// startup time by using the already optimized model and disabling all optimizations.
///
/// ## Notes:
/// - When running in offline mode, make sure to use the exact same options (e.g., execution providers, optimization
///   level) and hardware as the target machine that the model inference will run on (e.g., you cannot run a model
///   pre-optimized for a GPU execution provider on a machine that is equipped only with CPU).
/// - When layout optimizations are enabled, the offline mode can only be used on compatible hardware to the environment
///   when the offline model is saved. For example, if model has layout optimized for AVX2, the offline model would
///   require CPUs that support AVX2.
#[derive(Debug)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum GraphOptimizationLevel {
	/// Disables all graph optimizations.
	Disable = sys::GraphOptimizationLevel_ORT_DISABLE_ALL as OnnxEnumInt,
	/// Level 1 includes semantics-preserving graph rewrites which remove redundant nodes and redundant computation.
	/// They run before graph partitioning and thus apply to all the execution providers. Available basic/level 1 graph
	/// optimizations are as follows:
	///
	/// - Constant Folding: Statically computes parts of the graph that rely only on constant initializers. This
	///   eliminates the need to compute them during runtime.
	/// - Redundant node eliminations: Remove all redundant nodes without changing the graph structure. The following
	///   such optimizations are currently supported:
	///   * Identity Elimination
	///   * Slice Elimination
	///   * Unsqueeze Elimination
	///   * Dropout Elimination
	/// - Semantics-preserving node fusions : Fuse/fold multiple nodes into a single node. For example, Conv Add fusion
	///   folds the Add operator as the bias of the Conv operator. The following such optimizations are currently
	///   supported:
	///   * Conv Add Fusion
	///   * Conv Mul Fusion
	///   * Conv BatchNorm Fusion
	///   * Relu Clip Fusion
	///   * Reshape Fusion
	Level1 = sys::GraphOptimizationLevel_ORT_ENABLE_BASIC as OnnxEnumInt,
	#[rustfmt::skip]
	/// Level 2 optimizations include complex node fusions. They are run after graph partitioning and are only applied to
	/// the nodes assigned to the CPU or CUDA execution provider. Available extended/level 2 graph optimizations are as follows:
	///
	/// | Optimization                    | EPs       | Comments                                                                       |
	/// |:------------------------------- |:--------- |:------------------------------------------------------------------------------ |
	/// | GEMM Activation Fusion          | CPU       |                                                                                |
	/// | Matmul Add Fusion               | CPU       |                                                                                |
	/// | Conv Activation Fusion          | CPU       |                                                                                |
	/// | GELU Fusion                     | CPU, CUDA |                                                                                |
	/// | Layer Normalization Fusion      | CPU, CUDA |                                                                                |
	/// | BERT Embedding Layer Fusion     | CPU, CUDA | Fuses BERT embedding layers, layer normalization, & attention mask length      |
	/// | Attention Fusion*               | CPU, CUDA |                                                                                |
	/// | Skip Layer Normalization Fusion | CPU, CUDA | Fuse bias of fully connected layers, skip connections, and layer normalization |
	/// | Bias GELU Fusion                | CPU, CUDA | Fuse bias of fully connected layers & GELU activation                          |
	/// | GELU Approximation*             | CUDA      | Disabled by default; enable with `OrtSessionOptions::EnableGeluApproximation`  |
	///
	/// > **NOTE**: To optimize performance of the BERT model, approximation is used in GELU Approximation and Attention
	/// Fusion for the CUDA execution provider. The impact on accuracy is negligible based on our evaluation; F1 score
	/// for a BERT model on SQuAD v1.1 is almost the same (87.05 vs 87.03).
	Level2 = sys::GraphOptimizationLevel_ORT_ENABLE_EXTENDED as OnnxEnumInt,
	/// Level 3 optimizations include memory layout optimizations, which may optimize the graph to use the NCHWc memory
	/// layout rather than NCHW to improve spatial locality for some targets.
	Level3 = sys::GraphOptimizationLevel_ORT_ENABLE_ALL as OnnxEnumInt
}

impl From<GraphOptimizationLevel> for sys::GraphOptimizationLevel {
	fn from(val: GraphOptimizationLevel) -> Self {
		match val {
			GraphOptimizationLevel::Disable => sys::GraphOptimizationLevel_ORT_DISABLE_ALL,
			GraphOptimizationLevel::Level1 => sys::GraphOptimizationLevel_ORT_ENABLE_BASIC,
			GraphOptimizationLevel::Level2 => sys::GraphOptimizationLevel_ORT_ENABLE_EXTENDED,
			GraphOptimizationLevel::Level3 => sys::GraphOptimizationLevel_ORT_ENABLE_ALL
		}
	}
}

/// Execution provider allocator type.
#[derive(Debug, Clone)]
#[repr(i32)]
pub enum AllocatorType {
	/// Default device-specific allocator.
	Device = sys::OrtAllocatorType_OrtDeviceAllocator,
	/// Arena allocator.
	Arena = sys::OrtAllocatorType_OrtArenaAllocator
}

impl From<AllocatorType> for sys::OrtAllocatorType {
	fn from(val: AllocatorType) -> Self {
		match val {
			AllocatorType::Device => sys::OrtAllocatorType_OrtDeviceAllocator,
			AllocatorType::Arena => sys::OrtAllocatorType_OrtArenaAllocator
		}
	}
}

/// Memory types for allocated memory.
#[derive(Debug, Clone)]
#[repr(i32)]
pub enum MemType {
	/// Any CPU memory used by non-CPU execution provider.
	CPUInput = sys::OrtMemType_OrtMemTypeCPUInput,
	/// CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED.
	CPUOutput = sys::OrtMemType_OrtMemTypeCPUOutput,
	/// The default allocator for an execution provider.
	Default = sys::OrtMemType_OrtMemTypeDefault
}

impl MemType {
	/// Temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED.
	pub const CPU: MemType = MemType::CPUOutput;
}

impl From<MemType> for sys::OrtMemType {
	fn from(val: MemType) -> Self {
		match val {
			MemType::CPUInput => sys::OrtMemType_OrtMemTypeCPUInput,
			MemType::CPUOutput => sys::OrtMemType_OrtMemTypeCPUOutput,
			MemType::Default => sys::OrtMemType_OrtMemTypeDefault
		}
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_char_p_to_string() {
		let s = ffi::CString::new("foo").unwrap();
		let ptr = s.as_c_str().as_ptr();
		assert_eq!("foo", char_p_to_string(ptr).unwrap());
	}
}
