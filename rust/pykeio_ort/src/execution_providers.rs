#![allow(unused_imports)]

use std::{collections::HashMap, ffi::CString, os::raw::c_char};

use crate::{error::status_to_result, ortsys, sys, OrtApiError, OrtResult};

extern "C" {
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_CPU(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr;
	#[cfg(feature = "acl")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_ACL(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr;
	#[cfg(feature = "onednn")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_Dnnl(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr;
	#[cfg(feature = "coreml")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_CoreML(options: *mut sys::OrtSessionOptions, flags: u32) -> sys::OrtStatusPtr;
	#[cfg(feature = "directml")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_DML(options: *mut sys::OrtSessionOptions, device_id: std::os::raw::c_int) -> sys::OrtStatusPtr;
	#[cfg(feature = "rocm")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_ROCm(options: *mut sys::OrtSessionOptions, device_id: std::os::raw::c_int) -> sys::OrtStatusPtr;
}

/// Execution provider container. See [the ONNX Runtime docs](https://onnxruntime.ai/docs/execution-providers/) for more
/// info on execution providers. Execution providers are actually registered via the `with_execution_providers()`
/// functions [`crate::SessionBuilder`] (per-session) or [`crate::EnvBuilder`] (default for all sessions in an
/// environment).
#[derive(Debug, Clone)]
pub struct ExecutionProvider {
	provider: String,
	options: HashMap<String, String>
}

macro_rules! ep_providers {
	($($fn_name:ident = $name:expr),*) => {
		$(
			/// Creates a new `
			#[doc = $name]
			#[doc = "` configuration object."]
			pub fn $fn_name() -> Self {
				Self::new($name)
			}
		)*
	}
}

macro_rules! ep_options {
	($(
		$(#[$meta:meta])*
		pub fn $fn_name:ident($opt_type:ty) = $option_name:ident;
	)*) => {
		$(
			$(#[$meta])*
			pub fn $fn_name(mut self, v: $opt_type) -> Self {
				self = self.with(stringify!($option_name), v.to_string());
				self
			}
		)*
	}
}

impl ExecutionProvider {
	/// Creates an `ExecutionProvider` for the given execution provider name.
	///
	/// You probably want the dedicated methods instead, e.g. [`ExecutionProvider::cuda`].
	pub fn new(provider: impl Into<String>) -> Self {
		Self {
			provider: provider.into(),
			options: HashMap::new()
		}
	}

	ep_providers! {
		cpu = "CPUExecutionProvider",
		cuda = "CUDAExecutionProvider",
		tensorrt = "TensorRTExecutionProvider",
		acl = "AclExecutionProvider",
		dnnl = "DnnlExecutionProvider",
		onednn = "DnnlExecutionProvider",
		coreml = "CoreMLExecutionProvider",
		directml = "DmlExecutionProvider",
		rocm = "ROCmExecutionProvider"
	}

	/// Returns `true` if this execution provider is available, `false` otherwise.
	/// The CPU execution provider will always be available.
	pub fn is_available(&self) -> bool {
		let mut providers: *mut *mut c_char = std::ptr::null_mut();
		let mut num_providers = 0;
		if status_to_result(ortsys![unsafe GetAvailableProviders(&mut providers, &mut num_providers)]).is_err() {
			return false;
		}

		for i in 0..num_providers {
			let avail = unsafe { std::ffi::CStr::from_ptr(*providers.offset(i as isize)) }
				.to_string_lossy()
				.into_owned();
			if self.provider == avail {
				let _ = ortsys![unsafe ReleaseAvailableProviders(providers, num_providers)];
				return true;
			}
		}

		let _ = ortsys![unsafe ReleaseAvailableProviders(providers, num_providers)];
		false
	}

	/// Configure this execution provider with the given option name and value
	pub fn with(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
		self.options.insert(name.into(), value.into());
		self
	}

	ep_options! {
		/// Whether or not to use the arena allocator.
		///
		/// Supported backends: CPU, ACL, oneDNN
		pub fn with_use_arena(bool) = use_arena;
		/// The device ID to initialize the execution provider on.
		///
		/// Supported backends: DirectML
		pub fn with_device_id(i32) = device_id;
	}
}

#[tracing::instrument(skip_all)]
pub(crate) fn apply_execution_providers(options: *mut sys::OrtSessionOptions, execution_providers: impl AsRef<[ExecutionProvider]>) {
	let status_to_result_and_log = |ep: &'static str, status: *mut sys::OrtStatus| {
		let result = status_to_result(status);
		tracing::debug!("{ep} execution provider registration {result:?}");
		result
	};
	for ep in execution_providers.as_ref() {
		let init_args = ep.options.clone();
		match ep.provider.as_str() {
			"CPUExecutionProvider" => {
				let use_arena = init_args.get("use_arena").map_or(false, |s| s.parse::<bool>().unwrap_or(false));
				let status = unsafe { OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena.into()) };
				if status_to_result_and_log("CPU", status).is_ok() {
					return; // EP found
				}
			}
			#[cfg(feature = "cuda")]
			"CUDAExecutionProvider" => {
				let mut cuda_options: *mut sys::OrtCUDAProviderOptionsV2 = std::ptr::null_mut();
				if status_to_result_and_log("CUDA", ortsys![unsafe CreateCUDAProviderOptions(&mut cuda_options)]).is_err() {
					continue; // next EP
				}
				let keys: Vec<CString> = init_args.keys().map(|k| CString::new(k.as_str()).unwrap()).collect();
				let values: Vec<CString> = init_args.values().map(|v| CString::new(v.as_str()).unwrap()).collect();
				assert_eq!(keys.len(), values.len()); // sanity check
				let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
				let value_ptrs: Vec<*const c_char> = values.iter().map(|v| v.as_ptr()).collect();
				let status = ortsys![unsafe UpdateCUDAProviderOptions(cuda_options, key_ptrs.as_ptr(), value_ptrs.as_ptr(), keys.len())];
				if status_to_result_and_log("CUDA", status).is_err() {
					ortsys![unsafe ReleaseCUDAProviderOptions(cuda_options)];
					continue; // next EP
				}
				let status = ortsys![unsafe SessionOptionsAppendExecutionProvider_CUDA_V2(options, cuda_options)];
				ortsys![unsafe ReleaseCUDAProviderOptions(cuda_options)];
				if status_to_result_and_log("CUDA", status).is_ok() {
					return; // EP found
				}
			}
			#[cfg(feature = "tensorrt")]
			"TensorRTExecutionProvider" => {
				let mut tensorrt_options: *mut sys::OrtTensorRTProviderOptionsV2 = std::ptr::null_mut();
				if status_to_result_and_log("TensorRT", ortsys![unsafe CreateTensorRTProviderOptions(&mut tensorrt_options)]).is_err() {
					continue; // next EP
				}
				let keys: Vec<CString> = init_args.keys().map(|k| CString::new(k.as_str()).unwrap()).collect();
				let values: Vec<CString> = init_args.values().map(|v| CString::new(v.as_str()).unwrap()).collect();
				assert_eq!(keys.len(), values.len()); // sanity check
				let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
				let value_ptrs: Vec<*const c_char> = values.iter().map(|v| v.as_ptr()).collect();
				let status = ortsys![unsafe UpdateTensorRTProviderOptions(tensorrt_options, key_ptrs.as_ptr(), value_ptrs.as_ptr(), keys.len())];
				if status_to_result_and_log("TensorRT", status).is_err() {
					ortsys![unsafe ReleaseTensorRTProviderOptions(tensorrt_options)];
					continue; // next EP
				}
				let status = ortsys![unsafe SessionOptionsAppendExecutionProvider_TensorRT_V2(options, tensorrt_options)];
				ortsys![unsafe ReleaseTensorRTProviderOptions(tensorrt_options)];
				if status_to_result_and_log("TensorRT", status).is_ok() {
					return; // EP found
				}
			}
			#[cfg(feature = "acl")]
			"AclExecutionProvider" => {
				let use_arena = init_args.get("use_arena").map_or(false, |s| s.parse::<bool>().unwrap_or(false));
				let status = unsafe { OrtSessionOptionsAppendExecutionProvider_ACL(options, use_arena.into()) };
				if status_to_result_and_log("ACL", status).is_ok() {
					return; // EP found
				}
			}
			#[cfg(feature = "onednn")]
			"DnnlExecutionProvider" => {
				let use_arena = init_args.get("use_arena").map_or(false, |s| s.parse::<bool>().unwrap_or(false));
				let status = unsafe { OrtSessionOptionsAppendExecutionProvider_Dnnl(options, use_arena.into()) };
				if status_to_result_and_log("oneDNN", status).is_ok() {
					return; // EP found
				}
			}
			#[cfg(feature = "coreml")]
			"CoreMLExecutionProvider" => {
				// TODO: Support additional CoreML flags
				// https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html#available-options
				let status = unsafe { OrtSessionOptionsAppendExecutionProvider_CoreML(options, 0) };
				if status_to_result_and_log("CoreML", status).is_ok() {
					return; // EP found
				}
			}
			#[cfg(feature = "directml")]
			"DmlExecutionProvider" => {
				let device_id = init_args.get("device_id").map_or(0, |s| s.parse::<i32>().unwrap_or(0));
				// TODO: extended options with OrtSessionOptionsAppendExecutionProviderEx_DML
				let status = unsafe { OrtSessionOptionsAppendExecutionProvider_DML(options, device_id) };
				if status_to_result_and_log("DirectML", status).is_ok() {
					return; // EP found
				}
			}
			#[cfg(feature = "rocm")]
			"ROCmExecutionProvider" => {
				let device_id = init_args.get("device_id").map_or(0, |s| s.parse::<i32>().unwrap_or(0));
				let status = unsafe { OrtSessionOptionsAppendExecutionProvider_ROCm(options, device_id) };
				if status_to_result_and_log("DirectML", status).is_ok() {
					return; // EP found
				}
			}
			_ => {}
		};
	}
}
