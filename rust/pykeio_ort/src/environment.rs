use std::{
	ffi::CString,
	sync::{atomic::AtomicPtr, Arc, Mutex}
};

use lazy_static::lazy_static;
use tracing::{debug, error, warn};

use super::{
	custom_logger,
	error::{status_to_result, OrtError, OrtResult},
	ort, ortsys, sys, ExecutionProvider, LoggingLevel
};

lazy_static! {
	static ref G_ENV: Arc<Mutex<EnvironmentSingleton>> = Arc::new(Mutex::new(EnvironmentSingleton {
		name: String::from("uninitialized"),
		env_ptr: AtomicPtr::new(std::ptr::null_mut())
	}));
}

#[derive(Debug)]
struct EnvironmentSingleton {
	name: String,
	env_ptr: AtomicPtr<sys::OrtEnv>
}

/// An [`Environment`] is the main entry point of the ONNX Runtime.
///
/// Only one ONNX environment can be created per process. A singleton (through `lazy_static!()`) is used to enforce
/// this.
///
/// Once an environment is created, a [`super::Session`] can be obtained from it.
///
/// **NOTE**: While the [`Environment`] constructor takes a `name` parameter to name the environment, only the first
/// name will be considered if many environments are created.
///
/// # Example
///
/// ```no_run
/// # use std::error::Error;
/// # use ort::{Environment, LoggingLevel};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let environment = Environment::builder().with_name("test").with_log_level(LoggingLevel::Verbose).build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Environment {
	env: Arc<Mutex<EnvironmentSingleton>>,
	pub(crate) execution_providers: Vec<ExecutionProvider>
}

impl Environment {
	/// Create a new environment builder using default values
	/// (name: `default`, log level: [`LoggingLevel::Warning`])
	pub fn builder() -> EnvBuilder {
		EnvBuilder {
			name: "default".into(),
			log_level: LoggingLevel::Warning,
			execution_providers: Vec::new()
		}
	}

	/// Return the name of the current environment
	pub fn name(&self) -> String {
		self.env.lock().unwrap().name.to_string()
	}

	/// Wraps this environment in an `Arc` for use with `SessionBuilder`.
	pub fn into_arc(self) -> Arc<Environment> {
		Arc::new(self)
	}

	pub(crate) fn env_ptr(&self) -> *const sys::OrtEnv {
		*self.env.lock().unwrap().env_ptr.get_mut()
	}

	fn new(name: String, log_level: LoggingLevel, execution_providers: Vec<ExecutionProvider>) -> OrtResult<Environment> {
		// NOTE: Because 'G_ENV' is a lazy_static, locking it will, initially, create
		//      a new Arc<Mutex<EnvironmentSingleton>> with a strong count of 1.
		//      Cloning it to embed it inside the 'Environment' to return
		//      will thus increase the strong count to 2.
		let mut environment_guard = G_ENV.lock().expect("Failed to acquire lock: another thread panicked?");
		let g_env_ptr = environment_guard.env_ptr.get_mut();
		if g_env_ptr.is_null() {
			debug!("Environment not yet initialized, creating a new one.");

			let mut env_ptr: *mut sys::OrtEnv = std::ptr::null_mut();

			let logging_function: sys::OrtLoggingFunction = Some(custom_logger);
			// FIXME: What should go here?
			let logger_param: *mut std::ffi::c_void = std::ptr::null_mut();

			let cname = CString::new(name.clone()).unwrap();

			let create_env_with_custom_logger = ortsys![CreateEnvWithCustomLogger];
			let status = unsafe { create_env_with_custom_logger(logging_function, logger_param, log_level.into(), cname.as_ptr(), &mut env_ptr) };
			status_to_result(status).map_err(OrtError::CreateEnvironment)?;

			debug!(env_ptr = format!("{:?}", env_ptr).as_str(), "Environment created.");

			*g_env_ptr = env_ptr;
			environment_guard.name = name;

			// NOTE: Cloning the lazy_static 'G_ENV' will increase its strong count by one.
			//       If this 'Environment' is the only one in the process, the strong count
			//       will be 2:
			//          * one lazy_static 'G_ENV'
			//          * one inside the 'Environment' returned
			Ok(Environment {
				env: G_ENV.clone(),
				execution_providers
			})
		} else {
			warn!(
				name = environment_guard.name.as_str(),
				env_ptr = format!("{:?}", environment_guard.env_ptr).as_str(),
				"Environment already initialized, reusing it.",
			);

			// NOTE: Cloning the lazy_static 'G_ENV' will increase its strong count by one.
			//       If this 'Environment' is the only one in the process, the strong count
			//       will be 2:
			//          * one lazy_static 'G_ENV'
			//          * one inside the 'Environment' returned
			Ok(Environment {
				env: G_ENV.clone(),
				execution_providers
			})
		}
	}
}

impl Default for Environment {
	fn default() -> Self {
		// NOTE: Because 'G_ENV' is a lazy_static, locking it will, initially, create
		//      a new Arc<Mutex<EnvironmentSingleton>> with a strong count of 1.
		//      Cloning it to embed it inside the 'Environment' to return
		//      will thus increase the strong count to 2.
		let mut environment_guard = G_ENV.lock().expect("Failed to acquire lock: another thread panicked?");
		let g_env_ptr = environment_guard.env_ptr.get_mut();
		if g_env_ptr.is_null() {
			debug!("Environment not yet initialized, creating a new one.");

			let mut env_ptr: *mut sys::OrtEnv = std::ptr::null_mut();

			let logging_function: sys::OrtLoggingFunction = Some(custom_logger);
			// FIXME: What should go here?
			let logger_param: *mut std::ffi::c_void = std::ptr::null_mut();

			let cname = CString::new("default".to_string()).unwrap();

			let create_env_with_custom_logger = ortsys![CreateEnvWithCustomLogger];
			let status = unsafe { create_env_with_custom_logger(logging_function, logger_param, LoggingLevel::Warning.into(), cname.as_ptr(), &mut env_ptr) };
			status_to_result(status).map_err(OrtError::CreateEnvironment).unwrap();

			debug!(env_ptr = format!("{:?}", env_ptr).as_str(), "Environment created.");

			*g_env_ptr = env_ptr;
			environment_guard.name = "default".to_string();

			// NOTE: Cloning the lazy_static 'G_ENV' will increase its strong count by one.
			//       If this 'Environment' is the only one in the process, the strong count
			//       will be 2:
			//          * one lazy_static 'G_ENV'
			//          * one inside the 'Environment' returned
			Environment {
				env: G_ENV.clone(),
				execution_providers: vec![]
			}
		} else {
			// NOTE: Cloning the lazy_static 'G_ENV' will increase its strong count by one.
			//       If this 'Environment' is the only one in the process, the strong count
			//       will be 2:
			//          * one lazy_static 'G_ENV'
			//          * one inside the 'Environment' returned
			Environment {
				env: G_ENV.clone(),
				execution_providers: vec![]
			}
		}
	}
}

impl Drop for Environment {
	#[tracing::instrument]
	fn drop(&mut self) {
		debug!(global_arc_count = Arc::strong_count(&G_ENV), "Dropping the Environment.",);

		let mut environment_guard = self.env.lock().expect("Failed to acquire lock: another thread panicked?");

		// NOTE: If we drop an 'Environment' we (obviously) have _at least_
		//       one 'G_ENV' strong count (the one in the 'env' member).
		//       There is also the "original" 'G_ENV' which is a the lazy_static global.
		//       If there is no other environment, the strong count should be two and we
		//       can properly free the sys::OrtEnv pointer.
		if Arc::strong_count(&G_ENV) == 2 {
			let release_env = ort().ReleaseEnv.unwrap();
			let env_ptr: *mut sys::OrtEnv = *environment_guard.env_ptr.get_mut();

			debug!(global_arc_count = Arc::strong_count(&G_ENV), "Releasing the Environment.",);

			assert_ne!(env_ptr, std::ptr::null_mut());
			if env_ptr.is_null() {
				error!("Environment pointer is null, not dropping!");
			} else {
				unsafe { release_env(env_ptr) };
			}

			environment_guard.env_ptr = AtomicPtr::new(std::ptr::null_mut());
			environment_guard.name = String::from("uninitialized");
		}
	}
}

/// Struct used to build an environment [`Environment`].
///
/// This is ONNX Runtime's main entry point. An environment _must_ be created as the first step. An [`Environment`] can
/// only be built using `EnvBuilder` to configure it.
///
/// Libraries using `ort` should **not** create an environment, as only one is allowed per process. Instead, allow the
/// user to pass their own environment to the library.
///
/// **NOTE**: If the same configuration method (for example [`EnvBuilder::with_name()`] is called multiple times, the
/// last value will have precedence.
pub struct EnvBuilder {
	name: String,
	log_level: LoggingLevel,
	execution_providers: Vec<ExecutionProvider>
}

impl EnvBuilder {
	/// Configure the environment with a given name
	///
	/// **NOTE**: Since ONNX can only define one environment per process, creating multiple environments using multiple
	/// [`EnvBuilder`]s will end up re-using the same environment internally; a new one will _not_ be created. New
	/// parameters will be ignored.
	pub fn with_name<S>(mut self, name: S) -> EnvBuilder
	where
		S: Into<String>
	{
		self.name = name.into();
		self
	}

	/// Configure the environment with a given log level
	///
	/// **NOTE**: Since ONNX can only define one environment per process, creating multiple environments using multiple
	/// [`EnvBuilder`]s will end up re-using the same environment internally; a new one will _not_ be created. New
	/// parameters will be ignored.
	pub fn with_log_level(mut self, log_level: LoggingLevel) -> EnvBuilder {
		self.log_level = log_level;
		self
	}

	/// Configures a list of execution providers sessions created under this environment will use by default. Sessions
	/// may override these via [`SessionBuilder::with_execution_providers()`].
	///
	/// Execution providers are loaded in the order they are provided until a suitable execution provider is found. Most
	/// execution providers will silently fail if they are unavailable or misconfigured (see notes below), however, some
	/// may log to the console, which is sadly unavoidable. The CPU execution provider is always available, so always
	/// put it last in the list (though it is not required).
	///
	/// Execution providers will only work if the corresponding `onnxep-*` feature is enabled and ONNX Runtime was built
	/// with support for the corresponding execution provider. Execution providers that do not have their corresponding
	/// feature enabled are currently ignored.
	///
	/// Execution provider options can be specified in the second argument. Refer to ONNX Runtime's
	/// [execution provider docs](https://onnxruntime.ai/docs/execution-providers/) for configuration options. In most
	/// cases, passing `None` to configure with no options is suitable.
	///
	/// It is recommended to enable the `cuda` EP for x86 platforms and the `acl` EP for ARM platforms for the best
	/// performance, though this does mean you'll have to build ONNX Runtime for these targets. Microsoft's prebuilt
	/// binaries are built with CUDA and TensorRT support, if you built `ort` with the `onnxep-cuda` or
	/// `onnxep-tensorrt` features enabled.
	///
	/// Supported execution providers:
	/// - `cpu`: Default CPU/MLAS execution provider. Available on all platforms.
	/// - `acl`: Arm Compute Library
	/// - `cuda`: NVIDIA CUDA/cuDNN
	/// - `tensorrt`: NVIDIA TensorRT
	///
	/// ## Notes
	///
	/// - Using the CUDA/TensorRT execution providers **can terminate the process if the CUDA/TensorRT installation is
	///   misconfigured**. Configuring the execution provider will seem to work, but when you attempt to run a session,
	///   it will hard crash the process with a "stack buffer overrun" error. This can occur when CUDA/TensorRT is
	///   missing a DLL such as `zlibwapi.dll`. To prevent your app from crashing, you can check to see if you can load
	///   `zlibwapi.dll` before enabling the CUDA/TensorRT execution providers.
	pub fn with_execution_providers(mut self, execution_providers: impl AsRef<[ExecutionProvider]>) -> EnvBuilder {
		self.execution_providers = execution_providers.as_ref().to_vec();
		self
	}

	/// Commit the configuration to a new [`Environment`].
	pub fn build(self) -> OrtResult<Environment> {
		Environment::new(self.name, self.log_level, self.execution_providers)
	}
}

#[cfg(test)]
mod tests {
	use std::sync::{RwLock, RwLockWriteGuard};

	use test_log::test;

	use super::*;

	impl G_ENV {
		fn is_initialized(&self) -> bool {
			Arc::strong_count(self) >= 2
		}

		fn env_ptr(&self) -> *const sys::OrtEnv {
			*self.lock().unwrap().env_ptr.get_mut()
		}
	}

	struct ConcurrentTestRun {
		lock: Arc<RwLock<()>>
	}

	lazy_static! {
		static ref CONCURRENT_TEST_RUN: ConcurrentTestRun = ConcurrentTestRun { lock: Arc::new(RwLock::new(())) };
	}

	impl CONCURRENT_TEST_RUN {
		fn single_test_run(&self) -> RwLockWriteGuard<()> {
			self.lock.write().unwrap()
		}
	}

	#[test]
	fn env_is_initialized() {
		let _run_lock = CONCURRENT_TEST_RUN.single_test_run();

		assert!(!G_ENV.is_initialized());
		assert_eq!(G_ENV.env_ptr(), std::ptr::null_mut());

		let env = Environment::builder()
			.with_name("env_is_initialized")
			.with_log_level(LoggingLevel::Warning)
			.build()
			.unwrap();
		assert!(G_ENV.is_initialized());
		assert_ne!(G_ENV.env_ptr(), std::ptr::null_mut());

		std::mem::drop(env);
		assert!(!G_ENV.is_initialized());
		assert_eq!(G_ENV.env_ptr(), std::ptr::null_mut());
	}

	#[ignore]
	#[test]
	fn sequential_environment_creation() {
		let _concurrent_run_lock_guard = CONCURRENT_TEST_RUN.single_test_run();

		let mut prev_env_ptr = G_ENV.env_ptr();

		for i in 0..10 {
			let name = format!("sequential_environment_creation: {}", i);
			let env = Environment::builder()
				.with_name(name.clone())
				.with_log_level(LoggingLevel::Warning)
				.build()
				.unwrap();
			let next_env_ptr = G_ENV.env_ptr();
			assert_ne!(next_env_ptr, prev_env_ptr);
			prev_env_ptr = next_env_ptr;

			assert_eq!(env.name(), name);
		}
	}

	#[test]
	fn concurrent_environment_creations() {
		let _concurrent_run_lock_guard = CONCURRENT_TEST_RUN.single_test_run();

		let initial_name = String::from("concurrent_environment_creation");
		let main_env = Environment::new(initial_name.clone(), LoggingLevel::Warning, Vec::new()).unwrap();
		let main_env_ptr = main_env.env_ptr() as usize;

		assert_eq!(main_env.name(), initial_name);
		assert_eq!(main_env.env_ptr() as usize, main_env_ptr);

		assert!(
			(0..10)
				.map(|t| {
					let initial_name_cloned = initial_name.clone();
					std::thread::spawn(move || {
						let name = format!("concurrent_environment_creation: {}", t);
						let env = Environment::builder()
							.with_name(name)
							.with_log_level(LoggingLevel::Warning)
							.build()
							.unwrap();

						assert_eq!(env.name(), initial_name_cloned);
						assert_eq!(env.env_ptr() as usize, main_env_ptr);
					})
				})
				.map(|child| child.join())
				.all(|r| std::result::Result::is_ok(&r))
		);
	}
}
