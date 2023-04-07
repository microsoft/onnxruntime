//! Module containing environment types

use crate::{
    error::{status_to_result, OrtError, OrtResult},
    onnxruntime::custom_logger,
    session::SessionBuilder,
    LoggingLevel,
};
use once_cell::sync::OnceCell;
use onnxruntime_sys as sys;
use onnxruntime_sys::library_filename;
use std::{
    ffi::CString,
    ptr::null_mut,
    sync::{Arc, Mutex, MutexGuard, 
        atomic::{AtomicPtr, Ordering}},
};
use sys::{onnxruntime, ORT_API_VERSION};
use tracing::{debug, warn};

static ENV: OnceCell<Arc<Mutex<_EnvironmentSingleton>>> = OnceCell::new();

static LIB: OnceCell<onnxruntime> = OnceCell::new();

static API: OnceCell<AtomicPtr<sys::OrtApi>> = OnceCell::new();

pub(crate) fn ort_api() -> sys::OrtApi {
    let atomic_ptr = API.get().expect("Environment not initialized");
    let api_ref = atomic_ptr.load(Ordering::Relaxed) as *const sys::OrtApi;
    unsafe { *api_ref }
}

#[derive(Debug)]
pub(crate) struct _EnvironmentSingleton {
    name: CString,
    pub(crate) env_ptr: *mut sys::OrtEnv,
}

unsafe impl Send for _EnvironmentSingleton {}

unsafe impl Sync for _EnvironmentSingleton {}

/// An [`Environment`](session/struct.Environment.html) is the main entry point of the ONNX Runtime.
///
/// Only one ONNXRuntime environment can be created per process. The `onnxruntime` crate
/// uses a singleton (through `lazy_static!()`) to enforce this.
///
/// Once an environment is created, a [`Session`](../session/struct.Session.html)
/// can be obtained from it.
///
/// **NOTE**: While the [`Environment`](environment/struct.Environment.html) constructor takes a `name` parameter
/// to name the environment, only the first name will be considered if many environments
/// are created.
///
/// # Example
///
/// ```no_run
/// # use std::error::Error;
/// # use std::env::var;
/// # use onnxruntime::{environment::Environment, LoggingLevel};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// # let path = var("RUST_ONNXRUNTIME_LIBRARY_PATH").ok();
///
/// let builder = Environment::builder()
///     .with_name("test")
///     .with_log_level(LoggingLevel::Warning);
///
/// let builder = if let Some(path) = path {
///     builder.with_library_path(path)
/// } else {
///     builder
/// };
/// let environment = builder.build()?;
/// # Ok(())
/// # }
/// ```
pub struct Environment {
    pub(crate) env: _Environment,
}

#[derive(Debug, Clone)]
pub(crate) struct _Environment {
    env: Arc<Mutex<_EnvironmentSingleton>>,
}

impl _Environment {
    pub(crate) fn env(&self) -> MutexGuard<_EnvironmentSingleton> {
        self.env.lock().expect("The lock is poisoned")
    }
}

impl std::fmt::Debug for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.env.fmt(f)
    }
}

impl Environment {
    /// Create a new environment builder using default values
    /// (name: `default`, log level: [`LoggingLevel::Warning`](../enum.LoggingLevel.html#variant.Warning))
    #[must_use]
    pub fn builder() -> EnvBuilder {
        EnvBuilder {
            name: "default".into(),
            log_level: LoggingLevel::Warning,
            path: None,
        }
    }

    /// Return the name of the current environment
    #[must_use]
    pub fn name(&self) -> String {
        self.env().name.to_str().unwrap().to_string()
    }

    pub(crate) fn env(&self) -> MutexGuard<_EnvironmentSingleton> {
        self.env.env()
    }

    #[tracing::instrument]
    fn new(name: &str, log_level: LoggingLevel, path: Option<String>) -> OrtResult<Environment> {
        // Load library
        let lib = if let Some(path) = path {
            LIB.get_or_try_init(|| unsafe { onnxruntime::new(path) })?
        } else {
            LIB.get_or_try_init(|| unsafe { onnxruntime::new(library_filename("onnxruntime")) })?
        };
        
        // Initialize static pointer to library's API
        API.get_or_init(|| {
            let api_ptr: *mut sys::OrtApi = unsafe {
                (*lib.OrtGetApiBase()).GetApi.unwrap()(ORT_API_VERSION) as *mut sys::OrtApi
            };
            AtomicPtr::new(api_ptr)
        });

        let env = ENV.get_or_try_init(|| {
            debug!("Environment not yet initialized, creating a new one.");

            let mut env_ptr: *mut sys::OrtEnv = std::ptr::null_mut();

            let logging_function: sys::OrtLoggingFunction = Some(custom_logger);
            // FIXME: What should go here?
            let logger_param: *mut std::ffi::c_void = std::ptr::null_mut();

            let cname = CString::new(name).unwrap();
            unsafe {
                let create_env_with_custom_logger = ort_api().CreateEnvWithCustomLogger.unwrap();
                let status = create_env_with_custom_logger(
                    logging_function,
                    logger_param,
                    log_level.into(),
                    cname.as_ptr(),
                    &mut env_ptr,
                );

                status_to_result(status).map_err(OrtError::Environment)?;
            }
            debug!(
                env_ptr = format!("{:?}", env_ptr).as_str(),
                "Environment created."
            );

            Ok::<_, OrtError>(Arc::new(Mutex::new(_EnvironmentSingleton {
                name: cname,
                env_ptr,
            })))
        })?;

        let mut guard = env.lock().expect("Lock is poisoned");

        if guard.env_ptr.is_null() {
            debug!("Environment not yet initialized, creating a new one.");

            let mut env_ptr: *mut sys::OrtEnv = std::ptr::null_mut();

            let logging_function: sys::OrtLoggingFunction = Some(custom_logger);
            // FIXME: What should go here?
            let logger_param: *mut std::ffi::c_void = std::ptr::null_mut();

            let cname = CString::new(name).unwrap();
            unsafe {
                let create_env_with_custom_logger = ort_api().CreateEnvWithCustomLogger.unwrap();
                let status = create_env_with_custom_logger(
                    logging_function,
                    logger_param,
                    log_level.into(),
                    cname.as_ptr(),
                    &mut env_ptr,
                );

                status_to_result(status).map_err(OrtError::Environment)?;
            }
            debug!(
                env_ptr = format!("{:?}", env_ptr).as_str(),
                "Environment created."
            );

            guard.env_ptr = env_ptr;
            guard.name = cname;
        }

        Ok(Environment {
            env: _Environment { env: env.clone() },
        })
    }

    /// Create a new [`SessionBuilder`](../session/struct.SessionBuilder.html)
    /// used to create a new ONNXRuntime session.
    pub fn new_session_builder(&self) -> OrtResult<SessionBuilder> {
        SessionBuilder::new(self)
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        if Arc::strong_count(ENV.get().unwrap()) == 2 {
            let env = &mut *ENV.get().unwrap().lock().expect("Lock is poisoned");

            unsafe {
                let release_env = ort_api().ReleaseEnv.unwrap();
                release_env(env.env_ptr);

                env.env_ptr = null_mut();
                env.name = CString::default();
            };
        }
    }
}

/// Struct used to build an environment [`Environment`](environment/struct.Environment.html)
///
/// This is the crate's main entry point. An environment _must_ be created
/// as the first step. An [`Environment`](environment/struct.Environment.html) can only be built
/// using `EnvBuilder` to configure it.
///
/// **NOTE**: If the same configuration method (for example [`with_name()`](struct.EnvBuilder.html#method.with_name))
/// is called multiple times, the last value will have precedence.
pub struct EnvBuilder {
    name: String,
    log_level: LoggingLevel,
    path: Option<String>,
}

impl EnvBuilder {
    /// Configure the environment with a given name
    ///
    /// **NOTE**: Since ONNXRuntime can only define one environment per process,
    /// creating multiple environments using multiple `EnvBuilder` will
    /// end up re-using the same environment internally; a new one will _not_
    /// be created. New parameters will be ignored.
    pub fn with_name<S>(mut self, name: S) -> EnvBuilder
    where
        S: Into<String>,
    {
        self.name = name.into();
        self
    }

    /// Add a library path to the Onnxruntime shared library.
    ///
    /// **Note**: The library path can be an absolute path or relative (to the executable) path.
    /// If no library path is specified, it is expected that the OS can find the Onnxruntime shared
    /// library in the normal manner to that OS.
    pub fn with_library_path<P: Into<String>>(mut self, path: P) -> EnvBuilder {
        self.path = Some(path.into());
        self
    }

    /// Configure the environment with a given log level
    ///
    /// **NOTE**: Since ONNXRuntime can only define one environment per process,
    /// creating multiple environments using multiple `EnvBuilder` will
    /// end up re-using the same environment internally; a new one will _not_
    /// be created. New parameters will be ignored.
    #[must_use]
    pub fn with_log_level(mut self, log_level: LoggingLevel) -> EnvBuilder {
        self.log_level = log_level;
        self
    }

    /// Commit the configuration to a new [`Environment`](environment/struct.Environment.html)
    pub fn build(self) -> OrtResult<Environment> {
        Environment::new(&self.name, self.log_level, self.path)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::env::var;

    use super::*;
    use test_log::test;

    pub(crate) static ONNX_RUNTIME_LIBRARY_PATH: &str = "RUST_ONNXRUNTIME_LIBRARY_PATH";

    #[test]
    fn sequential_environment_creation() {
        let first_name: String = "sequential_environment_creation".into();

        let path = var(ONNX_RUNTIME_LIBRARY_PATH).ok();

        let builder = Environment::builder()
            .with_name(first_name.clone())
            .with_log_level(LoggingLevel::Warning);

        let builder = if let Some(path) = path.clone() {
            builder.with_library_path(path)
        } else {
            builder
        };

        let env = builder.build().unwrap();

        let mut prev_env_ptr = env.env().env_ptr;

        for i in 0..10 {
            let name = format!("sequential_environment_creation: {}", i);
            let builder = Environment::builder()
                .with_name(name.clone())
                .with_log_level(LoggingLevel::Warning);

            let builder = if let Some(ref path) = path {
                builder.with_library_path(path)
            } else {
                builder
            };

            let env = builder.build().unwrap();
            let next_env_ptr = env.env().env_ptr;
            assert_eq!(next_env_ptr, prev_env_ptr);
            prev_env_ptr = next_env_ptr;
        }
    }

    #[test]
    fn concurrent_environment_creations() {
        let initial_name = "concurrent_environment_creation";

        let path = var(ONNX_RUNTIME_LIBRARY_PATH).ok();

        let main_env = Environment::new(initial_name, LoggingLevel::Warning, path.clone()).unwrap();
        let main_env_ptr = main_env.env().env_ptr as usize;

        let children: Vec<_> = (0..10)
            .map(|t| {
                let path = path.clone();

                std::thread::spawn(move || {
                    let name = format!("concurrent_environment_creation: {}", t);
                    let builder = Environment::builder()
                        .with_name(name.clone())
                        .with_log_level(LoggingLevel::Warning);

                    let builder = if let Some(path) = path {
                        builder.with_library_path(path)
                    } else {
                        builder
                    };

                    let env = builder.build().unwrap();

                    assert_eq!(env.env().env_ptr as usize, main_env_ptr);
                })
            })
            .collect();

        assert_eq!(main_env.env().env_ptr as usize, main_env_ptr);

        let res: Vec<std::thread::Result<_>> = children
            .into_iter()
            .map(std::thread::JoinHandle::join)
            .collect();
        assert!(res.into_iter().all(|r| std::result::Result::is_ok(&r)));
    }
}
