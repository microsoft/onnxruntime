//! Module containing session types

use std::{
    convert::TryFrom,
    ffi::{CStr, CString},
    fmt::Debug,
    os::raw::{c_char, c_void},
    panic::{catch_unwind, AssertUnwindSafe},
    path::Path,
    ptr,
    sync::Arc,
};

#[cfg(not(target_family = "windows"))]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_family = "windows")]
use std::os::windows::ffi::OsStrExt;

#[cfg(feature = "model-fetching")]
use std::env;

use crate::{
    char_p_to_string,
    environment::{_Environment, Environment},
    error::{
        assert_not_null_pointer, assert_null_pointer, status_to_result, status_to_result_with_api,
        EpContextDataCallbackError, NonMatchingDimensionsError, OrtApiError, OrtError, Result,
    },
    memory::MemoryInfo,
    tensor::{
        construct::ConstructTensor,
        ort_output_tensor::{OrtOutput, OrtOwnedTensorExtractor},
        OrtOutputTensor,
    },
    AllocatorType, GraphOptimizationLevel, MemType, TensorElementDataType,
};
use once_cell::sync::OnceCell;
use onnxruntime_sys as sys;

use tracing::{debug, error};

#[cfg(feature = "model-fetching")]
use crate::{download::AvailableOnnxModel, error::OrtDownloadError};

type CreateStatusFn = extern_system_fn! {
    unsafe fn(sys::OrtErrorCode, *const c_char) -> *mut sys::OrtStatus
};
type ReleaseSessionFn = extern_system_fn! {
    unsafe fn(*mut sys::OrtSession)
};
type ReleaseEpContextDataReadOptionsFn = extern_system_fn! {
    unsafe fn(*mut sys::OrtEpContextDataReadOptions)
};

struct SessionPointerGuard {
    session_ptr: *mut sys::OrtSession,
    release_session: ReleaseSessionFn,
}

impl SessionPointerGuard {
    fn into_raw(mut self) -> *mut sys::OrtSession {
        let session_ptr = self.session_ptr;
        self.session_ptr = ptr::null_mut();
        session_ptr
    }
}

impl Drop for SessionPointerGuard {
    fn drop(&mut self) {
        if !self.session_ptr.is_null() {
            unsafe { (self.release_session)(self.session_ptr) };
        }
    }
}

struct EpContextDataReadOptionsGuard {
    read_options: *mut sys::OrtEpContextDataReadOptions,
    release_options: ReleaseEpContextDataReadOptionsFn,
}

impl Drop for EpContextDataReadOptionsGuard {
    fn drop(&mut self) {
        unsafe { (self.release_options)(self.read_options) };
    }
}

struct EpContextDataReadRegistration {
    callback:
        Box<dyn Fn(&str) -> std::result::Result<Vec<u8>, EpContextDataCallbackError> + Send + Sync>,
    create_status: CreateStatusFn,
    max_data_size: usize,
}

impl Debug for EpContextDataReadRegistration {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EpContextDataReadRegistration")
            .field("max_data_size", &self.max_data_size)
            .finish_non_exhaustive()
    }
}

const CALLBACK_PANICKED: &[u8] = b"Rust EPContext data callback panicked\0";
const CALLBACK_STATE_NULL: &[u8] = b"EPContext callback state must not be null\0";
static EP_CONTEXT_CREATE_STATUS: OnceCell<CreateStatusFn> = OnceCell::new();

fn static_callback_status(code: sys::OrtErrorCode, message: &'static [u8]) -> *mut sys::OrtStatus {
    match EP_CONTEXT_CREATE_STATUS.get() {
        Some(create_status) => unsafe { create_status(code, message.as_ptr().cast()) },
        None => ptr::null_mut(),
    }
}

fn callback_status(
    registration: &EpContextDataReadRegistration,
    code: sys::OrtErrorCode,
    message: &str,
) -> *mut sys::OrtStatus {
    let mut sanitized = Vec::with_capacity(message.len() + 1);
    sanitized.extend(
        message
            .bytes()
            .map(|byte| if byte == 0 { b'?' } else { byte }),
    );
    sanitized.push(0);
    unsafe { (registration.create_status)(code, sanitized.as_ptr().cast()) }
}

unsafe fn ep_context_data_read_callback_impl(
    state: *mut c_void,
    name: *const c_char,
    allocator: *mut sys::OrtAllocator,
    buffer: *mut *mut c_void,
    data_size: *mut usize,
) -> *mut sys::OrtStatus {
    if !buffer.is_null() {
        *buffer = ptr::null_mut();
    }
    if !data_size.is_null() {
        *data_size = 0;
    }

    if state.is_null() {
        return static_callback_status(
            sys::OrtErrorCode::ORT_INVALID_ARGUMENT,
            CALLBACK_STATE_NULL,
        );
    }
    let registration = &*state.cast::<EpContextDataReadRegistration>();

    if buffer.is_null() || data_size.is_null() {
        return callback_status(
            registration,
            sys::OrtErrorCode::ORT_INVALID_ARGUMENT,
            "EPContext callback output pointers must not be null",
        );
    }
    if name.is_null() {
        return callback_status(
            registration,
            sys::OrtErrorCode::ORT_INVALID_ARGUMENT,
            "EPContext data name must not be null",
        );
    }
    if allocator.is_null() {
        return callback_status(
            registration,
            sys::OrtErrorCode::ORT_INVALID_ARGUMENT,
            "EPContext callback allocator must not be null",
        );
    }
    let alloc = match (*allocator).Alloc {
        Some(alloc) => alloc,
        None => {
            return callback_status(
                registration,
                sys::OrtErrorCode::ORT_INVALID_ARGUMENT,
                "EPContext callback allocator has no allocation function",
            )
        }
    };
    let name = match CStr::from_ptr(name).to_str() {
        Ok(name) => name,
        Err(_) => {
            return callback_status(
                registration,
                sys::OrtErrorCode::ORT_INVALID_ARGUMENT,
                "EPContext data name is not valid UTF-8",
            )
        }
    };

    let data = match (registration.callback)(name) {
        Ok(data) => data,
        Err(EpContextDataCallbackError::InvalidArgument(message)) => {
            return callback_status(
                registration,
                sys::OrtErrorCode::ORT_INVALID_ARGUMENT,
                &message,
            )
        }
        Err(EpContextDataCallbackError::Fail(message)) => {
            return callback_status(registration, sys::OrtErrorCode::ORT_FAIL, &message)
        }
    };

    if data.len() > registration.max_data_size {
        return callback_status(
            registration,
            sys::OrtErrorCode::ORT_INVALID_ARGUMENT,
            "EPContext callback data exceeds the configured maximum size",
        );
    }
    if data.is_empty() {
        return ptr::null_mut();
    }

    let output = alloc(allocator, data.len());
    if output.is_null() {
        return callback_status(
            registration,
            sys::OrtErrorCode::ORT_FAIL,
            "EPContext callback allocator failed",
        );
    }

    ptr::copy_nonoverlapping(data.as_ptr(), output.cast(), data.len());
    *buffer = output;
    *data_size = data.len();
    ptr::null_mut()
}

extern_system_fn! {
    unsafe fn ep_context_data_read_callback(
        state: *mut c_void,
        name: *const c_char,
        allocator: *mut sys::OrtAllocator,
        buffer: *mut *mut c_void,
        data_size: *mut usize,
    ) -> *mut sys::OrtStatus {
        match catch_unwind(AssertUnwindSafe(|| {
            ep_context_data_read_callback_impl(state, name, allocator, buffer, data_size)
        })) {
            Ok(status) => status,
            Err(payload) => {
                // A user-provided panic payload may itself panic when dropped.
                std::mem::forget(payload);
                if !buffer.is_null() {
                    *buffer = ptr::null_mut();
                }
                if !data_size.is_null() {
                    *data_size = 0;
                }
                if state.is_null() {
                    return static_callback_status(
                        sys::OrtErrorCode::ORT_FAIL,
                        CALLBACK_PANICKED,
                    );
                }
                let registration = &*state.cast::<EpContextDataReadRegistration>();
                (registration.create_status)(
                    sys::OrtErrorCode::ORT_FAIL,
                    CALLBACK_PANICKED.as_ptr().cast(),
                )
            }
        }
    }
}

/// Type used to create a session using the _builder pattern_
///
/// A `SessionBuilder` is created by calling the
/// [`Environment::new_session_builder()`](../env/struct.Environment.html#method.new_session_builder)
/// method on the environment.
///
/// Once created, use the different methods to configure the session.
///
/// Once configured, use the [`SessionBuilder::with_model_from_file()`](../session/struct.SessionBuilder.html#method.with_model_from_file)
/// method to "commit" the builder configuration into a [`Session`](../session/struct.Session.html).
///
/// # Example
///
/// ```no_run
/// # use std::error::Error;
/// # use std::env::var;
/// # use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel};
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
///
/// let mut session = environment
///     .new_session_builder()?
///     .with_graph_optimization_level(GraphOptimizationLevel::Basic)?
///     .with_intra_op_num_threads(1)?
///     .with_model_from_file("squeezenet.onnx")?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SessionBuilder<'a> {
    env: &'a Environment,
    session_options_ptr: *mut sys::OrtSessionOptions,

    allocator: AllocatorType,
    memory_type: MemType,
    // SessionBuilder::drop releases native options before this strong reference is dropped.
    ep_context_data_read_registration: Option<Arc<EpContextDataReadRegistration>>,
}

impl<'a> Drop for SessionBuilder<'a> {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.session_options_ptr.is_null() {
            error!("Session options pointer is null, not dropping");
        } else {
            debug!("Dropping the session options.");
            unsafe {
                self.env.env().api().ReleaseSessionOptions.unwrap()(self.session_options_ptr)
            };
        }
    }
}

impl<'a> SessionBuilder<'a> {
    pub(crate) fn new(env: &'a Environment) -> Result<SessionBuilder<'a>> {
        let mut session_options_ptr: *mut sys::OrtSessionOptions = std::ptr::null_mut();
        let status =
            unsafe { env.env().api().CreateSessionOptions.unwrap()(&mut session_options_ptr) };

        status_to_result(status).map_err(OrtError::SessionOptions)?;
        assert_null_pointer(status, "SessionStatus")?;
        assert_not_null_pointer(session_options_ptr, "SessionOptions")?;

        Ok(SessionBuilder {
            env,
            session_options_ptr,
            allocator: AllocatorType::Arena,
            memory_type: MemType::Default,
            ep_context_data_read_registration: None,
        })
    }

    /// Configure the session to use a number of threads
    pub fn with_intra_op_num_threads(self, num_threads: i16) -> Result<SessionBuilder<'a>> {
        // FIXME: Pre-built binaries use OpenMP, set env variable instead

        // We use a u16 in the builder to cover the 16-bits positive values of a i32.
        let num_threads = i32::from(num_threads);
        let status = unsafe {
            self.env.env().api().SetIntraOpNumThreads.unwrap()(
                self.session_options_ptr,
                num_threads,
            )
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;
        assert_null_pointer(status, "SessionStatus")?;
        Ok(self)
    }

    /// Set the session's optimization level
    pub fn with_graph_optimization_level(
        self,
        opt_level: GraphOptimizationLevel,
    ) -> Result<SessionBuilder<'a>> {
        // Sets graph optimization level
        unsafe {
            self.env
                .env()
                .api()
                .SetSessionGraphOptimizationLevel
                .unwrap()(self.session_options_ptr, opt_level.into())
        };
        Ok(self)
    }

    /// Set the session's allocator
    ///
    /// Defaults to [`AllocatorType::Arena`](../enum.AllocatorType.html#variant.Arena)
    pub fn with_allocator(mut self, allocator: AllocatorType) -> Result<SessionBuilder<'a>> {
        self.allocator = allocator;
        Ok(self)
    }

    /// Set the session's memory type
    ///
    /// Defaults to [`MemType::Default`](../enum.MemType.html#variant.Default)
    pub fn with_memory_type(mut self, memory_type: MemType) -> Result<SessionBuilder<'a>> {
        self.memory_type = memory_type;
        Ok(self)
    }

    /// Register a callback that supplies external EPContext data by logical name.
    ///
    /// ONNX Runtime may call the callback concurrently from different execution-provider
    /// instances or worker threads. The callback must therefore be [`Send`] and [`Sync`].
    /// The returned [`Vec`] is copied into memory allocated by ONNX Runtime; the vector may be
    /// dropped as soon as the callback returns. Callback failures stop session initialization
    /// and do not fall back to reading from disk.
    /// [`EpContextDataCallbackError::InvalidArgument`] maps to `ORT_INVALID_ARGUMENT`;
    /// [`EpContextDataCallbackError::Fail`] and callback panics map to `ORT_FAIL`.
    ///
    /// `max_data_size` must be greater than zero and less than [`usize::MAX`]. The callback and
    /// its captured state remain alive for the builder and every successfully created
    /// [`Session`]. Calling this method again replaces the previous callback only after native
    /// registration succeeds.
    ///
    /// This API reads data referenced by EPContext models. The Rust bindings do not currently
    /// expose model compilation or an EPContext data write callback.
    ///
    /// Available since ONNX Runtime 1.30.
    pub fn with_ep_context_data_read_callback<F>(
        mut self,
        max_data_size: usize,
        callback: F,
    ) -> Result<SessionBuilder<'a>>
    where
        F: Fn(&str) -> std::result::Result<Vec<u8>, EpContextDataCallbackError>
            + Send
            + Sync
            + 'static,
    {
        if max_data_size == 0 || max_data_size == usize::MAX {
            return Err(OrtError::SessionOptions(OrtApiError::Msg(
                "EPContext callback maximum data size must be greater than zero and less than usize::MAX"
                    .to_owned(),
            )));
        }

        let api = unsafe { self.env.env().api() };
        let create_status = api.CreateStatus.ok_or_else(|| {
            OrtError::SessionOptions(OrtApiError::Msg(
                "ONNX Runtime CreateStatus function is unavailable".to_owned(),
            ))
        })?;
        let registration = Arc::new(EpContextDataReadRegistration {
            callback: Box::new(callback),
            create_status,
            max_data_size,
        });
        let _ = EP_CONTEXT_CREATE_STATUS.set(create_status);
        let read_func: sys::OrtReadNamedBufferFunc = Some(ep_context_data_read_callback);
        let state = Arc::as_ptr(&registration) as *mut c_void;

        let create_options = api.CreateEpContextDataReadOptions.ok_or_else(|| {
            OrtError::SessionOptions(OrtApiError::Msg(
                "ONNX Runtime EPContext read options API is unavailable".to_owned(),
            ))
        })?;
        let release_options = api.ReleaseEpContextDataReadOptions.ok_or_else(|| {
            OrtError::SessionOptions(OrtApiError::Msg(
                "ONNX Runtime EPContext read options release API is unavailable".to_owned(),
            ))
        })?;
        let set_max_data_size = api.EpContextDataReadOptionsSetMaxDataSize.ok_or_else(|| {
            OrtError::SessionOptions(OrtApiError::Msg(
                "ONNX Runtime EPContext maximum data size API is unavailable".to_owned(),
            ))
        })?;
        let set_read_func = api.SessionOptionsSetEpContextDataReadFunc.ok_or_else(|| {
            OrtError::SessionOptions(OrtApiError::Msg(
                "ONNX Runtime EPContext read callback API is unavailable".to_owned(),
            ))
        })?;

        let mut read_options = EpContextDataReadOptionsGuard {
            read_options: ptr::null_mut(),
            release_options,
        };
        let status = unsafe { create_options(&mut read_options.read_options) };
        status_to_result_with_api(status, &api).map_err(OrtError::SessionOptions)?;

        let status = unsafe { set_max_data_size(read_options.read_options, max_data_size) };
        status_to_result_with_api(status, &api).map_err(OrtError::SessionOptions)?;

        let status = unsafe {
            set_read_func(
                self.session_options_ptr,
                read_func,
                state,
                read_options.read_options,
            )
        };
        status_to_result_with_api(status, &api).map_err(OrtError::SessionOptions)?;

        self.ep_context_data_read_registration = Some(registration);
        Ok(self)
    }

    /// Clear a previously registered EPContext data read callback.
    ///
    /// The old callback state is dropped only after ONNX Runtime has successfully cleared the
    /// native registration. Available since ONNX Runtime 1.30.
    pub fn without_ep_context_data_read_callback(mut self) -> Result<SessionBuilder<'a>> {
        let api = unsafe { self.env.env().api() };
        let set_read_func = api.SessionOptionsSetEpContextDataReadFunc.ok_or_else(|| {
            OrtError::SessionOptions(OrtApiError::Msg(
                "ONNX Runtime EPContext read callback API is unavailable".to_owned(),
            ))
        })?;
        let status =
            unsafe { set_read_func(self.session_options_ptr, None, ptr::null_mut(), ptr::null()) };
        status_to_result_with_api(status, &api).map_err(OrtError::SessionOptions)?;
        self.ep_context_data_read_registration = None;
        Ok(self)
    }

    /// Download an ONNX pre-trained model from the [ONNX Model Zoo](https://github.com/onnx/models) and commit the session
    #[cfg(feature = "model-fetching")]
    pub fn with_model_downloaded<M>(self, model: M) -> Result<Session>
    where
        M: Into<AvailableOnnxModel>,
    {
        self.with_model_downloaded_monomorphized(model.into())
    }

    #[cfg(feature = "model-fetching")]
    fn with_model_downloaded_monomorphized(self, model: AvailableOnnxModel) -> Result<Session> {
        let download_dir = env::current_dir().map_err(OrtDownloadError::IoError)?;
        let downloaded_path = model.download_to(download_dir)?;
        self.with_model_from_file(downloaded_path)
    }

    // TODO: Add all functions changing the options.
    //       See all OrtApi methods taking a `options: *mut OrtSessionOptions`.

    /// Load an ONNX graph from a file and commit the session
    pub fn with_model_from_file<P>(self, model_filepath_ref: P) -> Result<Session>
    where
        P: AsRef<Path> + 'a,
    {
        let model_filepath = model_filepath_ref.as_ref();
        let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();

        if !model_filepath.exists() {
            return Err(OrtError::FileDoesNotExists {
                filename: model_filepath.to_path_buf(),
            });
        }

        // Build an OsString than a vector of bytes to pass to C
        let model_path = std::ffi::OsString::from(model_filepath);
        #[cfg(target_family = "windows")]
        let model_path: Vec<u16> = model_path
            .encode_wide()
            .chain(std::iter::once(0)) // Make sure we have a null terminated string
            .collect();
        #[cfg(not(target_family = "windows"))]
        let model_path: Vec<std::os::raw::c_char> = model_path
            .as_bytes()
            .iter()
            .chain(std::iter::once(&b'\0')) // Make sure we have a null terminated string
            .map(|b| *b as std::os::raw::c_char)
            .collect();

        unsafe {
            let api = self.env.env().api();

            let status = api.CreateSession.unwrap()(
                self.env.env().env_ptr,
                model_path.as_ptr(),
                self.session_options_ptr,
                &mut session_ptr,
            );

            status_to_result(status).map_err(OrtError::Session)?;
            assert_null_pointer(status, "SessionStatus")?;
            assert_not_null_pointer(session_ptr, "Session")?;
        };
        let session_guard = SessionPointerGuard {
            session_ptr,
            release_session: unsafe { self.env.env().api().ReleaseSession.unwrap() },
        };
        let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
        let status = unsafe {
            self.env.env().api().GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr)
        };
        status_to_result(status).map_err(OrtError::Allocator)?;
        assert_null_pointer(status, "SessionStatus")?;
        assert_not_null_pointer(allocator_ptr, "Allocator")?;

        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default, &self.env)?;
        unsafe {
            // Extract input and output properties
            let num_input_nodes =
                dangerous::extract_inputs_count(session_ptr, self.env.env.clone())?;
            let num_output_nodes =
                dangerous::extract_outputs_count(session_ptr, self.env.env.clone())?;
            let inputs = (0..num_input_nodes)
                .map(|i| {
                    dangerous::extract_input(session_ptr, allocator_ptr, i, self.env.env.clone())
                })
                .collect::<Result<Vec<Input>>>()?;
            let outputs = (0..num_output_nodes)
                .map(|i| {
                    dangerous::extract_output(session_ptr, allocator_ptr, i, self.env.env.clone())
                })
                .collect::<Result<Vec<Output>>>()?;

            Ok(Session {
                env: self.env.env.clone(),
                session_ptr: session_guard.into_raw(),
                allocator_ptr,
                memory_info,
                _ep_context_data_read_registration: self.ep_context_data_read_registration.clone(),
                inputs,
                outputs,
            })
        }
    }

    /// Load an ONNX graph from memory and commit the session
    pub fn with_model_from_memory<B>(self, model_bytes: B) -> Result<Session>
    where
        B: AsRef<[u8]>,
    {
        self.with_model_from_memory_monomorphized(model_bytes.as_ref())
    }

    fn with_model_from_memory_monomorphized(self, model_bytes: &[u8]) -> Result<Session> {
        let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();
        unsafe {
            let api = self.env.env().api();

            let model_data = model_bytes.as_ptr().cast::<std::ffi::c_void>();
            let model_data_length = model_bytes.len();
            let status = api.CreateSessionFromArray.unwrap()(
                self.env.env().env_ptr,
                model_data,
                model_data_length,
                self.session_options_ptr,
                &mut session_ptr,
            );

            status_to_result(status).map_err(OrtError::Session)?;
            assert_null_pointer(status, "SessionStatus")?;
            assert_not_null_pointer(session_ptr, "Session")?;
        };
        let session_guard = SessionPointerGuard {
            session_ptr,
            release_session: unsafe { self.env.env().api().ReleaseSession.unwrap() },
        };
        let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
        let status = unsafe {
            self.env.env().api().GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr)
        };
        status_to_result(status).map_err(OrtError::Allocator)?;
        assert_null_pointer(status, "SessionStatus")?;
        assert_not_null_pointer(allocator_ptr, "Allocator")?;

        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default, &self.env)?;
        unsafe {
            // Extract input and output properties
            let num_input_nodes =
                dangerous::extract_inputs_count(session_ptr, self.env.env.clone())?;
            let num_output_nodes =
                dangerous::extract_outputs_count(session_ptr, self.env.env.clone())?;
            let inputs = (0..num_input_nodes)
                .map(|i| {
                    dangerous::extract_input(session_ptr, allocator_ptr, i, self.env.env.clone())
                })
                .collect::<Result<Vec<Input>>>()?;
            let outputs = (0..num_output_nodes)
                .map(|i| {
                    dangerous::extract_output(session_ptr, allocator_ptr, i, self.env.env.clone())
                })
                .collect::<Result<Vec<Output>>>()?;

            Ok(Session {
                env: self.env.env.clone(),
                session_ptr: session_guard.into_raw(),
                allocator_ptr,
                memory_info,
                _ep_context_data_read_registration: self.ep_context_data_read_registration.clone(),
                inputs,
                outputs,
            })
        }
    }
}

/// Type storing the session information, built from an [`Environment`](environment/struct.Environment.html)
#[derive(Debug)]
pub struct Session {
    env: _Environment,
    session_ptr: *mut sys::OrtSession,
    allocator_ptr: *mut sys::OrtAllocator,
    memory_info: MemoryInfo,
    // Dropped after Drop::drop releases the native session.
    _ep_context_data_read_registration: Option<Arc<EpContextDataReadRegistration>>,
    /// Information about the ONNX's inputs as stored in loaded file
    pub inputs: Vec<Input>,
    /// Information about the ONNX's outputs as stored in loaded file
    pub outputs: Vec<Output>,
}

/// Information about an ONNX's input as stored in loaded file
#[derive(Debug)]
pub struct Input {
    /// Name of the input layer
    pub name: String,
    /// Type of the input layer's elements
    pub input_type: TensorElementDataType,
    /// Shape of the input layer
    ///
    /// C API uses a i64 for the dimensions. We use an unsigned of the same range of the positive values.
    pub dimensions: Vec<Option<u32>>,
}

/// Information about an ONNX's output as stored in loaded file
#[derive(Debug)]
pub struct Output {
    /// Name of the output layer
    pub name: String,
    /// Type of the output layer's elements
    pub output_type: TensorElementDataType,
    /// Shape of the output layer
    ///
    /// C API uses a i64 for the dimensions. We use an unsigned of the same range of the positive values.
    pub dimensions: Vec<Option<u32>>,
}

impl Input {
    /// Return an iterator over the shape elements of the input layer
    ///
    /// Note: The member [`Input::dimensions`](struct.Input.html#structfield.dimensions)
    /// stores `u32` (since ONNX uses `i64` but which cannot be negative) so the
    /// iterator converts to `usize`.
    pub fn dimensions(&self) -> impl Iterator<Item = Option<usize>> + '_ {
        self.dimensions.iter().map(|d| d.map(|d2| d2 as usize))
    }
}

impl Output {
    /// Return an iterator over the shape elements of the output layer
    ///
    /// Note: The member [`Output::dimensions`](struct.Output.html#structfield.dimensions)
    /// stores `u32` (since ONNX uses `i64` but which cannot be negative) so the
    /// iterator converts to `usize`.
    pub fn dimensions(&self) -> impl Iterator<Item = Option<usize>> + '_ {
        self.dimensions.iter().map(|d| d.map(|d2| d2 as usize))
    }
}

impl Drop for Session {
    #[tracing::instrument]
    fn drop(&mut self) {
        debug!("Dropping the session.");
        if self.session_ptr.is_null() {
            error!("Session pointer is null, not dropping.");
        } else {
            unsafe { self.env.env().api().ReleaseSession.unwrap()(self.session_ptr) };
        }

        self.session_ptr = std::ptr::null_mut();
        self.allocator_ptr = std::ptr::null_mut();
    }
}

unsafe impl Send for Session {}

unsafe impl Sync for Session {}

impl Session {
    /// Run the input data through the ONNX graph, performing inference.
    ///
    /// Note that ONNX models can have multiple inputs; a `Vec<_>` is thus
    /// used for the input data here.
    pub fn run<'input>(
        &self,
        mut input_arrays: impl AsMut<[Box<dyn ConstructTensor + 'input>]> + 'input,
    ) -> Result<Vec<OrtOutput>> {
        let mut output_tensor_extractors_ptrs: Vec<*mut sys::OrtValue> =
            vec![std::ptr::null_mut(); self.outputs.len()];

        let output_names_cstring: Vec<CString> = self
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .map(|n| CString::new(n).unwrap())
            .collect();
        let output_names_ptr: Vec<*const i8> = output_names_cstring
            .iter()
            .map(|n| n.as_ptr().cast::<i8>())
            .collect();

        let input_names_ptr: Vec<*const i8> = self
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .map(|n| CString::new(n).unwrap())
            .map(|n| n.into_raw() as *const i8)
            .collect();

        {
            let memory_info = &self.memory_info;

            let allocator = self.allocator_ptr;

            let arr = input_arrays.as_mut();

            let input_tensors = arr
                .into_iter()
                .map(|v| v.construct(memory_info, allocator))
                .collect::<Result<Vec<_>>>()?;

            let input_arrays_shapes: Vec<Vec<usize>> =
                input_tensors.iter().map(|v| v.shape().to_vec()).collect();

            self.validate_input_shapes(&input_arrays_shapes)?;

            // Build arguments to Run()

            let input_ort_values: Vec<*const sys::OrtValue> = input_tensors
                .iter()
                .map(|input_array_ort| input_array_ort.ptr() as *const sys::OrtValue)
                .collect();

            let run_options_ptr: *const sys::OrtRunOptions = std::ptr::null();

            let status = unsafe {
                self.env.env().api().Run.unwrap()(
                    self.session_ptr,
                    run_options_ptr,
                    input_names_ptr.as_ptr(),
                    input_ort_values.as_ptr(),
                    input_ort_values.len(),
                    output_names_ptr.as_ptr(),
                    output_names_ptr.len(),
                    output_tensor_extractors_ptrs.as_mut_ptr(),
                )
            };
            status_to_result(status).map_err(OrtError::Run)?;
        }

        let outputs: Result<Vec<OrtOutputTensor>> = output_tensor_extractors_ptrs
            .into_iter()
            .map(|ptr| {
                let mut tensor_info_ptr: *mut sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
                let status = unsafe {
                    self.env.env().api().GetTensorTypeAndShape.unwrap()(
                        ptr,
                        &mut tensor_info_ptr as _,
                    )
                };
                status_to_result(status).map_err(OrtError::GetTensorTypeAndShape)?;
                let dims = unsafe { get_tensor_dimensions(tensor_info_ptr, self.env.clone()) };

                unsafe {
                    self.env.env().api().ReleaseTensorTypeAndShapeInfo.unwrap()(tensor_info_ptr)
                };
                let dims: Vec<_> = dims?.iter().map(|&n| n as usize).collect();

                let mut output_tensor_extractor =
                    OrtOwnedTensorExtractor::new(dims, self.env.clone());
                output_tensor_extractor.tensor_ptr = ptr;

                output_tensor_extractor.extract()
            })
            .collect();

        // Reconvert to CString so drop impl is called and memory is freed
        let cstrings: Result<Vec<CString>> = input_names_ptr
            .into_iter()
            .map(|p| {
                assert_not_null_pointer(p, "i8 for CString")?;
                unsafe { Ok(CString::from_raw(p as *mut i8)) }
            })
            .collect();
        cstrings?;

        outputs?
            .into_iter()
            .map(|v| OrtOutput::try_from(v))
            .collect()
    }

    fn validate_input_shapes(&self, input_array_shapes: &[Vec<usize>]) -> Result<()> {
        // ******************************************************************
        // FIXME: Properly handle errors here
        // Make sure all dimensions match (except dynamic ones)

        // Verify length of inputs
        if input_array_shapes.len() != self.inputs.len() {
            error!(
                "Non-matching number of inputs: {} (inference) vs {} (model)",
                input_array_shapes.len(),
                self.inputs.len()
            );
            return Err(OrtError::NonMatchingDimensions(
                NonMatchingDimensionsError::InputsCount {
                    inference_input_count: 0,
                    model_input_count: 0,
                    inference_input: input_array_shapes.to_vec(),
                    model_input: self
                        .inputs
                        .iter()
                        .map(|input| input.dimensions.clone())
                        .collect(),
                },
            ));
        }

        // Verify length of each individual inputs
        let inputs_different_length = input_array_shapes
            .iter()
            .zip(self.inputs.iter())
            .any(|(l, r)| l.len() != r.dimensions.len());
        if inputs_different_length {
            error!(
                "Different input lengths: {:?} vs {:?}",
                self.inputs, input_array_shapes
            );
            return Err(OrtError::NonMatchingDimensions(
                NonMatchingDimensionsError::InputsLength {
                    inference_input: input_array_shapes
                        .iter()
                        .map(|input_array| input_array.to_vec())
                        .collect(),
                    model_input: self
                        .inputs
                        .iter()
                        .map(|input| input.dimensions.clone())
                        .collect(),
                },
            ));
        }

        // Verify shape of each individual inputs
        let inputs_different_shape =
            input_array_shapes
                .iter()
                .zip(self.inputs.iter())
                .any(|(l, r)| {
                    let l_shape = l;
                    let r_shape = r.dimensions.as_slice();
                    l_shape.iter().zip(r_shape.iter()).any(|(l2, r2)| match r2 {
                        Some(r3) => *r3 as usize != *l2,
                        None => false, // None means dynamic size; in that case shape always match
                    })
                });
        if inputs_different_shape {
            error!(
                "Different input lengths: {:?} vs {:?}",
                self.inputs, input_array_shapes
            );
            return Err(OrtError::NonMatchingDimensions(
                NonMatchingDimensionsError::InputsLength {
                    inference_input: input_array_shapes
                        .iter()
                        .map(|input_array| input_array.to_vec())
                        .collect(),
                    model_input: self
                        .inputs
                        .iter()
                        .map(|input| input.dimensions.clone())
                        .collect(),
                },
            ));
        }

        Ok(())
    }
}

unsafe fn get_tensor_dimensions(
    tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo,
    env: _Environment,
) -> Result<Vec<i64>> {
    let mut num_dims = 0;
    let status = env.env().api().GetDimensionsCount.unwrap()(tensor_info_ptr, &mut num_dims);
    status_to_result(status).map_err(OrtError::GetDimensionsCount)?;
    (num_dims != 0)
        .then_some(())
        .ok_or(OrtError::InvalidDimensions)?;

    let mut node_dims: Vec<i64> = vec![0; num_dims as usize];
    let status = env.env().api().GetDimensions.unwrap()(
        tensor_info_ptr,
        node_dims.as_mut_ptr(), // FIXME: UB?
        num_dims,
    );
    status_to_result(status).map_err(OrtError::GetDimensions)?;
    Ok(node_dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{environment::tests::ONNX_RUNTIME_LIBRARY_PATH, LoggingLevel};
    use std::{
        env::var,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Mutex, Weak,
        },
        thread,
    };

    static TEST_ENVIRONMENT: OnceCell<Environment> = OnceCell::new();

    fn test_environment() -> &'static Environment {
        TEST_ENVIRONMENT.get_or_init(|| {
            let builder = Environment::builder()
                .with_name("ep_context_data_callback_tests")
                .with_log_level(LoggingLevel::Warning);
            match var(ONNX_RUNTIME_LIBRARY_PATH) {
                Ok(path) => builder.with_library_path(path).build().unwrap(),
                Err(_) => builder.build().unwrap(),
            }
        })
    }

    fn test_registration<F>(max_data_size: usize, callback: F) -> Arc<EpContextDataReadRegistration>
    where
        F: Fn(&str) -> std::result::Result<Vec<u8>, EpContextDataCallbackError>
            + Send
            + Sync
            + 'static,
    {
        let create_status = unsafe { test_environment().env().api().CreateStatus.unwrap() };
        let _ = EP_CONTEXT_CREATE_STATUS.set(create_status);
        Arc::new(EpContextDataReadRegistration {
            callback: Box::new(callback),
            create_status,
            max_data_size,
        })
    }

    fn default_allocator() -> *mut sys::OrtAllocator {
        let mut allocator = ptr::null_mut();
        let api = unsafe { test_environment().env().api() };
        let status = unsafe { api.GetAllocatorWithDefaultOptions.unwrap()(&mut allocator) };
        status_to_result(status).unwrap();
        assert!(!allocator.is_null());
        allocator
    }

    unsafe fn take_status(status: *mut sys::OrtStatus) -> (sys::OrtErrorCode, String) {
        assert!(!status.is_null());
        let api = test_environment().env().api();
        let code = api.GetErrorCode.unwrap()(status);
        let message = CStr::from_ptr(api.GetErrorMessage.unwrap()(status))
            .to_string_lossy()
            .into_owned();
        api.ReleaseStatus.unwrap()(status);
        (code, message)
    }

    unsafe fn invoke(
        registration: &Arc<EpContextDataReadRegistration>,
        name: &CStr,
        allocator: *mut sys::OrtAllocator,
    ) -> (*mut sys::OrtStatus, *mut c_void, usize) {
        let mut buffer = ptr::null_mut();
        let mut data_size = 0;
        let status = ep_context_data_read_callback(
            Arc::as_ptr(registration) as *mut c_void,
            name.as_ptr(),
            allocator,
            &mut buffer,
            &mut data_size,
        );
        (status, buffer, data_size)
    }

    #[test]
    fn trampoline_returns_non_empty_and_empty_data() {
        let requested_names = Arc::new(Mutex::new(Vec::new()));
        let names = requested_names.clone();
        let registration = test_registration(16, move |name| {
            names.lock().unwrap().push(name.to_owned());
            if name == "empty.bin" {
                Ok(Vec::new())
            } else {
                Ok(vec![1, 2, 3, 4])
            }
        });
        let allocator = default_allocator();

        unsafe {
            let (status, buffer, data_size) = invoke(
                &registration,
                CStr::from_bytes_with_nul(b"context.bin\0").unwrap(),
                allocator,
            );
            assert!(status.is_null());
            assert_eq!(data_size, 4);
            assert_eq!(
                std::slice::from_raw_parts(buffer.cast::<u8>(), data_size),
                [1, 2, 3, 4]
            );
            (*allocator).Free.unwrap()(allocator, buffer);

            let (status, buffer, data_size) = invoke(
                &registration,
                CStr::from_bytes_with_nul(b"empty.bin\0").unwrap(),
                allocator,
            );
            assert!(status.is_null());
            assert!(buffer.is_null());
            assert_eq!(data_size, 0);
        }
        assert_eq!(
            *requested_names.lock().unwrap(),
            ["context.bin".to_owned(), "empty.bin".to_owned()]
        );
    }

    #[test]
    fn trampoline_maps_callback_errors_and_contains_panics() {
        let registration = test_registration(16, |name| match name {
            "invalid" => Err(EpContextDataCallbackError::InvalidArgument(
                "bad\0argument".to_owned(),
            )),
            "failure" => Err(EpContextDataCallbackError::Fail("read\0failed".to_owned())),
            "panic" => panic!("callback panic"),
            _ => Ok(Vec::new()),
        });
        let allocator = default_allocator();

        unsafe {
            let (status, _, _) = invoke(
                &registration,
                CStr::from_bytes_with_nul(b"invalid\0").unwrap(),
                allocator,
            );
            let (code, message) = take_status(status);
            assert_eq!(code, sys::OrtErrorCode::ORT_INVALID_ARGUMENT);
            assert_eq!(message, "bad?argument");

            let (status, _, _) = invoke(
                &registration,
                CStr::from_bytes_with_nul(b"failure\0").unwrap(),
                allocator,
            );
            let (code, message) = take_status(status);
            assert_eq!(code, sys::OrtErrorCode::ORT_FAIL);
            assert_eq!(message, "read?failed");

            let (status, buffer, data_size) = invoke(
                &registration,
                CStr::from_bytes_with_nul(b"panic\0").unwrap(),
                allocator,
            );
            assert!(buffer.is_null());
            assert_eq!(data_size, 0);
            let (code, message) = take_status(status);
            assert_eq!(code, sys::OrtErrorCode::ORT_FAIL);
            assert_eq!(message, "Rust EPContext data callback panicked");
        }
    }

    struct PanicOnDrop;

    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("panic payload drop");
        }
    }

    #[test]
    fn trampoline_does_not_drop_panicking_panic_payload() {
        let registration = test_registration(16, |_| std::panic::panic_any(PanicOnDrop));
        let allocator = default_allocator();

        unsafe {
            let (status, buffer, data_size) = invoke(
                &registration,
                CStr::from_bytes_with_nul(b"panic\0").unwrap(),
                allocator,
            );
            assert!(buffer.is_null());
            assert_eq!(data_size, 0);
            let (code, message) = take_status(status);
            assert_eq!(code, sys::OrtErrorCode::ORT_FAIL);
            assert_eq!(message, "Rust EPContext data callback panicked");
        }
    }

    #[repr(C)]
    struct CountingAllocator {
        allocator: sys::OrtAllocator,
        allocations: AtomicUsize,
    }

    extern_system_fn! {
        unsafe fn counting_alloc(allocator: *mut sys::OrtAllocator, _size: usize) -> *mut c_void {
            let allocator = &*allocator.cast::<CountingAllocator>();
            allocator.allocations.fetch_add(1, Ordering::SeqCst);
            ptr::null_mut()
        }
    }

    #[test]
    fn trampoline_rejects_oversized_data_before_allocator_call() {
        let registration = test_registration(3, |_| Ok(vec![1, 2, 3, 4]));
        let mut allocator = CountingAllocator {
            allocator: unsafe { std::mem::zeroed() },
            allocations: AtomicUsize::new(0),
        };
        allocator.allocator.version = sys::ORT_API_VERSION;
        allocator.allocator.Alloc = Some(counting_alloc);

        unsafe {
            let (status, buffer, data_size) = invoke(
                &registration,
                CStr::from_bytes_with_nul(b"large\0").unwrap(),
                &mut allocator.allocator,
            );
            assert!(buffer.is_null());
            assert_eq!(data_size, 0);
            let (code, _) = take_status(status);
            assert_eq!(code, sys::OrtErrorCode::ORT_INVALID_ARGUMENT);
        }
        assert_eq!(allocator.allocations.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn trampoline_maps_allocator_failure() {
        let registration = test_registration(4, |_| Ok(vec![1, 2, 3, 4]));
        let mut allocator = CountingAllocator {
            allocator: unsafe { std::mem::zeroed() },
            allocations: AtomicUsize::new(0),
        };
        allocator.allocator.version = sys::ORT_API_VERSION;
        allocator.allocator.Alloc = Some(counting_alloc);

        unsafe {
            let (status, buffer, data_size) = invoke(
                &registration,
                CStr::from_bytes_with_nul(b"data\0").unwrap(),
                &mut allocator.allocator,
            );
            assert!(buffer.is_null());
            assert_eq!(data_size, 0);
            assert_eq!(take_status(status).0, sys::OrtErrorCode::ORT_FAIL);
        }
        assert_eq!(allocator.allocations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn trampoline_defends_against_null_and_invalid_name_pointers() {
        let registration = test_registration(16, |_| Ok(Vec::new()));
        let state = Arc::as_ptr(&registration) as *mut c_void;
        let allocator = default_allocator();
        let name = CStr::from_bytes_with_nul(b"context\0").unwrap();

        unsafe {
            let mut buffer = 1_usize as *mut c_void;
            let mut data_size = 7;
            let status = ep_context_data_read_callback(
                state,
                ptr::null(),
                allocator,
                &mut buffer,
                &mut data_size,
            );
            assert!(buffer.is_null());
            assert_eq!(data_size, 0);
            assert_eq!(
                take_status(status).0,
                sys::OrtErrorCode::ORT_INVALID_ARGUMENT
            );

            let status = ep_context_data_read_callback(
                state,
                name.as_ptr(),
                ptr::null_mut(),
                &mut buffer,
                &mut data_size,
            );
            assert_eq!(
                take_status(status).0,
                sys::OrtErrorCode::ORT_INVALID_ARGUMENT
            );

            let status = ep_context_data_read_callback(
                state,
                b"\xff\0".as_ptr().cast(),
                allocator,
                &mut buffer,
                &mut data_size,
            );
            assert_eq!(
                take_status(status).0,
                sys::OrtErrorCode::ORT_INVALID_ARGUMENT
            );

            let status = ep_context_data_read_callback(
                state,
                name.as_ptr(),
                allocator,
                ptr::null_mut(),
                &mut data_size,
            );
            assert_eq!(
                take_status(status).0,
                sys::OrtErrorCode::ORT_INVALID_ARGUMENT
            );

            let status = ep_context_data_read_callback(
                state,
                name.as_ptr(),
                allocator,
                &mut buffer,
                ptr::null_mut(),
            );
            assert_eq!(
                take_status(status).0,
                sys::OrtErrorCode::ORT_INVALID_ARGUMENT
            );

            let status = ep_context_data_read_callback(
                ptr::null_mut(),
                name.as_ptr(),
                allocator,
                &mut buffer,
                &mut data_size,
            );
            assert_eq!(
                take_status(status).0,
                sys::OrtErrorCode::ORT_INVALID_ARGUMENT
            );
        }
    }

    #[test]
    fn trampoline_supports_concurrent_calls() {
        const THREAD_COUNT: usize = 8;
        let call_count = Arc::new(AtomicUsize::new(0));
        let callback_count = call_count.clone();
        let registration = test_registration(16, move |name| {
            callback_count.fetch_add(1, Ordering::SeqCst);
            Ok(name.as_bytes().to_vec())
        });
        let state = Arc::as_ptr(&registration) as usize;
        let allocator = default_allocator() as usize;

        let threads: Vec<_> = (0..THREAD_COUNT)
            .map(|index| {
                thread::spawn(move || unsafe {
                    let name = CString::new(format!("data-{index}")).unwrap();
                    let mut buffer = ptr::null_mut();
                    let mut data_size = 0;
                    let status = ep_context_data_read_callback(
                        state as *mut c_void,
                        name.as_ptr(),
                        allocator as *mut sys::OrtAllocator,
                        &mut buffer,
                        &mut data_size,
                    );
                    assert!(status.is_null());
                    assert_eq!(
                        std::slice::from_raw_parts(buffer.cast::<u8>(), data_size),
                        name.as_bytes()
                    );
                    (*(allocator as *mut sys::OrtAllocator)).Free.unwrap()(
                        allocator as *mut sys::OrtAllocator,
                        buffer,
                    );
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(call_count.load(Ordering::SeqCst), THREAD_COUNT);
    }

    unsafe fn registered_callback(
        builder: &SessionBuilder<'_>,
    ) -> (sys::OrtReadNamedBufferFunc, *mut c_void, usize) {
        let api = builder.env.env().api();
        let ep_api = &*api.GetEpApi.unwrap()();
        let mut config = ptr::null_mut();
        status_to_result(ep_api.SessionOptionsGetEpContextConfig.unwrap()(
            builder.session_options_ptr,
            &mut config,
        ))
        .unwrap();

        let mut read_func = None;
        let mut state = ptr::null_mut();
        let mut max_data_size = 0;
        status_to_result(ep_api.EpContextConfigGetEpContextDataReadFunc.unwrap()(
            config,
            &mut read_func,
            &mut state,
            &mut max_data_size,
        ))
        .unwrap();
        ep_api.ReleaseEpContextConfig.unwrap()(config);
        (read_func, state, max_data_size)
    }

    #[test]
    fn registration_replaces_and_clears_native_callback_transactionally() {
        let first_lifetime = Arc::new(());
        let first_weak = Arc::downgrade(&first_lifetime);
        let first_capture = first_lifetime.clone();
        let builder = test_environment()
            .new_session_builder()
            .unwrap()
            .with_ep_context_data_read_callback(8, move |_| {
                let _ = &first_capture;
                Ok(vec![1])
            })
            .unwrap();
        drop(first_lifetime);

        let (_, first_state, first_max_data_size) = unsafe { registered_callback(&builder) };
        assert!(!first_state.is_null());
        assert_eq!(first_max_data_size, 8);
        assert!(first_weak.upgrade().is_some());

        let second_lifetime = Arc::new(());
        let second_weak = Arc::downgrade(&second_lifetime);
        let second_capture = second_lifetime.clone();
        let requested_name = Arc::new(Mutex::new(None));
        let callback_name = requested_name.clone();
        let builder = builder
            .with_ep_context_data_read_callback(16, move |name| {
                let _ = &second_capture;
                *callback_name.lock().unwrap() = Some(name.to_owned());
                Ok(vec![9])
            })
            .unwrap();
        drop(second_lifetime);

        let (read_func, second_state, second_max_data_size) =
            unsafe { registered_callback(&builder) };
        assert!(read_func.is_some());
        assert!(!second_state.is_null());
        assert_ne!(first_state, second_state);
        assert_eq!(second_max_data_size, 16);
        assert!(first_weak.upgrade().is_none());
        assert!(second_weak.upgrade().is_some());

        unsafe {
            let allocator = default_allocator();
            let mut buffer = ptr::null_mut();
            let mut data_size = 0;
            let name = CStr::from_bytes_with_nul(b"native-config.bin\0").unwrap();
            let status = read_func.unwrap()(
                second_state,
                name.as_ptr(),
                allocator,
                &mut buffer,
                &mut data_size,
            );
            assert!(status.is_null());
            assert_eq!(
                std::slice::from_raw_parts(buffer.cast::<u8>(), data_size),
                [9]
            );
            (*allocator).Free.unwrap()(allocator, buffer);
        }
        assert_eq!(
            *requested_name.lock().unwrap(),
            Some("native-config.bin".to_owned())
        );

        let builder = builder.without_ep_context_data_read_callback().unwrap();
        let (read_func, state, _) = unsafe { registered_callback(&builder) };
        assert!(read_func.is_none());
        assert!(state.is_null());
        assert!(second_weak.upgrade().is_none());
    }

    #[test]
    fn registration_rejects_non_finite_maximum_sizes() {
        let result = test_environment()
            .new_session_builder()
            .unwrap()
            .with_ep_context_data_read_callback(0, |_| Ok(Vec::new()));
        assert!(matches!(result, Err(OrtError::SessionOptions(_))));

        let result = test_environment()
            .new_session_builder()
            .unwrap()
            .with_ep_context_data_read_callback(usize::MAX, |_| Ok(Vec::new()));
        assert!(matches!(result, Err(OrtError::SessionOptions(_))));
    }

    #[test]
    fn session_retains_callback_registration() {
        let lifetime = Arc::new(());
        let weak: Weak<()> = Arc::downgrade(&lifetime);
        let capture = lifetime.clone();
        let builder = test_environment()
            .new_session_builder()
            .unwrap()
            .with_ep_context_data_read_callback(16, move |_| {
                let _ = &capture;
                Ok(Vec::new())
            })
            .unwrap();
        drop(lifetime);

        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../onnxruntime/test/testdata/mul_1.onnx");
        let session = builder.with_model_from_file(model_path).unwrap();
        assert!(weak.upgrade().is_some());
        drop(session);
        assert!(weak.upgrade().is_none());
    }
}

/// This module contains dangerous functions working on raw pointers.
/// Those functions are only to be used from inside the
/// `SessionBuilder::with_model_from_file()` method.
mod dangerous {
    use std::convert::TryFrom;

    use super::{
        assert_not_null_pointer, assert_null_pointer, char_p_to_string, get_tensor_dimensions,
        status_to_result, sys, Input, OrtApiError, OrtError, Output, Result, TensorElementDataType,
    };

    use crate::environment::_Environment;

    pub(super) unsafe fn extract_inputs_count(
        session_ptr: *mut sys::OrtSession,
        env: _Environment,
    ) -> Result<usize> {
        let f = env.env().api().SessionGetInputCount.unwrap();
        extract_io_count(f, session_ptr)
    }

    pub(super) unsafe fn extract_outputs_count(
        session_ptr: *mut sys::OrtSession,
        env: _Environment,
    ) -> Result<usize> {
        let f = env.env().api().SessionGetOutputCount.unwrap();
        extract_io_count(f, session_ptr)
    }

    fn extract_io_count(
        f: extern_system_fn! { unsafe fn(*const sys::OrtSession, *mut usize) -> *mut sys::OrtStatus },
        session_ptr: *mut sys::OrtSession,
    ) -> Result<usize> {
        let mut num_nodes: usize = 0;
        let status = unsafe { f(session_ptr, &mut num_nodes) };
        status_to_result(status).map_err(OrtError::InOutCount)?;
        assert_null_pointer(status, "SessionStatus")?;
        (num_nodes != 0).then_some(()).ok_or_else(|| {
            OrtError::InOutCount(OrtApiError::Msg("No nodes in model".to_owned()))
        })?;
        Ok(num_nodes)
    }

    unsafe fn extract_input_name(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: usize,
        env: _Environment,
    ) -> Result<String> {
        let f = env.env().api().SessionGetInputName.unwrap();
        extract_io_name(f, session_ptr, allocator_ptr, i, env)
    }

    unsafe fn extract_output_name(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: usize,
        env: _Environment,
    ) -> Result<String> {
        let f = env.env().api().SessionGetOutputName.unwrap();
        extract_io_name(f, session_ptr, allocator_ptr, i, env)
    }

    fn extract_io_name(
        f: extern_system_fn! { unsafe fn(
            *const sys::OrtSession,
            usize,
            *mut sys::OrtAllocator,
            *mut *mut i8,
        ) -> *mut sys::OrtStatus },
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: usize,
        env: _Environment,
    ) -> Result<String> {
        let mut name_bytes: *mut i8 = std::ptr::null_mut();

        let status = unsafe { f(session_ptr, i, allocator_ptr, &mut name_bytes) };
        status_to_result(status).map_err(OrtError::InputName)?;
        assert_not_null_pointer(name_bytes, "InputName")?;

        let name = char_p_to_string(name_bytes)?;

        unsafe {
            env.env().api().AllocatorFree.unwrap()(
                allocator_ptr,
                name_bytes as *mut std::ffi::c_void,
            )
        };

        Ok(name)
    }

    pub(super) unsafe fn extract_input(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: usize,
        env: _Environment,
    ) -> Result<Input> {
        let input_name = extract_input_name(session_ptr, allocator_ptr, i, env.clone())?;
        let f = env.env().api().SessionGetInputTypeInfo.unwrap();
        let (input_type, dimensions) = extract_io(f, session_ptr, i, env)?;
        Ok(Input {
            name: input_name,
            input_type,
            dimensions,
        })
    }

    pub(super) unsafe fn extract_output(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: usize,
        env: _Environment,
    ) -> Result<Output> {
        let output_name = extract_output_name(session_ptr, allocator_ptr, i, env.clone())?;
        let f = env.env().api().SessionGetOutputTypeInfo.unwrap();
        let (output_type, dimensions) = extract_io(f, session_ptr, i, env)?;
        Ok(Output {
            name: output_name,
            output_type,
            dimensions,
        })
    }

    fn extract_io(
        f: extern_system_fn! { unsafe fn(
            *const sys::OrtSession,
            usize,
            *mut *mut sys::OrtTypeInfo,
        ) -> *mut sys::OrtStatus },
        session_ptr: *mut sys::OrtSession,
        i: usize,
        env: _Environment,
    ) -> Result<(TensorElementDataType, Vec<Option<u32>>)> {
        let mut typeinfo_ptr: *mut sys::OrtTypeInfo = std::ptr::null_mut();

        let status = unsafe { f(session_ptr, i, &mut typeinfo_ptr) };
        status_to_result(status).map_err(OrtError::GetTypeInfo)?;
        assert_not_null_pointer(typeinfo_ptr, "TypeInfo")?;

        let mut tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status = unsafe {
            env.env().api().CastTypeInfoToTensorInfo.unwrap()(typeinfo_ptr, &mut tensor_info_ptr)
        };
        status_to_result(status).map_err(OrtError::CastTypeInfoToTensorInfo)?;
        assert_not_null_pointer(tensor_info_ptr, "TensorInfo")?;

        let mut type_sys = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        let status = unsafe {
            env.env().api().GetTensorElementType.unwrap()(tensor_info_ptr, &mut type_sys)
        };
        status_to_result(status).map_err(OrtError::TensorElementType)?;
        let io_type = TensorElementDataType::try_from(type_sys)?;

        // info!("{} : type={}", i, type_);

        let node_dims = unsafe { get_tensor_dimensions(tensor_info_ptr, env.clone())? };

        // for j in 0..num_dims {
        //     info!("{} : dim {}={}", i, j, node_dims[j as usize]);
        // }

        unsafe { env.env().api().ReleaseTypeInfo.unwrap()(typeinfo_ptr) };

        Ok((
            io_type,
            node_dims
                .into_iter()
                .map(|d| if d == -1 { None } else { Some(d as u32) })
                .collect(),
        ))
    }
}
