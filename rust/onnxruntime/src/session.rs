//! Module containing session types

use std::{convert::TryFrom, ffi::CString, fmt::Debug, path::Path};

#[cfg(not(target_family = "windows"))]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_family = "windows")]
use std::os::windows::ffi::OsStrExt;

#[cfg(feature = "model-fetching")]
use std::env;

use crate::{
    char_p_to_string,
    environment::{Environment, _Environment},
    error::{
        assert_not_null_pointer, assert_null_pointer, status_to_result, NonMatchingDimensionsError,
        OrtApiError, OrtError, Result,
    },
    memory::MemoryInfo,
    tensor::{
        construct::ConstructTensor,
        ort_output_tensor::{OrtOutput, OrtOwnedTensorExtractor},
        OrtOutputTensor,
    },
    AllocatorType, GraphOptimizationLevel, MemType, TensorElementDataType,
};
use onnxruntime_sys as sys;

use tracing::{debug, error};

#[cfg(feature = "model-fetching")]
use crate::{download::AvailableOnnxModel, error::OrtDownloadError};

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
        let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
        let status = unsafe {
            self.env.env().api().GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr)
        };
        status_to_result(status).map_err(OrtError::Allocator)?;
        assert_null_pointer(status, "SessionStatus")?;
        assert_not_null_pointer(allocator_ptr, "Allocator")?;

        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default, self.env)?;
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
                session_ptr,
                allocator_ptr,
                memory_info,
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
        let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
        let status = unsafe {
            self.env.env().api().GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr)
        };
        status_to_result(status).map_err(OrtError::Allocator)?;
        assert_null_pointer(status, "SessionStatus")?;
        assert_not_null_pointer(allocator_ptr, "Allocator")?;

        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default, self.env)?;
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
                session_ptr,
                allocator_ptr,
                memory_info,
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
    pub fn run<'input, 'output>(
        &'output self,
        mut input_arrays: impl AsMut<[Box<dyn ConstructTensor + 'input>]> + 'input,
    ) -> Result<Vec<OrtOutput<'output>>> {
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
                .iter_mut()
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

        outputs?.into_iter().map(OrtOutput::try_from).collect()
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

    let mut node_dims: Vec<i64> = vec![0; num_dims];
    let status = env.env().api().GetDimensions.unwrap()(
        tensor_info_ptr,
        node_dims.as_mut_ptr(), // FIXME: UB?
        num_dims,
    );
    status_to_result(status).map_err(OrtError::GetDimensions)?;
    Ok(node_dims)
}

/// This module contains dangerous functions working on raw pointers.
/// Those functions are only to be used from inside the
/// `SessionBuilder::with_model_from_file()` method.
mod dangerous {
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
        (type_sys != sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
            .then_some(())
            .ok_or(OrtError::UndefinedTensorElementType)?;
        // This transmute should be safe since its value is read from GetTensorElementType which we must trust.
        let io_type: TensorElementDataType = unsafe { std::mem::transmute(type_sys) };

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
