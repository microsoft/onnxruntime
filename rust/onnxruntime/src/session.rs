//! Module containing session types

use crate::{
    ort_api,
    error::{status_to_result, assert_not_null_pointer, assert_null_pointer, OrtResult, OrtError, OrtApiError},
    allocator::Allocator,
    char_ptr_to_string,
    environment::{Environment, _Environment},
    IoBinding,
    memory_info::MemoryInfo,
    allocator::AllocatorType, DeviceName, GraphOptimizationLevel, MemType,
    TensorElementDataType, OrtValue, Metadata, AsOrtValue, 
};
use indexmap::IndexMap;
use onnxruntime_sys as sys;
use std::{os::raw::c_char, sync::{Arc, Weak}};
use std::{collections::HashMap, io::Read};
use std::{ffi::CString, ffi::OsString, fmt::Debug, path::Path};

#[cfg(target_os = "windows")]
use std::os::windows::ffi::OsStrExt;

use tracing::{error, trace};

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
/// # use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let environment = Environment::builder()
///     .with_name("test")
///     .with_log_level(LoggingLevel::Verbose)
///     .build()?;
/// let mut session = environment
///     .new_session_builder()?
///     .with_optimization_level(GraphOptimizationLevel::Basic)?
///     .with_number_threads(1)?
///     .with_model_from_file("squeezenet.onnx")?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SessionBuilder<'e> {
    env: &'e Environment,
    session_options_ptr: *mut sys::OrtSessionOptions,
    allocator: AllocatorType,
    memory_type: MemType,
    load_metadata: bool,
}

impl<'e> Drop for SessionBuilder<'e> {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.session_options_ptr.is_null() {
            error!("SessionBuilder pointer is null, not dropping");
        } else {
            trace!("Dropping SessionBuilder.");
            unsafe { ort_api().ReleaseSessionOptions.unwrap()(self.session_options_ptr) };
        }

        self.session_options_ptr = std::ptr::null_mut();
    }
}

impl<'e> SessionBuilder<'e> {
    pub(crate) fn new(env: &'e Environment) -> OrtResult<SessionBuilder<'e>> {
        let mut session_options_ptr: *mut sys::OrtSessionOptions = std::ptr::null_mut();
        let status = unsafe { ort_api().CreateSessionOptions.unwrap()(&mut session_options_ptr) };

        status_to_result(status).map_err(OrtError::SessionOptions)?;
        assert_null_pointer(status, "SessionStatus")?;
        assert_not_null_pointer(session_options_ptr, "SessionOptions")?;

        Ok(SessionBuilder {
            env,
            session_options_ptr,
            allocator: AllocatorType::Arena,
            memory_type: MemType::Default,
            load_metadata: false,
        })
    }

    /// Configure the session to use a number of threads
    pub fn with_intra_op_num_threads(self, num_threads: u16) -> OrtResult<SessionBuilder<'e>> {
        // We use a u16 in the builder to cover the 16-bits positive values of a i32.
        let status = unsafe {
            ort_api().SetIntraOpNumThreads.unwrap()(self.session_options_ptr, num_threads as i32)
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;
        Ok(self)
    }

    /// Configure the session to use a number of threads
    pub fn with_inter_op_num_threads(self, num_threads: u16) -> OrtResult<SessionBuilder<'e>> {
        // We use a u16 in the builder to cover the 16-bits positive values of a i32.
        let status = unsafe {
            ort_api().SetInterOpNumThreads.unwrap()(self.session_options_ptr, num_threads as i32)
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;
        Ok(self)
    }

    /// Set the session's optimization level
    pub fn with_optimization_level(
        self,
        opt_level: GraphOptimizationLevel,
    ) -> OrtResult<SessionBuilder<'e>> {
        // Sets graph optimization level
        let status = unsafe {
            ort_api().SetSessionGraphOptimizationLevel.unwrap()(
                self.session_options_ptr,
                opt_level.into(),
            )
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;
        Ok(self)
    }

    /// Set the session's disable per session threads
    pub fn with_disable_per_session_threads(self) -> OrtResult<SessionBuilder<'e>> {
        let status = unsafe { ort_api().DisablePerSessionThreads.unwrap()(self.session_options_ptr) };
        status_to_result(status).map_err(OrtError::SessionOptions)?;
        Ok(self)
    }

    /// Enable profiling for a session.
    pub fn with_profiling(
        self,
        profile_file_prefix: Option<impl Into<String>>,
    ) -> OrtResult<SessionBuilder<'e>> {
        let status = unsafe {
            if let Some(profile_file_prefix) = profile_file_prefix {
                #[cfg(target_os = "windows")]
                {
                    // Convert string to UTF-16
                    let profile_file_prefix: Vec<u16> = OsString::from(profile_file_prefix.into()).encode_wide().collect();
                    ort_api().EnableProfiling.unwrap()(
                        self.session_options_ptr,
                        profile_file_prefix.as_ptr(),
                    )
                }
                #[cfg(not(target_os = "windows"))]
                {
                    let profile_file_prefix = CString::new(profile_file_prefix.into()).unwrap();
                    ort_api().EnableProfiling.unwrap()(
                        self.session_options_ptr,
                        profile_file_prefix.as_ptr(),
                    )
                }
            } else {
                ort_api().DisableProfiling.unwrap()(self.session_options_ptr)
            }
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;

        Ok(self)
    }

    /// Set ExecutionMode for a session.
    pub fn with_execution_mode(self, exection_mode: ExecutionMode) -> OrtResult<SessionBuilder<'e>> {
        let status = unsafe {
            ort_api().SetSessionExecutionMode.unwrap()(self.session_options_ptr, exection_mode.into())
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;

        Ok(self)
    }

    /// Set MemPattern for a session.
    pub fn with_mem_pattern(self, mem_pattern: bool) -> OrtResult<SessionBuilder<'e>> {
        let status = unsafe {
            if mem_pattern {
                ort_api().EnableMemPattern.unwrap()(self.session_options_ptr)
            } else {
                ort_api().DisableMemPattern.unwrap()(self.session_options_ptr)
            }
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;

        Ok(self)
    }

    /// Set CpuMemArena for a session.
    pub fn with_cpu_mem_arena(self, cpu_mem_arena: bool) -> OrtResult<SessionBuilder<'e>> {
        let status = unsafe {
            if cpu_mem_arena {
                ort_api().EnableCpuMemArena.unwrap()(self.session_options_ptr)
            } else {
                ort_api().DisableCpuMemArena.unwrap()(self.session_options_ptr)
            }
        };
        status_to_result(status).map_err(OrtError::SessionOptions)?;

        Ok(self)
    }

    /// Set the session to use cpu
    pub fn with_cpu(self, _use_arena: bool) -> OrtResult<SessionBuilder<'e>> {
        unimplemented!("Requires non-standard build of ORT")
        /* 
        unsafe {
            sys::OrtSessionOptionsAppendExecutionProvider_CPU(
                self.session_options_ptr,
                i32::from(use_arena),
            );
        };

        Ok(self)
        */
    }

    /// Set the session to use cuda
    #[cfg(feature = "cuda")]
    pub fn with_cuda(self, options: CUDAProviderOptions) -> OrtResult<SessionBuilder<'e>> {
        unsafe {
            let mut cuda_options_ptr: *mut sys::OrtCUDAProviderOptionsV2 = std::ptr::null_mut();
            let status = ort_api().CreateCUDAProviderOptions.unwrap()(&mut cuda_options_ptr);
            status_to_result(status).map_err(OrtError::CreateCUDAProviderOptions)?;
            assert_not_null_pointer(cuda_options_ptr, "CreateCUDAProviderOptions")?;

            let (keys, values) = options.get_keys_values();

            let status = ort_api().UpdateCUDAProviderOptions.unwrap()(
                cuda_options_ptr,
                keys.iter().map(|k| k.as_ptr()).collect::<Vec<_>>().as_ptr(),
                values
                    .iter()
                    .map(|v| v.as_ptr())
                    .collect::<Vec<_>>()
                    .as_ptr(),
                keys.len(),
            );
            status_to_result(status).map_err(OrtError::UpdateCUDAProviderOptions)?;

            let status = ort_api()
                .SessionOptionsAppendExecutionProvider_CUDA_V2
                .unwrap()(self.session_options_ptr, cuda_options_ptr);

            status_to_result(status)
                .map_err(OrtError::SessionOptionsAppendExecutionProviderCudaV2)?;

            ort_api().ReleaseCUDAProviderOptions.unwrap()(cuda_options_ptr);
        }
        Ok(self)
    }

    /// Set the session to use cuda
    #[cfg(feature = "cuda")]
    pub fn with_tensorrt(self, options: TensorrtProviderOptions) -> OrtResult<SessionBuilder<'e>> {
        unsafe {
            let mut trt_options_ptr: *mut sys::OrtTensorRTProviderOptionsV2 = std::ptr::null_mut();
            let status = ort_api().CreateTensorRTProviderOptions.unwrap()(&mut trt_options_ptr);
            status_to_result(status).map_err(OrtError::Allocator)?;
            assert_not_null_pointer(trt_options_ptr, "OrtTensorRTProviderOptionsV2")?;

            let (keys, values) = options.get_keys_values();

            let status = ort_api().UpdateTensorRTProviderOptions.unwrap()(
                trt_options_ptr,
                keys.iter().map(|k| k.as_ptr()).collect::<Vec<_>>().as_ptr(),
                values
                    .iter()
                    .map(|v| v.as_ptr())
                    .collect::<Vec<_>>()
                    .as_ptr(),
                keys.len(),
            );
            status_to_result(status).map_err(OrtError::Allocator)?;

            let status = ort_api()
                .SessionOptionsAppendExecutionProvider_TensorRT_V2
                .unwrap()(self.session_options_ptr, trt_options_ptr);

            status_to_result(status).map_err(OrtError::Allocator)?;

            ort_api().ReleaseTensorRTProviderOptions.unwrap()(trt_options_ptr);
        }
        Ok(self)
    }

    /// Set the session's allocator
    ///
    /// Defaults to [`AllocatorType::Arena`](../enum.AllocatorType.html#variant.Arena)
    pub fn with_allocator(mut self, allocator: AllocatorType) -> OrtResult<SessionBuilder<'e>> {
        self.allocator = allocator;
        Ok(self)
    }

    /// Set the session's memory type
    ///
    /// Defaults to [`MemType::Default`](../enum.MemType.html#variant.Default)
    pub fn with_memory_type(mut self, memory_type: MemType) -> OrtResult<SessionBuilder<'e>> {
        self.memory_type = memory_type;
        Ok(self)
    }

    /// Load model metadata
    pub fn with_metadata(mut self) -> OrtResult<SessionBuilder<'e>> {
        self.load_metadata = true;
        Ok(self)
    }

    // TODO: Add all functions changing the options.
    //       See all OrtApi methods taking a `options: *mut OrtSessionOptions`.

    /// Load an ONNX graph from a file and commit the session
    #[tracing::instrument]
    pub fn with_model_from_file<P>(self, model_filepath_ref: P) -> OrtResult<Arc<Session>>
    where
        P: AsRef<Path> + Debug + 'e,
    {
        let model_filepath = model_filepath_ref.as_ref();

        if !model_filepath.exists() {
            return Err(OrtError::FileDoesNotExist {
                filename: model_filepath.to_path_buf(),
            });
        }

        let mut onnx_file =
            std::fs::File::open(model_filepath).map_err(|err| OrtError::FileRead {
                filename: model_filepath.to_path_buf(),
                err,
            })?;
        let mut model = Vec::new();
        onnx_file
            .read_to_end(&mut model)
            .map_err(|err| OrtError::FileRead {
                filename: model_filepath.to_path_buf(),
                err,
            })?;

        self.with_model_from_memory(&model)
    }

    /// Load an ONNX graph from memory and commit the session
    #[tracing::instrument(skip(model_bytes))]
    pub fn with_model_from_memory<B>(self, model_bytes: B) -> OrtResult<Arc<Session>>
    where
        B: AsRef<[u8]> + Debug,
    {
        self.with_model_from_memory_monomorphized(model_bytes.as_ref())
    }

    #[tracing::instrument(skip(model_bytes))]
    fn with_model_from_memory_monomorphized(self, model_bytes: &[u8]) -> OrtResult<Arc<Session>> {
        let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();

        let env = self.env.env();

        let status = unsafe {
            let model_data = model_bytes.as_ptr() as *const std::ffi::c_void;
            let model_data_length = model_bytes.len();
            ort_api().CreateSessionFromArray.unwrap()(
                env.env_ptr as *const sys::OrtEnv,
                model_data,
                model_data_length,
                self.session_options_ptr,
                &mut session_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::Session)?;
        assert_null_pointer(status, "SessionStatus")?;
        assert_not_null_pointer(session_ptr, "Session")?;

        let allocator = Allocator::try_new()?;

        let memory_info =
            MemoryInfo::new(DeviceName::Cpu, 0, AllocatorType::Arena, MemType::Default)?;

        // Extract input and output properties
        let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
        let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
        let inputs = (0..num_input_nodes)
            .map(|i| {
                Ok( dangerous::extract_input(session_ptr, allocator.ptr, i)? )
            })
            .collect::<OrtResult<IndexMap<String, GraphTensorInfo>>>()?;
        let outputs = (0..num_output_nodes)
            .map(|i| {
                Ok( dangerous::extract_output(session_ptr, allocator.ptr, i)? )
            })
            .collect::<OrtResult<IndexMap<String, GraphTensorInfo>>>()?;
        
        let metadata = 
        if self.load_metadata {
            Some(Metadata::new(session_ptr, &allocator)?)
        } else {
            None
        };

        trace!("Created Session: {session_ptr:?}");
        Ok( Arc::new_cyclic(|weak_self| { 
            Session {
            env: self.env.env.clone(),
            ptr: session_ptr,
            allocator,
            memory_info,
            weak_self: weak_self.clone(),
            inputs,
            outputs,
            metadata,
        }}))
    }
}

/// Execution mode
#[derive(Debug, Clone)]
pub enum ExecutionMode {
    /// Sequential
    Sequential,
    /// Parallel
    Parallel,
}

impl From<ExecutionMode> for sys::ExecutionMode {
    fn from(val: ExecutionMode) -> Self {
        match val {
            ExecutionMode::Sequential => sys::ExecutionMode::ORT_SEQUENTIAL,
            ExecutionMode::Parallel => sys::ExecutionMode::ORT_PARALLEL,
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
/// Configuration options for the CUDA Execution Provider.
pub struct CUDAProviderOptions {
    /// The device ID.
    device_id: usize,
    /// The size limit of the device memory arena in bytes.
    gpu_mem_limit: usize,
    /// The strategy for extending the device memory arena.
    arena_extend_strategy: ArenaExtendStrategy,
    /// The type of search done for cuDNN convolution algorithms.
    cudnn_conv_algo_search: CuDNNConvAlgoSearch,
    /// Whether to do copies in the default stream or use separate streams.
    do_copy_in_default_stream: bool,
    /// Allow ORT to allocate the maximum possible workspace as determined by CuDNN.
    cudnn_conv_use_max_workspace: bool,
    /// Convolution Input Padding in the CUDA EP.
    cudnn_conv1d_pad_to_nc1d: bool,
    /// Enable the usage of CUDA Graphs.
    enable_cuda_graph: bool,
}

#[cfg(feature = "cuda")]
impl Default for CUDAProviderOptions {
    fn default() -> Self {
        Self {
            device_id: 0,
            gpu_mem_limit: 18446744073709551615,
            arena_extend_strategy: ArenaExtendStrategy::NextPowerOfTwo,
            cudnn_conv_algo_search: CuDNNConvAlgoSearch::Exhaustive,
            do_copy_in_default_stream: true,
            cudnn_conv_use_max_workspace: false,
            cudnn_conv1d_pad_to_nc1d: false,
            enable_cuda_graph: false,
        }
    }
}

#[cfg(feature = "cuda")]
impl CUDAProviderOptions {
    fn get_keys_values(&self) -> (Vec<CString>, Vec<CString>) {
        let keys = vec![
            "device_id",
            "gpu_mem_limit",
            "arena_extend_strategy",
            "cudnn_conv_algo_search",
            "do_copy_in_default_stream",
            "cudnn_conv_use_max_workspace",
            "cudnn_conv1d_pad_to_nc1d",
            "enable_cuda_graph",
        ]
        .into_iter()
        .map(|k| CString::new(k).unwrap())
        .collect::<Vec<_>>();

        let values = vec![
            self.device_id.to_string(),
            self.gpu_mem_limit.to_string(),
            match self.arena_extend_strategy {
                ArenaExtendStrategy::NextPowerOfTwo => "kNextPowerOfTwo",
                ArenaExtendStrategy::SameAsRequested => "kSameAsRequested",
            }
            .to_string(),
            match self.cudnn_conv_algo_search {
                CuDNNConvAlgoSearch::Exhaustive => "EXHAUSTIVE",
                CuDNNConvAlgoSearch::Heuristic => "HEURISTIC",
                CuDNNConvAlgoSearch::Default => "DEFAULT",
            }
            .to_string(),
            i32::from(self.do_copy_in_default_stream).to_string(),
            i32::from(self.cudnn_conv_use_max_workspace).to_string(),
            i32::from(self.cudnn_conv1d_pad_to_nc1d).to_string(),
            i32::from(self.enable_cuda_graph).to_string(),
        ]
        .into_iter()
        .map(|k| CString::new(k).unwrap())
        .collect::<Vec<_>>();

        (keys, values)
    }

    /// Set device_id
    pub fn with_device_id(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Set gpu_mem_limit
    pub fn with_gpu_mem_limit(mut self, gpu_mem_limit: usize) -> Self {
        self.gpu_mem_limit = gpu_mem_limit;
        self
    }

    /// Set arena_extend_strategy
    pub fn with_arena_extend_strategy(
        mut self,
        arena_extend_strategy: ArenaExtendStrategy,
    ) -> Self {
        self.arena_extend_strategy = arena_extend_strategy;
        self
    }

    /// Set cudnn_conv_algo_search
    pub fn with_cudnn_conv_algo_search(
        mut self,
        cudnn_conv_algo_search: CuDNNConvAlgoSearch,
    ) -> Self {
        self.cudnn_conv_algo_search = cudnn_conv_algo_search;
        self
    }

    /// Set do_copy_in_default_stream
    pub fn with_do_copy_in_default_stream(mut self, do_copy_in_default_stream: bool) -> Self {
        self.do_copy_in_default_stream = do_copy_in_default_stream;
        self
    }

    /// Set cudnn_conv_use_max_workspace
    pub fn with_cudnn_conv_use_max_workspace(mut self, cudnn_conv_use_max_workspace: bool) -> Self {
        self.cudnn_conv_use_max_workspace = cudnn_conv_use_max_workspace;
        self
    }

    /// Set cudnn_conv1d_pad_to_nc1d
    pub fn with_cudnn_conv1d_pad_to_nc1d(mut self, cudnn_conv1d_pad_to_nc1d: bool) -> Self {
        self.cudnn_conv1d_pad_to_nc1d = cudnn_conv1d_pad_to_nc1d;
        self
    }

    /// Set enable_cuda_graph
    pub fn with_enable_cuda_graph(mut self, enable_cuda_graph: bool) -> Self {
        self.enable_cuda_graph = enable_cuda_graph;
        self
    }
}

#[derive(Debug, Clone)]
/// The strategy for extending the device memory arena.
pub enum ArenaExtendStrategy {
    /// subsequent extensions extend by larger amounts (multiplied by powers of two)
    NextPowerOfTwo = 0,
    /// extend by the requested amount
    SameAsRequested = 1,
}

#[derive(Debug, Clone)]
/// The type of search done for cuDNN convolution algorithms.
pub enum CuDNNConvAlgoSearch {
    /// expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
    Exhaustive,
    /// lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
    Heuristic,
    /// default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
    Default,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
/// Configuration options for the TensorRT Execution Provider.
pub struct TensorrtProviderOptions {
    /// The device ID.
    device_id: usize,
    /// The maximum workspace size for TensorRT engine.
    max_workspace_size: usize,
    /// The maximum number of iterations allowed in model partitioning for TensorRT.
    max_partition_iterations: usize,
    /// The minimum node size in a subgraph after partitioning.
    min_subgraph_size: usize,
    /// Enable FP16 mode in TensorRT.
    fp16_enable: bool,
    /// Enable FP16 mode in TensorRT.
    int8_enable: bool,
    /// Specify INT8 calibration table file for non-QDQ models in INT8 mode.
    int8_calibration_table_name: Option<String>,
    ///  Select what calibration table is used for non-QDQ models in INT8 mode.
    /// If true, native TensorRT generated calibration table is used
    /// If false, ONNXRUNTIME tool generated calibration table is used.
    int8_use_native_calibration_table: bool,
    /// Enable Deep Learning Accelerator (DLA).
    dla_enable: bool,
    /// Specify DLA core to execute on.
    dla_core: usize,
    /// Enable TensorRT engine caching.
    engine_cache_enable: bool,
    /// Specify path for TensorRT engine and profile files.
    engine_cache_path: Option<String>,
    /// Dumps the subgraphs that are transformed into TRT engines in onnx format to the filesystem.
    dump_subgraphs: bool,
    /// Sequentially build TensorRT engines across provider instances in multi-GPU environment.
    force_sequential_engine_build: bool,
    /// Enable context memory sharing between subgraphs.
    #[cfg(feature = "ort_1_14_0")]
    context_memory_sharing_enable: bool,
    /// Force Pow + Reduce ops in layer norm to FP32.
    #[cfg(feature = "ort_1_14_0")]
    layer_norm_fp32_fallback: bool,
}

#[cfg(feature = "cuda")]
impl Default for TensorrtProviderOptions {
    fn default() -> Self {
        Self {
            device_id: 0,
            max_workspace_size: 1073741824,
            max_partition_iterations: 1000,
            min_subgraph_size: 1,
            fp16_enable: false,
            int8_enable: false,
            int8_calibration_table_name: None,
            int8_use_native_calibration_table: false,
            dla_enable: false,
            dla_core: 0,
            engine_cache_enable: false,
            engine_cache_path: None,
            dump_subgraphs: false,
            force_sequential_engine_build: false,
            #[cfg(feature = "ort_1_14_0")]
            context_memory_sharing_enable: false,
            #[cfg(feature = "ort_1_14_0")]
            layer_norm_fp32_fallback: false,
        }
    }
}

#[cfg(feature = "cuda")]
impl TensorrtProviderOptions {
    fn get_keys_values(&self) -> (Vec<CString>, Vec<CString>) {
        let mut keys = vec![
            "device_id",
            "trt_max_workspace_size",
            "trt_max_partition_iterations",
            "trt_min_subgraph_size",
            "trt_fp16_enable",
            "trt_int8_enable",
            "trt_int8_use_native_calibration_table",
            "trt_dla_enable",
            "trt_dla_core",
            "trt_engine_cache_enable",
            "trt_dump_subgraphs",
            "trt_force_sequential_engine_build",
        ]
        .into_iter()
        .map(|k| CString::new(k).unwrap())
        .collect::<Vec<_>>();

        #[cfg(feature = "ort_1_14_0")]
        keys.append(
            &mut vec![
                "trt_context_memory_sharing_enable",
                "trt_layer_norm_fp32_fallback",
            ]
            .into_iter()
            .map(|k| CString::new(k).unwrap())
            .collect::<Vec<_>>(),
        );

        let mut values = vec![
            self.device_id.to_string(),
            self.max_workspace_size.to_string(),
            self.max_partition_iterations.to_string(),
            self.min_subgraph_size.to_string(),
            i32::from(self.fp16_enable).to_string(),
            i32::from(self.int8_enable).to_string(),
            i32::from(self.int8_use_native_calibration_table).to_string(),
            i32::from(self.dla_enable).to_string(),
            self.dla_core.to_string(),
            i32::from(self.engine_cache_enable).to_string(),
            i32::from(self.dump_subgraphs).to_string(),
            i32::from(self.force_sequential_engine_build).to_string(),
        ]
        .into_iter()
        .map(|k| CString::new(k).unwrap())
        .collect::<Vec<_>>();

        #[cfg(feature = "ort_1_14_0")]
        values.append(
            &mut vec![
                i32::from(self.context_memory_sharing_enable).to_string(),
                i32::from(self.layer_norm_fp32_fallback).to_string(),
            ]
            .into_iter()
            .map(|k| CString::new(k).unwrap())
            .collect::<Vec<_>>(),
        );

        if let Some(engine_cache_path) = &self.engine_cache_path {
            keys.push(CString::new("trt_engine_cache_path").unwrap());
            values.push(CString::new(engine_cache_path.clone()).unwrap());
        };

        if let Some(int8_calibration_table_name) = &self.int8_calibration_table_name {
            keys.push(CString::new("trt_int8_calibration_table_name").unwrap());
            values.push(CString::new(int8_calibration_table_name.clone()).unwrap());
        };

        (keys, values)
    }

    /// Set device_id
    pub fn with_device_id(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Set trt_max_workspace_size
    pub fn with_max_workspace_size(mut self, max_workspace_size: usize) -> Self {
        self.max_workspace_size = max_workspace_size;
        self
    }

    /// Set trt_max_partition_iterations
    pub fn with_max_partition_iterations(mut self, max_partition_iterations: usize) -> Self {
        self.max_partition_iterations = max_partition_iterations;
        self
    }

    /// Set min_subgraph_size
    pub fn with_min_subgraph_size(mut self, min_subgraph_size: usize) -> Self {
        self.min_subgraph_size = min_subgraph_size;
        self
    }

    /// Set fp16_enable
    pub fn with_fp16_enable(mut self, fp16_enable: bool) -> Self {
        self.fp16_enable = fp16_enable;
        self
    }

    /// Set int8_enable
    pub fn with_int8_enable(mut self, int8_enable: bool) -> Self {
        self.int8_enable = int8_enable;
        self
    }

    /// Set int8_calibration_table_name
    pub fn with_int8_calibration_table_name(
        mut self,
        int8_calibration_table_name: Option<&str>,
    ) -> Self {
        self.int8_calibration_table_name = int8_calibration_table_name.map(|v| v.to_string());
        self
    }

    /// Set int8_use_native_calibration_table
    pub fn with_int8_use_native_calibration_table(
        mut self,
        int8_use_native_calibration_table: bool,
    ) -> Self {
        self.int8_use_native_calibration_table = int8_use_native_calibration_table;
        self
    }

    /// Set dla_enable
    pub fn with_dla_enable(mut self, dla_enable: bool) -> Self {
        self.dla_enable = dla_enable;
        self
    }

    /// Set dla_core
    pub fn with_dla_core(mut self, dla_core: usize) -> Self {
        self.dla_core = dla_core;
        self
    }

    /// Set engine_cache_enable
    pub fn with_engine_cache_enable(mut self, engine_cache_enable: bool) -> Self {
        self.engine_cache_enable = engine_cache_enable;
        self
    }

    /// Set engine_cache_path
    pub fn with_engine_cache_path(mut self, engine_cache_path: Option<&str>) -> Self {
        self.engine_cache_path = engine_cache_path.map(|v| v.to_string());
        self
    }

    /// Set dump_subgraphs
    pub fn with_dump_subgraphs(mut self, dump_subgraphs: bool) -> Self {
        self.dump_subgraphs = dump_subgraphs;
        self
    }

    /// Set force_sequential_engine_build
    pub fn with_force_sequential_engine_build(
        mut self,
        force_sequential_engine_build: bool,
    ) -> Self {
        self.force_sequential_engine_build = force_sequential_engine_build;
        self
    }

    /// Set context_memory_sharing_enable
    #[cfg(feature = "ort_1_14_0")]
    pub fn with_context_memory_sharing_enable(
        mut self,
        context_memory_sharing_enable: bool,
    ) -> Self {
        self.context_memory_sharing_enable = context_memory_sharing_enable;
        self
    }

    /// Set layer_norm_fp32_fallback
    #[cfg(feature = "ort_1_14_0")]
    pub fn with_layer_norm_fp32_fallback(mut self, layer_norm_fp32_fallback: bool) -> Self {
        self.layer_norm_fp32_fallback = layer_norm_fp32_fallback;
        self
    }
}

/// Description of tensor provided by the ONNX Graph.
/// Some dimensions may be variable (i.e. unspecified).
#[derive(Debug)]
pub struct GraphTensorInfo {
    /// Optimization to prevent regenerating this for each run() call
    name_cstr: CString,
    /// Shape of the tensor. None = variable. C API uses an i64, though the value must be positive.
    pub tensor_shape: Vec<Option<usize>>,
    /// Teneor element data type
    pub element_type: TensorElementDataType,
}

impl GraphTensorInfo {
    pub(crate) fn new(name: String, tensor_shape: Vec<Option<usize>>, element_type: TensorElementDataType) -> GraphTensorInfo {
        GraphTensorInfo { name_cstr: CString::new(name).unwrap(), tensor_shape, element_type }
    }

    /// Return an iterator over the shape elements of the input layer
    ///
    /// Note: The member [`Input::dimensions`](struct.Input.html#structfield.dimensions)
    /// stores `u32` (since ONNX uses `i64` but which cannot be negative) so the
    /// iterator converts to `usize`.
    pub fn shape(&self) -> impl Iterator<Item = Option<usize>> + '_ {
        self.tensor_shape.iter().map(|d| d.map(|d2| d2 as usize))
    }
}

/// Type storing the session information, built from an [`Environment`](environment/struct.Environment.html)
#[derive(Debug)]
#[allow(dead_code)]
pub struct Session {
    env: _Environment,
    pub(crate) ptr: *mut sys::OrtSession,
    pub(crate) allocator: Allocator,
    pub(crate) memory_info: MemoryInfo,    
    weak_self: Weak<Session>,
    /// Information about the ONNX's inputs as stored in loaded file
    pub inputs: IndexMap<String, GraphTensorInfo>,
    /// Information about the ONNX's outputs as stored in loaded file
    pub outputs: IndexMap<String, GraphTensorInfo>,
    /// Standard metadata for all graphs: producer_name, description, graph_name, domain, graph_description,
    /// plus custom key / value pairs.
    metadata: Option<Metadata>,
}

// This makes sense because, once created, the contents of Session are effectively read-only
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Drop for Session {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.ptr.is_null() {
            error!("Session pointer is null, not dropping");
        } else {
            trace!("Dropping Session: {:?}.", self.ptr);
            unsafe { ort_api().ReleaseSession.unwrap()(self.ptr) };
        }

        self.ptr = std::ptr::null_mut();
    }
}

impl Session {
    /// Run the input data through the ONNX graph, performing inference.
    ///
    /// The inputs are specified as a map of input_name to &OrtValue. 
    /// The input name must exist in session.inputs.
    /// The requested outputs are specified as a list of names, which
    /// must exist in session.outputs.
    /// The returned list of OrtValues correspond to the list of output_names.
    /// 
    /// OrtValues may safely be acquired from MutableOrtValue, 
    /// MutableOrtValueTyped or NdarrayOrtValue, though the latter is
    /// better used with run_with_arrays().
    #[tracing::instrument]
    pub fn run(
        &self,
        inputs: &HashMap<String, &OrtValue>,
        output_names: &[String],
    ) -> OrtResult<Vec<OrtValue>> {
        // Construct list of input names: pointers to CString names in self.inputs
        let input_names_ptr: Vec<*const c_char> = inputs
            .keys()
            .map(|name| match self.inputs.get(name) {
                Some(input_info) => Ok(input_info.name_cstr.as_ptr()),
                None => Err(OrtError::PointerShouldNotBeNull(format!("Input tensor {:?} does not exist", name))),
            })
            .collect::<OrtResult<Vec<_>>>()?;

        // Construct list of output names: pointers to CString names in self.outputs
        let output_names_ptr: Vec<*const c_char> = output_names
            .iter()
            .map(|name| match self.outputs.get(name) {
                Some(output_info) => Ok(output_info.name_cstr.as_ptr()),
                None => Err(OrtError::PointerShouldNotBeNull(format!("Output tensor {:?} does not exist", name))),
            })
            .collect::<OrtResult<Vec<_>>>()?;

        // Construct list of inputs: pointers to sys::OrtValues
        let input_ort_values: Vec<*const sys::OrtValue> = inputs
            .values()
            .map(|input| input.ptr as *const sys::OrtValue)
            .collect();

        // Construct list of outputs: unassigned pointers to future sys::OrtValues
        let mut output_ort_values: Vec<*mut sys::OrtValue> =
            vec![std::ptr::null_mut(); output_names_ptr.len()];

        let run_options_ptr: *const sys::OrtRunOptions = std::ptr::null();

        let status = unsafe {
            ort_api().Run.unwrap()(
                self.ptr,
                run_options_ptr,
                input_names_ptr.as_ptr(),
                input_ort_values.as_ptr(),
                input_ort_values.len(),
                output_names_ptr.as_ptr(),
                output_names_ptr.len(),
                output_ort_values.as_mut_ptr(),
            )
        };
        status_to_result(status).map_err(OrtError::Run)?;

        Ok( output_ort_values
            .into_iter()
            .map(|ort_value_ptr| ort_value_ptr.into())
            .collect::<Vec<OrtValue>>() )
    }

    /// A helper function that acquires the OrtValue for each input and 
    /// calls run(). 
    /// 
    /// When caller code stores tensors in ndarray::Arrays, one may create
    /// for each Array an NdArrayOrtValue when building the inputs map to 
    /// this function. See NdArrayOrtValue for usage.
    pub fn run_with_arrays<'i, 'v>(
        &self,
        inputs: &'i HashMap<String, Box<dyn AsOrtValue + 'v>>,
        output_names: &[String],
    ) -> OrtResult<Vec<OrtValue>> 
    where 
        'i : 'v
    {
        let ort_inputs: HashMap<String, &'v OrtValue> = inputs.iter()
            .map(|(name, val)| 
                (name.clone(), val.as_ort_value()))
            .collect();

        self.run(&ort_inputs, output_names)
    }

    /// Run the input data through the ONNX graph, performing inference.
    /// This uses the IoBinding interface, where input and output OrtValues
    /// are preconfigured with the IoBinding.
    pub fn run_with_iobinding(&self, io_binding: &IoBinding) -> OrtResult<()> {
        let run_options_ptr: *const sys::OrtRunOptions = std::ptr::null();
        let status =
            unsafe { ort_api().RunWithBinding.unwrap()(self.ptr, run_options_ptr, io_binding.ptr) };
        status_to_result(status).map_err(OrtError::Run)?;
        Ok(())
    }

    /// Create or return the session [`IoBinding`](../io_binding/struct.IoBinding.html)
    pub fn io_binding(&self) -> OrtResult<IoBinding> {
        let arc_self = self.weak_self.upgrade().expect("Session's weak pointer to itself is null. This should be impossible.");
        IoBinding::new(arc_self)
    }

    /// Return reference to metadata if it was loaded by SessionBuilder,
    /// otherwise None. (Simpler than session.metadata.as_ref())
    pub fn get_metadata(&self) -> Option<&Metadata> {
        match &self.metadata {
            Some(md) => Some(&md),
            None => None,
        }
    }
}

/// This module contains dangerous functions working on raw pointers.
/// Those functions are only to be used from inside the
/// `SessionBuilder::with_model_from_file()` method.
mod dangerous {
    use std::convert::TryInto;

    use crate::tensor_type_and_shape_info::TensorTypeAndShapeInfo;

    use super::*;

    pub(super) fn extract_inputs_count(session_ptr: *mut sys::OrtSession) -> OrtResult<usize> {
        let f = ort_api().SessionGetInputCount.unwrap();
        extract_io_count(f, session_ptr)
    }

    pub(super) fn extract_outputs_count(session_ptr: *mut sys::OrtSession) -> OrtResult<usize> {
        let f = ort_api().SessionGetOutputCount.unwrap();
        extract_io_count(f, session_ptr)
    }

    fn extract_io_count(
        f: extern_system_fn! { unsafe fn(*const sys::OrtSession, *mut usize) -> *mut sys::OrtStatus },
        session_ptr: *mut sys::OrtSession,
    ) -> OrtResult<usize> {
        let mut num_nodes: usize = 0;
        let status = unsafe { f(session_ptr, &mut num_nodes) };
        status_to_result(status).map_err(OrtError::InOutCount)?;
        assert_null_pointer(status, "SessionStatus")?;
        (num_nodes != 0).then_some(()).ok_or_else(|| {
            OrtError::InOutCount(OrtApiError::Msg("No nodes in model".to_owned()))
        })?;
        Ok(num_nodes)
    }

    pub(super) fn extract_input(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: usize,
    ) -> OrtResult<(String, GraphTensorInfo)> {
        let name = extract_input_name(session_ptr, allocator_ptr, i)?;
        let f = ort_api().SessionGetInputTypeInfo.unwrap();
        let (element_type, tensor_shape) = extract_io_type_info(f, session_ptr, i)?;
        Ok((name.clone(), GraphTensorInfo::new(name, tensor_shape, element_type)))
    }

    fn extract_input_name(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: usize,
    ) -> OrtResult<String> {
        let mut name_ptr: *mut c_char = std::ptr::null_mut();
        let status = unsafe {
            ort_api().SessionGetInputName.unwrap()(session_ptr, i, allocator_ptr, &mut name_ptr)
        };
        status_to_result(status).map_err(OrtError::SessionGetInputName)?;
        assert_not_null_pointer(name_ptr, "SessionGetInputName")?;

        let input_name = char_ptr_to_string(name_ptr)?;

        let status = unsafe {
            ort_api().AllocatorFree.unwrap()(allocator_ptr, name_ptr as *mut std::ffi::c_void)
        };
        status_to_result(status).map_err(OrtError::AllocatorFree)?;

        Ok(input_name)
    }

    fn extract_output_name(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: usize,
    ) -> OrtResult<String> {
        let mut name_ptr: *mut c_char = std::ptr::null_mut();
        let status = unsafe {
            ort_api().SessionGetOutputName.unwrap()(session_ptr, i, allocator_ptr, &mut name_ptr)
        };
        status_to_result(status).map_err(OrtError::SessionGetOutputName)?;
        assert_not_null_pointer(name_ptr, "SessionGetOutputName")?;

        let output_name = char_ptr_to_string(name_ptr)?;

        let status = unsafe {
            ort_api().AllocatorFree.unwrap()(allocator_ptr, name_ptr as *mut std::ffi::c_void)
        };
        status_to_result(status).map_err(OrtError::AllocatorFree)?;

        Ok(output_name)
    }

    pub(super) fn extract_output(
        session_ptr: *mut sys::OrtSession,
        allocator_ptr: *mut sys::OrtAllocator,
        i: usize,
    ) -> OrtResult<(String, GraphTensorInfo)> {
        let name = extract_output_name(session_ptr, allocator_ptr, i)?;
        let f = ort_api().SessionGetOutputTypeInfo.unwrap();
        let (element_type, tensor_shape) = extract_io_type_info(f, session_ptr, i)?;
        Ok((name.clone(), GraphTensorInfo::new(name, tensor_shape, element_type)))
    }

    fn extract_io_type_info(
        f: extern_system_fn! { unsafe fn(
            *const sys::OrtSession,
            usize,
            *mut *mut sys::OrtTypeInfo,
        ) -> *mut sys::OrtStatus },
        session_ptr: *mut sys::OrtSession,
        i: usize,
    ) -> OrtResult<(TensorElementDataType, Vec<Option<usize>>)> {
        let mut typeinfo_ptr: *mut sys::OrtTypeInfo = std::ptr::null_mut();

        let status = unsafe { f(session_ptr, i, &mut typeinfo_ptr) };
        status_to_result(status).map_err(OrtError::GetTypeInfo)?;
        assert_not_null_pointer(typeinfo_ptr, "TypeInfo")?;

        let mut tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status = unsafe {
            ort_api().CastTypeInfoToTensorInfo.unwrap()(typeinfo_ptr, &mut tensor_info_ptr)
        };
        status_to_result(status).map_err(OrtError::CastTypeInfoToTensorInfo)?;
        assert_not_null_pointer(tensor_info_ptr, "TensorInfo")?;

        let type_and_shape_info: TensorTypeAndShapeInfo =
            (tensor_info_ptr as *mut sys::OrtTensorTypeAndShapeInfo).try_into()?;

        Ok((
            type_and_shape_info.element_data_type.clone(),
            type_and_shape_info
                .dimensions
                .clone()
                .into_iter()
                .map(|d| if d == -1 { None } else { Some(d as usize) })
                .collect(),
        ))
    }

}
