#![allow(non_snake_case)]

use std::env::args;
#[cfg(not(target_family = "windows"))]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_family = "windows")]
use std::os::windows::ffi::OsStrExt;

use onnxruntime_sys::{
    onnxruntime, GraphOptimizationLevel, ONNXTensorElementDataType, OrtAllocator, OrtAllocatorType,
    OrtApi, OrtEnv, OrtLoggingLevel, OrtMemType, OrtMemoryInfo, OrtRunOptions, OrtSession,
    OrtSessionOptions, OrtStatus, OrtTensorTypeAndShapeInfo, OrtTypeInfo, OrtValue,
    ORT_API_VERSION,
};

// https://github.com/microsoft/onnxruntime/blob/v1.4.0/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp

fn main() {
    let onnxruntime_path = args()
        .nth(1)
        .expect("This example expects a path to the ONNXRuntime shared library");

    let (_, g_ort) = unsafe {
        let ort = onnxruntime::new(onnxruntime_path);

        let ort = ort.expect("Error initializing onnxruntime");
        let g_ort = ort.OrtGetApiBase().as_ref().unwrap().GetApi.unwrap()(ORT_API_VERSION);

        (ort, g_ort)
    };
    assert_ne!(g_ort, std::ptr::null_mut());

    //*************************************************************************
    // initialize  enviroment...one enviroment per process
    // enviroment maintains thread pools and other state info
    let mut env_ptr: *mut OrtEnv = std::ptr::null_mut();
    let env_name = std::ffi::CString::new("test").unwrap();
    let status = unsafe {
        g_ort.as_ref().unwrap().CreateEnv.unwrap()(
            OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
            env_name.as_ptr(),
            &mut env_ptr,
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(env_ptr, std::ptr::null_mut());

    // initialize session options if needed
    let mut session_options_ptr: *mut OrtSessionOptions = std::ptr::null_mut();
    let status =
        unsafe { g_ort.as_ref().unwrap().CreateSessionOptions.unwrap()(&mut session_options_ptr) };
    CheckStatus(g_ort, status).unwrap();
    unsafe { g_ort.as_ref().unwrap().SetIntraOpNumThreads.unwrap()(session_options_ptr, 1) };
    assert_ne!(session_options_ptr, std::ptr::null_mut());

    // Sets graph optimization level
    unsafe {
        g_ort
            .as_ref()
            .unwrap()
            .SetSessionGraphOptimizationLevel
            .unwrap()(
            session_options_ptr,
            GraphOptimizationLevel::ORT_ENABLE_BASIC,
        )
    };

    // Optionally add more execution providers via session_options
    // E.g. for CUDA include cuda_provider_factory.h and uncomment the following line:
    // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

    //*************************************************************************
    // create session and load model into memory
    // NOTE: Original C version loaded SqueezeNet 1.0 (ONNX version: 1.3, Opset version: 8,
    //       https://github.com/onnx/models/blob/main/vision/classification/squeezenet/model/squeezenet1.0-8.onnx)
    //       Download it:
    //           curl -LO "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-8.onnx"
    //       Reference: https://github.com/onnx/models/tree/main/vision/classification/squeezenet#model
    let model_path = std::ffi::OsString::from("squeezenet1.0-8.onnx");

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

    let mut session_ptr: *mut OrtSession = std::ptr::null_mut();

    println!("Using Onnxruntime C API");
    let status = unsafe {
        g_ort.as_ref().unwrap().CreateSession.unwrap()(
            env_ptr,
            model_path.as_ptr(),
            session_options_ptr,
            &mut session_ptr,
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(session_ptr, std::ptr::null_mut());

    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    // size_t num_input_nodes;
    let mut allocator_ptr: *mut OrtAllocator = std::ptr::null_mut();
    let status = unsafe {
        g_ort
            .as_ref()
            .unwrap()
            .GetAllocatorWithDefaultOptions
            .unwrap()(&mut allocator_ptr)
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(allocator_ptr, std::ptr::null_mut());

    // print number of model input nodes
    let mut num_input_nodes: usize = 0;
    let status = unsafe {
        g_ort.as_ref().unwrap().SessionGetInputCount.unwrap()(session_ptr, &mut num_input_nodes)
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(num_input_nodes, 0);
    println!("Number of inputs = {:?}", num_input_nodes);
    let mut input_node_names: Vec<&str> = Vec::new();
    let mut input_node_dims: Vec<i64> = Vec::new(); // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                                    // Otherwise need vector<vector<>>

    // iterate over all input nodes
    for i in 0..num_input_nodes {
        // print input node names
        let mut input_name: *mut i8 = std::ptr::null_mut();
        let status = unsafe {
            g_ort.as_ref().unwrap().SessionGetInputName.unwrap()(
                session_ptr,
                i,
                allocator_ptr,
                &mut input_name,
            )
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(input_name, std::ptr::null_mut());

        // WARNING: The C function SessionGetInputName allocates memory for the string.
        //          We cannot let Rust free that string, the C side must free the string.
        //          We thus convert the pointer to a string slice (&str).
        let input_name = char_p_to_str(input_name).unwrap();
        println!("Input {} : name={}", i, input_name);
        input_node_names.push(input_name);

        // print input node types
        let mut typeinfo_ptr: *mut OrtTypeInfo = std::ptr::null_mut();
        let status = unsafe {
            g_ort.as_ref().unwrap().SessionGetInputTypeInfo.unwrap()(
                session_ptr,
                i,
                &mut typeinfo_ptr,
            )
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(typeinfo_ptr, std::ptr::null_mut());

        let mut tensor_info_ptr: *const OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status = unsafe {
            g_ort.as_ref().unwrap().CastTypeInfoToTensorInfo.unwrap()(
                typeinfo_ptr,
                &mut tensor_info_ptr,
            )
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(tensor_info_ptr, std::ptr::null_mut());

        let mut type_: ONNXTensorElementDataType =
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        let status = unsafe {
            g_ort.as_ref().unwrap().GetTensorElementType.unwrap()(tensor_info_ptr, &mut type_)
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(
            type_,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
        );

        println!("Input {} : type={}", i, type_ as i32);

        // print input shapes/dims
        let mut num_dims = 0;
        let status = unsafe {
            g_ort.as_ref().unwrap().GetDimensionsCount.unwrap()(tensor_info_ptr, &mut num_dims)
        };
        CheckStatus(g_ort, status).unwrap();
        assert_ne!(num_dims, 0);

        println!("Input {} : num_dims={}", i, num_dims);
        input_node_dims.resize_with(num_dims as usize, Default::default);
        let status = unsafe {
            g_ort.as_ref().unwrap().GetDimensions.unwrap()(
                tensor_info_ptr,
                input_node_dims.as_mut_ptr(),
                num_dims,
            )
        };
        CheckStatus(g_ort, status).unwrap();

        for j in 0..num_dims {
            println!("Input {} : dim {}={}", i, j, input_node_dims[j as usize]);
        }

        unsafe { g_ort.as_ref().unwrap().ReleaseTypeInfo.unwrap()(typeinfo_ptr) };
    }

    // Results should be...
    // Number of inputs = 1
    // Input 0 : name = data_0
    // Input 0 : type = 1
    // Input 0 : num_dims = 4
    // Input 0 : dim 0 = 1
    // Input 0 : dim 1 = 3
    // Input 0 : dim 2 = 224
    // Input 0 : dim 3 = 224

    //*************************************************************************
    // Similar operations to get output node information.
    // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
    // OrtSessionGetOutputTypeInfo() as shown above.

    //*************************************************************************
    // Score the model using sample data, and inspect values

    let input_tensor_size = 224 * 224 * 3; // simplify ... using known dim values to calculate size
                                           // use OrtGetTensorShapeElementCount() to get official size!

    let output_node_names = &["softmaxout_1"];

    // initialize input data with values in [0.0, 1.0]
    let mut input_tensor_values: Vec<f32> = (0..input_tensor_size)
        .map(|i| (i as f32) / ((input_tensor_size + 1) as f32))
        .collect();

    // create input tensor object from data values
    let mut memory_info_ptr: *mut OrtMemoryInfo = std::ptr::null_mut();
    let status = unsafe {
        g_ort.as_ref().unwrap().CreateCpuMemoryInfo.unwrap()(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault,
            &mut memory_info_ptr,
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(memory_info_ptr, std::ptr::null_mut());

    // FIXME: Check me!
    let mut input_tensor_ptr: *mut OrtValue = std::ptr::null_mut();
    let input_tensor_ptr_ptr: *mut *mut OrtValue = &mut input_tensor_ptr;
    let input_tensor_values_ptr: *mut std::ffi::c_void =
        input_tensor_values.as_mut_ptr().cast::<std::ffi::c_void>();
    assert_ne!(input_tensor_values_ptr, std::ptr::null_mut());

    let shape: *const i64 = input_node_dims.as_ptr();
    assert_ne!(shape, std::ptr::null_mut());

    let status = unsafe {
        g_ort
            .as_ref()
            .unwrap()
            .CreateTensorWithDataAsOrtValue
            .unwrap()(
            memory_info_ptr,
            input_tensor_values_ptr,
            input_tensor_size * std::mem::size_of::<f32>(),
            shape,
            4,
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            input_tensor_ptr_ptr,
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(input_tensor_ptr, std::ptr::null_mut());

    let mut is_tensor = 0;
    let status =
        unsafe { g_ort.as_ref().unwrap().IsTensor.unwrap()(input_tensor_ptr, &mut is_tensor) };
    CheckStatus(g_ort, status).unwrap();
    assert_eq!(is_tensor, 1);

    let input_tensor_ptr2: *const OrtValue = input_tensor_ptr as *const OrtValue;
    let input_tensor_ptr3: *const *const OrtValue = &input_tensor_ptr2;

    unsafe { g_ort.as_ref().unwrap().ReleaseMemoryInfo.unwrap()(memory_info_ptr) };

    // score model & input tensor, get back output tensor

    let input_node_names_cstring: Vec<std::ffi::CString> = input_node_names
        .into_iter()
        .map(|n| std::ffi::CString::new(n).unwrap())
        .collect();
    let input_node_names_ptr: Vec<*const i8> = input_node_names_cstring
        .into_iter()
        .map(|n| n.into_raw() as *const i8)
        .collect();
    let input_node_names_ptr_ptr: *const *const i8 = input_node_names_ptr.as_ptr();

    let output_node_names_cstring: Vec<std::ffi::CString> = output_node_names
        .iter()
        .map(|n| std::ffi::CString::new(*n).unwrap())
        .collect();
    let output_node_names_ptr: Vec<*const i8> = output_node_names_cstring
        .iter()
        .map(|n| n.as_ptr().cast::<i8>())
        .collect();
    let output_node_names_ptr_ptr: *const *const i8 = output_node_names_ptr.as_ptr();

    let _input_node_names_cstring =
        unsafe { std::ffi::CString::from_raw(input_node_names_ptr[0] as *mut i8) };
    let run_options_ptr: *const OrtRunOptions = std::ptr::null();
    let mut output_tensor_ptr: *mut OrtValue = std::ptr::null_mut();
    let output_tensor_ptr_ptr: *mut *mut OrtValue = &mut output_tensor_ptr;

    let status = unsafe {
        g_ort.as_ref().unwrap().Run.unwrap()(
            session_ptr,
            run_options_ptr,
            input_node_names_ptr_ptr,
            input_tensor_ptr3,
            1,
            output_node_names_ptr_ptr,
            1,
            output_tensor_ptr_ptr,
        )
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(output_tensor_ptr, std::ptr::null_mut());

    let mut is_tensor = 0;
    let status =
        unsafe { g_ort.as_ref().unwrap().IsTensor.unwrap()(output_tensor_ptr, &mut is_tensor) };
    CheckStatus(g_ort, status).unwrap();
    assert_eq!(is_tensor, 1);

    // Get pointer to output tensor float values
    let mut floatarr: *mut f32 = std::ptr::null_mut();
    let floatarr_ptr: *mut *mut f32 = &mut floatarr;
    let floatarr_ptr_void: *mut *mut std::ffi::c_void =
        floatarr_ptr.cast::<*mut std::ffi::c_void>();
    let status = unsafe {
        g_ort.as_ref().unwrap().GetTensorMutableData.unwrap()(output_tensor_ptr, floatarr_ptr_void)
    };
    CheckStatus(g_ort, status).unwrap();
    assert_ne!(floatarr, std::ptr::null_mut());

    assert!((unsafe { *floatarr.offset(0) } - 0.000_045).abs() < 1e-6);

    // score the model, and print scores for first 5 classes
    // NOTE: The C ONNX Runtime allocated the array, we shouldn't drop the vec
    //       but let C de-allocate instead.
    let floatarr_vec: Vec<f32> = unsafe { Vec::from_raw_parts(floatarr, 5, 5) };
    for i in 0..5 {
        println!("Score for class [{}] =  {}", i, floatarr_vec[i]);
    }
    std::mem::forget(floatarr_vec);

    // Results should be as below...
    // Score for class[0] = 0.000045
    // Score for class[1] = 0.003846
    // Score for class[2] = 0.000125
    // Score for class[3] = 0.001180
    // Score for class[4] = 0.001317

    unsafe { g_ort.as_ref().unwrap().ReleaseValue.unwrap()(output_tensor_ptr) };
    unsafe { g_ort.as_ref().unwrap().ReleaseValue.unwrap()(input_tensor_ptr) };
    unsafe { g_ort.as_ref().unwrap().ReleaseSession.unwrap()(session_ptr) };
    unsafe { g_ort.as_ref().unwrap().ReleaseSessionOptions.unwrap()(session_options_ptr) };
    unsafe { g_ort.as_ref().unwrap().ReleaseEnv.unwrap()(env_ptr) };

    println!("Done!");
}

fn CheckStatus(g_ort: *const OrtApi, status: *const OrtStatus) -> Result<(), String> {
    if status != std::ptr::null() {
        let raw = unsafe { g_ort.as_ref().unwrap().GetErrorMessage.unwrap()(status) };
        Err(char_p_to_str(raw).unwrap().to_string())
    } else {
        Ok(())
    }
}

fn char_p_to_str<'a>(raw: *const i8) -> Result<&'a str, std::str::Utf8Error> {
    let c_str = unsafe { std::ffi::CStr::from_ptr(raw as *mut i8) };
    c_str.to_str()
}
