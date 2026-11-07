// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#[test]
fn current_c_api() {
    use onnxruntime_sys::{onnxruntime, OrtAllocator, ORT_API_VERSION};

    let path = std::env::var("RUST_ONNXRUNTIME_LIBRARY_PATH")
        .expect("Bazel must provide the source-built runtime library path");
    unsafe {
        let library = onnxruntime::new(path).expect("Could not load the Bazel-built runtime");
        let base = library.OrtGetApiBase();
        assert!(!base.is_null());
        let api = (*base).GetApi.unwrap()(ORT_API_VERSION);
        assert!(!api.is_null(), "The runtime must support the generated C API");
        let mut allocator: *mut OrtAllocator = std::ptr::null_mut();
        let status = (*api).GetAllocatorWithDefaultOptions.unwrap()(&mut allocator);
        assert!(status.is_null());
        assert!(!allocator.is_null());
    }
}

#[test]
fn add_inference() {
    use onnxruntime::{environment::Environment, ndarray::Array};

    let path = std::env::var("RUST_ONNXRUNTIME_LIBRARY_PATH")
        .expect("Bazel must provide the source-built runtime library path");
    let environment = Environment::builder().with_library_path(path).build().unwrap();
    let session = environment
        .new_session_builder()
        .unwrap()
        .with_intra_op_num_threads(1)
        .unwrap()
        .with_model_from_memory(include_bytes!("../../go/testdata/add_f32.onnx"))
        .unwrap();
    let left = Array::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right = Array::from_shape_vec((2, 3), vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
    let outputs = session.run(vec![left.into(), right.into()]).unwrap();
    assert_eq!(outputs.len(), 1);
    let output = outputs[0].float_array().unwrap().view();
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(
        output.iter().copied().collect::<Vec<_>>(),
        vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
    );
}
