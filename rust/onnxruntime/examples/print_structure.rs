//! Display the input and output structure of an ONNX model.
use onnxruntime::{environment, LoggingLevel};
use std::{env::var, error::Error};

fn main() -> Result<(), Box<dyn Error>> {
    let path = var("RUST_ONNXRUNTIME_LIBRARY_PATH").ok();

    let builder = environment::Environment::builder()
        .with_name("onnx_metadata")
        .with_log_level(LoggingLevel::Verbose);

    let builder = if let Some(path) = path.clone() {
        builder.with_library_path(path)
    } else {
        builder
    };

    let environment = builder.build().unwrap();

    // provide path to .onnx model on disk
    let path = std::env::args()
        .nth(1)
        .expect("Must provide an .onnx file as the first arg");

    let session = environment
        .new_session_builder()?
        .with_optimization_level(onnxruntime::GraphOptimizationLevel::Basic)?
        .with_model_from_file(path)?;

    println!("Inputs:");
    for (index, (name, tensor_info)) in session.inputs.iter().enumerate() {
        println!(
            "  {}:\n    name = {}\n    type = {:?}\n    shape = {:?}",
            index, name, tensor_info.element_type, tensor_info.tensor_shape
        )
    }

    println!("Outputs:");
    for (index, (name, tensor_info)) in session.outputs.iter().enumerate() {
        println!(
            "  {}:\n    name = {}\n    type = {:?}\n    shape = {:?}",
            index, name, tensor_info.element_type, tensor_info.tensor_shape
        );
    }

    println!("Metadata: {:?}", session.get_metadata());

    Ok(())
}
