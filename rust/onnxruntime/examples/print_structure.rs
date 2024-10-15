//! Display the input and output structure of an ONNX model.
use onnxruntime::environment;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // provide path to .onnx model on disk
    let path = std::env::args()
        .skip(1)
        .next()
        .expect("Must provide an .onnx file as the first arg");

    let environment = environment::Environment::builder()
        .with_name("onnx metadata")
        .with_log_level(onnxruntime::LoggingLevel::Verbose)
        .build()?;

    let session = environment
        .new_session_builder()?
        .with_optimization_level(onnxruntime::GraphOptimizationLevel::Basic)?
        .with_model_from_file(path)?;

    println!("Inputs:");
    for (index, input) in session.inputs.iter().enumerate() {
        println!(
            "  {}:\n    name = {}\n    type = {:?}\n    dimensions = {:?}",
            index, input.name, input.input_type, input.dimensions
        )
    }

    println!("Outputs:");
    for (index, output) in session.outputs.iter().enumerate() {
        println!(
            "  {}:\n    name = {}\n    type = {:?}\n    dimensions = {:?}",
            index, output.name, output.output_type, output.dimensions
        );
    }

    Ok(())
}
