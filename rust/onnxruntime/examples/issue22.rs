//! Example reproducing issue #22.
//!
//! `model.onnx` available to download here:
//! https://drive.google.com/file/d/1FmL-Wpm06V-8wgRqvV3Skey_X98Ue4D_/view?usp=sharing

use ndarray::Array2;
use onnxruntime::{environment::Environment, GraphOptimizationLevel, LoggingLevel};
use std::env::var;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

fn main() {
    // a builder for `FmtSubscriber`.
    let subscriber = FmtSubscriber::builder()
        // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
        // will be written to stdout.
        .with_max_level(Level::TRACE)
        // completes the builder.
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let path = var("RUST_ONNXRUNTIME_LIBRARY_PATH").ok();

    let builder = Environment::builder()
        .with_name("env")
        .with_log_level(LoggingLevel::Warning);

    let builder = if let Some(path) = path.clone() {
        builder.with_library_path(path)
    } else {
        builder
    };

    let env = builder.build().unwrap();
    let session = env
        .new_session_builder()
        .unwrap()
        .with_graph_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_model_from_file("model.onnx")
        .unwrap();

    println!("{:#?}", session.inputs);
    println!("{:#?}", session.outputs);

    let input_ids = Array2::<i64>::from_shape_vec((1, 3), vec![1, 2, 3]).unwrap();
    let attention_mask = Array2::<i64>::from_shape_vec((1, 3), vec![1, 1, 1]).unwrap();

    let inputs = vec![input_ids.into(), attention_mask.into()];

    let outputs = session.run(inputs).unwrap();

    print!("outputs: {:#?}", outputs[0].float_array().unwrap());
}
