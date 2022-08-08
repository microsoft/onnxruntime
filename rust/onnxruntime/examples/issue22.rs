//! Example reproducing issue #22.
//!
//! `model.onnx` available to download here:
//! https://drive.google.com/file/d/1FmL-Wpm06V-8wgRqvV3Skey_X98Ue4D_/view?usp=sharing

use ndarray::Array2;
use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor, GraphOptimizationLevel};
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

    let env = Environment::builder().with_name("env").build().unwrap();
    let session = env
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .with_model_from_file("model.onnx")
        .unwrap();

    println!("{:#?}", session.inputs);
    println!("{:#?}", session.outputs);

    let input_ids = Array2::<i64>::from_shape_vec((1, 3), vec![1, 2, 3]).unwrap();
    let attention_mask = Array2::<i64>::from_shape_vec((1, 3), vec![1, 1, 1]).unwrap();

    let outputs: Vec<OrtOwnedTensor<f32, _>> =
        session.run(vec![input_ids, attention_mask]).unwrap();
    print!("outputs: {:#?}", outputs);
}
