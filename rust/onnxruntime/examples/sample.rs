#![forbid(unsafe_code)]

use onnxruntime::{environment::Environment, ndarray::Array, GraphOptimizationLevel, LoggingLevel};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

type Error = Box<dyn std::error::Error>;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    // Setup the example's log level.
    // NOTE: ONNX Runtime's log level is controlled separately when building the environment.
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let environment = Environment::builder()
        .with_name("test")
        // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
        .with_log_level(LoggingLevel::Info)
        .build()?;

    let session = environment
        .new_session_builder()?
        .with_graph_optimization_level(GraphOptimizationLevel::Basic)?
        .with_intra_op_num_threads(1)?
        // NOTE: The example uses SqueezeNet 1.0 (ONNX version: 1.3, Opset version: 8),
        //       _not_ SqueezeNet 1.1 as downloaded by '.with_model_downloaded(ImageClassification::SqueezeNet)'
        //       Obtain it with:
        //          curl -LO "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-8.onnx"
        .with_model_from_file("squeezenet1.0-8.onnx")?;

    let input0_shape: Vec<usize> = session.inputs[0]
        .dimensions()
        .map(std::option::Option::unwrap)
        .collect();
    let output0_shape: Vec<usize> = session.outputs[0]
        .dimensions()
        .map(std::option::Option::unwrap)
        .collect();

    assert_eq!(input0_shape, [1, 3, 224, 224]);
    assert_eq!(output0_shape, [1, 1000, 1, 1]);

    // initialize input data with values in [0.0, 1.0]
    let n: u32 = session.inputs[0]
        .dimensions
        .iter()
        .map(|d| d.unwrap())
        .product();
    let array = Array::linspace(0.0_f32, 1.0, n as usize)
        .into_shape(input0_shape)
        .unwrap();
    let input_tensor_values = vec![array.into()];

    let outputs = session.run(input_tensor_values)?;

    let output = outputs[0].float_array().unwrap();

    assert_eq!(output.shape(), output0_shape.as_slice());
    for i in 0..5 {
        println!("Score for class [{}] =  {}", i, output[[0, i, 0, 0]]);
    }

    Ok(())
}
