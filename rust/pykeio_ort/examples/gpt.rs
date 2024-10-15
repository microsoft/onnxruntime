use std::sync::Arc;

use ndarray::{array, concatenate, s, Array1, Axis};
use ort::{
	download::language::machine_comprehension::GPT2,
	tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
	Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder
};
use rand::Rng;
use tokenizers::Tokenizer;

const GEN_TOKENS: i32 = 45;
const TOP_K: usize = 5;

fn main() -> OrtResult<()> {
	let mut rng = rand::thread_rng();

	let environment = Arc::new(
		Environment::builder()
			.with_name("GPT-2")
			.with_execution_providers([ExecutionProvider::cuda()])
			.build()?
	);

	let session = SessionBuilder::new(&environment)?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.with_model_downloaded(GPT2::GPT2LmHead)?;

	let tokenizer = Tokenizer::from_file("tests/data/gpt2-tokenizer.json").unwrap();
	let tokens = tokenizer
		.encode("The corsac fox (Vulpes corsac), also known simply as a corsac, is a medium-sized fox found in", false)
		.unwrap();
	let tokens = tokens.get_ids().iter().map(|i| *i as i64).collect::<Vec<_>>();

	let tokens = &mut Array1::from_iter(tokens.iter().cloned());

	for _ in 0..GEN_TOKENS {
		let n_tokens = &tokens.shape()[0];
		let array = tokens.clone().insert_axis(Axis(0)).into_shape((1, 1, *n_tokens)).unwrap();
		let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> = session.run([InputTensor::from_array(array.into_dyn())])?;
		let generated_tokens: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
		let generated_tokens = generated_tokens.view();

		let probabilities = &mut generated_tokens
			.slice(s![0, 0, -1, ..])
			.insert_axis(Axis(0))
			.to_owned()
			.iter()
			.cloned()
			.enumerate()
			.collect::<Vec<_>>();
		probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

		let token = probabilities[rng.gen_range(0..=TOP_K)].0;
		*tokens = concatenate![Axis(0), *tokens, array![token.try_into().unwrap()]];
		let sentence = tokenizer.decode(tokens.iter().map(|i| *i as u32).collect::<Vec<_>>(), true).unwrap();
		println!("{}", sentence);
	}

	Ok(())
}
