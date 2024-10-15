#![allow(clippy::upper_case_acronyms)]

//! Models for machine language comprehension.

use crate::download::ModelUrl;

/// Machine comprehension models.
///
/// A subset of natural language processing models that answer questions about a given context paragraph.
#[derive(Debug, Clone)]
pub enum MachineComprehension {
	/// Answers a query about a given context paragraph.
	BiDAF,
	/// Answers questions based on the context of the given input paragraph.
	BERTSquad,
	/// Large transformer-based model that predicts sentiment based on given input text.
	RoBERTa(RoBERTa),
	/// Generates synthetic text samples in response to the model being primed with an arbitrary input.
	GPT2(GPT2)
}

/// Large transformer-based model that predicts sentiment based on given input text.
#[derive(Debug, Clone)]
pub enum RoBERTa {
	/// Base RoBERTa model.
	RoBERTaBase,
	/// RoBERTa model for sequence classification.
	RoBERTaSequenceClassification
}

/// Generates synthetic text samples in response to the model being primed with an arbitrary input.
#[derive(Debug, Clone)]
pub enum GPT2 {
	/// Base GPT-2 model.
	GPT2,
	/// GPT-2 model with a causal LM head.
	GPT2LmHead
}

impl ModelUrl for MachineComprehension {
	fn fetch_url(&self) -> &'static str {
		match self {
			MachineComprehension::BiDAF => "https://github.com/onnx/models/raw/main/text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx",
			MachineComprehension::BERTSquad => "https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx",
			MachineComprehension::RoBERTa(variant) => variant.fetch_url(),
			MachineComprehension::GPT2(variant) => variant.fetch_url()
		}
	}
}

impl ModelUrl for RoBERTa {
	fn fetch_url(&self) -> &'static str {
		match self {
			RoBERTa::RoBERTaBase => "https://github.com/onnx/models/raw/main/text/machine_comprehension/roberta/model/roberta-base-11.onnx",
			RoBERTa::RoBERTaSequenceClassification => {
				"https://github.com/onnx/models/raw/main/text/machine_comprehension/roberta/model/roberta-sequence-classification-9.onnx"
			}
		}
	}
}

impl ModelUrl for GPT2 {
	fn fetch_url(&self) -> &'static str {
		match self {
			GPT2::GPT2 => "https://github.com/onnx/models/raw/main/text/machine_comprehension/gpt-2/model/gpt2-10.onnx",
			GPT2::GPT2LmHead => "https://github.com/onnx/models/raw/main/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx"
		}
	}
}
