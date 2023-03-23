//! Module defining natural language models available to download.
//!
//! See [https://github.com/onnx/models#machine_comprehension](https://github.com/onnx/models#machine_comprehension).

use super::ModelUrl;

pub mod machine_comprehension;

// Re-exports
pub use machine_comprehension::MachineComprehension;

/// Natural language models
#[derive(Debug, Clone)]
pub enum Language {
    /// Machine comprehension
    MachineComprehension(MachineComprehension),
}

impl ModelUrl for Language {
    fn fetch_url(&self) -> &'static str {
        match self {
            Language::MachineComprehension(variant) => variant.fetch_url(),
        }
    }
}
