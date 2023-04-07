//! Module controlling models downloadable from ONNX Model Zoom
//!
//! Pre-trained models are available from the
//! [ONNX Model Zoo](https://github.com/onnx/models).
//!
//! A pre-trained model can be downloaded automatically using the
//! [`SessionBuilder`](../session/struct.SessionBuilder.html)'s
//! [`with_model_downloaded()`](../session/struct.SessionBuilder.html#method.with_model_downloaded) method.
//!
//! See [`AvailableOnnxModel`](enum.AvailableOnnxModel.html) for the different models available
//! to download.

#[cfg(feature = "model-fetching")]
use std::{
    fs, io,
    path::{Path, PathBuf},
    time::Duration,
};

#[cfg(feature = "model-fetching")]
use crate::error::{OrtDownloadError, OrtResult};

#[cfg(feature = "model-fetching")]
use tracing::info;

pub mod language;
pub mod vision;

/// Available pre-trained models to download from [ONNX Model Zoo](https://github.com/onnx/models).
///
/// According to [ONNX Model Zoo](https://github.com/onnx/models)'s GitHub page:
///
/// > The ONNX Model Zoo is a collection of pre-trained, state-of-the-art models in the ONNX format
/// > contributed by community members like you.
#[derive(Debug, Clone)]
pub enum AvailableOnnxModel {
    /// Computer vision model
    Vision(vision::Vision),
    /// Natural language model
    Language(language::Language),
}

trait ModelUrl {
    fn fetch_url(&self) -> &'static str;
}

impl ModelUrl for AvailableOnnxModel {
    fn fetch_url(&self) -> &'static str {
        match self {
            AvailableOnnxModel::Vision(model) => model.fetch_url(),
            AvailableOnnxModel::Language(model) => model.fetch_url(),
        }
    }
}

impl AvailableOnnxModel {
    #[cfg(feature = "model-fetching")]
    #[tracing::instrument]
    pub(crate) fn download_to<P>(&self, download_dir: P) -> Result<PathBuf>
    where
        P: AsRef<Path> + std::fmt::Debug,
    {
        let url = self.fetch_url();

        let model_filename = PathBuf::from(url.split('/').last().unwrap());
        let model_filepath = download_dir.as_ref().join(model_filename);

        if model_filepath.exists() {
            info!(
                model_filepath = format!("{}", model_filepath.display()).as_str(),
                "File already exists, not re-downloading.",
            );
            Ok(model_filepath)
        } else {
            info!(
                model_filepath = format!("{}", model_filepath.display()).as_str(),
                url = format!("{:?}", url).as_str(),
                "Downloading file, please wait....",
            );

            let resp = ureq::get(url)
                .timeout(Duration::from_secs(180)) // 3 minutes
                .call()
                .map_err(Box::new)
                .map_err(OrtDownloadError::UreqError)?;

            assert!(resp.has("Content-Length"));
            let len = resp
                .header("Content-Length")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap();
            info!(len, "Downloading {} bytes...", len);

            let mut reader = resp.into_reader();

            let f = fs::File::create(&model_filepath).unwrap();
            let mut writer = io::BufWriter::new(f);

            let bytes_io_count =
                io::copy(&mut reader, &mut writer).map_err(OrtDownloadError::IoError)?;

            if bytes_io_count == len as u64 {
                Ok(model_filepath)
            } else {
                Err(OrtDownloadError::CopyError {
                    expected: len as u64,
                    io: bytes_io_count,
                }
                .into())
            }
        }
    }
}
