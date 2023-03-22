#![allow(clippy::tabs_in_doc_comments)]

use std::{ffi::CString, os::raw::c_char};

use super::{char_p_to_string, error::OrtResult, ortfree, ortsys, sys, OrtError};

/// Container for model metadata, including name & producer information.
pub struct Metadata {
	metadata_ptr: *mut sys::OrtModelMetadata,
	allocator_ptr: *mut sys::OrtAllocator
}

impl Metadata {
	pub(crate) fn new(metadata_ptr: *mut sys::OrtModelMetadata, allocator_ptr: *mut sys::OrtAllocator) -> Self {
		Metadata { metadata_ptr, allocator_ptr }
	}

	/// Gets the model description, returning an error if no description is present.
	pub fn description(&self) -> OrtResult<String> {
		let mut str_bytes: *mut c_char = std::ptr::null_mut();
		ortsys![unsafe ModelMetadataGetDescription(self.metadata_ptr, self.allocator_ptr, &mut str_bytes) -> OrtError::GetModelMetadata; nonNull(str_bytes)];

		let value = char_p_to_string(str_bytes)?;
		ortfree!(unsafe self.allocator_ptr, str_bytes);
		Ok(value)
	}

	/// Gets the model producer name, returning an error if no producer name is present.
	pub fn producer(&self) -> OrtResult<String> {
		let mut str_bytes: *mut c_char = std::ptr::null_mut();
		ortsys![unsafe ModelMetadataGetProducerName(self.metadata_ptr, self.allocator_ptr, &mut str_bytes) -> OrtError::GetModelMetadata; nonNull(str_bytes)];

		let value = char_p_to_string(str_bytes)?;
		ortfree!(unsafe self.allocator_ptr, str_bytes);
		Ok(value)
	}

	/// Gets the model name, returning an error if no name is present.
	pub fn name(&self) -> OrtResult<String> {
		let mut str_bytes: *mut c_char = std::ptr::null_mut();
		ortsys![unsafe ModelMetadataGetGraphName(self.metadata_ptr, self.allocator_ptr, &mut str_bytes) -> OrtError::GetModelMetadata; nonNull(str_bytes)];

		let value = char_p_to_string(str_bytes)?;
		ortfree!(unsafe self.allocator_ptr, str_bytes);
		Ok(value)
	}

	/// Gets the model version, returning an error if no version is present.
	pub fn version(&self) -> OrtResult<i64> {
		let mut ver = 0i64;
		ortsys![unsafe ModelMetadataGetVersion(self.metadata_ptr, &mut ver) -> OrtError::GetModelMetadata];
		Ok(ver)
	}

	/// Fetch the value of a custom metadata key. Returns `Ok(None)` if the key is not found.
	pub fn custom(&self, key: &str) -> OrtResult<Option<String>> {
		let mut str_bytes: *mut c_char = std::ptr::null_mut();
		let key_str = CString::new(key)?;
		ortsys![unsafe ModelMetadataLookupCustomMetadataMap(self.metadata_ptr, self.allocator_ptr, key_str.as_ptr(), &mut str_bytes) -> OrtError::GetModelMetadata];
		if !str_bytes.is_null() {
			unsafe {
				let value = char_p_to_string(str_bytes)?;
				ortfree!(self.allocator_ptr, str_bytes);
				Ok(Some(value))
			}
		} else {
			Ok(None)
		}
	}
}

impl Drop for Metadata {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseModelMetadata(self.metadata_ptr)];
	}
}
