//! Metadata contains various information about the model.

use std::collections::HashMap;

use onnxruntime_sys as sys;

use crate::{
    ort_api,
    error::{OrtError, OrtResult, status_to_result},
    char_ptr_to_string, allocator::Allocator,
};

/// Container for various metadata provided alongside the model.
/// See the corresponding ModelMetadata* functions in the 
/// onnxruntime C API.
#[derive(Debug)]
pub struct Metadata {
    /// From onnxruntime API call: ModelMetadataGetGraphName
    pub graph_name: String,
    /// From onnxruntime API call: ModelMetadataGetDescription
    pub description: String,
    /// From onnxruntime API call: ModelMetadataGetGraphDescription
    pub graph_description: String,
    /// From onnxruntime API call: ModelMetadataGetVersion
    pub version: i64,
    /// From onnxruntime API call: ModelMetadataGetDomain
    pub domain: String,
    /// From onnxruntime API call: ModelMetadataGetProducerName
    pub producer_name: String,
    /// From onnxruntime API call: ModelMetadataGetCustomMetadataMapKeys / ModelMetadataLookupCustomMetadataMap
    pub custom: HashMap<String, String>,
}

#[allow(dead_code)]
enum MetadataField {
    GraphName, Description, GraphDescription, Version, Domain, ProducerName, Custom
}

impl Default for Metadata {
    fn default() -> Self {
        Self { graph_name: Default::default(), description: Default::default(), graph_description: Default::default(), version: Default::default(), producer_name: Default::default(), domain: Default::default(), custom: Default::default() }
    }
}

impl Metadata {
    pub(crate) fn new(session_ptr: *mut sys::OrtSession, allocator: &Allocator) -> OrtResult<Metadata> {
        let mut metadata = Self::default();

        // access the metadata
        let mut model_metadata: *mut sys::OrtModelMetadata = std::ptr::null_mut();
        let status = unsafe {
            ort_api().SessionGetModelMetadata.unwrap()(session_ptr, &mut model_metadata)
        };
        status_to_result(status).map_err(OrtError::MetadataFailure)?;

        let f_graph_name = ort_api().ModelMetadataGetGraphName.unwrap();
        let f_description = ort_api().ModelMetadataGetDescription.unwrap();
        let f_graph_description = ort_api().ModelMetadataGetGraphDescription.unwrap();
        let f_domain = ort_api().ModelMetadataGetDomain.unwrap();
        let f_producer_name = ort_api().ModelMetadataGetProducerName.unwrap();

        for (field_type, f) in [
            (MetadataField::GraphName, f_graph_name), 
            (MetadataField::Description, f_description), 
            (MetadataField::GraphDescription, f_graph_description), 
            (MetadataField::Domain, f_domain), 
            (MetadataField::ProducerName, f_producer_name) 
        ] {
            let mut cstr: *mut std::os::raw::c_char = std::ptr::null_mut();
            let status = unsafe {
                f(model_metadata,
                    allocator.ptr,
                    &mut cstr)
            };
            status_to_result(status).map_err(OrtError::MetadataFailure)?;

            let value = char_ptr_to_string(cstr)?;
            match field_type {
                MetadataField::GraphName => metadata.graph_name = value,
                MetadataField::Description => metadata.description = value,
                MetadataField::GraphDescription => metadata.graph_description = value,
                MetadataField::Domain => metadata.domain = value,
                MetadataField::ProducerName => metadata.producer_name = value,
                _ => unimplemented!(),
            };
    
            unsafe { ort_api().AllocatorFree.unwrap()(
                allocator.ptr,
                cstr as *mut std::ffi::c_void) 
            };
        }

        unsafe { 
            ort_api().ModelMetadataGetVersion.unwrap()(model_metadata, &mut metadata.version); 
        }
        status_to_result(status).map_err(OrtError::MetadataFailure)?;

        // load the custom metadata into another HashMap
        let mut keys: *mut *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut num_keys: i64 = 0;
        let status = unsafe {
            ort_api().ModelMetadataGetCustomMetadataMapKeys.unwrap()(
                model_metadata,
                allocator.ptr,
                &mut keys,
                &mut num_keys)
        };
        status_to_result(status).map_err(OrtError::MetadataFailure)?;
        let key_table = unsafe { std::slice::from_raw_parts(keys, num_keys as usize) };
        for idx in 0..(num_keys as usize) {
            let key = char_ptr_to_string(key_table[idx])?;
            let mut val_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
            let status = unsafe {
                ort_api().ModelMetadataLookupCustomMetadataMap.unwrap()(
                    model_metadata,
                    allocator.ptr,
                    key_table[idx],
                    &mut val_ptr)
            };
            status_to_result(status).map_err(OrtError::MetadataFailure)?;
            let value = char_ptr_to_string(val_ptr)?;
            metadata.custom.insert(key, value);
            unsafe { ort_api().AllocatorFree.unwrap()(
                allocator.ptr,
                val_ptr as *mut std::ffi::c_void ) };
            unsafe { ort_api().AllocatorFree.unwrap()(
                allocator.ptr,
                key_table[idx] as *mut std::ffi::c_void) };
        }
        unsafe { ort_api().AllocatorFree.unwrap()(
            allocator.ptr,
            keys as *mut std::ffi::c_void) };

        // release metatdata object
        unsafe { ort_api().ReleaseModelMetadata.unwrap()(model_metadata); }

        Ok( metadata )
    }
}

