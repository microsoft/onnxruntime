//! OrtNdArray is a container of an ndarray::Array
//! and its corresponding OrtValue. When the underlying 
//! allocation to the array changes, the OrtValue will be 
//! regenerated when a new reference to it is requested.


use std::collections::HashMap;

use onnxruntime_sys as sys;

use ndarray::{Array, ArrayBase, Data, Dimension};

use crate::OrtError;
use crate::TensorElementDataType;
use crate::TypeToTensorElementDataType;
use crate::OrtValue;
use crate::environment::ort_api;
use crate::error::{OrtResult, status_to_result};
use crate::session::Session;

/// A container of an ndarray::Array (owned data) that auto-manages
/// the lifetime of a corresponding OrtValue, which can be provided
/// as input to run a graph.
#[derive(Debug)]
pub struct OrtNdArray<T, D>
where
    T: TypeToTensorElementDataType,
    D: Dimension,
{
    /// ndarray::Array whose memory OrtValue will reference
    pub array: Array<T, D>,
    array_data_ptr: *mut std::ffi::c_void,
    array_data_len: usize,
    ort_value: OrtValue,
}

unsafe impl<T, D> Send for OrtNdArray<T, D> 
where
    T: TypeToTensorElementDataType,
    D: Dimension,
{}
unsafe impl<T, D> Sync for OrtNdArray<T, D> 
where
    T: TypeToTensorElementDataType,
    D: Dimension,
{}

impl<T, D> OrtNdArray<T, D>
where
    T: TypeToTensorElementDataType,
    D: Dimension,
{
    /// Constructor
    pub fn new(array: Array<T, D>) -> OrtNdArray<T, D> {
        OrtNdArray { 
            array, 
            array_data_ptr: std::ptr::null_mut(),
            array_data_len: 0,
            ort_value: OrtValue::default(),
        }
    }

    /// Create or return an existing OrtValue for the array.
    /// If the location of the data in the array has changed, or its size has changed,
    /// then a new OrtValue will be created.
    pub fn get_ort_value<'a>(&'a mut self, session: &Session) -> OrtResult<&'a OrtValue> {
        assert!(self.array.is_standard_layout());
        let array_data_ptr = self.array.as_ptr() as *mut std::ffi::c_void;
        assert!(array_data_ptr != std::ptr::null_mut());
        let array_data_len = self.array.len();

        if array_data_ptr != self.array_data_ptr || array_data_len != self.array_data_len {
            // where onnxruntime will put the sys::OrtValue
            let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();

            // shapes
            let shape: Vec<i64> = self.array.shape().iter().map(|d: &usize| *d as i64).collect();
            let shape_ptr = shape.as_ptr();
            let shape_len = shape.len();

            match T::tensor_element_data_type() {
                TensorElementDataType::Float
                | TensorElementDataType::Uint8
                | TensorElementDataType::Int8
                | TensorElementDataType::Uint16
                | TensorElementDataType::Int16
                | TensorElementDataType::Int32
                | TensorElementDataType::Int64
                | TensorElementDataType::Double
                | TensorElementDataType::Uint32
                | TensorElementDataType::Uint64 => {
                    // primitive data is already suitably laid out in memory; provide it to
                    // onnxruntime as is

                    let status = unsafe {
                        ort_api().CreateTensorWithDataAsOrtValue.unwrap()(
                            session.memory_info.ptr,
                            array_data_ptr,
                            array_data_len * std::mem::size_of::<T>(),
                            shape_ptr,
                            shape_len,
                            T::tensor_element_data_type().into(),
                            &mut tensor_ptr,
                        )
                    };
                    status_to_result(status).map_err(OrtError::IsTensor)?;
                }
                _ => todo!(),
            }

            // Drop any prior sys::OrtValue and create a new one
            self.ort_value = OrtValue::new(tensor_ptr);
        }

        Ok( &self.ort_value )
    }
}

/// Given an iterator over tensor names and a corresponding iterator over OrtNdArrays,
/// Construct the input to Session's run().
/// For the common case where all inputs are provided:
/// ```ignore
/// // These only need to be created once.
/// let Session: Arc<Session> = ...;
/// let input_arrays: HashMap<String, OrtNdArray<_,_>> = ...;
/// let output_names: Vec<String> = session.outputs.keys().map(|name| name.clone()).collect();
/// 
/// // For each call to session.run()
/// let inputs: HashMap<String, &OrtValue> = 
///     onnxruntime::create_graph_inputs(&session, input_arrays)?;
/// 
/// let outputs = session.run(inputs, output_names.as_slice())?;
/// ```
pub fn create_graph_inputs<'a, I, T, D>(session: &Session, inputs_iter: I) 
-> OrtResult<HashMap<&'a String, &'a OrtValue>>
where
    I: IntoIterator<Item = (&'a String, &'a mut OrtNdArray<T, D>)>,
    T: TypeToTensorElementDataType + 'a,
    D: Dimension + 'a,
{
    inputs_iter.into_iter()
        .map(|(name, ort_ndarray)| {
            ort_ndarray.get_ort_value(&session)
                .map(|ort_value| (name, ort_value))
        })
        .collect::<OrtResult<HashMap<&'a String, &'a OrtValue>>>()
}


/// Create an OrtValue with the shape of the array that owns its own data. 
/// Then copy the contents of array into the OrtValue.
/// This is an alternative to using an OrtNdArray, where the data memory
/// is owned by the ndarray::Array.
pub fn copy_array_to_ort_value<T, S, Dim>(session: &Session, array: &ArrayBase<S,Dim>) -> OrtResult<OrtValue>
where
    T: TypeToTensorElementDataType,
    S: Data<Elem = T>,
    Dim: Dimension,
{
    let shape: Vec<i64> = array.shape().iter().map(|d: &usize| *d as i64).collect();

    let (ort_value, tensor_data_ptr) = OrtValue::new_from_type_and_shape::<T>(&session, &shape)?;

    unsafe { std::ptr::copy_nonoverlapping(array.as_ptr(), tensor_data_ptr, array.len()); }

    Ok( ort_value )
}
