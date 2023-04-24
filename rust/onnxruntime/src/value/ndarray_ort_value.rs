//! Module for NdArrayOrtValue<T,D>
use std::fmt::Debug;

use ndarray::{Array};

use crate::{
    error::{OrtResult},
    AsOrtValue, OrtValue, Session, TypeToTensorElementDataType,
};

/// A transitive conversion of an ndarray::Array into an OrtValue
/// that borrows its data from the Array. The Array remains borrowed
/// for the lifetime of the OrtValue, preventing changes to the
/// Array that could invalidate its data memory.
/// 
/// NdArrayOrtValue can be used with Session::run_with_arrays():
/// ```ignore
/// let inputs: HashMap<String, Box<dyn AsOrtValue>> = HashMap::new();
/// inputs.insert("features".to_string(), NdArrayOrtValue::try_boxed_from(session, features)?);
/// let output_names = vec!["class_probs".to_string()];
/// let outputs = sesssion.run_with_arrays(inputs, outputs_names);
/// ```
/// 
/// If the same shape is to be repeatedly used, MutableOrtValue / 
/// MutableOrtValueTyped are more efficient.
#[derive(Debug)]
pub struct NdArrayOrtValue<'a, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    _array: &'a Array<T, D>,
    ort_value: OrtValue,
}

impl<'a, T, D> NdArrayOrtValue<'a, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    /// Try to convert the Array into an OrtValue that borrows the data
    /// memory from the Array.
    //pub fn try_from(session: &Session, array: &'a Array<T, D>) -> OrtResult<Box<dyn AsOrtValue + 'a>> {
    pub fn try_from(session: &Session, array: &'a Array<T, D>) -> OrtResult<NdArrayOrtValue<'a, T, D>> {
        let ort_value = OrtValue::try_from_array(session, array)?;
        Ok( Self { _array: array, ort_value } )
    }

    /// Like try_from, but boxes the NdArrayOrtValue as a dyn AsOortValue, suitable for use
    /// with session::run_with_arrays().
    pub fn try_boxed_from(session: &Session, array: &'a Array<T, D>) -> OrtResult<Box<dyn AsOrtValue + 'a>> {
        Ok( Self::try_from(session, array)?.into() )
    }
}

impl<'a, T, D> AsOrtValue for NdArrayOrtValue<'a, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    fn as_ort_value(&self) -> &OrtValue {
        &self.ort_value
    }
}

impl<'a, 'v, T, D> From<&'a NdArrayOrtValue<'v, T, D>> for &'a OrtValue
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
    'v : 'a
{
    fn from(mut_ort_value: &'a NdArrayOrtValue<'v, T, D>) -> Self {
        &mut_ort_value.ort_value
    }
}

impl<'a, T, D> From<NdArrayOrtValue<'a, T, D>> for Box<dyn AsOrtValue + 'a>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    fn from(ndarray_ort_value: NdArrayOrtValue<'a, T, D>) -> Self {
        Box::new(ndarray_ort_value)
    }
}

