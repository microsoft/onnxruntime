use std::{ffi, fmt::Debug, mem::MaybeUninit, ops::Deref};

use ndarray::Array;
use tracing::{debug, error};

use crate::{
	error::assert_non_null_pointer,
	memory::MemoryInfo,
	ortsys, sys,
	tensor::{ndarray_tensor::NdArrayTensor, IntoTensorElementDataType, TensorElementDataType},
	OrtError, OrtResult
};

/// Owned tensor, backed by an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
///
/// This tensor bounds the ONNX Runtime to `ndarray`; it is used to copy an
/// [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html) to the runtime's memory.
///
/// **NOTE**: The type is not meant to be used directly, use an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
/// instead.
#[derive(Debug)]
pub struct OrtTensor<'t, T, D>
where
	T: IntoTensorElementDataType + Debug + Clone,
	D: ndarray::Dimension
{
	pub(crate) c_ptr: *mut sys::OrtValue,
	array: Array<T, D>,
	#[allow(dead_code)]
	memory_info: &'t MemoryInfo
}

impl<'t, T, D> OrtTensor<'t, T, D>
where
	T: IntoTensorElementDataType + Debug + Clone,
	D: ndarray::Dimension
{
	pub(crate) fn from_array<'m>(memory_info: &'m MemoryInfo, allocator_ptr: *mut sys::OrtAllocator, array: &Array<T, D>) -> OrtResult<OrtTensor<'t, T, D>>
	where
		'm: 't // 'm outlives 't
	{
		// Ensure that the array is contiguous in memory.
		let mut contiguous_array: Array<MaybeUninit<T>, D> = Array::uninit(array.raw_dim());
		array.assign_to(&mut contiguous_array);
		let mut contiguous_array = unsafe { contiguous_array.assume_init() };

		// where onnxruntime will write the tensor data to
		let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
		let tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut tensor_ptr;

		let shape: Vec<i64> = contiguous_array.shape().iter().map(|d: &usize| *d as i64).collect();
		let shape_ptr: *const i64 = shape.as_ptr();
		let shape_len = contiguous_array.shape().len();

		match T::tensor_element_data_type() {
			TensorElementDataType::Float32
			| TensorElementDataType::Uint8
			| TensorElementDataType::Int8
			| TensorElementDataType::Uint16
			| TensorElementDataType::Int16
			| TensorElementDataType::Int32
			| TensorElementDataType::Int64
			| TensorElementDataType::Float64
			| TensorElementDataType::Uint32
			| TensorElementDataType::Uint64 => {
				// primitive data is already suitably laid out in memory; provide it to
				// onnxruntime as is
				let tensor_values_ptr: *mut std::ffi::c_void = contiguous_array.as_mut_ptr() as *mut std::ffi::c_void;
				assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;

				ortsys![
					unsafe CreateTensorWithDataAsOrtValue(
						memory_info.ptr,
						tensor_values_ptr,
						(contiguous_array.len() * std::mem::size_of::<T>()) as _,
						shape_ptr,
						shape_len as _,
						T::tensor_element_data_type().into(),
						tensor_ptr_ptr
					) -> OrtError::CreateTensorWithData;
					nonNull(tensor_ptr)
				];

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(tensor_ptr, &mut is_tensor) -> OrtError::FailedTensorCheck];
				assert_eq!(is_tensor, 1);
			}
			#[cfg(feature = "half")]
			TensorElementDataType::Bfloat16 | TensorElementDataType::Float16 => {
				// f16 and bf16 are repr(transparent) to u16, so memory layout should be identical to onnxruntime
				let tensor_values_ptr: *mut std::ffi::c_void = contiguous_array.as_mut_ptr() as *mut std::ffi::c_void;
				assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;

				ortsys![
					unsafe CreateTensorWithDataAsOrtValue(
						memory_info.ptr,
						tensor_values_ptr,
						(contiguous_array.len() * std::mem::size_of::<T>()) as _,
						shape_ptr,
						shape_len as _,
						T::tensor_element_data_type().into(),
						tensor_ptr_ptr
					) -> OrtError::CreateTensorWithData;
					nonNull(tensor_ptr)
				];

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(tensor_ptr, &mut is_tensor) -> OrtError::FailedTensorCheck];
				assert_eq!(is_tensor, 1);
			}
			TensorElementDataType::String => {
				// create tensor without data -- data is filled in later
				ortsys![
					unsafe CreateTensorAsOrtValue(allocator_ptr, shape_ptr, shape_len as _, T::tensor_element_data_type().into(), tensor_ptr_ptr)
						-> OrtError::CreateTensor
				];

				// create null-terminated copies of each string, as per `FillStringTensor` docs
				let null_terminated_copies: Vec<ffi::CString> = contiguous_array
					.iter()
					.map(|elt| {
						let slice = elt.try_utf8_bytes().expect("String data type must provide utf8 bytes");
						ffi::CString::new(slice)
					})
					.collect::<std::result::Result<Vec<_>, _>>()
					.map_err(OrtError::FfiStringNull)?;

				let string_pointers = null_terminated_copies.iter().map(|cstring| cstring.as_ptr()).collect::<Vec<_>>();

				ortsys![unsafe FillStringTensor(tensor_ptr, string_pointers.as_ptr(), string_pointers.len() as _) -> OrtError::FillStringTensor];
			}
			_ => unimplemented!("Tensor element data type {:?} not yet implemented", T::tensor_element_data_type())
		}

		assert_non_null_pointer(tensor_ptr, "Tensor")?;

		Ok(OrtTensor {
			c_ptr: tensor_ptr,
			array: contiguous_array,
			memory_info
		})
	}
}

impl<'t, T, D> Deref for OrtTensor<'t, T, D>
where
	T: IntoTensorElementDataType + Debug + Clone,
	D: ndarray::Dimension
{
	type Target = Array<T, D>;

	fn deref(&self) -> &Self::Target {
		&self.array
	}
}

impl<'t, T, D> Drop for OrtTensor<'t, T, D>
where
	T: IntoTensorElementDataType + Debug + Clone,
	D: ndarray::Dimension
{
	#[tracing::instrument]
	fn drop(&mut self) {
		// We need to let the C part free
		debug!("Dropping Tensor.");
		if self.c_ptr.is_null() {
			error!("Null pointer, not calling free.");
		} else {
			ortsys![unsafe ReleaseValue(self.c_ptr)];
		}

		self.c_ptr = std::ptr::null_mut();
	}
}

impl<'t, T, D> OrtTensor<'t, T, D>
where
	T: IntoTensorElementDataType + Debug + Clone,
	D: ndarray::Dimension
{
	/// Apply a softmax on the specified axis
	pub fn softmax(&self, axis: ndarray::Axis) -> Array<T, D>
	where
		D: ndarray::RemoveAxis,
		T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign
	{
		self.array.softmax(axis)
	}
}

#[cfg(test)]
mod tests {
	use std::ptr;

	use ndarray::{arr0, arr1, arr2, arr3};
	use test_log::test;

	use super::*;
	use crate::{AllocatorType, MemType};

	#[test]
	fn orttensor_from_array_0d_i32() -> OrtResult<()> {
		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default)?;
		let array = arr0::<i32>(123);
		let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), &array)?;
		let expected_shape: &[usize] = &[];
		assert_eq!(tensor.shape(), expected_shape);
		Ok(())
	}

	#[test]
	fn orttensor_from_array_1d_i32() -> OrtResult<()> {
		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default)?;
		let array = arr1(&[1_i32, 2, 3, 4, 5, 6]);
		let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), &array)?;
		let expected_shape: &[usize] = &[6];
		assert_eq!(tensor.shape(), expected_shape);
		Ok(())
	}

	#[test]
	fn orttensor_from_array_2d_i32() -> OrtResult<()> {
		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default)?;
		let array = arr2(&[[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]);
		let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), &array)?;
		assert_eq!(tensor.shape(), &[2, 6]);
		Ok(())
	}

	#[test]
	fn orttensor_from_array_3d_i32() -> OrtResult<()> {
		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default)?;
		let array = arr3(&[
			[[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
			[[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]],
			[[25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]]
		]);
		let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), &array)?;
		assert_eq!(tensor.shape(), &[3, 2, 6]);
		Ok(())
	}

	#[test]
	fn orttensor_from_array_1d_string() -> OrtResult<()> {
		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default)?;
		let array = arr1(&[String::from("foo"), String::from("bar"), String::from("baz")]);
		let tensor = OrtTensor::from_array(&memory_info, ort_default_allocator()?, &array)?;
		assert_eq!(tensor.shape(), &[3]);
		Ok(())
	}

	#[test]
	fn orttensor_from_array_3d_str() -> OrtResult<()> {
		let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default)?;
		let array = arr3(&[[["1", "2", "3"], ["4", "5", "6"]], [["7", "8", "9"], ["10", "11", "12"]]]);
		let tensor = OrtTensor::from_array(&memory_info, ort_default_allocator()?, &array)?;
		assert_eq!(tensor.shape(), &[2, 2, 3]);
		Ok(())
	}

	fn ort_default_allocator() -> OrtResult<*mut sys::OrtAllocator> {
		let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
		// this default non-arena allocator doesn't need to be deallocated
		ortsys![unsafe GetAllocatorWithDefaultOptions(&mut allocator_ptr) -> OrtError::GetAllocator];
		Ok(allocator_ptr)
	}
}
