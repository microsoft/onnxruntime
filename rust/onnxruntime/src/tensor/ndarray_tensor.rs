//! Module containing a tensor trait extending [`ndarray::ArrayBase`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html)

use ndarray::{Array, ArrayBase};

/// Trait extending [`ndarray::ArrayBase`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html)
/// with useful tensor operations.
///
/// # Generic
///
/// The trait is generic over:
/// * `S`: [`ndarray::ArrayBase`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html)'s data container
/// * `T`: Type contained inside the tensor (for example `f32`)
/// * `D`: Tensor's dimension ([`ndarray::Dimension`](https://docs.rs/ndarray/latest/ndarray/trait.Dimension.html))
pub trait NdArrayTensor<S, T, D> {
    /// Calculate the [softmax](https://en.wikipedia.org/wiki/Softmax_function) of the tensor along a given axis
    ///
    /// # Trait Bounds
    ///
    /// The function is generic and thus has some trait bounds:
    /// * `D: ndarray::RemoveAxis`: The summation over an axis reduces the dimension of the tensor. A 0-D tensor thus
    ///   cannot have a softmax calculated.
    /// * `S: ndarray::RawData + ndarray::Data + ndarray::RawData<Elem = T>`: The storage of the tensor can be an owned
    ///   array ([`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)) or an array view
    ///   ([`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html)).
    /// * `<S as ndarray::RawData>::Elem: std::clone::Clone`: The elements of the tensor must be `Clone`.
    /// * `T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign`: The elements of the tensor must be workable
    ///   as floats and must support `-=` and `/=` operations.
    fn softmax(&self, axis: ndarray::Axis) -> Array<T, D>
    where
        D: ndarray::RemoveAxis,
        S: ndarray::Data + ndarray::RawData<Elem = T>,
        <S as ndarray::RawData>::Elem: std::clone::Clone,
        T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign;
}

impl<S, T, D> NdArrayTensor<S, T, D> for ArrayBase<S, D>
where
    D: ndarray::RemoveAxis,
    S: ndarray::Data + ndarray::RawData<Elem = T>,
    <S as ndarray::RawData>::Elem: std::clone::Clone,
    T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign,
{
    fn softmax(&self, axis: ndarray::Axis) -> Array<T, D> {
        let mut new_array: Array<T, D> = self.to_owned();
        // FIXME: Change to non-overflowing formula
        // e = np.exp(A - np.sum(A, axis=1, keepdims=True))
        // np.exp(a) / np.sum(np.exp(a))
        new_array.map_inplace(|v| *v = v.exp());
        let sum = new_array.sum_axis(axis).insert_axis(axis);
        new_array /= &sum;

        new_array
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, arr3};
    use test_log::test;

    #[test]
    fn softmax_1d() {
        let array = arr1(&[1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]);

        let expected_softmax = arr1(&[
            0.023_640_54,
            0.064_261_66,
            0.174_681_3,
            0.474_833,
            0.023_640_54,
            0.064_261_66,
            0.174_681_3,
        ]);

        let softmax = array.softmax(ndarray::Axis(0));

        assert_eq!(softmax.shape(), expected_softmax.shape());

        let diff = softmax - expected_softmax;

        assert!(diff.iter().all(|d| d.abs() < 1.0e-7));
    }

    #[test]
    fn softmax_2d() {
        let array = arr2(&[
            [1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
            [1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
        ]);

        let expected_softmax = arr2(&[
            [
                0.023_640_54,
                0.064_261_66,
                0.174_681_3,
                0.474_833,
                0.023_640_54,
                0.064_261_66,
                0.174_681_3,
            ],
            [
                0.023_640_54,
                0.064_261_66,
                0.174_681_3,
                0.474_833,
                0.023_640_54,
                0.064_261_66,
                0.174_681_3,
            ],
        ]);

        let softmax = array.softmax(ndarray::Axis(1));

        assert_eq!(softmax.shape(), expected_softmax.shape());

        let diff = softmax - expected_softmax;

        assert!(diff.iter().all(|d| d.abs() < 1.0e-7));
    }

    #[test]
    fn softmax_3d() {
        let array = arr3(&[
            [
                [1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
                [1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
            ],
            [
                [1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
                [1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
            ],
            [
                [1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
                [1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
            ],
        ]);

        let expected_softmax = arr3(&[
            [
                [
                    0.023_640_54,
                    0.064_261_66,
                    0.174_681_3,
                    0.474_833,
                    0.023_640_54,
                    0.064_261_66,
                    0.174_681_3,
                ],
                [
                    0.023_640_54,
                    0.064_261_66,
                    0.174_681_3,
                    0.474_833,
                    0.023_640_54,
                    0.064_261_66,
                    0.174_681_3,
                ],
            ],
            [
                [
                    0.023_640_54,
                    0.064_261_66,
                    0.174_681_3,
                    0.474_833,
                    0.023_640_54,
                    0.064_261_66,
                    0.174_681_3,
                ],
                [
                    0.023_640_54,
                    0.064_261_66,
                    0.174_681_3,
                    0.474_833,
                    0.023_640_54,
                    0.064_261_66,
                    0.174_681_3,
                ],
            ],
            [
                [
                    0.023_640_54,
                    0.064_261_66,
                    0.174_681_3,
                    0.474_833,
                    0.023_640_54,
                    0.064_261_66,
                    0.174_681_3,
                ],
                [
                    0.023_640_54,
                    0.064_261_66,
                    0.174_681_3,
                    0.474_833,
                    0.023_640_54,
                    0.064_261_66,
                    0.174_681_3,
                ],
            ],
        ]);

        let softmax = array.softmax(ndarray::Axis(2));

        assert_eq!(softmax.shape(), expected_softmax.shape());

        let diff = softmax - expected_softmax;

        assert!(diff.iter().all(|d| d.abs() < 1.0e-7));
    }
}
