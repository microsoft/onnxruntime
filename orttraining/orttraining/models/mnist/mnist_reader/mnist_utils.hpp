//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains utility functions to manipulate the MNIST dataset
 */

#ifndef MNIST_UTILS_HPP
#define MNIST_UTILS_HPP

#include <cmath>

namespace mnist {

/*!
 * \brief Binarize each sub range inside the given range
 * \param values The collection of ranges to binarize
 * \param threshold The threshold for binarization
 */
template <typename Container>
void binarize_each(Container& values, double threshold = 30.0) {
    for (auto& vec : values) {
        for (auto& v : vec) {
            v = v > threshold ? 1.0 : 0.0;
        }
    }
}

/*!
 * \brief Return the mean value of the elements inside the given range
 * \param container The range to compute the average from
 * \return The average value of the range
 */
template <typename Container>
double mean(const Container& container) {
    double mean = 0.0;
    for (auto& value : container) {
        mean += value;
    }
    return mean / container.size();
}

/*!
 * \brief Return the standard deviation of the elements inside the given range
 * \param container The range to compute the standard deviation from
 * \param mean The mean of the given range
 * \return The standard deviation of the range
 */
template <typename Container>
double stddev(const Container& container, double mean) {
    double std = 0.0;
    for (auto& value : container) {
        std += (value - mean) * (value - mean);
    }
    return std::sqrt(std / container.size());
}

/*!
 * \brief Normalize each sub range inside the given range
 * \param values The collection of ranges to normalize
 */
template <typename Container>
void normalize_each(Container& values) {
    for (auto& vec : values) {
        //zero-mean
        auto m = mnist::mean(vec);
        for (auto& v : vec) {
            v -= m;
        }
        //unit variance
        auto s = mnist::stddev(vec, 0.0);
        for (auto& v : vec) {
            v /= s;
        }
    }
}

/*!
 * \brief Binarize the given MNIST dataset
 * \param dataset The dataset to binarize
 */
template <typename Dataset>
void binarize_dataset(Dataset& dataset) {
    mnist::binarize_each(dataset.training_images);
    mnist::binarize_each(dataset.test_images);
}

/*!
 * \brief Normalize the given MNIST dataset to zero-mean and unit variance
 * \param dataset The dataset to normalize
 */
template <typename Dataset>
void normalize_dataset(Dataset& dataset) {
    mnist::normalize_each(dataset.training_images);
    mnist::normalize_each(dataset.test_images);
}

} //end of namespace mnist

#endif
