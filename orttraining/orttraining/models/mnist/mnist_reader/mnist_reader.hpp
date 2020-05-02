//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains functions to read the MNIST dataset
 */

#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

#include "mnist_reader_common.hpp"

namespace mnist {

/*!
 * \brief Represents a complete mnist dataset
 * \tparam Container The container to use
 * \tparam Image The type of image
 * \tparam Label The type of label
 */
template <template <typename...> class Container, typename Image, typename Label>
struct MNIST_dataset {
  Container<Image> training_images;  ///< The training images
  Container<Image> test_images;      ///< The test images
  Container<Label> training_labels;  ///< The training labels
  Container<Label> test_labels;      ///< The test labels

  /*!
     * \brief Resize the training set to new_size
     *
     * If new_size is less than the current size, this function has no effect.
     *
     * \param new_size The size to resize the training sets to.
     */
  void resize_training(std::size_t new_size) {
    if (training_images.size() > new_size) {
      training_images.resize(new_size);
      training_labels.resize(new_size);
    }
  }

  /*!
     * \brief Resize the test set to new_size
     *
     * If new_size is less than the current size, this function has no effect.
     *
     * \param new_size The size to resize the test sets to.
     */
  void resize_test(std::size_t new_size) {
    if (test_images.size() > new_size) {
      test_images.resize(new_size);
      test_labels.resize(new_size);
    }
  }
};

/*!
 * \brief Read a MNIST image file inside the given flat container (ETL)
 * \param images The container to fill with the images
 * \param path The path to the image file
 * \param limit The maximum number of elements to read (0: no limit)
 * \param start The elements to ignore at the beginning
 * \param func The functor to create the image object
 */
template <typename Container>
bool read_mnist_image_file_flat(Container& images, const std::string& path, std::size_t limit, std::size_t start = 0) {
  auto buffer = read_mnist_file(path, 0x803);

  if (buffer) {
    auto count = read_header(buffer, 1);
    auto rows = read_header(buffer, 2);
    auto columns = read_header(buffer, 3);

    //Skip the header
    //Cast to unsigned char is necessary cause signedness of char is
    //platform-specific
    auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);

    if (limit > 0 && count > limit) {
      count = static_cast<unsigned int>(limit);
    }

    // Ignore "start" first elements
    image_buffer += start * (rows * columns);

    for (size_t i = 0; i < count; ++i) {
      for (size_t j = 0; j < rows * columns; ++j) {
        images(i)[j] = *image_buffer++;
      }
    }

    return true;
  } else {
    return false;
  }
}

/*!
 * \brief Read a MNIST image file inside the given container
 * \param images The container to fill with the images
 * \param path The path to the image file
 * \param limit The maximum number of elements to read (0: no limit)
 * \param func The functor to create the image object
 */
template <template <typename...> class Container = std::vector, typename Image, typename Functor>
void read_mnist_image_file(Container<Image>& images, const std::string& path, std::size_t limit, Functor func) {
  auto buffer = read_mnist_file(path, 0x803);

  if (buffer) {
    auto count = read_header(buffer, 1);
    auto rows = read_header(buffer, 2);
    auto columns = read_header(buffer, 3);

    //Skip the header
    //Cast to unsigned char is necessary cause signedness of char is
    //platform-specific
    auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);

    if (limit > 0 && count > limit) {
      count = static_cast<unsigned int>(limit);
    }

    images.reserve(count);

    for (size_t i = 0; i < count; ++i) {
      images.push_back(func());

      for (size_t j = 0; j < rows * columns; ++j) {
        auto pixel = *image_buffer++;
        images[i][j] = static_cast<typename Image::value_type>(pixel);
      }
    }
  }
}

/*!
 * \brief Read a MNIST label file inside the given container
 * \param labels The container to fill with the labels
 * \param path The path to the label file
 * \param limit The maximum number of elements to read (0: no limit)
 */
template <template <typename...> class Container = std::vector, typename Label = uint8_t>
void read_mnist_label_file(Container<Label>& labels, const std::string& path, std::size_t limit = 0) {
  auto buffer = read_mnist_file(path, 0x801);

  if (buffer) {
    auto count = read_header(buffer, 1);

    //Skip the header
    //Cast to unsigned char is necessary cause signedness of char is
    //platform-specific
    auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

    if (limit > 0 && count > limit) {
      count = static_cast<unsigned int>(limit);
    }

    labels.resize(count);

    for (size_t i = 0; i < count; ++i) {
      auto label = *label_buffer++;
      labels[i] = static_cast<Label>(label);
    }
  }
}

/*!
 * \brief Read a MNIST label file inside the given flat container (ETL).
 * \param labels The container to fill with the labels
 * \param path The path to the label file
 * \param limit The maximum number of elements to read (0: no limit)
 */
template <typename Container>
bool read_mnist_label_file_flat(Container& labels, const std::string& path, std::size_t limit = 0) {
  auto buffer = read_mnist_file(path, 0x801);

  if (buffer) {
    auto count = read_header(buffer, 1);

    //Skip the header
    //Cast to unsigned char is necessary cause signedness of char is
    //platform-specific
    auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

    if (limit > 0 && count > limit) {
      count = static_cast<unsigned int>(limit);
    }

    for (size_t i = 0; i < count; ++i) {
      labels(i) = *label_buffer++;
    }

    return true;
  } else {
    return false;
  }
}

/*!
 * \brief Read a MNIST label file inside the given flat categorical container (ETL).
 * \param labels The container to fill with the labels
 * \param path The path to the label file
 * \param limit The maximum number of elements to read (0: no limit)
 * \param start The elements to avoid at the beginning
 */
template <typename Container>
bool read_mnist_label_file_categorical(Container& labels, const std::string& path, std::size_t limit = 0, std::size_t start = 0) {
  auto buffer = read_mnist_file(path, 0x801);

  if (buffer) {
    auto count = read_header(buffer, 1);

    //Skip the header
    //Cast to unsigned char is necessary cause signedness of char is
    //platform-specific
    auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

    if (limit > 0 && count > limit) {
      count = static_cast<unsigned int>(limit);
    }

    // Ignore "start" first elements
    label_buffer += start;

    for (size_t i = 0; i < count; ++i) {
      labels(i)(static_cast<size_t>(*label_buffer++)) = 1;
    }

    return true;
  } else {
    return false;
  }
}

/*!
 * \brief Read all training images and return a container filled with the images.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \param limit The maximum number of elements to read (0: no limit)
 * \param func The functor to create the image objects.
 * \return Container filled with the images
 */
template <template <typename...> class Container = std::vector, typename Image, typename Functor>
Container<Image> read_training_images(const std::string& folder, std::size_t limit, Functor func) {
  Container<Image> images;
  read_mnist_image_file<Container, Image>(images, folder + "/train-images.idx3-ubyte", limit, func);
  return images;
}

/*!
 * \brief Read all test images and return a container filled with the images.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \param limit The maximum number of elements to read (0: no limit)
 * \param func The functor to create the image objects.
 * \return Container filled with the images
 */
template <template <typename...> class Container = std::vector, typename Image, typename Functor>
Container<Image> read_test_images(const std::string& folder, std::size_t limit, Functor func) {
  Container<Image> images;
  read_mnist_image_file<Container, Image>(images, folder + "/t10k-images.idx3-ubyte", limit, func);
  return images;
}

/*!
 * \brief Read all training label and return a container filled with the labels.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \param limit The maximum number of elements to read (0: no limit)
 * \return Container filled with the labels
 */
template <template <typename...> class Container = std::vector, typename Label = uint8_t>
Container<Label> read_training_labels(const std::string& folder, std::size_t limit) {
  Container<Label> labels;
  read_mnist_label_file<Container, Label>(labels, folder + "/train-labels.idx1-ubyte", limit);
  return labels;
}

/*!
 * \brief Read all test label and return a container filled with the labels.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \param limit The maximum number of elements to read (0: no limit)
 * \return Container filled with the labels
 */
template <template <typename...> class Container = std::vector, typename Label = uint8_t>
Container<Label> read_test_labels(const std::string& folder, std::size_t limit) {
  Container<Label> labels;
  read_mnist_label_file<Container, Label>(labels, folder + "/t10k-labels.idx1-ubyte", limit);
  return labels;
}

/*!
 * \brief Read dataset and assume images in 3D (1x28x28)
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \param training_limit The maximum number of elements to read from training set (0: no limit)
 * \param test_limit The maximum number of elements to read from test set (0: no limit)
 * \return The dataset
 */
template <template <typename...> class Container, typename Image, typename Label = uint8_t>
MNIST_dataset<Container, Image, Label> read_dataset_3d(const std::string& folder, std::size_t training_limit = 0, std::size_t test_limit = 0) {
  MNIST_dataset<Container, Image, Label> dataset;

  dataset.training_images = read_training_images<Container, Image>(folder, training_limit, [] { return Image(1, 28, 28); });
  dataset.training_labels = read_training_labels<Container, Label>(folder, training_limit);

  dataset.test_images = read_test_images<Container, Image>(folder, test_limit, [] { return Image(1, 28, 28); });
  dataset.test_labels = read_test_labels<Container, Label>(folder, test_limit);

  return dataset;
}

/*!
 * \brief Read dataset and assume images in 3D (1x28x28)
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \param training_limit The maximum number of elements to read from training set (0: no limit)
 * \param test_limit The maximum number of elements to read from test set (0: no limit)
 * \return The dataset
 */
template <template <typename...> class Container, typename Image, typename Label = uint8_t>
MNIST_dataset<Container, Image, Label> read_dataset_3d(std::size_t training_limit = 0, std::size_t test_limit = 0) {
  return read_dataset_3d<Container, Image, Label>("mnist", training_limit, test_limit);
}

/*!
 * \brief Read dataset from some location.
 *
 * \param training_limit The maximum number of elements to read from training set (0: no limit)
 * \param test_limit The maximum number of elements to read from test set (0: no limit)
 * \return The dataset
 */
template <template <typename...> class Container, typename Image, typename Label = uint8_t>
MNIST_dataset<Container, Image, Label> read_dataset_direct(const std::string& folder, std::size_t training_limit = 0, std::size_t test_limit = 0) {
  MNIST_dataset<Container, Image, Label> dataset;

  dataset.training_images = read_training_images<Container, Image>(folder, training_limit, [] { return Image(1 * 28 * 28); });
  dataset.training_labels = read_training_labels<Container, Label>(folder, training_limit);

  dataset.test_images = read_test_images<Container, Image>(folder, test_limit, [] { return Image(1 * 28 * 28); });
  dataset.test_labels = read_test_labels<Container, Label>(folder, test_limit);

  return dataset;
}

/*!
 * \brief Read dataset.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \param training_limit The maximum number of elements to read from training set (0: no limit)
 * \param test_limit The maximum number of elements to read from test set (0: no limit)
 * \return The dataset
 */
template <template <typename...> class Container, typename Image, typename Label = uint8_t>
MNIST_dataset<Container, Image, Label> read_dataset_direct(std::size_t training_limit = 0, std::size_t test_limit = 0) {
  return read_dataset_direct<Container, Image, Label>("mnist", training_limit, test_limit);
}

/*!
 * \brief Read dataset.
 *
 * The dataset is assumed to be in a mnist subfolder
 *
 * \param training_limit The maximum number of elements to read from training set (0: no limit)
 * \param test_limit The maximum number of elements to read from test set (0: no limit)
 * \return The dataset
 */
template <template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
MNIST_dataset<Container, Sub<Pixel>, Label> read_dataset(std::size_t training_limit = 0, std::size_t test_limit = 0) {
  return read_dataset_direct<Container, Sub<Pixel>>(training_limit, test_limit);
}

/*!
 * \brief Read dataset from some location.
 *
 * \param training_limit The maximum number of elements to read from training set (0: no limit)
 * \param test_limit The maximum number of elements to read from test set (0: no limit)
 * \return The dataset
 */
template <template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
MNIST_dataset<Container, Sub<Pixel>, Label> read_dataset(const std::string& folder, std::size_t training_limit = 0, std::size_t test_limit = 0) {
  return read_dataset_direct<Container, Sub<Pixel>>(folder, training_limit, test_limit);
}

}  //end of namespace mnist

#endif
