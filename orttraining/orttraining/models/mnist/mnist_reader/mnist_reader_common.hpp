//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains common functions to read the MNIST dataset
 */

#ifndef MNIST_READER_COMMON_HPP
#define MNIST_READER_COMMON_HPP

namespace mnist {

/*!
 * \brief Extract the MNIST header from the given buffer
 * \param buffer The current buffer
 * \param position The current reading positoin
 * \return The value of the mnist header
 */
inline uint32_t read_header(const std::unique_ptr<char[]>& buffer, size_t position) {
  auto header = reinterpret_cast<uint32_t*>(buffer.get());

  auto value = *(header + position);
  return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
}

/*!
 * \brief Read a MNIST file inside a raw buffer
 * \param path The path to the image file
 * \return The buffer of byte on success, a nullptr-unique_ptr otherwise
 */
inline std::unique_ptr<char[]> read_mnist_file(const std::string& path, uint32_t key) {
  std::ifstream file;
  file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

  if (!file) {
    std::cout << "Error opening file " << path << std::endl;
    std::cout << std::system_error(errno, std::system_category(), "failed to open " + path).what();
    return {};
  }

  auto size = file.tellg();
  std::unique_ptr<char[]> buffer(new char[size]);

  //Read the entire file at once
  file.seekg(0, std::ios::beg);
  file.read(buffer.get(), size);
  file.close();

  auto magic = read_header(buffer, 0);

  if (magic != key) {
    std::cout << "Invalid magic number, probably not a MNIST file " << path << std::endl;
    return {};
  }

  auto count = read_header(buffer, 1);

  if (magic == 0x803) {
    auto rows = read_header(buffer, 2);
    auto columns = read_header(buffer, 3);

    if (size < count * rows * columns + 16) {
      std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
      return {};
    }
  } else if (magic == 0x801) {
    if (size < count + 8) {
      std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
      return {};
    }
  }

  return buffer;
}

}  //end of namespace mnist

#endif
