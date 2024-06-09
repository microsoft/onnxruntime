// //////////////////////////////////////////////////////////
// md5.h
// Copyright (c) 2014 Stephan Brumme. All rights reserved.
// see http://create.stephan-brumme.com/disclaimer.html
//

#pragma once

//#include "hash.h"
#include <string>

// define fixed size integer types
#ifdef _MSC_VER
// Windows
typedef unsigned __int8  uint8_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#else
// GCC
#include <stdint.h>
#endif


/// compute MD5 hash
/** Usage:
    MD5 md5;
    std::string myHash  = md5("Hello World");     // std::string
    std::string myHash2 = md5("How are you", 11); // arbitrary data, 11 bytes

    // or in a streaming fashion:

    MD5 md5;
    while (more data available)
      md5.add(pointer to fresh data, number of new bytes);
    std::string myHash3 = md5.getHash();
  */
class MD5 //: public Hash
{
public:
  /// split into 64 byte blocks (=> 512 bits), hash is 16 bytes long
  enum { BlockSize = 512 / 8, HashBytes = 16 };

  /// same as reset()
  MD5();

  /// compute MD5 of a memory block
  std::string operator()(const void* data, size_t numBytes);
  /// compute MD5 of a string, excluding final zero
  std::string operator()(const std::string& text);

  /// add arbitrary number of bytes
  void add(const void* data, size_t numBytes);

  /// return latest hash as 32 hex characters
  std::string getHash();
  /// return latest hash as bytes
  void        getHash(unsigned char buffer[HashBytes]);

  /// restart
  void reset();

private:
  /// process 64 bytes
  void processBlock(const void* data);
  /// process everything left in the internal buffer
  void processBuffer();

  /// size of processed data in bytes
  uint64_t m_numBytes;
  /// valid bytes in m_buffer
  size_t   m_bufferSize;
  /// bytes not processed yet
  uint8_t  m_buffer[BlockSize];

  enum { HashValues = HashBytes / 4 };
  /// hash, stored as integers
  uint32_t m_hash[HashValues];
};
