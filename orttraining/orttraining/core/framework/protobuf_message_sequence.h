// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>
#include <vector>
#include <type_traits>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/message_lite.h"

#include "core/common/common.h"

namespace onnxruntime {

/**
 * Wire format:
 * [number of messages: varint32]
 * for each message:
 *   [number of bytes in message: varint32]
 *   [message bytes]
 */

/**
 * Serializes a sequence of protobuf messages to the output stream.
 * This function should be used in conjunction with ReadProtoMessageSequence().
 *
 * @tparam TMessage The protobuf message type.
 * @param messages The protobuf messages to write.
 * @param output The output stream to write to.
 * @return The status of the operation.
 */
template <typename TMessage>
Status WriteProtoMessageSequence(
    const std::vector<TMessage>& messages,
    google::protobuf::io::ZeroCopyOutputStream& output) {
  static_assert(
      std::is_base_of<google::protobuf::MessageLite, TMessage>::value,
      "TMessage must be derived from google::protobuf::MessageLite.");

  // limit size values to what's representable in an int because they will be
  // parsed as ints with CodedInputStream::ReadVarintSizeAsInt()
  static constexpr size_t k_max_size = std::numeric_limits<int>::max();

  google::protobuf::io::CodedOutputStream coded_output(&output);

  // message count
  const auto message_count = messages.size();
  ORT_RETURN_IF_NOT(message_count <= k_max_size, "message_count > k_max_size");
  coded_output.WriteVarint32(static_cast<int>(message_count));

  for (const auto& message : messages) {
    // message size
    const auto message_size = message.ByteSizeLong();
    ORT_RETURN_IF_NOT(message_size <= k_max_size, "message_count > k_max_size");
    coded_output.WriteVarint32(static_cast<int>(message_size));

    // message bytes
    ORT_RETURN_IF_NOT(message.SerializeToCodedStream(&coded_output), "message.SerializeToCodedStream failed");
  }

  return Status::OK();
}

/**
 * Deserializes a sequence of protobuf messages from the input stream.
 * This function should be used in conjunction with
 * WriteProtoMessageSequence().
 *
 * @tparam TMessage The protobuf message type.
 * @param[out] messages The read protobuf messages.
 * @param input The input stream to read from.
 * @return The status of the operation.
 */
template <typename TMessage>
Status ReadProtoMessageSequence(
    std::vector<TMessage>& messages,
    google::protobuf::io::ZeroCopyInputStream& input) {
  static_assert(
      std::is_base_of<google::protobuf::MessageLite, TMessage>::value,
      "TMessage must be derived from google::protobuf::MessageLite.");

  google::protobuf::io::CodedInputStream coded_input(&input);

  // message count
  int message_count;
  ORT_RETURN_IF_NOT(coded_input.ReadVarintSizeAsInt(&message_count), "coded_input.ReadVarintSizeAsInt failed");

  std::vector<TMessage> result(message_count);
  for (auto& message : result) {
    // message size
    int message_size;
    ORT_RETURN_IF_NOT(coded_input.ReadVarintSizeAsInt(&message_size), "coded_input.ReadVarintSizeAsInt failed");

    // message bytes
    const auto message_limit = coded_input.PushLimit(message_size);
    ORT_RETURN_IF_NOT(message.ParseFromCodedStream(&coded_input), "message.ParseFromCodedStream failed");
    ORT_RETURN_IF_NOT(coded_input.CheckEntireMessageConsumedAndPopLimit(message_limit),
                      "coded_input.CheckEntireMessageConsumedAndPopLimit failed");
  }

  messages = std::move(result);
  return Status::OK();
}

}  // namespace onnxruntime
