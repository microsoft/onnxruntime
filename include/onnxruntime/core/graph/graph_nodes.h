// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <type_traits>
#include <vector>

namespace onnxruntime {

class Node;

/**
Class that provides iteration services for nodes in the Graph. 
It's primary function is to hide holes in the nodes vector due to removed nodes.
*/
class GraphNodes {
  using TNodesContainer = std::vector<std::unique_ptr<Node>>;

 public:
  template <typename TIterator>
  class NodeIterator;

  // construct a wrapper of the nodes that provides iteration services
  explicit GraphNodes(TNodesContainer& nodes) noexcept : nodes_(nodes) {}

  using ConstNodeIterator = NodeIterator<TNodesContainer::const_iterator>;
  using MutableNodeIterator = NodeIterator<TNodesContainer::iterator>;

  ConstNodeIterator cbegin() const noexcept {
    return {nodes_.cbegin(), nodes_.cend()};
  }

  ConstNodeIterator cend() const noexcept {
    return {nodes_.cend(), nodes_.cend()};
  }

  ConstNodeIterator begin() const noexcept {
    return cbegin();
  }

  ConstNodeIterator end() const noexcept {
    return cend();
  }

  MutableNodeIterator begin() noexcept {
    return {nodes_.begin(), nodes_.end()};
  }

  MutableNodeIterator end() noexcept {
    return {nodes_.end(), nodes_.end()};
  }

  // Iterator to provide const and non-const access to nodes, skipping invalid nodes.
  template <typename TIterator>
  class NodeIterator {
    // get the type being returned by the iterator. can't use TIterator::value_type as that is always non-const
    using IterType = typename std::remove_reference<typename std::iterator_traits<TIterator>::reference>::type;
    // and determine what we will return based on its constness
    using T = typename std::conditional<std::is_const<IterType>::value,
                                        const Node,   // return const Node if this is a const iterator
                                        Node>::type;  // else return Node

   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = T;
    using difference_type = typename TIterator::difference_type;  // ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    using const_reference = std::add_const_t<reference>;

    // Constructor. Will move to a valid node or end.
    NodeIterator<TIterator>(TIterator current, const TIterator end) noexcept : current_{current}, end_{end} {
      // skip to valid node or end - whatever comes first
      while (current_ < end && *current_ == nullptr) {
        ++current_;
      }
    }

    bool operator==(const NodeIterator<TIterator>& other) const noexcept {
      return (current_ == other.current_);
    }

    bool operator!=(const NodeIterator<TIterator>& other) const noexcept {
      return (current_ != other.current_);
    }

    void operator++() {
      if (current_ < end_) {
        while (++current_ != end_) {
          if (*current_ != nullptr) break;
        }
      }
    }

    NodeIterator<TIterator> operator++(int) {
      NodeIterator<TIterator> tmp{*this};
      ++(*this);

      return tmp;
    }

    reference operator*() {
      // if iterator is valid we always have a non-nullptr node
      // if this is a nullptr we're at end_ and this shouldn't be being called
      return **current_;
    }

    pointer operator->() {
      return current_->get();
    }

   private:
    TIterator current_;
    const TIterator end_;
  };

 private:
  TNodesContainer& nodes_;
};

}  // namespace onnxruntime
