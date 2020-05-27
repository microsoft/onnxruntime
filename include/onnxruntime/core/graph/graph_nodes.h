// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <type_traits>
#include <vector>

namespace onnxruntime {

class Node;

/**
Class to filter out null entries from either a vector of unique_ptr<Node> or a vector of [const] Node* and
provide an iterator interface that returns [const] Node& for the valid entries.
*/
template <typename TNodesContainer>
class ValidNodes {
 public:
  template <typename TIterator>
  class NodeIterator;

  /**
  Construct a ValidNodes instance to provide iteration over all valid nodes in the TNodesCollection
  @param[in] nodes Nodes to iterate, skipping invalid entries.
  */
  explicit ValidNodes(TNodesContainer& nodes) noexcept : nodes_(nodes) {}

  using ConstNodeIterator = NodeIterator<typename TNodesContainer::const_iterator>;
  using MutableNodeIterator = NodeIterator<typename TNodesContainer::iterator>;
  using ConstReverseNodeIterator = NodeIterator<typename TNodesContainer::const_reverse_iterator>;

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

  ConstReverseNodeIterator rbegin() const noexcept {
    return {nodes_.crbegin(), nodes_.crend()};
  }

  ConstReverseNodeIterator rend() const noexcept {
    return {nodes_.crend(), nodes_.crend()};
  }

  MutableNodeIterator begin() noexcept {
    return {nodes_.begin(), nodes_.end()};
  }

  MutableNodeIterator end() noexcept {
    return {nodes_.end(), nodes_.end()};
  }

  bool empty() const noexcept { return nodes_.empty(); }

  /**
  @class NodeIterator
  Iterator to provide const and non-const access to valid Node instances in a Graph.
  @remarks Skips invalid nodes.
  */
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
    using difference_type = typename TIterator::difference_type;
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;

    /** Construct a NodeInterator and move to the first valid node. */
    NodeIterator<TIterator>(const TIterator current, const TIterator end) noexcept : current_{current}, end_{end} {
      // skip to next valid node, stopping at end if none are found
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

    /** Return the current Node&. This will be const if the iterator was returned from a const GraphNodes instance. */
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
    TIterator end_;
  };

 private:
  TNodesContainer& nodes_;
};

/**
Class that provides iteration over all valid nodes in the Graph.
*/
class GraphNodes : public ValidNodes<std::vector<std::unique_ptr<Node>>> {
 public:
  GraphNodes(std::vector<std::unique_ptr<Node>>& nodes) : ValidNodes(nodes) {}
};

}  // namespace onnxruntime
