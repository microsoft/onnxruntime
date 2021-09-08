#pragma once

#include <utility>

// inspired by https://stackoverflow.com/a/57650648

namespace onnxruntime::current_file_basename_detail {

template <char... Chars>
struct CharSeq {
  static constexpr char value[] = {Chars..., '\0'};
};

template <typename CStrAccessor, size_t Offset, typename RelativeIndexSequence>
struct CStrToCharSeqImpl;

template <typename CStrAccessor, size_t Offset, size_t... RelativeIndexes>
struct CStrToCharSeqImpl<CStrAccessor, Offset, std::index_sequence<RelativeIndexes...>> {
  using type = CharSeq<CStrAccessor::Get()[Offset + RelativeIndexes]...>;
};

template <typename CStrAccessor, size_t Offset, size_t Length>
struct CStrToCharSeq {
  using type = typename CStrToCharSeqImpl<CStrAccessor, Offset, std::make_index_sequence<Length>>::type;
};

constexpr inline std::pair<size_t, size_t> BasenameBeginAndEnd(const char* path) {
  const char* curr = path;
  const char* last_sep = nullptr;
  while (*curr != '\0') {
    if (*curr == '/' || *curr == '\\') last_sep = curr;
    ++curr;
  }
  return {last_sep ? last_sep + 1 - path : 0, curr - path};
}

}  // namespace onnxruntime::current_file_basename_detail

#define ORT_CURRENT_FILE_BASENAME()                                                              \
  []() {                                                                                         \
    struct PathCStrAccessor {                                                                    \
      static constexpr const char* Get() { return __FILE__; }                                    \
    };                                                                                           \
                                                                                                 \
    constexpr auto basename_begin_and_end =                                                      \
        onnxruntime::current_file_basename_detail::BasenameBeginAndEnd(PathCStrAccessor::Get()); \
                                                                                                 \
    using FileBasenameCharSeq =                                                                  \
        typename onnxruntime::current_file_basename_detail::CStrToCharSeq<                       \
            PathCStrAccessor,                                                                    \
            basename_begin_and_end.first,                                                        \
            basename_begin_and_end.second - basename_begin_and_end.first>::type;                 \
                                                                                                 \
    return FileBasenameCharSeq::value;                                                           \
  }()
