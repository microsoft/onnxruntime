# remove_math_h_dependency_from_fp16_h.patch

Remove dependency on math.h (with fabsf()) to work around Xcode 16 build error for iphonesimulator x86_64 target:

/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk/usr/include/math.h:614:27: error:
      _Float16 is not supported on this target
  614 | extern _Float16 __fabsf16(_Float16) __API_AVAILABLE(macos(15.0), ios(18.0), watchos(11.0), tvos(18.0));
      |

This patch was adapted from this PR: https://github.com/Maratyszcza/FP16/pull/32
See also: https://github.com/google/XNNPACK/issues/6989
