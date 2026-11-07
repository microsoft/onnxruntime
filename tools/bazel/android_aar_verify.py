# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Check a real cross-built AAR on the host without loading Android ELF files."""

import io
import pathlib
import struct
import sys
import zipfile


def verify(path):
    with zipfile.ZipFile(path) as archive:
        required = {
            "classes.jar",
            "AndroidManifest.xml",
            "R.txt",
            "proguard.txt",
            "headers/onnxruntime_c_api.h",
            "headers/onnxruntime_cxx_api.h",
            "headers/onnxruntime_cxx_inline.h",
            "META-INF/LICENSE",
            "META-INF/ThirdPartyNotices.txt",
            "META-INF/VERSION_NUMBER",
        }
        missing = required - set(archive.namelist())
        if missing:
            raise ValueError(f"{path}: missing AAR entries {sorted(missing)}; rebuild //:android_aar.")
        for name in ("libonnxruntime.so", "libonnxruntime4j_jni.so"):
            elf = archive.read("jni/arm64-v8a/" + name)
            # ELF64, little-endian, shared object, EM_AARCH64.
            if elf[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<HH", elf, 16) != (3, 183):
                raise ValueError(f"{name}: expected an ARM64 Android shared library; use --config=android_arm64.")
            phoff = struct.unpack_from("<Q", elf, 32)[0]
            phsize, phcount = struct.unpack_from("<HH", elf, 54)
            loads = []
            for index in range(phcount):
                offset = phoff + index * phsize
                if struct.unpack_from("<I", elf, offset)[0] == 1:  # PT_LOAD
                    file_offset, address = struct.unpack_from("<QQ", elf, offset + 8)
                    alignment = struct.unpack_from("<Q", elf, offset + 48)[0]
                    loads.append(alignment)
                    if alignment < 16384 or file_offset % 16384 != address % 16384:
                        raise ValueError(f"{name}: incompatible with 16 KB pages; link with -z,max-page-size=16384.")
            if not loads:
                raise ValueError(f"{name}: missing ELF load segments; rebuild the native library.")
        with zipfile.ZipFile(io.BytesIO(archive.read("classes.jar"))) as classes:
            for name in (
                "ai/onnxruntime/OrtEnvironment.class",
                "ai/onnxruntime/OrtSession.class",
                "ai/onnxruntime/platform/Fp16Conversions.class",
            ):
                if name not in classes.namelist():
                    raise ValueError(f"classes.jar: missing {name}; compile the Android Java sources.")
            fp16 = classes.read("ai/onnxruntime/platform/Fp16Conversions.class")
            if b"java/lang/invoke/MethodHandles" in fp16:
                raise ValueError("classes.jar: expected Android FP16 conversion implementation, not JVM sources.")
    print(f"Verified ARM64 AAR layout, Java classes, and 16 KB ELF segments: {pathlib.Path(path).name}")


if __name__ == "__main__":
    verify(sys.argv[1])
