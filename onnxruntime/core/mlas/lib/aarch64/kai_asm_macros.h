/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    kai_asm_macros.h

Abstract:

    Portable assembly macros for AArch64 kernels that must assemble under
    both the GNU assembler (Linux, macOS) and Microsoft armasm64 (Windows).

    The macro set and its names are adopted verbatim from Arm's KleidiAI
    library (kai/kai_common_sve_asm.S, SPDX-License-Identifier: Apache-2.0),
    which inlines this block at the top of each of its portable *_asm.S
    files; keeping the same names lets kernels move between the two trees
    unchanged. This header exists so ONNX Runtime kernels can share a single
    copy.

    Instructions that a given assembler may not accept as mnemonics (all SVE
    instructions, in the case of armasm64) are emitted as raw instruction
    words via KAI_ASM_INST, which expands to ".inst" under GAS and "DCD"
    under armasm64 — the same bytes either way.

--*/

#if defined(_MSC_VER)

    #define KAI_ASM_GLOBAL(name) GLOBAL name
    #define KAI_ASM_FUNCTION_TYPE(name)
    #define KAI_ASM_FUNCTION_LABEL(name) name PROC
    #define KAI_ASM_FUNCTION_END(name) ENDP

    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
    #define KAI_ASM_ALIGN
    #define KAI_ASM_LABEL(name) name
    #define KAI_ASM_INST(hex) DCD hex
    #define KAI_ASM_END END

#else

    #if defined(__APPLE__)
        #define KAI_ASM_GLOBAL(name) .globl _##name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) _##name:
        #define KAI_ASM_FUNCTION_END(name)
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
    #endif

    #define KAI_ASM_CODE(name) .text
    #define KAI_ASM_ALIGN .p2align 4,,11
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END

#endif

#if defined(__ARM_FEATURE_BTI_DEFAULT) && __ARM_FEATURE_BTI_DEFAULT == 1
    #define KAI_ASM_BTI_C KAI_ASM_INST(0xd503245f)

    #if defined(__ELF__)
        .pushsection .note.gnu.property, "a"
        .p2align 3
        .long 4
        .long 0x10
        .long 0x5
        .asciz "GNU"
        .long 0xc0000000
        .long 4
        .long 1
        .long 0
        .popsection
    #endif
#else
    #define KAI_ASM_BTI_C
#endif
