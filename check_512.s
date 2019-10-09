.intel_syntax noprefix
infiniteLoop:
 jmp main
main:
 vxorpd zmm0,zmm0,zmm0
 jmp infiniteLoop

