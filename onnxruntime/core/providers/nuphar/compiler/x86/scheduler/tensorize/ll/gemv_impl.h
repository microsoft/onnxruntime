// The string in this file is generated using clang:
// clang++.exe -fno-preserve-as-comments -S -emit-llvm gemv_impl.cpp

namespace onnxruntime {
namespace nuphar {

const char* gemv_stubs_ir = R"gemv_stub_escape(
; ModuleID = 'gemv_stubs.cpp'
source_filename = "gemv_stubs.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.25548"

; Function Attrs: noinline nounwind optnone uwtable
define i32 @gemv_update(float*, float*, float*, i32, i32, i32) #0 {
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca float*, align 8
  %11 = alloca float*, align 8
  %12 = alloca float*, align 8
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  store i32 %5, i32* %7, align 4
  store i32 %4, i32* %8, align 4
  store i32 %3, i32* %9, align 4
  store float* %2, float** %10, align 8
  store float* %1, float** %11, align 8
  store float* %0, float** %12, align 8
  store i32 0, i32* %13, align 4
  br label %15

; <label>:15:                                     ; preds = %50, %6
  %16 = load i32, i32* %13, align 4
  %17 = load i32, i32* %9, align 4
  %18 = icmp slt i32 %16, %17
  br i1 %18, label %19, label %53

; <label>:19:                                     ; preds = %15
  store i32 0, i32* %14, align 4
  br label %20

; <label>:20:                                     ; preds = %46, %19
  %21 = load i32, i32* %14, align 4
  %22 = load i32, i32* %8, align 4
  %23 = icmp slt i32 %21, %22
  br i1 %23, label %24, label %49

; <label>:24:                                     ; preds = %20
  %25 = load float*, float** %11, align 8
  %26 = load i32, i32* %14, align 4
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds float, float* %25, i64 %27
  %29 = load float, float* %28, align 4
  %30 = load float*, float** %10, align 8
  %31 = load i32, i32* %13, align 4
  %32 = load i32, i32* %7, align 4
  %33 = mul nsw i32 %31, %32
  %34 = load i32, i32* %14, align 4
  %35 = add nsw i32 %33, %34
  %36 = sext i32 %35 to i64
  %37 = getelementptr inbounds float, float* %30, i64 %36
  %38 = load float, float* %37, align 4
  %39 = fmul float %29, %38
  %40 = load float*, float** %12, align 8
  %41 = load i32, i32* %13, align 4
  %42 = sext i32 %41 to i64
  %43 = getelementptr inbounds float, float* %40, i64 %42
  %44 = load float, float* %43, align 4
  %45 = fadd float %44, %39
  store float %45, float* %43, align 4
  br label %46

; <label>:46:                                     ; preds = %24
  %47 = load i32, i32* %14, align 4
  %48 = add nsw i32 %47, 1
  store i32 %48, i32* %14, align 4
  br label %20

; <label>:49:                                     ; preds = %20
  br label %50

; <label>:50:                                     ; preds = %49
  %51 = load i32, i32* %13, align 4
  %52 = add nsw i32 %51, 1
  store i32 %52, i32* %13, align 4
  br label %15

; <label>:53:                                     ; preds = %15
  ret i32 0
}

; Function Attrs: noinline nounwind optnone uwtable
define i32 @gemv_reset(float*, i32) #0 {
  %3 = alloca i32, align 4
  %4 = alloca float*, align 8
  %5 = alloca i32, align 4
  store i32 %1, i32* %3, align 4
  store float* %0, float** %4, align 8
  store i32 0, i32* %5, align 4
  br label %6

; <label>:6:                                      ; preds = %15, %2
  %7 = load i32, i32* %5, align 4
  %8 = load i32, i32* %3, align 4
  %9 = icmp slt i32 %7, %8
  br i1 %9, label %10, label %18

; <label>:10:                                     ; preds = %6
  %11 = load float*, float** %4, align 8
  %12 = load i32, i32* %5, align 4
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds float, float* %11, i64 %13
  store float 0.000000e+00, float* %14, align 4
  br label %15

; <label>:15:                                     ; preds = %10
  %16 = load i32, i32* %5, align 4
  %17 = add nsw i32 %16, 1
  store i32 %17, i32* %5, align 4
  br label %6

; <label>:18:                                     ; preds = %6
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 6.0.1 (tags/RELEASE_601/final)"}
)gemv_stub_escape";

}  // namespace nuphar
}  // namespace onnxruntime
