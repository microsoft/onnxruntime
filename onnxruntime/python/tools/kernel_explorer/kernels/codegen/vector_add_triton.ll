define amdgpu_kernel void @vector_add_triton_kernel(half addrspace(1)* align 16 %0, half addrspace(1)* align 16 %1, half addrspace(1)* align 16 %2, i32 %3) {
entry:
  %4 = call i32 @llvm.amdgcn.workitem.id.x()
  %5 = urem i32 %4, 32
  %6 = udiv i32 %4, 32
  %7 = mul i32 %6, 32
  %8 = add i32 %7, %5
  %9 = mul i32 %8, 8
  %idx_0_0 = add i32 %9, 0
  %idx_0_1 = add i32 %9, 1
  %idx_0_2 = add i32 %9, 2
  %idx_0_3 = add i32 %9, 3
  %idx_0_4 = add i32 %9, 4
  %idx_0_5 = add i32 %9, 5
  %idx_0_6 = add i32 %9, 6
  %idx_0_7 = add i32 %9, 7
  %10 = call i32 @llvm.amdgcn.workgroup.id.x()
  %11 = mul i32 %10, 1024
  %12 = add i32 0, %9
  %13 = add i32 %12, 0
  %14 = add i32 0, %9
  %15 = add i32 %14, 1
  %16 = add i32 0, %9
  %17 = add i32 %16, 2
  %18 = add i32 0, %9
  %19 = add i32 %18, 3
  %20 = add i32 0, %9
  %21 = add i32 %20, 4
  %22 = add i32 0, %9
  %23 = add i32 %22, 5
  %24 = add i32 0, %9
  %25 = add i32 %24, 6
  %26 = add i32 0, %9
  %27 = add i32 %26, 7
  %28 = add i32 %11, %12
  %29 = add i32 %28, 0
  %30 = add i32 %11, %14
  %31 = add i32 %30, 1
  %32 = add i32 %11, %16
  %33 = add i32 %32, 2
  %34 = add i32 %11, %18
  %35 = add i32 %34, 3
  %36 = add i32 %11, %20
  %37 = add i32 %36, 4
  %38 = add i32 %11, %22
  %39 = add i32 %38, 5
  %40 = add i32 %11, %24
  %41 = add i32 %40, 6
  %42 = add i32 %11, %26
  %43 = add i32 %42, 7
  %44 = icmp slt i32 %29, %3
  %45 = icmp slt i32 %31, %3
  %46 = icmp slt i32 %33, %3
  %47 = icmp slt i32 %35, %3
  %48 = icmp slt i32 %37, %3
  %49 = icmp slt i32 %39, %3
  %50 = icmp slt i32 %41, %3
  %51 = icmp slt i32 %43, %3
  %52 = getelementptr half, half addrspace(1)* %0, i32 %28
  %53 = getelementptr half, half addrspace(1)* %52, i32 0
  %54 = getelementptr half, half addrspace(1)* %0, i32 %30
  %55 = getelementptr half, half addrspace(1)* %54, i32 1
  %56 = getelementptr half, half addrspace(1)* %0, i32 %32
  %57 = getelementptr half, half addrspace(1)* %56, i32 2
  %58 = getelementptr half, half addrspace(1)* %0, i32 %34
  %59 = getelementptr half, half addrspace(1)* %58, i32 3
  %60 = getelementptr half, half addrspace(1)* %0, i32 %36
  %61 = getelementptr half, half addrspace(1)* %60, i32 4
  %62 = getelementptr half, half addrspace(1)* %0, i32 %38
  %63 = getelementptr half, half addrspace(1)* %62, i32 5
  %64 = getelementptr half, half addrspace(1)* %0, i32 %40
  %65 = getelementptr half, half addrspace(1)* %64, i32 6
  %66 = getelementptr half, half addrspace(1)* %0, i32 %42
  %67 = getelementptr half, half addrspace(1)* %66, i32 7
  %68 = load half, half addrspace(1)* %53, align 2
  %69 = load half, half addrspace(1)* %55, align 2
  %70 = load half, half addrspace(1)* %57, align 2
  %71 = load half, half addrspace(1)* %59, align 2
  %72 = load half, half addrspace(1)* %61, align 2
  %73 = load half, half addrspace(1)* %63, align 2
  %74 = load half, half addrspace(1)* %65, align 2
  %75 = load half, half addrspace(1)* %67, align 2
  %76 = getelementptr half, half addrspace(1)* %1, i32 %28
  %77 = getelementptr half, half addrspace(1)* %76, i32 0
  %78 = getelementptr half, half addrspace(1)* %1, i32 %30
  %79 = getelementptr half, half addrspace(1)* %78, i32 1
  %80 = getelementptr half, half addrspace(1)* %1, i32 %32
  %81 = getelementptr half, half addrspace(1)* %80, i32 2
  %82 = getelementptr half, half addrspace(1)* %1, i32 %34
  %83 = getelementptr half, half addrspace(1)* %82, i32 3
  %84 = getelementptr half, half addrspace(1)* %1, i32 %36
  %85 = getelementptr half, half addrspace(1)* %84, i32 4
  %86 = getelementptr half, half addrspace(1)* %1, i32 %38
  %87 = getelementptr half, half addrspace(1)* %86, i32 5
  %88 = getelementptr half, half addrspace(1)* %1, i32 %40
  %89 = getelementptr half, half addrspace(1)* %88, i32 6
  %90 = getelementptr half, half addrspace(1)* %1, i32 %42
  %91 = getelementptr half, half addrspace(1)* %90, i32 7
  %92 = load half, half addrspace(1)* %77, align 2
  %93 = load half, half addrspace(1)* %79, align 2
  %94 = load half, half addrspace(1)* %81, align 2
  %95 = load half, half addrspace(1)* %83, align 2
  %96 = load half, half addrspace(1)* %85, align 2
  %97 = load half, half addrspace(1)* %87, align 2
  %98 = load half, half addrspace(1)* %89, align 2
  %99 = load half, half addrspace(1)* %91, align 2
  %100 = fadd half %68, %92
  %101 = fadd half %69, %93
  %102 = fadd half %70, %94
  %103 = fadd half %71, %95
  %104 = fadd half %72, %96
  %105 = fadd half %73, %97
  %106 = fadd half %74, %98
  %107 = fadd half %75, %99
  %108 = getelementptr half, half addrspace(1)* %2, i32 %28
  %109 = getelementptr half, half addrspace(1)* %108, i32 0
  %110 = getelementptr half, half addrspace(1)* %2, i32 %30
  %111 = getelementptr half, half addrspace(1)* %110, i32 1
  %112 = getelementptr half, half addrspace(1)* %2, i32 %32
  %113 = getelementptr half, half addrspace(1)* %112, i32 2
  %114 = getelementptr half, half addrspace(1)* %2, i32 %34
  %115 = getelementptr half, half addrspace(1)* %114, i32 3
  %116 = getelementptr half, half addrspace(1)* %2, i32 %36
  %117 = getelementptr half, half addrspace(1)* %116, i32 4
  %118 = getelementptr half, half addrspace(1)* %2, i32 %38
  %119 = getelementptr half, half addrspace(1)* %118, i32 5
  %120 = getelementptr half, half addrspace(1)* %2, i32 %40
  %121 = getelementptr half, half addrspace(1)* %120, i32 6
  %122 = getelementptr half, half addrspace(1)* %2, i32 %42
  %123 = getelementptr half, half addrspace(1)* %122, i32 7
  store half %100, half addrspace(1)* %109, align 2
  store half %101, half addrspace(1)* %111, align 2
  store half %102, half addrspace(1)* %113, align 2
  store half %103, half addrspace(1)* %115, align 2
  store half %104, half addrspace(1)* %117, align 2
  store half %105, half addrspace(1)* %119, align 2
  store half %106, half addrspace(1)* %121, align 2
  store half %107, half addrspace(1)* %123, align 2
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workgroup.id.x() #0

