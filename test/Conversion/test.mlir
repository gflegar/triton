// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [1, 0]}>
#slice = #triton_gpu.slice<{dim = 1, parent = #blocked}>
module attributes {"triton_gpu.num-warps" = 2 : i32} {
  tt.func public @reduce_bools(%arg: tensor<256x2xi1, #blocked>) {
    %24 = "tt.reduce"(%arg) <{axis = 1 : i32}> ({
    ^bb0(%arg4: i1, %arg5: i1):
      %48 = arith.ori %arg4, %arg5 : i1
      tt.reduce.return %48 : i1
    }) : (tensor<256x2xi1, #blocked>) -> tensor<256xi1, #slice>
    tt.return
  }
}
