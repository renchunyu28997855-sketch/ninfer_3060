# Ternary (PQ2_0 / t2_g128_fp16) row-split GEMM family: SIMT fallback plus the
# MMA kernels. The dispatch host entry is shared with the rotation pre-pass.
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/ternary_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ternary_rowsplit_gemm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ternary_rotation.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ternary_rotation.cu"
)
