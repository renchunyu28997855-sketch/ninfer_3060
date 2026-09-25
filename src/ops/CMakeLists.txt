# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops STATIC)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core PRIVATE ninfer_nvfp4_non_rdc)

# Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from
# producer to consumer warpgroups. RDC makes ptxas discard that contract. These
# self-contained kernels therefore use a non-RDC archive and host launchers.
add_library(ninfer_nvfp4_non_rdc STATIC)
ninfer_internal_includes(ninfer_nvfp4_non_rdc)
ninfer_cuda_non_rdc_archive(ninfer_nvfp4_non_rdc)
target_link_libraries(ninfer_nvfp4_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)

include(basic_sources.cmake)
include("${CMAKE_CURRENT_LIST_DIR}/attn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/candidate_selector/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_gating_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_topk/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sparse_moe/sources.cmake")

# Ampere/early-Ada (sm_86/sm_89): FP8 A8 and NVFP4 W4A4 mma translation units use
# Blackwell-only tensor-core qualifiers and cannot be compiled there. The A16
# dequantizing routes stay and remain functional; host launchers are provided by
# throwing stubs when referenced.
get_property(_ninfer_sm8x_compat GLOBAL PROPERTY _ninfer_sm8x_compat)
if(CMAKE_CUDA_ARCHITECTURES MATCHES "120a" OR NOT CMAKE_CUDA_ARCHITECTURES MATCHES "(^|;)?(86|89)")
  set(_ninfer_sm8x_compat OFF)
else()
  set(_ninfer_sm8x_compat ON)
endif()
if(_ninfer_sm8x_compat)
  get_target_property(_ninfer_ops_sources ninfer_ops SOURCES)
  list(FILTER _ninfer_ops_sources EXCLUDE REGEX "nvfp4/.+_w4a4_tma\.cu$")
  list(FILTER _ninfer_ops_sources EXCLUDE REGEX "nvfp4/.+_w4a4\.cu$")
  list(FILTER _ninfer_ops_sources EXCLUDE REGEX ".*/fp8/.+_a8\\.cu$")
  list(FILTER _ninfer_ops_sources EXCLUDE REGEX ".*/fp8/shapes/.+")
  list(FILTER _ninfer_ops_sources EXCLUDE REGEX ".*/causal_cache/(prompt|small_t)_fp8\\.cu$")
  list(FILTER _ninfer_ops_sources EXCLUDE REGEX ".*/causal_cache/(small_t|prompt)_(nvfp4|k8v4)\\.cu$")
  list(FILTER _ninfer_ops_sources EXCLUDE REGEX ".*/kv_cache/append/(nvfp4|k8v4)_launch\\.cu$")
  list(FILTER _ninfer_ops_sources EXCLUDE REGEX ".*/nvfp4/shapes/.+")
  set_property(TARGET ninfer_ops PROPERTY SOURCES ${_ninfer_ops_sources})

  # The non-RDC archive holds only Blackwell NVFP4 kernels; keep it buildable with a
  # placeholder translation unit on sm_86/sm_89.
  set_property(TARGET ninfer_nvfp4_non_rdc PROPERTY SOURCES "${CMAKE_CURRENT_LIST_DIR}/sm8x_placeholder.cpp")
endif()