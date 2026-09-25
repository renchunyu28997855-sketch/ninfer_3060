// sm_86/sm_89 compatibility stubs for op translation units that use instructions
// unavailable before sm_89 (FP8 e4m3 tensor-core matrix multiply, NVFP4/K8V4 KV
// storage codecs). These entry points are unreachable in sm_8x artifacts: the
// corresponding weight/KV formats cannot be represented by products built for
// these targets. Every function throws instead of dispatching.

#include "ninfer/types.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#if defined(NINFER_SM8X_COMPAT)

#include <stdexcept>

namespace {
[[noreturn]] void throw_unavailable(const char* what)
{
    throw std::runtime_error(std::string(what) +
                             " is not available on sm_86/sm_89 products");
}
} // namespace

namespace ninfer::ops::detail {

namespace {
void a8_w4a4_unavailable(const char* what)
{
    throw_unavailable(what);
}

void fp8_launch_unavailable(const Tensor&, const Weight&, Tensor&, cudaStream_t)
{
    throw_unavailable("fp8 linear");
}

void fp8_a8_unavailable(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t)
{
    throw_unavailable("fp8 A8 linear");
}

bool fp8_uses_a8_unavailable(std::int32_t, std::int32_t) { return false; }

void nvfp4_launch_unavailable(const Tensor&, const Weight&, Tensor&, cudaStream_t)
{
    throw_unavailable("nvfp4 linear");
}

void nvfp4_a4_unavailable(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace,
                          cudaStream_t)
{
    throw_unavailable("nvfp4 A4 linear");
}

bool nvfp4_uses_a4_unavailable(std::int32_t, std::int32_t) { return false; }

const char* kv_stub_message() { return "kv-cache fp8 append"; }
} // namespace

const Fp8LinearShape kFp8N14336K5120{14336, 5120, fp8_launch_unavailable, fp8_a8_unavailable,
                                      fp8_uses_a8_unavailable};
const Fp8LinearShape kFp8N16384K5120{16384, 5120, fp8_launch_unavailable, fp8_a8_unavailable,
                                      fp8_uses_a8_unavailable};
const Fp8LinearShape kFp8N34816K5120{34816, 5120, fp8_launch_unavailable, fp8_a8_unavailable,
                                      fp8_uses_a8_unavailable};
const Fp8LinearShape kFp8N5120K6144{5120, 6144, fp8_launch_unavailable, fp8_a8_unavailable,
                                    fp8_uses_a8_unavailable};
const Fp8LinearShape kFp8N5120K17408{5120, 17408, fp8_launch_unavailable, fp8_a8_unavailable,
                                     fp8_uses_a8_unavailable};
const Fp8LinearShape kFp8N248320K5120{248320, 5120, fp8_launch_unavailable, fp8_a8_unavailable,
                                      fp8_uses_a8_unavailable};

const Nvfp4LinearShape kNvfp4N14336K5120{14336, 5120, nvfp4_launch_unavailable,
                                         nvfp4_a4_unavailable, nvfp4_uses_a4_unavailable};
const Nvfp4LinearShape kNvfp4N16384K5120{16384, 5120, nvfp4_launch_unavailable,
                                          nvfp4_a4_unavailable, nvfp4_uses_a4_unavailable};
const Nvfp4LinearShape kNvfp4N34816K5120{34816, 5120, nvfp4_launch_unavailable,
                                          nvfp4_a4_unavailable, nvfp4_uses_a4_unavailable};
const Nvfp4LinearShape kNvfp4N5120K6144{5120, 6144, nvfp4_launch_unavailable,
                                        nvfp4_a4_unavailable, nvfp4_uses_a4_unavailable};
const Nvfp4LinearShape kNvfp4N5120K17408{5120, 17408, nvfp4_launch_unavailable,
                                         nvfp4_a4_unavailable, nvfp4_uses_a4_unavailable};
const Nvfp4LinearShape kNvfp4DFlash2Feature{0, 0, nvfp4_launch_unavailable,
                                            nvfp4_a4_unavailable, nvfp4_uses_a4_unavailable};
const Nvfp4LinearShape kNvfp4DFlash2Qkv{0, 0, nvfp4_launch_unavailable,
                                         nvfp4_a4_unavailable, nvfp4_uses_a4_unavailable};
const Nvfp4LinearShape kNvfp4DFlash2AttnOut{0, 0, nvfp4_launch_unavailable,
                                             nvfp4_a4_unavailable, nvfp4_uses_a4_unavailable};
const Nvfp4LinearShape kNvfp4DFlash2ConvProj{0, 0, nvfp4_launch_unavailable,
                                              nvfp4_a4_unavailable, nvfp4_uses_a4_unavailable};
const Nvfp4LinearShape kNvfp4DFlash2Selector{0, 0, nvfp4_launch_unavailable,
                                             nvfp4_a4_unavailable, nvfp4_uses_a4_unavailable};

void kv_cache_append_nvfp4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  PagedKVLayerView cache, cudaStream_t stream)
{
    (void)k;
    (void)v;
    (void)positions;
    (void)cache;
    (void)stream;
    throw_unavailable(kv_stub_message());
}

void kv_cache_append_nvfp4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& table_rows,
                                        PagedKVBatchLayerView cache, cudaStream_t stream)
{
    (void)k;
    (void)v;
    (void)positions;
    (void)valid_columns;
    (void)table_rows;
    (void)cache;
    (void)stream;
    throw_unavailable(kv_stub_message());
}

void kv_cache_append_k8v4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                 PagedKVLayerView cache, cudaStream_t stream)
{
    (void)k;
    (void)v;
    (void)positions;
    (void)cache;
    (void)stream;
    throw_unavailable("kv-cache K8V4 append");
}

void kv_cache_append_k8v4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                      const Tensor& valid_columns, const Tensor& table_rows,
                                      PagedKVBatchLayerView cache, cudaStream_t stream)
{
    (void)k;
    (void)v;
    (void)positions;
    (void)valid_columns;
    (void)table_rows;
    (void)cache;
    (void)stream;
    throw_unavailable("kv-cache K8V4 batch append");
}

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream)
{
    a8_w4a4_unavailable("fp8 attention-input A8");
}

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream)
{
    a8_w4a4_unavailable("fp8 GDN-input A8");
}

void fp8_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                              WorkspaceArena& workspace, cudaStream_t stream)
{
    a8_w4a4_unavailable("fp8 linear-add A8");
}

void fp8_linear_swiglu_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream)
{
    a8_w4a4_unavailable("fp8 linear-swiglu A8");
}

void nvfp4_attn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, Nvfp4W4a4Workspace workspace,
                                   cudaStream_t stream)
{
    a8_w4a4_unavailable("nvfp4 attention-input W4A4");
}

void nvfp4_gdn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  Nvfp4W4a4Workspace workspace, cudaStream_t stream)
{
    a8_w4a4_unavailable("nvfp4 GDN-input W4A4");
}

void nvfp4_linear_add_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  Nvfp4W4a4Workspace workspace, cudaStream_t stream)
{
    a8_w4a4_unavailable("nvfp4 linear-add W4A4");
}

void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream)
{
    a8_w4a4_unavailable("nvfp4 linear-swiglu W4A4");
}

void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                Nvfp4ScaleLayout layout, cudaStream_t stream)
{
    a8_w4a4_unavailable("nvfp4 W4A4 quantize");
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t* activation_codes,
                                         const std::uint8_t* activation_scales,
                                         const std::uint8_t* weight_codes,
                                         const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                         std::int32_t tokens, float alpha, cudaStream_t stream)
{
    a8_w4a4_unavailable("nvfp4 linear-swiglu W4A4 TMA");
}

} // namespace ninfer::ops::detail

namespace {
void causal_launcher_unavailable(const char* what)
{
    throw_unavailable(what);
}
} // namespace

namespace ninfer::ops::detail {

void causal_attention_small_t_nvfp4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream)
{
    (void)q;
    (void)k;
    (void)v;
    (void)positions;
    (void)valid_columns;
    (void)table_rows;
    (void)scale;
    (void)cache;
    (void)envelope;
    (void)column_begin;
    (void)width;
    (void)partial_acc;
    (void)partial_m;
    (void)partial_l;
    (void)out;
    (void)stream;
    causal_launcher_unavailable("causal attention nvfp4");
}

void causal_attention_cached_small_t_nvfp4_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  CausalAttentionExecutionEnvelope envelope,
                                                  Tensor& partial_acc, Tensor& partial_m,
                                                  Tensor& partial_l, Tensor& out,
                                                  cudaStream_t stream)
{
    (void)q;
    (void)positions;
    (void)scale;
    (void)cache;
    (void)envelope;
    (void)partial_acc;
    (void)partial_m;
    (void)partial_l;
    (void)out;
    (void)stream;
    causal_launcher_unavailable("causal attention nvfp4");
}

void causal_attention_small_t_k8v4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream)
{
    (void)q;
    (void)k;
    (void)v;
    (void)positions;
    (void)valid_columns;
    (void)table_rows;
    (void)scale;
    (void)cache;
    (void)envelope;
    (void)column_begin;
    (void)width;
    (void)partial_acc;
    (void)partial_m;
    (void)partial_l;
    (void)out;
    (void)stream;
    causal_launcher_unavailable("causal attention K8V4");
}

void causal_attention_cached_small_t_k8v4_launch(const Tensor& q, const Tensor& positions,
                                                float scale, const PagedKVLayerView& cache,
                                                CausalAttentionExecutionEnvelope envelope,
                                                Tensor& partial_acc, Tensor& partial_m,
                                                Tensor& partial_l, Tensor& out,
                                                cudaStream_t stream)
{
    (void)q;
    (void)positions;
    (void)scale;
    (void)cache;
    (void)envelope;
    (void)partial_acc;
    (void)partial_m;
    (void)partial_l;
    (void)out;
    (void)stream;
    causal_launcher_unavailable("causal attention K8V4");
}

void causal_attention_prompt_nvfp4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache, Tensor& out,
                                          cudaStream_t stream)
{
    (void)q;
    (void)k;
    (void)v;
    (void)positions;
    (void)valid_columns;
    (void)table_rows;
    (void)scale;
    (void)cache;
    (void)out;
    (void)stream;
    causal_launcher_unavailable("prompt attention nvfp4");
}

void causal_attention_prompt_nvfp4_attention_launch(const Tensor& q, const Tensor& positions,
                                                    float scale, const PagedKVLayerView& cache,
                                                    Tensor& out, cudaStream_t stream)
{
    (void)q;
    (void)positions;
    (void)scale;
    (void)cache;
    (void)out;
    (void)stream;
    causal_launcher_unavailable("prompt attention nvfp4");
}

void causal_attention_prompt_k8v4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& positions, const Tensor& valid_columns,
                                         const Tensor& table_rows, float scale,
                                         PagedKVBatchLayerView cache, Tensor& out,
                                         cudaStream_t stream)
{
    (void)q;
    (void)k;
    (void)v;
    (void)positions;
    (void)valid_columns;
    (void)table_rows;
    (void)scale;
    (void)cache;
    (void)out;
    (void)stream;
    causal_launcher_unavailable("prompt attention K8V4");
}

void causal_attention_prompt_k8v4_attention_launch(const Tensor& q, const Tensor& positions,
                                                   float scale, const PagedKVLayerView& cache,
                                                   Tensor& out, cudaStream_t stream)
{
    (void)q;
    (void)positions;
    (void)scale;
    (void)cache;
    (void)out;
    (void)stream;
    causal_launcher_unavailable("prompt attention K8V4");
}

void causal_attention_small_t_fp8_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale,
    PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin,
    std::int32_t width, Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out,
    cudaStream_t stream)
{
    (void)q;
    (void)k;
    (void)v;
    (void)positions;
    (void)valid_columns;
    (void)table_rows;
    (void)scale;
    (void)cache;
    (void)envelope;
    (void)column_begin;
    (void)width;
    (void)partial_acc;
    (void)partial_m;
    (void)partial_l;
    (void)out;
    (void)stream;
    causal_launcher_unavailable("causal attention fp8");
}

void causal_attention_cached_small_t_fp8_launch(const Tensor& q, const Tensor& positions,
                                                float scale, const PagedKVLayerView& cache,
                                                CausalAttentionExecutionEnvelope envelope,
                                                Tensor& partial_acc, Tensor& partial_m,
                                                Tensor& partial_l, Tensor& out, cudaStream_t stream)
{
    (void)q;
    (void)positions;
    (void)scale;
    (void)cache;
    (void)envelope;
    (void)partial_acc;
    (void)partial_m;
    (void)partial_l;
    (void)out;
    (void)stream;
    causal_launcher_unavailable("causal attention fp8");
}

void causal_attention_prompt_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, float scale,
                                        PagedKVBatchLayerView cache, Tensor& out,
                                        cudaStream_t stream)
{
    (void)q;
    (void)k;
    (void)v;
    (void)positions;
    (void)valid_columns;
    (void)table_rows;
    (void)scale;
    (void)cache;
    (void)out;
    (void)stream;
    causal_launcher_unavailable("prompt attention fp8");
}

void causal_attention_prompt_fp8_attention_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  Tensor& out, cudaStream_t stream)
{
    (void)q;
    (void)positions;
    (void)scale;
    (void)cache;
    (void)out;
    (void)stream;
    causal_launcher_unavailable("prompt attention fp8");
}

} // namespace ninfer::ops::detail

#endif // NINFER_SM8X_COMPAT
