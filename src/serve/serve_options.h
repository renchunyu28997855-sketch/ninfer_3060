#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// Protocol default when the client omits max_tokens. Engine independently
// clamps the request to its effective context capacity.
inline constexpr int kDefaultMaxTokens                    = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;

struct ServeOptions {
    bool help_requested = false;
    std::string artifact_path;
    std::filesystem::path chat_template_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact metadata.name
    std::string request_log_jsonl;                // empty => structured request logging disabled
    std::uint32_t max_context          = 8192;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(8192);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    // Long-context working set: device-resident token budget for long histories (0 = off) and
    // the always-resident leading sink. Host pool comes from --host-kv-mib or auto-sizing.
    // "auto" sizing resolves the budget from the device KV pool at startup instead of a fixed
    // --kv-working-set value; kv_sink_explicit records whether --kv-sink was given so the
    // engine can derive the sink only when it was not.
    std::uint32_t kv_working_set       = 0;
    bool         kv_working_set_auto   = false;
    std::uint8_t kv_ws_grant_mode      = 0; // 0 take-remaining | 1 fair-even | 2 elastic (auto only)
    std::uint32_t kv_sink              = 2048;
    bool         kv_sink_explicit      = false;
    // Sized-slots mode: one device-window percentage (0,100] per concurrency lane. Non-empty
    // selects per-lane fixed budgets and is mutually exclusive with --kv-working-set auto/explicit.
    std::vector<double> kv_slot_percentages;
    std::filesystem::path context_cost_presets;
    std::uint32_t log_stats_interval_ms    = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    int device                             = 0;
    KvCacheStorage kv_cache                = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    ContextCacheOptions context_cache;
    bool enable_vision      = false;
    bool use_cuda_graph     = true;
    bool allow_prefix_reuse = true;
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
    std::optional<std::uint32_t> default_thinking_budget;
    int default_max_tokens = kDefaultMaxTokens;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs
    // Process-level explicit overrides layered between registered model/mode defaults and request
    // fields. An omitted seed is replaced per request with a fresh random seed.
    SamplingOverrides sampling_overrides;
    bool greedy                 = false; // --greedy: force temperature 0 (exact argmax)
    bool wddm_evictable_budget  = false; // --wddm-evictable-budget: budget runtime against total VRAM (Windows)
    product::LogLevel log_level = product::LogLevel::Info;

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

ServeOptions parse_serve_options(int argc, char** argv);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name);
std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve
