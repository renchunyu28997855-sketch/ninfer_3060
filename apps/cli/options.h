#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::cli {

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    std::uint32_t max_context    = 2048;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t prefill_chunk  = 1024;
    int device                   = 0;

    // Long-context working set: keep a block-aligned token budget of history resident in the
    // device KV row and park the rest on pinned host memory. 0 = off (dense row, today's path).
    std::uint32_t kv_working_set = 0;
    std::uint32_t kv_sink        = 2048; // always-resident leading tokens
    std::uint64_t kv_host_capacity = 0;  // host KV pool bytes; 0 = auto-size

    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    bool enable_vision  = false;
    bool use_cuda_graph = true;

    bool raw_output      = false;
    bool print_token_ids = false;
    std::optional<bool> enable_thinking;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<ReasoningEffort> reasoning_effort;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy                 = false;
    product::LogLevel log_level = product::LogLevel::Info;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace ninfer::cli
