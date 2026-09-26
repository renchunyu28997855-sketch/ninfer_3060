#pragma once
// Host KV working set — M1 selection scaffold.
//
// Long-context sessions keep a bounded device-resident KV working set and park
// the remainder on the host (M2). M1 establishes the deterministic recency+sink
// selection policy, the per-session row remap state, and the true-position to
// row-local position mapping consumed by the model layer.
//
// Semantics (docs/maintainer/2026-09-21-host-kv-working-set-plan.md §2.2): after
// compaction at true frontier H the KV row holds B resident tokens; every true
// position p >= H maps to row-local position B + (p - H). The attention kernels
// consume those row-local positions while RoPE keeps using the true positions.

#include "core/paged_kv_cache.h"
#include "models/qwen3_5/program/storage/kv_store.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

inline constexpr std::uint32_t kWorkingSetBlockTokens = 128;
inline constexpr std::uint32_t kWorkingSetBlockPages   = kWorkingSetBlockTokens / kPagedKVPageSize;

struct WorkingSetPolicy {
    std::uint32_t budget_tokens = 0;  // device-resident token budget (block-aligned)
    std::uint32_t sink_tokens   = 0;  // always-resident prefix tokens (block-aligned)
};

// One entry per logical KV page owned by the session's true history, ordered by true
// position (block, then page within block). The page handle stays valid while the block's
// content exists on device or host; which replica holds it is derived from the page store
// at use time.
struct WorkingSetPage {
    std::uint32_t true_block = 0;
    LogicalKVPageHandle page{};
};

// Engine defaults (docs/maintainer/2026-09-21-host-kv-working-set-plan.md §4): the
// device-resident budget is 32K tokens and the always-resident sink prefix is 2K.
inline constexpr std::uint32_t kDefaultWorkingSetBudgetTokens = 32 * 1024;
inline constexpr std::uint32_t kDefaultWorkingSetSinkTokens   = 2048;

// Per-session working-set state established at a turn boundary.
struct WorkingSetSession {
    std::uint32_t resident_tokens          = 0;  // B: valid row length after compaction
    std::uint32_t compaction_true_frontier = 0;  // H: true frontier at compaction
    // Full-history page table: one entry per page covering true positions [0, H), in true
    // order. Entries for demoted (host-only) blocks stay in the table so later selections
    // can promote them back; the resident row is the selected subset of these pages.
    std::vector<WorkingSetPage> pages;

    // Row-local position for a true position p >= compaction_true_frontier.
    [[nodiscard]] std::uint32_t row_local(std::uint32_t true_position) const;
};

// Selection policy: the sink prefix plus the most recent blocks that fit the budget.
// history_tokens is the committed true frontier. Returns ascending true-position block
// indices (block i covers [i*128, (i+1)*128)).
[[nodiscard]] std::vector<std::uint32_t> select_working_set_blocks(const WorkingSetPolicy& policy,
                                                                   std::uint32_t history_tokens);

// Resident token count of the compacted row: every selected block is full
// except the newest, which holds the tail of the true frontier.
[[nodiscard]] std::uint32_t working_set_resident_tokens(const std::vector<std::uint32_t>& blocks,
                                                        std::uint32_t history_tokens);

// Page count covering a token count (ceil division by the physical page size); mirrors
// LogicalKVPageStore::pages_for_tokens, which is private to the store.
[[nodiscard]] constexpr std::uint32_t working_set_pages_for_tokens(std::uint32_t tokens)
    noexcept {
    return tokens == 0 ? 0U : 1U + (tokens - 1U) / kPagedKVPageSize;
}

// Row slots (ascending) of the given true blocks within a row-order page table. Blocks must
// be ascending and present in the table; a block that left the row is an error.
[[nodiscard]] std::vector<std::uint32_t> working_set_table_slots(std::span<const WorkingSetPage> table,
                                                                 std::span<const std::uint32_t> blocks);

// Pages of the given true blocks within a full-history page table, ordered by true position.
[[nodiscard]] std::vector<LogicalKVPageHandle>
working_set_pages(std::span<const WorkingSetPage> table, std::span<const std::uint32_t> blocks);

// Same, resolved through a session's table.
[[nodiscard]] std::vector<LogicalKVPageHandle> working_set_pages(const WorkingSetSession& session,
                                                                 std::span<const std::uint32_t> blocks);

// Rebuild the full-history page table after a turn boundary. The row currently holds the
// previously selected pages in true order followed by the pages appended since the previous
// compaction; this returns the table covering [0, history) — the previous entries plus the
// committed appended pages. With an empty previous table the row must be a dense prefix.
[[nodiscard]] std::vector<WorkingSetPage> working_set_build_table(
    std::span<const WorkingSetPage> prev_table, std::uint32_t prev_frontier,
    std::uint32_t prev_resident, std::span<const LogicalKVPageHandle> row_members,
    std::uint32_t history);

// Device KV window for admission under a working-set policy (plan §4): the resident row
// never exceeds the token budget plus the tokens appended since the last trigger — one
// prefill chunk at most (inter_trigger_headroom). Short contexts pass through unchanged,
// so requests below the window keep their exact current entitlement.
[[nodiscard]] constexpr std::uint32_t working_set_device_window(const WorkingSetPolicy& policy,
                                                                std::uint32_t reserved_context_tokens,
                                                                std::uint32_t inter_trigger_headroom)
    noexcept {
    const std::uint32_t window = policy.budget_tokens + inter_trigger_headroom;
    return reserved_context_tokens < window ? reserved_context_tokens : window;
}

// Default always-resident sink for an auto-sized working set: a quarter of the resolved
// budget capped at 16K tokens, block-aligned. Explicit budgets keep their configured sink.
[[nodiscard]] constexpr std::uint32_t working_set_derive_sink(std::uint32_t budget_tokens)
    noexcept {
    const std::uint32_t quarter = budget_tokens / 4;
    const std::uint32_t capped  = quarter < 16 * 1024 ? quarter : 16 * 1024;
    return capped & ~static_cast<std::uint32_t>(kWorkingSetBlockTokens - 1);
}

// Admission-time "take the remaining pool" grant for one new session under an auto-sized
// working set (plan §4.5): the session window is the smaller of the ceiling and the device
// pages still free, minus one inter-trigger headroom chunk, clamped to [sink, ceiling] and
// block-aligned. A pool that cannot fund the minimum viable window (sink + one chunk)
// returns 0, which keeps the request demanding its full window so it FIFO-queues until the
// pool frees up instead of starting a session too small to be useful.
[[nodiscard]] constexpr std::uint32_t working_set_grant_budget(std::uint32_t ceiling,
                                                               std::uint32_t free_tokens,
                                                               std::uint32_t sink_tokens,
                                                               std::uint32_t inter_trigger_headroom)
    noexcept {
    const std::uint32_t grant_window =
        ceiling < free_tokens ? ceiling : free_tokens;
    if (grant_window < sink_tokens + inter_trigger_headroom) { return 0; }
    std::uint32_t budget = grant_window - inter_trigger_headroom;
    if (budget > ceiling) { budget = ceiling; }
    budget &= ~static_cast<std::uint32_t>(kWorkingSetBlockTokens - 1);
    if (budget < sink_tokens) { budget = sink_tokens; }
    return budget;
}

// Host arena auto-capacity when no explicit host KV capacity is configured: every
// concurrent session may park its full true history (context_tokens) on host, rounded
// up to page granularity (plan §4.4).
[[nodiscard]] constexpr std::uint64_t working_set_auto_host_capacity(
    std::uint32_t concurrency, std::uint32_t context_tokens, std::uint64_t page_stride)
    noexcept {
    const std::uint64_t pages =
        static_cast<std::uint64_t>(concurrency) * working_set_pages_for_tokens(context_tokens);
    return pages * page_stride;
}

}  // namespace ninfer::models::qwen3_5::detail
