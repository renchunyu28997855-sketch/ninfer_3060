#include "models/qwen3_5/program/planning/working_set.h"

#include <cstddef>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

std::vector<std::uint32_t> select_working_set_blocks(const WorkingSetPolicy& policy,
                                                     std::uint32_t history_tokens) {
    if (policy.budget_tokens == 0 || policy.sink_tokens > policy.budget_tokens ||
        policy.budget_tokens % kWorkingSetBlockTokens != 0 ||
        policy.sink_tokens % kWorkingSetBlockTokens != 0) {
        throw std::invalid_argument("working-set policy must be block-aligned with budget >= sink");
    }
    const std::uint32_t total_blocks  = (history_tokens + kWorkingSetBlockTokens - 1) /
                                       kWorkingSetBlockTokens;
    const std::uint32_t budget_blocks = policy.budget_tokens / kWorkingSetBlockTokens;
    const std::uint32_t sink_blocks   = policy.sink_tokens / kWorkingSetBlockTokens;

    std::vector<std::uint32_t> blocks;
    if (total_blocks <= budget_blocks) {
        blocks.resize(total_blocks);
        for (std::uint32_t i = 0; i < total_blocks; ++i) { blocks[i] = i; }
        return blocks;
    }
    const std::uint32_t recent_begin = total_blocks - (budget_blocks - sink_blocks);
    for (std::uint32_t i = 0; i < sink_blocks; ++i) { blocks.push_back(i); }
    for (std::uint32_t i = recent_begin; i < total_blocks; ++i) { blocks.push_back(i); }
    return blocks;
}

std::uint32_t working_set_resident_tokens(const std::vector<std::uint32_t>& blocks,
                                           std::uint32_t history_tokens) {
    if (blocks.empty()) { return 0; }
    const std::uint32_t last    = blocks.back();
    const std::uint32_t tail    = history_tokens - last * kWorkingSetBlockTokens;
    return (static_cast<std::uint32_t>(blocks.size()) - 1) * kWorkingSetBlockTokens + tail;
}

std::vector<std::uint32_t> working_set_table_slots(std::span<const WorkingSetPage> table,
                                                   std::span<const std::uint32_t> blocks) {
    if (blocks.size() > 1) {
        for (std::size_t i = 1; i < blocks.size(); ++i) {
            if (blocks[i - 1] >= blocks[i]) {
                throw std::invalid_argument("working-set block list must be ascending");
            }
        }
    }
    std::vector<std::uint32_t> slots;
    slots.reserve(blocks.size() * kWorkingSetBlockPages);
    std::size_t cursor = 0;
    for (const std::uint32_t block : blocks) {
        while (cursor < table.size() && table[cursor].true_block < block) { ++cursor; }
        const std::size_t begin = cursor;
        while (cursor < table.size() && table[cursor].true_block == block) { ++cursor; }
        if (cursor == begin) {
            throw std::logic_error("working-set selection reached an evicted block");
        }
        for (std::size_t i = begin; i < cursor; ++i) {
            slots.push_back(static_cast<std::uint32_t>(i));
        }
    }
    return slots;
}

std::uint32_t WorkingSetSession::row_local(std::uint32_t true_position) const {
    if (true_position < compaction_true_frontier) {
        throw std::logic_error("working-set row_local called below the compaction frontier");
    }
    return resident_tokens + (true_position - compaction_true_frontier);
}

std::vector<LogicalKVPageHandle> working_set_pages(std::span<const WorkingSetPage> table,
                                                   std::span<const std::uint32_t> blocks) {
    const auto slots = working_set_table_slots(table, blocks);
    std::vector<LogicalKVPageHandle> pages;
    pages.reserve(slots.size());
    for (const std::uint32_t slot : slots) { pages.push_back(table[slot].page); }
    return pages;
}

std::vector<LogicalKVPageHandle> working_set_pages(const WorkingSetSession& session,
                                                   std::span<const std::uint32_t> blocks) {
    return working_set_pages(session.pages, blocks);
}

std::vector<WorkingSetPage> working_set_build_table(
    std::span<const WorkingSetPage> prev_table, std::uint32_t prev_frontier,
    std::uint32_t prev_resident, std::span<const LogicalKVPageHandle> row_members,
    std::uint32_t history) {
    const std::uint32_t target = working_set_pages_for_tokens(history);
    if (prev_table.empty()) {
        // First compaction: the row is a dense prefix — plus possibly uncommitted
        // window-slack pages that staging mapped ahead of the committed frontier. The table
        // covers [0, history) only; the trailing slack pages carry no data and leave the row
        // with the remap.
        if (row_members.size() < target) {
            throw std::logic_error("working-set row is not a dense prefix before first compaction");
        }
        std::vector<WorkingSetPage> table;
        table.reserve(target);
        for (std::size_t slot = 0; slot < target; ++slot) {
            table.push_back({static_cast<std::uint32_t>(slot / kWorkingSetBlockPages),
                             row_members[slot]});
        }
        return table;
    }
    if (prev_frontier > history) {
        throw std::logic_error("working-set history regressed below the compaction frontier");
    }
    if (prev_table.size() != working_set_pages_for_tokens(prev_frontier)) {
        throw std::logic_error("working-set page table does not cover its compaction frontier");
    }
    // Appended pages follow the previously resident pages in row order; they cover true
    // positions [prev_frontier, ...) one page per 64 tokens. Entry i of the dense true-
    // coordinate table covers tokens [64i, 64i+64), so its block label is i/2 — deriving it
    // from prev_frontier would mislabel every appended entry whenever the compaction frontier
    // is not block-aligned, and the ascending block scan would skip the tail block.
    const std::uint32_t resident_pages = working_set_pages_for_tokens(prev_resident);
    const std::size_t appended         = row_members.size() - static_cast<std::size_t>(resident_pages);
    if (appended < target - prev_table.size()) {
        throw std::logic_error("working-set row lost pages since the previous compaction");
    }
    std::vector<WorkingSetPage> table(prev_table.begin(), prev_table.end());
    table.reserve(target);
    for (std::size_t i = prev_table.size(); i < target; ++i) {
        const std::size_t row_slot = resident_pages + (i - prev_table.size());
        table.push_back({static_cast<std::uint32_t>(i / kWorkingSetBlockPages),
                         row_members[row_slot]});
    }
    return table;
}

}  // namespace ninfer::models::qwen3_5::detail
