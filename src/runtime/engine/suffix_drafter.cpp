#include "runtime/engine/suffix_drafter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace ninfer::runtime {

namespace {

// Deterministic 64-bit mix over a token gram: FNV-1a seeded by the length so
// grams of different lengths never collide on purpose. Collision safety does
// not rely on the hash: propose() re-verifies the bytes at the hit position.
std::uint64_t fnv1a(const TokenId* tokens, std::uint32_t length) noexcept {
    std::uint64_t h = 1469598103934665603ULL ^ static_cast<std::uint64_t>(length);
    for (std::uint32_t i = 0; i < length; ++i) {
        h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(tokens[i]));
        h *= 1099511628211ULL;
    }
    // Final avalanche so that short-gram patterns do not cluster.
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h;
}

std::size_t next_power_of_two(std::size_t n) noexcept {
    std::size_t p = 1;
    while (p < n) { p <<= 1; }
    return p;
}

} // namespace

SuffixDrafter::SuffixDrafter(std::uint32_t min_match, std::uint32_t max_gram)
    : min_match_(min_match), max_gram_(max_gram) {
    if (min_match == 0 || max_gram < min_match) {
        throw std::invalid_argument("SuffixDrafter requires 1 <= min_match <= max_gram");
    }
    if (max_gram > 64) {
        throw std::invalid_argument("SuffixDrafter max_gram exceeds 64");
    }
    table_.assign(next_power_of_two(16), Entry{});
    mask_ = table_.size() - 1;
}

void SuffixDrafter::rebuild(std::span<const TokenId> history) {
    reset();
    history_.reserve(history.size());
    for (std::uint32_t position = 0; position < static_cast<std::uint32_t>(history.size());
         ++position) {
        history_.push_back(history[position]);
        insert_grams_ending_at(position);
    }
}

void SuffixDrafter::append(TokenId token) {
    const std::uint32_t position = static_cast<std::uint32_t>(history_.size());
    history_.push_back(token);
    insert_grams_ending_at(position);
}

void SuffixDrafter::reset() noexcept {
    history_.clear();
    std::fill(table_.begin(), table_.end(), Entry{});
    occupied_ = 0;
}

std::uint64_t SuffixDrafter::mix(const TokenId* tokens, std::uint32_t length) noexcept {
    return fnv1a(tokens, length);
}

void SuffixDrafter::insert_grams_ending_at(std::uint32_t position) {
    for (std::uint32_t length = min_match_; length <= max_gram_; ++length) {
        if (position + 1 < length) { break; }  // not enough history yet
        const TokenId* gram = history_.data() + (position + 1 - length);
        const std::uint64_t key = mix(gram, length);
        std::size_t slot = key & mask_;
        for (;;) {
            Entry& entry = table_[slot];
            if (!entry.occupied) {
                entry.key     = key;
                entry.start   = position + 1 - length;
                entry.length  = length;
                entry.occupied = true;
                ++occupied_;
                break;
            }
            if (entry.key == key) {
                // Most-recent occurrence wins: it carries the followers that
                // matter for the next proposal. Keep the previous one as the
                // tail guard (the newest occurrence may BE the tail itself).
                entry.prev_start = entry.start;
                entry.start  = position + 1 - length;
                entry.length  = length;
                break;
            }
            slot = (slot + 1) & mask_;
        }
    }
    if (occupied_ * 10 >= table_.size() * 7) {
        rehash(table_.size() * 2);
    }
}

void SuffixDrafter::rehash(std::size_t new_capacity) {
    std::vector<Entry> old = std::move(table_);
    table_.assign(next_power_of_two(new_capacity), Entry{});
    mask_     = table_.size() - 1;
    occupied_ = 0;
    for (const Entry& entry : old) {
        if (!entry.occupied) { continue; }
        std::size_t slot = entry.key & mask_;
        for (;;) {
            if (!table_[slot].occupied) {
                table_[slot] = entry;
                ++occupied_;
                break;
            }
            slot = (slot + 1) & mask_;
        }
    }
}

const SuffixDrafter::Entry* SuffixDrafter::probe(std::uint64_t key) const {
    std::size_t slot = key & mask_;
    for (;;) {
        const Entry& entry = table_[slot];
        if (!entry.occupied) { return nullptr; }
        if (entry.key == key) { return &entry; }
        slot = (slot + 1) & mask_;
    }
}

SuffixDrafter::Proposal SuffixDrafter::propose() const {
    const std::size_t n = history_.size();
    if (n < min_match_) { return Proposal{}; }
    const TokenId* tail = history_.data() + n;
    for (std::uint32_t length = max_gram_; length >= min_match_; --length) {
        if (n <= length) { continue; }  // no earlier room for followers
        const std::uint64_t key = mix(tail - length, length);
        const Entry* entry = probe(key);
        if (entry == nullptr) { continue; }
        std::uint32_t start = entry->start;
        if (start + length >= n) {
            // The newest occurrence is the tail itself (no followers there);
            // fall back to the earlier occurrence.
            if (entry->prev_start == kNoPrev) { continue; }
            start = entry->prev_start;
            if (start + length >= n) { continue; }
        }
        // Byte-verify the hit (guards hash collisions and same-key entries).
        // Window pointer avoids unsigned-negative indexing: with uint32
        // operands, `tail[-length + i]` would underflow to a huge offset.
        const TokenId* window = tail - length;
        bool match = true;
        for (std::uint32_t i = 0; i < length; ++i) {
            if (history_[start + i] != window[i]) { match = false; break; }
        }
        if (!match) { continue; }
        const std::uint32_t followers =
            static_cast<std::uint32_t>(n - (start + length));
        const std::uint32_t proposed = std::min(max_gram_, followers);
        if (proposed == 0) { continue; }
        return Proposal{history_.data() + start + length, proposed, length};
    }
    return Proposal{};
}

// ---------------------------------------------------------------------------

void SuffixDraftPolicy::observe_suffix(std::uint32_t accepted, std::uint32_t window,
                                      std::uint32_t match_length) {
    const double rate = window == 0 ? 0.0
                                    : static_cast<double>(accepted) / static_cast<double>(window);
    suffix_ema_ = kEmaDecay * suffix_ema_ + (1.0 - kEmaDecay) * rate;
    // The first sample of a bucket sets it outright: a bucket that starts from
    // zero would need ~10 blended rounds to become trustworthy, and the
    // exploration phase only spends kExplorationRounds rounds gathering them.
    const std::size_t bucket = bucket_index(match_length);
    bucket_ema_[bucket] = bucket_seeded_[bucket]
                              ? kEmaDecay * bucket_ema_[bucket] + (1.0 - kEmaDecay) * rate
                              : rate;
    bucket_seeded_[bucket] = true;
    ++bucket_samples_[bucket];
    best_match_ = std::max(best_match_, match_length);
    ++observations_;
    bucket_last_sampled_[bucket] = observations_;
}

void SuffixDraftPolicy::observe_learned(std::uint32_t accepted, std::uint32_t window) {
    const double rate = window == 0 ? 0.0
                                    : static_cast<double>(accepted) / static_cast<double>(window);
    learned_ema_ = kEmaDecay * learned_ema_ + (1.0 - kEmaDecay) * rate;
    ++observations_;
}

SuffixDraftPolicy::Decision SuffixDraftPolicy::decide(std::uint32_t match_length) const {
    if (match_length < 2) { return Decision::No; }
    // The comparison needs a learned baseline; warmup rounds are learned rounds
    // because this predicate is the only way a suffix round can happen.
    if (observations_ < kWarmupRounds) { return Decision::Warmup; }
    const std::size_t bucket = bucket_index(match_length);
    // Unmeasured match length: the match itself is the only evidence available,
    // so a long one is worth spending verify rounds to measure.
    if (bucket_samples_[bucket] < kExplorationRounds) {
        return match_length >= kStrongMatchLength ? Decision::Probe : Decision::Floor;
    }
    const double expected = bucket_ema_[bucket];
    if (expected < kMinimumSuffixAcceptance) {
        // A locked bucket produces no samples while rejected, so without this
        // cadence a regime change can never pull it back above the floor. Re-
        // probe at most once per kReexploreInterval observations.
        if (observations_ - bucket_last_sampled_[bucket] >= kReexploreInterval) {
            return Decision::Probe;
        }
        return Decision::Floor;
    }
    if (expected < learned_ema_ * (1.0 + kAdoptionMargin)) { return Decision::Margin; }
    return Decision::Adopt;
}

} // namespace ninfer::runtime
