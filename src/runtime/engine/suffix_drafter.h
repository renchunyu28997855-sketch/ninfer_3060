#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace ninfer::runtime {

// Zero-parameter host-side draft source for speculative decoding.
//
// Maintains an n-gram table over the sequence's own committed token history
// (prompt + generated tokens) and proposes the tokens that followed the
// longest matching suffix of the current prefix. The proposal is a pure
// function of the token stream, so it is fully deterministic and carries no
// model weights, no device memory, and no extra forward pass: drafting cost
// is a hash probe plus a bounded copy. Verification of the proposal reuses
// the existing target-model verify pass, which makes the output distribution
// identical to drafting from any other source (the target model licenses the
// accepted prefix exactly).
//
// Memory: one history copy (2 bytes/token) plus open-addressed gram entries
// for each length in [min_match, max_gram] ending at every position
// (~24 B/gram). At 256K context with min_match=4, max_gram=5 this is on the
// order of tens of MB per active sequence in host RAM, and the whole state is
// derived: it can be rebuilt from the ledger in O(n * (max_gram - min_match + 1)).
class SuffixDrafter {
public:
    struct Proposal {
        const TokenId* tokens = nullptr;  // points into internal history
        std::uint32_t length        = 0;  // proposed tokens (0 = no usable match)
        std::uint32_t match_length  = 0;  // depth of the matched suffix (>= min_match)
    };

    SuffixDrafter(std::uint32_t min_match, std::uint32_t max_gram);

    // Seed or reset from a full history (request start, continuation resume).
    void rebuild(std::span<const TokenId> history);

    // Append one newly committed token (decode commit path).
    void append(TokenId token);

    void reset() noexcept;

    // Longest-match proposal at the tail of the current history: the most
    // recent earlier occurrence of the longest matching suffix, capped at
    // max_gram proposed tokens. Empty when no gram of length >= min_match
    // has an earlier occurrence with followers.
    [[nodiscard]] Proposal propose() const;

    [[nodiscard]] std::size_t history_size() const noexcept { return history_.size(); }
    [[nodiscard]] std::size_t entry_count() const noexcept { return occupied_; }
    [[nodiscard]] std::uint32_t min_match() const noexcept { return min_match_; }
    [[nodiscard]] std::uint32_t max_gram() const noexcept { return max_gram_; }

private:
    // Sentinel for "no earlier occurrence" in Entry::prev_start.
    static constexpr std::uint32_t kNoPrev = std::numeric_limits<std::uint32_t>::max();

    struct Entry {
        std::uint64_t key   = 0;
        std::uint32_t start      = 0;      // most recent gram start in history_
        std::uint32_t prev_start = kNoPrev; // occurrence before `start` (tail guard)
        std::uint8_t  length   = 0;        // gram length
        bool          occupied = false;
    };

    static std::uint64_t mix(const TokenId* tokens, std::uint32_t length) noexcept;
    void insert_grams_ending_at(std::uint32_t position);
    const Entry* probe(std::uint64_t key) const;
    void rehash(std::size_t new_capacity);

    std::uint32_t min_match_;
    std::uint32_t max_gram_;
    std::vector<TokenId> history_;
    std::vector<Entry> table_;
    std::size_t occupied_ = 0;
    std::size_t mask_     = 0;
};

// Per-sequence cost-model gate choosing between the zero-cost suffix source
// and the learned drafter for the next verify window.
//
// Both sources spend the same verify budget, so the dominant term of round cost
// is how many drafted tokens survive verification. Each source keeps an
// exponential moving average of accepted/window; the suffix source is adopted
// while its measured acceptance leads the learned drafter's by a margin.
//
// Acceptance is measured per match length. The longer the match, the more of the
// earlier context a proposal reproduces, and the more likely the target model
// follows it; averaging every match together hides that, so each length keeps
// its own EMA and the gate compares like with like.
//
// Measurement has a bootstrap problem: a source that is never adopted is never
// measured, and an unmeasured source cannot win a comparison. The first
// `kExplorationRounds` rounds of a long match are therefore adopted on the
// strength of the match itself, which gives that bucket its first samples;
// afterwards its own numbers decide. Without this the source stays locked out
// for the whole sequence whenever the learned drafter happens to be strong.
//
// The symmetric failure is lockout: a bucket whose EMA falls under the floor
// produces no samples while rejected, so a workload regime change (the context
// distribution shifting back to the one that suited it) can never pull it
// back above the floor within the sequence's lifetime. Buckets therefore go
// stale: after `kReexploreInterval` observations without a sample, a locked
// bucket is re-probed once, bounding the cost to one verify round per interval
// per locked bucket while keeping its measurement fresh enough to track the
// regime.
class SuffixDraftPolicy {
public:
    // Gate-decision outcome for a proposal of the given match length.
    enum class Decision : std::int32_t {
        No      = 0,  // no usable match (shorter than two tokens)
        Warmup  = 1,  // fewer than kWarmupRounds observations so far
        Probe   = 2,  // unmeasured bucket; adopted on match strength to gather samples
        Floor   = 3,  // measured bucket acceptance below kMinimumSuffixAcceptance
        Margin  = 4,  // measured, but does not lead the learned drafter by the margin
        Adopt   = 5,  // measured and leads
    };

    // Defaults chosen from the Strata prompt-lookup cost model: 3% adoption
    // margin and ~6-7 round EMA half-life. After local qualification the
    // measured acceptance rates and end-to-end delta are recorded here.
    static constexpr double kAdoptionMargin    = 0.03;
    static constexpr double kEmaDecay          = 0.90;
    static constexpr std::uint32_t kWarmupRounds = 8;
    // Never adopt a suffix source whose recent acceptance falls below this.
    static constexpr double kMinimumSuffixAcceptance = 0.50;
    // Exploration threshold: unmeasured buckets at or above this length get
    // kExplorationRounds of probe samples before their own numbers decide
    // adoption. Set to 2 so every bucket the drafter can propose (default
    // min_match=4, i.e. length 4) is measurable; shorter buckets stay dead
    // unless --suffix-min-match is lowered further.
    static constexpr std::uint32_t kStrongMatchLength = 2;
    // Samples a bucket needs before its own numbers decide adoption.
    static constexpr std::uint32_t kExplorationRounds = 4;
    // Cadence for re-probing a floor-locked bucket: at most one probe per this
    // many observations. At kEmaDecay=0.9 five consecutive good samples cross
    // the floor from a locked EMA, so 64 bounds probe overhead to 1/64 of the
    // gate's decisions per locked bucket while recovering within a few hundred
    // rounds of a regime shift.
    static constexpr std::uint32_t kReexploreInterval = 64;
    // Match lengths 2..7 measure separately; longer matches share the last.
    static constexpr std::size_t kMatchBuckets = 6;

    void observe_suffix(std::uint32_t accepted, std::uint32_t window, std::uint32_t match_length);
    void observe_learned(std::uint32_t accepted, std::uint32_t window);

    // Full gate decision with the rejection reason, for observability counters.
    [[nodiscard]] Decision decide(std::uint32_t match_length) const;

    // True when the suffix source (with a match of at least `match_length`)
    // should supply the next verify window.
    [[nodiscard]] bool use_suffix(std::uint32_t match_length) const {
        const Decision d = decide(match_length);
        return d == Decision::Adopt || d == Decision::Probe;
    }

    // Observability (tests, reporting).
    [[nodiscard]] double suffix_acceptance_ema() const { return suffix_ema_; }
    [[nodiscard]] double learned_acceptance_ema() const { return learned_ema_; }
    [[nodiscard]] double bucket_acceptance(std::uint32_t match_length) const {
        return bucket_ema_[bucket_index(match_length)];
    }
    [[nodiscard]] std::uint32_t bucket_samples(std::uint32_t match_length) const {
        return bucket_samples_[bucket_index(match_length)];
    }
    [[nodiscard]] std::uint32_t bucket_last_sampled(std::uint32_t match_length) const {
        return bucket_last_sampled_[bucket_index(match_length)];
    }
    [[nodiscard]] std::uint32_t observations() const { return observations_; }
    [[nodiscard]] std::uint32_t best_match_length() const { return best_match_; }

private:
    // Match length 2..7 map to their own bucket; longer matches share the last.
    [[nodiscard]] static constexpr std::size_t bucket_index(std::uint32_t match_length) {
        if (match_length < 2) { return 0; }
        if (match_length > kMatchBuckets + 1) { return kMatchBuckets - 1; }
        return static_cast<std::size_t>(match_length) - 2;
    }

    double suffix_ema_  = 0.0;
    double learned_ema_ = 0.0;
    double bucket_ema_[kMatchBuckets] = {};
    std::uint32_t bucket_samples_[kMatchBuckets] = {};
    bool bucket_seeded_[kMatchBuckets] = {};
    std::uint32_t bucket_last_sampled_[kMatchBuckets] = {};
    std::uint32_t observations_ = 0;
    std::uint32_t best_match_   = 0;
};

} // namespace ninfer::runtime
