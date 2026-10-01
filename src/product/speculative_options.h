#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

// Auto-enablement policy: under the MTP and DFlash backends the zero-parameter
// suffix draft source is on by default; its per-sequence cost model gates actual
// use per round, so enabling it costs nothing on workloads where it never pays
// off. --no-spec-suffix forces it off (baseline A/B runs). The None backend never
// uses it.
inline void normalize_speculative_options(SpeculativeOptions& options) {
    switch (options.backend) {
    case SpeculativeBackend::Mtp:
    case SpeculativeBackend::DFlash:
    case SpeculativeBackend::DFlash2:
        options.suffix_drafter = !options.no_suffix_drafter;
        break;
    case SpeculativeBackend::None:
        options.suffix_drafter = false;
        break;
    }
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.draft_tokens == 0 || options.draft_tokens > 5) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,5]");
        }
        if (options.suffix_drafter &&
            (options.suffix_min_match < 2 || options.suffix_min_match > options.draft_tokens)) {
            throw std::invalid_argument(
                "suffix min-match must be in [2,draft-tokens]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        if (options.suffix_drafter &&
            (options.suffix_min_match < 2 || options.suffix_min_match > options.draft_tokens)) {
            throw std::invalid_argument(
                "suffix min-match must be in [2,draft-tokens]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        if (options.suffix_drafter &&
            (options.suffix_min_match < 2 || options.suffix_min_match > options.draft_tokens)) {
            throw std::invalid_argument(
                "suffix min-match must be in [2,draft-tokens]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
