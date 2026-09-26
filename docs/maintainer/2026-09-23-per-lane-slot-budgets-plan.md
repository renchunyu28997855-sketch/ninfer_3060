# Per-lane slot budgets (WS "sized slots" mode)

Dated plan — active work. Remove when shipped or abandoned.

## Goal

Let the operator pin each concurrency lane to a **fixed share of the device KV pool**,
so N concurrent sessions get predictable, independently-sized working sets instead of the
arrival-order take-remaining behaviour of `--kv-working-set auto`. Both equal and uneven
splits must work; the shares may sum to ≤ 100 % (slack allowed).

## Semantics (fourth WS mode)

Modes today: `off` (ws=0), `auto` (take-remaining grant), `explicit-uniform` (`--kv-working-set N`).
New: **per-lane** via `--kv-slot-percentages "p0,p1,...,pN"`.

- `N` must equal `max_concurrency`; each `p_i ∈ (0,100]`; Σ `p_i ≤ 100`.
- Each `p_i` is a % of the **device KV pool** (the resolved `kv_capacity`, which may itself be `auto`).
- Resolved at startup (ProgramImpl ctor, where pool tokens are known):
  `lane_budgets_[i] = align_block(pool_tokens * p_i / 100)` (block = 128 tokens).
- Validation: count == concurrency, each > 0, and Σ(lane_budgets_) + N·prefill_chunk ≤ pool
  (each lane's resident window includes one prefill-chunk headroom).
- Per-lane sink = `working_set_derive_sink(lane_budgets_[i])`, clamped ≤ lane budget.
- Plan-time global policy budget = max(lane_budgets_) (a safe fit-check upper bound); the
  auto-grant refresh path is disabled in per-lane mode.
- Prefill: for every request on a lane (fresh, reuse, fork), override
  `sequence.working_set_budget = lane_budgets_[sequence.lane]` before the staging window is
  computed. This single knob drives `working_set_policy_for` → entitlement → window → swap.

### Why this is small
All heavy machinery (window/entitlement/compaction/swap) already keys off
`SequenceState::working_set_budget`; `SequenceState::lane` already exists and is stable per lane.
Per-lane mode only adds a ceiling table + one override point at prefill.

## Files

| File | Change |
|---|---|
| `include/ninfer/types.h` | `WorkingSetConfig` += `bool per_lane=false` (+ optional resolved table for reporting) |
| `src/serve/serve_options.{h,cpp}` | parse + validate `--kv-slot-percentages`; mutual-exclude with auto/explicit; usage text |
| `src/serve/generation_service.cpp` | plumb percentages into engine options |
| `src/models/qwen3_5/program/planning/startup.{h,cpp}` | carry percentages through inputs/plan |
| `src/models/qwen3_5/program/program_impl.{h,cpp}` | `lane_budgets_` array + `working_set_per_lane_` + `lane_count_`; resolve in ctor; accessor; gate auto-grant off |
| `src/models/qwen3_5/program/prefill.cpp` | one override point before staging window |
| `src/product/logging/startup_log.cpp` | log resolved per-lane table |
| launcher `app_ws/` + `app/` | WS group "sized slots" option + N % fields + even-split + live token calc + i18n |
| tests | percentage parse/resolution unit test; serve_options flag test; build green |

## Decisions

- Default lane assignment stays arrival-order (no session→lane pinning in v1).
- Fixed slots trade away auto's dynamic borrowing (a short session's spare VRAM is not lent to a
  neighbour) — accepted for predictability.
- Launcher shows both the % and the computed token count per slot (needs a pool value; if
  `--kv-capacity auto`, the calculator uses the last resolved pool from the engine's startup log).
