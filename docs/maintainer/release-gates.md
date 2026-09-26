# Release gates (判据)

Checklist a delivery must pass before its archive is cut. Each gate names the check, the pass
reading, and where the evidence lives. Gates that cannot run on the current machine say so and
the limitation is recorded on the release entry — a gate skipped without reason fails the release.

## 1. Build and test

| Gate | Pass reading |
|---|---|
| Delivery build (`scripts/build_sm86.bat`) | clean exit, exes land in `dist/apps/` with DLLs; `scripts/verify_configure.bat` and `scripts/check_objs.bat` pass |
| sm_120a dev build (`scripts/build_sm120a_dev.bat`, `build_120a/`) | clean exit — required for any 5090-side smoke/regression, since `dist/apps` carries sm_86 SASS only |
| Release-gate suite (`ctest --test-dir build-test`, when the tree is configured) | zero failures |
| Code review verdict | landed **before** the archive is cut. If it lands afterwards, re-cut and re-package; the archive is cheap to regenerate, a review-after-package cycle costs three re-cuts |

Sanitizer caveat learned from a lost 50-minute run: an ASan pull must name its host-only
tests explicitly and exclude every device test — a broad regex under ASan hangs the whole run.

## 2. Artifact

- Select artifacts **by explicit path**. Never glob order, modification time, or unqualified
  "latest".
- SHA-256 recorded next to the artifact (`models/README.md`, `docs/ARTIFACT_NOTES.md`) and
  verified before delivery (e.g. the spliced ternary artifact's pinned digest). A changed size or
  digest means the source was rebuilt or republished — re-pin deliberately, do not loosen the
  check.
- Provenance recorded in [`docs/ARTIFACT_NOTES.md`](../../docs/ARTIFACT_NOTES.md): which source
  built it, which tool spliced/repaired it, and the numeric proof (see §4).

## 3. Launcher and package

- `.bat` launchers are **ASCII-only**. GBK or UTF-8-BOM encoding breaks double-click start on a
  machine whose console code page differs; the failure mode is mojibake arguments or a dead
  window. Verify by byte inspection, not by eye.
- Every launcher checks engine and artifact exist before starting and leaves the failure on
  screen.
- Package inventory matches what actually ships: every exe/DLL referenced exists in `dist/apps`
  (exe + ffmpeg DLLs + `nvcudart_hybrid64.dll`, all beside each other — a missing one dies
  silently at startup with 0xC0000061); nothing references a path outside the package folder.
- GUI launcher state (`app/config.json`) stays out of the shipped surface; runtime stats under
  `stats/` stay gitignored.

## 4. Numerical sanity

- Perplexity on the fixed corpus lands in the band measured for this artifact on a
  development card (e.g. spliced ternary 27B: **5.628** quick-1M on the 5090 dev build).
  ~16.3 is the corruption signature of the official damaged artifact and fails this gate
  outright.
- Greedy smoke through the serving route returns coherent output; no NaN/empty completions.
- Spec lanes (when shipped) report draft acceptance within the band measured for that lane in
  the release notes (e.g. DFlash2 62–64%, MTP 58–64%); a lane far outside its measured band is
  a binding or kernel regression, not variance.

## 5. Performance claims

- Measured on the shipping card with the **exact launcher arguments**, variants **interleaved**
  in the same unpinned clock window (sequential A→B runs measure the window, not the variant —
  up to ~9% drift between windows here; two false findings were killed by interleaving,
  ADR-0003). Report ≥3 repetitions per variant; publish the median, not a single shot.
- Every published number records target hardware, workload/command, and the summarized result.
  Numbers are **not transferable across cards**: an estimate for another card is an
  extrapolation and must be labeled as such, never published as a measurement.
- Op microbenchmarks establish Op results only; end-to-end claims require whole-inference
  measurement ([`bench/README.md`](../../bench/README.md),
  [measurement rules](../performance/methodology.md)).

## 6. Versioning discipline

- Publish every version you build, **in order**, or do not build it — a gap reads as a
  withdrawn release and nothing records why (this happened to 1.0.1/1.0.2).
- The version is set in the packager, and the packager's recorded value is what gets checked —
  never hand-typed at release time.
- Create release tags **while on the release branch**: `git tag -f` tags HEAD, so tagging from
  `dev` puts the pointer on the wrong commit even when the trees are identical.
- Confirm every commit actually landed (`git log -1`, `git status` after each commit).
  Multi-line messages go through `git commit -F <file>` — a long `-m` message can be misparsed
  into pathspecs and commit nothing while reporting success. Note `tools/build/` paths are
  silently ignored (`.gitignore` has unanchored `build/`).

## 7. Documentation sync

- README Windows section, RELEASE_NOTES, and the performance table move together, in the same
  measurement session that produced their numbers.
- One authority per fact: no parallel "final/v2/new-design" docs; superseded content is
  deleted, not forked.

## Failure-mode quick reference

| Symptom | Diagnosis |
|---|---|
| Engine refuses artifact at startup with a magic/version error | v2 container on a v3 engine — upgrade offline with the v2→v3 tool (from the 5090 release line; it preserves weight bytes) |
| PPL ~16.3 (or 1e7) | corrupted/misplaced weight core; rebuild via the splice path, verify against GGUF source tensor-by-tensor |
| Download refused on size | upstream republished; refresh the pinned hash from blob metadata |
| Launcher double-click dies or args arrive mangled | bat file encoding (non-ASCII) or a missing-file check that exits silently |
| PowerShell script "succeeded" but wrote an empty file | inline quoting ate the content; check output byte size, never just exit code |
| Review/sanitizer verdict lands after packaging | re-cut the archive; do not ship the pre-review build |
| `--wddm-evictable-budget`: startup FATAL "D3D12 residency arena failed read-back verification" followed by `cudaErrorLaunchFailure`, or an access violation inside `D3D12CreateDevice` (seen once on a multi-adapter RTX 5090 workstation with wedged driver state after force-killing heavy CUDA apps) | D3D12↔CUDA shared-memory interop is environment-sensitive; set `NINFER_DISABLE_D3D12_ARENA=1` to skip the D3D12 arena entirely (checked before any D3D call, so it also prevents the CreateDevice AV) while keeping the evictable-budget accounting on plain `cudaMalloc`. Flag OFF has zero D3D12 involvement and is unaffected |
