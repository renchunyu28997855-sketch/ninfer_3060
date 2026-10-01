# NInfer 3060 树交接文档（HANDOVER）

> **本次更新：2026-10-01**。

## 0. 一句话现状
5090 主树的 **suffix-lookup drafter 全栈已移植到本树**并提交推送（`4613bf2`，26 文件 +905/-41），
sm_86 全量构建通过（`build_86/apps/ninfer-serve.exe`，192MB）。**本地无 3060 显卡，未上机验证**——
CUDA 侧新 op 只过了编译。本机目录已清理：`E:\download\123` 下只剩本树（见 §6）。

## 1. Git 状态
- **HEAD = `4613bf2`**（已推送 origin/main）：
  `feat(speculative): 移植 suffix-lookup drafter 全栈（5090 谱系）到 sm_8x`
- 前一提交 `3c43798`（host-KV working set 移植）是移植前基线。
- 工作树源码干净，但本节之后的目录整理留有**未提交改动**（§6.2 列全）：6 个修改文件 +
  2 个删除的 bat + 3 个未跟踪文档（HANDOVER.md / .port_note.md / build_86_run.bat）。

## 2. 移植内容（`4613bf2`）
### 2.1 逐字复制自 5090 树（逻辑不变，单测 12/12 在 5090 树验证过）
- `src/runtime/engine/suffix_drafter.{h,cpp}`：n-gram 历史 + 桶级 EMA 门控（warmup/floor/margin/adoption）
  + re-exploration 周期（`kReexploreInterval=64`）+ `kStrongMatchLength=4`；
- `include/ninfer/ops/suffix_drafter.h`（op 门面）。

### 2.2 模型侧接线
- `prefill.cpp`：`seed_suffix_drafter` + 两处 `append_suffix_tokens`（投机提交循环 + 首生成 token）——
  这是 5090 树上修过的**历史喂入回归**（`4d64c7fe`），本树直接带上修复版；
- `decode.cpp`：MTP / DFlash 双路径接线（gate 决策 → host proposal 填充 → 计数器 → observe）。

### 2.3 ops 层（用户补充的 per-row host-proposal 管道，本树新增 CUDA 代码）
- 新 op `speculative_apply_host_proposals`（`src/ops/{kernel,launcher,wrapper}` + `include/ninfer/ops/speculative_round.h`）：
  把每行最多 16 个 host 提议 token 写入 DFlash 草稿槽位，one-hot `q = (P==j ? 1 : 0)`（float 精确）接受语义；
- 数据路径（已逐环核对）：`decode.cpp` host fill → `memcpy D2H 1024B`（正好 sizeof 结构）→
  `draft.cpp` `memcpy H2D` 写回 → op 消费（`extent ∈ [0,16]` clamp 安全）；
- MTP 分支 host proposal 恒空，行为不变；图捕获兼容（fill 在 capture 外，kernel 在 capture 内）。

### 2.4 产品/服务层
- `types.h`：`SpeculativeStats` +12 个 `suffix_*` 字段；`SpeculativeOptions` +`suffix_drafter`/`no_suffix_drafter`/`suffix_min_match`；
- `speculative_options.h`：`normalize_speculative_options`（mtp/dflash/dflash2 默认自动启用，None 关闭）+ min-match 校验；
- `serve_options.cpp`：`--no-spec-suffix` / `--suffix-min-match N` 两个 flag；
- `generation_service.{h,cpp}` + `request_log.cpp`：12 个 `speculative_suffix_*` 指标贯穿到 JSONL。

### 2.5 架构差异（3060 vs 5090，移植时有意保留的差异）
- 本树**没有** model config 路线（无 `core/config.h` / `model.h` 的 suffix 字段）：
  走 `SpeculativeOptions → planning 直连`（`startup.cpp` 读 `options.speculative.suffix_*`），设计如此，不是遗漏；
- 本树 Program 持有直接成员 ingress（无 5090 的 PrefillIngress 抽象层）；bind 返回 `Program*`；
- 本树**无 tests/ 基建**：suffix 单测未随移植（逻辑与 5090 逐字相同，已在 5090 树 12/12 通过）。

## 3. 构建
- 配方（`build_86_run.bat`，已验证）：VS2022 BuildTools vcvars64 + CUDA v13.1 + Ninja，
  `cmake -B build_86 -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release`，`ninja -C build_86 -j 8`；
- 当前状态：**BUILD OK**（675/675，零错误），产物 `build_86/apps/ninfer-serve.exe`（2026-10-01 10:33）。
- 旧构建日志（`build_86_build.log` 等）已随清理删除（§6）；重建直接跑 `build_86_run.bat`（已改为
  `cd /d "%~dp0"`，位置无关，可随目录移动）。

## 4. 开放事项
1. **上机验证（最高优先）**：本地无 3060，新 op 的 CUDA 路径未经实卡运行。建议首个请求用强重复负载
   （prompt 内文本出现两次 + 要求复述），查 stats/JSONL 的 `suffix_rounds > 0` 且 `suffix_accepted > 0` 即为通；
   对照 5090 树经验：强重复负载学习式 ~74% 接受、后缀 ~90% 接受，合成负载总吞吐基本持平（学习式已是甜点）。
2. 若 5090 树后续再改 suffix 逻辑（如 cost-aware 窗口选择、EMA 持久化），需手工同步到本树并重新 sm_86 构建。
3. 可选：给本树补 tests/ 基建 + `ninfer_suffix_drafter_test`（参考 5090 树 `tests/cmake/RuntimeTests.cmake` 注册方式）。

## 5. 关键文件索引
| 路径 | 作用 |
|---|---|
| `src/runtime/engine/suffix_drafter.{h,cpp}` | drafter 核心（逐字自 5090） |
| `src/models/qwen3_5/program/decode.cpp` | MTP/DFlash 接线 + host proposal 填充 |
| `src/models/qwen3_5/program/prefill.cpp` | 种子 + 历史喂入 |
| `src/ops/{kernel,launcher,wrapper}/*speculative_round*` | per-row host-proposal 管道（本树新增） |
| `src/product/speculative_options.h` | 默认启用策略 + 校验 |
| `src/serve/{serve_options,generation_service,request_log}.cpp` | flag / 指标 / JSONL |
| `.port_note.md` | 移植过程备忘（本地，未入库） |
| `build_86_run.bat` | sm_86 一键构建脚本（本地，未入库；已改 `%~dp0` 位置无关） |
| `models/Swift-Bonsai-2-ninfer-v3-aux.ninfer` | Swift aux 工件（8,307,236,592 B，PPL overall 5.713） |
| `models/pack-tools/` | Swift 打包工具链（pack.py 等，自旧 `model\` 目录迁入） |

## 6. 目录清理与整理（2026-10-01 本会话）
### 6.1 树外（`E:\download\123` 现只剩 `ninfer_3060`）
- **删**（~85GB）：`model/`（82G，Ternary 全系 + Swift 源 GGUF/v2/v3）、`ninfer-5090-sm86`、
  `ninfer-5090-windows-main`、`ninfer-3060-win`、`ninfer-3060X2-master`、`ninfer-ada-ternary`、
  `ninfer-ternary-bonsai-ada-master`、`patches/`、`d3d_probe/`、`极速档（ninfer）/`、顶层 `_splice_ternary.py`
  与 4× `embed_dump.bin.*`。
- **留进树**：`Swift-Bonsai-2-ninfer-v3-aux.ninfer` → `models/`；`model/tools`（packer 工具链）→
  `models/pack-tools/`（含 LICENSE/NOTICE/README/swift-bonai2-hadamard-meta.json）。
- ⚠️ splice 输入源件（官方 v3 `.ninfer` + 三值化 GGUF）**本机已无任何副本**，再生三值化工件前需重新下载。

### 6.2 树内未提交改动（新会话第一件事建议先 commit）
- 修改：`README.md`（dist/apps 与模型路径去外部引用 + GUI 启动器改 `py -3` 可移植写法）、
  `app/ninfer_launcher.pyw`（去绝对路径 shebang）、`docs/ARTIFACT_NOTES.md`（spliced/packer 改 `models\` 路径）、
  `models/README.md`（补 Swift aux 条目）、`tools/splice_ternary.py`（REPO_ROOT 相对路径 + sys.path 指 pack-tools）、
  `.gitignore`（追加 `build_86/`、`ppl_*.log`、`ws_*.log`、`.decode_diff.txt`、`.pi/`）
- 删除：`scripts/run_ppl_dev.bat`、`scripts/check_objs.bat`（指向已删的 5090 树）
- 未跟踪保留：`HANDOVER.md`、`.port_note.md`、`build_86_run.bat`

### 6.3 树内调试残留清理（已执行，均未被 git 跟踪）
- 删 21 个开发日志（`ppl_*.log`×8、`ws_*.log`、`build_*`/`serve_wddm_env.log`、`.decode_diff.txt`）
  + `smoke/`（734MB WDDM 冒烟沙盒，可从 `dist/apps` 再生）+ `dist.rar`（564MB，可从 `dist/` 重压）
- 删 `.pi/arc/tmp_*`（11 个一次性调试文件；其中 tmp_repair_q4head.py / tmp_probe_mtp.py 引用的
  5090 model 目录已不存在，本就坏了）
- 保留：`build_86/`、`build_120a/` 构建树，`stats/`（运行时自动生成）

## 7. 移动目录后如何接上（新会话起步清单）
1. 整目录搬走 `ninfer_3060` 即可；**移动后第一次构建前**跑一次 `build_86_run.bat`（或
   `scripts/build_sm86.bat`）重新 configure——`build_86/`、`build_120a/` 的 CMakeCache.txt/build.ninja
   内嵌旧绝对路径，ninja 增量会自动触发 cmake 再生。
2. 先把 §6.2 的未提交改动 commit 掉（Conventional Commit，如 `chore: 目录整理与路径去硬编码`）。
3. 继续 §4 开放事项：最高优先仍是 3060 实卡验证 suffix drafter（强重复负载，查 JSONL
   `suffix_rounds > 0` 且 `suffix_accepted > 0`）。
