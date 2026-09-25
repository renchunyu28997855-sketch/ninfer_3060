# ninfer_3060

NInfer v3 推理引擎的 **RTX 3060 (GA106 / sm_86, 12GB)** 单卡适配分支。
模型目标：Ternary-Bonsai-2-27B (PQ2_0 三值化核心, Qwen3.5-27B 架构)。

上游基线为 5090/sm_120a 的 v3 引擎树；本分支在其上完成：
sm_86 架构门、GA106 48KB 静态共享内存上限适配、nvfp4/fp8-w4a4 编译隔离与
受支持 stub、三值化权重 (t2_g128_fp16) 注册/几何/内核移植，以及 hadamard
符号辅助对象的装载接线。

## 目录结构

| 路径 | 内容 |
|---|---|
| `src/`, `include/` | 引擎源码 (CUDA/C++) |
| `cmake/` | CMake 模块（依赖解析 / 目标定义，构建必需） |
| `ffmpeg/` | FFmpeg 开发件（headers + import libs，media decode 链接必需；已入库） |
| `apps/` | CLI / serve / perplexity 三个可执行目标 |
| `bench/`, `eval/` | 基准与评测工具 |
| `docs/` | 产品与维护者文档；`docs/ARTIFACT_NOTES.md` 为本分支特有的工件溯源说明；`docs/conversion/` 为模型转换文档集（含自 sister 仓库归档的 4 篇） |
| `models/` | 修改后的 `.ninfer` 制品本地存放目录（制品不入库，见 `models/README.md`） |
| `dist/apps/` | 预编译交付物（3 exe + 9 DLL，**不入库**；从 `E:\download\123\ninfer-3060-win\bin\` 拷入，启动器默认扫描此目录） |
| `scripts/` | Windows 构建驱动（开发机）：`build_sm86.bat`(交付构建) 等 |
| `tools/splice_ternary.py` | 从 GGUF 重建三值化核心的离线脚本（工件再生用） |

预编译产物**不入库**（~700MB exe + 依赖 DLL），本机位置见下文「快速开始」。

## 快速开始（3060 机器）

1. 取预编译包（本仓库外）：`E:\download\123\ninfer-3060-win\bin\`
   - `ninfer.exe` / `ninfer-serve.exe` / `ninfer-perplexity.exe`
   - 随包 ffmpeg DLL ×7 + `nvcudart_hybrid64.dll`（必须与 exe 同目录）
2. 模型工件（9.5GB，不入库）：`E:\download\123\model\Ternary-Bonsai-2-27B-ninfer-v3-spliced.ninfer`
3. 运行：
   ```bat
   ninfer.exe <model.ninfer> --prompt "你好" --max-context 8192
   ninfer-serve.exe <model.ninfer> --port 8080
   ninfer-perplexity.exe <model.ninfer> --corpus <manifest.json> --quick
   ```

显存预算（12GB）：权重 6.70 GiB + hadamard 符号表 ~0.12 GiB + KV(fp8)/激活余量，
默认配置总占用 < 10 GiB，有富余给更长上下文与并发。

## 从源码构建（sm_86 交付构建）

前置：Windows + Visual Studio 2022 Build Tools (cl) + CMake + Ninja + CUDA 13.1。
仓库已自带 `cmake/` 与 `ffmpeg/{include,lib}`（media decode 链接用），无需另找依赖。

```bat
cmake -B build_86 -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build_86 -j 8
```

关键适配（已包含在本树中，勿回退）：
- 根 CMake 架构门 `{120a,86,89}`，非 120a 时定义 `NINFER_SM8X_COMPAT`；
- `src/ops/CMakeLists.txt` 排除 e4m3-mma / nvfp4-w4a4(+tma) / fp8 形状 /
  causal kv-fp8 编译单元，改由 `src/ops/sm8x_compat_stubs.cpp` 提供受支持抛错桩；
- GA106 48KB 静态 smem 上限：q8 KSplit 调度 KWarps 钳到 4、6 处 grouped 内核
  转动态 `extern __shared__`（配合 `cudaFuncSetAttribute`）、swiglu rowsplit BK 64；
- sm_86 下 nvfp4 / fp8-w4a4 路线不可用（本模型不走这些路线，不影响推理）。

## 模型工件与验证

官方 `.ninfer` 的三值化核心经字节级比对确认**数据损坏/错位**（embedding 解码
与 GGUF 源不符，PPL 恒 ~16.3 近随机）。本分支交付的 **spliced 工件**用
`tools/splice_ternary.py` 从 GGUF 逐张量重建 322 个 t2 对象后生成，详见
`docs/ARTIFACT_NOTES.md`。

端到端验证（sm_120a 开发构建，5090 实机）：
`ninfer-ppl-1m-v1/quick` 261,167 tokens → **PPL 5.628**（英文参考 7.82 /
长文 8.66 / 中文 7.89 / 代码 1.86），速率 ~1.4k tok/s。对比损坏工件的
16.3–1e7，数值路径确认正确。最终需在真 3060 上过一遍完整推理。

## GUI 启动器（app/）

`app/ninfer_launcher.pyw`（PyQt6 托盘启动器，移植自上游配套启动器并适配本仓库布局）：
托管 ninfer-serve.exe 启停、实时日志、参数表单（读写 `app/config.json`）、基准/案例测试页。

```bat
:: 依赖 Python 3.12 + PyQt6 + requests（本机已装于 ...
:: C:\Users\sanbanfu\AppData\Local\Programs\Python\Python312）
cd E:\<repo>
start "" "C:\Users\sanbanfu\AppData\Local\Programs\Python\Python312\pythonw.exe" app\ninfer_launcher.pyw
```

界面选引擎 exe（自动扫描 `dist/apps/`、`build*/apps/`、`*/bin/`；exe 旁需同放 DLL）
与模型（把 .ninfer 放到 `models/` 目录或手工填路径），Start 后 API 在 http://127.0.0.1:8080/v1。
运行时统计落 `stats/`（已 gitignore）。

## 已知限制

- 单二进制仅含 sm_86 SASS，不能在 5090/sm_120a 上运行（开发回归用
  `scripts/build_sm120a_dev.bat` 的独立 120a 构建；双架构单二进制不支持）。
- MTP draft (`proposal/head`) 张量保留官方文件原始数据，仅主模型路径经过
  重建验证（spec 解码未启用时不影响）。
- 发行需自行随包提供 ffmpeg DLL 与 CUDA hybrid runtime DLL（缺失时进程以
  0xC0000061 静默启动失败）。
