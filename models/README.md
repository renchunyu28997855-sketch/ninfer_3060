# 模型工件目录（models/）

本目录用于放置 `.ninfer` 模型制品。**制品本身不入库**（`.gitignore` 忽略 `models/` 下
全部文件，仅提交本 README 与 `.gitkeep`）——9.5 GB 的模型请按下述方式获取或自行重建。

## 目标模型卡：Ternary-Bonsai-2-27B

| 项 | 值 |
|---|---|
| 架构 | Qwen 27B（hidden 5120 / ffn 17408 / 64 层，全注意力层 = 每 4 层第 1 层，其余为 GDN） |
| 词表 / 上下文 | 248,320 / 262,144（推理建议 ≤ 32K，受 12GB 显存约束） |
| 权重格式 | v3 容器；三值化核心 `t2_g128_fp16`（PQ2_0，128 组 34 字节）+ int 侧（q4/q5/q8）+ bf16 hadamard 符号辅助对象 ×3（Σ 28,672 符号） |
| 引擎 | ninfer_3060（sm_86 交付构建 / sm_120a 开发构建均可加载） |

### 本地已有工件

- **`Ternary-Bonsai-2-27B-ninfer-v3-spliced.ninfer`**（若存在）
  - 大小：**9,520,051,456 B**
  - SHA-256：`6d8b62b589cfc57736b515f4dc13e7c5c06c09a726ceee3628ac20a86d8aa883`
  - 来源：官方 v3 制品 + GGUF 逐张量重建 322 个三值化对象（官方件的三值化核心
    数据损坏，直接加载 PPL 恒 ~16.3；spliced 件实测 **PPL 5.628**，quick 集
    261,167 tokens）。详见 `docs/ARTIFACT_NOTES.md` 与 `docs/conversion/`。

### 校验命令（PowerShell / Python）

```bat
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" %~dp0Ternary-Bonsai-2-27B-ninfer-v3-spliced.ninfer
:: 期望 6d8b62b5...（分块读取大文件时同上逻辑即可）
```

### 快速功能验证（~1 分钟，任意 GPU）

```bat
dist\apps\ninfer-perplexity.exe <本目录>\<模型>.ninfer --corpus eval\corpora\perplexity-1m\manifest.json --quick
:: spliced 件参考值：overall ≈ 5.63（英文长文 8.66 / 代码 1.86 / 中文 7.89）
```

## 使用方式（路径可任意，不限本目录）

```bat
:: CLI 对话
dist\apps\ninfer.exe <模型>.ninfer --prompt "你好" --max-context 8192

:: HTTP 服务（OpenAI/Anthropic 兼容 API）
dist\apps\ninfer-serve.exe <模型>.ninfer --port 8080

:: GUI 启动器：把 .ninfer 放本目录，界面自动扫描
start "" "%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe" app\ninfer_launcher.pyw
```

**注意**：exe 旁的 9 个依赖 DLL（ffmpeg ×8 + nvcudart_hybrid64.dll）必须与 exe 同目录，
缺失时进程以 0xC0000061 静默启动失败。`dist/apps/` 已随包配齐（该目录不入库，从
`E:\download\123\ninfer-3060-win\bin\` 拷贝）。

## 没有 spliced 件时如何重建

前置：官方 `Ternary-Bonsai-2-27B-ninfer-v3.ninfer`（9,520,051,456 B）+
GGUF `Ternary-Bonsai-2-27B-PQ2_0.gguf`。

```bat
python tools\splice_ternary.py --base 官方.ninfer --gguf 模型.gguf --out models\Ternary-Bonsai-2-27B-ninfer-v3-spliced.ninfer
```

参数与完整推导见 `tools/splice_ternary.py` 头部注释、`docs/ARTIFACT_NOTES.md`、
`docs/conversion/03-三元模型转-NInfer.md`。

## 显存预算（RTX 3060 12GB）

权重 6.70 GiB + hadamard 符号表 ~0.12 GiB + KV(fp8)/激活余量 → 默认配置总占用
< 10 GiB；剩余可给更长上下文与并发（默认配置 `--max-concurrency 4`）。
