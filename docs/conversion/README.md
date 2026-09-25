# 模型转换文档（自 sister 仓库 ninfer-ada-ternary 归档）

| 文档 | 内容 | 与 ninfer_3060 的关系 |
|---|---|---|
| [01-4080S-接入-NInfer.md](01-4080S-接入-NInfer.md) | 消费级 Ada 卡 + Windows 上从源码构建/运行引擎 | 背景参考；本仓库的 sm_86 交付构建见根 `README.md`「从源码构建」 |
| [02-微调模型转-NInfer.md](02-微调模型转-NInfer.md) | 普通（微调/safetensors）模型 → `.ninfer`，零引擎改动 | 方法学参考；本仓库不做通用转换器 |
| [03-三元模型转-NInfer.md](03-三元模型转-NInfer.md) | 三元 + Hadamard 旋转基模型的完整移植路径 | **核心参照**：本仓库 t2_g128_fp16 注册/几何/内核/hadamard 符号装载即按其路线实施 |
| [04-工程实录-坑与方法学.md](04-工程实录-坑与方法学.md) | 症状原文 + 判据式排查经验 | 踩坑清单与 `docs/ARTIFACT_NOTES.md` 互补（fp16↔bf16 误读、流式拼接损坏等同类事故均在此有先例） |

本仓库内的对应物：

- `tools/splice_ternary.py` —— 从 GGUF 逐张量重建三值化核心的离线脚本（本文档集所述流程的本仓库实现）；
- `models/` —— 放置修改后的 `.ninfer` 制品目录（**不入库**，gitignore 忽略 `*.ninfer`）。
