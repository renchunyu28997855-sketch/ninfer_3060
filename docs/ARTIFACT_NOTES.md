# 工件溯源（Ternary-Bonsai-2-27B）

## 结论先行

官方交付的 `Ternary-Bonsai-2-27B-ninfer-v3.ninfer` 中，**三值化 (t2_g128_fp16)
对象的数据损坏/错位**：按其声明几何解码出的 embedding 与 GGUF 源不符，引擎端
PPL 恒 ~16.3（近随机）。引擎几何与作者打包器规范一致，问题不在引擎。
交付请使用 **spliced** 工件（本机
`E:\download\123\model\Ternary-Bonsai-2-27B-ninfer-v3-spliced.ninfer`，
9,520,051,456 B，未入库——9.5GB）。

## 格式规范（权威来源）

- 打包器：`E:\download\123\model\tools\pack.py`（"M1-D packer"，958 行）。
  `assemble_ternary` (ln≈305)：每行先写 **codes 平面**
  (`base_offset + i*base_row_bytes + sel*32` ← GGUF blk[2:34])，再写
  **scales 平面** (`scale_offset + i*80 + sel*2`，fp16 LE ← blk[0:2])；
  `k_pad = align_up(k,128)`；行映射恒等（dest 行 i ← GGUF 行 i）；
  PQ2_0 块格式 34B `[fp16 scale@0][32B codes@2]`，组大小 128。
- 位序：`code_j = byte[j//4] >> (2*(j%4))`（lsb-first），见 pack.py dq_pq2_0。
- hadamard/prism 符号：GGUF meta `sign_values` int32 ±1，块 1024 归一化
  Sylvester，宽度 [5120,6144,17408] Σ28672；v3 工件以 3 个 bf16 aux 对象
  携带（role `hadamard_signs`），+1 占比 0.4846 与 GGUF 逐值吻合。

## spliced 工件的构造（可再生）

1. 基线：官方文件（其 dir JSON / 资源区 / int 侧保持不变）。
2. `tools/splice_ternary.py`：按 dir JSON 的 parts 绑定把 323 个 t2 物理对象
   映射到 packer 生产者名，用 GGUF 逐张量重建载荷；显式 seek 的原地覆写
   （本机增量文件读不可靠，**必须**显式 seek，且最终断言文件大小不变）。
   跳过 1 个（`proposal/head`，无 GGUF 源，仅 MTP 用）。
3. 校验：embedding 行 tok {0,19,74157} 对 GGUF 真值 **rel_l2 = 0**；
   全文件 diff 仅 322 个 t2 区段 + 3 个 aux 区段不同；tokenizer 等资源区未动。

历史踩坑记录（勿重复）：
- 曾用有状态流式拷贝写入，导致 dst 膨胀到 16.65GB、token 区尾部被截断
  （表现为 `malformed tokenizer.json ... ill-formed UTF-8`）。重写为
  「整文件复制 + 逐对象 seek/write」后解决。
- fp16→fp32 不能简单 `bits<<16`（指数偏置不同）；bf16 才可以。aux 早期
  "看起来是垃圾" 的误判即源于此。

## 验证结果

sm_120a 开发构建 @ RTX 5090，`ninfer-ppl-1m-v1/quick`（261,167 scored
tokens，4 域）：**overall PPL 5.628**（chinese_reference 7.89 /
english_long_form 8.66 / english_reference 7.82 / ninfer_code 1.86），
~1.4k tok/s。损坏工件同条件 16.3–1e7。
