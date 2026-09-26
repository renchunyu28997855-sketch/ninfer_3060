# Host KV 工作集（长上下文工作集）实施计划

状态：草案，待实施

适用基线：2026-09-21 当前工作树（main @ `9a4fbc76`）

适用产品：单 GPU、单 resident model、启动时固定 1–8 个 active requests 的 NInfer Engine/serve 路径

本文是一次有终点的实施计划，不是新的长期架构权威。全部工作项完成后，应把稳定不变量合并到
[`paged-kv-cache`](paged-kv-cache.md)、[`resource-scheduling-and-context-cache`](resource-scheduling-and-context-cache.md)
和 [`engine-architecture`](engine-architecture.md)，产品面命令合并到 `cli.md` / `serving.md`，随后删除本文。

## 1. 背景与目标

参考实现是 `F:\develop\bak\kvmem-llama.cpp-master`（llama.cpp fork）：GPU 有界工作集 + CPU pinned
二级缓存，按块换入换出，宣称 32K 活跃窗口近似无损 256K 历史（ΔPPL≈0.02）。**不移植其代码**
（ggml 耦合深），只借鉴算法思路，在 NInfer 现有 paged store + HostKVArena 之上实现。

现状约束：活动请求的全部 committed KV 必须驻留 device。`kv_capacity` 必须 ≥ `max_context`
（`src/models/qwen3_5/program/planning/startup.cpp:763`），因此有效上下文 = min(模型 S, 池容量)，
长上下文 × 高并发直接撞显存墙。Host 侧现有的 HostKVArena 只服务**非活动检查点副本**，救不了
活动序列自身。

本计划引入**工作集模式**：活动会话只把选定子集的 KV 块驻留 device，其余块以字节原样存放在
pinned host 层；每 turn 边界重新选择、换入换出。目标：

1. 单请求长上下文（至 S=256K）在多轮 agent 会话中可运行，device 只占有界工作集。
2. 工作集模式下**注意力语义精确**：对选定块集合做真稀疏注意力，无位置失真（见 §2，这是与
   kvmem 的关键差异）。
3. 默认关闭；关闭时基础路径行为与性能 bit-identical，现有测试套件直接兜底。

完成标准（全部满足才算完成）：

- `--kv-working-set` 开启后，27B NVFP4 artifact 上 256K 历史的多轮会话可运行，PPL/needle 相对
  全历史基线的退化在验收阈值内（§6）。
- 工作集模式每 turn 的传输量、TTFT、TPS 有实测数据（§7）；关闭模式回归零。
- 新增合同（行重映射、host 块元数据、admission 方程）落入 §5 所列文档，本文删除。

## 2. 核心机制：稀疏行 + 预旋转 K（机制与 kernel 改动面）

kvmem 需要 **re-RoPE**（块回迁后原地 delta 旋转、CPU 存 raw-K 镜像防漂移），是因为它的 KV
布局里 key 的逻辑位置由物理 slot 隐式决定，块挪位后位置失效。NInfer 没有这个问题，原因有二：

1. **K 在写入前已旋转到真实位置**。`text_rope(positions, config, qn, kn, s)`
   （`src/models/qwen3_5/execution/attention.h`）在 `ops::kv_cache_append` 之前执行，页里存的
   就是按原始绝对位置旋转后的 K。RoPE 注意力的数学性质：`q(P)·k(j) = f(P-j)` 只依赖**相对**
   位置。只要查询按真实位置 P 旋转、存储 K 按各自真实位置 j 旋转，任意块子集上的点积自动给出
   正确的相对距离——**无需知道块现在住在哪个物理页**。
2. **物理寻址已经间接化**。attention 经 `kv_table_rows`（行号）+ 块表解析物理页
   （`context_query.cuh:292` `policy.page(position, lane)`；prompt kernel 的 `block_table[kb]`
   ，`prompt_bf16.cuh:200`）。换一组驻留块 = 重建该行的页列表。

因此工作集模式下的注意力 = **对声明可见集合 J 的精确稀疏注意力**：被逐出的块对输出零影响
（模型真的看不见它们），驻留块之间的相对几何完全保真。这与 kvmem 的"压缩重排 + delta 旋转"
语义不同——kvmem 会折叠块间空隙造成位置失真，我们不会。**推论：不需要 re-RoPE Op、不需要
raw-K 镜像、host 层就是打包页的纯字节缓存，回迁是 memcpy。**

### 2.1 kernel 空间审计（已完成）：causal_cache 全程在行本地逻辑空间工作

初稿声称"零 kernel 改动"，后续审计先判其被证伪：两条 causal_cache 路线用 positions[] 做
算术推导（prompt 的 `base_pos = positions[0]`、small_t 的 `window = last_pos+1`），若传
**真实位置**则紧凑行下有掩码泄漏与 block_table 越界。更深的复审发现更好的事实：
**positions[] 在这些 kernel 里从头到尾被当作行本地逻辑索引使用**——

- append 写入：`position = positions[0] + token` →
  `paged_kv_physical_page(block_table, position)`（`src/ops/kv_cache/append/kernel.cuh` 全部
  dtype 变体）→ 写入行槽位 `[positions[0], positions[0]+W)`；
- prompt 掩码：query 行 i 见键 `[0, base_pos+i]`（`prompt_bf16.cuh:68-70,112`）；键迭代上界
  `n_block_max` 由 `max_query_abs = base_pos + q0 + tile_rows - 1` 推导（:185-186），块表
  索引 `block_table[kb]`（:200）；
- small_t：`window = positions[tokens-1]+1`（`small_t.cuh:221`），逐 token 因果掩码
  `position = positions[token]`（`small_t_fp8.cuh:192`）。

稠密模式下真实位置 == 行本地索引（行是从 0 开始的稠密前缀），所以现状恒正确。**工作集
模式因此完全不需要改 kernel**：模型层给 op 传行本地 positions `[B..B+W-1]`（B = 驻留历史
行长度），上述算术自动给出正确结果——掩码 `[0, B+i]`、迭代上界 ≤ 行长、append 落行槽
`[B, B+W)`。RoPE 继续用真实位置：`text_rope` 在 op 之前独立执行，调用点已有两个独立位置
张量（`active_rope_positions_` vs `active_cache_positions_`，待 M1.5 核实赋值来源）。

（中途否决的方案：给 envelope 加 `visible_context_tokens` 并改 ~7 个 kernel 文件的锚点——
可行但多余，kernel 本来就活在行本地空间；gap 填充页 / LSE 合并 / per-key 真实位置数组同样
否决，理由见 git 历史版本本文档。）

### 2.2 语义定案：行本地位置 + 精确稀疏注意力

- Op 合同澄清（文字层面，非代码）：`positions[]` 是**行本地逻辑索引**，稠密模式下恰等于
  真实位置；声明可见集 J = 行前缀 `[0, B+i]`、chunk 内保因果；oracle 沿用 "declared
  visible key set J" 定义（`include/ninfer/ops/softmax_attention.h` 既有合同文字微调措辞）。
- 稠密模式行本地 == 真实 → **关闭模式行为 bit-identical**，现有测试套件直接兜底回归。
- conformance 测试增加稀疏行用例（prompt/small_t 两路由），把"行本地 positions 语义"钉死为
  合同的一部分。

## 3. 范围

**v1（本计划）**：

- C=1（单活动请求），turn 边界选择；turn 内不驱逐（行在生成期间稳定）
- 选择器：MeanK（块均值 K 与最近查询的相似度）+ recency + sink（前 N token 常驻）
- MTP/投机路径自动关闭（开启工作集时 CLI 告警）
- host 层复用 HostKVArena + HostKVExtentStore，与检查点副本共池（§4.4）

**v2（不在本计划，仅列接口预留）**：多请求（C>1）跨会话统一驱逐；累计注意力热度打分
（需 attention Op 导出 per-block attention mass）；MRoPE 三轴场景验证；MTP 与工作集一致性；
NVMe 第三层。

**明确不做**：抢占、多 GPU、通用 plugin 化选择策略、向后兼容任何旧 CLI 形态。

## 4. 设计细节

### 4.1 粒度与容量账

- 块 = **128 token = 2 个物理页**（`kPagedKVPageSize = 64`，`src/core/paged_kv_cache.h:20`），
  与 `HostKVExtentStore` 的 extent 语义（≥2 页有序页组）天然对齐。
- 27B 几何（Qwen3.8-27B，来自 artifact config）：24 Q heads / **4 KV heads / head_dim 256**，
  64 层 = **16 full_attention + 48 linear_attention(GDN)**，S=262144。
- 只有 16 层 full-attn 存 KV；GDN 层状态 O(1) 且无位置输入，**完全不受驱逐影响**——混合架构
  使工作集的稀疏化代价天然只有全密的 1/4。
- NVFP4 下每 token KV ≈ 16 层 × 4 heads × 288B ≈ **18KB/token**：

  | 量 | 大小 |
  |---|---|
  | 256K 全历史（单会话） | ≈ 4.7GB（host） |
  | 32K 工作集 | ≈ 590MB（device） |
  | 一块（128 token） | ≈ 2.3MB |
  | C=8 × 256K host 总量 | ≈ 37GB RAM |
  | 对照：BF16 单会话 256K | ≈ 16GB（device 不可行） |

- PCIe Gen5 x16 pinned 实测 ~45–55GB/s：整工作集换血（590MB）≈ 11–13ms，典型 turn 增量
  （几块～几十块）亚毫秒到几毫秒；摊到几百 token 的生成段可忽略。常驻模型设计下推理期间
  PCIe 无其他流量，链路专用于 KV。

### 4.2 工作集生命周期

一切选择/换入换出发生在 **turn 边界**（同一 session key 的新请求 admission 时刻，即
materialization transaction 处，`src/models/qwen3_5/program/transactions/materialization.cpp`），
与现有 admission target search 同点挂钩：

1. **规划**：planner 已知本 turn 形状（新 prompt 长 P_new、输出上限 O_max），
   `B = budget − P_new − O_max − sink` 为旧历史可选块预算。
2. **打分选择**：对旧历史的每个块计算分数（§4.3），取 top-B + sink 块，按真实位置升序排列。
3. **换出（D2H）**：device 驻留但未选中的块经 `DeviceKVPagePool::copy_to_host`
   （`src/core/paged_kv_cache.h:234`，`transfer_stream`）写入 host extent，
   `HostKVExtentStore::publish` 记录 (page, epoch, coverage)。
4. **换入（H2D）**：选中但只在 host 的块经 `copy_from_host` 拷回 device 页，标记
   device_resident。字节相同、host replica 为源；回迁到新页时 content_epoch 按既有
   transfer-destination publish 流程递增（checkpoint restore 已在用同一路径，
   `materialization.cpp:1018`），不发明新校验。
5. **行重建**：会话 KV 行的页列表 = 选中块页（升序）+ 本 turn 新页；`valid_columns` =
   驻留 token 数。op 的 positions 传**行本地** `[B..B+W-1]`（B = 驻留历史行长度；kernel 全程
   行本地空间，§2.1），RoPE 单独传真实位置——两个位置张量在调用栈上本就独立
   （`active_cache_positions_` vs `active_rope_positions_`，text.cpp:875/877）。行内虚拟索引
   仅服务于掩码顺序与页寻址。
6. **turn 内**：新 token 以行本地位置追加在行尾（cache 侧 B+1…），RoPE 用真实位置，行不再
   变动；若 turn 实际生成超出 admission 投影（超长输出），v1 直接拒绝超预算 turn（admission
   已有拒收概念），不做 turn 内驱逐。

不变量：**历史从未超过可用 budget 的会话行为与今天完全一致**（全驻留、无 host 往返）；
工作集逻辑只在历史 > budget 时激活。

### 4.3 打分

- **MeanK**：每块维护均值 K 向量（F32 `[Hkv, head_dim]` = `[4, 256]`，host 元数据表）。
  块首次换出时对其 device 页做一次归约落表；常驻未换出块在打分时按需现算（块在 device，便宜）。
  分数 = `mean_K_block · q_last`（双方均为已旋转值，点积即相对位置编码，与 MeanK 文献一致）。
- **recency**：越靠近 frontier 的块加分（指数衰减，衰减常数为调优项）。
- **sink**：前 `--kv-sink` token（默认 2048）无条件常驻，不参与竞争。
- 最终按 `score = meank_sim + λ·recency` 排序取 top-B；λ 为调优项，初值 1.0，验收阶段扫描。
- 元数据表随 extent 迁移：换出时落表，回迁不重算。表容量 = 会话块数（256K/128 = 2048 块 ×
  4KB = 8MB/会话，可忽略）。

### 4.4 Host 层

- 复用 `HostKVArena`（`src/core/host_kv_arena.h`）pinned 分配与 `HostKVExtentStore`
  （`src/models/qwen3_5/program/storage/host_kv_store.h`）extent 生命周期
  （prepare → writable_view → publish → abort/release）。
- **统一池**：工作集溢出块与检查点副本页共用同一 arena，共享一套 LRU/驱逐（现有
  pressure victim 机制可扩展覆盖工作集块）。避免两个半空池子。
- sizing：启动参数 `--kv-host-capacity <bytes>`（默认按 C×S 上界自动算，机器 RAM 不足时报错
  而非静默降级）。

### 4.5 Planner / admission 挂钩

- 方程变化：`kv_capacity` 从 "≥ max_context" 变为 "≥ working_set_budget"；
  `max_context`（= 模型 S）保持不变，是**逻辑**上限；host 容量约束 `C×S`。
- admission 判据：会话总历史 H ≤ S 且 `H ≤ budget + host_capacity_share` 即可接纳——
  今天 H > device 容量即拒收的路径变成可接纳。
- 新文件：`src/models/qwen3_5/program/planning/working_set.{h,cpp}` 承载选择策略与块元数据，
  被 materialization transaction 调用；不进入普通 decode 热路径（turn 内零开销）。

### 4.6 检查点 / StateImage 交互

- GDN 状态（StateImage：linear conv + recurrent + continuation hidden，
  `src/models/qwen3_5/state/state_image.h`）每层 O(1)、体积小，**始终全量物化**，不受工作集
  影响——它本身就是线性注意力的完整历史压缩。
- continuation 合同（frontier 处完整 StateImage + typed KV 覆盖）扩展：typed KV 覆盖允许部分
  块位于 host replica（覆盖校验按 epoch/coverage 对 host 块同样生效，机制已存在）。
- 会话结束/turn 收尾的 checkpoint 保留策略不变；工作集块只是多了 host 副本位置。

### 4.7 MTP 隔离

MTP 有自己的 KV store（`mtp_kv_.layer_view(0)`，`src/models/qwen3_5/execution/text.cpp:491`），
draft/verify 路径与主池的一致性证明是 v2 工作。v1：`--kv-working-set` 与 MTP 互斥，
同时指定时 CLI 报错（不是静默关闭投机，避免用户误以为还在加速）。

### 4.8 CLI 面

```
--kv-working-set <tokens>   工作集预算（0=关闭，默认 0；须 ≥ --kv-sink）
--kv-sink <tokens>          常驻 sink token 数（默认 2048）
--kv-host-capacity <bytes>  host KV 池容量（默认自动）
```

文档：`cli.md` / `serving.md` 增加"长上下文模式"小节，写明适用负载、质量边界（§6 的评测
结论）、与 MTP 互斥、host RAM 需求。

## 5. 合同与所有权边界

| 域 | 变化 |
|---|---|
| Core（物理原语） | `DeviceKVPagePool` 增按 extent 的批量 copy 便捷接口（若现有粒度不够）；无新原语 |
| Artifact | 无变化（KV 是运行时产物，不进 artifact 格式） |
| Ops | **无数学新 Op、无接口改动**（§2.1：kernel 全程行本地空间）。`causal_softmax_attention`
  合同文字澄清：positions[] 为行本地逻辑索引（稠密模式 == 真实位置）；声明可见集 J = 行前缀
  `[0, B+i]`、chunk 内保因果；conformance 测试增加稀疏行用例（prompt/small_t 两路由），
  oracle 沿用 "declared visible key set J" 定义 |
| Model（qwen3_5） | 新增 `planning/working_set.{h,cpp}`（选择策略、块元数据）；
  materialization transaction 增工作集分支；`RopeConfig`/执行路径零改动 |
| Runtime | admission 方程与拒收理由扩展；`RuntimeStats` 增工作集计数器
  （换入/换出字节、每 turn 传输时间、选择耗时、命中分布）进 `/stats` |
| Serve | CLI 选项透传；文档更新 |

不引入：通用模型图、字符串驱动执行、隐藏 device 分配、假想目标的 placeholder。

## 6. 数值与验证计划

工作集模式**不产生新的数学 Op**，数值风险集中在"稀疏行 == 精确稀疏注意力"这一系统级命题：

1. **Op conformance 扩展**：`causal_softmax_attention` 在稀疏行（乱序物理页、非前缀可见集）
   下对独立 FP64 oracle（declared set J 上的 naive 计算）过既有数值判据。这是 M1 的硬门槛。
2. **端到端质量**：真实 27B NVFP4 artifact，固定语料多轮 agent 会话 + needle（256K 深度
   检索探针），工作集 32K/64K/128K 各跑一遍，对照全历史基线；报告 ΔPPL 与 needle 命中率。
   验收阈值在 M4 前根据基线噪声确定并写入报告（目标：32K 档 ΔPPL < 0.05 且 needle 不退化）。
3. **合同测试**：行重映射正确性、epoch/coverage 在 host 往返后不变、驱逐引用计数
   （无悬挂/双释放）、transfer stream 排序、统一池 LRU 与检查点副本共存。
4. **基础路径回归**：`--kv-working-set 0` 时现有测试套件全绿 + 关键 benchmark 前后对比
   （合入门槛）。

## 7. 性能验收

- 指标：每 turn 换入/换出字节分布、选择耗时、TTFT、TPS（32K/64K/128K/256K × 关闭/开启）。
- 证据：nsys 抓 turn 边界（确认 H2D 与 prefill 重叠、无 per-token 传输）；
  `profiles/nsys/` 留档。
- 盈亏分界：找到"开启比关闭慢"的上下文长度拐点，写入文档作为使用指引
  （预期：短上下文必亏、超长上下文因 KV 读取带宽下降可能反超）。

## 8. 里程碑

| ID | 内容 | 完成标准 | 状态 |
|---|---|---|---|
| M1 | 稀疏行 conformance 测试（prompt/small_t 两路由，钉住行本地 positions 语义；§2.1 确认零 kernel 改动）+ 行重映射脚手架（确定性 recency+sink 选择，C=1，模型层传行本地 positions） | 两路由稀疏行 oracle 测试过；关闭模式回归零 | ✅ 完成（ctest 9/9 定向绿） |
| M2 | host 层接入：换出/换入事务、块元数据表、统一池 sizing | 256K 会话在 32K 工作集下可连续多 turn 运行（功能正确，质量未验收） | ✅ 完成（swap 回环测试绿；真跑见 M2-G） |
| M3 | MeanK 打分 + CLI 面 + 文档 + /stats 计数器 | 端到端跑通真实 artifact；CLI 文档同步 | ✅ 完成（CLI 面/stats/文档同步；MeanK 打分已于 2026-09-22 移除，见 §15） |
| M4 | 数值验收（§6.2）+ 性能验收（§7）+ 合同文档合并、删本文 | 阈值达成或明确失败原因；文档合并完成 | ✅ 功能完成（多会话+DFlash 共存，构建 369/369、四回归套件绿）；数值/性能验收与文档合并待 GPU 空闲后收尾（§14） |

M1 失败（kernel 有不可绕过的稠密假设）时立即上报，走 §2 备选方案 (a)/(b) 重新评估，
不带病推进 M2。

## 9. 风险与开放问题

1. ~~decode 路由的稠密假设~~（§2 风险点）——**已解决且比预期好**：审计（§2.1）证明 kernel
   本就工作在行本地逻辑空间，零 kernel 改动、零 Op 接口改动；残余风险移到模型层（稀疏行
   构建、行增长机制、行本地 positions 张量），由 M1.5 脚手架 + conformance 测试覆盖。
2. **质量边界依赖负载**：近无损结论只在多轮 agent 类负载上有证据；长文档一次性问答、
   远距离检索可能真掉点。产品定位为"长上下文模式"，文档明示适用场景。
3. **MeanK 与真实注意力热度的差距**：MeanK 是查询-键代理，极端负载下可能选错块；
   v2 的累计热度（attention mass 导出）是对冲，v1 接受该风险并在验收中暴露。
4. **host RAM**：C=8×256K ≈ 37GB pinned；机器 RAM 不足时 `--kv-host-capacity` 自动下调
   并发上限并告警（不静默）。
5. **sink 大小与 agent 系统提示词长度**：默认 2048 可能不够（工具定义长的会话），
   验收阶段用真实负载校准默认值。

## 10. 参考

- kvmem 参考实现：`F:\develop\bak\kvmem-llama.cpp-master`（`kvmem/include/kvmem/*`、
  `src/adapter/llama-memory-kvmem*`、`llama-kvmem-stagein.cu`）——只借鉴块选择与分层思想，
  re-RoPE/raw-K 机制**不采用**（§2 论证了不需要）。
- 现有合同：`paged-kv-cache.md`（typed pools / 页 / 副本 / 消费者视图）、
  `resource-scheduling-and-context-cache.md`（admission / retention / materialization）、
  `qwen3_5-model.md`（混合架构与状态语义）、`op-development.md`（oracle 合同）。

## 11. M2 工作分解（host 层接入）

M2 目标：256K 会话 × 32K device 工作集，连续多 turn 功能正确。全部生命周期挂在既有
materialization transaction 上（admission 同点），turn 内不驱逐。

### 基础设施现状（已核实）

- `LogicalKVPageStore::Page`（kv_store.h:780）：device_replica / host_replica / content_epoch /
  committed_columns / references / active_references / source_pins / destination_pinned。
- 换出原语链：`HostKVExtentStore::prepare(pages, membership)`（pin source、arena 分配、逐页记
  Membership{epoch,coverage}）→ `DeviceKVPagePool::copy_to_host` → `publish(reservation)`
  （attach host replica + unpin）→ `release_active_reference` + `release_reference(writer)`
  （refs→0 时 device lease 自动 reset；descriptor 因 host_replica 存活）。
- 换入原语链：`materialize_transfer_destination(reservation, coverage)`（新逻辑页 + lease +
  epoch++ + destination_pinned）→ `copy_from_host`（materialization.cpp:1018 同款 subview 批拷贝）
  → `publish_transfer_destination(handle, writer=true)` → `retain_active_reference` +
  `commit_coverage`；旧 host replica 经 `release_page_replicas` 释放 arena。
- 事务框架：MaterializationTransaction（program_impl.h:736+）已有 text_restores/destinations、
  transfer timer、publish/abort 路径——M2 的换血走同一框架。
- trigger 点：逻辑上是 admission 同点的 turn 边界；物理钩子为 `bind_sequence_kv`
  （storage/context.cpp，prefill staging 唯一咽喉，4 个调用点都过它）——swap 原语要求行处于
  active 态，bind 激活之后立即触发，与后续 ensure/prefill 同 stream 自动保序。
- 行发布：`KVExecutionTablePool::publish(row, logical_begin, indices, stream)`（行重发行）；
  `compact_working_set`（M1 脚手架，kv_store.h:1504）只能选当前 membership 子集，不能加入新页，
  已被 `swap_remap_working_set` 取代。

### 分块实现

- **A. 块元数据表**（✅ 完成）：`WorkingSetSession.pages` = 全历史页表（per-page 条目
  `WorkingSetPage{true_block, page}`，覆盖 [0,H) 全部块、true 位置序）；被降级块保留条目供再提升。
- **B. 行重映射原语**（✅ 完成）：`KVAddressSpaceStore::swap_remap_working_set(handle, new_row,
  resident_tokens, HostKVExtentStore&, stream)`——任意合法逻辑页列表重建行（双向：换出 D2H+
  host publish / 换入 materialize+H2D+publish）；coverage 守卫（逐槽 committed_columns ≥
  needed=min(64, resident−i·64)）；bad_alloc 传播；committed_frontier 重定基为 resident_tokens。
- **C. 换出**（✅ 完成）：leaving 页批量进一个 extent（prepare_active_owner → copy_to_host →
  publish）；release 失败 terminate；checkpoint/引用校验。
- **D. 换入**（✅ 完成）：host-only 入选页 → materialize_transfer_destination + H2D +
  publish_transfer_destination(writer) + retain active ref；reclaimed 尾部 release_page_replicas；
  内容回环测试绿。
- **E. trigger**（✅ 完成）：钩子 = `bind_sequence_kv` 激活末尾 + prefill 分块循环内
  `commit_sequence_kv` 之后 + 普通 decode 每轮 `commit_sequence_kv` 之后（maybe_apply_working_set；
  prefill 逐块流式换出，decode 窗口每跨一个 128-token 块滑一次——行常驻恒 ≤ budget+128+尾页，
  被 F 的 entitlement cap（budget+prefill_chunk）覆盖；短序列 bit-identical）。仅当
  `working_set_policy_` 已设且 text_kv_valid>0。语义：H≤budget → identity 短路（不建 session，
  坐标恒等）；否则 build_table（全历史表扩展）→ 选择 → 幂等检查（session.frontier==H 或
  行==new_row → 跳过 republish）→ swap_remap_working_set → **从重发行后的行回刷驻留条目句柄**
  （提升页落在新逻辑页上，表不得引用被取代句柄）→ session={B,H,table}。使能 =
  Program::set_working_set_policy（默认 off；校验块对齐 + speculative 互斥 + host extents）；
  budget/sink 默认 32K/2K 常量在 working_set.h，CLI 接法在 M3。
- **F. 统一池 + admission**（✅ 完成）：池无需新建——工作集溢出与检查点副本已共用同一
  HostKVArena + HostKVExtentStore；arena 容量仍由 engine options `host_kv_capacity_bytes`
  驱动（自动 sizing + --kv-host-capacity CLI 在 M3），不足时 swap 原语干净 bad_alloc 拒收 turn。
  admission 放宽 = entitlement cap：`plan_request` 在策略激活时把 `text_kv_page_entitlement`
  从全上下文改为 `pft(min(reserved, budget + prefill_chunk))`（peak 行 = 驻留窗口 + 两次
  trigger 间最多追加一个 prefill chunk；短上下文透传不变，main_kv_pages admission 随之看到
  真实峰值，零调度改动）。纯函数 seam `working_set_device_window(policy, reserved,
  headroom)`（working_set.h）+ 单测 test_working_set_device_window。speculative backend 的
  backend_kv_page_entitlement 不 cap（与 policy 互斥）。
- **G. 测试**（✅ 完成，真跑待 GPU）：每块先红后绿——块表迁移单测、remap 引用计数、换出/换入
  内容回环（packed 字节精确比较）均已覆盖（test_context_store.cpp + test_runtime_mechanisms.cpp）；
  E 的胶水组合测试 = test_kv_working_set_trigger（两 turn 窗口滑动 + 加宽预算提升 + 幂等 +
  build_table 错误路径）；多 turn 长会话功能测试 = tests/models/qwen3_5/test_engine_working_set_real.cpp
  （NINFER_TEST_ARTIFACT 未设 → RC 77 skip；场景 NINFER_WS_REAL_SCENARIO=all|working-set|control）：
  max_context=8192 + kv_capacity explicit 2048 + budget=1024/sink=128 + prefill_chunk=256 +
  host arena 256MB；①短请求走 identity 路径；②~2.5K-token prompt（>kv_capacity）完成 =
  entitlement cap + prefill 流式换出 + decode 窗口滑动端到端证明；③同 session 续接 turn
  （前缀复用 + 表扩展/提升）；④control：同请求关 WS → 永不准入 → pending deadline 触发
  RequestError(QueueTimeout)。关闭模式 bit-identical 回归 = 既有套件全绿。真实 artifact 运行需
  GPU 空闲（用户 serve 占 ~30GB 时不可并发加载 27B）。

顺序：A → B → C+D（事务扩展）→ E → F → G 交织。

## 12. M3 工作分解（产品面 + 打分选择器）

M3 目标：把工作集做成产品能力（CLI/serve/stats 面），并把选择从纯 recency 升级为
mean-K 相似度打分（plan §4.3）。全部完成，关闭模式 bit-identical 不变。

- **A. CLI/serve 选项面**（✅ 完成）：`--kv-working-set N`（token 预算，128 对齐，0=关）、
  `--kv-sink N`（默认 2048）、CLI `--kv-host-capacity BYTES`（显式 host 池；0=auto）。校验：块对齐、
  sink≤budget、与 --spec 互斥、--kv-host-capacity 须配 --kv-working-set、
  --kv-capacity ≥ --kv-working-set + --prefill-chunk。serve 复用既有 --host-kv-mib 作 host 池
  （不加新旗标）。tests/test_cli_options.cpp + tests/test_serve_options.cpp 覆盖接受/拒绝例。
- **B. policy 走 plan + auto sizing**（✅ 完成）：WorkingSetPolicy 经 SequencePlanningInputs/
  SequencePlanImpl 进入 ProgramImpl（startup 校验）；capacity==0 时 ctor 自动 sizing
  （working_set_auto_host_capacity + 物理内存预检，>3/4 RAM 报错）；旧 setter 链
  （set_working_set_policy）全删。
- **C. /stats 计数器**（✅ 完成）：WorkingSetStats（selections/swaps/demoted+promoted pages/
  D2H+H2D bytes+seconds/selection seconds）→ RuntimeStats → /stats JSON `working_set.*`
  + request_log 单调增量。
- **D. Mean-K 打分选择器**（✅ 完成；含一次线上修复）：capture = RoPE 后每个 full-attn 层把 tapped token 的
  旋转 query 列（全部 q heads）异步拷入 pinned host 槽（WorkingSetQueryTap；text.cpp hook，
  decode/prefill/graph 三路径绑定，仅块边界步绑定）；harvest = 逐层组归约 bf16→f32 得
  last-query 签名 [kv_heads*head_dim]；块签名 = 候选块 K 平面 128-token 均值（选择时按需抽取：
  双 host 直读 / 双 device 一次性 D2H 进 scratch arena / 混合驻留拒绝）；
  score = mean_h((1+cos_h)/2) + λ·exp(−age/64)（λ=1，τ=64 块），top-k ∪ sink 升序；几何门控 =
  plain BF16 KV 存储（量化存储自动回退 recency，位相同）；任何坏输入（span 尺寸/头数/整除性）
  回退 M1 确定性 recency 路径。单测 test_working_set_scoring_selection（相似旧块击败新块、
  零签名→recency、malformed→回退、identity 忽略 scoring）。
  - **修复（用户实测发现，2026-09-22）**：prefill chunk 可被 split_frontier（thinking 模板
    rewrite frontier / capture frontier）截断到短于名义长度（如 512 边界只算 505 列），
    原 hook 对越界 tap index 抛 logic_error → worker fail_all_locked → 引擎整体 failed
    （后续请求全 503）。改为越界即跳过采样：被截断后的下一个 chunk 恰好落在块边界补采，
    最坏情况该轮选择回退 recency（stamp 条件本就要求 processed==nominal，语义一致）。
- **E. 文档同步**（✅ 完成）：docs/cli.md（选项表 + 语义段）、docs/serving.md（选项表 +
  /stats working_set 字段 + 适用场景段）、README（long-context 小节）。

未完成/待办：真实 artifact 端到端跑（test_engine_working_set_real，等 GPU 空闲）与性能验收
（§7）——需用户 serve 不占 GPU 的窗口。

## 13. M4 工作分解（多会话并发 + DFlash 投机共存）

M4 目标：解除两个人为互斥——(a) WS 不再限定单会话（C=1 guard 拆除，支持
`--max-concurrency` 1..8）；(b) WS 与 `--spec dflash`/`--spec dflash2` 共存（仅 MTP 仍互斥，
其 verify 路径未接线）。设计原则：draft 池独立于主 KV store，WS 驱逐不触碰 draft 数据；
tap 泛化为逐行采样；target 验证基址与 draft 执行位置分离。

- **A. tap 泛化**（✅ 完成）：`WorkingSetQueryTap.pinned` 布局 `[rows][layers][slice]`
  （rows=kMaximumConcurrency=8 槽，cudaMallocHost ~1MiB@8×16×256×2B）；setter 改
  `set_working_set_query_tap(const WorkingSetQueryTap*, std::span<const int32_t> row_columns)`；
  attn_mix hook 逐行 r：`row_columns[r]<0||>=width` 跳过（宽容，不抛），dst=
  `pinned+((r*layers+fidx)*slice)`，src=`qn.data+(r*width+index)*slice`。每行至多采 1 列：
  ordinary col=0（A=1）；dflash col=rem−1，rem=(128−F%128)%128∈[1,W]（本轮跨越的块边界列）；
  prefill col=remaining−1（chunk 尾恰落 128 边界时）。非边界步 tap=nullptr → 走 CUDA Graph
  （图捕获恒不绑 tap，烧录安全）。
- **B. 多会话 ordinary**（✅ 完成）：删 C=1 guard（"working-set mode is single-session"）→
  per-row `kv_row_coordinate` 翻译、envelope `{min+1,max+1}`、graph profile 按 max frontier；
  graph key 用真值 maximum_frontier（bucket 覆盖保证）；边界交叉步 eager 执行（防 tap 被烧录）。
  post-loop stamp 仅对 `ws_row_columns[row]>=0` 的行写入 `working_set_query_position/slot`
  （非边界行保留旧 position → harvest freshness 检查失败 → 回退 recency，避免读陈旧 pinned）。
- **C. DFlash 接线**（✅ 完成）：`DFlashDecodeIngress += verify_base_positions[8]`（row-local
  target 验证基址，整结构 H2D 自动带上）；`speculative_prepare_verify_inputs` 改用它；
  `execution_frontiers` 保持真值（draft 池 prepare_ragged_prefix / propose / sliding-window
  attention 都在 draft 空间）；`TargetVerifyFrameView.frontiers`（真值）仍喂 accept ops（RNG
  domain）；DFlashBatchContext += tap+row_columns（body 内绑定，graph 下从 context 取）；
  commit 在 resolve_pending_raw：`text_kv_valid%128==0` 时 stamp (row, text_kv_valid−1) +
  maybe_apply。
- **D. 互斥放宽五处**（✅ 完成）：CLI/serve/startup/model_instance/engine validate_options
  全部改为 ws+Mtp 拒绝、ws+DFlash/DFlash2 放行（错误串统一 "...mutually exclusive with MTP
  speculative decoding (--spec mtp); dflash/dflash2 are supported"）；apply_sequence_working_set
  删 speculative logic_error 守卫。tests/test_cli_options.cpp + test_serve_options.cpp 加
  ws+dflash2 正例（--draft-tokens 15 --kv-capacity 33792 → DFlash2）。
- **E. 文档/启动器**（✅ 完成）：docs/cli.md + docs/serving.md 互斥措辞更新；app_ws 启动器
  守卫修复（原条件误拦 dflash/dflash2，改为仅拦 mtp，i18n 文案同步）；新增“上下文容量速览”
  面板（总历史=并发×max-context、驻留窗口估算、⚠ 警告行）+ 六个上下文参数 tooltip；
  SERVE_EXE 切到 ninfer-serve-20260922m4.exe。

正确性要点：① 双坐标系不变（模型层恒真实空间，store 行本地，桥=SequenceState::working_set）；
② WS 驱逐只动主 KV store 物理页，draft CyclicKVCache/full 池完全独立；③ 每行至多 1 列采样 +
harvest freshness（query_position==history−1）保证签名新鲜度；④ 任何几何/span 非法输入回退
确定性 recency（位相同）。

验证状态：构建 369/369 RC=0；四回归套件绿（cli_options/serve_options/context_store/
runtime_mechanisms）。真模型 e2e（engine_working_set_real + ws+dflash2 组合）待 GPU 空闲。

## 14. 里程碑记录（2026-09-25，M1–M4 交付完成）

**交付状态：功能可用（用户已用 app_ws 启动器 + m4 二进制真实 serve 运行），仍有已知问题。**

证据：
- 构建 369/369 targets RC=0；四回归套件绿（ninfer_cli_options_test / ninfer_serve_options_test /
  ninfer_qwen3_5_context_store_test / ninfer_qwen3_5_runtime_mechanisms_test）。
- 真实 serve 启动成功：weights 18.0GiB、host KV pinned 18.7GiB、CUDA graphs 1.7s、ready 8.7s；
  `capacity | KV 254,016 tokens, k8v4, auto | pages 3,969/12,190`；用户实测输入缓存命中正常。

已知问题与限制：
1. **k8v4 下 mean-K 相似度打分关闭**（几何门控 BF16-only，设计如此）：选择退化为 sink+recency；
   要启用需 artifact 转 BF16 注意力 KV。
2. **kv_sink 全局静态**（一刀切）：pi 会话 system 前缀 ~20K 而 kv_sink 默认小 → 长会话超窗后
   系统提示词可能整体滑出注意力。后续方案 B（admission 时把匹配到的共享前缀长度作为
   per-session sink 自动探测）已设计未实施。
3. **真模型 e2e 未跑**（todo #13 M2-G 真跑 / #25 M4-E 数值+性能验收）：需 GPU 空闲窗口 +
   NINFER_TEST_ARTIFACT=out/qwen3_6_27b.ninfer。
4. **三个预存 ctest 失败**（stash 基线确认与 WS 无关，勿修）：ninfer_resource_manager_test
   （"FAIL candidate-stratified reuse closure: one-step eviction outranked the multi-step
   preserving reuse closure"）、ninfer_gdn_replay_records_test（0xC0000374）、
   ninfer_linear_nvfp4_a16_test。
5. **长上下文 decode 速度**：KV 带宽 bound，ms/step 随驻留窗口线性（k8v4 每 token≈25.3KiB，
   5090 ~1.8TB/s 下界：30K→~0.4ms、130K→~1.8ms、200K→~2.8ms）；是否超线性（kernel 水分）
   待受控测量（扫窗口 30/130/200K + nsys）。

## 15. 里程碑记录（2026-09-22，移除 mean-K 打分）

用户要求去掉 score 部分功能（速度归因未找到具体原因，先移除待测）。选择器回到
M1 确定性 recency+sink 策略，WS 其余能力（换出/换入、行本地双坐标系、多会话+DFlash
共存、CLI/serve 面）不变。

移除范围（12 文件）：
- `working_set.{h,cpp}`：`WorkingSetScoring`、`select_working_set_blocks` 三参重载与
  sim/recency 混合排序（`kWorkingSetRecencyWeight/HorizonBlocks` 常量）。
- `text.{h,cpp}`：`WorkingSetQueryTap` 结构与 `attn_mix` 内的 per-layer query 采样 hook。
- `context.h`：三个 Context 的 tap/row_columns 字段。
- `decode.cpp`：边界步 eager 逻辑（`working_set_step`→`use_graph=false`）、per-row
  tap 列计算（ordinary col0 / dflash rem−1）、post-loop query 位置 stamp；图路径恢复
  无条件使用 CUDA Graph。
- `prefill.cpp`：chunk 边界 tap 绑定与 query 位置 stamp。
- `program_impl.{h,cpp}`：`SequenceState` 五个 scorer 字段、pinned 分配与 BF16 几何门控、
  `harvest_working_set_query`/`ensure_block_means`。
- `storage/context.cpp`：apply 内 scoring 构造、`harvest`/`ensure_block_means` 定义、
  bf16→f32 helper；`kv_store.h`：四个 scorer-only 页访问包装。
- `draft.cpp`/`graphs.cpp`：tap 绑定与 aggregate 尾字段；`test_runtime_mechanisms.cpp`：
  `test_working_set_scoring_selection` 整函数。

行为变化：块边界步不再走 eager（无 tap memcpy/同步开销）；k8v4 与 BF16 几何下选择
行为一致（纯 recency+sink）。保留项：`swap_remap_working_set`、`row_local` 双坐标系、
`maybe_apply_working_set` 触发点、`verify_base_positions`（target 验证 row-local 基址）、
CLI/serve/launcher 面。§4.3 设计保留作历史记录，代码不再实现。

验证：构建 RC=0；四回归套件绿；新二进制 ninfer-serve-ws-20260922.exe（WS 分支标签命名）交付，
app_ws 启动器默认切换。引擎命名约定：主分支=ninfer-serve-main-<date>.exe，
WS 分支=ninfer-serve-ws-<date>.exe。真 e2e（#13/#25）仍待 GPU 空闲。

剩余工作：#13/#25（GPU 空闲后）；长窗口 decode 受控测量；方案 B 自动 sink（可选）；
若用户测试后确认 score 无关性能问题，本移除即终态（§13-E launcher 守卫无需再改）。

环境与构建铁律（本机必用）：vcvars64.bat broken（VSCMD_* 标记使 nvcc 8.3 名 ccbin 校验必败）→
hand-rolled env bat（scratch/build_m4a.bat 配方：绝对长路径 VC/KITS INCLUDE/LIB/PATH，
CC=CXX=cl.exe，不调 vcvars；bat 必须 CRLF；cmd //c 绝对路径调用）；构建目录 build-v3；
测试 exe 在 build-v3/tests/*.exe，DLL 部署=复制 third_party\vcpkg\bin+third_party\ffmpeg\bin
的 ~10 个 dll 到 apps/tests 旁；MSVC 日志 GBK→python decode('gbk',errors='replace')。

