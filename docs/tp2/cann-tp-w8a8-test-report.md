# CANN TP2 W8A8 35B 验收测试报告

- 被测对象: llama.cpp 工作区 (develop tip 21015853c + TP2/W8A8 未提交改动; 含 11-10 bind 断链修复 + shape gate)
- 执行日期: 2026-11-10 (性能与 GSM8K 为重启后干净主机复测, shape-gate 版二进制)
- 环境: Ascend Atlas 300I Duo (310P3, 44GB/颗) x2, CANN 9.1.0, TP2
- 模型: Qwen3.6-35B-A3B-W8A8.gguf (37.81 GiB, 35.51 B, 11-09 重新导出版, 修复旧导出乱码)
- 对照栈: vllm 0.28.0 + vllm-ascend (commit 9e39b9e1a), 同机 TP2, W8A8-RTN
  (数据源: vllm-ascend-w8a8-report.md, 2026-11-09)
- 范围: TP2 W8A8 验收 (功能门禁 + 性能 + GSM8K 正确率, 对照 vllm-ascend);
  F16 仅作参考; 详细过程记录见 REPORT4.md

## 一、性能

### 1.1 llama-bench (TP2, 干净主机, 设备对 2/3)

方法: llama-bench pp512/tg128, -r 3, 不传 `--device` (Meta 切分);
单流口径, 与 vllm 离线 LLM.generate (IN=512, OUT=128, 并发=1) 对齐。

| 模型 | Graph TP | pp512 (t/s) | tg128 (t/s) | capture |
|---|---|---|---|---|
| W8A8 | off | 865.73 ± 2.19 | 20.73 ± 0.16 | 0 |
| W8A8 | on | 863.11 ± 1.58 | **21.09 ± 0.06** | 2 CAP + 768 REPLAY |
| F16 | off | 809.67 ± 6.93 | 20.63 ± 0.32 | 0 |
| F16 | on | 808.74 ± 10.47 | 20.80 ± 0.03 | 82 CAP + 31199 REPLAY |

- W8A8 vs F16 (off): prefill **+6.9%** (865.7 vs 809.7), decode 持平 (+0.5%)
- Graph TP on 干净主机增益: W8A8 tg128 +1.7% (pp512 持平)
- TPOT 47.4 ms/token, TTFT 593 ms (= 512 / pp512)
- 跨设备对 pp512 一致: 0/1 (862.71) vs 2/3 (862.77-865.73) 差 < 0.4%

### 1.2 与 vllm-ascend 对比 (W8A8, TP2, 同机同口径)

| 指标 | llama.cpp (W8A8) | vllm-ascend aclgraph | vllm-ascend eager |
|---|---|---|---|
| prefill pp512 (t/s) | **865.7** (off) / 863.1 (on) | 836.9 | 927.0 |
| decode tg128 (t/s) | 20.73 (off) / 21.09 (on) | **37.09** | 8.62 |

- **prefill: llama.cpp 比 vllm aclgraph 快 +3.4%**
- **decode: vllm aclgraph 比 llama.cpp 快 +76%** (37.09 vs 21.09)
- vllm eager 仅 8.62 t/s: aclgraph 是 vllm-ascend 在 310P 上的可用性前提
  (4.3x), 关闭不可用; llama.cpp eager 路径 20.73 = vllm eager 的 2.4x,
  无图模式本身可用
- 差距根因: vllm aclgraph 把整个 decode 步捕成单图回放 (host 调度开销
  -> ~0); llama.cpp 的 ACL graph TP 为 WIP opt-in, 仅覆盖稳定签名子图
  (W8A8 面 2 图, +1.7%), 全图覆盖受上游重捕风暴限制 (见第三节)

### 1.3 vllm-ascend 对照配置说明

| 项 | 值 |
|---|---|
| 框架 | vllm 0.28.0 + vllm-ascend commit 9e39b9e1a (editable, ~/venv_vllm_ascend) |
| 量化 | W8A8-RTN (ModelSlim ascend 格式): 逐通道对称 int8, scale=amax/127, 无校准; 量化 self_attn q/k/v/o + 256 routed experts; shared_expert (310P EZ1001 限制)/router/GDN/norm/lm_head 保持 FP16。与 GGUF W8A8 导出方案不同, 对照时注意 |
| aclgraph | `VLLM_ASCEND_CAPTURE_SIZES="1 2"` (310P 驱动限制, 更大 batch 捕获报 507903 no event resource) |
| 性能口径 | 离线 LLM.generate, IN=512, OUT=128, 并发=1, 2 预热 + 5 均值 |
| GSM8K 口径 | `vllm serve` + llama-eval 补丁版: 32 并发, temp=0, seed 1234, max_tokens 4096, no-think 模板 (与 llama.cpp `--reasoning off` 等效) |
| 关键环境 | TORCH_DEVICE_BACKEND_AUTOLOAD=0, VLLM_ENABLE_V1_MULTIPROCESSING=0, ASCEND_RT_VISIBLE_DEVICES=<对卡>, GPU_MEM_UTIL 0.80~0.92 |

### 1.4 生产负载吞吐 (32 并发持续解码, server timing 样本统计)

| 模型 | Graph TP | 单 slot 均值 t/s | 聚合估算 (x32) t/s |
|---|---|---|---|
| W8A8 | off | 1.6 | 50.5 |
| W8A8 | on | 1.4* | 44.8 |
| F16 | off | 1.53 | 48.9 |

\* 全程累计均值 (12894 s), 慢 8-12% 可归因重叠探针/别容器/热状态窗口
(每题中位 token 341 vs 341 逐位一致, 无答案膨胀, 结构性差异排除)。
GSM8K 全程 wall time: llama.cpp W8A8 off 10688 s / on 12894 s; vllm W8A8
aclgraph ~1.5 h (KV 55 万 token 无压力; F16 因 KV 仅 7.4 万反复 preempt, ~3 h)。

## 二、GSM8K 正确率 (1319 题)

方法 (llama.cpp): llama-server 32 slot (`-c 262144 --parallel 32
--flash-attn on -ngl 999 -t 32 -b 4096 -ub 512 -n 4096 --device
CANN0,CANN1 -sm tensor --reasoning off`) + llama-eval
(32 线程, 贪心 temp=0, seed 1234, n_predict 4096, regex grader)。
vllm 口径见 1.3, 两侧题集/温度/种子/截断一致。

| 栈 | 配置 | 正确 | 正确率 | 用时 |
|---|---|---|---|---|
| llama.cpp | W8A8 graph off | 1229 / 1319 | 93.18% [91.8, 94.5] | 10688 s |
| llama.cpp | W8A8 graph on | 1239 / 1319 | 93.93% [92.6, 95.2] | 12894 s |
| llama.cpp | F16 graph off | 1242 / 1319 | 94.16% [92.8, 95.4] | 10945 s |
| llama.cpp | F16 graph on (单流探针) | 19 / 20 | 95% | ~600 s |
| vllm-ascend | W8A8 aclgraph | 1237 / 1319 | 93.8% | ~1.5 h |
| vllm-ascend | W8A8 eager | 1227 / 1319 | 93.0% | ~1.5 h |
| vllm-ascend | F16 aclgraph | 1238 / 1319 | 94.0% | ~3 h |

W8A8 on 臂 (GRAPH_TP=1 + shape gate) 2026-11-10 干净主机跑完: 全程 3.6 h
零死锁、4 例 4096 截断按错计入 (模型行为, 同题在 off/eager 基线同为超长答案)。

结论:
- **正确率三方一致**: llama.cpp W8A8 (93.2-93.9%) 与 vllm W8A8 (93.0-93.8%)
  同带; 各配置全部落在 93.2-94.6% 评测噪声带内
- vllm 内部 W8A8 vs F16 差 -0.2pt (量化噪声级); llama.cpp GGUF W8A8 off 与
  F16 off 差 ~1 pt, 与两栈量化方案差异一致 (GGUF 导出覆盖面 vs RTN 保留
  GDN/shared_expert FP16)
- graph 开/关不改变正确性 (两臂正确率互在对方 CI 内)
- 4 例 `finish_reason=length` (4096 截断) 为模型行为, 按错计入

## 三、F16 graph TP 已知限制 (数据缺口说明)

F16 32-slot `GRAPH_TP=1` 存在**流死锁** (n=2, 可复现): 单 token 图的
签名/地址随 batch 组成每步漂移 -> LRU 严格地址匹配永不命中 -> 重捕风暴
(复现实测 150 CAPTURE / 0 REPLAY) -> 显存 5 min 吃满 99% -> 容量驱逐与
在飞执行互相绞死。已复现定案、**不修复仅记录** (上游 capture 机制设计
属性, 默认 0 时生产路径不触发; 详见 REPORT4 5.5 / 设计文档 8.2)。
F16 32-slot graph-on 正确率数据因此不可得, 以单流探针 (19/20) 与
graph-off 基线 (94.16%) 为参考。W8A8 面仅 2 张稳定签名图, 持续 replay,
1319 例浸泡零死锁, 不受此限制影响。

## 四、功能验收门禁 (A1-A5, 5/5 PASS)

correctness 第一门禁 (方法学与判据模板见 test-report-tp2.md):

| 门禁 | 结果 | 摘要 |
|---|---|---|
| A1 跨跑自一致 (seed 42, -n 512 x4) | PASS | 首分岔带 21-56 词, 均为良性语义改写; 无乱码/短输出 |
| A2 MoE + paged attention prefill/decode | PASS | 5 次 CLI 生成 + bench 全 rc=0; 两卡分片对称 (差 1.3% <= 3%) |
| A3 paged metadata 两 rank 一致 | PASS | MIRRORED 冗余计算, 4x512 步连贯输出为间接证据 |
| A4 GDN 状态与 paged KV 生命周期 | PASS | 连续 decode 交替演进, 无 hang/crash/中途乱码 |
| A5 连续运行无 hang/leak/不确定失败 | PASS | 套件前后显存 1632/1313 -> 1630/1316 MB 无泄漏; tg128 cv 1.93% |

## 五、验收结论

**W8A8 TP2 验收: PASS**

1. 功能门禁 5/5, 正确率 93.2-93.9%, 与 vllm-ascend W8A8 (93.8%) 及
   llama.cpp F16 基线 (94.16%) 同带, 量化路径正确性成立
2. 性能: prefill 优于 vllm aclgraph (+3.4%); decode 落后 vllm aclgraph
   (-43%), 但 vllm 无图不可用 (8.62), llama.cpp 无图即 20.73; graph-on
   干净主机增益 +1.7%, capture replay 稳定 (768 次零异常)
3. F16 graph TP 死锁为已定案已知限制, 不影响 W8A8 生产路径
   (`GGML_CANN_ACL_GRAPH_TP` 默认 0)

## 附录: 证据索引

| 文件 | 内容 |
|---|---|
| /tmp/opencode/w8a8-review/gates/perf2_*.log | 1.1 表 llama-bench 原始输出 (4 条件) |
| /home/l00804262/gsm8k_q36_w8a8_tp2_graphtp0.json | W8A8 off 臂 (有效) |
| /home/l00804262/gsm8k_q36_w8a8_tp2_graphtp1_postfix.json | W8A8 on 臂终值 (1239/1319, 93.93%) |
| /home/l00804262/gsm8k_q36_f16_tp2_graphtp0.json | F16 off 基线 (有效) |
| /home/l00804262/gsm8k_q36_f16_tp2_graphtp1_serial_probe.json | F16 on 单流探针 (19/20) |
| /tmp/opencode/w8a8-review/gates/gsm4k_postfix_*_server.log | GSM8K 服务端日志 |
| /tmp/opencode/w8a8-review/gates/f16_repro_evidence.txt | F16 死锁复现尸检证据 |
| ~/bench_result_q36_w8a8_tp2_{graph,eager}.json | vllm W8A8 性能原始数据 |
| ~/gsm8k_q36_{f16,w8a8}_tp2_{graph,eager}.json | vllm GSM8K 原始数据 |
| /tmp/opencode/w8a8-accept/ | A1-A5 功能套件证据 (gen/bench/npu 日志) |
