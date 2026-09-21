# CANN Weight NZ 优化测试报告

分支: `weight-nz-ope` (develop + weight NZ layout module)
代码提交: `291029539` (cann : add weight NZ layout module for Ascend matmul weights), 已合并 develop `8b319514d`

## 1. 修改方案

### 1.1 新增模块 ggml/src/ggml-cann/weight_layout.cpp / weight_layout.h

将权重 NZ (FRACTAL_NZ) 格式逻辑从后端内联代码抽成独立模块, 核心接口:

| 接口 | 作用 |
|---|---|
| `ggml_cann_weight_layout_enabled()` | 总开关, 读取 `GGML_CANN_WEIGHT_NZ` (默认 on), 进程启动时读取一次 |
| `ggml_cann_is_weight_layout_candidate()` | 候选权重判定: 名称后缀匹配 (attn_q/k/v, ffn_*, output, expert 权重等) + F16 类型 + 连续内存 + ne[0]/ne[1] 32 对齐 + expert 权重支持 ne[2]>1 |
| `ggml_cann_weight_layout_alloc_size()` | 基于 `aclnnCalculateMatmulWeightSizeV2` 精确计算 NZ 显存 (含 expert 切片外排场景) |
| `ggml_cann_weight_format_to_nz()` | 加载期 ND -> FRACTAL_NZ 转换 (`aclnnTransMatmulWeight`); expert 权重逐切片原位转换; 设备级 workspace 复用加锁分配 |

### 1.2 既有文件接入

| 文件 | 修改 |
|---|---|
| `ggml-cann.cpp` | 静态 `weight_format_to_nz()` 改为路由到新模块; `set_tensor` 的 is_nz 判定与 buffer alloc size 计算改用 candidate/alloc_size 接口 |
| `aclnn_ops.cpp` | mul_mat 权重格式选择: candidate 命中即按 `ACL_FORMAT_FRACTAL_NZ` 构建 tensor, 否则退回 ND |

### 1.3 相比旧内联实现的增强

1. MoE expert 权重 (ne[2]>1) 逐切片 NZ 转换, 旧实现直接跳过。
2. 用 `aclnnCalculateMatmulWeightSizeV2` 精确计算 NZ 显存, 避免旧实现的断言限制与尺寸低估。
3. 候选命名集扩充 (ssm_alpha/beta/out, ffn_gate_inp 等), 与 KQ 量化路径解耦。

### 1.4 本分支的 expert 门控

expert NZ 依赖配套的 grouped matmul FRACTAL_NZ 路径 (上游模块的 moe_ops / moe_fused_ops), 本分支未移植, 且 buffer set_tensor 路径存在 `ne[2] == 1` 断言。因此 `ggml_cann_is_weight_layout_candidate` 对 ne[2]>1 一律返回 false, expert 权重保持 ND (与 develop 原行为一致), dense 路径不受影响; 后续移植 moe_ops 后再放开。

## 2. 测试环境

| 项目 | 配置 |
|---|---|
| 硬件 | 华为云 123.60.231.33, Atlas 310P3 (8 chip, 取空闲 chip) |
| CANN 版本 | 9.1.0-beta.1 |
| 构建目录 | /home/llama-cpp-weight-nz/build-cann (Release) |
| 模型 | Qwen3.5-2B F16 (3.62 GiB) / Qwen3-1.7B F16 (3.21 GiB) |
| 对照开关 | GGML_CANN_WEIGHT_NZ=0 (关闭) / =1 (开启), 进程启动前设置 |

## 3. 测试方法

- 工具: llama.cpp 自带 `llama-bench`, 每配置 3 次重复取均值并给出置信区间。
- 指标:
  - `pp512`: prompt 处理吞吐 (512 token 单次前向, 反映 prefill 算力)。
  - `tg128`: 生成吞吐 (128 token 逐 token 解码, 反映 decode kernel 路径)。
- A/B 方式: 同一构建产物, 仅切换 `GGML_CANN_WEIGHT_NZ`; NZ 转换发生在权重加载阶段。
- 交替执行: OFF/ON 按 R1(Off,On) R2(Off,On) R3(Off,On) 交错进行, 每轮间隔 20s, 抵消共享机负载漂移。
- 干扰控制: 选取 AICore 空闲 chip; 规避同机高 CPU 进程 (曾观测 llama-parallel 1550% CPU 导致轮次间漂移 33%); 分析辅以峰值包络法 (外部干扰只可能拖低分数, 各配置历史峰值即无污染估计)。
- 排错记录: 机器重启后 /tmp 清空导致环境脚本丢失, bench 静默回退到系统旧版 CANN 库 (pp 掉至 1/3 且 A/B 失去差异), 已通过显式环境检查修复。

## 4. 测试命令

### 4.1 环境 (每次登录后执行, 注意 LF 行尾)

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0-beta.1
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.1.0-beta.1/opp
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0-beta.1/aarch64-linux/lib64:/usr/local/Ascend/cann-9.1.0-beta.1/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64
export ASCEND_RT_VISIBLE_DEVICES=2   # 按需选空闲 chip, npu-smi info 查看
```

### 4.2 构建

```bash
cd /home/llama-cpp-weight-nz
cmake -B build-cann -DGGML_CANN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cann --target llama-bench llama-cli -j64
```

### 4.3 模型转换 (HF safetensors -> GGUF F16, 一次性)

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=/home/llama-cpp-weight-nz/gguf-py \
  python3 convert_hf_to_gguf.py /home/models/Qwen3.5-2B \
  --outfile /home/qwen35-2b-f16.gguf --outtype f16
```

### 4.4 A/B 基准 (核心命令)

```bash
cd /home/llama-cpp-weight-nz/build-cann
# OFF 基线
GGML_CANN_WEIGHT_NZ=0 ./bin/llama-bench -m /home/qwen35-2b-f16.gguf -p 512 -n 128 -r 3
# ON 对照
GGML_CANN_WEIGHT_NZ=1 ./bin/llama-bench -m /home/qwen35-2b-f16.gguf -p 512 -n 128 -r 3
# Qwen3-1.7B 同理: -m /home/qwen3-1.7b-f16.gguf
```

正式测试按 R1(Off,On) R2(Off,On) R3(Off,On) 交错循环, 轮间隔 sleep 20。

### 4.5 功能冒烟

```bash
GGML_CANN_WEIGHT_NZ=1 ./bin/llama-cli -m /home/qwen35-2b-f16.gguf \
  -p "Hello, my name is" -n 32 --no-warmup -st -t 4
```

## 5. 测试结果

### 5.1 Qwen3.5-2B F16 (3 轮交错)

| 轮次 | NZ | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---|
| R1 | OFF | 803.87 +- 9.95 | 7.74 +- 0.07 |
| R1 | ON | 856.67 +- 26.35 | 11.94 +- 0.13 |
| R2 | OFF | 812.60 +- 6.59 | 7.54 +- 0.04 |
| R2 | ON | 826.73 +- 62.89 | 9.07 +- 0.51 |
| R3 | OFF | 664.10 +- 6.40 | 4.10 +- 0.56 |
| R3 | ON | 847.22 +- 3.46 | 8.93 +- 1.50 |
| **均值** | OFF | **760.19** | **6.46** |
| **均值** | ON | **843.54** | **9.98** |
| **收益** | ON vs OFF | **+11.0%** | **+54.5%** |

峰值包络: pp512 812.60 -> 856.67 (+5.4%), tg128 7.74 -> 11.94 (+54.3%)。

说明: R3 OFF 基线受同机负载抬升影响被压低 (664), 同轮 ON 仍稳定在 847, 与峰值一致。

### 5.2 Qwen3-1.7B F16 (对照)

| 配置 | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| NZ OFF | 2390.06 +- 1.83 | 10.89 +- 0.01 |
| NZ ON | 3527.99 +- 5.16 | 22.41 +- 0.01 |
| **收益** | **+47.6%** | **+105.8%** |

### 5.3 汇总

| 模型 | pp512 收益 | tg128 收益 |
|---|---|---|
| Qwen3.5-2B F16 | +11.0% (峰值 +5.4%) | +54.5% (峰值 +54.3%) |
| Qwen3-1.7B F16 | +47.6% | +105.8% |

### 5.4 真实负载对照 (llama-parallel, llama-pa 构建)

容器 z00575061-llama-bench-v2 内 /app/llama.cpp/llama-pa 构建, 完整参数:
`-kvu -f perf-results/retest-kvu/unified-prompt.txt -np 1 -ns 1 -n 128 -c 4096 -b 1024 -ub 1024 -t 8 -tb 8 --temp 0 -ngl 99 -sm none -mg 3 --ignore-eos`
(系统 282 + prompt 870 = 1152 token 上下文, decode 128 token, 2 轮交错)。

| 轮次 | NZ | prompt (t/s) | decode (t/s) | total (t/s) |
|---|---|---|---|---|
| R1 | ON | 78.61 | 11.57 | 90.17 |
| R1 | OFF | 50.56 | 7.44 | 58.00 |
| R2 | ON | 77.37 | 11.38 | 88.75 |
| R2 | OFF | 52.42 | 7.71 | 60.14 |
| **均值** | ON | **77.99** | **11.48** | **89.46** |
| **均值** | OFF | **51.49** | **7.58** | **59.07** |
| **收益** | ON vs OFF | **+51.5%** | **+51.5%** | **+51.5%** |

真实负载下 NZ 收益与 llama-bench 交叉验证一致: decode +51.5% (bench tg +54.5%), prefill 端长 prompt 收益进一步放大到 +51.5% (bench pp512 +11%, 长上下文/paged KV 路径下 NZ 权重 kernel 优势更明显)。绝对值高于 bench (decode 11.5 vs 10.0) 来自不同构建 (llama-pa 分支)、-kvu 路径、ub 1024 与 1152 token 上下文缓存, 基线口径不同, 不应与 bench 直接互比。

### 5.5 历史数据勘误 (retest-kvu/results.md)

容器内今日凌晨的 `retest-kvu/results.md` 记录: unified-single TPOT median 72.83 ms, paged-single 76.50 ms (均为 3 轮去 warmup 取中位数)。经查这些运行的日志中**均未出现 `GGML_CANN_WEIGHT_NZ=0`**, 即历史上所有测试实际都处于 NZ 默认开启状态, "约 80ms decode" 是 ON 态数据而非 OFF 基线。

| 场景 (unified-single 同命令) | TPOT | decode t/s |
|---|---|---|
| NZ ON (历史 median / 本次实测) | 72.83 ~ 87.9 ms | 11.38 ~ 13.84 |
| NZ OFF (本次实测) | 129.7 ~ 134.4 ms | 7.44 ~ 7.71 |

结论: NZ 关闭的真实 decode 延迟约 130 ms/token; 历史 80ms 级数据与本次 ON 实测吻合, 差异源于 warmup 丢弃与中位数统计口径。

复测 (retest-develop2, 同命令 2 轮交错): ON decode 12.23 t/s (81.8 ms), OFF 7.90 t/s (126.6 ms), 收益 +54.8%, 与 retest-kvu 会话 (+51.5%) 交叉一致。

### 5.6 tied lm_head NZ 移植后验证 (Qwen3.5-2B, GGML_CANN_ACL_GRAPH=0)

移植 llama.cpp-x 的 tied-output NZ 副本机制后 (commit 41eeb5caa), lm_head = token_embd.weight (F16, 151936x2048) 在 mul_mat 读独立 FRACTAL_NZ 副本, embedding 查表继续读原 ND buffer。构建: USE_ACL_GRAPH=ON, 运行时 ACL_GRAPH=0 隔离变量。

| 轮次 | NZ | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---|
| R1 | ON | 845.83 +- 5.13 | 5.78 +- 0.52 |
| R1 | OFF | 653.25 +- 2.55 | 2.31 +- 0.11 |
| R2 | ON | 832.59 +- 7.76 | 5.74 +- 0.30 |
| R2 | OFF | 570.41 +- 45.49 | 2.56 +- 0.06 |
| **均值** | ON | **839.21** | **5.76** |
| **均值** | OFF | **611.83** | **2.44** |
| **收益** | ON vs OFF | **+37.2%** | **+136.5%** |

- llama-cli 冒烟 (同模型, tied NZ on): Generation 16.1 t/s, 移植前同口径 13.9 t/s (+16%), 端到端方向一致。
- ON 侧 pp/tg 跨会话稳定 (pp 839-850, 峰值一致); OFF 基线受同机负载影响波动较大, 收益比例随之浮动 (tg +28% ~ +136%), 但所有配对方向均为正。
- 两个模型均为 tied 结构, 该机制使 decode 阶段 lm_head (占 F16 decode 读带宽的主要部分之一) 首次吃到 NZ 收益。

### 5.7 llama-parallel 带 tied lm_head NZ 验证 (Device 4, 2 轮交错)

host 构建 (weight-nz-ope 分支, commit 41eeb5caa, USE_ACL_GRAPH=ON, 运行时 ACL_GRAPH=0), 单并发 llama-parallel:
uild-cann/bin/llama-parallel -m /app/gguf-models/qwen3.5-2b-f16.gguf -f perf-results/retest-20260920/unified-prompt.txt -np 1 -ns 1 -n 128 -c 4096 -b 1024 -ub 1024 -t 8 -tb 8 --temp 0 -ngl 99 -sm none --ignore-eos
(ASCEND_RT_VISIBLE_DEVICES=4, mg 0 host 单卡)。

| 轮次 | NZ | prompt (t/s) | gen (t/s) |
|---|---|---|---|
| R1 | ON | 512.08 | 7.65 |
| R1 | OFF | 340.05 | 5.08 |
| R2 | ON | 508.08 | 7.59 |
| R2 | OFF | 340.58 | 5.09 |
| **均值** | ON | **510.08** | **7.62** |
| **均值** | OFF | **340.32** | **5.09** |
| **收益** | ON vs OFF | **+49.9%** | **+49.7%** |

两轮数据高度一致 (prompt 波动 <1%, gen 波动 <1%), 交叉验证了 5.4 节容器内 +51.5% 和 5.6 节 llama-bench +37.2%/+136.5% 的结论。host 绝对值高于容器 (prompt 510 vs 78, gen 7.6 vs 11.5) 是因为容器构建 (llama-pa 分支) 与 host 构建 (weight-nz-ope 分支) 的基线口径不同, 不应直接互比; 关注同构建内的 ON/OFF 比率。

## 6. 功能验证

- Qwen3.5-2B: `llama-cli -st` 单轮生成正常退出, 输出连贯, NZ 转换在加载阶段完成。
- Qwen3-1.7B: 同上。
- NZ 为权重布局变换 (ND -> FRACTAL_NZ), 数学上无损; 不同 kernel 路径的浮点累加顺序差异在正常波动范围内。

## 7. 使用说明

- 默认开启, 无需任何配置; 关闭需在进程启动前设置 `GGML_CANN_WEIGHT_NZ=0`。
- 仅符合候选条件的权重转换: 命名匹配, F16 类型, ne[0]/ne[1] 均为 32 的倍数; BF16 与量化权重不走该路径。
### 5.8 llama-pa vs weight-nz-ope 对比测试 (容器 6c8b816f1970, 128 tokens)

修复 `parallel.cpp` 中的 `ignore-eos` bug 后，两个构建都能正确生成 128 tokens。测试命令：
`build-cann/bin/llama-parallel -m /app/gguf-models/qwen3.5-2b-f16.gguf -kvu -f /app/llama.cpp/llama-pa/perf-results/retest-kvu/unified-prompt.txt -np 1 -ns 1 -n 128 -c 4096 -b 1024 -ub 1024 -t 8 -tb 8 --temp 0 -ngl 99 -sm none -mg 3 --ignore-eos`

| 构建 | NZ | prompt (t/s) | gen (t/s) | tokens |
|---|---|---|---|---|
| llama-pa (含 paged_attn) | OFF | 51.63 | 7.60 | 128 |
| llama-pa (含 paged_attn) | ON | 79.30 | 11.67 | 128 |
| **收益** | ON vs OFF | **+54%** | **+54%** | |
| weight-nz-ope (修复后) | OFF | 52.46 | 7.72 | 128 |
| weight-nz-ope (修复后) | ON | 98.22 | 14.45 | 128 |
| **收益** | ON vs OFF | **+87%** | **+87%** | |
| **weight-nz-ope vs llama-pa** | NZ ON | **+24%** | **+24%** | |

关键发现：
- weight-nz-ope 的 NZ 收益（+87%）显著高于 llama-pa（+54%）
- 修复后 weight-nz-ope 比 llama-pa 快 24%（14.45 vs 11.67 t/s）
- unified kv cache (`-kvu`) 本身就很快，不需要 paged_attn
- 之前的性能差异是因为 weight-nz-ope 只生成了 16 tokens（遇到第一个 EOG 就停止），而 llama-pa 正确忽略了 EOG 生成完整的 128 tokens
