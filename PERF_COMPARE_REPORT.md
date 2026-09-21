# Weight NZ 优化对比测试报告

日期: 2026-09-21
模型: Qwen3.5-2B F16 (`/app/gguf-models/qwen3.5-2b-f16.gguf`)
环境: 容器 `6c8b816f1970`, Atlas 310P3, CANN 9.1.0

## 1. 测试命令

```bash
build-cann/bin/llama-parallel \
  -m /app/gguf-models/qwen3.5-2b-f16.gguf \
  -kvu \
  -f /app/llama.cpp/llama-pa/perf-results/retest-kvu/unified-prompt.txt \
  -np 1 -ns 1 -n 128 -c 4096 -b 1024 -ub 1024 \
  -t 8 -tb 8 --temp 0 -ngl 99 -sm none -mg 3 --ignore-eos
```

NZ 开关: `GGML_CANN_WEIGHT_NZ=0|1`
Tied lm_head 开关: `GGML_CANN_TIED_OUTPUT_NZ=0|1`
ACL Graph: `GGML_CANN_ACL_GRAPH=0`

## 2. llama-pa vs weight-nz-ope 对比 (3 轮交替)

| 轮次 | 构建 | NZ | prompt (t/s) | gen (t/s) |
|------|------|-----|-------------|-----------|
| R1 | llama-pa | ON | 81.43 | 11.98 |
| R1 | weight-nz-ope | ON | 98.26 | 14.46 |
| R1 | llama-pa | OFF | 52.80 | 7.77 |
| R1 | weight-nz-ope | OFF | 52.06 | 7.66 |
| R2 | llama-pa | ON | 80.40 | 11.83 |
| R2 | weight-nz-ope | ON | 98.14 | 14.44 |
| R2 | llama-pa | OFF | 51.81 | 7.62 |
| R2 | weight-nz-ope | OFF | 51.74 | 7.61 |
| R3 | llama-pa | ON | 79.56 | 11.70 |
| R3 | weight-nz-ope | ON | 95.77 | 14.09 |
| R3 | llama-pa | OFF | 52.18 | 7.68 |
| R3 | weight-nz-ope | OFF | 52.16 | 7.67 |

### 汇总

| 构建 | NZ | prompt 均值 (t/s) | gen 均值 (t/s) | NZ 收益 (prompt) | NZ 收益 (gen) |
|------|-----|-------------------|----------------|-----------------|--------------|
| llama-pa | OFF | 52.26 | 7.69 | - | - |
| llama-pa | ON | 80.46 | 11.84 | +54.0% | +54.0% |
| weight-nz-ope | OFF | 51.99 | 7.65 | - | - |
| weight-nz-ope | ON | 97.39 | 14.33 | +87.3% | +87.3% |
| **weight-nz-ope vs llama-pa** | **ON** | **97.39 vs 80.46** | **14.33 vs 11.84** | **+21.0%** | **+21.0%** |

## 3. lm_head (tied output) NZ 影响测试 (3 轮交替)

验证 weight-nz-ope 比 llama-pa 快 21% 是否由 lm_head NZ 优化贡献。

| 轮次 | TIED | prompt (t/s) | gen (t/s) |
|------|------|-------------|-----------|
| R1 | ON | 98.07 | 14.43 |
| R1 | OFF | 80.21 | 11.80 |
| R2 | ON | 98.26 | 14.46 |
| R2 | OFF | 78.80 | 11.59 |
| R3 | ON | 101.40 | 14.92 |
| R3 | OFF | 80.28 | 11.81 |

### 汇总

| 配置 | prompt 均值 (t/s) | gen 均值 (t/s) | 提升 (prompt) | 提升 (gen) |
|------|-------------------|----------------|--------------|-----------|
| TIED ON | 99.24 | 14.60 | - | - |
| TIED OFF | 79.76 | 11.73 | - | - |
| **TIED ON vs OFF** | **99.24 vs 79.76** | **14.60 vs 11.73** | **+24.4%** | **+24.5%** |

## 4. 结论

1. **weight-nz-ope 比 llama-pa 快 21%**, 核心原因是 lm_head (tied output) NZ 优化。
2. **lm_head NZ 单独贡献 +24.5% 的 gen 提升**, 与 WNZ vs PA 的 21% 差异高度吻合。
3. **weight-nz-ope 的 NZ 总收益 (+87%)** 远高于 llama-pa (+54%), 因为 llama-pa 没有 lm_head NZ 优化。
4. **两个构建的 NZ OFF 基线基本一致** (52.26 vs 51.99 t/s), 说明底层代码差异不影响基线性能。
5. **unified kv cache (`-kvu`)** 本身就足够高效, 不需要 paged_attn 也能达到优秀性能。
