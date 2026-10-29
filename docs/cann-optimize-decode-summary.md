# `cann-optimize-decode` 相对 `develop` 的改动总结

## 改动范围

基线：`gitcode/develop`，提交为 `642fcab26bc9b7d5e410846f1eab0afbd3b8f6de`。

当前分支：`cann-optimize-decode`，最新提交为 `70aba7e65`。

净差异：12 个文件，新增 161 行，删除 59 行。

| 模块 | 改动 | 目的 |
|---|---|---|
| Paged KV | `llama_kv_cache_paged::init_update()` 返回 `LLAMA_MEMORY_STATUS_NO_UPDATE` | 避免 decode 每步重置并重建 worst-case reserve graph |
| 执行阶段 | `graph_compute()` 新增 `prefill` 参数，并把 `graph_prefill` 设置到 CANN backend | 显式区分 prefill 和 decode，不再依赖 matmul shape 推断 |
| ACL Graph | prefill 默认 eager；decode 可以复用 graph；`GGML_CANN_PREFILL_USE_GRAPH=1` 才开启 prefill graph | 保留 decode graph replay 收益，同时避免 prefill graph capture 不稳定 |
| Warmup | paged scheduler warmup 期间临时设置 `GGML_CANN_NO_LRU=1`，结束后恢复 | 让 warmup decode 强制捕获新 graph，真实 decode 恢复正常 LRU replay |
| Block table | 固定 block table capacity，未使用的位置用 `-1` padding | 保持 decode 各步 tensor shape 和 graph replay key 稳定 |
| Hybrid Paged Backend | 设备判断从 `== GPU` 改为 `!= CPU` | 让 Ascend CANN accelerator 设备正确进入 paged hybrid GPU backend 路径 |
| 清理无效实验 | 移除 GDN 自定义 kernel、matmul cache 和 F16 input cache 相关实验 | 只保留经过 A/B 验证的 decode 稳定性和 graph 复用优化 |
| 兼容性 | `--kv-block-size` 允许任意正的 16 的倍数 | 适配 PagedAttention block 对齐要求 |

## 功能测试

构建并运行 scheduler 测试：

```bash
cmake -B build-cann -DGGML_CANN=ON
cmake --build build-cann --target test-paged-kv -j
./build-cann/bin/test-paged-kv
```

| 测试用例 | 覆盖内容 |
|---|---|
| `test_scheduler_single_request_decode` | 单请求执行 32 token prefill，随后 decode 1 token；校验 batch shape 和 context length |
| `test_scheduler_multi_request_decode` | 两个 32 token 请求分别 prefill，随后合批 decode；校验 `n_seq=2`、每请求 1 token、context length |

结果：`test-paged-kv: ALL PASSED`。

## 性能测试

模型：Qwen3-1.7B F16。

运行模式：PagedAttention、单请求、`N=128`。

测试方法：每个分支或配置连续执行 5 轮，取平均值。

```bash
export GGML_CANN_WEIGHT_NZ=1
export GGML_CANN_ACL_GRAPH=0

./build-cann/bin/llama-paged \
  -m /home/z00575061/models/Qwen3-1.7B-F16.gguf \
  -kvp -np 1 -ns 1 -n 128 -c 4096 -b 1024 -ub 1024 \
  -t 8 -tb 8 --temp 0 -ngl 99 -sm none -mg 6 --ignore-eos
```

| 分支 / 配置 | Decode t/s | TPOT | 对比 develop |
|---|---:|---:|---:|
| `cann-optimize-decode`，保留 `NO_UPDATE` | 33.41 | 30.40 ms | t/s 提升 12.3% |
| 回退 `NO_UPDATE` | 29.68 | 34.22 ms | t/s 降低 0.2% |
| `develop` | 29.75 | 34.14 ms | 基线 |

## 结论

主要收益来自 `LLAMA_MEMORY_STATUS_NO_UPDATE`，它避免 PagedAttention decode 每步重建 llama.cpp 计算图。Matmul、F16 input cache 和 GDN 相关实验没有稳定收益，已从当前分支移除。prefill 默认保持 eager，decode 保留 graph reuse 路径。

## 无收益项回退后的复测

无收益项已按 `develop` 原样回退，保留 `NO_UPDATE`、显式 prefill / decode 阶段标记、block table 稳定化、warmup graph 支撑和新增测试用例。

| 配置 | Decode t/s | TPOT | 说明 |
|---|---:|---:|---|
| `cann-optimize-decode`，清理后 | 32.41 | 31.38 ms | 5 轮平均 |
| develop 基线 | 29.16 | 34.82 ms | 5 轮平均 |

清理后仍保留约 `11.2%` 的 decode t/s 收益。
