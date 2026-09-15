# CANN Hybrid PagedAttention 轻量接入设计

状态：已实现第一版

日期：2026-09-15

## 1. 背景

Qwen3.5-MoE 这类混合架构模型包含两类层：

- full-attention 层：可复用现有 PagedAttention
- linear-attention 或 GDN 层：继续使用 `llama_memory_recurrent`

当前 llama.cpp 的普通 hybrid 实现是：

```text
llama_memory_hybrid
  mem_attn : llama_kv_cache
  mem_recr : llama_memory_recurrent
```

现有 paged 实现是：

```text
llama_kv_cache_paged
  block manager
  paged K/V pools
  write slots
  block table
  context lens
```

两条路径还没有组合。本文定义一个轻量 hybrid-paged 接入方案。

## 2. 设计边界

PagedAttention 本身尚未合入社区，因此 paged 生态内的文件可以继续演进。重点是避免修改 llama.cpp 原生框架代码。

| 类别 | 修改策略 |
|---|---|
| `llama-kv-cache-paged.*` | 可以修改 |
| `llama-paged-scheduler*` | 可以修改 |
| `ggml-cann/paged_attn*` | 可以修改 |
| 新增 `llama-memory-hybrid-paged.*` | 主要实现位置 |
| 新增 hybrid-paged graph helper | 主要实现位置 |
| `llama-memory-hybrid.*` | 不修改 |
| `llama-kv-cache.*` | 不修改 |
| `llama-memory-recurrent.*` | 不修改 |
| `llama-graph.*` | 目标是不修改 |
| `llama-model.cpp` | 只保留一个 memory factory hook |
| `models/qwen35moe.cpp` | 只保留一个 graph input 分支 |

原生代码的理想 diff 只包含：

```text
src/llama-model.cpp       : paged memory factory hook
src/models/qwen35moe.cpp  : hybrid-paged graph input 分支
src/CMakeLists.txt        : 新增 plugin source
```

## 3. 核心结论：组合，不继承

不建议让 `llama_memory_hybrid_paged` 继承 `llama_memory_hybrid`，也不建议让 paged cache 伪装成普通 `llama_kv_cache`。

原因是原生 hybrid 类把 attention cache 写死为：

```cpp
std::unique_ptr<llama_kv_cache> mem_attn;
```

context 也返回：

```cpp
const llama_kv_cache_context *
```

而 paged cache 需要暴露：

```cpp
const llama_kv_cache_paged_context *
paged_write_slots
paged_block_table
paged_context_lens
paged_batch_offsets
paged_batch_lens
```

两者不是同一接口。

推荐结构是两个兄弟实现：

```text
llama_memory_hybrid        = normal KV cache + recurrent cache
llama_memory_hybrid_paged  = paged KV cache + recurrent cache
```

这样可以复用原生 hybrid 的逻辑，但不复用它的类接口。

## 4. 总体架构

### 4.1 Memory 组合

新增：

```cpp
class llama_memory_hybrid_paged : public llama_memory_i {
private:
    std::unique_ptr<llama_kv_cache_paged> mem_attn;
    std::unique_ptr<llama_memory_recurrent> mem_recr;
};
```

新增 context：

```cpp
class llama_memory_hybrid_paged_context : public llama_memory_context_i {
public:
    const llama_kv_cache_paged_context * get_paged_attn() const;
    const llama_memory_recurrent_context * get_recr() const;

private:
    llama_memory_context_ptr ctx_attn;
    llama_memory_context_ptr ctx_recr;
};
```

context 的职责：

1. 持有两个子 context。
2. `next()` 同时推进两个子 context。
3. `apply()` 合并两个子 context 的状态。
4. `get_ubatch()` 返回当前 ubatch。

paged cache 在 graph 执行期间写入 K/V，所以 paged 子 context 的 `apply()` 通常可以是 no-op。recurrent 子 context 继续沿用原生生命周期。

### 4.2 不新增组合 graph input

不新增 `llm_graph_input_mem_hybrid_paged`。

graph 层直接创建两个现有 input：

```text
llm_graph_input_attn_kv_paged
llm_graph_input_rs
```

这样避免了新增组合 input 的 `set_input()`、`can_reuse()` 和生命周期逻辑。

### 4.3 graph helper 放在 plugin 文件

不要给 `llm_graph_context` 增加新的成员函数。新增独立 helper：

```cpp
llm_graph_input_attn_kv_paged * build_hybrid_paged_attn_inp(
        llm_graph_context & gctx,
        const llama_kv_cache_paged_context * paged_mctx);

llm_graph_input_rs * build_hybrid_paged_rs_inp(
        llm_graph_context & gctx,
        const llama_memory_recurrent_context * recr_mctx);
```

这些 helper 可以放在新文件，例如：

```text
src/llama-hybrid-paged-graph.h
src/llama-hybrid-paged-graph.cpp
```

它们只使用 `gctx.ctx0`、`gctx.ubatch`、`gctx.res` 等现有成员，并通过 `res->add_input()` 注册 graph input。

这使 `llama-graph.h` 和 `llama-graph.cpp` 不需要新增 hybrid-paged 接口。

## 5. Memory 生命周期

### 5.1 初始化

`llama_memory_hybrid_paged` 构造时创建：

```cpp
mem_attn = std::make_unique<llama_kv_cache_paged>(...);
mem_recr = std::make_unique<llama_memory_recurrent>(...);
```

paged cache 使用现有 block manager 和物理 K/V 池。recurrent cache 使用原生实现。

### 5.2 init_batch

不要直接调用 `mem_attn->init_batch()`。paged cache 自己的 `init_batch()` 会使用 `split_simple()`，不符合 GDN 层需要的 per-sequence equal split。

hybrid-paged memory 自己切分：

```cpp
balloc.split_reset();

while (true) {
    llama_ubatch ubatch;

    if (embd_all) {
        ubatch = balloc.split_seq(n_ubatch);
    } else {
        ubatch = balloc.split_equal(
            n_ubatch,
            true,
            mem_recr->n_rs_seq > 0 ? mem_recr->n_rs_seq + 1 : 0);
    }

    if (ubatch.n_tokens == 0) {
        break;
    }

    ubatches.push_back(std::move(ubatch));
}
```

然后：

1. 调用 `mem_recr->prepare(ubatches)`。
2. 用当前 paged batch metadata 创建 paged context。
3. 用同一组 ubatches 创建 recurrent context。
4. 组合成 `llama_memory_hybrid_paged_context`。

paged context 的创建建议收在 paged 文件里：

```cpp
llama_memory_context_ptr llama_kv_cache_paged::create_batch_context(
        const std::vector<llama_ubatch> & ubatches) {
    GGML_ASSERT(last_paged_info && "no paged batch info set");

    auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, ubatches);
    ctx->set_batch_data(*last_paged_info);

    return ctx;
}
```

这是 `llama-kv-cache-paged` 侧的一个小 helper，可以接受。

因为 GDN 层要求 per-sequence equal split，而 paged scheduler 给出的 metadata 描述的是整批 token，所以 paged context 还需要按当前 ubatch 派生 metadata：

```text
global write slots      -> 当前 ubatch 的 write slots
global context lens     -> 当前 ubatch 结束时的 context lens
global batch offsets    -> 当前 ubatch 内的 local offsets
global batch lens       -> 当前 ubatch 的 per-sequence token 数
```

普通 dense paged 路径继续使用原始全局 metadata；只有 hybrid-paged memory 通过
`create_batch_context()` 显式启用 per-ubatch metadata。

### 5.3 next 和 apply

```cpp
bool llama_memory_hybrid_paged_context::next() {
    ctx_attn->next();
    ctx_recr->next();

    return ++i_next < ubatches.size();
}

bool llama_memory_hybrid_paged_context::apply() {
    bool res = ctx_attn->apply();
    res &= ctx_recr->apply();
    return res;
}
```

两个子 context 必须使用同一组 ubatches 和相同顺序。

### 5.4 清理

请求结束时必须同时释放：

```text
paged KV blocks
recurrent state slots
```

如果只释放 paged block，recurrent slot 会泄漏；如果只释放 recurrent state，下一批请求可能复用旧 GDN state。

## 6. Graph 接入

### 6.1 Qwen3.5-MoE 分支

原逻辑：

```cpp
auto * inp = build_inp_mem_hybrid();

if (hparams.is_recr(il)) {
    cur = build_layer_attn_linear(inp->get_recr(), cur, il);
} else {
    cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
}
```

轻量版改为：

```cpp
auto * paged_hybrid =
    dynamic_cast<const llama_memory_hybrid_paged_context *>(mctx);

llm_graph_input_i * inp_attn = nullptr;
llm_graph_input_rs * inp_rs  = nullptr;

if (paged_hybrid) {
    inp_attn = build_hybrid_paged_attn_inp(*this, paged_hybrid->get_paged_attn());
    inp_rs   = build_hybrid_paged_rs_inp(*this, paged_hybrid->get_recr());
} else {
    auto * inp = build_inp_mem_hybrid();
    inp_attn = inp->get_attn();
    inp_rs   = inp->get_recr();
}

for (...) {
    if (hparams.is_recr(il)) {
        cur = build_layer_attn_linear(inp_rs, cur, il);
    } else {
        cur = build_layer_attn(inp_attn, cur, inp_pos, sections, il);
    }
}
```

### 6.2 build_layer_attn 放宽输入类型

原函数签名：

```cpp
ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ...);
```

改为：

```cpp
ggml_tensor * build_layer_attn(
        llm_graph_input_i * inp,
        ...);
```

内部调用：

```cpp
cur = build_attn(
        inp,
        nullptr,
        nullptr,
        nullptr,
        Qcur,
        Kcur,
        Vcur,
        nullptr,
        nullptr,
        nullptr,
        kq_scale,
        il);
```

`llm_graph_context::build_attn(llm_graph_input_i *, ...)` 已经能 dispatch 到：

```text
normal KV attention
ISWA attention
paged attention
no-cache attention
```

因此 Qwen3.5-MoE 不需要理解 paged metadata。

paged attention 的输出宽度必须按 Q 实际宽度计算：

```cpp
const int64_t attn_width = q_cur->ne[0] * q_cur->ne[1];
cur = ggml_reshape_2d(ctx0, cur, attn_width, n_tokens);
```

不能写成 `hparams.n_embd`。Qwen3.5-MoE 的 attention 输出宽度是
`head_dim * n_head`，与其 hidden size 不一定相同。

另外，hybrid-paged 模式允许 `n_batch >= n_ubatch`。paged scheduler 对普通 dense
模型仍保持 `n_batch == n_ubatch` 的约束。

## 7. Memory factory hook

在 `llama-model.cpp` 中只加一个入口：

```cpp
if (cparams.kv_paged) {
    res = llama_paged_memory_try_create(
            *this,
            params,
            cparams,
            filter_attn,
            filter_recr);

    if (res) {
        return res;
    }
}
```

paged plugin 内部决定：

```text
dense model  -> llama_kv_cache_paged
hybrid model -> llama_memory_hybrid_paged
不支持       -> nullptr，回落原生路径
```

第一版可以只处理 Qwen3.5-MoE。其他架构返回 `nullptr`，走原生分支。

hybrid-paged memory 的创建参数包括：

```text
model
type_k / type_v
block_size
n_ctx_seq
n_ubatch
n_gpu_blocks
n_cpu_blocks
watermark
n_seq_max
n_rs_seq
offload
filter_attn
filter_recr
```

## 8. Paged cache 收口

paged cache 可以继续自由演进，不需要受原生 KV cache 抽象限制。

### 8.1 第一版：不改 layer mapping

第一版可以让 `llama_kv_cache_paged` 按 `hparams.n_layer()` 分配物理池。对 Qwen3.5-MoE 来说，模型有 32 层，但只有约 8 层是 full-attention。recurrent 层的 paged pool 不会被访问。

这个版本显存利用率不理想，但侵入最小，适合先打通正确性。

### 8.2 第二版：layer mapping

正确性稳定后，在 paged cache 内加入映射：

```text
model layer id -> physical paged layer id
recurrent layer id -> invalid
```

例如：

```text
model layer 3  -> paged layer 0
model layer 7  -> paged layer 1
model layer 11 -> paged layer 2
...
```

物理池数量从 32 降到 8。

该逻辑应完全收在：

```text
llama-kv-cache-paged.h
llama-kv-cache-paged.cpp
```

不暴露给 graph 或 model 层。

## 9. Scheduler 集成

现有 paged scheduler 直接假设顶层 memory 是 `llama_kv_cache_paged`：

```cpp
dynamic_cast<llama_kv_cache_paged *>(ctx->get_memory());
```

hybrid-paged 模式下顶层 memory 是组合对象，需要调整为：

```cpp
auto * memory = ctx->get_memory();

auto * hybrid_paged =
        dynamic_cast<llama_memory_hybrid_paged *>(memory);

auto * paged_kv =
        hybrid_paged
                ? hybrid_paged->get_paged_mem()
                : dynamic_cast<llama_kv_cache_paged *>(memory);
```

请求结束时：

```cpp
paged_kv->free_blocks(group);

if (hybrid_paged) {
    hybrid_paged->seq_rm(group.request_id, -1, -1);
}
```

第一阶段可以先只支持单请求，第二步再补齐多请求的准入和释放。

多请求准入要检查两类资源：

```text
free paged KV blocks   >= 本批请求需要的 KV blocks
free recurrent slots   >= 本批 sequence 数量
```

## 10. CANN backend

graph 层保持 backend 无关：

```text
K/V pools 是普通 GGML tensor
paged metadata 是 integer tensor
graph 调用 ggml_paged_attn
CANN backend 内部包装成 ATB tensor
```

CANN 后端继续隐藏：

1. NZ cache layout。
2. ND 到 NZ 的转换。
3. PA、PFA、Splitfuse 的 phase dispatch。
4. reshape-and-cache。

不在 graph 或 model 层引入 CANN layout flag。

## 11. 实施计划

### Phase 0：基线验证

1. Qwen3-8B dense PagedAttention 回归。
2. Qwen3.5-MoE 普通 hybrid 路径回归。
3. 记录 prefill 和 decode 基线。

### Phase 1：单请求 POC

1. 新增 `llama-memory-hybrid-paged.*`。
2. 新增 hybrid-paged graph helper。
3. 给 `llama_kv_cache_paged` 增加 `create_batch_context()`。
4. 在 `llama-model.cpp` 加 memory factory hook。
5. 在 Qwen3.5-MoE 加 graph input 分支。
6. 将 `build_layer_attn()` 输入类型放宽为 `llm_graph_input_i *`。
7. 单请求固定 sampling 对比普通 hybrid 输出。

退出标准：

1. logits 或 perplexity 与普通 hybrid 路径一致。
2. CANN 后端完整执行不触发 AICORE timeout。
3. full-attention 层确实调用 paged attention。
4. GDN 层 state 读写正确。
5. 请求结束后 KV block 和 recurrent state 都释放。

### Phase 2：多请求

1. paged scheduler 识别 hybrid-paged memory。
2. 请求结束同时释放 KV blocks 和 recurrent slots。
3. 增加双资源准入检查。
4. mixed prefill 和 decode batch 测试。
5. 测量 TTFT、TPOT、吞吐和显存利用率。

### Phase 3：显存优化

1. paged cache 增加 layer mapping。
2. 只为 full-attention 层分配物理池。
3. 更新 memory breakdown。
4. 重新评估可支撑 context 和并发数。

### Phase 4：性能优化

1. 保持 NZ 和 layout conversion 在 CANN 后端内。
2. 减少 graph 调度和 backend 同步空泡。
3. 如收益明确，再做 Q/K/V projection 和 normalization 融合。
4. 后续再考虑 checkpoint 和 prefix reuse。

## 12. 复用矩阵

| 组件 | 复用策略 |
|---|---|
| `llama_memory_recurrent` | 原样复用 |
| recurrent graph input | 复用 |
| `build_layer_attn_linear()` | 复用 |
| GDN ops | 复用 |
| paged graph input | 复用 |
| paged attention graph builder | 复用 |
| CANN PagedAttention | 复用 |
| hybrid ubatch splitting | 复用逻辑，新类内实现 |
| `llama_memory_hybrid` | 不复用类接口 |
| normal `llama_kv_cache` | hybrid-paged 模式不使用 |

## 13. 风险

### 13.1 ubatch 顺序不一致

paged metadata 和 recurrent context 必须使用同一组 ubatch 和相同 sequence 顺序。否则 write slot、block table、GDN state 都会错位。

### 13.2 recurrent state 泄漏

paged scheduler 释放 KV block 时，必须同时通知 hybrid memory 释放 recurrent slot。

### 13.3 sequence copy 不支持

当前 paged cache 不支持 `seq_cp()`。hybrid-paged 第一版应在 context 创建或 operation 层 fail fast。

### 13.4 layer id 误用

如果直接把 model layer id 当作 paged pool id，普通 dense 模型没问题，但 hybrid 模型会出现大量未使用池。后续必须显式 layer mapping。

### 13.5 speculative rollback

recurrent snapshot 和 paged KV rollback 生命周期不同。第一阶段禁用 speculative decoding。

### 13.6 CPU/CANN block 混用

CPU swap pool 和 CANN device pool 必须保持相同 block size 和元素布局，否则拷贝和 NZ 解释会出错。

## 14. 测试计划

### 14.1 单元测试

1. hybrid-paged context 能创建两个子 context。
2. ubatch 顺序在 paged 和 recurrent 子 context 中一致。
3. `seq_rm()` 同时清理 paged 位置信息和 recurrent state。
4. paged metadata 尺寸与 ubatch 一致。
5. 不支持 `seq_cp()` 时 fail fast。

### 14.2 功能测试

1. Qwen3.5-MoE 普通 hybrid 输出与 hybrid-paged 输出一致。
2. Qwen3-8B dense PagedAttention 回归通过。
3. 长 prompt 不越界。
4. context clear 后不残留 state。
5. 请求结束不泄漏 state。

### 14.3 性能测试

普通 hybrid 和 hybrid-paged 分别记录：

```text
PP=512, TG=0
PP=128, TG=0
TG=32, batch=1
PP=512, TG=128, batch=1/4/10
```

CANN kernel 维度记录：

```text
PA
PFA
MatMul
GroupedMatmul
TransData
Cast
TensorMove
```

## 15. 待定问题

1. recurrent state snapshot 是否与 paged KV 使用同一个 device pool？
2. layer mapping 应该在 paged cache 构造时固定，还是运行时可扩展？
3. 不支持 sequence copy 时，是回退普通 hybrid，还是直接创建失败？
4. Phase 2 是否需要新的 hybrid scheduler wrapper，还是扩展现有 paged scheduler？

## 16. 当前验证记录

### 16.1 功能

容器 `z00575061-llama-bench-v2` 使用 Qwen3.5-MoE 4.7B F16 GGUF：

```text
test-paged-kv-e2e
  top-5 argmax match : ref=16, paged=16
  top-5 set overlap  : 5/5
  token window       : passed
```

Qwen3-8B dense PagedAttention 的 top-5 overlap 保持 5/5；strict token
window 在 CANN 累加顺序下会偶发漂移，只作为观察项。Qwen3.5-MoE 普通 hybrid
与 hybrid-paged 在同一 prompt、未套 chat template 时输出一致；两者都会
重复输出 token "1"。这不能作为 hybrid-paged 正确性失败信号。

### 16.2 多请求

```text
4 requests, 64 prompt tokens, 32 generated tokens
aggregate decode throughput : 11.5 tok/s
average TPOT                : 370 ms
result                      : no hang, no crash
```

该结果包含首次图创建和 ATB warmup 关闭后的算子编译开销；后续稳定
运行会低于首次运行延迟。

### 16.3 实现备注

ATB PagedAttention / SplitFuse operation cache 不能只按 op 是否为空判断。
backend 初始化 warmup 使用的 head_dim 可能与当前模型不同。现在缓存会按
`headNum`, `kvHeadNum`, `headDim` 失效并重建，避免不同形状模型复用错误
tiling。debug metadata dump 只在 `GGML_CANN_PAGED_DEBUG=1` 时开启。

### 16.4 dense Qwen3.5 2B

dense `QWEN35` 也接入 hybrid-paged memory。使用 `Q4_K_M` 权重时，
non-paged 与 hybrid-paged 路径的 e2e 结果一致：

```text
top-5 argmax match : ref=2545, paged=2545
top-5 set overlap  : 5/5
strict token window: passed
```

这说明 dense 分支的 paged metadata、GDN state 和 graph input 生命周期
没有错位。
