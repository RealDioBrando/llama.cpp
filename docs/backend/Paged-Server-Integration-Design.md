# Paged Scheduler 对接 llama-server 集成设计

状态:第一阶段已实现(未编译验证,见第 10 节实施记录)

日期:2026-10-25

## 1. 背景与目标

本仓库已在 llama 层实现完整的 paged KV cache 与 vLLM 风格连续批处理调度器
(`llama_paged_scheduler_*`,见 `src/llama-paged-scheduler-impl.cpp`),并配有两个块级前缀缓存
与 hybrid GDN checkpoint 的设计文档(`CANN-Hybrid-PagedAttention-Design.md`,
`CANN-Hybrid-PagedAttention-PrefixCache-Design.md`)。目前该调度器仅被
`examples/paged/paged.cpp` 与 `tests/test-paged-kv*.cpp` 使用,llama-server 仍走默认的
静态 slot 调度(`get_available_slot` + `pre_decode/decode/post_decode`)。

本文档调研两侧现状,给出 paged_scheduler 对接 server 的集成方案:

- 侵入式修改尽量少:通过策略接口 + 工厂把调度器做成可插拔组件,默认路径行为不变;
- 做合理的抽象:接口边界与 server 现有解耦点(queue 回调、server_slot)对齐;
- 第一阶段只提供采样的基础功能,工具调用、投机推理、结构化输出、多模态、
  embedding、reranker 等拓展功能留到第二阶段设计。

## 2. 现状盘点

### 2.1 paged 调度器(llama 层)

公共 C API(`include/llama.h:1660-1713`):

| API | 作用 |
|---|---|
| `llama_paged_scheduler_init(ctx)` | 从 ctx 取出 paged KV 内存,初始化调度器;非 paged ctx 返回 nullptr |
| `llama_paged_scheduler_warmup(sched, ctx)` | 一次小批 + 一次图尺寸的合成 prefill/decode,预热后端与 ATB 内核 |
| `llama_paged_scheduler_add_request(sched, tokens, n, request_id)` | 入队一个请求;request_id 即 llama seq_id |
| `llama_paged_scheduler_prepare_batch(sched, batch)` | 产出一个 `llama_batch`(调度器持有并复用);false 表示死锁/活锁,服务端对这类失败不做任何重试或自愈,直接把受影响的请求按错误路径收尾,而不是像默认调度路径那样尝试恢复 |
| `llama_paged_scheduler_update(sched, batch, tokens, stop_flags)` | decode 后回填:每 seq 一个采样 token 与一个停止标志 |
| `llama_paged_scheduler_get_seq_state(sched, id, out)` | 查询 `n_prompt/n_prompt_total/n_decoded/n_past/t_arrival_us/t_first_token_us` |
| `llama_paged_scheduler_get_batch_info(sched)` | 当前批的路由元数据(`n_seq/batch_offsets/batch_lens/...`),下次 prepare 前有效 |
| `llama_paged_scheduler_set_on_finish(sched, cb, ud)` | 请求被回收时回调(主线程,prepare_batch 内部触发) |

调用契约(以 `examples/paged/paged.cpp:215-289` 为准):

- 调用方自己采样,调度器不做任何采样;`update` 的 `tokens[i]` 在非采样步可为
  `LLAMA_TOKEN_NULL`;
- 仅在 decode 步(`batch_lens[i] == 1`)或 prefill 末块
  (`batch_lens[i] > 1 && state.n_past + batch_lens[i] == n_prompt_total`)产生 logits
  并采样,采样下标为 `batch_offsets[i] + batch_lens[i] - 1`;
- `stop_flags[i] != 0` 或 `n_past >= n_ctx` 时请求标记 FINISHED,下一次
  `prepare_batch` 的 `step()` 内回收并触发 `on_finish`;
- prefill 独占步:任一候选处于 prefill 时整批只保留该请求
  (`llama-paged-scheduler-impl.cpp:90-107`),prefill 与 decode 不混批;
- 内部三级队列 running/swapped/waiting:KV 块不足时先驱逐缓存块,再抢占最新运行请求
  (换出到 CPU 块池,CPU 池满则释放重算);块级前缀缓存(链式哈希 + 块共享,
  hybrid 模型另配 GDN checkpoint 槽)在入队时自动命中。

约束(第一阶段,`examples/paged/README.md:22-28`):

- 必须单卡全量卸载(`validate_paged_kv_placement`,`src/llama-context.cpp:87-111`);
- 纯注意力路径要求 `n_batch == n_ubatch`(`src/llama-paged-scheduler.cpp:42-44`);
- 不支持 SWA;hybrid paged 目前仅 `LLM_ARCH_QWEN35/QWEN35MOE`;
- `request_id` 必须小于 `n_seq_max`,且 hybrid 模型要避开 checkpoint 保留段
  (`llama-paged-scheduler-impl.cpp:29-38,137-141`);
- 单请求 prompt 长度必须小于 `n_ctx`(逐请求上限,而非总预算)。

### 2.2 server 现状

文件布局:本 fork 已将 server 拆成约 13 个文件,核心是 `tools/server/server-context.cpp`
(约 5558 行)+ `server-context.h`(pimpl,`server_context` 公开接口只暴露
`load_model/start_loop/terminate/get_meta` 等)。

请求主链路(均为 `server-context.cpp`):

```
HTTP handler (server_routes, 4257+)
  -> server_task -> server_queue::post
  -> start_loop: process_new_tasks -> process_single_task (2375)
       COMPLETION/EMBEDDING/RERANK/INFILL:
         get_available_slot (1547)  无空闲槽则 queue_tasks.defer
         launch_slot_with_task (1707)  建 common_sampler,槽状态 STARTED
  -> on_update_slots 回调 (1408-1416 接线)
       update_slots (2793)             <- llama_context (ctx_tgt) 只在这三段被消费
         pre_decode (2909):  上下文迁移/前缀复用/构建 server_batch
                             [ctx] slot.mem.* 做 KV 维护 (seq_rm/seq_cp/seq_add)
         decode (3647):      llama_decode + 分批重试
                             [ctx] llama_decode(ctx_tgt, batch) 前向计算, KV 按 (seq_id, pos) 写入
         post_decode (3784): 采样 -> process_token -> send_partial/final_response
                             [ctx] 从 ctx 的 logits 缓冲采样
  -> server_task_result_* -> queue_results -> SSE/JSON
```

`llama_context`(全局唯一 `ctx_tgt`)在图中不可见,因为它不属于任务/槽链路:
调度决策(`process_single_task`/`get_available_slot`)不碰计算,slot 只持有它的指针。
真正消费它的是 `update_slots` 的三段:

| 环节 | 消费点 | 作用 |
|---|---|---|
| pre_decode | `slot.mem.*`(`common_memory` 包装 ctx 的 `llama_memory_i`) | KV 维护:seq_rm/seq_cp/seq_add、state 序列化 |
| decode | `llama_decode(ctx_tgt, batch)`(3681) | 唯一前向计算入口;批内 token 的 KV 按 `(seq_id = slot.id, pos)` 写入 ctx 的 KV 内存,`seq_id = {id_slot}` 由 `server_batch::render` 填入 |
| post_decode | `common_sampler_sample(smpl, ctx_tgt, tok_idx)`(3856) | 从 ctx 的 logits 缓冲取分布采样;detokenize 也用 ctx 的 vocab |

主循环之外还有零星消费:元数据查询(`llama_n_batch/n_ctx_seq`)、embeddings
(`llama_set_embeddings`/`llama_get_embd_seq`)、后端采样开关(`llama_set_sampler`),
以及投机推理的第二个 context `ctx_dft`。

这也是集成的前提:slot 与请求状态不含任何自有计算上下文。paged 模式下仍是同一个
`ctx_tgt` 走 `llama_decode`,变化的只是批由谁构建(`prepare_batch` 替代 server_batch)
与 KV 写到哪里(路由元数据 `llama_paged_batch_info` 由调度器给出,在 `llama_decode`
内部被 ctx 的 paged 内存消费,server 侧无感)。

与本次集成相关的要点:
- `n_parallel` 个 slot 即 `cparams.n_seq_max`(`common/common.cpp:1791`),
  slot 在 `load_model` 中创建(`server-context.cpp:1254-1314`);
- 采样在 `post_decode` 用每槽独立 `common_sampler` 链
  (含语法/结构化输出,`server-context.cpp:3856`);停止规则集中在
  `process_token`(1833-1962:stop 串/antiprompt/EOG/n_predict/t_max/UTF-8 缓冲);
- 所有 KV 操作走 `common_memory` 包装;上下文迁移(`seq_add`)、
  `n_cmpl>1` 状态复制(`seq_cp`)、槽位存取与 RAM 提示缓存(`state_write/read`)
  都依赖经典 KV 语义;
- `server_queue` 通过 `std::function` 回调(`on_new_task/on_update_slots/
  on_sleeping_state`,`server-queue.h:121-137`)与 impl 解耦,是现成的插件接缝;
- server 内没有任何虚基类调度抽象;`common_speculative`(投机推理)以普通辅助结构 +
  钩子函数的方式挂在 pre/decode/post 三段上,是"外挂策略"的先例。

### 2.3 能力映射与冲突分析

| server 依赖 | 功能用途 | paged 路径下的情况 | 处理 |
|---|---|---|---|
| 批构建(server_batch,固定 slot 顺序) | 决定每轮批内进哪些槽、各占多少 token(prefill/decode 混批) | 调度器 `prepare_batch` 全权决定批内容与 seq 集合 | paged 路径不用 server_batch |
| LCP 选槽 + 槽内前缀复用 | 把新请求路由到缓存前缀最长的空闲槽,免重算公共前缀 | 调度器块级前缀缓存取代;槽亲和不再有价值 | paged 路径退化为"任取空闲槽" |
| `seq_rm`(截断旧尾部) | 复用槽时清掉公共前缀之后的陈旧 KV;回滚被拒 draft token | `llama_kv_cache_paged` 有 seq_rm,但调度器自管块生命周期 | paged 路径下绕开 slot.mem,由调度器全权管理 |
| `seq_add`(上下文迁移、n_cache_reuse) | 上下文满时删中段并把后续 KV 整体前移;n_cache_reuse 借它挪缓存块 | no-op(`llama-kv-cache-paged.h:81-83`) | 禁用,`get_can_shift()==false` 时现有代码本就自动降级 |
| `seq_cp`(n_cmpl>1 并行采样) | 把父槽 prompt KV 整段复制到子槽,N 个候选免重复 prefill | 桩实现(`llama-kv-cache-paged.h:73-77`) | 一阶段禁用 |
| `state_write/read`(槽位存取、RAM 提示缓存、checkpoint) | 把序列 KV 序列化到 RAM/文件,腾显存或跨槽、跨重启复用 | no-op | 一阶段禁用 |
| 采样/停止规则/流式文本/工具解析 | 每生成一个 token 后判定输出内容与停止条件 | 全在 KV 层之上,与批来源无关 | 直接复用 |
| 任务队列/defer/取消框架 | 请求排队;无空闲槽时延迟入队;中断在途请求 | 与调度策略无关 | 直接复用 |
| 指标 | TTFT/吞吐等性能统计与喂桶 | `server_metrics` 按 token 计数喂桶,不依赖批结构 | 复用,时间戳取自 `llama_paged_seq_state` |

结论:采样类主链路可以较低成本切换到 paged 调度器;所有依赖经典 KV
写操作(seq_add/seq_cp/state 序列化)的功能在 paged 模式下明确禁用并报可读错误。

## 3. 总体设计

### 3.1 设计原则

1. 默认路径零行为变化:现有 `get_available_slot/launch_slot_with_task/update_slots`
   原样保留,只被一层薄适配器包住;
2. 调度器做成策略对象:新接口 `server_scheduler_interface` + 工厂,paged 实现放独立文件,
   与 `server-context.cpp` 的耦合限定为若干已存在的 impl 成员函数;
3. 每请求状态复用 `server_slot`:采样链、停止规则(`process_token`)、流式文本拼装
   (`find_stopping_strings` 等)零重复实现;
4. 分层排队:server 侧固定 slot 池负责请求级并发上限与溢出 defer(现有机制),
   调度器内部 waiting/抢占负责 KV 块级动态调度。两层职责不重叠。

### 3.2 组件与文件

```
tools/server/
  server-scheduler.h          (新) server_scheduler_interface 接口 + 工厂声明
  server-scheduler-paged.h    (新) paged 实现声明
  server-scheduler-paged.cpp  (新) paged 连续批处理主循环
  server-slot.h               (新) server_slot 结构体(自 server-context.cpp 纯移动)
  server-context.cpp           (改) 工厂接线、任务路由、默认适配器、功能门禁
```

结构:

```
HTTP handlers (server_routes)
        | server_task
        v
server_queue ------> server_context_impl::process_single_task
        ^                  |  推理类任务(COMPLETION/EMBEDDING/RERANK/INFILL)与 CANCEL
        | on_update_slots  v
        +------------- server_scheduler_interface          <-- 工厂: --kv-paged 开关
                            |
              +-------------+------------------+
              v                                v
   server_scheduler_default         server_scheduler_paged
   (默认路径薄适配器,                (llama_paged_scheduler_*:
    包装 get_available_slot /          add_request / prepare_batch /
    launch_slot_with_task /            llama_decode / update)
    update_slots 三段式)                    |
              |                                |
              +----- 共用每请求状态 -----> server_slot (server-slot.h)
                                            采样链 / 停止规则 / 流式文本 / 统计
```

### 3.3 接口定义(草图)

```cpp
// tools/server/server-scheduler.h
#pragma once

#include <memory>
#include <cstdint>

struct server_context_impl;
struct server_task;

// scheduling strategy over the shared llama_context.
// all methods are called from the main thread only, like the rest of server_context_impl
struct server_scheduler_interface {
    virtual ~server_scheduler_interface() = default;

    // called once after load_model()
    virtual bool init() = 0;

    // accept one inference-family task. return false when no capacity:
    // the caller defers the task via queue_tasks.defer(), like today
    virtual bool launch(server_task && task) = 0;

    // abort a queued or running task (CANCEL, client disconnect)
    virtual void cancel(int64_t id_task) = 0;

    // one iteration: build batch, llama_decode, sample, emit results.
    // return true while any request is in flight
    virtual bool update() = 0;
};

// picks default or paged based on params_base.kv_paged
std::unique_ptr<server_scheduler_interface> server_scheduler_create(server_context_impl & impl);
```

说明:

- `launch` 返回 false 对应今天的"无空闲槽 -> defer",溢出排队完全复用现有
  `queue_tasks_deferred` 机制,deferred 任务由槽释放回调
  (`callback_on_release -> pop_deferred_task`,`server-context.cpp:1301-1304`)唤醒;
- 非推理类任务(METRICS/CONTROL/SLOT_*/LoRA/tokenize 等)不经调度器,
  保留在 `process_single_task` 原分支;
- 默认适配器 `server_scheduler_default` 定义在 `server-context.cpp` 内
  (需要访问 impl 私有成员),仅做转发:

```cpp
struct server_scheduler_default final : server_scheduler_interface {
    server_context_impl & impl;

    bool init() override;                 // no-op, 现有路径在 load_model 里已就绪
    bool launch(server_task && t) override {
        server_slot * slot = impl.get_available_slot(t);
        if (!slot) return false;
        impl.launch_slot_with_task(*slot, std::move(t));
        return true;
    }
    void cancel(int64_t id) override;     // 现有 CANCEL 分支逻辑(2454-2462)移入
    bool update() override { impl.update_slots(); return impl.any_slot_processing(); }
};
```

### 3.4 paged 实现要点(`server-scheduler-paged.cpp`)

持有:`llama_paged_scheduler * sched`、`std::vector<server_slot> & slots`(引用 impl 的
slot 池)、`llama_context * ctx`。`server_scheduler_paged` 需要调用 impl 的
`process_token/send_partial_response/send_final_response` 等既有成员,
通过 `server-context.cpp` 内声明的一个内部访问头(或 friend)获得,不再新建抽象层。

launch(task):

1. 取空闲槽(paged 模式不做 LCP 匹配,任取 LRU 空闲槽);
2. 初始化 `slot.smpl`(复用 `launch_slot_with_task` 的采样链构建逻辑,
   `server-context.cpp:1781-1804`),空 prompt / `n_predict == 0` 走 legacy 同款短路;
3. `llama_paged_scheduler_add_request(sched, task.tokens.data(), n, slot.id)`,
   失败(prompt 超长、id 冲突)立即回错误结果并释放槽。

update() 主循环(对齐 `examples/paged/paged.cpp:215-289`):

```
1. // 死锁/活锁检测在库内完成,结论经返回值暴露给调用方
   if (!llama_paged_scheduler_prepare_batch(sched, &batch)) {
       // 返回 false: 死锁/活锁,不可恢复,批内容不可用
       // 所有在途请求按错误收尾,不重试(槽位强制释放见下方 finish 处理)
       fail_all_running();
       return false;
   }
   if (batch.n_tokens == 0) {
       return false;   // 调用成功但没有可运行的 seq: 空闲
   }

2. if (llama_decode(ctx, batch) != 0) {
       // 批内所有 seq 对应槽按错误收尾并停止该请求(无"减半重试"语义)
       fail_batch_seq();
       return false;
   }
   llama_synchronize(ctx);

3. info = get_batch_info(sched)
   逐 seq i:
     request_id = batch.seq_id[info->batch_offsets[i]][0]   // 即 slot.id
     final_prefill = batch_lens[i] > 1
                     && state.n_past + batch_lens[i] == state.n_prompt_total
     if (batch_lens[i] == 1 || final_prefill) {   // 本步产生 logits,需要采样
       common_sampler_sample(slot.smpl, ctx, batch_offsets[i]+batch_lens[i]-1)
       common_sampler_accept(...)
       running = impl.process_token(text, slot)   // 复用: stop 串/EOG/n_predict/t_max/流式
       if (!running) { send_final_response(slot); slot.stopping = true; }
     }
     tokens[i] = 采样结果或 LLAMA_TOKEN_NULL
     stop_flags[i] = (slot.stopping || 槽被 CANCEL 标记)

4. llama_paged_scheduler_update(sched, &batch, tokens, stop_flags)
   // 调度器标记 FINISHED; 下一次 prepare_batch 的 step() 回收并触发 on_finish

5. return true(仍有在途请求)
```

空批的含义:`n_tokens == 0` 指本次调用成功但无可运行的 seq,正常情况等价于调度器内
没有任何在途请求(running/swapped/waiting 全空),`update()` 返回 false 后队列循环挂回
条件变量等待,与默认路径全部槽空闲时的行为一致,不耗 CPU。新到达的请求不会落在空批
分支:launch 与 update 在同一循环迭代内(队列循环先处理任务、后调 update),请求经
`add_request` 进入 waiting 后,running 为空时 `step()` 每步必接纳一个 waiting 请求,
产出该请求的非空 prefill 批。仍在 server defer 队列里的请求不在调度器内,与空批无关;
已入 waiting 但因 KV 块预算未接纳、且 running 为空的情形由库内 `check_deadlock` 拦截
(prepare 返回 false,走致命路径),同样不表现为空批。

finish 处理与槽释放时序(关键细节):

- 停止决定发生时立即 `send_final_response`(与 legacy 一致,客户端无感延迟);
- 但 `slot.release()` 延迟到 `on_finish(request_id)` 回调里执行:请求被 `step()`
  回收之前 `id_to_group` 仍持有该 id,此时同 id 重新 `add_request` 会失败;
  `on_finish` 在主线程 `prepare_batch` 内触发,回调里 release -> `pop_deferred_task`,
  时序安全。最坏延迟一个调度步;
- 调度器侧 `n_past >= n_ctx` 自动结束(服务端未主动 stop)时同样经 `on_finish`:
  映射为 `STOP_TYPE_LIMIT`(上下文满),补发 final response;
- `prepare_batch` 返回 false(死锁)时,除报错外需兜底强制 release 批内槽,
  避免依赖不再会触发的 `on_finish`。

cancel(id_task):

1. 任务还在 server defer 队列:由现有队列机制处理(不经调度器);
2. 已 `add_request`:标记 `slot.stopping`,并调用新增的
   `llama_paged_scheduler_cancel(sched, request_id)`(见 L1):waiting 中的请求立即
   移除并触发 `on_finish`;running/swapped 中的请求保证在下一个 `update()` 步进回收,
   避免"长期换出不上批、stop_flags 够不着"的悬挂请求。

warmup:kv_paged 模式下跳过 server 默认 warmup,改为
`llama_paged_scheduler_warmup`(对齐 `paged.cpp` 的做法),避免默认 warmup 占用/扰动
paged 块池。

指标:沿用 `impl.metrics_*` 喂桶接口;TTFT/e2e 取 `llama_paged_seq_state` 的
`t_arrival_us/t_first_token_us`,`t_prompt_last` 在 final_prefill 步设置,
`n_prompt_cached = n_prompt_total - 入队时剩余 n_prompt`(前缀命中量)。

### 3.5 启用方式与启动校验

不新增调度器参数:复用已有的 `--kv-paged`(L3 解封 server 例子)作为 paged 调度器的
唯一开关。这与默认路径的现有模式一致:内存模式决定调度行为(`kv_unified` 才启用
`try_clear_idle_slots`、`!can_shift` 自动关 ctx_shift 等先例)。开 `--kv-paged` ->
工厂创建 paged 调度器;缺省 -> 默认调度器,行为不变。

不引入独立 `--scheduler` 参数的理由:调度器与 paged KV 在库层硬绑定
(`llama_paged_scheduler_init` 在非 paged ctx 上返回 nullptr),两个参数的有效组合
只有一格,独立开关反而引入"paged KV 走默认调度器"这类劣化且不测试的配置矩阵。
将来若出现第二个调度轴(如二阶段动态准入),再加参数即可:接口与工厂结构不变,
只换工厂判据,向后兼容。

启动校验(fail-fast,给出可读错误与修复建议):

- `--kv-paged` 与 `--kv-unified` 互斥,组合使用启动报错;
- `-b == -ub`、单卡全卸载(库层 `validate_paged_kv_placement` 已有,错误信息透传);
- `n_parallel` 收敛为可用 request_id 上限(见 L2;hybrid 模型 checkpoint 保留段占用
  `n_seq_max` 尾部);
- 功能互斥检查:`--kv-paged` 与 speculative/mtmd/`cache_ram`/`n_cache_reuse`/
  ctx_shift/embeddings/rerank 组合时,启动期直接报错而非静默降级
  (比默认路径"自动关闭"更严格,避免用户误配)。

请求期门禁:EMBEDDING/RERANK/INFILL 任务与
`n_cmpl > 1`、grammar、多模态输入在 paged 模式下回
`server_task_result_error`(HTTP 501 语义,消息注明去掉 `--kv-paged` 即可用默认
调度器),chat 端点第一阶段同样回未支持错误。

## 4. 关键设计决策

| 决策 | 选择与理由 |
|---|---|
| 启用开关 | 复用 `--kv-paged` 单开关,不新增 `--scheduler`:调度器与 paged KV 库层硬绑定,独立参数的有效组合只有一格;将来出现第二个调度轴再加,工厂只换判据 |
| 抽象层级 | 接口只覆盖"launch/cancel/update"三动作,不试图抽象批构建/KV 操作。默认路径包一层即可,paged 路径整体另写,避免为统一而过度设计 |
| 排队模型 | server slot 池(n_parallel,溢出 defer)+ 调度器 waiting/抢占两层。slot 池给请求级背压与取消锚点;块级动态性交给调度器。第二阶段可评估"全量直入调度器 waiting"的 vLLM 式模型 |
| request_id 映射 | request_id = slot.id = llama seq_id,天然落在 `[0, n_seq_max)` 内,与批内 `seq_id` 反查 O(1) |
| 停止条件归属 | 服务端计算 stop_flags(复用 process_token 全部规则),调度器只执行。库层不新增任何停止语义 |
| 采样 | 每槽 `common_sampler` 链与 legacy 完全一致,`init_sampler` 时机等价为 final_prefill 步后首个 token 之前 |
| 槽释放时序 | 停止即发 final response,释放延到 `on_finish`(id 复用安全,主线程安全) |
| KV 生命周期 | paged 模式下 `slot.mem` 系列调用全部绕开;`prompt_clear` 中的 `seq_rm` 在 paged 模式跳过(块释放由调度器 finish 路径负责) |
| 前缀复用 | 不做槽亲和;调度器块级前缀缓存负责跨请求命中,语义更强(块粒度 vs 整段前缀) |
| 失败语义 | `prepare_batch` false 或 decode 失败不做任何重试或自愈:批内请求全部按错误路径收尾,不做默认路径的 n_batch 减半重试(块不足应由调度器抢占规避) |

## 5. 第一阶段功能范围与降级清单

范围(验收):`/completion` 与 `/v1/completions`,含采样参数、流式 SSE、
stop/antiprompt、EOG、`n_predict`、`t_max_predict_ms`、并发与排队、取消、`/metrics`。

paged 模式下禁用清单:

| 功能 | 处理 |
|---|---|
| `/v1/chat/completions`、`/infill` 端点 | 请求期 501 风格错误(第二阶段开启) |
| embeddings / rerank 端点 | 请求期错误(需要 paged 图上池化支持,第二阶段) |
| 工具调用、结构化输出(grammar) | 请求期错误(技术上随采样链可用,但验收放第二阶段) |
| 投机推理(`--spec-*`) | 启动期互斥报错 |
| 多模态(mtmd 输入) | 启动期互斥报错 |
| RAM 提示缓存/`--cache-ram`、`/slots/{save,restore}` | 启动期/请求期错误(需块级 state 序列化,第二阶段) |
| `n_cmpl > 1` 并行采样 | 请求期错误(需 seq_cp 或块引用 fork,第二阶段) |
| 上下文迁移 ctx_shift、`n_cache_reuse` | 启动期关闭并告警(paged 下无 `seq_add`;上下文满即按 `n_ctx` 结束) |
| LoRA | 第一阶段 paged 模式禁用(批内混 LoRA 集合的约束暂不处理) |

`/slots`(只读)、`/props`、`/tokenize` 等非推理端点保留可用。

## 6. 子任务分解与工作量估算

### 6.1 llama 层

| 编号 | 内容 | 文件 | 类型 | 行数 |
|---|---|---|---|---|
| L1 | `llama_paged_scheduler_cancel(sched, request_id)`:waiting 立即移除并 on_finish;running/swapped 标记停止,下一步回收 | `include/llama.h`、`src/llama-paged-scheduler.cpp`、`src/llama-paged-scheduler-impl.{h,cpp}` | 新增 | ~100-120 |
| L2 | `llama_paged_scheduler_max_request_id(sched)`:返回可用 request_id 上限(hybrid 保留段感知) | 同上四文件 | 新增 | ~25-30 |
| L3 | paged 参数解封(`--kv-paged` 即 paged 调度器开关):`-kvp/-ngpub/-ncpub/-kvbls/--kv-paged-watermark` 的 `set_examples` 加 `LLAMA_EXAMPLE_SERVER` | `common/arg.cpp` | 修改 | ~10 |
| L4(可选) | `common_fit_paged_kv_blocks` 允许 server 例子按空闲显存自适应块数 | `common/common.cpp` | 修改 | ~15 |
| L5 | e2e 测试补 cancel 用例(复用现有测试文件,不新建) | `tests/test-paged-kv-e2e.cpp` | 修改 | ~40 |

llama 层合计:约 150-215 行(L4/L5 可选)。

### 6.2 server 层

| 编号 | 内容 | 文件 | 类型 | 行数 |
|---|---|---|---|---|
| S1 | `server_scheduler_interface` 接口 + 工厂 | `tools/server/server-scheduler.h` | 新增 | ~80-110 |
| S2 | 抽取 `server_slot`(含 `slot_state` 枚举与内联方法)到独立头;`server_batch` 留在原处 | `tools/server/server-slot.h`(新)、`server-context.cpp` | 移动 | ~500(纯移动,零语义变化) |
| S3 | paged 调度器实现:init/warmup/on_finish、launch、cancel、update 主循环、finish 与错误路径、指标 | `tools/server/server-scheduler-paged.{h,cpp}` | 新增 | ~650-850 |
| S4 | 工厂接线与任务路由(`process_single_task` 推理类/CANCEL 分支改经调度器)、默认适配器、启动校验与功能门禁、warmup 分支、`prompt_clear` 的 paged 分支 | `tools/server/server-context.cpp` | 修改+新增 | ~230-300 |
| S5 | 用户文档:`--kv-paged` 在 server 的语义、第一阶段限制、降级清单 | `tools/server/README.md` | 修改 | ~30-40(文档行) |

server 层合计:新增约 975-1300 行(其中 S2 为 ~500 行纯移动,新逻辑约 475-800),
修改约 240-340 行。

### 6.3 汇总

- 去除纯移动部分,预计净新增/修改约 870-1170 行(llama 层 ~150-215,server 层 ~725-970);
- S2 + S4 是仅有的两处触碰现有代码的任务,其余全部为新增文件;
- 建议实施顺序:L3 -> L2 -> S2 -> S1 -> S4(默认路径回归)-> L1 -> S3 -> L4/L5 -> S5。
  其中 S2 合入后先跑全量 server 回归,确认纯移动无回归再继续。

## 7. 第二阶段扩展预览

| 功能 | 预估路径 |
|---|---|
| chat/工具调用 | HTTP 层已就绪(`task_result_state` 流式解析),开启端点验收即可,调度器零改动 |
| 结构化输出(grammar) | 已随每槽采样链存在,补验收测试即可 |
| embedding/rerank | 需验证 `llama_set_embeddings` + 池化在 paged 图上的支持(llama 层) |
| `n_cmpl>1` 并行采样 | 基于块引用共享实现调度器侧 fork(复制 block_table 引用),替代 seq_cp |
| 槽位存取/RAM 提示缓存 | paged 内存的块级 `state_write/read` 序列化 |
| 投机推理 | 需调度器支持每步多 token 反馈与 draft 上下文 paged 化,复杂度最高,需专项设计 |
| 多模态 | mtmd 块编码发生在 prefill 中途,与"整段 prompt 一次性入队"冲突,需分段入队 API |
| 动态准入 | 评估去掉 server slot 池、请求全量入调度器 waiting 的 vLLM 式模型(需 request 注册表与取消重构) |

## 8. 风险与开放问题

1. prefill 独占步:长 prompt 分块期间,在跑请求的 decode 被推迟,TPOT 出现毛刺;
   每步 token 数受 `n_batch` 预算约束,毛刺上界可控,但与默认路径 join-batching
   的行为差异需要在验收时量化(`scripts/perf-paged-vs-unified.py` 可复用)。
2. 抢占/重算毛刺:KV 块满时换出或重算带来尾延迟;`--kv-paged-watermark`(默认 0.05)
   与 CPU 块数是主要调参项。
3. 空批衔接:正常情况"批空 <=> 无在途请求"(含义见 3.4),返回 false 挂起等待即可。
   风险在反常情况:在途请求非空却产出空批,属 `check_deadlock`/`check_livelock`
   两道检测缝隙里的理论残留(实现 bug、边界竞争)。此时 `update()` 因在途非空返回
   true,循环立即重投递再调 `update()` 又得空批,形成无 sleep 的忙轮询(烧满一核但
   不做任何 decode)。兜底:连续 N 次空批且在途非空时记告警,必要时把受影响请求按
   错误收尾。
4. `on_finish` 依赖:`prepare_batch` 返回 false(死锁)后回调不再触发,必须走强制
   释放兜底(见 3.4);另一路径是库层提供显式 reap,作为备选。
5. hybrid 模型 checkpoint 保留段使可用 slot 数少于 `n_parallel`,启动校验需以 L2
   的返回值为准裁剪槽池。
6. `n_ctx` 语义变化(总预算 -> 单请求上限):`--kv-unified-per-slot` 等参数互斥报错,
   文档明确说明;上下文满即结束(无 ctx_shift),依赖超长上下文的客户端会观察到行为差异。
7. 空 prompt / `n_predict == 0` / 请求 id 复用窗口(停止到 on_finish 之间)等边界,
   已在 3.4 设计,实现时需要专项用例覆盖。
8. 默认路径回归面:S4 虽小,但 `process_single_task` 分支重构影响所有任务类型,
   S2 的纯移动需要全量 server 测试佐证(见 9)。

## 9. 验证计划

- 库层:复用 `tests/test-paged-kv.cpp` / `test-paged-kv-e2e.cpp`,L1 落地时补 cancel
  用例;不新增测试文件(遵循仓库贡献约定)。
- 默认路径回归:S2/S4 合入后跑现有 server 全量测试,确认行为不变。
- server 手测清单(`--kv-paged`):
  - `/completion` 非流式 + 流式 SSE;`/v1/completions`;
  - 采样参数(smoothing/temperature/penalties)、stop 串、antiprompt、`n_predict`、EOG;
  - 并发压到 2x 槽位:超额排队与释放后续跑;
  - 取消:流式中断(client disconnect)与 CANCEL 任务,确认槽位与 KV 块回收
    (`/metrics` 的 `n_cache_tokens` 类指标);
  - 前缀缓存命中:相同前缀二次请求 TTFT 下降;
  - 降级端点逐一确认返回明确错误;
  - 启动校验:`-b != -ub`、多卡分布、`--kv-paged` 与 `--kv-unified` 互斥、
    hybrid 保留段等组合的报错信息。
- 性能:复用 `scripts/perf-paged-vs-unified.py` 思路,补 server 场景
  (并发 10 请求,对比 default 与 paged 的 TTFT/TPOT)。

## 10. 实施记录(第一阶段)

实际改动(`git diff`,不含设计文档与 README 表格微调):

| 层 | 文件 | 改动 |
|---|---|---|
| llama | `include/llama.h` | +11:cancel / max_request_id 声明 |
| llama | `src/llama-paged-scheduler.cpp`、`-impl.{h,cpp}` | +74:cancel(running/swapped/waiting 同步回收)、max_request_id、queue_request 重复 id 拒绝 |
| llama | `src/llama-kv-cache-paged.h`、`llama-paged-prefix-cache.h` | +3:n_seq_max / ckpt_slot_base 访问器 |
| llama | `common/arg.cpp` | paged 参数解封 LLAMA_EXAMPLE_SERVER |
| llama | `tests/test-paged-kv-e2e.cpp` | +109:cancel 用例(waiting/running 取消、未知 id、id 复用、max_request_id) |
| server | `tools/server/server-slot.h`(新) | 648 行纯移动(slot_state + server_batch + server_slot) |
| server | `tools/server/server-scheduler.h`(新) | 79 行:宿主接口 + 调度器接口 + 工厂声明 |
| server | `tools/server/server-scheduler-paged.{h,cpp}`(新) | 415 行:launch/cancel/update/init/on_finish/错误路径/指标 |
| server | `tools/server/server-context.cpp` | +113 / -641(删除部分为纯移动):启动门禁、任务路由、宿主实现块、warmup 跳过 |
| server | `tools/server/CMakeLists.txt`、`README.md` | 构建列表、参数表与 Paged KV cache 章节 |

与设计文档的偏差(均为实现期确认后的改进):

1. `server_batch` 随 `server_slot` 一并移入 `server-slot.h`:`handle_last_sampled_token(server_batch&)` 依赖完整定义,不移动则纯移动不成立;仍为零语义变化。
2. 未引入 `server_scheduler_default` 适配器:默认路径的推理任务分支含 `n_cmpl>1` 父子槽与 `cache_idle_slots` 保存逻辑,搬进适配器会放大回归面,与"默认路径零行为变化"冲突。改为 `process_single_task` 内一个 `if (scheduler)` 分支,默认路径原样保留;接口与工厂仅服务 paged 路径。
3. 宿主访问采用 `server_scheduler_host` 抽象基类(impl 继承实现),替代文中"内部访问头或 friend";原有同名方法(process_token/send_*/metrics_*)直接满足接口。
4. `update()` 返回 void:实际队列循环的保活机制是重投递 NEXT_RESPONSE(默认路径同款),接口返回"是否在途"无处消费;空批 + 在途非空的忙轮询用"连续 64 步兜底报错"实现。
5. cancel 对三种状态(waiting/running/swapped)一律同步回收:`free_blocks` 同时释放 GPU 与 CPU 池块,换出中的请求无需先 swap-in,彻底消除"悬挂请求";`finish()` 触发 on_finish,时序与文中一致。
6. `queue_request` 增加重复 request_id 拒绝(槽位复用窗口与 cancel 幂等性依赖它干净失败)。
7. `--kv-paged` + `--kv-unified` 改为强制关闭并告警(而非启动报错):kv_unified 在 paged 路径本就是 no-op,硬报错会使默认 `-np auto`(隐含 kv_unified)与 `--kv-paged` 无法组合。
8. L4 无需改码:`fit_params` 默认开启,`common_init_from_params` 已对 kv_paged 自动适配块数。
9. paged warmup 失败降级为告警(默认 warmup 失败同样非致命)。

## 11. 调试记录(910B4 + CANN 9.0.0)

编译:openEuler 22.03 aarch64、CANN 9.0.0(ops + nnal 9.0.0)、GCC 10.3,`-DGGML_CANN=on`
下 llama-server 与 test-paged-kv-e2e 构建通过。

| 问题 | 结论 |
|---|---|
| 服务器缺 aclnn 算子头文件 | 补装 `Ascend-cann-910b-ops_9.0.0`;nnal 原为 8.5.0,与 toolkit 9.0.0 错配,已换装 9.0.0 |
| ggml-cann.cpp 910B 编译失败 | 上游 bug:`transform_kq` 定义在 `#ifdef ASCEND_310P` 内,`buffer_set_tensor` 的调用点未加保护。已按文件既有模式把 `if (is_kq)` 块包进 `ASCEND_310P`(910B 下该块运行时本就不可达) |
| server_tokens 拷贝赋值 | `server_tokens` 有意禁止拷贝(留有 TODO),paged launch 改用 `get_text_tokens()` 构造后 move 赋值 |

运行时:非 paged 路径(NPU)正常出 token;**paged 路径在 910B4 上被 ATB PagedAttention
内核阻塞**:`op->Setup` 返回 error 8(ERROR_INVALID_TENSOR_DIM),ATB 日志明确报
`headSize of keyCache and query should be same`。本仓库的 CANN paged-attn wrapper
按 310P ATB 编写(KV 池以 FRACTAL_NZ 解释,`K_CACHE_V_CACHE_NZ` 写入路径,310P 的
SelfAttention/PA 支持 NZ cache),910B 的 ATB 9.0.0 PagedAttention 对 NZ cache 的
维度解释不同,直接复用会触发维度校验失败。head_dim=80(Qwen3-4B)非 2 的幂亦需复核
310P 约束。**结论:paged attention 内核的 910B 适配是独立的内核层工作**(评估方向:
改喂 ND 布局 cache 或改用 910B 对应的 ATB/原生算子),不在本次调度器集成范围内;
LLAMA_PAGED 前缀的调度器与 server 集成代码与设备无关,内核打通后即可复验。
