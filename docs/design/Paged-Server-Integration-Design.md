# Paged Scheduler 对接 llama-server 集成设计

状态:第一阶段与第二阶段已实现,含多模态(9.4);310P 功能验收已完成,见 9.6;

日期:2026-10-25(最近刷新:2026-11-03,对齐优先级队列适配与多模态增量)

## 1. 背景与目标

本仓库已在 llama 层实现完整的 paged KV cache 与 vLLM 风格连续批处理调度器(`llama_paged_scheduler_*`,见 `src/llama-paged-scheduler-impl.cpp`),并配有两个块级前缀缓存与 hybrid GDN checkpoint 的设计文档(`CANN-Hybrid-PagedAttention-Design.md`, `CANN-Hybrid-PagedAttention-PrefixCache-Design.md`)。

本文给出 paged_scheduler 对接 server 的集成方案:

- 侵入式修改尽量少:通过策略接口 + 工厂把调度器做成可插拔组件,默认路径行为不变;
- 做合理的抽象:接口边界与 server 现有解耦点(queue 回调、server_slot)对齐;
- 第一阶段只提供采样的基础功能; 第二阶段提供工具调用、投机推理、结构化输出、多模态等拓展功能; 遗留embedding、reranker 等拓展功能未设计。

## 2. 现状盘点

### 2.1 paged 调度器(llama 层)

公共 C API(`include/llama.h:1669-1836`):

| API | 作用 |
|---|---|
| `llama_paged_scheduler_init(ctx)` | 从 ctx 取出 paged KV 内存,初始化调度器;非 paged ctx 返回 nullptr。模式由 `llama_context_params.paged_priority`(`--enable-priority`)在此时固定:FCFS 或按请求优先级 |
| `llama_paged_scheduler_warmup(sched, ctx)` | 一次小批 + 一次图尺寸的合成 prefill/decode,预热后端与 ATB 内核 |
| `llama_paged_scheduler_add_request(sched, tokens, n, request_id, priority)` | 入队一个请求;request_id 即 llama seq_id。priority 数小者先服务(默认 0),仅在 paged_priority 模式生效,钳制到 `[-LLAMA_PAGED_PRIORITY_LIMIT, LLAMA_PAGED_PRIORITY_LIMIT]` |
| `llama_paged_scheduler_add_request_segs(sched, segs, n_segs, request_id)` | 多模态分段入队(9.4):文本段带 token id,媒体段可 `embd = NULL` 只声明行数 |
| `llama_paged_scheduler_append_segment(sched, request_id, seg)` | 补供媒体段的编码输出,请求从暂停态恢复 |
| `llama_paged_scheduler_prepare_batch(sched, batch)` | 产出一个 `llama_batch`(调度器持有并复用);false 表示死锁/活锁,服务端对这类失败不做任何重试或自愈,直接把受影响的请求按错误路径收尾,而不是像默认调度路径那样尝试恢复 |
| `llama_paged_scheduler_update(sched, batch, feedback, n_feedback)` | decode 后回填:每 seq 一条 `llama_paged_feedback`(request_id/tokens/n_tokens/stop),批内推进加批外记账 |
| `llama_paged_scheduler_fork_request(sched, src_id, dst_id)` | `n_cmpl > 1`:满块引用共享 + 尾块急拷贝 fork 已预填的请求(9.2) |
| `llama_paged_scheduler_attach_draft(sched, id, draft, n_draft)` | 投机推理:下一步扩为 1 + n_draft 行(9.3) |
| `llama_paged_scheduler_truncate(sched, id, n_past_new)` | 投机部分接受时位置级回滚(9.3) |
| `llama_paged_scheduler_get_seq_state(sched, id, out)` | 查询 `n_prompt/n_prompt_total/n_decoded/n_past/paused/priority/t_arrival_us/t_first_token_us/status` |
| `llama_paged_scheduler_get_batch_info(sched)` | 当前批的路由元数据(`n_seq/batch_offsets/batch_lens/...`),下次 prepare 前有效 |
| `llama_paged_scheduler_set_on_finish(sched, cb, ud)` | 请求被回收时回调(主线程,prepare_batch 内部触发) |
| `llama_paged_scheduler_cancel(sched, request_id)` | 取消排队/在途请求:waiting/running/swapped 一律同步回收并触发 on_finish |
| `llama_paged_scheduler_max_request_id(sched)` | 可用 request_id 上限(hybrid 保留段感知) |

调用契约(以 `examples/paged/paged.cpp` 为准;`update` 的反馈契约见 9.0,为 feedback 数组形式):

- 调用方自己采样,调度器不做任何采样;每 seq 一条 `llama_paged_feedback`,批内 seq 先按批行数推进 `n_past`,再以 feedback 的 token 列表为准推进,条目 token 数少于批行数即为隐式回滚,批外条目仅做 token 记账(fork 子请求);
- 仅在 decode 步(`batch_lens[i] == 1`)或 prefill 末块(`batch_lens[i] > 1 && state.n_past + batch_lens[i] == n_prompt_total`)产生 logits并采样,采样下标为 `batch_offsets[i] + batch_lens[i] - 1`;
- feedback 的 `stop` 字段非 0 或 `n_past >= n_ctx` 时请求标记 FINISHED,下一次`prepare_batch` 的 `step()` 内回收并触发 `on_finish`;
- prefill 独占步:任一候选处于 prefill 时整批只保留该请求(`llama-paged-scheduler-impl.cpp:90-107`),prefill 与 decode 不混批;
- 内部三级队列 running/swapped/waiting:waiting/swapped 队列在 init 时按模式二选一(`src/llama-paged-queue.h`):FCFS 按到达序(deque + 按键插入),优先级模式为`group_precedes`(priority, t_arrival, seq)比较器的最小堆;两种实现均支持`take(request_id)` 定点摘除(cancel 依赖)。KV 块不足时先驱逐缓存块,再抢占运行请求(换出到 CPU 块池,CPU 池满则释放重算;牺牲者选择 FCFS 取最新到达、优先级模式取最低优先级,并带步内优先级反转保护);前缀缓存(链式哈希 + 块共享,hybrid 模型另配 GDN checkpoint 槽)在入队时自动命中;媒体组(has_media)不参与前缀缓存与重算,只能换出(9.4)。

约束(第一阶段,`examples/paged/README.md:22-28`):

- 必须单卡全量卸载(`validate_paged_kv_placement`,`src/llama-context.cpp:87-111`);
- 纯注意力路径要求 `n_batch == n_ubatch`(`src/llama-paged-scheduler.cpp:42-44`);
- 不支持 SWA;hybrid paged 目前仅 `LLM_ARCH_QWEN35/QWEN35MOE`;
- `request_id` 必须小于 `n_seq_max`,且 hybrid 模型要避开 checkpoint 保留段(`llama-paged-scheduler-impl.cpp:29-38,137-141`);
- 单请求 prompt 长度必须小于 `n_ctx`(逐请求上限,而非总预算)。

### 2.2 server 现状

文件布局:本 fork 已将 server 拆成约 13 个文件,核心是 `tools/server/server-context.cpp`(约 5558 行)+ `server-context.h`(pimpl,`server_context` 公开接口只暴露`load_model/start_loop/terminate/get_meta` 等)。

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

`llama_context`(全局唯一 `ctx_tgt`)在图中不可见,因为它不属于任务/槽链路:调度决策(`process_single_task`/`get_available_slot`)不碰计算,slot 只持有它的指针。真正消费它的是 `update_slots` 的三段:

| 环节 | 消费点 | 作用 |
|---|---|---|
| pre_decode | `slot.mem.*`(`common_memory` 包装 ctx 的 `llama_memory_i`) | KV 维护:seq_rm/seq_cp/seq_add、state 序列化 |
| decode | `llama_decode(ctx_tgt, batch)`(3681) | 唯一前向计算入口;批内 token 的 KV 按 `(seq_id = slot.id, pos)` 写入 ctx 的 KV 内存,`seq_id = {id_slot}` 由 `server_batch::render` 填入 |
| post_decode | `common_sampler_sample(smpl, ctx_tgt, tok_idx)`(3856) | 从 ctx 的 logits 缓冲取分布采样;detokenize 也用 ctx 的 vocab |

主循环之外还有零星消费:元数据查询(`llama_n_batch/n_ctx_seq`)、embeddings(`llama_set_embeddings`/`llama_get_embd_seq`)、后端采样开关(`llama_set_sampler`),以及投机推理的第二个 context `ctx_dft`。

这也是集成的前提:slot 与请求状态不含任何自有计算上下文。paged 模式下仍是同一个`ctx_tgt` 走 `llama_decode`,变化的只是批由谁构建(`prepare_batch` 替代 server_batch)与 KV 写到哪里(路由元数据 `llama_paged_batch_info` 由调度器给出,在 `llama_decode`内部被 ctx 的 paged 内存消费,server 侧无感)。

与本次集成相关的要点:
- `n_parallel` 个 slot 即 `cparams.n_seq_max`(`common/common.cpp:1791`),
  slot 在 `load_model` 中创建(`server-context.cpp:1254-1314`);
- 采样在 `post_decode` 用每槽独立 `common_sampler` 链(含语法/结构化输出,`server-context.cpp:3856`);停止规则集中在`process_token`(1833-1962:stop 串/antiprompt/EOG/n_predict/t_max/UTF-8 缓冲);
- 所有 KV 操作走 `common_memory` 包装;上下文迁移(`seq_add`)、`n_cmpl>1` 状态复制(`seq_cp`)、槽位存取与 RAM 提示缓存(`state_write/read`)都依赖经典 KV 语义;
- `server_queue` 通过 `std::function` 回调(`on_new_task/on_update_slots/on_sleeping_state`,`server-queue.h:121-137`)与 impl 解耦,是现成的插件接缝;
- server 内没有任何虚基类调度抽象;`common_speculative`(投机推理)以普通辅助结构 + 钩子函数的方式挂在 pre/decode/post 三段上,是"外挂策略"的先例。

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

1. 默认路径零行为变化:现有 `get_available_slot/launch_slot_with_task/update_slots` 原样内联保留,不包适配器;paged 调度器经 `if (scheduler)` 分支旁路接入(取舍见 3.2);
2. 调度器做成策略对象:新接口 `server_scheduler_interface` + 工厂,paged 实现放独立文件, 与 `server-context.cpp` 的耦合限定为若干已存在的 impl 成员函数;
3. 每请求状态复用 `server_slot`:采样链、停止规则(`process_token`)、流式文本拼装(`find_stopping_strings` 等)零重复实现;
4. 分层排队:server 侧固定 slot 池负责请求级并发上限与溢出 defer(现有机制),调度器内部 waiting/抢占负责 KV 块级动态调度。两层职责不重叠。

### 3.2 组件与文件

```
tools/server/
  server-scheduler.h          (新) server_context_port 接口(context 暴露给调度器的 port)+ server_scheduler_interface
                                调度器接口 + 工厂声明
  server-scheduler-paged.h    (新) paged 实现声明
  server-scheduler-paged.cpp  (新) paged 连续批处理主循环:launch/cancel/update、多模态分段
                                与编码预取、投机起草与验证、n_cmpl fork
  server-slot.h               (新) server_slot + server_batch(自 server-context.cpp 纯移动)
  server-context.cpp           (改) impl 继承实现 server_context_port、`if (scheduler)`
                                任务路由、启动门禁、paged 模式跳过默认 warmup
```

设计取舍:不设默认适配器。默认路径的推理任务分支含 `n_cmpl>1` 父子槽派生与
`cache_idle_slots` 保存逻辑,包一层适配器就要把这两段搬进新文件,回归面覆盖全部任务
类型,换来的只是形式上的策略统一;`if (scheduler)` 分支旁路即可达成可插拔,默认路径
零移动。因此调度器接口只服务 paged 路径,工厂缺省返回 nullptr;paged 实现对 impl 的
访问经 `server_context_port` 抽象基类(`server_context_impl` 继承实现)取得,
不新建 friend 或内部头。

结构:

```
HTTP handlers (server_routes)
        | server_task
        v
server_queue ------> server_context_impl::process_single_task
        ^                  |  推理类任务(COMPLETION 等)与 CANCEL:
        |                  |    if (scheduler) -> scheduler->launch / cancel
        |                  |    else           -> 内联默认分支原样保留
        | on_update_slots  |                     (get_available_slot / launch_slot_with_task)
        +-------------- update 钩子: if (scheduler) scheduler->update();
                                    else                update_slots();
                            |
                            v
              server_scheduler_interface
                            |      <-- 工厂 server_scheduler_create(port):
                            v          --kv-paged -> server_scheduler_paged
                   server_scheduler_paged   缺省 -> nullptr(默认路径不进调度器)
                    (llama_paged_scheduler_*:
                     add_request(_segs) / prepare_batch /
                     llama_decode / update)
                            |
                            |  server_context 侧服务经 server_context_port 暴露(server_context_impl 继承实现):
                            |    launch_slot_with_task / process_token / send_* / metrics_* /
                            |    spec_ctx / mmctx / yield / post_next_response
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

    // one iteration: build batch, llama_decode, sample, emit results
    virtual void update() = 0;
};

// returns the paged scheduler when params.kv_paged is set, nullptr otherwise
// (the default path stays inline in server-context.cpp)
std::unique_ptr<server_scheduler_interface> server_scheduler_create(server_context_port & port);
```

说明:

- `launch` 返回 false 对应今天的"无空闲槽 -> defer",溢出排队完全复用现有`queue_tasks_deferred` 机制,deferred 任务由槽释放回调(`callback_on_release -> pop_deferred_task`,`server-context.cpp:1301-1304`)唤醒;
- 非推理类任务(METRICS/CONTROL/SLOT_*/LoRA/tokenize 等)不经调度器,保留在 `process_single_task` 原分支;
- 默认路径不进调度器接口:`process_single_task` 的推理任务与 CANCEL 分支、update 钩子均为 `if (scheduler)` 双分支,默认路径原样内联(取舍见 3.2);
- paged 实现对 impl 的访问经 `server_context_port` 抽象基类(`server_context_impl` 继承实现,提供 `launch_slot_with_task/process_token/send_*/metrics_*/spec_ctx/mmctx/yield/post_next_response`):impl 原有同名方法直接满足接口,零包装成本;实际接口与服务面见 `tools/server/server-scheduler.h`。

### 3.4 paged 实现要点(`server-scheduler-paged.cpp`)

持有:`llama_paged_scheduler * sched`、`std::vector<server_slot> & slots`(引用 impl 的
slot 池)、`llama_context * ctx`。`server_scheduler_paged` 需要调用 impl 的
`process_token/send_partial_response/send_final_response` 等既有成员,
经 `server_context_port` 接口(`server_context_impl` 继承实现)获得,不再新建抽象层;
实际类还含多模态编码/预取与投机、fork 路径,见 9.3/9.4。

launch(task):

1. 取空闲槽(paged 模式不做 LCP 匹配,任取 LRU 空闲槽);
2. 初始化 `slot.smpl`(复用 `launch_slot_with_task` 的采样链构建逻辑,`server-context.cpp:1781-1804`),空 prompt / `n_predict == 0` 走 legacy 同款短路;
3. `llama_paged_scheduler_add_request(sched, task.tokens.data(), n, slot.id)`,失败(prompt 超长、id 冲突)立即回错误结果并释放槽。

update() 主循环(对齐 `examples/paged/paged.cpp`;反馈契约见 9.0,`update()` 返回 void,
保活由 port 侧重投递 NEXT_RESPONSE 承担(经 `post_next_response`),同默认路径):

```
1. // 死锁/活锁检测在库内完成,结论经返回值暴露给调用方
   if (!llama_paged_scheduler_prepare_batch(sched, &batch)) {
       // 返回 false: 死锁/活锁,不可恢复,批内容不可用
       // 所有在途请求按错误收尾,不重试(槽位强制释放见下方 finish 处理)
       fail_all_running();
       return;
   }
   if (batch.n_tokens == 0) {
       return;   // 调用成功但没有可运行的 seq: 空闲(含义见下文)
   }

2. if (llama_decode(ctx, batch) != 0) {
       // 批内所有 seq 对应槽按错误收尾并停止该请求(无"减半重试"语义)
       fail_batch_seq();
       return;
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
       feedback.push({request_id, &sampled, 1, slot.stopping ? 1 : 0})
     }

4. llama_paged_scheduler_update(sched, &batch, feedback.data(), feedback.size())
   // 调度器标记 FINISHED; 下一次 prepare_batch 的 step() 回收并触发 on_finish
```

空批的含义:`n_tokens == 0` 指本次调用成功但无可运行的 seq,正常情况等价于调度器内没有任何在途请求(running/swapped/waiting 全空),`update()` 返回后无在途请求,队列循环挂回条件变量等待,与默认路径全部槽空闲时的行为一致,不耗 CPU。新到达的请求不会落在空批分支:launch 与 update 在同一循环迭代内(队列循环先处理任务、后调 update),请求经`add_request` 进入 waiting 后,running 为空时 `step()` 每步必接纳一个 waiting 请求,产出该请求的非空 prefill 批。仍在 server defer 队列里的请求不在调度器内,与空批无关;已入 waiting 但因 KV 块预算未接纳、且 running 为空的情形由库内 `check_deadlock` 拦截(prepare 返回 false,走致命路径),同样不表现为空批。反常的"在途非空却连续空批"由连续 64 步兜底报错收尾。

finish 处理与槽释放时序(关键细节):

- 停止决定发生时立即 `send_final_response`(与 legacy 一致,客户端无感延迟);
- 但 `slot.release()` 延迟到 `on_finish(request_id)` 回调里执行:请求被 `step()`回收之前 `id_to_group` 仍持有该 id,此时同 id 重新 `add_request` 会失败;`on_finish` 在主线程 `prepare_batch` 内触发,回调里 release -> `pop_deferred_task`,时序安全。最坏延迟一个调度步;
- 调度器侧 `n_past >= n_ctx` 自动结束(服务端未主动 stop)时同样经 `on_finish`: 映射为 `STOP_TYPE_LIMIT`(上下文满),补发 final response;
- `prepare_batch` 返回 false(死锁)时,除报错外需兜底强制 release 批内槽,避免依赖不再会触发的 `on_finish`。

cancel(id_task):

1. 任务还在 server defer 队列:由现有队列机制处理(不经调度器);
2. 已 `add_request`:标记 `slot.stopping`,并调用 `llama_paged_scheduler_cancel(sched, request_id)`:waiting/running/swapped 一律同步回收——`free_blocks` 同时释放 GPU 与 CPU 池块,换出中的请求无需先 swap-in,`finish()` 触发 `on_finish`,彻底消除"长期换出不上批、stop 够不着"的悬挂请求;未知 id(已结束或未入队)干净返回 false。

warmup:kv_paged 模式下跳过 server 默认 warmup,改为`llama_paged_scheduler_warmup`(对齐 `paged.cpp` 的做法),避免默认 warmup 占用/扰动paged 块池。

指标:沿用 `impl.metrics_*` 喂桶接口;TTFT/e2e 取 `llama_paged_seq_state` 的`t_arrival_us/t_first_token_us`,`t_prompt_last` 在 final_prefill 步设置,`n_prompt_cached = n_prompt_total - 入队时剩余 n_prompt`(前缀命中量)。

### 3.5 启用方式与启动校验

不新增调度器参数:复用已有的 `--kv-paged`(L3 解封 server 例子)作为 paged 调度器的唯一开关。这与默认路径的现有模式一致:内存模式决定调度行为(`kv_unified` 才启用`try_clear_idle_slots`、`!can_shift` 自动关 ctx_shift 等先例)。开 `--kv-paged` ->工厂创建 paged 调度器;缺省 -> 工厂返回 nullptr,默认路径内联原样,行为不变。

启动校验(fail-fast,给出可读错误与修复建议):

- `--kv-paged` 与 `--kv-unified` 组合:强制关闭 kv_unified 并告警,而非报错。理由:kv_unified 在 paged 路径本就是 no-op,硬报错会挡住 `-np auto`(隐含 kv_unified)与 `--kv-paged` 的默认组合;
- `-b == -ub`、单卡全卸载(库层 `validate_paged_kv_placement` 已有,错误信息透传);
- `n_parallel` 收敛为可用 request_id 上限(`max_request_id`;hybrid 模型 checkpoint 保留段占用`n_seq_max` 尾部);
- `--cache-ram`/embeddings/rerank 与 `--kv-paged` 组合:启动期直接报错而非静默降级(比默认路径"自动关闭"更严格,避免用户误配);synthetic speculative acceptance 同样启动期拒绝(部分接受回滚走 truncate,无 replay 语义);
- speculative 与多模态输入放行(9.3/9.4:draft ctx 强制经典 KV;媒体分段预填);`n_cache_reuse` 与 ctx_shift 告警关闭(paged 下无 `seq_add`,上下文满即按 `n_ctx` 结束)。

请求期门禁:EMBEDDING/RERANK/INFILL 任务在 paged 模式下回`server_task_result_error`(HTTP 501 语义,消息注明去掉 `--kv-paged` 即可用默认调度器);chat、grammar、`n_cmpl > 1` 与多模态输入均已放行(9.1/9.2/9.4)。

## 4. 关键设计决策

| 决策 | 选择与理由 |
|---|---|
| 启用开关 | 复用 `--kv-paged` 单开关,不新增 `--scheduler`:调度器与 paged KV 库层硬绑定,独立参数的有效组合只有一格;将来出现第二个调度轴再加,工厂只换判据 |
| 抽象层级 | 接口只覆盖"launch/cancel/update"三动作,不试图抽象批构建/KV 操作。默认路径保持内联,paged 路径整体另写,避免为统一而过度设计 |
| 排队模型 | server slot 池(n_parallel,溢出 defer)+ 调度器 waiting/抢占两层。slot 池给请求级背压与取消锚点;块级动态性交给调度器。第二阶段可评估"全量直入调度器 waiting"的 vLLM 式模型 |
| request_id 映射 | request_id = slot.id = llama seq_id,天然落在 `[0, n_seq_max)` 内,与批内 `seq_id` 反查 O(1) |
| 停止条件归属 | 服务端计算 stop_flags(复用 process_token 全部规则),调度器只执行。库层不新增任何停止语义 |
| 采样 | 每槽 `common_sampler` 链与 legacy 完全一致,`init_sampler` 时机等价为 final_prefill 步后首个 token 之前 |
| 槽释放时序 | 停止即发 final response,释放延到 `on_finish`(id 复用安全,主线程安全) |
| KV 生命周期 | paged 模式下 `slot.mem` 系列调用全部绕开;`prompt_clear` 中的 `seq_rm` 在 paged 模式跳过(块释放由调度器 finish 路径负责) |
| 前缀复用 | 不做槽亲和;调度器块级前缀缓存负责跨请求命中,语义更强(块粒度 vs 整段前缀) |
| 失败语义 | `prepare_batch` false 或 decode 失败不做任何重试或自愈:批内请求全部按错误路径收尾,不做默认路径的 n_batch 减半重试(块不足应由调度器抢占规避) |

## 5. 功能范围与降级清单(当前状态)

第一阶段验收范围曾是 `/completion` 与 `/v1/completions` 基础链路;第二阶段(含多模态)落地后,paged 模式当前支持:

- `/completion`、`/v1/completions`、`/v1/chat/completions`(含工具调用、grammar、reasoning 解析、`logprobs`),采样参数、流式 SSE、stop/antiprompt、EOG、`n_predict`;
- 并发与排队、取消、`/metrics`;
- `n_cmpl > 1` 并行采样(块引用 fork,9.2);
- 投机推理 `--spec-*`(draft ctx 保持经典 KV;synthetic acceptance 在 paged 下启动期拒绝;媒体请求跳过起草,9.3/9.4);
- 多模态输入(分段预填 + 暂停边界按需编码,9.4);
- 请求优先级调度(`--enable-priority` 打开调度器优先级模式;server 侧暂未接每请求优先级,见 9.7)。

paged 模式下禁用清单:

| 功能 | 处理 |
|---|---|
| `/infill` 端点 | 请求期 501 风格错误 |
| embeddings / rerank 端点 | 启动期报错(需要 paged 图上池化支持,未设计) |
| synthetic speculative acceptance | 启动期拒绝(部分接受回滚走 truncate,无 replay) |
| RAM 提示缓存/`--cache-ram`、`/slots/{save,restore}` | 启动期/请求期错误(需块级 state 序列化,未设计) |
| 上下文迁移 ctx_shift、`n_cache_reuse` | 启动期关闭并告警(paged 下无 `seq_add`;上下文满即按 `n_ctx` 结束) |
| LoRA | paged 模式禁用(批内混 LoRA 集合的约束暂不处理) |
| 显式 `id_slot` 请求 | 请求期错误(槽亲和无意义,任取空闲槽) |
| SWA 架构(gemma3、llama4 等) | 不支持;hybrid 保留段使有效槽数少于 `n_parallel` |

`/slots`(只读)、`/props`、`/tokenize` 等非推理端点保留可用。


## 6. 第二阶段扩展预览

| 功能 | 路径与状态 |
|---|---|
| chat/工具调用 | **已实现**:HTTP 层已就绪(`task_result_state` 流式解析),调度器零改动(见 9.1) |
| 结构化输出(grammar) | **已实现**:随每槽采样链存在(随 9.1 一并放开) |
| embedding/rerank | 未设计:需验证 `llama_set_embeddings` + 池化在 paged 图上的支持(llama 层) |
| `n_cmpl>1` 并行采样 | **已实现**:块引用共享 fork(复制 block_table 引用),替代 seq_cp(见 9.2) |
| 槽位存取/RAM 提示缓存 | 未设计:paged 内存的块级 `state_write/read` 序列化 |
| 投机推理 | **已实现**:目标侧验证经调度器,draft 上下文保持经典 KV;变长反馈与位置级回滚(见 9.3) |
| 多模态 | **已实现**:分段入队 + embd 批 + prefill 中途编码暂停态(见 9.4) |
| 动态准入 | 未设计:评估去掉 server slot 池、请求全量入调度器 waiting 的 vLLM 式模型(需 request 注册表与取消重构) |

> 注:9.3 的早期预判"draft 上下文 paged 化"已修正——draft 是独立 context,保持经典 KV
> 即可,复杂度集中在目标侧验证与回滚。

## 7. 风险与开放问题

1. prefill 独占步:长 prompt 分块期间,在跑请求的 decode 被推迟,TPOT 出现毛刺;每步 token 数受 `n_batch` 预算约束,毛刺上界可控,但与默认路径 join-batching的行为差异需要在验收时量化(`scripts/perf-paged-vs-unified.py` 可复用)。
2. 抢占/重算毛刺:KV 块满时换出或重算带来尾延迟;`--kv-paged-watermark`(默认 0.05)与 CPU 块数是主要调参项。
3. 空批衔接:正常情况"批空 <=> 无在途请求"(含义见 3.4),无在途时挂起等待即可。风险在反常情况:在途请求非空却产出空批,属 `check_deadlock`/`check_livelock`两道检测缝隙里的理论残留(实现 bug、边界竞争)。此时在途非空使循环重投递,再调 `update()` 又得空批,形成无 sleep 的忙轮询(烧满一核但不做任何 decode)。兜底:连续 64 步空批且在途非空时把受影响请求按错误收尾。
4. `on_finish` 依赖:`prepare_batch` 返回 false(死锁)后回调不再触发,必须走强制释放兜底(见 3.4);另一路径是库层提供显式 reap,作为备选。
5. hybrid 模型 checkpoint 保留段使可用 slot 数少于 `n_parallel`,启动校验需以 `llama_paged_scheduler_max_request_id` 的返回值为准裁剪槽池。
6. `n_ctx` 语义变化(总预算 -> 单请求上限):`--kv-unified-per-slot` 等参数互斥报错,文档明确说明;上下文满即结束(无 ctx_shift),依赖超长上下文的客户端会观察到行为差异。
7. 空 prompt / `n_predict == 0` / 请求 id 复用窗口(停止到 on_finish 之间)等边界,已在 3.4 设计,实现时需要专项用例覆盖。
8. 默认路径回归面:`process_single_task` 的路由分支重构影响所有任务类型,`server_slot`/`server_batch` 的纯移动需要全量 server 测试佐证;已回归通过(见 9.6)。

## 8. 实施记录(第一阶段)

实际改动(`git diff`,不含设计文档与 README 表格微调):

| 层 | 文件 | 改动 |
|---|---|---|
| llama | `include/llama.h` | +11:cancel / max_request_id 声明 |
| llama | `src/llama-paged-scheduler.cpp`、`-impl.{h,cpp}` | +74:cancel(running/swapped/waiting 同步回收)、max_request_id、queue_request 重复 id 拒绝 |
| llama | `src/llama-kv-cache-paged.h`、`llama-paged-prefix-cache.h` | +3:n_seq_max / ckpt_slot_base 访问器 |
| llama | `common/arg.cpp` | paged 参数解封 LLAMA_EXAMPLE_SERVER |
| llama | `tests/test-paged-kv-e2e.cpp` | +109:cancel 用例(waiting/running 取消、未知 id、id 复用、max_request_id) |
| server | `tools/server/server-slot.h`(新) | 648 行纯移动(slot_state + server_batch + server_slot) |
| server | `tools/server/server-scheduler.h`(新) | 79 行:port 接口(server_context_port)+ 调度器接口 + 工厂声明 |
| server | `tools/server/server-scheduler-paged.{h,cpp}`(新) | 415 行:launch/cancel/update/init/on_finish/错误路径/指标 |
| server | `tools/server/server-context.cpp` | +113 / -641(删除部分为纯移动):启动门禁、任务路由、port 实现块、warmup 跳过 |
| server | `tools/server/CMakeLists.txt`、`README.md` | 构建列表、参数表与 Paged KV cache 章节 |

与最初草案的偏差(实现期确认后已并入上文设计,此处留档):

1. `server_batch` 随 `server_slot` 一并移入 `server-slot.h`:`handle_last_sampled_token(server_batch&)` 依赖完整定义,不移动则纯移动不成立;仍为零语义变化。
2. 未引入 `server_scheduler_default` 适配器:默认路径的推理任务分支含 `n_cmpl>1` 父子槽与 `cache_idle_slots` 保存逻辑,搬进适配器会放大回归面,与"默认路径零行为变化"冲突。改为 `process_single_task` 内一个 `if (scheduler)` 分支,默认路径原样保留;接口与工厂仅服务 paged 路径。
3. 调度器对 impl 的访问采用 `server_context_port` 抽象基类(impl 继承实现,类名描述 impl 扮演的角色:server_context 上暴露服务的 port),替代文中"内部访问头或 friend";原有同名方法(process_token/send_*/metrics_*)直接满足接口。
4. `update()` 返回 void:实际队列循环的保活机制是重投递 NEXT_RESPONSE(默认路径同款),接口返回"是否在途"无处消费;空批 + 在途非空的忙轮询用"连续 64 步兜底报错"实现。
5. cancel 对三种状态(waiting/running/swapped)一律同步回收:`free_blocks` 同时释放 GPU 与 CPU 池块,换出中的请求无需先 swap-in,彻底消除"悬挂请求";`finish()` 触发 on_finish,时序与文中一致。
6. `queue_request` 增加重复 request_id 拒绝(槽位复用窗口与 cancel 幂等性依赖它干净失败)。
7. `--kv-paged` + `--kv-unified` 改为强制关闭并告警(而非启动报错):kv_unified 在 paged 路径本就是 no-op,硬报错会使默认 `-np auto`(隐含 kv_unified)与 `--kv-paged` 无法组合。
8. L4 无需改码:`fit_params` 默认开启,`common_init_from_params` 已对 kv_paged 自动适配块数。
9. paged warmup 失败降级为告警(默认 warmup 失败同样非致命)。

## 9. 第二阶段设计:四个明确支持的特性

范围:a) 投机推理、b) 多模态、c) `n_cmpl > 1` 并行采样、d) `/v1/chat/completions`。
共性前提:默认路径行为不变;所有扩展只作用于 `--kv-paged` 分支;运行时验收以 310P
为准。

### 9.0 前置小修与统一反馈契约

**P0(过渡闸,已随 9.4 移除)**:phase-1 的 paged launch 未检查 `task.tokens.has_mtmd`,多模态请求会被`get_text_tokens()` 静默丢掉媒体内容,位置在 `server-scheduler-paged.cpp:124`。在 b) 落地前补一道闸,使媒体请求明确返回未支持错误;9.4 分段启动落地后,该闸随 `has_mtmd` 分流一并移除。过渡期一个版本,无残留逻辑。

**统一反馈契约,一次设计到位**:由于 c) 需要 fork 出的子请求的批外反馈,a) 需要投机验证一次接受多个 token 的每 seq 变长 token 列表,直接将 `llama_paged_scheduler_update` 的签名改为 feedback 数组形式,不引入 `_v2` 新符号。理由:该 API 的调用方全部在树内且仅4 处,即 examples/paged、tests/test-paged-kv-e2e、shim 内 warmup、server-scheduler-paged,一次机械迁移即可;本 fork 新增 API 无外部消费者;llama.cpp 惯例是原位演进 C API 签名,`_ext` 类后缀仅在两种能力需长期并存时使用,而新形式是严格超集,旧语义等价于每项n_tokens = 1,无并存价值:

```c
struct llama_paged_feedback {
    int32_t            request_id;
    const llama_token * tokens;    // accepted/sampled tokens this step
    int32_t            n_tokens;   // 1 for plain decode, 1..n_draft+1 for spec verify
    int8_t             stop;       // finish the request after applying this feedback
};

// batch may be NULL when all feedback entries are out-of-batch (n_cmpl children)
LLAMA_API void llama_paged_scheduler_update(struct llama_paged_scheduler * sched,
                                             struct llama_batch * batch,
                                             const struct llama_paged_feedback * feedback,
                                             int32_t n_feedback);
```

语义:批内 seq 先按 `n_past += batch_lens[i]` 推进,再以 feedback 的 token 列表为准推进
`logical_seq/n_decoded`;token 数少于批行数即为隐式回滚,供 a) 的投机部分接受使用;批外的
fork 子请求只做 token 追加与状态推进。4 处旧调用点原为两个并行数组的用法,随签名一并
迁移,各约 5-10 行,计入 c) 的工作量。

### 9.1 d) /v1/chat/completions,含工具调用

**结论:调度器层零结构改动,纯放闸 + 验收。**

- 任务到达 `process_single_task` 时端点无关:chat 经 `oaicompat_chat_params_parse` 在
  HTTP 层套模板后,与 `/completion` 一样是 `SERVER_TASK_TYPE_COMPLETION`,仅 `res_type`
  不同;而 `res_type` 只被结果层消费,即 `server_task_result_*::to_json` 与 SSE 格式选择;
- 工具调用 grammar 由 chat 模板解析器自动生成,类型为 `COMMON_GRAMMAR_TYPE_TOOL_CALLS`,
  完全活在 `common_sampler` 链内;paged 的每槽采样链与默认路径一致,技术上已兼容;
- 流式工具调用解析位于 `task_result_state::update_chat_msg`,与 reasoning 解析一样均在
  结果层。

改动:

| 位置 | 内容 | 行数 |
|---|---|---|
| `server-scheduler-paged.cpp` | `paged_task_supported` 放开 `OAI_CHAT` 与 grammar 闸 | ~10 |
| 同上 | P0 的 `has_mtmd` 闸 | ~10 |
| `README.md` | 限制章节更新 | ~10 |

风险:低。验收矩阵:chat 流式与非流式、工具调用含流式 tool_call 增量、reasoning 解析、
`logprobs`,以及 grammar 与 n_predict/stop 的组合。llama 层 0 行,server 层约 15-25 行。

### 9.2 c) `n_cmpl > 1` 并行采样

默认路径:父槽加 N-1 个子槽,由 `add_child` 派生并递增种子;子槽处于 WAIT_OTHER 状态等待
父槽 DONE_PROMPT,`decode()` 内 `copy_state_to` 通过 `mem.seq_cp` 整段复制父 prompt KV,
子槽从继承的 `i_batch` 行采各自的首 token,见 `server-context.cpp:3198-3219`。

paged 方案:由于 KV 池的 `seq_cp` 是桩,改用块引用 fork,直接复用前缀缓存的引用计数
共享机制 `add_ref_blocks`:

```c
// fork a fully-prefilled request: dst shares src's full prompt blocks by
// reference; a partially-filled tail block is eagerly copied per child.
// hybrid recurrent state is seq_cp-ed. src must have n_past == n_prompt_total.
LLAMA_API bool llama_paged_scheduler_fork_request(struct llama_paged_scheduler * sched,
                                                   int32_t src_id, int32_t dst_id);
```

- 实现要点:校验 src 已完成 prefill;复制 group,包括 `block_table` 引用与`add_ref_blocks`、`logical_seq` 截到 `n_prompt_total`、`n_past` 继承、`n_decoded = 0`,状态置 RUNNING 并按到达时间插入 running 列表;hybrid 模型补`hybrid_memory_manager->seq_cp`,与 GDN checkpoint fork 同机制,`llama-paged-scheduler-impl.cpp:215` 已有先例;重复 id 复用 `queue_request` 的拒绝逻辑;
- 尾块处理:若 `n_past` 不在块边界上,部分填充的尾块是多写者冲突点,因为父请求与全部子请求的下一个 decode 都要往该尾块的同一位置写入各自的 token。处理方式为**整块共享 +尾块急拷贝**:满块按引用共享,部分填充的尾块为每个子序列即时复制一份,父请求保留原尾块所有权;需要在 `llama_kv_cache_paged` 增加块内容拷贝入口,复用 swap 路径 `do_block_copy`的设备内拷贝,每个子请求只多一次单块拷贝。不选 vLLM 式 copy-on-write,因为调度器没有懒拷贝基础设施;也不选把 `n_past` 回退到块边界让子请求重 prefill 尾部,因为那会推迟同批首 token 采样;
- 子请求首 token:fork 发生在父槽 final_prefill 的同一 `update()` 迭代内,采样前 logits仍在 ctx 缓冲,所以子槽直接用父槽的末行下标采样,即`common_sampler_sample(child.smpl, ctx, tok_idx)`,经 `process_token` 后以批外 feedback回填。语义等价于默认路径的 `i_batch` 继承。

### 9.3 a) 投机推理

默认路径结构:draft 是独立 context,使用经典 KV,与 paged 目标正交;`common_speculative_draft`
产 draft,验证批含已采样 token 加 draft 且每行都要 logits,由 `handle_last_sampled_token`
装批;`common_sampler_sample_and_accept_n` 逐位验证,部分接受时 checkpoint 恢复或 `seq_rm`
回滚;`common_speculative_process` 把验证批镜像进 draft ctx,见
`server-context.cpp:2424-2541, 3338-3462`。

paged 方案:draft ctx 保持经典 KV,启动时强制 draft 参数 `kv_paged = false`,其 `seq_rm`
与 state 操作全部可用;目标侧验证经调度器,新增两个原语:

```c
// tentatively extend the next decode step with draft tokens: the next batch
// holds 1 + n_draft rows for this seq, logits on every row. auto-cleared
// after the step; acceptance is decided by the caller via the update feedback.
LLAMA_API bool llama_paged_scheduler_attach_draft(struct llama_paged_scheduler * sched,
                                                   int32_t request_id,
                                                   const llama_token * draft, int32_t n_draft);

// roll the request back to n_past_new tokens (KV beyond is released, partial
// tail block kept). used on partial draft rejection.
LLAMA_API bool llama_paged_scheduler_truncate(struct llama_paged_scheduler * sched,
                                               int32_t request_id, int32_t n_past_new);
```

- `attach_draft`:draft 暂存于 group,`populate_batch_from` 对该 seq 产出 `1 + n_draft` 行,块预算按 `allocate(n_draft + 1)` 预留,`batch.logits` 全 1,行位置为 `n_past + i`;行下标由 `batch_info` 的 offsets/lens 推导,等价默认路径的 `spec_i_batch`;
- 验证与回滚,在 server 侧 `update()` 内:先以 `common_sampler_clone` 备份,再用`sample_and_accept_n` 验证;`n_rollback > 0` 时执行`truncate(sched, id, n_past_before + accepted.size())`,并以 `common_sampler_copy` 恢复采样器、`prompt.tokens.keep_first` 收缩 prompt。由于 paged 支持位置级 seq_rm 加块收缩,无需 checkpoint,对齐默认路径 PART 分支;最后 update 反馈携带 accepted 列表;
- `update()` 内 draft 生成:上一轮采样后,按 `get_n_draft_max` 与 `spec_draft` 是否为空组装 `common_speculative_get_draft_params`,并经 port 的 yield 钩子调`common_speculative_draft`;draft ctx 解码期间让 queue 保持响应,对齐默认路径`yield_to_queue`;
- `common_speculative_process(spec, batch)`:直接喂 paged 产出的验证批,seq_id 即槽位 id,与 draft ctx 的 n_seq_max 约定一致,无需改动;
- `common_context_can_seq_rm` 特判:由于探测式分类会对 paged ctx 断言失败,探测 decode 缺`set_paged_batch_info`,在 `common.cpp` 对 `kv_paged` 直接返回`COMMON_CONTEXT_SEQ_RM_TYPE_PART`;paged 支持范围 seq_rm,走最简单的无 checkpoint 回滚;
- 统计:`n_accepted_per_pos`、`spec_is_replay` 与 draft 计数全部沿用槽位字段,与默认路径同构。

### 9.4 b) 多模态:prefill 中途编码,对齐默认路径

默认路径:媒体块在 `server_tokens` 里是 `LLAMA_TOKEN_NULL` 哨兵行,index 与 position 不
相等,`n_tokens != n_pos`;prefill 遇哨兵即暂停 token 装批,经 `process_mtmd_chunk` 编码
后,由 `mtmd_helper_decode_image_chunk` 用自带的 `llama_decode` 把 embd 行写入 KV 并自行
管理位置,再以 `push_back_placeholder` 记账。该 helper 完全绕过调度器,与 paged 的
`write_slots/block_table` 路由不兼容,所以不能复用其解码半段,只复用编码半段,即
`mtmd_batch_*` 加 `mtmd_batch_get_output_embd`。

**编码/解码两半段的准确含义**(命名易误导,`llama_decode` 的 decode 指"对 context 跑
一次前向",与 `llama_encode` 相对,不是"解码生成 token"):

- 编码半段:图像 -> "虚拟 token 的 embedding"。`mtmd_batch_encode` 对 mmproj 视觉模型
  跑前向,输入图像像素,输出每个图块一行浮点向量,维度等于目标 LLM 的输入嵌入维度
  `n_embd_in`(如图 -> 700 行 x 2048 维 float)。产出的不是 token,而是"图块作为词表外
  虚拟 token 的 embedding"。与文本链路对比:文本是 token id 查 embedding 表得向量,
  媒体是 vision encoder 直接产出向量(无 token id 可查),两条路在"输入向量"处汇合;
- 解码半段:embedding -> 该位置的 K/V 入缓存,不生成任何 token。
  `mtmd_helper_decode_image_chunk` 把编码向量行装批(`batch.token = NULL,
  batch.embd = 向量行`)调 `llama_decode`:这些行像普通 token 一样流过 transformer
  各层,每行在自己位置算出 K/V 写入 KV cache,`batch.logits` 全 0,不采样。目的只有
  一个:让图像内容成为上下文记忆,后续文本 token 的 attention 从 KV 读到它。token
  生成只发生在 prefill 末尾与 decode 循环,与两半段均无关。

paged 侧要替换的正是解码半段(embd 行前向、K/V 落缓存改走调度器统一批入口:纯 embd
切片 -> `write_slots` 路由 -> 块池);编码半段产出的只是内存里一段 float,与调度器
正交,原样复用。

**阶段对齐**:两条路径跑在同一个循环阶段。server_queue 主循环每轮迭代经 on_update_slots
回调驱动一次推理:默认路径进入 `server_context_impl::update_slots`,子阶段为
pre_decode 装批 -> decode -> post_decode 采样;paged 路径进入
`server_scheduler_paged::update`,子阶段为 prepare_batch 装批 -> decode -> 采样/反馈。
所以默认路径的 pre_decode 装批与 paged 的 update 迭代是同一阶段的对应物,差异在回调
内部:装批归属不同,默认路径由 server 构建 `server_batch`,paged 路径由调度器
prepare_batch 构建、server 不接触批内容;媒体编码时点基本等价,两条路径的编码都落在
迭代 A 的 decode 与迭代 B 的 decode 之间,默认路径发生在 B 的 pre_decode 开头,paged
路径发生在 B 的 update() 开头(prepare_batch 之前),同为两轮 decode 之间的同一段
墙钟;机制差异在于默认路径经 yield_to_queue 在装批内部插入媒体处理,paged 在 update
入口轮询 seq_state.paused 后直接内联编码,媒体 decode 与调度步同线程,少一次线程
交接。真正多付的一个调度步来自媒体行的解码方式:
默认路径的媒体行由 helper 在 B 的 pre_decode 内私有 `llama_decode` 写 KV,不占共享批,
B 的共享 decode 直接装 text2,请求两轮迭代走完 prefill;paged 路径的媒体行必须作为
纯 embd 切片经 prepare_batch 进共享 decode,而文本行与 embd 行不能混批,embd 切片
独占迭代 B,text2 顺延到迭代 C,请求三轮迭代走完 prefill。这一步是调度器统一批入口、
禁止混批的设计代价。

默认路径流程,以 [text1][img][text2] 三段 prompt 为例,哨兵把 text1 与 text2 拆进
相邻两轮迭代:

```
queue 循环迭代 A -> on_update_slots -> server_context_impl::update_slots
  +-- pre_decode: 装批,server 拥有批构建
  |     text1 token 逐个 add 进 server_batch
  |     text 装批循环遇 LLAMA_TOKEN_NULL 哨兵即 break,该槽本轮装批结束
  +-- decode: llama_decode(batch)                        text1 行,共享 decode
  +-- post_decode: text1 为中途 prefill,无 logits,不采样

queue 循环迭代 B -> on_update_slots -> update_slots
  +-- pre_decode:
  |     mtmd 循环见哨兵,yield_to_queue 内执行 process_mtmd_chunk:
  |       mtmd_batch_encode                               vision 上下文编码
  |       mtmd_helper_decode_image_chunk                  自带 llama_decode,
  |                                                        embd 行、自管位置,
  |                                                        不经 server_batch
  |       push_back_placeholder                           prompt.tokens 记账,
  |                                                        pos_next 前进
  |     img 段消费完毕,text 装批循环继续,text2 进 server_batch
  +-- decode: llama_decode(batch)                         text2 行,共享 decode
  +-- post_decode: 末行 logits -> 采样
```

paged 方案:分段入队加外部暂停态。请求以分段骨架入队,媒体段先声明行数,编码前
`n_tokens` 已知,编码数据后补;调度器把 embd 行当普通逻辑行调度,批构建时产出纯
embd 切片:

```c
struct llama_paged_segment {
    const llama_token * tokens;      // text segment rows; NULL for media
    const float *       embd;        // media segment rows [n_tokens x embd_stride]; NULL = pending encode
    int32_t             n_tokens;    // rows in this segment
    int32_t             embd_stride; // media: floats per embd row (n_embd_inp); 0 for text
    const llama_pos *   pos;         // media mrope positions [n_tokens x 4] row-major; NULL for text
};

// media segments may pass embd = NULL (declared size only): the request pauses
// at that boundary until append_segment supplies the encoder output
LLAMA_API bool llama_paged_scheduler_add_request_segs(struct llama_paged_scheduler *      sched,
                                                       const struct llama_paged_segment * segs,
                                                       int32_t                             n_segs,
                                                       int32_t                             request_id);
LLAMA_API bool llama_paged_scheduler_append_segment(struct llama_paged_scheduler *      sched,
                                                     int32_t                             request_id,
                                                     const struct llama_paged_segment * seg);
```

paged 路径流程,同一 [text1][img][text2] 请求,与默认路径逐迭代对照:

```
launch
  -> add_request_segs
       text1 段:tokens 实数据
       img 段:  embd = NULL,仅声明 n_tokens              pending 声明
       text2 段:tokens 实数据
  -> start_mm_prefetch: worker 线程预取首个媒体 chunk 的
       clip 编码,与调度步重叠(mm_encode_mutex 串行化
       共享 mmproj 上的编码;失败由消费方回退内联编码)

queue 循环迭代 A -> on_update_slots -> server_scheduler_paged::update
  +-- prepare_batch: 装批,调度器拥有批构建
  |     text1 段可跑 -> 产出 token 切片
  |     到达 img 段边界且 embd 为 NULL -> 请求置 PAUSED_EXTERNAL
  |       不进候选,不计入 check_deadlock 分母
  |       其余请求照常组批,互不阻塞
  +-- decode: llama_decode(本批切片)                     text1 段行
  +-- 采样/反馈: text1 为中途 prefill 块,无 logits、无反馈

queue 循环迭代 B -> on_update_slots -> update
  +-- update 开头: 轮询 seq_state 发现该请求 paused
  |     join 预取 worker(通常已编完)或内联编码:
  |       mtmd_batch_encode -> mtmd_batch_get_output_embd
  |     append_segment(id, embd):embd 为借用指针,
  |       由槽持有的 mbatch 保活,请求回 RUNNING
  |     再启动下一个媒体 chunk 的预取 worker
  +-- prepare_batch: 为 img 段产出纯 embd 切片
  |     batch.token = NULL,batch.embd 指向段缓冲
  |     write_slots 按每行位置数组路由,KV 写入块池
  +-- decode: llama_decode(embd 切片)
  +-- img 段完成

queue 循环迭代 C -> on_update_slots -> update
  +-- prepare_batch: text2 段回到普通 token 切片
  +-- decode: llama_decode(text2 切片)
  +-- 采样/反馈: text2 为最终段,末行 logits -> 采样
```

**时延影响**:默认路径两轮迭代走完 prefill,paged 三轮,多的一个调度步代价如下。

- 计算量不变:目标模型的 decode 调用两边都是三次(text1、embd 行、text2),KV 计算与
  写入完全相同,多的只是循环边界的组织方式;
- 单请求场景:额外开销仅为一轮循环边界的主机侧工作(prepare_batch 调度与反馈记账,
  亚毫秒级),相对图片编码的几十到几百毫秒可忽略;且编码由预取 worker 与调度步重叠
  (launch 即开编),通常不在关键路径,内联编码仅作回退;
- 编码批处理两路径等价:mtmd_batch_encode 在 vision 上下文执行,默认路径会把同请求
  后续图片贪婪攒入一个 encoder 批;paged 的 update() 开头编码复用同一逻辑,同样可
  一次编完该请求全部 pending 媒体段,编码次数与耗时相同,与节拍延迟无关;
- 并发场景:循环节拍由在跑 decode 的批决定,媒体请求的 TTFT 约延后一个节拍,即系统
  当前 TPOT,典型 10-50ms 量级;相对编码与 prefill 总耗时通常为个位数百分比;
- "n 张图累积 n 个节拍"仅指**单请求、n 张图被文本隔开**的场景(如
  [text1][img1][text2][img2]...),不可一步写入的原因有两个:其一,因果链:img_k 的
  attention 要读中间文本行的 K/V,而中间文本又要读 img_{k-1} 的 K/V,同一 seq 的
  位置轴上前向依赖串行不可折叠,attention 无法读还不存在的 K/V(prefill 大批一次算完
  靠的是 ubatch 内先行后算 + causal mask,但要求批内行同型);其二,批类型同质性:
  llama_batch 的 token/embd 是批级开关,build_inp_embd 对整个 ubatch 二选一,text 行
  与 embd 行无法同批。两个约束与调度器无关,是因果注意力与 batch 抽象的固有性质;
- 默认路径并非并行:同一请求的媒体前向默认路径同样逐次串行调用(helper 的
  mtmd_helper_decode_image_chunk 内部还按 n_batch 切片),n 张图同样是 n 次 LLM 前向,
  GPU 计算量两边完全相同;默认路径省的是**调度节拍**而不是前向次数:媒体前向发生在
  pre_decode 内部的私有 llama_decode,不占共享批、不产生 queue 迭代,对外表现为
  "两轮迭代走完 prefill",媒体步被藏进文本步的迭代里;
- 节拍账目按场景分:单请求 n 图被文本隔开 -> paged 2n+1 个调度步(默认 n+1 个共享
  步,媒体步藏内,前向次数两边相同);单请求 n 图相邻 -> 连续同型,合并为一个 embd
  段,paged 1 个节拍;n 个请求各 1 图 -> 跨 seq 无位置依赖,不存在因果约束,paged 的
  embd 步只要求同型、不限请求,全部 pending embd 段可合入同一调度步,paged 1 个节拍;
- 吞吐不受影响,并发多模态时甚至更优:默认路径的 helper 私有 decode 逐槽串行、永远
  单请求,n 个请求的图片要 n 次私有 decode 顺序执行;paged 可跨请求合批。n 节拍惩罚
  只作用于"单请求多图且非相邻"场景下该请求自身的 TTFT,系统总计算量不变;
- 若需消除该步,后续可评估同一轮内连续两次 llama_decode(先 embd 后文本)或内核支持
  混合 ubatch,第一版不做。

- 序列模型:`llama_sequence_group` 增加分段描述与每行位置数组,打破现有"序号等于位置"的
  假设,`pos = n_past + token_idx` 改为查表;`write_slots` 按位置算块,天然兼容;媒体行
  携带 [t][y][x][z] 的 mrope 位置数组——M-RoPE 模型正是多模态主力,直接支持而非按架构
  闸掉(多序列并发曾出批位置错乱,已修复,见 9.6 第 8 条);媒体组(has_media)不参与
  前缀缓存与重算:媒体行无 token id(逻辑序以 LLAMA_TOKEN_NULL 占位),块哈希链与
  重算均以 token id 为前提,抢占只能换出;
- PAUSED_EXTERNAL 调度态:到达 pending 段边界即暂停,不进候选、不计入死锁/活锁检测,
  `check_deadlock` 分母排除;`append_segment` 供给 embd 后回到 RUNNING。对外经
  `seq_state.paused` 标志而非并入生命周期枚举:暂停与 WAITING/RUNNING/SWAPPED 正交
  (等媒体可发生在任一阶段),独立标志避免状态组合枚举膨胀;
- 缓冲所有权:embd 按借用指针挂段,编码输出由槽持有的 mbatch 保活——段消费必然先于
  请求 finish/槽释放(update 内编码 -> append -> 后续步消费的顺序保证),借用安全,
  免一次整段拷贝(典型图 700 行 x 2048 维 float 约 5.7MB);pos 数组每行 4 个
  llama_pos,体量小,拷贝进段;
- 批构建:同段同型才可同批,因为文本 token 与 embd 行不混批,`build_inp_embd` 只在
  `ubatch.token == nullptr` 时走 embd 路径;embd 切片中 `batch.token = nullptr`、
  `batch.embd` 指向段缓冲;logits 仍只在最终段末行;
- server 侧:`update()` 开头轮询暂停槽,内联编码(mtmd vision 上下文独立,期间其余
  请求照常跑批)后 `append_segment` 恢复;launch 与每次 append 后由 worker 线程预取
  下一个媒体 chunk 的编码,与调度步重叠;媒体请求跳过 spec 起草(b + a 组合不报错,
  退化为无草稿,见 9.5);`prompt.tokens` 以 placeholder 记账,`pos_next()` 仍是
  位置权威;embedding/rerank 依旧禁用,`need_embd` 与 paged 池化问题相互独立。

### 9.5 特性交互与实施顺序

| 组合 | 默认路径 | paged 第二阶段 |
|---|---|---|
| d + 任意 | - | 无冲突 |
| c + a,n_cmpl 加投机 | 支持,fork 不复制 spec 状态,子槽独立起草 | 支持,fork 同理只复制 prompt 段 |
| b + a,多模态加投机 | 支持 | 组合放行:媒体请求跳过 spec 起草(spec 退化为无草稿),由此绕开 draft prompt 媒体段文本化问题 |
| b + c | 支持 | 支持,fork 含媒体段,块引用共享不受段型影响 |

实施顺序:P0 -> d,放闸即刻可用 -> c,fork 基础设施与契约重构 -> a,复用新契约,工作量
最大 -> b,独立、可与 a 并行开发,运行时验收依赖内核 embd 支持。9.0 的契约重构必须先于
c/a 落地。

### 9.6 第二阶段实施记录

以下为落地时的关键实现事实;9.0-9.4 正文即最终方案,与实现一致:

1. `update` 契约按 9.0 修订原位改签名,4 处树内调用点(examples/paged、e2e 测试两处、shim 内 warmup)随签名迁移。
2. `can_seq_rm` 的 paged 特判落在 `server-context.cpp` 的 `ctx_tgt_seq_rm_type` 赋值处(以 `params_base.kv_paged` 判 PART),而非 `common.cpp`:common 层只有 C API,看不到cparams,server 侧判断零成本且避免新增公共接口。该特判同时修复了一个 phase-1 隐患:探测式分类会对 paged ctx 断言失败,此前 server 带 `--kv-paged` 启动会在 load 阶段崩溃。
3. 投机推理走 PART 语义:部分接受经 `prompt.tokens.keep_first + insert` 修复后直接`truncate` 到新的验证末尾,无 checkpoint、无 replay(`spec_is_replay` 恒为 false),与默认路径 PART 分支同构;synthetic acceptance 在 `--kv-paged` 下启动期拒绝。
4. 验证采样复用 `common_sampler_sample_and_accept_n`,不在 server 侧重写逐行采样。
5. fork 的尾块急拷贝:满块 `add_ref_blocks` 引用共享,`n_past % block_size != 0` 时为子请求分配私有块并经新增的 `copy_block_content`(同池直拷/跨池走 `do_block_copy`)复制父尾块;`truncate` 释放块时调用新增的 `prefix_cache.drop_block` 撤销已释放块的前缀缓存登记,避免 LRU 驱逐命中陈旧块号。
6. 多模态按 9.4 方案落地(正文即实现方案)。实现期附带修复与规避:qwen3vl / qwen3vlmoe 图构建改 `build_attn_inp_kv_auto`(统一缓存的 cpy_k 未接入 paged 分派,此前 paged 路径加载即段错误);llama-kv-cache-paged 的 init_batch 增加分割失败诊断日志;paged 路径跳过 mmproj fit 显存测量,规避 310P CANN buffer 释放缺陷;
7. 图文 TTFT 优化(`79d9eb6c0`):launch 与每次 append 之后在 worker 线程预取下一个媒体 chunk 的 clip 编码,与调度步重叠;`mm_encode_mutex` 串行化共享 mmproj 上的编码,消费方 join 后复用结果,失败或过期回退内联编码;slot reset 先 join 再释放task;init 时 `mm_warmup` 用合成图预热 clip 图,消除首图请求的首次编译开销。310P 实测(Qwen3-VL-4B F16):首图 TTFT 497ms -> 317ms,热图 392ms -> 312ms,与 default 路径热态差距从 124ms 缩至约 45ms;
8. M-RoPE 多序列批位置修复(`3121b0ef4`):`populate_batch_from` 对 `n_pos_per_embd > 1`的模型曾按 [section][本序列行数] 平铺覆盖写 `populate_pos`,未加批内偏移且基数错,第二个序列起覆盖前序序列的位置。修复后平铺写仅用于媒体(embd)行且基数改为批内总行数并加 token 偏移,文本行只写每 token 单值。310P 实测:修复前 4 并发 2/4 失败,修复后 4/4 与 2/3 并发全通过,总吞吐 56.5 tok/s(单流 3.55 倍),13 用例回归通过。

验证:CANN aarch64 编译通过(llama-server + test-paged-kv-e2e)。默认调度回归通过
(/completion、/v1/completions、/v1/chat/completions 含工具调用、n=2 并行)。paged 路径
的运行时功能验收已在 310P 完成:图文请求贪婪输出与默认路径逐 token 一致,
`test_paged_server_mm.py` 3 用例(逐 token 等价、图像后文本携带图像上下文、prompt token
记账)与 10 个 paged server pytest 用例通过;此前记录的投机推理缺陷(verify 门位、attach
重复推入、truncate 时序、n_keep_blocks 向下取整)修复后,n_max 2/3/4/6/8 的 128 token
输出与无投机基线逐 token 一致(2026-10-28 复测);`--mmproj` 进程退出 abort 已修复
(`mtmd_free` 前移至 `destroy()` 开头),paged 与 default 各 8/8 启停无 abort。

### 9.7 develop 优先级架构适配与 sdk-demo(2026-10-29 至 2026-11-02)

rebase 到含请求优先级调度(!23/!29)的 origin/develop 后的增量适配:

1. 队列抽象与定点摘除(`98d97d995`):waiting/swapped 改经 `llama_paged_queue` 接口
   (`src/llama-paged-queue.h`),FCFS 实现(按到达序 deque)与优先级实现
   (`group_precedes` 最小堆)均支持 `take(request_id)`(FCFS 线性扫描,优先级队列
   弹出重建);cancel 经队列 take 摘除组(running 仍为到达序链表);
2. `process_waiting_list` 预算参数改传值:prefill 独占步使用全新预算,头部准入语义
   保留;抢占牺牲者选择与换入次序按模式分化(`eviction_precedes`:FCFS 牺牲最新
   到达,优先级模式牺牲最低优先级),并保留步内优先级反转保护;
3. 模式与参数:`--enable-priority`(`common_params.enable_priority`)经
   `common_init_from_params` 映射为 `llama_context_params.paged_priority`,
   在 `llama_paged_scheduler_init` 时固定;`add_request` 增加 `priority` 实参
   (钳制到 `[-LLAMA_PAGED_PRIORITY_LIMIT, LLAMA_PAGED_PRIORITY_LIMIT]`,
   非 priority 模式记录并忽略);`llama_paged_seq_state` 增加 `priority` 字段;
4. server 侧暂未接每请求优先级:`server_scheduler_paged` 的 `add_request` 仍以默认
   0 入队(每请求优先级需从 server 任务层透传,后续增量);库层行为已由
   `examples/paged`(priority 池)与 `tests/req_priority`(单测 + e2e,含换入/换出
   优先级)覆盖;
5. 测试迁移:`test-paged-kv` 与 req_priority 测试的 update 调用迁至 feedback 契约,
   并补 `n_pos_per_embd = 1` 构造(纯文本合成 fixture,多模态提交给 impl 构造函数
   加参导致的接口漂移,`718566a3d` 修复);mid-prefill chunk 仅推进 KV 不带 feedback
   token,避免块状预填被误计为解码;
6. llama-sdk-demo(`0fc96a6b5`):新增 `tools/server/sdk-demo.cpp`,进程内(无 HTTP)
   经 `server_context` + `server_response_reader` 提交任务与收取结果,paged 调度器下
   10 路并发推理验证通过;cli 输入经 `tokenize_cli_input` 在服务侧分词(vocab/mmproj
   不外露)。附带修复:defer 后重新处理的 cli 任务会以已清空的 cli_prompt 再次分词,
   覆盖首次分词结果导致空响应(tokenize_cli_input 跳过已分词任务)。


## 10. 调试与验收记录:310P + CANN 9.1.0

运行时功能验收只在 310P 上进行。

### 10.1 环境与构建

- Atlas 300I Pro(310P3)单卡(`ASCEND_RT_VISIBLE_DEVICES=1`),CANN 9.1.0,GCC 11.4;
- `-DGGML_CANN=ON -DCMAKE_BUILD_TYPE=Release` 下 llama-server 与 test-paged-kv-e2e 构建通过;server 侧功能测试复用 pytest 基础设施(`tools/server/tests`);
- 模型 Qwen3-1.7B F16 GGUF(head_dim=128,满足 310P 约束),由本地 safetensors 经`convert_hf_to_gguf.py` 转换;
- 服务命令:`llama-server -m <model> --kv-paged --cache-ram 0 -b 2048 -ub 2048 -ngl 99 -c 4096 -np 4`。注意 `--cache-ram` 默认 8192M 会被门禁拒绝,须显式置 0。

### 10.2 验收结果(2026-10-26)

| 组 | 项目 | 结果 |
|---|---|---|
| 阶段一 | /completion 非流式 + 流式 SSE、/v1/completions、stop 串、n_predict、EOG、采样参数、温度 0 确定性 | 全部通过 |
| 阶段一 | 并发 8 请求 / 4 槽:超额排队与释放续跑 | 通过 |
| 阶段一 | 块级前缀缓存:451 token 前缀二次请求 cache_n=448,prompt 时延 195.9ms -> 51.7ms | 通过 |
| 阶段一 | 客户端断连取消:槽位与 KV 回收,后续请求正常 | 通过 |
| 阶段一 | /metrics(需 `--metrics`)、/props、/tokenize、/slots | 通过 |
| 阶段二 | /v1/chat/completions 非流式 + 流式(reasoning_content) | 通过 |
| 阶段二 | 工具调用:get_weather({"city": "Paris"}),finish_reason=tool_calls | 通过 |
| 阶段二 | n_cmpl>1:n=2 返回 2 choices | 通过 |
| 阶段二 | json_schema 严格约束输出 | 通过 |
| 阶段二 | 投机推理:启动放行,运行时输出错误 | 未通过,见 10.3 缺陷 2 |
| 门禁 | -b != -ub 报错;--kv-paged + --kv-unified 告警关闭;双卡报错附修复建议;--embeddings / --reranking / --cache-ram 启动报错;--cache-reuse 告警关闭;/infill 请求期 501;embeddings / rerank 端点 501;图像输入明确报错 | 全部符合设计 |
| 对比 | 同 prompt 贪婪 16 token,paged 与 default 逐字一致 | 通过 |

### 10.3 缺陷与修复

1. (已修复,server +1 行)`--kv-paged` + 投机推理启动崩溃:draft ctx 继承
   `kv_paged=true`,load_model 对 ctx_dft 的探测式 `common_context_can_seq_rm` 在无路由
   元数据的 paged ctx 上触发 `llama-kv-cache-paged.cpp:375` 断言(目标 ctx 在
   server-context.cpp 已有同款特判,draft 漏了)。修复:`params_dft.kv_paged = false`,
   对齐 9.3"draft 保持经典 KV"。
2. (已修复 4 处)投机推理在 paged 路径的账本缺陷:update post 循环的 verify 门位于
   sample_now 判断之后,verify 批(len = 1 + n_draft)被当作 prefill 中途块跳过,
   verify_slot 不可达;attach 重复推入 pending token(plain 路径已推);truncate 在
   调度器 update 之前执行,被 "cannot truncate" 静默拒绝;truncate 的 n_keep_blocks
   向下取整,把承载有效 KV 的部分块整个释放,块表轮换后旧位置全部读空。修复后
   paged+spec 全深度贪婪等价:n_max 2/3/4/6/8 的 128 token 输出与无投机基线逐 token
   一致(2026-10-28 复测);此前深度 >= 3 观察到的毒化系上述缺陷所致,见 10.4。
3. (已澄清)test-paged-kv-e2e 曾在 310P 失败:CANN paged 图必须先经
   `llama_paged_scheduler_warmup` 预热,否则 decode 步采样到陈旧 logits(server 有预热
   而 e2e 没有)。已在 e2e 补预热;残余 token 级漂移属在线/两遍 softmax 的 F16 数值差,
   同 prompt 下 server paged 与 default 逐字一致,top-5 重合 5/5。
4. (已修复,server +5/-3 行)`--mmproj` 进程退出时 CANN abort(`rtFree invalid
   value` + `ctx is NULL`):`destroy()` 先 `llama_init.reset()` 释放主 backend
   并 `aclrtResetDevice` 销毁设备上下文,其后的 `mtmd_free` 在上下文销毁后释放
   clip 的 CANN 缓冲。310P 实测修复前 paged+mmproj 8/8 次退出 abort(default 路径
   同序但 8/8 干净),启动阶段两路径 8/8 均正常,即问题仅在进程退出阶段出现。修复将
   `mtmd_free` 前移至 `destroy()` 开头(设备上下文存活时释放);修复后 paged 与
   default 各 8/8 启停无 abort,mm 请求 sanity 与 13 用例回归(--noconftest 串行)通过。

观察:`t_max_predict_ms` 在 default 与 paged 路径均未按预期截断(fork 级问题,非 paged
回归)。

### 10.4 遗留项

- ATB SF op(CALC_TYPE_SPEC)排查记录:深度 >= 3 的毒化曾疑其多行写路径问题,排查已
  排除调度器/服务端账本、ReshapeCache 跨块写入、ACL graph;独立探针(atb_probe.cpp)
  在共享多租户设备上算子静默无效;vllm-ascend 310P 投机(torch_npu
  rejection_sampler.py + attention_v1.py)掩码/布局语义逐项比对一致。缺陷 2 修复后
  毒化消失,该怀疑解除;若更换内核或驱动后再现,循 server 进程内探针 + 向华为获取
  CALC_TYPE_SPEC 张量契约重查;
- e2e 的 N_COMPARE=4 等价窗口是否按平台/模型放宽,由 owner 决定;
- 910B 内核适配若立项,另开文档。
