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
| chat/工具调用 | HTTP 层已就绪(`task_result_state` 流式解析),开启端点验收即可,调度器零改动(已细化,见 12.1) |
| 结构化输出(grammar) | 已随每槽采样链存在,补验收测试即可(随 12.1 一并放开) |
| embedding/rerank | 需验证 `llama_set_embeddings` + 池化在 paged 图上的支持(llama 层) |
| `n_cmpl>1` 并行采样 | 基于块引用共享实现调度器侧 fork(复制 block_table 引用),替代 seq_cp(已细化,见 12.2) |
| 槽位存取/RAM 提示缓存 | paged 内存的块级 `state_write/read` 序列化 |
| 投机推理 | 目标侧验证经调度器,draft 上下文保持经典 KV;需变长反馈与位置级回滚(已细化,见 12.3) |
| 多模态 | 分段入队 + embd 批 + prefill 中途编码暂停态(已细化,见 12.4) |
| 动态准入 | 评估去掉 server slot 池、请求全量入调度器 waiting 的 vLLM 式模型(需 request 注册表与取消重构) |

> 注:12.3 的早期预判"draft 上下文 paged 化"已修正——draft 是独立 context,保持经典 KV
> 即可,复杂度集中在目标侧验证与回滚。

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

## 11. 调试记录:910B4 + CANN 9.0.0

编译环境为 openEuler 22.03 aarch64、CANN 9.0.0,含 ops 与 nnal 9.0.0,以及 GCC 10.3。在
`-DGGML_CANN=on` 下 llama-server 与 test-paged-kv-e2e 构建通过。

| 问题 | 结论 |
|---|---|
| 服务器缺 aclnn 算子头文件 | 补装 `Ascend-cann-910b-ops_9.0.0`。nnal 原为 8.5.0,与 toolkit 9.0.0 版本错配,已换装 9.0.0 |
| ggml-cann.cpp 910B 编译失败 | 上游 bug:`transform_kq` 定义在 `#ifdef ASCEND_310P` 内,而 `buffer_set_tensor` 的调用点未加保护。已按文件既有模式把 `if (is_kq)` 块包进 `ASCEND_310P`。910B 下 `kq_path` 恒为 `GGML_CANN_KQ_NONE`,该块运行时本就不可达,行为无变化 |
| server_tokens 拷贝赋值 | `server_tokens` 有意禁止拷贝,源码留有 TODO;paged launch 改为 `get_text_tokens()` 构造后 move 赋值 |

运行时:非 paged 路径在 NPU 上正常出 token。paged 路径在 910B4 上被 ATB PagedAttention
内核阻塞:`op->Setup` 返回 error 8,即 ERROR_INVALID_TENSOR_DIM,ATB 日志明确报
`headSize of keyCache and query should be same`。原因:本仓库的 CANN paged-attn wrapper
按 310P ATB 编写,KV 池以 FRACTAL_NZ 解释并走 `K_CACHE_V_CACHE_NZ` 写入路径,而 310P 的
SelfAttention/PA 支持 NZ cache;910B 的 ATB 9.0.0 PagedAttention 对 NZ cache 的维度解释
不同,直接复用会触发维度校验失败。另外 Qwen3-4B 的 head_dim 为 80,不是 2 的幂,是否满足
310P 约束亦需复核。

结论:paged attention 内核的 910B 适配是独立的内核层工作,评估方向为改喂 ND 布局 cache,
或改用 910B 对应的 ATB/原生算子;该项不在本次调度器集成范围内。LLAMA_PAGED 前缀的调度器
与 server 集成代码与设备无关,内核打通后即可复验。

## 12. 第二阶段设计:四个明确支持的特性

范围:a) 投机推理、b) 多模态、c) `n_cmpl > 1` 并行采样、d) `/v1/chat/completions`。
共性前提:默认路径行为不变;所有扩展只作用于 `--kv-paged` 分支;由于 910B 内核问题
尚未解决,见第 11 节,运行时验收仍以 CPU/310P 为准。

### 12.0 前置小修与统一反馈契约

**P0,必须先做**:phase-1 的 paged launch 未检查 `task.tokens.has_mtmd`,多模态请求会被
`get_text_tokens()` 静默丢掉媒体内容,位置在 `server-scheduler-paged.cpp:124`。补一道闸,
使媒体请求在 b) 落地前明确返回未支持错误。约 10 行。

**统一反馈契约,一次设计到位**:由于 c) 需要 fork 出的子请求的批外反馈,a) 需要投机验证
一次接受多个 token 的每 seq 变长 token 列表,直接将 `llama_paged_scheduler_update` 的
签名改为 feedback 数组形式,不引入 `_v2` 新符号。理由:该 API 的调用方全部在树内且仅
4 处,即 examples/paged、tests/test-paged-kv-e2e、shim 内 warmup、server-scheduler-paged,
一次机械迁移即可;本 fork 新增 API 无外部消费者;llama.cpp 惯例是原位演进 C API 签名,
`_ext` 类后缀仅在两种能力需长期并存时使用,而新形式是严格超集,旧语义等价于每项
n_tokens = 1,无并存价值:

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

### 12.1 d) /v1/chat/completions,含工具调用

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

### 12.2 c) `n_cmpl > 1` 并行采样

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

- 实现要点:校验 src 已完成 prefill;复制 group,包括 `block_table` 引用与
  `add_ref_blocks`、`logical_seq` 截到 `n_prompt_total`、`n_past` 继承、`n_decoded = 0`,
  状态置 RUNNING 并按到达时间插入 running 列表;hybrid 模型补
  `hybrid_memory_manager->seq_cp`,与 GDN checkpoint fork 同机制,
  `llama-paged-scheduler-impl.cpp:215` 已有先例;重复 id 复用 `queue_request` 的拒绝逻辑;
- 尾块处理:若 `n_past` 不在块边界上,部分填充的尾块是多写者冲突点,因为父请求与全部
  子请求的下一个 decode 都要往该尾块的同一位置写入各自的 token。处理方式为**整块共享 +
  尾块急拷贝**:满块按引用共享,部分填充的尾块为每个子序列即时复制一份,父请求保留原尾块
  所有权;需要在 `llama_kv_cache_paged` 增加块内容拷贝入口,复用 swap 路径 `do_block_copy`
  的设备内拷贝,每个子请求只多一次单块拷贝。不选 vLLM 式 copy-on-write,因为调度器没有
  懒拷贝基础设施;也不选把 `n_past` 回退到块边界让子请求重 prefill 尾部,因为那会推迟
  同批首 token 采样;
- 子请求首 token:fork 发生在父槽 final_prefill 的同一 `update()` 迭代内,采样前 logits
  仍在 ctx 缓冲,所以子槽直接用父槽的末行下标采样,即
  `common_sampler_sample(child.smpl, ctx, tok_idx)`,经 `process_token` 后以批外 feedback
  回填。语义等价于默认路径的 `i_batch` 继承。

| 层 | 位置 | 内容 | 行数 |
|---|---|---|---|
| llama | `-impl.{h,cpp}` + shim + `llama.h` | `fork_request`,含 `llama_kv_cache_paged` 的块拷贝入口 | ~125-175 |
| llama | 同上 | 批外反馈路径,与 12.0 共享 | ~60-80 |
| server | `server-scheduler-paged.cpp` | launch:父任务需同时容纳全部子槽,空闲槽不足则整体 defer,对齐 `launch_slots_with_parent_task`;子槽 WAIT_OTHER 且不 `add_request` | ~80-110 |
| server | 同上 | update:父 final_prefill 时 fork + 子槽同批首采样 + stats/prompt 继承 + 批外 feedback | ~90-130 |
| server | 同上 | 释放与取消路径,子槽 finish 时块引用自然减引用;门禁移除 | ~30-40 |

合计约 385-535 行。风险:中。hybrid GDN fork 的 recurrent seq_cp 有现成机制;子槽首 token
依赖 logits 在同迭代内有效,该前提成立,因为下一次 decode 才会覆盖 logits;尾块急拷贝走
设备内单块复制,与 swap 同路径;finish 时父与子的尾块各自独立释放,无引用纠缠;n_parallel
需大于等于 n_cmpl,与默认路径一致,schema 层已限制。

### 12.3 a) 投机推理

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

- `attach_draft`:draft 暂存于 group,`populate_batch_from` 对该 seq 产出 `1 + n_draft` 行,
  块预算按 `allocate(n_draft + 1)` 预留,`batch.logits` 全 1,行位置为 `n_past + i`;行下标
  由 `batch_info` 的 offsets/lens 推导,等价默认路径的 `spec_i_batch`;
- 验证与回滚,在 server 侧 `update()` 内:先以 `common_sampler_clone` 备份,再用
  `sample_and_accept_n` 验证;`n_rollback > 0` 时执行
  `truncate(sched, id, n_past_before + accepted.size())`,并以 `common_sampler_copy` 恢复
  采样器、`prompt.tokens.keep_first` 收缩 prompt。由于 paged 支持位置级 seq_rm 加块收缩,
  无需 checkpoint,对齐默认路径 PART 分支;最后 update 反馈携带 accepted 列表;
- `update()` 内 draft 生成:上一轮采样后,按 `get_n_draft_max` 与 `spec_draft` 是否为空
  组装 `common_speculative_get_draft_params`,并经 host yield 钩子调
  `common_speculative_draft`;draft ctx 解码期间让 queue 保持响应,对齐默认路径
  `yield_to_queue`;
- `common_speculative_process(spec, batch)`:直接喂 paged 产出的验证批,seq_id 即槽位 id,
  与 draft ctx 的 n_seq_max 约定一致,无需改动;
- `common_context_can_seq_rm` 特判:由于探测式分类会对 paged ctx 断言失败,探测 decode 缺
  `set_paged_batch_info`,在 `common.cpp` 对 `kv_paged` 直接返回
  `COMMON_CONTEXT_SEQ_RM_TYPE_PART`;paged 支持范围 seq_rm,走最简单的无 checkpoint 回滚;
- 统计:`n_accepted_per_pos`、`spec_is_replay` 与 draft 计数全部沿用槽位字段,与默认路径
  同构。

| 层 | 内容 | 行数 |
|---|---|---|
| llama | `attach_draft` + populate 草稿行/全行 logits/块预算 | ~150-200 |
| llama | update 变长反馈语义,与 12.0 共享,含隐式回滚 | ~100-140 |
| llama | `truncate`:位置级 seq_rm + 块表收缩释放,需对齐块边界 | ~80-110 |
| llama | `can_seq_rm` paged 特判 | ~20-30 |
| server | 启动门禁放开,spec 加 kv_paged 允许;draft ctx 强制非 paged;n_outputs 预算沿用 `server_output_limits` | ~40-60 |
| server | update:draft 生成与 yield、验证、部分接受回滚、replay、逐位置统计 | ~250-350 |

合计约 600-890 行,四项中复杂度最高。风险:update 契约重构依赖 12.0 先行;块部分释放需
对齐 block_size 边界;`spec_is_replay` 语义迁移;draft ctx 与 paged 目标的流同步。
spec 加多模态的组合第一版禁用,默认路径支持,paged 侧两个特性都落地后再开。

### 12.4 b) 多模态:prefill 中途编码,对齐默认路径

默认路径:媒体块在 `server_tokens` 里是 `LLAMA_TOKEN_NULL` 哨兵行,index 与 position 不
相等,`n_tokens != n_pos`;prefill 遇哨兵即暂停 token 装批,经 `process_mtmd_chunk` 编码
后,由 `mtmd_helper_decode_image_chunk` 用自带的 `llama_decode` 把 embd 行写入 KV 并自行
管理位置,再以 `push_back_placeholder` 记账。该 helper 完全绕过调度器,与 paged 的
`write_slots/block_table` 路由不兼容,所以不能复用其解码半段,只复用编码半段,即
`mtmd_batch_*` 加 `mtmd_batch_get_output_embd`。

**阶段对齐**:两条路径跑在同一个循环阶段。server_queue 主循环每轮迭代经 on_update_slots
回调驱动一次推理:默认路径进入 `server_context_impl::update_slots`,子阶段为
pre_decode 装批 -> decode -> post_decode 采样;paged 路径进入
`server_scheduler_paged::update`,子阶段为 prepare_batch 装批 -> decode -> 采样/反馈。
所以默认路径的 pre_decode 装批与 paged 的 update 迭代是同一阶段的对应物,差异在回调
内部:装批归属不同,默认路径由 server 构建 `server_batch`,paged 路径由调度器
prepare_batch 构建、server 不接触批内容;媒体编码时点基本等价,两条路径的编码都落在
迭代 A 的 decode 与迭代 B 的 decode 之间,默认路径发生在 B 的 pre_decode 开头,paged
路径发生在 A 的迭代尾部,墙钟顺序几乎相同;真正多付的一个调度步来自媒体行的解码方式:
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

paged 方案:分段入队加外部暂停态。请求以分段骨架入队,媒体段先声明规模,编码前
`n_tokens/n_pos` 已知,编码数据后补;调度器把 embd 行当普通逻辑行调度,批构建时产出纯
embd 切片:

```c
struct llama_paged_segment {
    const llama_token * tokens;  // text segment: n_tokens ids
    const float *       embd;    // media segment: n_tokens rows x n_embd; NULL = pending
    int32_t             n_tokens;
    int32_t             n_pos;   // positions spanned; 0 = same as n_tokens (text)
};

// media segments may pass embd = NULL (declared size only): the request pauses
// at that boundary until append_segment supplies the encoder output
LLAMA_API bool llama_paged_scheduler_add_request_segs(struct llama_paged_scheduler * sched,
                                                       const struct llama_paged_segment * segs,
                                                       int32_t n_segs, int32_t request_id);
LLAMA_API bool llama_paged_scheduler_append_segment(struct llama_paged_scheduler * sched,
                                                     int32_t request_id,
                                                     const struct llama_paged_segment * seg);
```

paged 路径流程,同一 [text1][img][text2] 请求,与默认路径逐迭代对照:

```
launch
  -> add_request_segs
       text1 段:tokens 实数据
       img 段:  embd = NULL,仅声明 n_tokens/n_pos          pending 声明
       text2 段:tokens 实数据

queue 循环迭代 A -> on_update_slots -> server_scheduler_paged::update
  +-- prepare_batch: 装批,调度器拥有批构建
  |     text1 段可跑 -> 产出 token 切片
  |     到达 img 段边界且 embd 为 NULL -> 请求置 PAUSED_EXTERNAL
  |       不进候选,不计入 check_deadlock 分母
  |       其余请求照常组批,互不阻塞
  +-- decode: llama_decode(本批切片)                     text1 段行
  +-- 采样/反馈: text1 为中途 prefill 块,无 logits、无反馈
  +-- 迭代尾部: server 查 seq_state,发现该请求 PAUSED_EXTERNAL
        经 host yield 钩子编码:mtmd_batch_encode -> mtmd_batch_get_output_embd
        append_segment(id, embd):拷入调度器自有缓冲,请求回 RUNNING

queue 循环迭代 B -> on_update_slots -> update
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
  亚毫秒级),相对图片编码的几十到几百毫秒可忽略;
- 编码批处理两路径等价:mtmd_batch_encode 在 vision 上下文执行,默认路径会把同请求
  后续图片贪婪攒入一个 encoder 批;paged 的迭代尾部编码复用同一逻辑,同样可一次编完
  该请求全部 pending 媒体段,编码次数与耗时相同,与节拍延迟无关;
- 并发场景:循环节拍由在跑 decode 的批决定,媒体请求的 TTFT 约延后一个节拍,即系统
  当前 TPOT,典型 10-50ms 量级;n 张图片累积 n 个节拍,原因在解码侧而非编码侧:图片
  之间隔着文本段时,受位置顺序约束,后面图片的 embd 行要等中间文本行写入 KV 才能
  计算,非相邻 embd 段无法合并,每个 embd 段独占一步,即使编码已一次完成;相对编码
  与 prefill 总耗时通常为个位数百分比;
- 吞吐不受影响,并发多模态时甚至更优:默认路径的 helper 私有 decode 逐槽串行、永远
  单请求;paged 的 embd 步只要求同型、不限请求,多个请求的 pending embd 段可合入
  同一调度步。n 节拍惩罚只作用于单个请求自身的 TTFT,系统总计算量不变;
- 若需消除该步,后续可评估同一轮内连续两次 llama_decode(先 embd 后文本)或内核支持
  混合 ubatch,第一版不做。

- 序列模型:`llama_sequence_group` 增加分段描述与每行位置数组,打破现有"序号等于位置"的
  假设,`pos = n_past + token_idx` 改为查表;`write_slots` 按位置算块,天然兼容;
- PAUSED_EXTERNAL 调度态:到达 pending 段边界即暂停,不进候选、不计入死锁/活锁检测,
  `check_deadlock` 分母排除;`append_segment` 供给 embd 后回到 WAITING/RUNNING,embd 拷入
  调度器自有缓冲以规避 mbatch 生命周期;`llama_paged_seq_state` 增加状态枚举供 server
  检测暂停;
- 批构建:同段同型才可同批,因为文本 token 与 embd 行不混批,`build_inp_embd` 只在
  `ubatch.token == nullptr` 时走 embd 路径;embd 切片中 `batch.token = nullptr`、
  `batch.embd` 指向段缓冲;logits 仍只在最终段末行;
- server 侧:`update()` 每轮迭代检查暂停请求,经 host yield 钩子编码,mtmd vision 上下文
  独立,期间其余请求照常跑批,然后 `append_segment` 恢复;`prompt.tokens` 以 placeholder
  记账,`pos_next()` 仍是位置权威;embedding/rerank 依旧禁用,`need_embd` 与 paged 池化
  问题相互独立。

| 层 | 内容 | 行数 |
|---|---|---|
| llama | 序列分段模型 + 每行位置数组 | ~120-160 |
| llama | `add_request_segs` / `append_segment`,含 pending 声明与 embd 缓冲 | ~80-110 |
| llama | PAUSED_EXTERNAL 状态机 + seq_state 状态枚举 + 死锁排除 | ~60-90 |
| llama | populate 纯 embd 切片 + 段边界不混批 + logits 规则 | ~120-170 |
| server | host yield 钩子 | ~15 |
| server | launch:分段骨架入队,复用 `find_chunk` 遍历 | ~50-70 |
| server | update:暂停检测 -> 编码 -> 恢复,复用 `process_mtmd_chunk` 编码半段 | ~130-180 |
| server | placeholder 记账 + 释放路径 | ~60-80 |

合计约 535-875 行,风险最高。风险点:paged 图的 embd 输入分支,`build_inp_embd` 选择逻辑
内核侧待验证;M-RoPE 每行多位置,第一版可按架构闸掉,仅支持单位置媒体;编码期间调度空转
的边界,其余请求全部空闲且唯一请求暂停时,update 返回空闲并由编码驱动唤醒。

### 12.5 特性交互与实施顺序

| 组合 | 默认路径 | paged 第二阶段 |
|---|---|---|
| d + 任意 | - | 无冲突 |
| c + a,n_cmpl 加投机 | 支持,fork 不复制 spec 状态,子槽独立起草 | 支持,fork 同理只复制 prompt 段 |
| b + a,多模态加投机 | 支持 | 第一版禁,因为 draft prompt 的媒体段文本化有问题,后开 |
| b + c | 支持 | 支持,fork 含媒体段,块引用共享不受段型影响 |

实施顺序:P0 -> d,放闸即刻可用 -> c,fork 基础设施与契约重构 -> a,复用新契约,工作量
最大 -> b,独立、可与 a 并行开发,运行时验收依赖内核 embd 支持。12.0 的契约重构必须先于
c/a 落地。

### 12.6 工作量汇总

| 特性 | llama 层 | server 层 | 合计 | 风险 |
|---|---|---|---|---|
| P0 + d) chat/工具调用 | 0 | ~25-35 | ~25-35 | 低 |
| c) n_cmpl>1 | ~185-255 | ~200-280 | ~385-535 | 中 |
| a) 投机推理 | ~350-480 | ~290-410 | ~600-890 | 高 |
| b) 多模态 | ~380-530 | ~255-345 | ~535-875 | 最高 |
| **合计** | **~915-1265** | **~770-1070** | **~1545-2335** | - |

验收沿用第 9 节框架,按特性补矩阵:chat 工具调用流式增量、n_cmpl 多路种子一致性,
与默认路径同 prompt 对比;投机接受率与无投机等价性,top-k 对比复用 `test-paged-kv-e2e`
思路;多模态图文混合 prompt 与默认路径逐 token 等价。

### 12.7 第二阶段实施记录

范围:P0 + 12.1 + 12.2 + 12.3;12.4 多模态按计划暂缓。

实现与设计偏差:

1. `update` 契约按 12.0 修订原位改签名,4 处树内调用点(examples/paged、e2e 测试两处、
   shim 内 warmup)随签名迁移。
2. `can_seq_rm` 的 paged 特判落在 `server-context.cpp` 的 `ctx_tgt_seq_rm_type` 赋值处
   (以 `params_base.kv_paged` 判 PART),而非 `common.cpp`:common 层只有 C API,看不到
   cparams,server 侧判断零成本且避免新增公共接口。该特判同时修复了一个 phase-1 隐患:
   探测式分类会对 paged ctx 断言失败,此前 server 带 `--kv-paged` 启动会在 load 阶段崩溃。
3. 投机推理走 PART 语义:部分接受经 `prompt.tokens.keep_first + insert` 修复后直接
   `truncate` 到新的验证末尾,无 checkpoint、无 replay(`spec_is_replay` 恒为 false),
   与默认路径 PART 分支同构;synthetic acceptance 在 `--kv-paged` 下启动期拒绝。
4. 验证采样复用 `common_sampler_sample_and_accept_n`,不在 server 侧重写逐行采样。
5. fork 的尾块急拷贝:满块 `add_ref_blocks` 引用共享,`n_past % block_size != 0` 时为
   子请求分配私有块并经新增的 `copy_block_content`(同池直拷/跨池走 `do_block_copy`)
   复制父尾块;`truncate` 释放块时调用新增的 `prefix_cache.drop_block` 撤销已释放块的
   前缀缓存登记,避免 LRU 驱逐命中陈旧块号。

验证:CANN 9.0.0 aarch64 编译通过(llama-server + test-paged-kv-e2e);910B4 上默认
调度回归通过(/completion、/v1/completions、/v1/chat/completions 含工具调用、n=2 并行)。
paged 路径运行时验收依赖第 11 节的 910B 内核适配,当前以 CPU/310P 为准。
