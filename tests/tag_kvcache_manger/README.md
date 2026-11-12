# tag KV block 统计与驱逐策略：测试

对应设计文档：[`docs/kvcache-manager/tag-kv-block-usage.md`](../../docs/kvcache-manager/tag-kv-block-usage.md)。

本目录包含：

1. **服务化（HTTP）端到端测试**：启动真实 `llama-server --kv-paged`，验证 tag → KV block
   的统计/查询口径，以及 tag 保护型驱逐策略下的 prefix 命中；
2. **C++ 单元/集成测试**：`test-paged-evict-policy.cpp`，直接测块管理器与调度器，
   不需要模型。

服务化测试通过公共接口验证 tag → KV block 的统计与查询口径。核心实现位于
`src/llama-paged-tag-manager.*`（tag 聚合/查询）、`src/llama-block-manager.*`
（物理块 owner tag）、`src/llama-paged-scheduler-impl.*`（请求 tag 与快照）、
`tools/server/server-scheduler-paged.cpp` 与 `server-context.cpp`（HTTP 端点）。

## 被测接口

| 接口 | 说明 |
| --- | --- |
| `POST /completion`、`POST /v1/completions`、`POST /v1/chat/completions` | 请求体可带 `tag`（JSON 整数，uint64，0 = 无 tag）；paged 模式下在首次调度前写入 owner tag |
| `GET /kv_stats` | `npu.full` / `npu.linear` 两块池的 `total/used/free`；`cpu.*` 恒为 0 |
| `GET /tag_usage` | 全量 tag 计数（默认不带 id 列表） |
| `GET /tag_usage?tag=<uint64>` | 单 tag 计数 + `blocks`（默认带 id） |
| `GET /tag_usage?blocks=1` | 全量列表附带每个 tag 的 id 列表 |
| `GET /slots` | 处理中的请求在 `params.tag` 暴露 tag |

## 测试覆盖

| 用例 | 断言 |
| --- | --- |
| `test_kv_stats_initial_consistency` | 初始 `used+free==total`；`cpu.*` 全 0；warmup 后 `used==0` |
| `test_tag_counts_owned_reusable_blocks` | 请求结束后 `full_blocks>0`、`len(owned)==full_blocks`、id 升序去重、`shared` 为空；`used>=full_blocks` |
| `test_tag_linear_gpu_checkpoint` | hybrid 模型 `linear_blocks>=1` 且等于 `linear.owned` 长度；full/linear id 空间不相交（非 hybrid 自动 skip） |
| `test_unknown_tag_is_zero` | 未知 tag 计 0、id 为空、HTTP 200 |
| `test_tag_listing_and_blocks_flag` | 列表默认无 `blocks`，`?blocks=1` 时与单 tag 查询一致；每个 tag 的 `full_blocks<=used`（会员制计数，允许跨 tag 重复） |
| `test_shared_prefix_is_counted_for_both_tags` | 命中方复用的块同时出现在自己和 owner 的 `owned` 里；正常路径 `shared` 为空（tag 已并入集合）；owner 归属保持不变 |
| `test_oai_endpoints_accept_tag` | `/v1/completions`、`/v1/chat/completions` 的 tag 生效 |
| `test_slots_expose_request_tag` | `/slots` 的 `params.tag` 可见 |
| `test_non_uint64_tag_is_rejected` | 字符串 tag / 非数字 `?tag=` 返回 400 `invalid_request_error` |
| `test_kv_stats_consistency_after_activity` | 活动后池统计仍自洽 |
| `test_non_paged_server_reports_not_supported` | 非 `--kv-paged` 服务返回 501 `not_supported_error` |

## 前置条件

1. 已编译带 paged 支持的 `llama-server`（本仓库默认 `build/bin/llama-server`）：

   ```bash
   cmake --build build --target llama-server -j 64
   ```

2. 一个本地 GGUF 模型（Ascend 上使用 F16 权重）：
   ```bash
   export TAG_TEST_MODEL=/home/qwen35-2b-f16.gguf   # 或 PAGED_TEST_MODEL
   ```
3. Ascend 环境：脚本会自动 source `$ASCEND_TOOLKIT_HOME/set_env.sh`
   （默认 `/usr/local/Ascend/cann-9.1.0`）；未设置时直接使用当前环境。

paged 路径要求**单设备全卸载**（`-sm none -mg 0`）且不接受默认
`--cache-ram`，测试里已自动带上 `--kv-paged -ngl 99 -sm none -mg 0
--cache-ram 0 -kvbls 16 -np 4 -c 8192 --slots`。

## 运行

```bash
# 全量（约 1.5-2 分钟，含两次模型加载）
python3 -m pytest -v tests/tag_kvcache_manger/test_tag_kv_server_e2e.py

# 只跑共享前缀用例
python3 -m pytest -v tests/tag_kvcache_manger/test_tag_kv_server_e2e.py \
    -k shared_prefix

# 指定二进制 / 模型
LLAMA_SERVER_BIN=/path/to/llama-server \
TAG_TEST_MODEL=/path/to/model.gguf \
python3 -m pytest -v tests/tag_kvcache_manger/test_tag_kv_server_e2e.py
```

## 口径说明（断言为什么这样写）

- **只统计可复用块**：full 侧是 prefix index 中 live（GPU、ref>0、generation
  匹配）的块；linear 侧是 live 的 GPU checkpoint。partial tail、linear ACTIVE
  state、CPU 池、swap 出去的块都不计数，所以计数是实际物理占用的下界。
- **owned 是 tag 集合的会员关系**：一个块在 cache-resident 期间被多个 tag 使用就会
  计入每个 tag 的 `owned`，因此 `sum(tag.full_blocks)` 可以大于 `npu.full.used`；
  `blocks.full.shared` 只是“引用的块集合不含该 tag”时的兜底口径，正常命中复用
  会把 tag 并入集合，shared 通常为空。
- **hybrid 模型的前缀复用受 checkpoint 限制**：默认
  `--kv-prefix-ckpt-interval 4`、block=16，前缀至少要覆盖 64 token 才能命中；
  因此测试 prompt 均长于 64 token，纯 full-attention 模型会跳过 linear 断言。
- **归属可能随重新注册变化**：命中方从 checkpoint 边界之后重算的块会用
  自己的物理块重新注册同一 hash 链，原块的计数可能下降，这是设计行为；
  复用用例只断言“复用的 id 仍属于 owner”，不假设 owner 总数不变。
- 测试通过 `/kv_stats` 与 `/tag_usage` 的交叉约束（`used+free==total`、
  单 tag `full_blocks<=used`、`len(owned)==count`）自校验，不读取内部状态。


## 驱逐策略 e2e

`test_paged_evict_policy_e2e.py` 通过环境变量
`LLAMA_SERVER_PAGED_EVICT_PROTECT_TAGS=1` 安装内置的 tag 保护策略：

1. 用 tag `keep` 发送长 prompt，生成可复用 prefix；
2. 用多个不同 tag 的请求填满小块池，强制淘汰；
3. 重新发送 `keep` prompt，断言 `timings.cache_n >= 64`（prefix 仍命中），
   并至少有一个压力 tag 的缓存块已被淘汰。

```bash
TAG_TEST_MODEL=/home/qwen35-2b-f16.gguf \
  python3 -m pytest -v tests/tag_kvcache_manger/test_paged_evict_policy_e2e.py
```
