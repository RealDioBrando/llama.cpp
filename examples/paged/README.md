# llama.cpp/example/paged

Minimal end-to-end demo of the paged KV cache and continuous-batching scheduler. Queues N requests up front, then runs a loop of `llama_paged_scheduler_prepare_batch` → `llama_decode` → `llama_paged_scheduler_update` until every request finishes.

## Example

Run 10 sequences in parallel with greedy sampling, up to 50 generated tokens per sequence on a single GPU:

```bash
llama-paged -m model.gguf -kvp -ngl 100 -sm none -mg 0 \
            -ns 10 -np 10 -n 50 -b 512 -ub 512 \
            -ngpub 500 -ncpub 100
```

- `-kvp` / `--kv-paged`: enable the paged KV cache (required)
- `-ngpub N`: number of GPU KV blocks to allocate (auto-fitted from free VRAM if omitted)
- `-ncpub N`: number of CPU KV blocks for swap-out (optional)
- `-ns N`: number of sequences to queue
- `-np N`: maximum sequences decoded in parallel (must equal `-ns` in this example)
- `--no-warmup`: skip the paged prefill/decode warmup; warmup is enabled by default

## Request priority (optional)

By default the scheduler is strict FCFS. Pass `--enable-priority` (or set
`LLAMA_ARG_ENABLE_PRIORITY=1`) to order the waiting/swapped queues by request
priority instead. The mode is fixed when the scheduler is created; the per-request
`priority` argument only carries the value.

Semantics, matching vLLM:

- `priority` is an `int32_t`, default `0`. **Smaller values are served first**
  (`-10` is urgent, `0` is normal, `10` is background). Values are clamped to
  `[-1000000, 1000000]`.
- Equal priorities are served in arrival order.
- Scheduling is strict head-of-queue: if the head request cannot be admitted
  (not enough KV blocks) or swapped in, that queue makes no progress this step.
  A lower-priority request behind the head is **not** served out of order.
- Priority does not preempt a running request. It affects admission, swap-in and
  (in priority mode) which running request is evicted first: the lowest-priority
  request (largest value) is sacrificed. Under `FCFS` the most recent arrival is
  sacrificed, as before.
- Priority is compared per phase only: prefills are drained before decodes, and
  the `running -> swapped -> waiting` stage order is unchanged. A high-priority
  request can still be delayed by a large low-priority prefill that is already
  running or by the strict head-of-queue rule; "higher priority" is not a
  guarantee that it is never delayed.
- In FCFS mode a non-zero `priority` is recorded but ignored (the scheduler logs
  one warning).

The example assigns priorities with two environment knobs (only read when
`--enable-priority` is active):

```text
LLAMA_PAGED_PRIORITY         default 0   base priority for every request
LLAMA_PAGED_PRIORITY_STEP    default 0   priority = base - seq_id * step
                                        (step > 0 makes later arrivals higher priority)
```

Combine with `LLAMA_PAGED_SEQ_DELAY_MS` to stagger arrivals and observe the
effect on TTFT:

```bash
# Later arrivals are more urgent: priorities 0, -1, -2, ...
LLAMA_PAGED_SEQ_DELAY_MS=50 LLAMA_PAGED_PRIORITY_STEP=1 \
  llama-paged -m model.gguf -kvp --enable-priority -ngl 100 -sm none -mg 0 \
              -ns 10 -np 10 -n 50 -b 512 -ub 512 -ngpub 500 -ncpub 100
```

Drop `--enable-priority` for the FCFS control run. The summary prints a
`priority` column and the average TTFT per priority.

## Phase 1 restrictions
The paged path currently requires:
- Single device (fully loaded on one GPU or CPU). Use `-sm none -mg <id>` to pin to one GPU on multi-GPU machines.
- Full offload of the model (`-ngl` must cover all layers).
- `-b == -ub` (batch and ubatch sizes must match).

Note: SWA architectures (gemma3, llama4, etc.) are not yet supported.

## Output
- Prints all the generated outputs at the end.
- Prints performance timings (general and per-request).

## Performance benchmark

Use the fixed Paged vs Unified performance case when comparing KV cache modes:

```bash
python3 scripts/perf-paged-vs-unified.py \
  --model /app/gguf-models/qwen3.5-2b-f16.gguf \
  --bin-dir build-cann/bin \
  --device 5
```

The case runs single-request and 10-request workloads, captures per-run logs, and writes `results.json` and `results.md`. Use `--scenarios paged-single unified-single` for a short smoke run and `--dry-run` to inspect the exact commands.

The case discards warmups per scenario, alternates Paged and Unified ordering, waits for host and NPU idle state, adds a quiet gap between runs, and reports median values. It records a relative-range stability score; pass `--require-stable` to turn an unstable result into a failure.
