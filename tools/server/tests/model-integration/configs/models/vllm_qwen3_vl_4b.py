from ais_bench.benchmark.models import VLLMCustomAPIChatStream


models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr="vllm-qwen3-vl4b",
        path=".",
        model="",
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=18120,
        max_out_len=128,
        batch_size=1,
        trust_remote_code=False,
        api_key="EMPTY",
        generation_kwargs=dict(
            temperature=0.0,
            ignore_eos=True,
        ),
    )
]
