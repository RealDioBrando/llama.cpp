from ais_bench.benchmark.models import VLLMCustomAPIChatStream


models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr="llama-qwen3-17b",
        path="/home/c30081098/models/Qwen3-1.7B",
        model="",
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=18112,
        max_out_len=2048,
        batch_size=10,
        trust_remote_code=False,
        api_key="EMPTY",
        generation_kwargs=dict(
            temperature=0.0,
            ignore_eos=True,
        ),
    )
]
