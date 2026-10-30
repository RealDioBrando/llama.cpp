"""Run small acceptance and first-block capture controls outside the CI test suite."""

import json
import signal
import sys
import time

import benchmark


def check_capture_model(path):
    sys.path.insert(0, str(benchmark.ROOT / 'gguf-py'))
    import gguf

    reader = gguf.GGUFReader(str(path))
    tensors = {t.name: t for t in reader.tensors}
    architecture = reader.fields['general.architecture'].contents()
    if (architecture != 'dflash' or tuple(tensors['fc.weight'].data.shape) != (2560, 12800)
            or reader.fields['dflash.block_count'].contents() != 5
            or 'dflash.rope.dimension_sections' not in reader.fields):
        raise ValueError('capture supports the documented five-tap Qwen3-VL draft only')


def main():
    parser = benchmark.parser()
    parser.description = __doc__
    parser.add_argument('--variants', nargs='+', default=['batch128', 'batch1024'],
                        choices=['batch128', 'batch1024', 'legacy1d', 'image256', 'capture3d', 'capture1d'])
    parser.add_argument('--legacy-draft', type=benchmark.Path)
    parser.add_argument('--capture-library', type=benchmark.Path)
    args = parser.parse_args()
    if any(v in args.variants for v in ('legacy1d', 'capture1d')) and not args.legacy_draft:
        parser.error('--legacy-draft is required for legacy controls')
    if any(v.startswith('capture') for v in args.variants) and not args.capture_library:
        parser.error('--capture-library is required for numerical capture')
    benchmark.configure(args)
    if any(v.startswith('capture') for v in args.variants):
        check_capture_model(benchmark.DRAFT)
        if 'capture1d' in args.variants:
            check_capture_model(args.legacy_draft)
    work = benchmark.WORK
    if args.prepare_only:
        return
    image_cases = benchmark.prepare('selection.jsonl')[:3]
    for variant in args.variants:
        options = {}
        if variant == 'batch1024':
            options.update(batch=1024, ubatch=1024)
        if variant in ('legacy1d', 'capture1d'):
            options['draft'] = args.legacy_draft.resolve()
        if variant == 'image256':
            options['image_tokens'] = 256
        capture = variant.startswith('capture')
        if capture:
            options.update(capture_library=args.capture_library, capture_width=12800)
        cases = []
        if variant == 'batch128':
            for name, prompt in [('text_numbers', 'List the integers from 1 to 60, separated by commas.'),
                                 ('text_explanation', 'Explain how photosynthesis works in five clear sentences.'),
                                 ('text_code', 'Write a short Python function that computes the Fibonacci numbers up to n.')]:
                cases.append((name, {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 128,
                                     'stream': True, 'stream_options': {'include_usage': True},
                                     'temperature': 0, 'top_k': 1, 'top_p': 1, 'seed': args.seed,
                                     'cache_prompt': False, 'id_slot': 0, 'return_tokens': True, 'verbose': True}))
        for item, payload, _ in image_cases[:1] if capture else image_cases:
            payload = dict(payload)
            if capture:
                payload['max_tokens'] = 32
            cases.append(('image_' + str(item['question_id']), payload))
        directory = work / variant
        directory.mkdir()
        process, log, session, url = benchmark.launch('dflash', directory, **options)
        try:
            if capture:
                (directory / 'capture').mkdir(exist_ok=True)
                (directory / 'capture/ARMED').write_text('ready', encoding='utf-8')
            with (directory / 'requests.jsonl').open('w', encoding='utf-8') as output:
                for index, (name, payload) in enumerate(cases):
                    before = benchmark.metrics(session, url)
                    result = benchmark.request(session, url, payload)
                    after = benchmark.metrics(session, url)
                    counters = {key: after[key] - before[key] for key in before}
                    result.update({'case': name, 'variant': variant, 'counters': counters})
                    result['accepted_per_round'] = counters['accepted'] / counters['rounds'] if counters['rounds'] else None
                    output.write(json.dumps(result) + '\n')
                    output.flush()
                    progress = {'variant': variant, 'completed': index + 1, 'total': len(cases), 'case': name,
                                'accepted_per_round': result['accepted_per_round'], 'counters': counters}
                    benchmark.save(work / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            if capture and not (directory / 'capture/noise.json').is_file():
                raise RuntimeError('capture library did not produce a first block; inspect server.log')
        finally:
            benchmark.stop(process)
            log.close()
            session.close()
            benchmark.save(directory / 'stopped.json', {'pid': process.pid, 'stopped': process.poll() is not None})
    benchmark.save(work / 'complete.json', {'completed_at': time.time()})


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, benchmark.interrupted)
    main()
