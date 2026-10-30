"""Serial paired VL benchmark with real SSE TTFT and per-request evidence."""

import argparse
import base64
import csv
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import random
import re
import signal
import socket
import statistics
import subprocess
import threading
import time
import traceback
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[5]
WORK: Path
BINARY: Path
TARGET: Path
MMPROJ: Path
DRAFT: Path
PLAN: dict[str, Any] = {}
STATE: dict[str, Any] = {}
STOP = threading.Event()
OWNED = []


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--server', type=Path, required=True)
    result.add_argument('--target', type=Path, required=True)
    result.add_argument('--mmproj', type=Path, required=True)
    result.add_argument('--draft', type=Path, required=True)
    result.add_argument('--dataset', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--repo-root', type=Path, default=ROOT)
    result.add_argument('--visible-devices', help='One physical Ascend device ID; omit to keep the caller mask')
    result.add_argument('--device', default='CANN0')
    result.add_argument('--mmproj-device', default=None)
    result.add_argument('--samples', type=int, default=200)
    result.add_argument('--warmups', type=int, default=5)
    result.add_argument('--selection-seed', type=int, default=20261029)
    result.add_argument('--seed', type=int, default=1234)
    result.add_argument('--draft-max', type=int, default=15)
    result.add_argument('--max-tokens', type=int, default=64)
    result.add_argument('--context', type=int, default=4096)
    result.add_argument('--batch', type=int, default=128)
    result.add_argument('--ubatch', type=int, default=128)
    result.add_argument('--threads', type=int, default=4)
    result.add_argument('--image-tokens', type=int, default=1024)
    result.add_argument('--prepare-only', action='store_true')
    result.add_argument('--monitor-npu', action='store_true')
    return result


def configure(args):
    global ROOT, WORK, BINARY, TARGET, MMPROJ, DRAFT, PLAN, STATE
    for name in ('samples', 'draft_max', 'max_tokens', 'context', 'batch', 'ubatch', 'threads', 'image_tokens'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.warmups < 0 or args.ubatch > args.batch:
        raise ValueError('warmups must be nonnegative and ubatch must not exceed batch')
    if args.draft_max > 15:
        raise ValueError('these controls use a trained block size of 16; draft-max must not exceed 15')
    ROOT = args.repo_root.resolve()
    output = args.output.resolve()
    BINARY, TARGET, MMPROJ, DRAFT = (p.resolve() for p in (args.server, args.target, args.mmproj, args.draft))
    for path in (BINARY, TARGET, MMPROJ, DRAFT, args.dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset = args.dataset.resolve()
    data = dataset.read_bytes()
    rows = [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    if args.samples + args.warmups > len(rows):
        raise ValueError('not enough distinct rows for samples and warmups')
    rng = random.Random(args.selection_seed)
    chosen = rng.sample(range(len(rows)), args.samples)
    warmup = rng.sample([i for i in range(len(rows)) if i not in set(chosen)], args.warmups)
    output.mkdir(parents=True, exist_ok=False)
    WORK = output
    (WORK / 'evidence').mkdir()
    for name, indices in [('selection.jsonl', chosen), ('warmup.jsonl', warmup)]:
        selected = []
        for index in indices:
            item: dict[str, Any] = dict(rows[index], dataset_index=index)
            if not isinstance(item['prompt'], str) or not isinstance(item['image_files'], list):
                raise ValueError(f'invalid dataset row {index}')
            item.setdefault('question_id', index)
            item['image_files'] = [str((dataset.parent / p).resolve()) for p in item['image_files']]
            for path in item['image_files']:
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            selected.append(item)
        (WORK / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected), encoding='utf-8')
    PLAN = dict(vars(args))
    PLAN.update({'dataset': str(dataset), 'dataset_sha256': hashlib.sha256(data).hexdigest(),
                 'total_rows': len(rows), 'selection_indices': chosen, 'warmup_indices': warmup,
                 'mmproj_device': args.mmproj_device or args.device, 'concurrency': 1,
                 'server': str(BINARY), 'target': str(TARGET), 'mmproj': str(MMPROJ), 'draft': str(DRAFT)})
    PLAN = {k: str(v) if isinstance(v, Path) else v for k, v in PLAN.items()}
    STATE = {'stage': 'preparing', 'phase': None, 'completed': 0, 'total': len(chosen)}
    save(WORK / 'plan.json', PLAN)


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def progress(**values):
    STATE.update(values)
    STATE['updated_at'] = time.time()
    save(WORK/'progress.json', STATE)


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered)-1)*fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


def prepare(filename):
    result = []
    for line in (WORK/filename).read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        content = [{'type': 'text', 'text': item['prompt']}]
        digests = []
        for name in item['image_files']:
            path = Path(name)
            data = path.read_bytes()
            digests.append(hashlib.sha256(data).hexdigest())
            mime = mimetypes.guess_type(path.name)[0] or 'image/jpeg'
            content.append({'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,'+base64.b64encode(data).decode()}})
        payload = {'messages': [{'role': 'user', 'content': content}], 'stream': True,
                   'stream_options': {'include_usage': True}, 'max_tokens': PLAN['max_tokens'],
                   'temperature': 0, 'top_k': 1, 'top_p': 1, 'seed': PLAN['seed'],
                   'cache_prompt': False, 'id_slot': 0, 'return_tokens': True, 'verbose': True}
        result.append((item, payload, digests))
    return result


def metrics(session, url):
    response = session.get(url+'/metrics', timeout=10)
    response.raise_for_status()
    result = {}
    for name, suffix in [('drafted', 'num_draft_tokens'), ('accepted', 'num_accepted_tokens'), ('rounds', 'num_drafts')]:
        values = re.findall(r'^[^#\s]*spec_decode_'+suffix+r'_total(?:\{[^\n]*\})?\s+([0-9.eE+-]+)\s*$', response.text, re.M)
        assert len(values) == 1, (name, values)
        result[name] = int(float(values[0]))
    return result


def request(session, url, payload) -> dict[str, Any]:
    start = time.perf_counter()
    first = None
    fragments, finish = [], None
    usage, timings, verbose = {}, {}, {}
    chunks = 0
    done = False
    # Read complete SSE lines without requests' default 512-byte buffering delay.
    with session.post(url+'/v1/chat/completions', json=payload, stream=True, timeout=(10, 180),
                      headers={'Accept': 'text/event-stream', 'Accept-Encoding': 'identity'}) as response:
        if response.status_code != 200:
            raise RuntimeError(f'HTTP {response.status_code}: {response.text[:2000]}')
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True
                break
            part = json.loads(data)
            chunks += 1
            if 'error' in part:
                raise RuntimeError(str(part['error']))
            for choice in part.get('choices', []):
                text = choice.get('delta', {}).get('content')
                if text:
                    if first is None:
                        first = time.perf_counter()
                    fragments.append(text)
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
            usage = part.get('usage', usage)
            timings = part.get('timings', timings)
            verbose = part.get('__verbose', verbose)
    end = time.perf_counter()
    assert done and usage and timings and 'tokens' in verbose, (done, usage, timings, list(verbose))
    assert timings['cache_n'] == 0, f'prompt cache was reused: {timings}'
    return {'ok': True, 'wall_seconds': end-start, 'ttft_seconds': first-start if first is not None else None,
            'content': ''.join(fragments), 'tokens': verbose['tokens'], 'stop_type': verbose.get('stop_type'),
            'finish_reason': finish, 'usage': usage, 'timings': timings, 'sse_chunks': chunks}


def summarize(records: list[dict[str, Any]], wall: float) -> dict[str, Any]:
    good = [r for r in records if r.get('ok')]
    latencies = [r['wall_seconds'] for r in good]
    ttfts = [r['ttft_seconds'] for r in good if r['ttft_seconds'] is not None]
    output = sum(r['usage']['completion_tokens'] for r in good)
    decoded = sum(max(r['timings']['predicted_n'] - 1, 0) for r in good)
    decode_seconds = sum(r['timings']['predicted_ms'] for r in good)/1000
    counters = {name: sum(r['counters'][name] for r in good) for name in ['drafted', 'accepted', 'rounds']}
    return {'requests': len(records), 'successes': len(good), 'failures': len(records)-len(good),
            'phase_wall_seconds': wall, 'request_wall_seconds': sum(latencies),
            'latency_mean_s': statistics.mean(latencies) if latencies else None,
            'latency_p50_s': percentile(latencies, .5), 'latency_p95_s': percentile(latencies, .95),
            'ttft_mean_s': statistics.mean(ttfts) if ttfts else None, 'ttft_p50_s': percentile(ttfts, .5),
            'ttft_p95_s': percentile(ttfts, .95), 'ttft_samples': len(ttfts),
            'output_tokens': output, 'output_tokens_per_second': output/wall if wall else None,
            'requests_per_second': len(good)/wall if wall else None,
            'server_decode_tokens': decoded, 'server_decode_seconds': decode_seconds,
            'server_decode_tokens_per_second': decoded/decode_seconds if decode_seconds else None,
            'length_limited': sum(r['finish_reason']=='length' for r in good), 'counters': counters,
            'draft_acceptance_rate': counters['accepted']/counters['drafted'] if counters['drafted'] else None,
            'accepted_draft_tokens_per_round': counters['accepted']/counters['rounds'] if counters['rounds'] else None,
            'mean_acc_len': 1 + counters['accepted']/counters['rounds'] if counters['rounds'] else None,
            'output_tokens_per_draft_round': output/counters['rounds'] if counters['rounds'] else None}


def launch(mode, directory, *, batch=None, ubatch=None, draft=None, image_tokens=None,
           capture_library=None, capture_width=None):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command: list[str] = [str(BINARY), '-m', str(TARGET), '--mmproj', str(MMPROJ),
               '--host', '127.0.0.1', '--port', str(port), '--device', PLAN['device'],
               '--mmproj-device', PLAN['mmproj_device'], '--split-mode', 'none', '-ngl', '999',
               '-c', str(PLAN['context']), '-b', str(batch or PLAN['batch']), '-ub', str(ubatch or PLAN['ubatch']),
               '-np', '1', '-t', str(PLAN['threads']), '-tb', str(PLAN['threads']),
               '-fa', 'off', '--fit', 'off', '--jinja', '--reasoning', 'off', '--offline', '--metrics',
               '--cache-ram', '0', '--no-cache-prompt', '--no-cache-idle-slots',
               '--cache-type-k', 'f16', '--cache-type-v', 'f16',
               '--image-min-tokens', str(image_tokens or PLAN['image_tokens']),
               '--image-max-tokens', str(image_tokens or PLAN['image_tokens'])]
    if mode == 'dflash':
        command += ['-md', str(draft or DRAFT), '--spec-type', 'draft-dflash',
                    '--spec-draft-device', PLAN['device'], '-ngld', '999', '--spec-draft-n-min', '1',
                    '--spec-draft-n-max', str(PLAN['draft_max']), '--spec-draft-p-min', '0',
                    '--spec-draft-type-k', 'f16', '--spec-draft-type-v', 'f16']
    else:
        command += ['--spec-type', 'none']
    env: dict[str, str] = {**os.environ, 'OMP_NUM_THREADS': str(PLAN['threads']), 'OPENBLAS_NUM_THREADS': str(PLAN['threads']),
           'LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    if PLAN['visible_devices'] is not None:
        env['ASCEND_RT_VISIBLE_DEVICES'] = PLAN['visible_devices']
    for name in ['LD_PRELOAD', 'LLAMA_SERVER_DEBUG_FAKE_TIMING', 'ROPE_TRACE_DIR', 'BATCH_OBSERVER_DIR',
                 'GGML_SCHED_DEBUG', 'DFLASH_CAPTURE_DIR', 'DFLASH_CAPTURE_WIDTH']:
        env.pop(name, None)
    if capture_library is not None:
        if not capture_library.is_file() or not capture_width or capture_width < 1:
            raise ValueError('capture needs a compiled library and positive feature width')
        env['LD_PRELOAD'] = str(capture_library.resolve())
        env['DFLASH_CAPTURE_DIR'] = str(directory / 'capture')
        env['DFLASH_CAPTURE_WIDTH'] = str(capture_width)
    log = (directory / 'server.log').open('w', encoding='utf-8')
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    OWNED.append(process)
    save(directory / 'server.json', {'pid': process.pid, 'command': command,
                                   'visible_devices': env.get('ASCEND_RT_VISIBLE_DEVICES'),
                                   'fake_timing': False, 'preload': capture_library is not None})
    session = requests.Session()
    session.trust_env = False
    url = f'http://127.0.0.1:{port}'
    deadline = time.monotonic() + 240
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'{mode} startup exited: {process.returncode}')
            try:
                if session.get(url + '/health', timeout=2).status_code == 200:
                    return process, log, session, url
            except requests.RequestException:
                pass
            time.sleep(.2)
        raise TimeoutError('server startup timeout')
    except BaseException:
        stop(process)
        log.close()
        session.close()
        raise


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run_mode(mode, selected, warmups):
    directory = WORK/mode
    directory.mkdir()
    progress(stage='loading', phase=mode, completed=0)
    process, log, session, url = launch(mode, directory)
    records: list[dict[str, Any]] = []
    try:
        progress(stage='warmup', phase=mode, completed=0)
        warmup_results = [request(session, url, payload) for _, payload, _ in warmups]
        save(directory/'warmup.json', warmup_results)
        progress(stage='measuring', phase=mode, completed=0,
                 warmup_mean_s=statistics.mean(r['wall_seconds'] for r in warmup_results) if warmup_results else None)
        started = time.perf_counter()
        with (directory/'requests.jsonl').open('w', encoding='utf-8') as output:
            for index, (item, payload, digests) in enumerate(selected):
                before = metrics(session, url)
                try:
                    result = request(session, url, payload)
                    after = metrics(session, url)
                    result['counters'] = {k: after[k]-before[k] for k in before}
                    assert result['timings'].get('draft_n', 0)==result['counters']['drafted']
                    assert result['timings'].get('draft_n_accepted', 0)==result['counters']['accepted']
                except Exception as error:
                    result = {'ok': False, 'error': str(error), 'traceback': traceback.format_exc()}
                result = {**result, **item, 'sample_index': index, 'image_sha256': digests}
                records.append(result)
                output.write(json.dumps(result, ensure_ascii=False)+'\n')
                output.flush()
                elapsed = time.perf_counter()-started
                progress(stage='measuring', phase=mode, completed=index+1, successes=sum(r.get('ok', False) for r in records),
                         elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed/(index+1)*(len(selected)-index-1),
                         last_wall_seconds=result.get('wall_seconds'), last_error=result.get('error'))
                if process.poll() is not None:
                    raise RuntimeError(f'{mode} server stopped during the measurement')
        result = summarize(records, time.perf_counter()-started)
        save(directory/'summary.json', result)
        return records, result
    finally:
        stop(process)
        log.close()
        session.close()
        save(directory/'stopped.json', {'pid': process.pid, 'returncode': process.returncode, 'stopped': process.poll() is not None})


def monitor():
    with (WORK/'evidence/npu-samples.jsonl').open('w') as output:
        index = 0
        while not STOP.is_set():
            try:
                info = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=20)
                (WORK/'evidence'/f'npu-{index:04d}.txt').write_text(info.stdout+info.stderr)
                item = {'time': time.time(), 'phase': STATE['phase'], 'stage': STATE['stage'], 'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None}
                for line in info.stdout.splitlines():
                    parts = [v.strip() for v in line.split('|')[1:-1]]
                    if len(parts) >= 3 and PLAN['visible_devices'] and re.fullmatch(r'\d+\s+' + re.escape(PLAN['visible_devices']), parts[0]):
                        values = re.findall(r'\d+', parts[2])
                        if len(values) >= 3:
                            item.update({'aicore_percent': int(values[0]), 'memory_mb': int(values[-2]), 'total_mb': int(values[-1])})
                output.write(json.dumps(item)+'\n')
                output.flush()
                index += 1
            except Exception as error:
                output.write(json.dumps({'error': str(error)})+'\n')
                output.flush()
            STOP.wait(15)


def compare_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if left['question_id'] != right['question_id'] or left['image_sha256'] != right['image_sha256']:
        raise ValueError('paired inputs differ')
    ok = bool(left.get('ok') and right.get('ok'))
    ids_available = ok and bool(left.get('tokens')) and bool(right.get('tokens'))
    return {'sample_index': left['sample_index'], 'question_id': left['question_id'], 'both_succeeded': ok,
            'tokens_equal': left['tokens'] == right['tokens'] if ids_available else None,
            'text_equal': ok and left['content'] == right['content'],
            'stop_equal': ok and left['stop_type'] == right['stop_type'],
            'baseline_latency': left.get('wall_seconds'), 'dflash_latency': right.get('wall_seconds')}


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def main():
    args = parser().parse_args()
    configure(args)
    if args.prepare_only:
        return
    save(WORK / 'driver.json', {'pid': os.getpid(), 'started_at': time.time()})
    selected, warmups = prepare('selection.jsonl'), prepare('warmup.jsonl')
    thread = threading.Thread(target=monitor, daemon=True) if args.monitor_npu else None
    if thread:
        thread.start()
    try:
        baseline, baseline_stats = run_mode('baseline', selected, warmups)
        draft, draft_stats = run_mode('dflash', selected, warmups)
        pairs = [compare_pair(left, right) for left, right in zip(baseline, draft)]
        save(WORK / 'comparisons.json', pairs)
        available = sum(p['tokens_equal'] is not None for p in pairs)
        result = {'plan': PLAN, 'baseline': baseline_stats, 'dflash': draft_stats,
                  'paired_requests': len(pairs), 'native_token_id_pairs_available': available,
                  'token_matches': sum(p['tokens_equal'] is True for p in pairs) if available else None,
                  'text_matches': sum(p['text_equal'] for p in pairs), 'stop_matches': sum(p['stop_equal'] for p in pairs),
                  'latency_speedup': ratio(baseline_stats['latency_mean_s'], draft_stats['latency_mean_s']),
                  'decode_tps_speedup': ratio(draft_stats['server_decode_tokens_per_second'], baseline_stats['server_decode_tokens_per_second'])}
        with (WORK / 'comparisons.tsv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(pairs)
        save(WORK / 'summary.json', result)
        passed = all(p['both_succeeded'] and p['text_equal'] and p['stop_equal'] for p in pairs)
        save(WORK / 'complete.json', {'passed': passed, 'completed_at': time.time()})
        progress(stage='complete', phase=None, completed=len(pairs))
        if not passed:
            raise RuntimeError('requests or paired output checks failed; see results')
    finally:
        STOP.set()
        if thread:
            thread.join(timeout=25)
        for process in OWNED:
            stop(process)


def interrupted(_signum, _frame):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException as error:
        if 'WORK' not in globals() or not WORK.is_dir():
            raise
        save(WORK/'error.json', {'error': str(error), 'traceback': traceback.format_exc()})
        progress(stage='error', error=str(error))
        raise
