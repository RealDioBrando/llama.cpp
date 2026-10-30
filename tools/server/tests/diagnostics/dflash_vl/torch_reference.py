"""Check the captured first draft block against an independent eager arithmetic reference."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(os.environ.get('LLAMA_CPP_ROOT', Path(__file__).resolve().parents[5])) / 'gguf-py'))
import gguf

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--capture', type=Path, required=True)
p.add_argument('--hf-dir', type=Path, required=True)
p.add_argument('--target-gguf', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
CAP = args.capture
config = json.loads((args.hf_dir / 'config.json').read_text(encoding='utf-8'))
rope_config = config.get('rope_parameters') or config.get('rope_scaling') or {}
if (config['hidden_size'] != 2560 or config['head_dim'] != 128 or config['num_hidden_layers'] != 5
        or config['num_attention_heads'] != 32 or config['num_key_value_heads'] != 8
        or rope_config.get('mrope_section') != [24, 20, 20] or not rope_config.get('mrope_interleaved')
        or config.get('rope_theta') != 5000000 or config.get('rms_norm_eps') != 1e-6):
    raise ValueError('reference supports the documented five-layer Qwen3-VL contract only')

torch.set_num_threads(8)
torch.set_grad_enabled(False)
with safe_open(str(args.hf_dir / 'model.safetensors'), framework='pt', device='cpu') as state:
    weights = {k: state.get_tensor(k).float() for k in state.keys()}
reader = gguf.GGUFReader(str(args.target_gguf))
target = {t.name: t for t in reader.tensors}
embedding = torch.tensor(target['token_embd.weight'].data, dtype=torch.float32)
head = torch.tensor(target['output.weight'].data, dtype=torch.float32) if 'output.weight' in target else embedding
metadata = json.loads((CAP / 'noise.json').read_text())
parts = sorted(CAP.glob('inject_*.json'), key=lambda p: int(p.stem.split('_')[-1]))
feature_parts, position_parts = [], []
for p in parts:
    d = json.loads(p.read_text())
    feature_parts.append(np.fromfile(p.with_suffix('.bin'), np.float32).reshape(d['rows'], d['width']))
    position_parts.append(np.fromfile(p.with_suffix('.pos'), np.int32).reshape(4, d['rows'])[:3].T)
features = torch.from_numpy(np.concatenate(feature_parts))
positions = torch.from_numpy(np.concatenate(position_parts)).long()
noise_ids = torch.from_numpy(np.fromfile(CAP / 'noise.tokens', np.int32)).long()
noise_pos = torch.from_numpy(np.fromfile(CAP / 'noise.pos', np.int32)).long()
noise_pos = noise_pos[:, None].expand(-1, 3)
actual = torch.from_numpy(np.fromfile(CAP / 'logits.bin', np.float32).reshape(metadata['rows'], metadata['vocab']))


def norm(x, weight):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * weight


def rope(x, pos):
    axis = torch.zeros(64, dtype=torch.long)
    axis[1:60:3] = 1
    axis[2:60:3] = 2
    frequency = 5000000.0 ** (-torch.arange(64, dtype=torch.float32) / 64)
    angle = pos[:, axis].float() * frequency
    angle = torch.cat([angle, angle], dim=-1)[None]
    rotated = torch.cat([-x[..., 64:], x[..., :64]], dim=-1)
    return x * angle.cos() + rotated * angle.sin()


def run(include_images):
    keep = torch.ones(features.shape[0], dtype=torch.bool) if include_images else (positions[:, 0] == positions[:, 1]) & (positions[:, 0] == positions[:, 2])
    encoded = norm(F.linear(features[keep], weights['fc.weight']), weights['hidden_norm.weight'])
    pos = positions[keep]
    hidden = embedding[noise_ids]
    layer_stats = []
    for layer in range(5):
        prefix = f'layers.{layer}.'
        def get(name):
            return weights[prefix + name + '.weight']

        def project(x, name, heads):
            return F.linear(x, get('self_attn.' + name)).reshape(-1, heads, 128).transpose(0, 1)
        context_k = rope(norm(project(encoded, 'k_proj', 8), get('self_attn.k_norm')), pos)
        context_v = project(encoded, 'v_proj', 8)
        normalized = norm(hidden, get('input_layernorm'))
        query = rope(norm(project(normalized, 'q_proj', 32), get('self_attn.q_norm')), noise_pos)
        key = rope(norm(project(normalized, 'k_proj', 8), get('self_attn.k_norm')), noise_pos)
        value = project(normalized, 'v_proj', 8)
        # The deployed cache stores K/V as F16; the attention arithmetic is evaluated in F32 here.
        key = torch.cat([context_k, key], dim=1).half().float().repeat_interleave(4, dim=0)
        value = torch.cat([context_v, value], dim=1).half().float().repeat_interleave(4, dim=0)
        scores = query @ key.transpose(-1, -2) / (128 ** .5)
        attention = scores.softmax(dim=-1)
        layer_stats.append({'layer': layer, 'context_attention_mass_mean': float(attention[..., :len(pos)].sum(-1).mean())})
        output = (attention @ value).transpose(0, 1).reshape(-1, 4096)
        residual = hidden + F.linear(output, get('self_attn.o_proj'))
        normalized = norm(residual, get('post_attention_layernorm'))
        gate = F.silu(F.linear(normalized, get('mlp.gate_proj')))
        hidden = residual + F.linear(gate * F.linear(normalized, get('mlp.up_proj')), get('mlp.down_proj'))
    logits = F.linear(norm(hidden, weights['norm.weight']), head)
    difference = logits - actual
    result = {'included_context_rows': int(keep.sum()),
              'top1_matches_all': int((logits.argmax(-1) == actual.argmax(-1)).sum()),
              'top1_matches_draft_masks': int((logits.argmax(-1)[1:] == actual.argmax(-1)[1:]).sum()),
              'reference_choices_in_native_max_set': int((actual.gather(1, logits.argmax(-1)[:, None]).flatten()[1:] == actual.amax(-1)[1:]).sum()),
              'reference_ids': logits.argmax(-1).tolist(), 'native_ids': actual.argmax(-1).tolist(),
              'relative_l2': float(difference.norm() / actual.norm()),
              'cosine_mean': float(F.cosine_similarity(logits, actual, dim=-1).mean()),
              'max_abs_error': float(difference.abs().max()), 'layer_attention': layer_stats,
              'disagreements': [{'row': i, 'reference_id': int(logits.argmax(-1)[i]),
                                 'native_id': int(actual.argmax(-1)[i]),
                                 'native_score_gap': float(actual[i].max() - actual[i, logits.argmax(-1)[i]])}
                                for i in range(len(logits)) if logits.argmax(-1)[i] != actual.argmax(-1)[i]]}
    print(json.dumps({'include_images': include_images, **result}), flush=True)
    return result


result = {'reference_scope': 'Standard DFlash arithmetic with the checkpoint MRoPE configuration, given captured native target features; not the missing original training forward.',
          'context_rows': len(features), 'full_context': run(True), 'without_spatial_rows': run(False)}
args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
