"""Check DFlash GGUF tensors against a single-file HF checkpoint after dtype conversion."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open
import torch

sys.path.insert(0, str(Path(os.environ.get('LLAMA_CPP_ROOT', Path(__file__).resolve().parents[5])) / 'gguf-py'))
import gguf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf-dir', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.hf_dir / 'config.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4)
    reader = gguf.GGUFReader(str(args.gguf))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.DFLASH, config['num_hidden_layers'])
    rows = []
    with safe_open(str(args.hf_dir / 'model.safetensors'), framework='pt', device='cpu') as hf:
        for name in hf.keys():
            converted_name = mapping.get_name('model.' + name, try_suffixes=('.weight',))
            if converted_name is None or converted_name not in tensors:
                raise ValueError(f'missing GGUF tensor for {name}')
            actual = tensors[converted_name].data
            expected = hf.get_tensor(name).float().numpy().astype(actual.dtype)
            if expected.shape != actual.shape:
                raise ValueError(f'shape mismatch for {name}: {expected.shape} != {actual.shape}')
            error = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
            rows.append({'source': name, 'gguf': converted_name, 'shape': list(actual.shape),
                         'dtype': str(actual.dtype), 'exact_after_cast': bool(np.array_equal(expected, actual)),
                         'max_abs_error': float(error.max()), 'finite': bool(np.isfinite(actual).all())})
    result = {'checked_tensors': len(rows), 'gguf_tensors': len(tensors),
              'exact_tensors': sum(r['exact_after_cast'] is True for r in rows), 'all_finite': all(r['finite'] for r in rows),
              'rows': rows}
    result['passed'] = bool(rows) and len(rows) == len(tensors) and result['exact_tensors'] == len(rows) and result['all_finite']
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
