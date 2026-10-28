#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "w8a8"))

from convert_modelslim import w8a8_blocks, w8a8_expert_blocks  # noqa: E402


class W8A8PackingTest(unittest.TestCase):
    def test_expert_blocks_preserve_payload_and_scale(self) -> None:
        weight = np.arange(2 * 3 * 64, dtype=np.int8).reshape(2, 3, 64)
        scale = np.array([[0.5, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        raw = w8a8_expert_blocks(weight, scale)
        self.assertEqual(raw.shape, (2, 3, 2 * 34))

        decoded = raw.reshape(2, 3, 2, 34)
        self.assertEqual(decoded.dtype, np.uint8)
        for expert in range(2):
            for row in range(3):
                self.assertEqual(decoded[expert, row, :, 0:2].view('<f2').reshape(-1).tolist(), [scale[expert, row]] * 2)
                payload = decoded[expert, row, :, 2:].view(np.int8).reshape(64)
                self.assertTrue(np.array_equal(payload, weight[expert, row]))

    def test_singleton_scale_plane_is_normalized(self) -> None:
        weight = np.arange(2 * 2 * 32, dtype=np.int8).reshape(2, 2, 32)
        scale = np.array([[[0.25], [0.5]], [[1.0], [2.0]]], dtype=np.float32)
        np.testing.assert_array_equal(w8a8_expert_blocks(weight, scale), w8a8_expert_blocks(weight, scale[..., 0]))

    def test_dense_blocks_are_byte_stable(self) -> None:
        weight = np.arange(4 * 32, dtype=np.int8).reshape(4, 32)
        scale = np.linspace(0.125, 0.5, 4, dtype=np.float32).reshape(4, 1)
        raw = w8a8_blocks(weight, scale)
        decoded = raw.reshape(4, 1, 34)
        payload = decoded[:, :, 2:].view(np.int8).reshape(4, 32)
        np.testing.assert_array_equal(payload, weight)
        np.testing.assert_array_equal(decoded[:, :, 0:2].view('<f2').reshape(-1), scale.astype(np.float16).reshape(-1))


if __name__ == "__main__":
    unittest.main()
