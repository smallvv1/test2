#!/usr/local/bin/python3
"""评估器名单、缺失资产和指纹复用门禁的单元测试。"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import gpc_v2_expanded_dev_eval as evaluator


class EvaluatorProtocolTests(unittest.TestCase):
    def test_full_b0_to_a5_list(self):
        labels = [f"screen_{arm}_seed1234" for arm in ("B0", "A1", "A2", "A3", "A4", "A5")]
        specs = evaluator.build_specs(Path("/tmp/training"), labels=labels)
        self.assertEqual([label for label, _ in specs], labels)

    def test_current_b0_a4_r2_two_seed_list(self):
        specs = evaluator.build_specs(Path("/tmp/training"), arms=("B0", "A4", "R2"), seeds=(1234, 4321))
        self.assertEqual(len(specs), 6)
        self.assertNotIn("screen_A1_seed1234", [label for label, _ in specs])

    def test_missing_expected_checkpoint_is_error(self):
        with tempfile.TemporaryDirectory() as temp:
            specs = evaluator.build_specs(Path(temp), labels=["screen_B0_seed1234", "screen_A4_seed1234"])
            with self.assertRaises(FileNotFoundError):
                evaluator.validate_spec_files(specs)

    def test_protocol_mismatch_blocks_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "model_final.pth"
            checkpoint.write_bytes(b"checkpoint")
            result_path = root / "screen_B0_seed1234.json"
            result_path.write_text(json.dumps({
                "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                "protocol_sha256": "old-protocol",
                "query_sha256": "query",
                "gallery_sha256": "gallery",
                "evaluation_config_sha256": "config",
                "metrics": {"query_count": 1, "gallery_count": 1, "effective_pid_count": 1},
                "query_metrics": [{}],
            }), encoding="utf-8")
            reused, reason = evaluator.reusable_result(
                result_path, checkpoint,
                {"protocol_sha256": "new-protocol", "query_sha256": "query", "gallery_sha256": "gallery"},
                "config", 1, 1, 1,
            )
            self.assertIsNone(reused)
            self.assertIn("protocol_sha256", reason)

    def test_checkpoint_mismatch_blocks_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "model_final.pth"
            checkpoint.write_bytes(b"new-checkpoint")
            result_path = root / "screen_B0_seed1234.json"
            result_path.write_text(json.dumps({
                "checkpoint_sha256": "old-checkpoint",
                "protocol_sha256": "protocol",
                "query_sha256": "query",
                "gallery_sha256": "gallery",
                "evaluation_config_sha256": "config",
                "metrics": {"query_count": 1, "gallery_count": 1, "effective_pid_count": 1},
                "query_metrics": [{}],
            }), encoding="utf-8")
            reused, reason = evaluator.reusable_result(
                result_path, checkpoint,
                {"protocol_sha256": "protocol", "query_sha256": "query", "gallery_sha256": "gallery"},
                "config", 1, 1, 1,
            )
            self.assertIsNone(reused)
            self.assertIn("checkpoint_sha256", reason)


if __name__ == "__main__":
    unittest.main()
