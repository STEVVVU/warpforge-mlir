import unittest

from scripts.benchmark_pipeline import require_expected_subset
from scripts.generate_workload import generate_workload, validate_dimensions


class GenerateWorkloadTest(unittest.TestCase):
    def test_is_deterministic_and_encodes_shapes(self) -> None:
        first = generate_workload(32, 64, 16, "f16")
        second = generate_workload(32, 64, 16, "f16")
        self.assertEqual(first, second)
        self.assertIn("tensor<32x16xf16>", first)
        self.assertIn("tensor<16x64xf16>", first)
        self.assertEqual(first.count("linalg.matmul"), 1)
        self.assertEqual(first.count("linalg.add"), 1)
        self.assertEqual(first.count("linalg.max"), 1)

    def test_rejects_unsupported_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "f16 or f32"):
            generate_workload(16, 16, 16, "f64")

    def test_rejects_non_divisible_tile(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            validate_dimensions(31, 16, 16, 16)

    def test_evidence_subset_detects_drift(self) -> None:
        require_expected_subset({"delta": {"add": -1}}, {"delta": {"add": -1}})
        with self.assertRaisesRegex(ValueError, "expected -1, got 0"):
            require_expected_subset({"delta": {"add": -1}}, {"delta": {"add": 0}})


if __name__ == "__main__":
    unittest.main()
