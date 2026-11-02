import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import verify_driver


class VerifyDriverTest(unittest.TestCase):
    def test_rejected_fixture_must_fail(self):
        def fake_run(command, output, name, stdin=None):
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"
            args = ["verify_driver", "--opt", sys.executable, "--filecheck", sys.executable, "--output", str(output)]
            with patch.object(sys, "argv", args), patch.object(verify_driver, "run", fake_run):
                with self.assertRaisesRegex(SystemExit, "verify-error: unexpected exit code 0"):
                    verify_driver.main()
            self.assertTrue((output / "identity.json").is_file())

    def test_failed_ir_reparse_stops_verification(self):
        def fake_run(command, output, name, stdin=None):
            code = 1 if name == "verify-reparse" else 0
            return SimpleNamespace(returncode=code, stdout="IR", stderr="")

        with tempfile.TemporaryDirectory() as directory:
            args = ["verify_driver", "--opt", sys.executable, "--filecheck", sys.executable, "--output", str(Path(directory) / "evidence")]
            with patch.object(sys, "argv", args), patch.object(verify_driver, "run", fake_run):
                with self.assertRaisesRegex(SystemExit, "verify: transformed IR failed to verify"):
                    verify_driver.main()

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "identity.json"
            receipt.write_text("preserved", encoding="utf-8")
            args = ["verify_driver", "--opt", sys.executable, "--filecheck", sys.executable, "--output", directory]
            with patch.object(sys, "argv", args):
                with self.assertRaises(FileExistsError):
                    verify_driver.main()
            self.assertEqual(receipt.read_text(encoding="utf-8"), "preserved")


if __name__ == "__main__":
    unittest.main()
