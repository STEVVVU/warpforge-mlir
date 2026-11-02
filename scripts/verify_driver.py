#!/usr/bin/env python3
"""Drive the built optimizer and retain outputs and identity for inspection."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("verify", "--warpforge-verify-inference", True),
    ("verify-error", "--warpforge-verify-inference", False),
    ("fusion", "--warpforge-fuse-matmul-epilogue", True),
    ("tile-vectorize", "--warpforge-tile-vectorize", True),
    ("tile-vectorize-error", "--warpforge-tile-vectorize", False),
)


def run(command, output, name, stdin=None):
    result = subprocess.run(command, input=stdin, capture_output=True, text=True, timeout=60)
    (output / f"{name}.stdout").write_text(result.stdout, encoding="utf-8")
    (output / f"{name}.stderr").write_text(result.stderr, encoding="utf-8")
    (output / f"{name}.json").write_text(
        json.dumps({"command": [str(arg) for arg in command], "exit_code": result.returncode}, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--filecheck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    opt, filecheck, output = args.opt.resolve(), args.filecheck.resolve(), args.output.resolve()
    output.mkdir(parents=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    files = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(set(files.split("\0"))) if name and (ROOT / name).is_file()}
    identity = {"source_commit": sha, "working_tree_status": status, "source_file_sha256": source_hashes, "binary_sha256": hashlib.sha256(opt.read_bytes()).hexdigest()}
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    version = run([opt, "--version"], output, "version")
    if version.returncode:
        raise SystemExit("optimizer version check failed")
    for name, flag, succeeds in CASES:
        fixture = ROOT / "test" / f"{name}.mlir"
        result = run([opt, fixture, flag], output, name)
        if (result.returncode == 0) != succeeds:
            raise SystemExit(f"{name}: unexpected exit code {result.returncode}")
        checked = run([filecheck, fixture], output, f"{name}-filecheck", result.stdout + result.stderr)
        if checked.returncode:
            raise SystemExit(f"{name}: FileCheck failed")
        if succeeds:
            verified = run([opt], output, f"{name}-reparse", result.stdout)
            if verified.returncode:
                raise SystemExit(f"{name}: transformed IR failed to verify")
    print(f"Verified {len(CASES)} cases; evidence: {output}")


if __name__ == "__main__":
    main()
