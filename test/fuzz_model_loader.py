#!/usr/bin/env python3
"""Black-box mutation fuzzer for the model loading path.

Generates random and mutated safetensors/GGUF files and feeds them to sd-cli.
A finding is any run that dies with a signal (exit >= 128) or prints an
AddressSanitizer report. Using an ASan build of sd-cli is recommended.

Usage:
    python3 test/fuzz_model_loader.py <path-to-sd-cli> [--seconds 300] [--seed 0]

Exit code 0 = no findings, 1 = findings (inputs saved next to the script).
"""
import argparse
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import time

DTYPES = ["F32", "F16", "BF16", "I64", "I32", "I8", "U8", "BOOL", "F8_E4M3", "XX"]


def random_safetensors(rng: random.Random) -> bytes:
    mode = rng.randrange(5)
    if mode == 0:  # random bytes
        return rng.randbytes(rng.randrange(0, 4096))
    if mode == 1:  # random header length + random payload
        return struct.pack("<Q", rng.randrange(0, 2**63)) + rng.randbytes(rng.randrange(0, 512))
    # structured header with hostile values
    tensors = {}
    for i in range(rng.randrange(0, 8)):
        shape = [rng.choice([0, 1, -1, 2**31, 2**62, rng.randrange(0, 10000)])
                 for _ in range(rng.randrange(0, 6))]
        start = rng.choice([0, 1, 2**31, 2**63 - 1, rng.randrange(0, 10**6)])
        end = rng.choice([start, start - 1, start + rng.randrange(0, 10**9), 2**63 - 1])
        tensors[f"t{i}" * rng.choice([1, 50])] = {
            "dtype": rng.choice(DTYPES),
            "shape": shape,
            "data_offsets": [start, end],
        }
    if rng.random() < 0.3:
        tensors["__metadata__"] = {"k" * rng.randrange(1, 100): "v" * rng.randrange(1, 100)}
    header = json.dumps(tensors).encode()
    if mode == 3:  # corrupt the JSON
        header = bytearray(header)
        for _ in range(rng.randrange(1, 10)):
            if header:
                header[rng.randrange(len(header))] = rng.randrange(256)
        header = bytes(header)
    declared = len(header) if mode != 4 else rng.randrange(0, 2**63)
    return struct.pack("<Q", declared) + header + rng.randbytes(rng.randrange(0, 256))


def random_gguf(rng: random.Random) -> bytes:
    out = b"GGUF" if rng.random() < 0.8 else rng.randbytes(4)
    out += struct.pack("<I", rng.choice([0, 1, 2, 3, 2**31]))
    out += struct.pack("<QQ", rng.randrange(0, 2**63), rng.randrange(0, 2**63))
    return out + rng.randbytes(rng.randrange(0, 2048))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sd_cli")
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    findings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fuzz_findings")
    deadline = time.monotonic() + args.seconds
    runs = 0
    findings = 0

    with tempfile.TemporaryDirectory() as tmp:
        while time.monotonic() < deadline:
            runs += 1
            is_gguf = rng.random() < 0.3
            data = random_gguf(rng) if is_gguf else random_safetensors(rng)
            path = os.path.join(tmp, "fuzz.gguf" if is_gguf else "fuzz.safetensors")
            with open(path, "wb") as f:
                f.write(data)
            proc = subprocess.run(
                [args.sd_cli, "-m", path, "-p", "t", "-o", os.path.join(tmp, "o.png")],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
            stderr = proc.stderr.decode(errors="replace")
            crashed = proc.returncode < 0 or proc.returncode >= 128
            asan = "ERROR: AddressSanitizer" in stderr or "ERROR: LeakSanitizer" in stderr
            if crashed or asan:
                findings += 1
                os.makedirs(findings_dir, exist_ok=True)
                name = f"finding_{findings}_{'gguf' if is_gguf else 'safetensors'}"
                with open(os.path.join(findings_dir, name + ".bin"), "wb") as f:
                    f.write(data)
                with open(os.path.join(findings_dir, name + ".log"), "w") as f:
                    f.write(f"returncode: {proc.returncode}\n\n{stderr}")
                print(f"FINDING #{findings} (run {runs}): rc={proc.returncode} saved as {name}")
            if runs % 100 == 0:
                print(f"... {runs} runs, {findings} findings, "
                      f"{int(deadline - time.monotonic())}s left", flush=True)

    print(f"\ndone: {runs} runs, {findings} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
