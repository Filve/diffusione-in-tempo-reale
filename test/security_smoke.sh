#!/usr/bin/env bash
# Security/robustness smoke test for sd-cli.
# Usage: ./security_smoke.sh [path-to-sd-cli]
# Exit code 0 = all checks passed, 1 = at least one failure.

set -u

SD_CLI="${1:-$(dirname "$0")/../build/bin/sd-cli}"
if [[ ! -x "$SD_CLI" ]]; then
    echo "sd-cli not found or not executable: $SD_CLI" >&2
    echo "usage: $0 [path-to-sd-cli]" >&2
    exit 1
fi
# The script cd's into a temp dir: the binary path must survive that.
SD_CLI="$(cd "$(dirname "$SD_CLI")" && pwd)/$(basename "$SD_CLI")"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR"

failures=0

check() {
    local label="$1"
    shift
    "$@" >out.log 2>&1
    local code=$?
    if [[ $code -ge 128 ]]; then
        echo "FAIL  $label  (crashed with signal $((code - 128)))"
        failures=$((failures + 1))
    elif [[ $code -eq 126 || $code -eq 127 ]]; then
        echo "FAIL  $label  (test setup error: binary not runnable, exit $code)"
        failures=$((failures + 1))
    elif [[ $code -eq 0 ]]; then
        echo "FAIL  $label  (unexpectedly succeeded)"
        failures=$((failures + 1))
    else
        echo "ok    $label  (clean failure, exit $code)"
    fi
}

# --- hostile model files -----------------------------------------------------
head -c 4096 /dev/urandom > garbage.safetensors
printf '\xff\xff\xff\xff\xff\xff\xff\x7f' > huge_header.safetensors
head -c 100 /dev/urandom >> huge_header.safetensors
printf '\xe8\x03\x00\x00\x00\x00\x00\x00{"incomplete' > truncated.safetensors
python3 - <<'EOF'
import json, struct
h = json.dumps({'a': {'dtype': 'F32', 'shape': [1000000, 1000000],
                      'data_offsets': [0, 4000000000000]}}).encode()
open('bad_offsets.safetensors', 'wb').write(struct.pack('<Q', len(h)) + h)
EOF
printf 'GGUF\x99\x99\x99\x99' > bad_magic.gguf
head -c 64 /dev/urandom >> bad_magic.gguf

for f in nonexistent.safetensors garbage.safetensors huge_header.safetensors \
         truncated.safetensors bad_offsets.safetensors bad_magic.gguf; do
    check "hostile model: $f" "$SD_CLI" -m "$f" -p test -o out.png
done

# --- extreme/malformed option values ----------------------------------------
long_arg="$(python3 -c 'print("A" * 5000)')"
check "option: --memory-guard huge"   "$SD_CLI" -m garbage.safetensors -p t --memory-guard 99999999999
check "option: --memory-guard text"   "$SD_CLI" -m garbage.safetensors -p t --memory-guard abc
check "option: --threads huge"        "$SD_CLI" -m garbage.safetensors -p t --threads 99999999999
check "option: --seed huge"           "$SD_CLI" -m garbage.safetensors -p t --seed 99999999999999999999999
check "option: --skip-layers huge"    "$SD_CLI" -m garbage.safetensors -p t --skip-layers 99999999999
check "option: --cfg-scale overflow"  "$SD_CLI" -m garbage.safetensors -p t --cfg-scale 1e999
check "option: --max-vram text"       "$SD_CLI" -m garbage.safetensors -p t --max-vram abc
check "option: --backend unknown"     "$SD_CLI" -m garbage.safetensors -p t --backend no_such_backend --disable-backend-fallback
check "option: --backend long"        "$SD_CLI" -m garbage.safetensors -p t --backend "$long_arg"

echo
if [[ $failures -gt 0 ]]; then
    echo "$failures check(s) FAILED"
    exit 1
fi
echo "all checks passed"
