#!/usr/bin/env bash
# Test of scripts/package_windows_cuda_vulkan.sh with small fake zips that have
# the file names of the real release zips. Every fake file holds "<zip>:<name>",
# so the test can tell which zip a file of the package came from (the tags cuda
# and vulk have the same length, so a shared file has the same size in both).
# Needs zip and unzip; no build, GPU or model.
# Usage: ./test/package_windows_cuda_vulkan_test.sh

set -uo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
script="$repo_root/scripts/package_windows_cuda_vulkan.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
failures=0

ok() { echo "ok   $*"; }
bad() {
    echo "FAIL $*"
    failures=$((failures + 1))
}
expect() { # expect <description> <command...>
    local what="$1"
    shift
    if "$@" >/dev/null 2>&1; then ok "$what"; else bad "$what"; fi
}

# Files that both programs zips have under the same name.
common=(ggml-base.dll ggml-cpu-alderlake.dll ggml-cpu-cannonlake.dll ggml-cpu-cascadelake.dll ggml-cpu-haswell.dll
    ggml-cpu-icelake.dll ggml-cpu-sandybridge.dll ggml-cpu-skylakex.dll ggml-cpu-sse42.dll ggml-cpu-x64.dll
    ggml.dll ggml.txt libsharpyuv.dll libwebp.dll libwebpmux.dll sd-cli.exe sd-server.exe
    stable-diffusion.cpp.txt stable-diffusion.dll webm.dll)
runtime=(cublas64_12.dll cublasLt64_12.dll cudart64_12.dll)

make_zip() { # make_zip <zip> <tag> <file...>
    local zip="$1" tag="$2"
    shift 2
    local dir
    dir="$(mktemp -d "$work/zip.XXXXXX")"
    for file in "$@"; do echo "$tag:$file" >"$dir/$file"; done
    (cd "$dir" && zip -q -j "$zip" "$@")
}

make_inputs() { # make_inputs <dir> <cuda_commit> <vulkan_commit>
    mkdir -p "$1"
    make_zip "$1/sd-master-$2-bin-win-cuda12-x64.zip" cuda "${common[@]}" ggml-cuda.dll
    make_zip "$1/sd-master-$3-bin-win-vulkan-x64.zip" vulk "${common[@]}" ggml-vulkan.dll
    make_zip "$1/cudart-sd-bin-win-cu12-x64.zip" cudart "${runtime[@]}"
}

member() { # member <zip> <name>: the content of one file of the package
    unzip -p "$1" "$2" 2>/dev/null
}

# expect_failure <description> <input_dir> <message regex>
expect_failure() {
    local what="$1" dir="$2" pattern="$3" log="$work/failure.log"
    if bash "$script" "$dir" "$work/failure-out" >"$log" 2>&1; then
        bad "$what (the script succeeded)"
    elif ! grep -Eq "$pattern" "$log"; then
        bad "$what (unexpected message: $(head -c 200 "$log"))"
    elif compgen -G "$work/failure-out/*.zip" >/dev/null || compgen -G "$dir/*cuda12-vulkan*.zip" >/dev/null; then
        bad "$what (a zip was left behind)"
    else
        ok "$what"
    fi
    rm -rf "$work/failure-out"
}

# 1. The normal case.
name=sd-master-abc1234-bin-win-cuda12-vulkan-x64
make_inputs "$work/normal" abc1234 abc1234
if bash "$script" "$work/normal" "$work/normal-out" >"$work/normal.log" 2>&1; then
    ok "assembles the package"
else
    bad "assembles the package"
    cat "$work/normal.log"
fi
zip="$work/normal-out/$name.zip"
expect "the zip is named after the build" test -f "$zip"
listing="$(unzip -Z1 "$zip" 2>/dev/null)"
[[ -n "$listing" && -z "$(grep -v "^$name/" <<<"$listing")" ]] &&
    ok "everything is in one folder named like the zip" || bad "everything is in one folder named like the zip"
[[ "$(wc -l <<<"$listing" | tr -d ' ')" -eq 26 ]] &&
    ok "20 shared files + CUDA module + 3 runtime DLLs + Vulkan module + LEGGIMI.txt" || bad "the package has $(wc -l <<<"$listing" | tr -d ' ') files, expected 26"
[[ "$(member "$zip" "$name/sd-cli.exe")" == "cuda:sd-cli.exe" ]] && ok "the programs come from the CUDA zip" || bad "the programs come from the CUDA zip"
[[ "$(member "$zip" "$name/stable-diffusion.dll")" == "cuda:stable-diffusion.dll" ]] && ok "stable-diffusion.dll comes from the CUDA zip" || bad "stable-diffusion.dll comes from the CUDA zip"
[[ "$(member "$zip" "$name/ggml-cuda.dll")" == "cuda:ggml-cuda.dll" ]] && ok "ggml-cuda.dll is there" || bad "ggml-cuda.dll is there"
[[ "$(member "$zip" "$name/ggml-vulkan.dll")" == "vulk:ggml-vulkan.dll" ]] && ok "ggml-vulkan.dll comes from the Vulkan zip" || bad "ggml-vulkan.dll comes from the Vulkan zip"
[[ "$(member "$zip" "$name/cublasLt64_12.dll")" == "cudart:cublasLt64_12.dll" ]] && ok "the CUDA runtime DLLs are there" || bad "the CUDA runtime DLLs are there"
member "$zip" "$name/LEGGIMI.txt" | grep -q 'commit abc1234' && ok "LEGGIMI.txt names the commit" || bad "LEGGIMI.txt names the commit"
grep -qi 'warning' "$work/normal.log" && bad "no warning when the shared files have the same size" || ok "no warning when the shared files have the same size"

# 2. Running it again in the input directory: its own output is not taken for an input.
make_inputs "$work/rerun" abc1234 abc1234
bash "$script" "$work/rerun" >/dev/null 2>&1 && bash "$script" "$work/rerun" >/dev/null 2>&1 &&
    [[ "$(compgen -G "$work/rerun/*cuda12-vulkan*.zip" | wc -l | tr -d ' ')" -eq 1 ]] &&
    ok "can be run twice in the same directory" || bad "can be run twice in the same directory"

# 3. A shared file of another size is a warning, not an error.
make_inputs "$work/resized" abc1234 abc1234
rm "$work/resized/sd-master-abc1234-bin-win-vulkan-x64.zip"
dir="$(mktemp -d "$work/zip.XXXXXX")"
for file in "${common[@]}" ggml-vulkan.dll; do echo "vulk:$file" >"$dir/$file"; done
echo "a ggml-base.dll that is longer than the one of the CUDA zip" >"$dir/ggml-base.dll"
(cd "$dir" && zip -q -j "$work/resized/sd-master-abc1234-bin-win-vulkan-x64.zip" "${common[@]}" ggml-vulkan.dll)
if bash "$script" "$work/resized" >"$work/resized.log" 2>&1 && grep -q 'warning:' "$work/resized.log" && grep -q 'ggml-base.dll' "$work/resized.log"; then
    ok "warns when the two builds have a shared file of a different size"
else
    bad "warns when the two builds have a shared file of a different size"
fi

# 4. Inputs that must be refused.
make_inputs "$work/other-commit" abc1234 def5678
expect_failure "refuses the CUDA and Vulkan zips of different builds" "$work/other-commit" 'different builds'

make_inputs "$work/no-runtime" abc1234 abc1234
rm "$work/no-runtime/cudart-sd-bin-win-cu12-x64.zip"
expect_failure "refuses a missing CUDA runtime zip" "$work/no-runtime" 'expected cudart-sd-bin-win-cu12-x64.zip'

make_inputs "$work/no-vulkan-zip" abc1234 abc1234
rm "$work/no-vulkan-zip/sd-master-abc1234-bin-win-vulkan-x64.zip"
expect_failure "refuses a missing Vulkan zip" "$work/no-vulkan-zip" 'expected one sd-\*-bin-win-vulkan-x64.zip in .*found 0'

make_inputs "$work/no-cuda-zip" abc1234 abc1234
rm "$work/no-cuda-zip/sd-master-abc1234-bin-win-cuda12-x64.zip"
expect_failure "refuses a missing CUDA zip" "$work/no-cuda-zip" 'expected one sd-\*-bin-win-cuda12-x64.zip in .*found 0'

make_inputs "$work/no-vulkan-module" abc1234 abc1234
rm "$work/no-vulkan-module/sd-master-abc1234-bin-win-vulkan-x64.zip"
make_zip "$work/no-vulkan-module/sd-master-abc1234-bin-win-vulkan-x64.zip" vulk "${common[@]}"
expect_failure "refuses a Vulkan zip without ggml-vulkan.dll" "$work/no-vulkan-module" 'ggml-vulkan.dll is not in'

make_inputs "$work/no-cuda-module" abc1234 abc1234
rm "$work/no-cuda-module/sd-master-abc1234-bin-win-cuda12-x64.zip"
make_zip "$work/no-cuda-module/sd-master-abc1234-bin-win-cuda12-x64.zip" cuda "${common[@]}"
expect_failure "refuses a CUDA zip without ggml-cuda.dll" "$work/no-cuda-module" 'missing: .*ggml-cuda.dll'

make_inputs "$work/no-cuda-dlls" abc1234 abc1234
rm "$work/no-cuda-dlls/cudart-sd-bin-win-cu12-x64.zip"
make_zip "$work/no-cuda-dlls/cudart-sd-bin-win-cu12-x64.zip" cudart cudart64_12.dll
expect_failure "refuses a CUDA runtime zip without cublas" "$work/no-cuda-dlls" 'missing: .*cublas'

make_inputs "$work/has-readme" abc1234 abc1234
rm "$work/has-readme/sd-master-abc1234-bin-win-cuda12-x64.zip"
make_zip "$work/has-readme/sd-master-abc1234-bin-win-cuda12-x64.zip" cuda "${common[@]}" ggml-cuda.dll LEGGIMI.txt
expect_failure "refuses to overwrite a LEGGIMI.txt that a zip already has" "$work/has-readme" 'already contain a LEGGIMI.txt'

make_inputs "$work/same-name" abc1234 abc1234
rm "$work/same-name/cudart-sd-bin-win-cu12-x64.zip"
make_zip "$work/same-name/cudart-sd-bin-win-cu12-x64.zip" cudart "${runtime[@]}" ggml.dll
expect_failure "refuses two zips that both hold the same file" "$work/same-name" 'both contain: ggml.dll'

make_inputs "$work/two-builds" abc1234 abc1234
make_zip "$work/two-builds/sd-master-def5678-bin-win-cuda12-x64.zip" cuda "${common[@]}" ggml-cuda.dll
expect_failure "refuses a directory with two CUDA zips" "$work/two-builds" 'expected one sd-\*-bin-win-cuda12-x64.zip'

if ((failures)); then
    echo "$failures check(s) failed"
    exit 1
fi
echo "all checks passed"
