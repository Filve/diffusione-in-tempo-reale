#!/usr/bin/env bash
# Create a relocatable distribution package from a portable build
# (see docs/build.md, "Build a portable package").
# Usage: scripts/package_portable.sh [build_dir] [output_dir]
#
# Environment:
#   PORTABLE_VARIANT           suffix of the zip name and wording of LEGGIMI.txt:
#                              cuda12, vulkan or metal (empty: CPU only)
#   PORTABLE_REQUIRE_BACKENDS  space/comma separated ggml backends (cuda, vulkan,
#                              metal) whose module must be in the build; the
#                              script fails instead of shipping a package that
#                              would silently fall back to the CPU

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="${1:-$repo_root/build}"
output_dir="${2:-$repo_root/dist}"

if [[ ! -x "$build_dir/bin/sd-cli" && ! -x "$build_dir/bin/sd-cli.exe" ]]; then
    echo "error: $build_dir/bin/sd-cli not found; build the project first" >&2
    echo "usage: $0 [build_dir] [output_dir]" >&2
    exit 1
fi

# The zip is created from another directory, so the paths must not be relative.
mkdir -p "$output_dir"
build_dir="$(cd "$build_dir" && pwd)"
output_dir="$(cd "$output_dir" && pwd)"

cd "$repo_root"
version="$(git describe --tags --always 2>/dev/null || echo unknown)"
case "$(uname -s)" in
    Darwin) os=darwin ;;
    Linux) os=linux ;;
    MINGW* | MSYS* | CYGWIN*) os=windows ;;
    *) os="$(uname -s | tr '[:upper:]' '[:lower:]')" ;;
esac
arch="$(uname -m)"
variant="${PORTABLE_VARIANT:-}"
name="diffusione-in-tempo-reale-${version}-${os}-${arch}${variant:+-$variant}"
stage="$(mktemp -d)/$name"
mkdir -p "$stage/licenses" "$output_dir"
trap 'rm -rf "$(dirname "$stage")"' EXIT

# Binaries plus every shared library/backend module next to them.
cp -R "$build_dir/bin/." "$stage/"

missing=()
required="${PORTABLE_REQUIRE_BACKENDS:-}"
for backend in ${required//,/ }; do
    compgen -G "$stage/*ggml-$backend.*" >/dev/null || missing+=("ggml-$backend module")
    if [[ "$backend" == cuda && "$os" == windows ]]; then
        # ggml-cuda.dll cannot load on a machine without the CUDA toolkit
        # unless its runtime libraries sit next to it.
        for lib in cudart64_ cublas64_ cublasLt64_; do
            compgen -G "$stage/${lib}*.dll" >/dev/null || missing+=("${lib}*.dll")
        done
    fi
done
if ((${#missing[@]})); then
    echo "error: the package could not use its GPU backend; missing: ${missing[*]}" >&2
    exit 1
fi

case "$variant" in
    cuda*) gpu_note="GPU: questo pacchetto usa CUDA 12 (schede NVIDIA). Serve solo il driver
NVIDIA aggiornato: le DLL di CUDA (cudart, cublas) sono incluse e devono
restare nella stessa cartella di sd-cli e sd-server." ;;
    vulkan) gpu_note="GPU: questo pacchetto usa Vulkan (schede NVIDIA, AMD, Intel). Serve il
driver grafico aggiornato del produttore, che installa Vulkan insieme a se'." ;;
    metal) gpu_note="GPU: questo pacchetto usa Metal (Apple Silicon)." ;;
    *) gpu_note="GPU: questo pacchetto usa solo la CPU." ;;
esac

help_note=""
if [[ -n "$variant" ]]; then
    keep_files="i file ggml-*"
    if [[ "$variant" == cuda* ]]; then
        keep_files+=" e le DLL di CUDA"
    fi
    help_note="
Se --list-devices mostra solo la CPU: aggiorna il driver della scheda video e
non spostare ${keep_files} fuori dalla cartella degli
eseguibili. Il programma scrive nel log perche' un modulo GPU non e' stato
caricato.
"
fi
if [[ "$os" == windows ]]; then
    help_note+="
Su Windows serve anche Microsoft Visual C++ Redistributable 2015-2022 (x64),
di solito gia' presente.
"
fi

cp LICENSE "$stage/"
cp ggml/LICENSE "$stage/licenses/ggml.LICENSE"
cp thirdparty/libwebp/COPYING "$stage/licenses/libwebp.COPYING" 2>/dev/null || true
cp thirdparty/libwebm/LICENSE.TXT "$stage/licenses/libwebm.LICENSE.TXT" 2>/dev/null || true
cp thirdparty/oniguruma/COPYING "$stage/licenses/oniguruma.COPYING" 2>/dev/null || true
cp thirdparty/utf8proc/LICENSE.md "$stage/licenses/utf8proc.LICENSE.md" 2>/dev/null || true

cat > "$stage/LEGGIMI.txt" <<EOF
diffusione-in-tempo-reale ${version} (${os}/${arch}${variant:+, $variant})
Pacchetto portabile: i backend (ggml-*) sono caricati a runtime, solo se
la macchina li supporta. Nessuna installazione richiesta.

${gpu_note}

Comandi principali:
  sd-cli --list-devices          elenca i dispositivi: deve comparire la GPU
  sd-cli -m modello.safetensors -p "un gatto"   genera un'immagine
  sd-server                      avvia il server HTTP su 127.0.0.1:1234
                                 (parte anche senza modello)
${help_note}
Opzioni utili: --memory-guard 90 (protezione memoria), --max-vram,
--params-backend disk (modelli piu' grandi della RAM/VRAM).

I modelli NON sono inclusi e hanno licenze proprie.
Basato su stable-diffusion.cpp (MIT, leejet); modifiche del fork di
Francesco Simeoni (MIT). Licenze complete: LICENSE e licenses/.
EOF

(
    cd "$(dirname "$stage")"
    if command -v zip >/dev/null 2>&1; then
        zip -qry "$output_dir/$name.zip" "$name"
    else
        # Windows runners have python but usually no zip.
        "$(command -v python3 || command -v python)" -m zipfile -c "$output_dir/$name.zip" "$name"
    fi
)
echo "created: $output_dir/$name.zip"
unzip -l "$output_dir/$name.zip" 2>/dev/null | tail -3 || true
