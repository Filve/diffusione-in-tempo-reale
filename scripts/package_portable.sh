#!/usr/bin/env bash
# Create a relocatable distribution package from a portable build
# (see docs/build.md, "Build a portable package").
# Usage: scripts/package_portable.sh [build_dir] [output_dir]

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="${1:-$repo_root/build}"
output_dir="${2:-$repo_root/dist}"

if [[ ! -x "$build_dir/bin/sd-cli" && ! -x "$build_dir/bin/sd-cli.exe" ]]; then
    echo "error: $build_dir/bin/sd-cli not found; build the project first" >&2
    echo "usage: $0 [build_dir] [output_dir]" >&2
    exit 1
fi

cd "$repo_root"
version="$(git describe --tags --always 2>/dev/null || echo unknown)"
case "$(uname -s)" in
    Darwin) os=darwin ;;
    Linux) os=linux ;;
    MINGW* | MSYS* | CYGWIN*) os=windows ;;
    *) os="$(uname -s | tr '[:upper:]' '[:lower:]')" ;;
esac
arch="$(uname -m)"
name="diffusione-in-tempo-reale-${version}-${os}-${arch}"
stage="$(mktemp -d)/$name"
mkdir -p "$stage/licenses" "$output_dir"
trap 'rm -rf "$(dirname "$stage")"' EXIT

# Binaries plus every shared library/backend module next to them.
cp -R "$build_dir/bin/." "$stage/"

cp LICENSE "$stage/"
cp ggml/LICENSE "$stage/licenses/ggml.LICENSE"
cp thirdparty/libwebp/COPYING "$stage/licenses/libwebp.COPYING" 2>/dev/null || true
cp thirdparty/libwebm/LICENSE.TXT "$stage/licenses/libwebm.LICENSE.TXT" 2>/dev/null || true
cp thirdparty/oniguruma/COPYING "$stage/licenses/oniguruma.COPYING" 2>/dev/null || true
cp thirdparty/utf8proc/LICENSE.md "$stage/licenses/utf8proc.LICENSE.md" 2>/dev/null || true

cat > "$stage/LEGGIMI.txt" <<EOF
diffusione-in-tempo-reale ${version} (${os}/${arch})
Pacchetto portabile: i backend (libggml-*) sono caricati a runtime, solo se
la macchina li supporta. Nessuna installazione richiesta.

Comandi principali:
  ./sd-cli --list-devices        elenca i dispositivi disponibili
  ./sd-cli -m modello.safetensors -p "un gatto"   genera un'immagine
  ./sd-server                    avvia il server HTTP su 127.0.0.1:1234
                                 (parte anche senza modello)

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
