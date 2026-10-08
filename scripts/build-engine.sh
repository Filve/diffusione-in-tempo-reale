#!/usr/bin/env bash
# Selettore interattivo delle build del motore per macOS e Linux.
# Ogni sistema operativo compila soltanto i propri binari: le voci per gli
# altri sistemi spiegano come ottenerli (build su quel sistema o workflow
# release-portable.yml con un tag portable-v*).
# Uso: scripts/build-engine.sh [scelta] [cartella-destinazione]

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
scelta="${1:-}"
destinazione="${2:-}"
host="$(uname -s)"
arch="$(uname -m)"

case "$host" in
    Darwin) host_label="macOS" ;;
    Linux) host_label="Linux" ;;
    *) echo "Sistema non riconosciuto: $host. Per Windows usare scripts/build-engine.ps1." >&2; exit 1 ;;
esac

if [[ -z "$scelta" ]]; then
    echo "Quale motore vuoi compilare? (sistema attuale: $host_label $arch)"
    echo "  1) macOS - CPU + Metal (richiede macOS)"
    echo "  2) Linux - solo CPU (pacchetto portabile)"
    echo "  3) Linux - CPU + Vulkan (richiede Vulkan SDK/driver)"
    echo "  4) Linux - CPU + CUDA (solo NVIDIA, richiede CUDA Toolkit)"
    echo "  5) Windows - istruzioni"
    read -r -p "Scelta [1-5]: " scelta
fi

cross_help() {
    echo "I binari per $1 non si possono compilare da $host_label:" >&2
    echo "compilare su quel sistema ($2) oppure usare il workflow" >&2
    echo "release-portable.yml con un tag portable-v* (zip per" >&2
    echo "linux-x86_64, darwin-arm64 e windows-x86_64)." >&2
    exit 1
}

build_dir=""
cmake_args=(-DCMAKE_BUILD_TYPE=Release -DSD_BUILD_SHARED_GGML_LIB=ON
            -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON)
require_accelerator=1

case "$scelta" in
    1)
        [[ "$host" == "Darwin" ]] || cross_help "macOS" "scripts/build-engine.sh"
        build_dir="build-macos"
        cmake_args+=(-DSD_METAL=ON)
        ;;
    2)
        [[ "$host" == "Linux" ]] || cross_help "Linux" "scripts/build-engine.sh"
        build_dir="build-linux-cpu"
        require_accelerator=0
        ;;
    3)
        [[ "$host" == "Linux" ]] || cross_help "Linux" "scripts/build-engine.sh"
        build_dir="build-linux-vulkan"
        cmake_args+=(-DSD_VULKAN=ON)
        ;;
    4)
        [[ "$host" == "Linux" ]] || cross_help "Linux" "scripts/build-engine.sh"
        command -v nvcc >/dev/null || { echo "CUDA Toolkit mancante (nvcc non trovato)." >&2; exit 1; }
        build_dir="build-linux-cuda"
        cmake_args+=(-DSD_CUDA=ON)
        ;;
    5)
        echo "I binari Windows si compilano su Windows con scripts/build-engine.ps1,"
        echo "oppure con il workflow release-portable.yml (tag portable-v*)."
        exit 0
        ;;
    *)
        echo "Scelta non valida: '$scelta'." >&2
        exit 1
        ;;
esac

if [[ "$arch" == "x86_64" ]]; then
    cmake_args+=(-DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF)
fi

cd "$repo_root"
if command -v ninja >/dev/null; then
    cmake -S . -B "$build_dir" -G Ninja "${cmake_args[@]}"
else
    cmake -S . -B "$build_dir" "${cmake_args[@]}"
fi
cmake --build "$build_dir" --parallel 2

cli="$build_dir/bin/sd-cli"
devices="$("$cli" --list-devices)"
echo "$devices"
if ! grep -qE $'\t(cpu|gpu|igpu|accel)\t' <<<"$devices"; then
    echo "Enumerazione dispositivi fallita; installazione esistente non toccata." >&2
    exit 1
fi
if [[ "$require_accelerator" == 1 ]] && ! grep -qE $'\t(gpu|igpu|accel)\t' <<<"$devices"; then
    echo "Nessun acceleratore rilevato: controlla driver e backend." >&2
    echo "Installazione esistente non toccata." >&2
    exit 1
fi

if [[ -n "$destinazione" ]]; then
    case "$(cd "$(dirname "$destinazione")" 2>/dev/null && pwd)/$(basename "$destinazione")" in
        "$repo_root/$build_dir/bin") echo "La destinazione deve essere diversa dalla build." >&2; exit 1 ;;
    esac
    stage="$(mktemp -d)/bin"
    mkdir -p "$stage"
    cp -R "$build_dir/bin/." "$stage/"
    staged="$("$stage/sd-cli" --list-devices)"
    if [[ "$require_accelerator" == 1 ]] && ! grep -qE $'\t(gpu|igpu|accel)\t' <<<"$staged"; then
        echo "La copia temporanea non supera la verifica; installazione non toccata." >&2
        exit 1
    fi
    if [[ -d "$destinazione" ]]; then
        backup="${destinazione%/}.backup-$(date +%Y%m%d-%H%M%S)"
        mv "$destinazione" "$backup"
        echo "Installazione precedente conservata in $backup"
    fi
    mkdir -p "$(dirname "$destinazione")"
    mv "$stage" "$destinazione"
    echo "Motore verificato e installato in $destinazione. Riavvia l'applicazione."
fi
