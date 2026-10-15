#!/usr/bin/env bash
# Assemble the Windows package that has both GPU backends in one folder, from
# the three zips that .github/workflows/build.yml builds for every release:
#
#   sd-<branch>-<commit>-bin-win-cuda12-x64.zip    programs + CUDA module (ggml-cuda.dll)
#   cudart-sd-bin-win-cu12-x64.zip                 CUDA runtime DLLs (cudart, cublas, cublasLt)
#   sd-<branch>-<commit>-bin-win-vulkan-x64.zip    programs + Vulkan module (ggml-vulkan.dll)
#
# The package is the CUDA zip, plus the runtime DLLs, plus ggml-vulkan.dll. The
# rest of the Vulkan zip is not copied: it holds the same file names built
# without the CUDA code (stable-diffusion.dll differs), so unpacking one zip over
# the other would leave a mixture of two builds.
#
# Usage: scripts/package_windows_cuda_vulkan.sh <dir_with_the_three_zips> [output_dir]
# Output: <output_dir>/sd-<branch>-<commit>-bin-win-cuda12-vulkan-x64.zip
#         (output_dir defaults to the input directory)

set -euo pipefail

usage() {
    echo "usage: $0 <dir_with_the_three_zips> [output_dir]" >&2
    exit 1
}

fail() {
    echo "error: $*" >&2
    exit 1
}

warn() {
    if [[ "${GITHUB_ACTIONS:-}" == true ]]; then
        echo "::warning::$*"
    else
        echo "warning: $*" >&2
    fi
}

[[ $# -ge 1 && $# -le 2 ]] || usage
for tool in zip unzip; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool is required"
done

[[ -d "$1" ]] || fail "$1 is not a directory"
input_dir="$(cd "$1" && pwd)"
output_dir="${2:-$input_dir}"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

shopt -s nullglob
cuda_zips=("$input_dir"/sd-*-bin-win-cuda12-x64.zip)
vulkan_zips=("$input_dir"/sd-*-bin-win-vulkan-x64.zip)
shopt -u nullglob
cudart_zip="$input_dir/cudart-sd-bin-win-cu12-x64.zip"

[[ ${#cuda_zips[@]} -eq 1 ]] || fail "expected one sd-*-bin-win-cuda12-x64.zip in $input_dir, found ${#cuda_zips[@]}"
[[ ${#vulkan_zips[@]} -eq 1 ]] || fail "expected one sd-*-bin-win-vulkan-x64.zip in $input_dir, found ${#vulkan_zips[@]}"
[[ -f "$cudart_zip" ]] || fail "expected cudart-sd-bin-win-cu12-x64.zip in $input_dir"
cuda_zip="${cuda_zips[0]}"
vulkan_zip="${vulkan_zips[0]}"

# sd-<branch>-<commit>: the two programs zips must come from the same build.
prefix="$(basename "$cuda_zip")"
prefix="${prefix%-bin-win-cuda12-x64.zip}"
vulkan_prefix="$(basename "$vulkan_zip")"
vulkan_prefix="${vulkan_prefix%-bin-win-vulkan-x64.zip}"
[[ "$prefix" == "$vulkan_prefix" ]] || fail "the CUDA and Vulkan zips come from different builds: $prefix and $vulkan_prefix"
commit="${prefix##*-}"
name="$prefix-bin-win-cuda12-vulkan-x64"
output_zip="$output_dir/$name.zip"

entries() {
    unzip -Z1 "$1"
}

overlap="$(comm -12 <(entries "$cuda_zip" | LC_ALL=C sort) <(entries "$cudart_zip" | LC_ALL=C sort))"
[[ -z "$overlap" ]] || fail "the CUDA zip and the CUDA runtime zip both contain: ${overlap//$'\n'/ }"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
root="$work/$name"
mkdir -p "$root"

# -n: never overwrite, and never ask.
unzip -q -n "$cuda_zip" -d "$root"
unzip -q -n "$cudart_zip" -d "$root"
unzip -q -n -j "$vulkan_zip" ggml-vulkan.dll -d "$root" || fail "ggml-vulkan.dll is not in $(basename "$vulkan_zip")"

missing=()
for file in sd-cli.exe sd-server.exe stable-diffusion.dll ggml.dll ggml-base.dll ggml-cuda.dll ggml-vulkan.dll; do
    [[ -f "$root/$file" ]] || missing+=("$file")
done
for pattern in 'ggml-cpu-*.dll' 'cudart64_*.dll' 'cublas64_*.dll' 'cublasLt64_*.dll'; do
    compgen -G "$root/$pattern" >/dev/null || missing+=("$pattern")
done
if ((${#missing[@]})); then
    fail "the package would be incomplete; missing: ${missing[*]}"
fi

# ggml-vulkan.dll was built next to the ggml-base.dll and ggml.dll of the Vulkan
# zip and now runs with the ones of the CUDA zip. Same commit and same options
# give the same size; a different size means a different ggml build.
for shared in ggml-base.dll ggml.dll; do
    vulkan_size="$(unzip -p "$vulkan_zip" "$shared" | wc -c | tr -d ' ')"
    package_size="$(wc -c <"$root/$shared" | tr -d ' ')"
    if [[ "$vulkan_size" != "$package_size" ]]; then
        warn "$shared is $package_size bytes in the CUDA zip and $vulkan_size bytes in the Vulkan zip: the two builds may not share the same ggml"
    fi
done

[[ ! -e "$root/LEGGIMI.txt" ]] || fail "the zips already contain a LEGGIMI.txt, which would be overwritten"
cat >"$root/LEGGIMI.txt" <<EOF
${name}
Windows: NVIDIA (CUDA 12) e Vulkan nella stessa cartella.
Nessuna installazione: scompatta lo zip e avvia i programmi da questa cartella.

Che cosa contiene (versione ${prefix}, commit ${commit})
  sd-cli.exe, sd-server.exe, stable-diffusion.dll
      il motore.
  ggml-cuda.dll, cudart64_*.dll, cublas64_*.dll, cublasLt64_*.dll
      backend NVIDIA (CUDA 12) e le sue DLL, di NVIDIA, ridistribuite con i
      termini del CUDA Toolkit. Serve il driver NVIDIA aggiornato.
  ggml-vulkan.dll
      backend Vulkan: schede NVIDIA, AMD (anche la grafica integrata) e Intel.
      Vulkan arriva con il driver grafico del produttore.
  ggml-cpu-*.dll
      backend CPU: il motore sceglie la variante adatta al processore.
I backend sono moduli separati, caricati insieme se il driver c'e': non vanno
in conflitto. Tieni tutti i file nella stessa cartella e non mescolarli con
quelli di un'altra versione.

Controllo (PowerShell, dentro la cartella)
  .\sd-cli.exe --version         deve finire con "commit ${commit}"
  .\sd-cli.exe --list-devices    deve elencare CUDA0 (la scheda NVIDIA), i
                                 dispositivi Vulkan e la CPU

La stessa scheda NVIDIA compare due volte, come CUDA0 e come Vulkan0: e' una
scheda sola vista da due backend. Il pianificatore automatico (auto-fit) la
conta una volta e la fa usare con CUDA. Una GPU integrata (igpu) usa la RAM del
computer: il pianificatore la lascia da parte finche' una scheda ha memoria
libera.

Opzioni utili
  --profile auto         sceglie tra max, balanced e lite in base a dimensione
                         del modello, RAM libera e memoria della GPU
  --diffusion-fa         flash attention per il modello di diffusione
  --type q8_0            quantizza i pesi al caricamento: occupano poco piu' della
                         meta' di f16
  --memory-guard 90      libera pesi dalla memoria prima che finisca
  --backend, --params-backend
                         scegli tu dove girano i moduli e dove stanno i pesi
                         (con --params-backend l'auto-fit non pianifica)
Le opzioni indicate a mano battono sempre quelle del profilo.

Se --list-devices mostra solo la CPU: aggiorna il driver della scheda video e
non spostare i file ggml-* e le DLL di CUDA fuori da questa cartella. Il
programma scrive nel log perche' un modulo GPU non e' stato caricato.

Su Windows serve Microsoft Visual C++ Redistributable 2015-2022 (x64),
versione 14.44 o successiva. Se manca o e' piu' vecchio, Windows chiude il
programma subito, senza scrivere nulla: il codice di uscita e' 0xC0000135
(3221225781, oppure -1073741515 in PowerShell). Controlla la versione con:
  (Get-Item C:\Windows\System32\msvcp140.dll).VersionInfo.FileVersion
e installa l'ultima da https://aka.ms/vc14/vc_redist.x64.exe

I modelli NON sono inclusi e hanno licenze proprie.
Basato su stable-diffusion.cpp (MIT, leejet); modifiche del fork di
Francesco Simeoni (MIT). Licenze: stable-diffusion.cpp.txt e ggml.txt.
EOF

rm -f "$output_zip"
(cd "$work" && zip -q -r -D "$output_zip" "$name")

# Every file of the CUDA and runtime zips, the Vulkan module and LEGGIMI.txt.
expected=$(($(entries "$cuda_zip" | wc -l) + $(entries "$cudart_zip" | wc -l) + 2))
actual="$(entries "$output_zip" | wc -l)"
if [[ "$actual" -ne "$expected" ]]; then
    rm -f "$output_zip"
    fail "the zip has $actual files, expected $expected"
fi

echo "created: $output_zip ($(du -h "$output_zip" | cut -f1), $actual files)"
unzip -l "$output_zip"
