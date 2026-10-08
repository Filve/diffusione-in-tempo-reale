param(
    [int]$Scelta = 0,
    [string]$Destinazione = ""
)

# Selettore interattivo delle build del motore. Da Windows si possono
# compilare soltanto binari Windows: le voci macOS/Linux spiegano come
# ottenerli senza fingere una cross-compilazione che non esiste.

$ErrorActionPreference = "Stop"
$gpuScript = Join-Path $PSScriptRoot "build-windows-gpu.ps1"

$voci = @(
    @{ Testo = "Windows - solo CPU (pacchetto portabile, varianti x86)"; Backend = "CPU"; Dir = "build-windows-cpu" },
    @{ Testo = "Windows - CPU + Vulkan (NVIDIA/AMD/Intel con driver Vulkan)"; Backend = "Vulkan"; Dir = "build-windows-vulkan" },
    @{ Testo = "Windows - CPU + CUDA (solo NVIDIA, richiede CUDA Toolkit)"; Backend = "CUDA"; Dir = "build-windows-cuda" },
    @{ Testo = "Windows - CPU + CUDA + Vulkan (richiede entrambi gli SDK)"; Backend = "CUDA,Vulkan"; Dir = "build-windows-multi" },
    @{ Testo = "macOS (Apple Silicon, Metal) - istruzioni"; Backend = $null },
    @{ Testo = "Linux (CPU/Vulkan/CUDA) - istruzioni"; Backend = $null }
)

if ($Scelta -lt 1 -or $Scelta -gt $voci.Count) {
    Write-Host "Quale motore vuoi compilare?"
    for ($i = 0; $i -lt $voci.Count; $i++) {
        Write-Host ("  {0}) {1}" -f ($i + 1), $voci[$i].Testo)
    }
    $risposta = Read-Host "Scelta [1-$($voci.Count)]"
    if (-not [int]::TryParse($risposta, [ref]$Scelta) -or $Scelta -lt 1 -or $Scelta -gt $voci.Count) {
        throw "Scelta non valida: '$risposta'."
    }
}

$voce = $voci[$Scelta - 1]

if ($null -eq $voce.Backend) {
    Write-Host ""
    Write-Host "I binari per macOS e Linux non si possono compilare da Windows:"
    Write-Host "ogni sistema operativo compila soltanto i propri eseguibili."
    Write-Host ""
    Write-Host "Per ottenerli:"
    Write-Host "  - sul computer macOS/Linux: ./scripts/build-engine.sh"
    Write-Host "  - oppure, senza quel computer: push di un tag 'portable-v*' su GitHub."
    Write-Host "    Il workflow release-portable.yml produce gli zip per"
    Write-Host "    linux-x86_64, darwin-arm64 e windows-x86_64."
    Write-Host ""
    Write-Host "Il bilanciamento GPU/CPU/RAM/disco e' identico su ogni piattaforma:"
    Write-Host "auto-fit colloca i pesi secondo la memoria disponibile e --memory-guard"
    Write-Host "riduce la residenza dei pesi sotto pressione (vedi docs/performance.md)."
    exit 0
}

if ($env:OS -ne "Windows_NT") {
    throw "Questa voce compila binari Windows e richiede Windows."
}

$parametri = @{ Backend = $voce.Backend; BuildDirectory = $voce.Dir }
if ($Destinazione) { $parametri.Destination = $Destinazione }
& $gpuScript @parametri
