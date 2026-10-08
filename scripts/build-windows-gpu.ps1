param(
    [ValidateSet("CPU", "Vulkan", "CUDA", "CUDA,Vulkan")]
    [string]$Backend = "Vulkan",
    [string]$BuildDirectory = "build-windows-gpu",
    [string]$Destination = ""
)

# With -Backend CPU the build is CPU-only and deployment does not require an
# accelerator; GPU backends keep requiring a successful accelerator check.
$requireAccelerator = $Backend -ne "CPU"

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$build = [System.IO.Path]::GetFullPath((Join-Path $root $BuildDirectory))

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    $ErrorActionPreference = "Continue"
    & $Program @Arguments
    if (-not $? -or $LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE."
    }
}

function Read-Devices([string]$Program) {
    $ErrorActionPreference = "Continue"
    $output = @(& $Program --list-devices 2>&1 | ForEach-Object { $_.ToString() })
    $code = $LASTEXITCODE
    $output | ForEach-Object { Write-Host $_ }
    if ($code -ne 0 -or -not ($output | Select-String "`t(cpu|gpu|igpu|accel)`t")) {
        throw "Device enumeration failed in $Program. Existing installation has not been replaced."
    }
    if ($requireAccelerator -and -not ($output | Select-String "`t(gpu|igpu|accel)`t")) {
        throw "No accelerator detected in $Program. Check drivers and backend DLLs. Existing installation has not been replaced."
    }
}

$arguments = @("-S", $root, "-B", $build, "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release", "-DSD_BUILD_SHARED_GGML_LIB=ON",
    "-DBUILD_SHARED_LIBS=ON", "-DGGML_BACKEND_DL=ON",
    "-DGGML_NATIVE=OFF", "-DGGML_CPU_ALL_VARIANTS=ON",
    "-DSD_CUDA=OFF", "-DSD_VULKAN=OFF")

if ($Backend.Contains("CUDA")) {
    if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
        throw "CUDA Toolkit missing from PATH. Install a Toolkit compatible with your GPU and MSVC, then reopen Developer PowerShell."
    }
    $arguments += "-DSD_CUDA=ON"
}
if ($Backend.Contains("Vulkan")) {
    if (-not $env:VULKAN_SDK -or -not (Test-Path "$env:VULKAN_SDK\Include\vulkan\vulkan.h")) {
        throw "Vulkan SDK missing. Install the SDK, then reopen Developer PowerShell so VULKAN_SDK is set."
    }
    $arguments += "-DSD_VULKAN=ON"
}

Invoke-Checked "cmake" $arguments
Invoke-Checked "cmake" @("--build", $build, "--parallel", "2")
$cli = Join-Path $build "bin\sd-cli.exe"
Read-Devices $cli

if ($Destination) {
    $target = [System.IO.Path]::GetFullPath($Destination)
    if ($target.TrimEnd('\') -eq [System.IO.Path]::GetPathRoot($target).TrimEnd('\') -or
        $target.TrimEnd('\') -eq $HOME.TrimEnd('\') -or
        $target.TrimEnd('\') -eq $root.TrimEnd('\')) {
        throw "Destination must be a dedicated engine directory, not a drive, home or repository root."
    }
    if ($target -eq (Join-Path $build "bin")) {
        throw "Destination must differ from the build output directory."
    }
    $running = Get-CimInstance Win32_Process | Where-Object {
        $_.ExecutablePath -and
        [System.IO.Path]::GetDirectoryName($_.ExecutablePath) -eq $target -and
        $_.Name -in @("sd-cli.exe", "sd-server.exe")
    }
    if ($running) {
        throw "Engine still running in destination. Close the host application before deployment."
    }
    $stage = "$target.staging-$([Guid]::NewGuid().ToString('N'))"
    $backup = "$target.backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    Copy-Item -Path (Join-Path $build "bin\*") -Destination $stage -Recurse -Force
    Read-Devices (Join-Path $stage "sd-cli.exe")
    if (Test-Path $target) { Move-Item -Path $target -Destination $backup }
    try {
        Move-Item -Path $stage -Destination $target
    } catch {
        if (Test-Path $backup) { Move-Item -Path $backup -Destination $target }
        throw
    }
    Write-Host "Validated engine copied to $target. Restart the host application to use it."
    if (Test-Path $backup) { Write-Host "Previous engine preserved in $backup." }
}
