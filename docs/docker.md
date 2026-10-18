# Docker

The container images are built from this repository by the CI
(`.github/workflows/build.yml`) and published to the GitHub Container Registry
as `ghcr.io/filve/diffusione-in-tempo-reale`, so they always contain the code of
this fork. Every push to `master` updates these tags:

| Tag | Backend | Platform | Notes |
|---|---|---|---|
| `master-cpu` | CPU only | `linux/amd64` | no GPU needed |
| `master-cuda` | CUDA 12 | `linux/amd64` | NVIDIA GPUs; start it with `--gpus all` (needs the NVIDIA Container Toolkit) |
| `master-vulkan` | Vulkan | `linux/amd64` | AMD and Intel GPUs through Mesa; pass the GPU with `--device /dev/dri` |
| `master-musa` | MUSA | `linux/amd64` | Moore Threads GPUs |
| `master-sycl` | SYCL | `linux/amd64` | Intel GPUs |
| `master-cuda-spark` | CUDA 13 | `linux/arm64` | NVIDIA DGX Spark |

The examples below use the CPU image; swap the tag for the one that matches your
hardware.

## Run CLI

```shell
docker run --rm -v /path/to/models:/models -v /path/to/output/:/output ghcr.io/filve/diffusione-in-tempo-reale:master-cpu [args...]
# For example
# docker run --rm -v ./models:/models -v ./build:/output ghcr.io/filve/diffusione-in-tempo-reale:master-cpu -m /models/sd-v1-4.ckpt -p "a lovely cat" -v -o /output/output.png
```

Check that an NVIDIA GPU is visible to the container before generating:

```shell
docker run --rm --gpus all ghcr.io/filve/diffusione-in-tempo-reale:master-cuda --list-devices
```

## Run server

```shell
docker run --rm --init -v /path/to/models:/models -v /path/to/output/:/output -p "1234:1234" --entrypoint "/sd-server" ghcr.io/filve/diffusione-in-tempo-reale:master-cpu [args...]
# For example
# docker run --rm --init -v ./models:/models -v ./build:/output -p "1234:1234" --entrypoint "/sd-server" ghcr.io/filve/diffusione-in-tempo-reale:master-cpu -m /models/sd-v1-4.ckpt -p "a lovely cat" -v -o /output/output.png
```

## Building using Docker

```shell
docker build -f docker/Dockerfile -t sd .
```

## Building variants using Docker

### Vulkan:

```shell
docker build -f docker/Dockerfile.vulkan -t sd .
```

### ROCm (AMD GPU):

Defining one or multiple targets with `--build-arg AMDGPU_TARGETS` is strongly recommended.
Building the image will otherwise likely fail or build for a target that does not fit your
hardware.

```shell
# Build for one target
docker build -f docker/Dockerfile.rocm --build-arg "AMDGPU_TARGETS=gfx1100" -t sd .
# Build for multiple targets
docker build -f docker/Dockerfile.rocm --build-arg "AMDGPU_TARGETS=gfx1030;gfx1100" -t sd .
```

## Run locally built image's CLI

```shell
docker run --rm -v /path/to/models:/models -v /path/to/output/:/output sd [args...]
# For example
# docker run --rm -v ./models:/models -v ./build:/output sd -m /models/sd-v1-4.ckpt -p "a lovely cat" -v -o /output/output.png
```

### ROCm (AMD GPU)

The user that executes this command needs to be in the `video` group and have read and write
access to `/dev/dri` and `/dev/kfd`.

`--device` and `--group-add` allow the container access to the GPU.

```shell
docker run --rm --device=/dev/dri --device=/dev/kfd --group-add video \
  -v /path/to/models:/models -v /path/to/output/:/output \
  sd [args...]
# For example
# docker run --rm --device=/dev/dri --device=/dev/kfd --group-add video \
#   -v ./models:/models -v ./build:/output \
#   sd -m /models/sd-v1-4.ckpt -v -o /output/output.png \
#   -p "a lovely cat"
```