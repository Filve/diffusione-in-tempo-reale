# Backend selection

`stable-diffusion.cpp` has two backend assignments:

- `--backend` selects the runtime backend used to execute model graphs.
- `--params-backend` selects where model parameters are kept.

If `--params-backend` is not set, auto-fit chooses parameter placement. With
`--auto-fit off`, parameters use the same backend as their module runtime backend.

## Syntax

A backend assignment can be a single backend name:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cpu
```

This applies to every module that does not have a more specific assignment.

Assignments can also target individual modules:

```shell
sd-cli -m model.safetensors -p "a cat" --backend te=cpu,vae=cuda0,diffusion=vulkan0
```

The same syntax is used for parameter placement:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cuda0 --params-backend te=cpu,vae=cpu
```

`--params-backend` also accepts the special value `disk`:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cuda0 --params-backend disk
```

`--max-vram` can target resolved backend/device names:

```shell
sd-cli -m model.safetensors -p "a cat" --backend diffusion=cuda0,vae=vulkan0 --max-vram cuda0=6,vulkan0=2
```

The value is a shared per-device budget for managed weights and registered
runner compute/cache buffers. Live free memory can lower the effective limit
for each graph run. Driver contexts and allocations made outside the managed
model runners are not part of this accounting, so it is not a hard physical
VRAM cap.

Module names are case-insensitive. Hyphens and underscores in module names are ignored, so `clip_vision`, `clip-vision`, and `clipvision` are equivalent.

`all=`, `default=`, and `*=` can be used to set the default backend inside a mixed assignment:

```shell
sd-cli -m model.safetensors -p "a cat" --backend all=cuda0,te=cpu
```

## Multiple devices per module (layer split)

A `--backend` module assignment can list several devices separated by `&`:

```shell
sd-cli -m model.safetensors -p "a cat" --backend "diffusion=cuda0&cuda1"
```

The module's transformer blocks are then distributed across the listed devices
in contiguous ranges: the first device takes blocks until its free memory (minus a
compute-buffer headroom of about 2 GiB, and any `--max-vram` limit) is used up,
then the next device, and so on. The module's graphs are executed with a
`ggml_backend_sched` that runs each block
on the device holding its weights, copying the residual stream at the range
boundaries. The first device in the list is the module's main device: it also
holds the non-block tensors (embeddings, final norms, small sub-runners such as
CLIP models or projectors) and the graph inputs/outputs.

Layer split is supported for the `diffusion` and `te` modules. For `te` it
applies to the dominant text encoder (`t5xxl` or the LLM); other modules accept
only a single device. If the module has no recognizable transformer blocks, the
assignment falls back to the first listed device.

`--params-backend` accepts no device lists. If the module has no explicit
params assignment, each block range's parameters are loaded directly to (and,
with `--params-backend diffusion=disk`, released directly from) its own device;
an explicit assignment such as `te=cpu` keeps the parameters on that backend
and stages each range to its device on demand.

Layer split uses the fixed graph-cut plan to assign blocks across devices, but
single-device segmented execution and next-segment prefetch are disabled for
the split module. `--max-vram` can still provide the per-device limits used by
layer split and auto-fit.

Use `--list-devices` to see the device names available on the system. Each line
reports `name`, device type (`cpu`, `gpu`, `igpu`, `accel`), description, and
free/total memory when the backend reports it. Host applications can query the
same information through `sd_get_device_count()` / `sd_get_device_info()` in
the C API.

## Diagnosing a GPU that does not show up

With dynamic backends (`GGML_BACKEND_DL`), ggml loads the GPU modules silently in
Release builds, and on Windows it never reports the operating system error. After
the first load, this fork logs on stderr, for each GPU module found next to the
executable or in the working directory that provides no device, why it could not be
used (the `dlopen` error on Linux/macOS; on Windows the error code plus the DLLs the
module imports that cannot be found, such as `cublas64_12.dll`), with a hint. When no
GPU module is present at all it says where it looked. These are warnings and notes,
not errors: the CPU backend keeps working.

## CPU fallback

If a requested backend is unavailable at context creation (for example a
binary built with CUDA running on a machine without a CUDA device, or a device
that fails to initialize), the context falls back to the `cpu` backend with a
warning instead of failing. Explicit `--params-backend` and `--split-mode`
assignments are discarded in that case. Pass `--disable-backend-fallback`
(or set `disable_backend_fallback` in `sd_ctx_params_t`) to fail instead.

### Row split (`--split-mode row`)

`--split-mode` selects how a multi-device module distributes its weights:
`layer` (the default, described above) or `row`. It accepts a single mode or
per-module assignments:

```shell
sd-cli -m model.safetensors -p "a cat" --backend "diffusion=cuda0&cuda1" --split-mode row
sd-cli -m model.safetensors -p "a cat" --backend "diffusion=cuda0&cuda1,te=cuda0&cuda1" --split-mode diffusion=row,te=layer
```

In row mode the module keeps executing on its main (first listed) device, but
its transformer-block matmul weights are allocated in the backend's row-split
buffer type, which slices each weight's rows across the listed devices in
proportion to free memory and runs those matmuls on all devices in parallel.
Compared to a layer split this uses all GPUs within every layer (instead of
sequentially device by device) at the cost of a cross-device reduction per
matmul - usually the faster option when the devices have fast interconnect.

Row split requires a compatible split-buffer export from the linked GGML
backend. If it is unavailable (or the listed devices belong to different backend
registries), the module falls back to a layer split.
Embeddings, normalization weights, biases and other non-block tensors stay in
regular buffers on the main device.

Row-split execution can use graph segments, but split weights are loaded
synchronously instead of using the normal single-device prefetch path. Because
GGML does not expose exact shard allocation sizes, the managed budget currently
counts a split buffer's full size on each participating device. This is a
conservative bound and can reject otherwise feasible layouts.

Direct ("immediately") LoRA application cannot patch row-split tensors; with
`--split-mode row` the automatic LoRA mode selects runtime application, and an
explicit `--lora-apply-mode immediately` skips the split tensors with a
warning.

## Automatic placement (`--auto-fit on|off`)

`--auto-fit` requires `on` or `off` and defaults to `on` when omitted.
Explicit `--params-backend` assignments disable auto-fit,
regardless of argument order, even with `--auto-fit on`.

Auto-fit preserves explicit `--backend` assignments, including per-module
assignments and device lists. For modules without a runtime assignment, it chooses
the GPU with the largest available memory budget (the first device on a tie); with
several GPUs of the same backend it can also move or split modules across them, see
[Several GPUs](#several-gpus---multi-gpu-autoonoff). It then derives parameter
placements from the model metadata, each module's compute devices, and the remaining
memory budgets. The chosen backend specifications are printed.

```shell
sd-cli -m model.safetensors -p "a cat" --auto-fit on
sd-cli -m model.safetensors -p "a cat" --auto-fit on --max-vram cuda0=8,cuda1=14
sd-cli -m model.safetensors -p "a cat" --backend cuda0
sd-cli -m model.safetensors -p "a cat" --backend diffusion=cuda0,te=cpu,vae=cuda1
sd-cli -m model.safetensors -p "a cat" --auto-fit off
```

Budgets reuse `--max-vram`: a positive per-device value caps what auto-fit
plans with on that device, a negative value means "free memory minus that many
GiB", and with no budget set each device's free memory minus a 512 MiB margin
is used. These resolved GPU budgets, including the safety margin, also drive
the runner's graph-cut capacity checks.

Runtime capacity checks also leave 512 MiB of currently free device memory for
backend scratch buffers and pipelines, including with explicit backend assignments.
They cap free-memory reports by the device's total memory minus tracked
resident allocations. Vulkan reports exceeding total memory are rejected because
its heap-budget subtraction can underflow. Other backends use the cap instead of
treating such reports as zero free memory. Failed checks log the reported free and
total memory alongside tracked weight and runtime allocations.
With `--mmap`, device-backed mappings count toward these budgets at their full
mapped-file size, once per device buffer even when multiple parameter blocks
share it. Mappings retained in the loader cache continue to count.

Components are considered in `diffusion`, `te`, `vae` order so that repeatedly
used diffusion weights have priority. Each component's weights use the first
storage location with enough remaining budget:

1. The component's compute GPU, leaving estimated space for computation and weight staging.
2. CPU RAM, reserving the larger of 2 GiB or 10% of available RAM for other work.
3. Another GPU, choosing the one with the largest remaining budget that fits.
4. Disk, reloading weights on demand.

GPU cache space follows the same component priority. Before a lower-priority
component can become permanently resident, the planner leaves room for the full
weights and estimated compute space of higher-priority offloaded components.
If offloaded diffusion already needs the entire main GPU budget, TE and VAE also
use offloaded parameters. Their GPU copies can then be released after their
phases, leaving more room to reuse diffusion weights across sampling steps.
CPU parameter residency allows GPU weight caching; it does not force every
weight to be copied again at every step.

RAM and GPU budgets are shared across components. Each component uses a single
parameter backend; several other GPUs' capacities are not combined to store
one component (a module that auto-fit splits across GPUs, see below, keeps its
weights on those GPUs instead). If available RAM cannot be queried, RAM residency
is skipped.

Without a GPU (a CPU-only machine, or every GPU backend unavailable) the same
planning runs against the CPU: the diffusion model, text encoder and VAE are
considered in the usual order, and each one stays in RAM if its weights fit in
what is left of the available RAM after the 2 GiB / 10% margin and the largest
compute reserve (computation also needs RAM). A component that does not fit
reads its weights from disk, and auto-fit then turns the memory guard on at 90%
(unless `--memory-guard` is set) so that the graph runs in segments and weights
are released between them. When everything fits, the plan changes nothing and
the log says `auto-fit: CPU only; the weights fit in RAM`. If available RAM
cannot be queried, the default backend and placement are used.
Weights stored on another GPU are copied to the component's compute devices for
execution. CPU modules use RAM or disk. Compute reserves and cache priority are
accounted for separately on each device, so a CPU module does not reserve GPU
space. Storage on another module's GPU also leaves room for that module's work.

Explicit device lists and `--split-mode` still control the computation of the
modules they name. Before the runners have built their split plans, auto-fit
conservatively counts the full component size on each listed GPU when checking
residency and cache space. This can offload parameters even when a split layout
would fit; use `--auto-fit off` to keep the default split-device parameter
placement, or leave the module out of `--backend` and let `--multi-gpu` split it.

For example, a diffusion model whose full weights exceed the main GPU's budget
can use `--backend diffusion=cuda0 --params-backend diffusion=cpu` when RAM is
sufficient. Automatic graph segmentation can then load the required weights
for each segment and reclaim idle GPU copies. `--disable-segmented-compute`
still disables segmentation.

Initial compute reserves are estimates (2 GiB for diffusion and text encoders,
1 GiB for VAE); higher-priority placements also leave staging space for the
largest weight tensor of each lower-priority offloaded component. Actual segment
weights, compute buffers and caches must
still fit the runner's capacity checks. Offloading weights does not guarantee
that every resolution or frame count will fit, and auto-fit does not change a
component to CPU computation solely because its full weights exceed VRAM.
If a VAE decode fails, decoding retries with spatial tiling even when `--auto-fit`
is off; supported video decoders try temporal tiling first and can then add
spatial tiling. Spatial retries use half-size tiles along each latent dimension.

### Several GPUs (`--multi-gpu auto|on|off`)

With more than one discrete GPU of the same backend (`CUDA0` and `CUDA1`, or
`Vulkan0` and `Vulkan1`), auto-fit also decides which of them runs which module.
`--multi-gpu` selects how far it goes and defaults to `auto`:

| Value | Behavior |
| --- | --- |
| `auto` | Everything that fits stays on the GPU with the largest budget. A module that does not fit there next to the modules placed before it moves to the first other GPU that can hold it (in practice the text encoder or the VAE, because the diffusion model has priority). A diffusion model or text encoder that fits on no single GPU is split across the fewest GPUs that can hold it, as with `--backend "diffusion=cuda0&cuda1"`. |
| `on` | As `auto`, and in addition the text encoder and the VAE run on a GPU other than the one running the diffusion model even when everything would fit on one, so the diffusion model keeps its GPU's memory for itself. |
| `off` | One GPU, as before this option existed. |

```shell
sd-cli -m model.safetensors -p "a cat"                    # auto: uses the second GPU only if the model needs it
sd-cli -m model.safetensors -p "a cat" --multi-gpu on     # also puts the text encoder and the VAE on the second GPU
sd-cli -m model.safetensors -p "a cat" --multi-gpu off    # only the best GPU
```

The plan the log prints shows the result: `split:` lines give the share of the
weights on each GPU, and `auto-fit:` notes say which module moved and why, or
that the other GPUs stay idle because everything already fits on one.

What it does and does not do:

- Only discrete GPUs of the same backend are combined. Integrated GPUs share the
  system RAM, and a CUDA GPU cannot share a module with a Vulkan one; in that case
  the log says so and only the best GPU is used unless modules are assigned with `--backend`.
- A card that two backends list counts once. With the CUDA and the Vulkan builds
  installed side by side, an NVIDIA card is both `CUDA0` and `Vulkan0`. Listings that
  report the same PCI bus id are one GPU: the vendor's backend keeps it, and the Vulkan
  or OpenCL listing appears in the plan as `same GPU as CUDA0, not used automatically`.
  It can still be named with `--backend`, and then the run goes through that backend, usually
  slower on the same card; the log says so with an `auto-fit:` warning, once per listing.
  A backend that reports no PCI bus id is not matched.
  See [cuda_e_vulkan_stessa_scheda.md](./cuda_e_vulkan_stessa_scheda.md) (Italian) for the log lines, measured times and what to do.
- Weights that fit neither the compute device nor the RAM budget stay on disk (`params disk` in
  the plan) and are read again every time their module runs, the diffusion model at every sampling
  step. The log then warns with the size of those weights, the free RAM and the budget for weights,
  and suggests closing programs, `--type` or a quantized model file. No warning when the free RAM is
  unknown or with `--cpu-only`.
- An integrated GPU is the main device only when no graphics card has memory left.
  Its memory is the system RAM, so it reports more than a card does, and it computes
  several times slower. Weights are never parked in it either, because that is the
  same RAM the plan has just found too small.
- A module assigned with `--backend` is never moved or split, and a module whose
  `--split-mode` is `row` is never split automatically. An explicit
  `--params-backend` disables auto-fit, and with it this planning.
- The split is a layer split: whole transformer blocks run on one GPU after
  another. It lets a model that does not fit in one GPU's memory stay in VRAM
  instead of streaming from RAM or disk, but a single image is not generated faster
  by a second GPU. To use several GPUs for throughput, run one process (or
  `sd-server`) per GPU, each with its own `--backend cudaN`.
- The planner is conservative about a split: the free memory of the GPUs, less the compute
  reserve of each, must exceed the weights by 10% plus 256 MiB for every boundary between
  GPUs, because blocks cannot be cut exactly where the memory ends. When it does not,
  the module keeps its old placement (parameters in RAM or on disk).
- `--profile auto` counts the free memory of the GPUs together (less 2 GiB of
  working memory for each GPU after the first) when `--multi-gpu` is not `off`.

## Modules

| Module | Purpose | Accepted names |
| --- | --- | --- |
| `diffusion` | UNet, DiT, MMDiT, Flux, Wan, Qwen Image, and other diffusion models | `diffusion`, `model`, `unet`, `dit` |
| `te` | Text encoders and conditioners | `te`, `clip`, `text`, `textencoder`, `textencoders`, `conditioner`, `cond`, `llm`, `t5`, `t5xxl` |
| `clip_vision` | CLIP vision encoder | `clip_vision`, `clipvision`, `clip-vision`, `vision` |
| `vae` | VAE and TAE | `vae`, `firststage`, `autoencoder`, `tae` |
| `controlnet` | ControlNet | `controlnet`, `control` |
| `photomaker` | PhotoMaker ID encoder and PhotoMaker LoRA | `photomaker`, `photomakerid`, `pmid`, `photo` |
| `upscaler` | ESRGAN upscaler | `upscaler`, `esrgan`, `hires` |
| `detector` | ADetailer YOLOv8 detector | `detector`, `adetailer`, `yolo` |

`te` is the preferred module name for text encoders. `clip` is kept as an accepted alias because many existing commands and model names use CLIP terminology.

## Backend names

Backend names are resolved against the GGML backend device list. Matching is case-insensitive and accepts exact names or unique prefixes, so common values include names such as:

- `cpu`
- `cuda0`
- `vulkan0`
- `metal`

The special values `auto`, `default`, and an empty backend name select the default backend. The default preference is GPU, then integrated GPU, then CPU.

The special value `gpu` selects the first GPU backend, falling back to the first integrated GPU backend.

The special value `disk` is accepted only by `--params-backend`. `--backend disk` is invalid because `disk` is a parameter residency mode, not a runtime compute backend.

## Runtime backend vs. parameter backend

The runtime backend controls where graph execution runs. The parameter backend controls where model weights are allocated or whether they are reloaded from disk on demand.

For example:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cuda0 --params-backend cpu
```

This runs all modules on `cuda0`, but stores parameters in CPU RAM. During execution, parameters are moved to the runtime backend as needed.

For example:

```shell
sd-cli -m model.safetensors -p "a cat" --backend cuda0 --params-backend disk
```

This runs all modules on `cuda0`, reloads parameters from the model file as needed, and releases those parameter buffers after use.

Outside `--auto-fit`, `disk` is never selected implicitly. If `--params-backend` is not set, parameters use the runtime backend.

Per-module assignments can be mixed:

```shell
sd-cli -m model.safetensors -p "a cat" --backend diffusion=cuda0,te=cpu,vae=cpu --params-backend diffusion=cuda0,te=cpu,vae=cpu
```

This keeps text encoding and VAE execution on CPU while the diffusion model runs on GPU.

## Backend sharing and lifetime

Backends are managed by `SDBackendManager`.

Within one manager, backend instances are cached by resolved backend device name. If multiple modules request the same backend, they share the same `ggml_backend_t`.

For example:

```shell
--backend te=cpu,vae=cpu
```

uses one shared CPU backend for both `te` and `vae` runtime execution.

Runtime and parameter assignments also share the same backend cache. If `--backend diffusion=cuda0` and `--params-backend diffusion=cuda0` resolve to the same device, both use the same backend instance.

`--params-backend disk` does not create a separate backend instance. Parameters are loaded lazily using the module runtime backend.

`SDBackendManager` owns the backend instances and frees them when the context or upscaler is destroyed. Model runners receive non-owning runtime and parameter backend pointers and do not free them.

## Compatibility flags

The example CLI/server still accepts these older CPU placement flags as compatibility aliases:

- `--clip-on-cpu`
- `--vae-on-cpu`
- `--control-net-cpu`
- `--offload-to-cpu`

`--clip-on-cpu`, `--vae-on-cpu`, and `--control-net-cpu` are deprecated. The example argument layer prepends `te=cpu`, `vae=cpu`, and `controlnet=cpu` to `--backend` before creating the context.

`--offload-to-cpu` prepends a CPU default to the parameter assignment in the caller before creating the context:

```shell
--params-backend '*=cpu'
```

Because this default is inserted first, later explicit `--params-backend` entries can still override it, for example `--offload-to-cpu --params-backend te=disk` keeps non-TE parameters on CPU and reloads TE parameters from disk.

Library callers should set `backend` and `params_backend` directly. `sd_ctx_params_init()`
enables `auto_fit` by default; a nonempty `params_backend` assignment disables it.
The `backend` assignment constrains auto-fit's compute placement.
The old CPU/offload fields are no longer part of the C API. Explicit `--backend` and
`--params-backend` assignments are preferred for new commands.
