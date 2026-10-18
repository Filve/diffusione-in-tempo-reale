# Example

The following example starts `sd-server` with a standalone diffusion model, VAE, and LLM text encoder:

```
.\bin\Release\sd-server.exe --diffusion-model  ..\models\diffusion_models\z_image_turbo_bf16.safetensors --vae ..\models\vae\ae.sft  --llm ..\models\text_encoders\qwen_3_4b.safetensors --diffusion-fa --offload-to-cpu -v --cfg-scale 1.0
```

What this example does:

* `--diffusion-model` selects the standalone diffusion model
* `--vae` selects the VAE decoder
* `--llm` selects the text encoder / language model used by this pipeline
* `--diffusion-fa` enables flash attention in the diffusion model
* `--offload-to-cpu` reduces VRAM pressure by keeping weights in RAM when possible
* `-v` enables verbose logging (equivalent to `--log-level verbose`)
* `--cfg-scale 1.0` sets the default CFG scale for generation

Logging defaults to `info`. Use `--log-level <level>` to select `debug`, `verbose`,
`info`, `warn`, or `error` (from most to least detailed). Each level includes
messages at that level and all less detailed levels. `-v` and `--verbose` are
equivalent to `--log-level verbose`. If repeated, the last logging option wins.

After the server starts successfully:

* the web UI is available at `http://127.0.0.1:1234/`
* the native async API is available under `/sdcpp/v1/...`
* the compatibility APIs are available under `/v1/...` and `/sdapi/v1/...`

If you want to use a different host or port, pass:

```bash
--listen-ip <ip> --listen-port <port>
```

# When the memory runs short

A request that makes the machine run out of memory does not stop the server. It frees the model, loads it again
with a lighter configuration (profiles `max > balanced > lite > lite-disk > lite-cpu`, see `--profile`) and runs the
same request again; from `lite` on, the VAE is decoded in tiles. The request is then served normally. When even the
lightest profile does not fit, the request is answered with `503` and a `Retry-After` header, and the server stays up.
After a quiet period the server climbs back one profile at a time.

* `--degradation-cooldown <seconds>` sets how long the server waits before climbing back (default `300`; `0` = stay
  on the lighter profile until the next restart)
* `--disable-degradation` turns this off: a request that runs out of memory is answered with the error at once
* `GET /sdcpp/v1/health` tells which profile is in use, how many times the server had to step down and the last
  error; `GET /sdcpp/v1/resources` reports the free memory of the machine and of each device

If the model does not fit at startup, the server starts on the first profile that loads it (the log says which). When
the model cannot be loaded at all, or the address cannot be used, `sd-server` exits with a non-zero code and says why.
See [Failures and recovery](api.md#failures-and-recovery) for the answers a client gets.

# Frontend

## Build with Frontend

The server can optionally build the web frontend and embed it into the binary as `gen_index_html.h`.

### Requirements

Install the following tools:

* **Node.js** ≥ 20
  https://nodejs.org/

* **pnpm** ≥ 10
  Install via npm:

```bash
npm install -g pnpm
```

Verify installation:

```bash
node -v
pnpm -v
```

### Install frontend dependencies

Go to the frontend directory and install dependencies:

```bash
cd examples/server/frontend
pnpm install
```

### Build the server with CMake

Enable the frontend build option when configuring CMake:

```bash
cmake -B build -DSD_SERVER_BUILD_FRONTEND=ON
cmake --build build --config Release
```

If `pnpm` is available, the build system will automatically run:

```
pnpm run build
pnpm run build:header
```

and embed the generated frontend into the server binary.

## Frontend Repository

The web frontend is maintained in a **separate repository**, https://github.com/leejet/sdcpp-webui.

If you want to modify the UI or frontend logic, please submit pull requests to the **frontend repository**.

This repository (`stable-diffusion.cpp`) only vendors the frontend periodically. Changes from the frontend repo are synchronized:

* approximately **every 1–2 weeks**, or
* when there are **major frontend updates**

Because of this, frontend changes will **not appear here immediately** after being merged upstream.

## Using an external frontend

By default, the server uses the **embedded frontend** generated during the build (`gen_index_html.h`).

You can also serve a custom frontend file instead of the embedded one by using:

```bash
--serve-html-path <path-to-index.html>
```

For example:

```bash
sd-server --serve-html-path ./index.html
```

In this case, the server will load and serve the specified `index.html` file instead of the embedded frontend. This is useful when:

* developing or testing frontend changes
* using a custom UI
* avoiding rebuilding the binary after frontend modifications

# Usage

For detailed command-line arguments, run:

```bash
./bin/sd-server -h
```

For completely black or white images or videos, NaNs, and the `--linear-scale` /
`--attn-scale` startup options, see [Troubleshooting](../../docs/troubleshooting.md).
