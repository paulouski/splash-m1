# Development

Splash M1 is an unofficial fork based on Inco's Splash and paperniuk's Apple7 kernel work. The macOS 15 build targets Metal 3.2 and Apple7/8 GPUs. Its supported public model configuration is text-only Qwen3.8-27B with an MLX affine group-64 checkpoint using uniform Q4 or mixed Q4/Q5 tensors, F16 or BF16 target tensors, a matching BF16 DFlash draft, and INT8 KV.

This build excludes GGUF targets, MoE models, vision, and BF16 KV. The reference test machine is an M1 Max with a 24-core GPU and 32 GB memory on macOS 15.7. M2 support is intended but unverified; Apple9 and newer GPUs are not a supported guarantee.

## Build from source

Requirements: macOS 15 or newer, Apple Silicon, Xcode with Metal compiler tools available through xcrun, and Python 3.12–3.14. The commands use python3.13; substitute python3.12 or python3.14 if that is your installed supported interpreter.

~~~sh
git clone https://github.com/paulouski/splash-m1.git
cd splash-m1
python3.13 -m venv .venv
.venv/bin/python -m pip install -r install/requirements.txt
make MACOS15=1 BUILD=build
~~~

The build produces build/splash and build/splash.metallib. Run the source launcher with:

~~~sh
./splash serve --model mlx-community/Qwen3.8-27B-4bit --language-only --max-context 32K
~~~

The source launcher uses the checkout's Python environment when available. The first run downloads the selected model and matching draft from Hugging Face and prepares the weights. Prepared model weights are stored in ~/Library/Caches/Splash/weights by default; set SPLASH_WEIGHT_CACHE to choose another location.

The model configuration currently supported by this build is text-only. Image and PDF input are not available in the macOS 15 serving path.

## Upstream model loading

The source launcher accepts a Hugging Face owner/repository ID through --model for the supported model configuration. Use --revision to select a target branch, tag, or commit; --language-only for the text-only macOS 15 path; and --draft-model only for a compatible DFlash draft repository or local folder. For a local target and draft, the direct server entry point accepts separate paths; the target directory also supplies tokenizer files. See [RUN.md](RUN.md) for an example.

## Server configuration

The default listener is `127.0.0.1:8000`; use `--port` or `SPLASH_PORT` for another port. `splash-m1 serve --help` (`./splash serve --help` from a checkout) lists every option. Options that apply to this build:

| Option | Default | Purpose |
| --- | --- | --- |
| `--host` | `127.0.0.1` | HTTP bind address; `0.0.0.0` accepts LAN connections (set `--api-key` too). |
| `--allowed-host NAME` | none | Additional HTTP Host name, e.g. `mymac.local`; repeatable. Does not change the bind address. |
| `--allowed-origin ORIGIN` | none | Origin whose pages may call the API from a browser or webview, e.g. `tauri://localhost`; `'*'` for any (set `--api-key` too); repeatable. |
| `--api-key` | `SPLASH_API_KEY` or none | Require a bearer token or `x-api-key`. |
| `--max-memory` / `--max-context` | auto | Metal allocation ceiling (e.g. `28G`) and context limit (up to `256K`). |
| `--max-cache-disk` | `0` (off) | SSD cache for KV pages and states, e.g. `16G`. |
| `--served-model-name NAME` | none | Additional API model ID; repeatable. With `--announce-served-name`, responses report the first alias. |
| `--default-reasoning-effort` | `SPLASH_DEFAULT_REASONING_EFFORT` or model template | Fallback for Chat `reasoning_effort` and Responses `reasoning.effort`. |
| `--request-timeout` | none | Seconds a request may take from its arrival. |
| `--queue-size` | `32` | Requests admitted at once, running or waiting; more get 503. |
| `--max-request-size` | `128M` | Maximum HTTP request body size. |
| `--idle-unload` | `0` (off) | Unload the model after this many idle seconds; the next request reloads it. |
| `--prefill-mode` | `bounded` | `bounded` keeps each prefill GPU command short so macOS does not abort long-context prefill; `full` sends whole 2048-token chunks. |
| `--no-webui` | off | Disable the chat page. |

`--kv-format bf16` and `--max-image-pixels` are accepted by the parsers but unsupported on the macOS 15 build (INT8 KV, no vision). `--decode-share` (decode time owed per unit of prefill time) is a scheduler option whose support follows the runtime.

A Chat request's `chat_template_kwargs` are passed to the chat template and outrank the effort, so `{"enable_thinking": false}` turns reasoning off. A failed engine restarts at once; after three failures within 60 s of starting each, Splash stops restarting it and requests get 500 `engine_failed` until the server is restarted.

## Development checks

Run the model-free source, native CPU, and Python checks with:

~~~sh
make check-source check-native-cpu check-python-engine
~~~

This requires the development dependencies and a Mac with the Xcode Metal tools. It does not download model weights or run a model on the GPU. Model-backed runtime checks require a separate model and GPU run.

## Environment variables

Runtime tuning variables are experimental and unsupported as a user interface. Unset means the default shown.

| Variable | Effect | Default | Read |
| --- | --- | --- | --- |
| `SPLASH_DECODE_M1` | `1` decodes prompt-lookup misses as one row through the PQ2_0 GEMV (Bonsai only; allocates its scratch) | off | once per model runtime |
| `SPLASH_DECODE_LADDER` | `1` enables the adaptive autoregressive/speculative decode policy | off | once per model runtime |
| `SPLASH_DECODE_LADDER_WINDOW`, `_ELIGIBLE`, `_PROBE`, `_MIN_EVIDENCE`, `_MAX_PERIOD` | Policy cycle counts: rolling window, cycles between probes, probe length, probe evidence, longest probe period | 8, 8, 8, 8, 64 | once per model runtime |
| `SPLASH_DECODE_LADDER_GATE`, `_BACKOFF_LOSS` | Probe speed ratio that switches mode; loss fraction that doubles the probe period | 1.05, 0.20 | once per model runtime |
| `SPLASH_LOOKUP_MIN` | Shortest earlier match (tokens) that prompt-lookup drafting uses; `0` turns it off | 16 | once per process |
| `SPLASH_DRAFT_TEMP_SCALE` | Factor on the sampling temperature of the draft | 0.8 | once per process |
| `SPLASH_ACCEPT_LOG` | Path; appends one tab-separated line per verify cycle (analysis only) | unset | at process start |
| `SPLASH_WEIGHT_CACHE` | Directory of prepared model weights | `~/Library/Caches/Splash/weights` | at each preparation |
| `SPLASH_VERSION`, `SPLASH_REPO`, `SPLASH_BASE_URL`, `SPLASH_TOKEN`, `SPLASH_APP_DIR`, `SPLASH_BIN_DIR` | `dev/tools/install.sh`: release version, GitHub repository, asset base URL, its bearer token, app and command directories | latest release | per install |

## Package for review

The macOS 15 package is implemented but is not a published release. To build the review artifact locally:

~~~sh
make BUILD=build/release/native package RELEASE_VERSION=0.1.0-local
~~~

For the example version, the package command creates the runtime archive dist/splash-m1-0.1.0-local-arm64-macos15.tar.gz and the desktop archive dist/splash-m1-0.1.0-local-arm64-macos15-app.zip, their SHA-256 files, splash-m1.rb, install.sh, and the latest version pointer. The app is compiled with the Xcode Swift toolchain and embeds the same runtime under Contents/Resources/runtime. The manual GitHub Actions workflow uploads both artifacts for review; it does not publish a GitHub Release. No public release or Homebrew tap is currently available.

When published, the installer will default to the latest GitHub Release and install the desktop app (Splash M1.app, into /Applications or ~/Applications; `--cli-only` skips it) and the splash-m1 command without requiring an access token for public assets. SPLASH_VERSION selects a version; SPLASH_REPO, SPLASH_BASE_URL, and SPLASH_APP_DIR, and SPLASH_BIN_DIR override the repository, asset base URL, app destination, and command destination. The app and its checksum are verified, extracted beside the destination, and swapped in; a local directory served over HTTP (for example `python3 -m http.server` with SPLASH_BASE_URL) can test the installer without GitHub. SPLASH_TOKEN is only needed for a private custom asset source. The package includes bundled CPython and dependencies, so users do not need to install Python or a compiler.

The desktop app starts the existing launcher only after the user accepts the initial model download. It opens the browser after the owned server becomes ready, and stops that server when the user stops or quits the app. App preferences are separate from the browser's settings. Weights and runtime data remain in the existing per-user locations, outside the app bundle.

Review artifacts are not Developer ID signed or notarized. Signing and notarization are still required before normal public desktop distribution; the local package command does not publish or change repository visibility.

## Release check

A release candidate needs review of the package workflow artifact and one complete first-run check on the supported M1 Max/macOS 15 setup: unpack the app, start the download, get a short Qwen3.8-27B response, quit, and reopen using the cached model. Check cancellation and shutdown without starting a second large model process. The workflow only builds and uploads review artifacts; it does not perform the device/model check or publish a GitHub Release.
