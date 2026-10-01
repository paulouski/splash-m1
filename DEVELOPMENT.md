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

## Development checks

Run the model-free source, native CPU, and Python checks with:

~~~sh
make check-source check-native-cpu check-python-engine
~~~

This requires the development dependencies and a Mac with the Xcode Metal tools. It does not download model weights or run a model on the GPU. Model-backed runtime checks require a separate model and GPU run.

## Package for review

The macOS 15 package is implemented but is not a published release. To build the review artifact locally:

~~~sh
make BUILD=build/release/native package RELEASE_VERSION=0.1.0-local
~~~

For the example version, the package command creates the runtime archive dist/splash-m1-0.1.0-local-arm64-macos15.tar.gz and the desktop archive dist/splash-m1-0.1.0-local-arm64-macos15-app.zip, their SHA-256 files, splash-m1.rb, install.sh, and the latest version pointer. The app is compiled with the Xcode Swift toolchain and embeds the same runtime under Contents/Resources/runtime. The manual GitHub Actions workflow uploads both artifacts for review; it does not publish a GitHub Release. No public release or Homebrew tap is currently available.

When published, the installer will default to the latest GitHub Release and install the splash-m1 command without requiring an access token for public assets. SPLASH_VERSION selects a version; SPLASH_REPO, SPLASH_BASE_URL, and SPLASH_BIN_DIR override the repository, asset base URL, and command destination. SPLASH_TOKEN is only needed for a private custom asset source. The package includes bundled CPython and dependencies, so users do not need to install Python or a compiler.

The desktop app starts the existing launcher only after the user accepts the initial model download. It opens the browser after the owned server becomes ready, and stops that server when the user stops or quits the app. App preferences are separate from the browser's settings. Weights and runtime data remain in the existing per-user locations, outside the app bundle.

Review artifacts are not Developer ID signed or notarized. Signing and notarization are still required before normal public desktop distribution; the local package command does not publish or change repository visibility.

## Release check

A release candidate needs review of the package workflow artifact and one complete first-run check on the supported M1 Max/macOS 15 setup: unpack the app, start the download, get a short Qwen3.8-27B response, quit, and reopen using the cached model. Check cancellation and shutdown without starting a second large model process. The workflow only builds and uploads review artifacts; it does not perform the device/model check or publish a GitHub Release.
