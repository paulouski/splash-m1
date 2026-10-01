# Splash M1

Splash M1 is an unofficial community fork of [Inco's Splash](https://github.com/incoai/splash), based on [paperniuk's Apple7 kernel work](https://github.com/paperniuk/splash/tree/apple7-m1-kernels-1.1). It ports Splash's text inference path to Metal 3.2 on macOS 15 for Apple7/8 GPUs.

## Start with the Mac app

The desktop app is being prepared for release. No public download has been published yet. Check the [releases page](https://github.com/paulouski/splash-m1/releases) for availability.

Once a desktop release is available:

1. Download the **Mac app** ZIP, unzip it, and drag **Splash M1.app** to Applications.
2. Open Splash M1 and click **Download & Start**. It downloads the recommended Qwen3.8-27B model and its matching draft from Hugging Face, then prepares them for your Mac.
3. The app opens your browser when the chat is ready. Keep Splash M1 running while you chat; use **Stop** or quit the app to release the model.

Python, Xcode, and Terminal are not required for the packaged app. The first setup needs internet access and additional disk space for model weights; download and preparation status appears in the app. Later launches reuse cached weights and settings. The default desktop configuration uses a 32K context and requires 32 GB of memory.

Local review builds are not Developer ID signed or notarized. A public desktop release still needs signing and a check of the complete first-run flow.

## Platform and model support

The tested configuration is an M1 Max with a 24-core GPU and 32 GB unified memory on macOS 15.7. M2 support is intended but unverified. Apple9 and newer GPUs are not a supported guarantee for this build.

The supported public model configuration is text-only Qwen3.8-27B from an MLX affine group-64 checkpoint: uniform Q4 or mixed Q4/Q5 tensors, F16 or BF16 target tensors, a matching BF16 DFlash draft, and INT8 KV cache.

The macOS 15 build does not support GGUF targets, MoE models, image or PDF input, or BF16 KV cache. Model weights and their licenses are provided by their respective publishers.

## Features

- Metal 3.2 kernels for the Apple7/8 text inference path, including mixed Q4/Q5 weights.
- DFlash speculative decoding with the model's matching draft.
- OpenAI-compatible Chat Completions and Responses, Anthropic Messages, tool calls, and optional token logprobs.
- A local browser chat UI with Markdown and KaTeX rendering, plus a Pi client launcher.
- Optional web search and page-reading tools that use internet access.
- Exact generated-token prefix reuse across turns and optional idle model unload.

This fork builds on Inco's Splash engine and the community Apple7 kernel port. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) for code licensing and attribution.

## Command-line installation

The command-line package is also implemented but has not been published yet. After a release containing the installer asset is published, install and run it with:

~~~sh
curl -fsSL https://github.com/paulouski/splash-m1/releases/latest/download/install.sh | sh
splash-m1 serve --model mlx-community/Qwen3.8-27B-4bit --language-only --max-context 32K
~~~

Follow any PATH instruction printed by the installer. When the server reports Ready, open http://127.0.0.1:8000. Press Ctrl+C to stop it. No Homebrew tap is available.

## Build from source

Build requirements: an Apple Silicon Mac running macOS 15 or newer, Xcode with the Metal compiler tools available through xcrun, and Python 3.12, 3.13, or 3.14. Downloading and preparing model weights requires additional disk space. The commands use python3.13; substitute python3.12 or python3.14 if that is your installed supported interpreter.

~~~sh
git clone https://github.com/paulouski/splash-m1.git
cd splash-m1
python3.13 -m venv .venv
.venv/bin/python -m pip install -r install/requirements.txt
make -j8
./splash serve --model mlx-community/Qwen3.8-27B-4bit --language-only --max-context 32K
~~~

The server downloads the selected model and matching draft from Hugging Face, then prepares weights for this build. When the server reports Ready, open http://127.0.0.1:8000 or connect an agent. Press Ctrl+C to stop it.

## API

The server provides OpenAI Chat Completions at /v1/chat/completions, OpenAI Responses at /v1/responses, and Anthropic Messages at /v1/messages. Tool calls are supported. For example:

~~~sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen3.8-27B-4bit",
    "messages": [{"role": "user", "content": "Explain speculative decoding in one sentence."}]
  }'
~~~

See [DEVELOPMENT.md](DEVELOPMENT.md) for source build details and [RUN.md](RUN.md) for local checkpoint examples.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

Model behavior, memory use, and performance depend on the checkpoint and workload. This fork does not publish upstream benchmark numbers as its own results.
