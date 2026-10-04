# Splash M1

Splash M1 is an unofficial community fork of [Inco's Splash](https://github.com/incoai/splash), based on [paperniuk's Apple7 kernel work](https://github.com/paperniuk/splash/tree/apple7-m1-kernels-1.1). It ports Splash's text inference path to Metal 3.2 on macOS 15 for Apple7/8 GPUs.

## Start with the Mac app

Release candidates are published on the [releases page](https://github.com/paulouski/splash-m1/releases). They are ad-hoc signed and not notarized.

Install the app and the `splash-m1` command with one line in Terminal:

~~~sh
curl -fsSL https://github.com/paulouski/splash-m1/releases/latest/download/install.sh | bash
~~~

The installer verifies the download and puts **Splash M1.app** in /Applications (or ~/Applications if that is not writable). Files downloaded by curl carry no quarantine flag, so the app opens without a Gatekeeper warning. Add `-s -- --cli-only` after `bash` to skip the app. Alternatively, download the **Mac app** ZIP from the releases page, unzip it, and drag **Splash M1.app** to Applications (see **First launch** below).

To remove Splash M1 with its app, command, settings and downloaded models, run `splash-m1 uninstall` (`--keep-models` keeps the models, `--yes` skips the prompt), or choose **Uninstall Splash…** in the app menu.

1. Open Splash M1 (`open -a "Splash M1"`). Use the recommended model, or paste a Hugging Face model link or repository ID and click **Check Model**. Unsupported configurations are refused before model weights are downloaded. Click **Download & Start** to download the selected model and its matching draft, then prepare them for your Mac.
3. The app opens your browser when the chat is ready. Keep Splash M1 running while you chat; use **Stop** or quit the app to release the model.

The **Installed models** menu lists compatible models already installed through Splash. Select one and start it to reuse its files; stop the running model before choosing another. **Show Details** displays installation and server output. The compatibility check inspects model metadata; the engine validates the actual tensors during startup.

For [Pi](https://pi.dev/) or [OpenCode](https://opencode.ai/docs/), install the client first, start Splash, and wait until it is ready. Click **Copy Pi Command** or **Copy OpenCode Command**, then paste the command into Terminal in your project folder. The launcher connects the client to the currently loaded model and context. Pi's launcher adds a Splash provider to its model configuration; OpenCode receives settings for that launch. Keep Splash M1 running while the client uses it. **Copy API URL** provides the OpenAI-compatible endpoint for other clients.

**First launch of a manually downloaded ZIP:** the app is ad-hoc signed and not notarized, so macOS 15 shows "“Splash M1” is damaged and can't be opened" for a quarantined copy, and Open Anyway may not be offered. Clear the quarantine flag in Terminal: `xattr -dr com.apple.quarantine "/Applications/Splash M1.app"`. The app also checks free disk space before downloading and stops with the needed and available amounts if the model, its prepared weights, and a 2 GiB reserve do not fit.

Python, Xcode, and Terminal are not required for the packaged app. The first setup needs internet access and additional disk space for model weights; download and preparation status appears in the app. Later launches reuse cached weights and settings. The default desktop configuration uses a 32K context and requires 32 GB of memory. Macs with 16 GB can run only the 2-bit Prism checkpoint (Ternary Bonsai 2), with a context sized automatically to the available memory; this 16 GB configuration has not been tested on 16 GB hardware.

Local review builds are not Developer ID signed or notarized. A public desktop release still needs signing and a check of the complete first-run flow.

## Measured performance

On an M1 Max with a 24-core GPU and 32 GB running macOS 15.7, selected measurements showed:

| Workload | Comparison | Result |
| --- | --- | --- |
| English Python source edits | Prompt lookup off vs on | 37–38 → ~61 tok/s (~59–66% higher) |
| Synthetic 2,048-token prefill | Combined package vs previous build (4 matched ABBA samples) | 20.0 → 18.8 s (~5.9% less wall time) |

The code-edit A/B on 2026-09-30 used `Swift-Qwen3.8-27B-Uncensored-oQ4e-fp16-mtp` with mixed Q4/Q5 group-64 weights, Swift DFlash 2, INT8 KV cache, and 32K context. This experimental checkpoint differs from the app's recommended download. Prompt lookup used a 16-token minimum source match. The two prompts had 2,179 and 2,230 input tokens; each prompt and setting ran once with fixed seeds, `temperature=1.0`, `top_p=0.95`, `top_k=20`, no thinking, and a cap of 800 generated tokens. Outputs reached the cap. Rates count generation after the first token. These prompts allowed substantial source reuse, a best-case workload; this pre-release A/B does not measure complete task time, answer quality, or current default generation speed. The separate prefill result measures synthetic prompt time, not decode.

## What this fork changes

The fork builds on Splash's APIs, DFlash, and tool integrations, with a focus on the Apple7/8 text path and a local Mac workflow.

- Runs the Apple7/8 text path on Metal 3.2 and macOS 15 without requiring a Metal 4 runtime for this path.
- Supports mixed Q4/Q5 group-64 weights. Apple7 kernels handle weight unpacking and addressing, plus GDN and prefill work tuned for that GPU generation.
- Offers prompt lookup that reuses likely next source tokens and asks the target model to verify them; the benefit depends on the prompt.
- Preserves exact generated tokens and model state across turns so follow-up requests can reuse earlier answer text.
- Adds a packaged Mac app and CLI workflow, an English browser UI, and Pi client setup. The app and CLI packages are implemented but not yet published.

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

The same one-line installer provides the command (`-s -- --cli-only` skips the app); run it with:

~~~sh
curl -fsSL https://github.com/paulouski/splash-m1/releases/latest/download/install.sh | bash
splash-m1 start
~~~

Follow any PATH instruction printed by the installer. `start` uses the recommended Qwen3.8-27B Q4 model, text-only mode, and a 32K context. The server stays in the terminal and prints its address and logs; when it reports Ready, open http://127.0.0.1:8000. Press Ctrl+C to stop it.

To connect an installed coding client, open a second terminal in your project folder and run `splash-m1 pi` or `splash-m1 opencode`. Keep the server terminal open. For another checkpoint or server settings, use `splash-m1 start --model OWNER/REPO --max-context 32K` or `splash-m1 serve --help`. No Homebrew tap is available.

## Build from source

Build requirements: an Apple Silicon Mac running macOS 15 or newer, Xcode with the Metal compiler tools available through xcrun, and Python 3.12, 3.13, or 3.14. Downloading and preparing model weights requires additional disk space. The commands use python3.13; substitute python3.12 or python3.14 if that is your installed supported interpreter.

~~~sh
git clone https://github.com/paulouski/splash-m1.git
cd splash-m1
python3.13 -m venv .venv
.venv/bin/python -m pip install -r install/requirements.txt
make -j8
./splash serve --model mlx-community/Qwen3.8-27B-4bit --language-only --max-context 32K
# Macs below 32 GB of memory: ./splash serve --model prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --language-only
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

## Settings

Memory and context are sized automatically. To set your own limits, add these to `splash-m1 serve` (or `./splash serve`):

| Option | Purpose |
| --- | --- |
| `--max-memory 28G` | Cap Metal memory use. |
| `--max-context 100K` | Set the context limit. |
| `--language-only` | Skip vision preparation; serve text only. |
| `--max-cache-disk 16G` | Offload KV cache and states to SSD as needed. Off by default. |
| `--allowed-origin ORIGIN` | Let a browser app on another origin call the API (repeatable). |

The server listens on localhost without authentication by default. For LAN access, authentication and other options, see [server configuration](DEVELOPMENT.md#server-configuration) or `splash-m1 serve --help`.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

Model behavior, memory use, and performance depend on the checkpoint and workload. This fork does not publish upstream benchmark numbers as its own results.
