# Run Splash M1 from source

Build the macOS 15 variant and start the supported Hugging Face model. The commands use python3.13; substitute python3.12 or python3.14 if that is your installed supported interpreter:

~~~sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r install/requirements.txt
make MACOS15=1 BUILD=build
./splash serve --model mlx-community/Qwen3.8-27B-4bit --language-only --max-context 32K
~~~

The first start downloads the selected target and its matching DFlash draft, then prepares the weights. It serves locally on http://127.0.0.1:8000 by default. Press Ctrl+C to stop.

## Use local target and draft folders

The direct server entry point accepts a target directory and a separate DFlash draft directory. Point tokenizer at the directory containing the target tokenizer files. Replace the example folders with your own local checkpoint paths:

~~~sh
.venv/bin/python -m server.server \
  "$HOME/Models/qwen38-target" \
  "$HOME/Models/qwen38-draft" \
  --tokenizer "$HOME/Models/qwen38-target" \
  --model mlx-community/Qwen3.8-27B-4bit \
  --port 8000 \
  --max-context 32K \
  --kv-format int8 \
  --binary build/splash
~~~

In this direct mode, the --model value names the model in API responses; the target and draft paths provide the local weights. Use a compatible text-only Qwen3.8-27B MLX target and its matching DFlash draft. This macOS 15 build does not load vision or BF16 KV.
