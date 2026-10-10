# fn_quantize

Requantizes Q8_0 (or any ggml-readable) GGUF shards tensor by tensor from a recipe, with an optional imatrix
(per-expert entries are honoured), into one GGUF. Needs a llama.cpp build for libggml.

    L=/path/to/llama.cpp   # with build/bin/libggml*.dylib
    clang++ -std=c++17 -O2 -I$L/ggml/include dev/tools/fn_quantize.cpp -L$L/build/bin -lggml-base -lggml-cpu -lggml \
      -Wl,-rpath,$L/build/bin -o fn_quantize

    # target without the PLE table
    fn_quantize --out target.gguf --recipe dev/tools/fn_recipes/b1-uniform.recipe --imatrix imatrix.gguf \
      --drop 'per_layer_token_embd\.weight' Q8_0-0000{1,2,3,4,5,6}-of-00006.gguf
    # PLE table alone, IQ4_NL (recipe line: per_layer_token_embd\.weight = IQ4_NL)
    fn_quantize --out ple-iq4nl.gguf --recipe ple.recipe --only 'per_layer_token_embd\.weight' Q8_0-00003-of-00006.gguf

`--dry-run` prints sizes per type without writing; `--info` lists imatrix entries (values, experts, experts with no
samples). Key-values come from the first shard given (pass the metadata shard 00001 first for a loadable model).

The engine takes the table from `SPLASH_PLE_TABLE=<gguf>` when the target has none.

An imatrix for a new calibration set: `llama-imatrix -m <Q8_0 shard 1> -f corpus.txt` on CPU (mmap streams the 175 GiB).
