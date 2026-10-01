#!/usr/bin/env python3
"""Prepare a Splash-loadable directory for a text-only MLX Qwen3.8 checkpoint.

Builds a directory the affine loader (runtime/model/AffineTarget.cpp,
SafetensorsCheckpoint.mm) can point `serve-native` at, without copying or
rewriting any weight bytes: a flattened text-only config.json, symlinks to
the source safetensors shards and index, and symlinks to the tokenizer and
chat template files. Vision and MTP tensors are left in the shards
unchanged -- the affine loader never rejects tensors it does not reference,
so nothing needs to be filtered out.

Also runs a preflight that mirrors the C++ loader's own checks (config
numbers/strings, layer schedule, per-projection quantization bits, tensor
dtypes) against the Qwen3.8 layout in runtime/model/Qwen3_8.hpp, and prints
every mismatch it finds -- this only reads safetensors headers and JSON, no
MLX/torch and no tensor payloads.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

# Mirrors runtime/model/Qwen3_8.hpp's QwenHybridLayout and the checks
# validateConfiguration() in runtime/model/AffineTarget.cpp runs against it.
QWEN3_8_NUMBERS = {
    "num_hidden_layers": 64,
    "hidden_size": 5120,
    "vocab_size": 248320,
    "head_dim": 256,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4,
    "rms_norm_eps": 1e-6,
    "attention_bias": 0,
    "attn_output_gate": 1,
    "tie_word_embeddings": 0,
    "rope_parameters.rope_theta": 10_000_000,
    "rope_parameters.partial_rotary_factor": 0.25,
    "intermediate_size": 17408,
}
QWEN3_8_STRINGS = {
    "hidden_act": "silu",
    "rope_parameters.rope_type": "default",
    "model_type": "qwen3_5_text",
}
FULL_ATTENTION_PERIOD = QWEN3_8_NUMBERS["full_attention_interval"]
NUM_LAYERS = QWEN3_8_NUMBERS["num_hidden_layers"]

# Files the tokenizer/chat side needs; symlinked as-is. Vision preprocessing
# config is skipped: this prep is text-only.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
)

DEFAULT_OUTPUT = Path("build/prepared-qwen38")


def config_value(config, dotted_key):
    value = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def flatten_config(source_config):
    """text_config fields promoted to top level; model_type and rope_type
    fixed up to what the affine loader requires. Quantization maps and
    vision_config are left as source has them (vision_config is inert once
    model_type says qwen3_5_text; loader only reads what it needs)."""
    text_config = source_config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("source config.json has no text_config to flatten")
    flat = dict(source_config)
    del flat["text_config"]
    flat.update(text_config)
    flat["model_type"] = "qwen3_5_text"
    rope_parameters = flat.get("rope_parameters")
    if isinstance(rope_parameters, dict) and "rope_type" not in rope_parameters:
        rope_parameters = dict(rope_parameters)
        # Source names this field "type"; the loader requires "rope_type".
        if "type" in rope_parameters:
            rope_parameters["rope_type"] = rope_parameters["type"]
        flat["rope_parameters"] = rope_parameters
    return flat


def read_safetensors_header(path):
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def check_config(flat_config, report):
    for key, expected in QWEN3_8_NUMBERS.items():
        actual = config_value(flat_config, key)
        if not isinstance(actual, (int, float)) or actual != expected:
            report.fail("config", f"{key}: expected {expected!r}, got {actual!r}")
    for key, expected in QWEN3_8_STRINGS.items():
        actual = config_value(flat_config, key)
        if actual != expected:
            report.fail("config", f"{key}: expected {expected!r}, got {actual!r}")
    layer_types = flat_config.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != NUM_LAYERS:
        report.fail(
            "layer_types",
            f"expected {NUM_LAYERS} entries, got "
            f"{len(layer_types) if isinstance(layer_types, list) else layer_types!r}",
        )
    else:
        for layer, kind in enumerate(layer_types):
            expected = (
                "linear_attention"
                if (layer + 1) % FULL_ATTENTION_PERIOD
                else "full_attention"
            )
            if kind != expected:
                report.fail(
                    "layer_types", f"layer {layer}: expected {expected!r}, got {kind!r}"
                )
                break


def check_quantization(source_config, report):
    """The affine loader's dense (non-MoE, single-expert) projections accept
    a per-tensor override of 4 or 5 bit (AffineTarget.cpp's projection()
    bitsOf lookup, Q5Pack.hpp); any other width, or a non-4/64 default,
    fails requireQuantization() at load time (SafetensorsCheckpoint.mm)."""
    quantization = source_config.get("quantization") or source_config.get(
        "quantization_config"
    )
    if not isinstance(quantization, dict):
        report.fail("quantization", "source config has no quantization map")
        return
    default_bits = quantization.get("bits")
    default_group = quantization.get("group_size")
    if default_bits != 4 or default_group != 64:
        report.fail(
            "quantization",
            f"default bits/group_size {default_bits}/{default_group}, "
            "loader requires 4/64",
        )
    mismatched = [
        (name, entry.get("bits"))
        for name, entry in quantization.items()
        if isinstance(entry, dict) and entry.get("bits") not in (None, 4, 5)
    ]
    if mismatched:
        by_bits = {}
        for name, bits in mismatched:
            by_bits.setdefault(bits, []).append(name)
        for bits, names in sorted(by_bits.items()):
            example = ", ".join(names[:5])
            more = f" (+{len(names) - 5} more)" if len(names) > 5 else ""
            report.fail(
                "quantization",
                f"{len(names)} projections at {bits}-bit, loader requires 4 or 5-bit: "
                f"{example}{more}",
            )


def check_tensor_dtypes(shard_paths, report):
    """copy()/halfCopy() in AffinePlan.hpp accept BF16 or F16 for every
    plain/scales/biases field (A_log also accepts F32); U32 is for packed
    weight codes. Mismatches here fail bind() in AffinePlan.hpp at load
    time."""
    mismatched_by_suffix = {}
    seen_any = False
    for path in shard_paths:
        header = read_safetensors_header(path)
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if not name.startswith("language_model."):
                continue
            seen_any = True
            dtype = tensor.get("dtype")
            if name.endswith(".weight"):
                # Plain BF16/F16 weights (norms, conv1d, ...) vs packed U32
                # quantized codes are both suffixed ".weight".
                if dtype not in ("BF16", "F16", "U32"):
                    mismatched_by_suffix.setdefault(dtype, []).append(name)
            elif name.endswith("A_log"):
                if dtype not in ("BF16", "F16", "F32"):
                    mismatched_by_suffix.setdefault(dtype, []).append(name)
            elif dtype not in ("BF16", "F16"):
                mismatched_by_suffix.setdefault(dtype, []).append(name)
    if not seen_any:
        report.fail("dtype", "no language_model.* tensors found in the source shards")
        return
    for dtype, names in sorted(mismatched_by_suffix.items()):
        example = ", ".join(names[:5])
        more = f" (+{len(names) - 5} more)" if len(names) > 5 else ""
        report.fail(
            "dtype",
            f"{len(names)} tensors are {dtype}, loader requires BF16 or F16 "
            f"(F32 only for A_log): {example}{more}",
        )


class Report:
    def __init__(self):
        self.failures = []

    def fail(self, group, message):
        self.failures.append((group, message))

    def print(self):
        print("\n=== Splash affine-loader compatibility preflight ===")
        if not self.failures:
            print("OK: no mismatches found against the Qwen3.8 layout.")
            return
        by_group = {}
        for group, message in self.failures:
            by_group.setdefault(group, []).append(message)
        for group, messages in by_group.items():
            print(f"\n[{group}] {len(messages)} issue(s):")
            for message in messages:
                print(f"  - {message}")
        print(
            f"\nTOTAL: {len(self.failures)} blocking issue(s). "
            "The loader (runtime/model/AffineTarget.cpp, SafetensorsCheckpoint.mm) "
            "will reject this source until they are resolved upstream of this "
            "script -- it only arranges files, it does not rewrite weight bytes."
        )


def symlink(target_dir, source, name=None):
    name = name or source.name
    link = target_dir / name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(source.resolve())


def prepare(source_dir, output_dir):
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_config = json.loads((source_dir / "config.json").read_text())
    flat_config = flatten_config(source_config)
    (output_dir / "config.json").write_text(json.dumps(flat_config, indent=2) + "\n")

    shard_paths = sorted(source_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"no .safetensors shards in {source_dir}")
    for shard in shard_paths:
        symlink(output_dir, shard)
    index_path = source_dir / "model.safetensors.index.json"
    if index_path.exists():
        symlink(output_dir, index_path)

    linked, missing = [], []
    for name in TOKENIZER_FILES:
        path = source_dir / name
        if path.exists():
            symlink(output_dir, path)
            linked.append(name)
        else:
            missing.append(name)

    report = Report()
    check_config(flat_config, report)
    check_quantization(source_config, report)
    check_tensor_dtypes(shard_paths, report)

    return shard_paths, linked, missing, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source MLX model directory")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="directory to build"
    )
    args = parser.parse_args(argv)

    shard_paths, linked, missing, report = prepare(args.source, args.output)

    print(f"Prepared {args.output}")
    print("  config.json: flattened, written")
    print(f"  safetensors shards linked: {len(shard_paths)}")
    print(f"  tokenizer/template files linked: {', '.join(linked) or '(none)'}")
    if missing:
        print(f"  tokenizer/template files missing from source: {', '.join(missing)}")
    report.print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
