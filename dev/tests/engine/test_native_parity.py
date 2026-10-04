"""The Python metadata gates accept and reject the checkpoints the native loader
does. The native verdicts are encoded here from its source (not executed); a
row's comment names the lines that decide it."""

import copy
import re
import unittest
from pathlib import Path
from unittest import mock

from test_desktop_models import prism_config, target_config

from install import desktop_models, families, models, upstream

ROOT = Path(__file__).resolve().parents[3]


def python_accepts(config):
    """What the preflight does for a text-only MLX target: upstream's format
    screen (upstream._mlx_target), then the desktop gate."""
    prism = config.get("model_type") == upstream.PRISM_MODEL_TYPE
    try:
        if prism:
            upstream.require_prism_config(config)
        elif not upstream.is_affine_q4(config.get("quantization")):
            return False
        desktop_models._require_target_metadata(
            config, upstream.PRISM_FORMAT if prism else "mlx-affine"
        )
    except models.ModelError:
        return False
    return True


def affine(**changes):
    config = target_config()
    config["quantization"] = {"mode": "affine", "bits": 4, "group_size": 64} | changes
    return config


def prism(**changes):
    config = prism_config()
    config["quantization"] = config["quantization"] | changes.pop("quantization", {})
    return config | changes


def override(entry):
    config = affine()
    config["quantization"]["language_model.model.layers.0.mlp.gate_proj"] = entry
    return config


def text(**changes):
    config = affine()
    config["text_config"] |= changes
    return config


# (name, config, native accepts). Native rules, runtime/model:
#   affine: SafetensorsCheckpoint.mm:168-177 requireQuantization (entry bits ==
#     the part's bits, group_size == 64, mode absent or "affine"),
#     AffineTarget.cpp:50-51 (a part's bits must be 4 or 5),
#     AffineTarget.cpp:82-106 validateConfiguration (text_config fields).
#   Prism: PrismMlx.mm:337-338 requirePrismMlxConfig (model_type, schema 2,
#     bits 2, group_size 128, mode absent or "affine").
NATIVE = (
    ("plain Qwen Q4/g64", affine(), True),
    ("Q5 override on a part", override({"bits": 5}), True),
    ("explicit Q4 override", override({"bits": 4}), True),
    ("3-bit override (AffineTarget.cpp:50-51)", override({"bits": 3}), False),
    (
        "override with group 128 (mm:174)",
        override({"bits": 5, "group_size": 128}),
        False,
    ),
    ("override with another mode (mm:172-175)", override({"mode": "mxfp4"}), False),
    ("8-bit default (mm:173)", affine(bits=8), False),
    ("group 128 default (mm:174)", affine(group_size=128), False),
    ("mode mxfp4 default (mm:175)", affine(mode="mxfp4"), False),
    (
        "wrong intermediate_size (AffineTarget.cpp:105)",
        text(intermediate_size=12288),
        False,
    ),
    ("wrong conv kernel (AffineTarget.cpp:89)", text(linear_conv_kernel_dim=3), False),
    (
        "wrong rope theta (AffineTarget.cpp:91)",
        text(
            rope_parameters={
                **target_config()["text_config"]["rope_parameters"],
                "rope_theta": 10000,
            }
        ),
        False,
    ),
    ("tied embeddings (AffineTarget.cpp:91)", text(tie_word_embeddings=True), False),
    ("Prism bits 2 / group 128 / schema 2", prism(), True),
    ("Prism schema 1 (PrismMlx.mm:337)", prism(schema_version=1), False),
    ("Prism schema 3 (PrismMlx.mm:337)", prism(schema_version=3), False),
    ("Prism 4-bit (PrismMlx.mm:337)", prism(quantization={"bits": 4}), False),
    ("Prism group 64 (PrismMlx.mm:338)", prism(quantization={"group_size": 64}), False),
    (
        "Prism mode mxfp4 (PrismMlx.mm:338)",
        prism(quantization={"mode": "mxfp4"}),
        False,
    ),
)


class NativeParityTests(unittest.TestCase):
    def test_python_gate_matches_the_native_loader(self):
        for name, config, native in NATIVE:
            with self.subTest(name):
                self.assertEqual(python_accepts(copy.deepcopy(config)), native)

    def test_every_layout_field_is_a_field_the_native_target_checks(self):
        source = (ROOT / "runtime/model/AffineTarget.cpp").read_text()
        for key, _ in families.named(desktop_models.DESKTOP_FAMILY).layout:
            with self.subTest(key):
                self.assertIn(f'"{key}"', source)

    def test_layout_covers_the_native_required_fields(self):
        # The numeric and string fields of AffineTarget.cpp validateConfiguration
        # the signature does not already state.
        source = (ROOT / "runtime/model/AffineTarget.cpp").read_text()
        block = source[
            source.index("void validateConfiguration") : source.index(
                "Image layerImage"
            )
        ]
        native = set(re.findall(r'\{"([a-z_.]+)",', block)) | set(
            re.findall(r'requireConfig(?:Number|String)\("([a-z_.]+)"', block)
        )
        family = families.named(desktop_models.DESKTOP_FAMILY)
        stated = {key for key, _ in family.signature + family.layout}
        # Checked by the layer schedule and the MoE branch, not per field here.
        missing = (
            native
            - stated
            - {
                "model_type",
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
            }
        )
        self.assertEqual(missing, set())


class MemoryThresholdTests(unittest.TestCase):
    def test_installer_and_desktop_agree_on_the_full_memory_threshold(self):
        script = (ROOT / "dev/tools/install.sh").read_text()
        (threshold,) = re.findall(r'"\$mem" -ge (\d+)', script)
        self.assertEqual(int(threshold), upstream.FULL_MEMORY_BYTES)

    def test_desktop_gate_admits_exactly_the_threshold(self):
        for memory, full in (
            (16 * 1024**3, False),
            (upstream.FULL_MEMORY_BYTES - 1, False),
            (upstream.FULL_MEMORY_BYTES, True),
        ):
            with (
                self.subTest(memory),
                mock.patch.object(upstream, "_memory_bytes", return_value=memory),
            ):
                if full:
                    upstream.require_memory("mlx-affine")
                else:
                    with self.assertRaises(models.ModelError):
                        upstream.require_memory("mlx-affine")
                upstream.require_memory(upstream.PRISM_FORMAT)


if __name__ == "__main__":
    unittest.main()
