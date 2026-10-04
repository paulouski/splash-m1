import fcntl
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import assembly, desktop_models, families, hub, models, paths, upstream

MODEL = "team/Qwen3.8-desktop-test"
REVISION = "a" * 40


def target_config():
    return {
        "text_config": {
            "model_type": "qwen3_5_text",
            "max_position_embeddings": 262144,
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "vocab_size": 248320,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 17408,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "attn_output_gate": True,
            "tie_word_embeddings": False,
            "hidden_act": "silu",
            "rope_parameters": {
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "type": "default",
            },
            "layer_types": [
                "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
                for layer in range(64)
            ],
        },
        "quantization": {
            "mode": "affine",
            "bits": 4,
            "group_size": 64,
            "language_model.model.layers.0.mlp.gate_proj": {"bits": 5},
        },
    }


def draft_config():
    config = {}
    for key, value in families.FAMILIES[0].draft.signature:
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = list(value) if isinstance(value, tuple) else value
    return config


class FakeRepository:
    def __init__(self, root, name, files, *, config):
        self.name = name
        self.revision = REVISION
        self.files = set(files)
        self.directory = root
        self.downloads = []
        root.mkdir(parents=True)
        (root / "config.json").write_text(json.dumps(config))
        index = {
            "weight_map": {
                "language_model.model.layers.0.mlp.gate_proj.weight": (
                    "model-00001.safetensors"
                )
            }
        }
        (root / "model.safetensors.index.json").write_text(json.dumps(index))
        (root / "model-00001.safetensors").write_bytes(b"checkpoint")

    def file(self, name):
        if name not in self.files:
            raise models.ModelError(f"missing {name} in {self.name}")
        return self.directory / name

    def open(self, name):
        return self.file(name).open("rb")

    def download(self, names):
        self.downloads.append(set(names))
        raise AssertionError("metadata check must not download weights")


def repositories(root, *, target=None, draft=None):
    target_files = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-00001.safetensors",
    }
    draft_files = {
        "config.json",
        "model.safetensors.index.json",
        "model-00001.safetensors",
    }
    target_repo = FakeRepository(
        root / "target",
        MODEL,
        target_files,
        config=target or target_config(),
    )
    draft_repo = FakeRepository(
        root / "draft",
        families.FAMILIES[0].draft.repo,
        draft_files,
        config=draft or draft_config(),
    )
    return target_repo, draft_repo


DRAFT_REVISION = "b" * 40
OTHER_REVISION = "c" * 40
OTHER = "team/other-model"


def cache_snapshot(cache, repo, revision, files):
    folder = cache / hub.folder_name(repo)
    for name, content in files.items():
        blob = folder / "blobs" / hashlib.sha256(content).hexdigest()
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(content)
        link = folder / "snapshots" / revision / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(blob)
    return folder / "snapshots" / revision


def install(root, link, model, target_revision, draft_repo=None):
    """Install model at link from fake Hub snapshots, pinned as launcher does."""
    cache = root / "hub"
    draft_repo = draft_repo or families.FAMILIES[0].draft.repo
    target = cache_snapshot(
        cache, MODEL, target_revision, {"model.safetensors": target_revision.encode()}
    )
    draft = cache_snapshot(
        cache, draft_repo, DRAFT_REVISION, {"model.safetensors": b"d"}
    )
    record = {
        "version": 1,
        "model": model,
        "family": "Qwen3.8-27B",
        "target_format": "mlx-affine",
        "vision_format": "none",
        "sources": {
            "target": {"repo": MODEL, "revision": target_revision},
            "draft": {"repo": draft_repo, "revision": DRAFT_REVISION},
        },
    }
    files = {
        "target/model.safetensors": target / "model.safetensors",
        "draft/model.safetensors": draft / "model.safetensors",
    }
    record["files"] = {name: assembly.file_record(path) for name, path in files.items()}
    entry = assembly.build(root / "models", record, files)
    link.parent.mkdir(parents=True, exist_ok=True)
    models.link_selection(link, entry)
    hub.pin(target, MODEL, link)
    hub.pin(draft, draft_repo, link)
    return entry


def prepared(cache, key, entry, part):
    directory = cache / (key * 64)
    directory.mkdir(parents=True)
    (directory / "weights").write_bytes(b"w" * 10)
    (directory / "source").write_text(
        f"splash-prepared-weight-v1\ncomponent {part}\ninputs x\n"
        f"source {entry}/{part}/model.safetensors\n"
    )
    return directory


class DeleteModelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.models = self.root / "models"
        self.cache = self.root / "hub"
        self.weights = self.root / "weights"
        self.weights.mkdir()
        for patcher in (
            mock.patch.dict(os.environ, {"SPLASH_WEIGHT_CACHE": str(self.weights)}),
            mock.patch.object(paths, "RUNTIME", self.root / "runtime"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        draft = families.FAMILIES[0].draft.repo
        self.draft_snapshot = (
            self.cache / hub.folder_name(draft) / "snapshots" / DRAFT_REVISION
        )
        self.link1 = self.models / ".selections" / "one"
        self.link2 = self.models / "team" / "copy"
        self.link3 = self.models / "team" / "other"
        self.entry1 = install(self.root, self.link1, MODEL, REVISION)
        self.entry2 = install(self.root, self.link2, MODEL, OTHER_REVISION)
        self.entry3 = install(self.root, self.link3, OTHER, OTHER_REVISION)
        self.snapshot = lambda revision: (
            self.cache / hub.folder_name(MODEL) / "snapshots" / revision
        )

    def test_plan_changes_nothing(self):
        plan = desktop_models.delete_plan(MODEL, self.models)
        self.assertGreater(plan["bytes"], 0)
        self.assertTrue(self.link1.is_symlink() and self.snapshot(REVISION).is_dir())

    def test_delete_removes_every_link_and_keeps_shared_state(self):
        t1 = prepared(self.weights, "1", self.entry1, "target")
        draft1 = prepared(self.weights, "2", self.entry1, "draft")
        t3 = prepared(self.weights, "3", self.entry3, "target")
        desktop_models.delete(MODEL, self.models)

        self.assertFalse(self.link1.is_symlink() or self.link2.is_symlink())
        self.assertFalse(self.entry1.exists() or self.entry2.exists())
        self.assertTrue(self.link3.is_symlink() and self.entry3.is_dir())
        self.assertFalse(self.snapshot(REVISION).exists())
        self.assertTrue(self.snapshot(OTHER_REVISION).is_dir())
        self.assertTrue(self.draft_snapshot.is_dir())
        self.assertFalse(t1.exists())
        self.assertTrue(draft1.is_dir() and t3.is_dir())
        self.assertEqual(
            [
                pin.parent.name
                for pin in self.draft_snapshot.parents[1].glob("refs/splash/*/*")
            ],
            [hub.pin_owner(self.link3)],
        )

        desktop_models.delete(OTHER, self.models)
        self.assertFalse(self.snapshot(OTHER_REVISION).exists())
        self.assertFalse(t3.exists())
        self.assertTrue(self.draft_snapshot.is_dir())

    def test_bonsai_draft_stays_shared(self):
        link = self.models / "team" / "bonsai"
        entry = install(
            self.root, link, "team/Bonsai", REVISION, upstream.PRISM_DRAFT_REPO
        )
        draft = (
            self.cache
            / hub.folder_name(upstream.PRISM_DRAFT_REPO)
            / "snapshots"
            / DRAFT_REVISION
        )
        desktop_models.delete("team/Bonsai", self.models)
        self.assertFalse(entry.exists())
        self.assertTrue(draft.is_dir())

    def test_refused_while_serving_or_held(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        with (runtime / "serve-8000.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.assertRaises(models.ModelError):
                desktop_models.delete(MODEL, self.models)
        with (self.entry2 / "model.json").open("rb") as record:
            fcntl.flock(record, fcntl.LOCK_SH)
            with self.assertRaises(models.ModelError):
                desktop_models.delete(MODEL, self.models)
        self.assertTrue(self.link1.is_symlink() and self.snapshot(REVISION).is_dir())


class DesktopModelTests(unittest.TestCase):
    def test_normalizes_ids_and_default_branch_urls_only(self):
        self.assertEqual(desktop_models.normalize_model_id(MODEL), MODEL)
        self.assertEqual(
            desktop_models.normalize_model_id(f"https://huggingface.co/{MODEL}"),
            MODEL,
        )
        self.assertEqual(
            desktop_models.normalize_model_id(f"https://hf.co/{MODEL}/tree/main/"),
            MODEL,
        )
        for value in (
            f"{MODEL}:Q4_K_M",
            f"https://huggingface.co/{MODEL}/tree/dev",
            f"https://huggingface.co/{MODEL}/resolve/main/config.json",
            f"https://example.com/{MODEL}",
        ):
            with self.subTest(value=value), self.assertRaises(models.ModelError):
                desktop_models.normalize_model_id(value)

    def test_compatible_target_and_draft_metadata_pass_without_weight_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, draft = repositories(root)
            with (
                mock.patch.object(
                    hub.Repository,
                    "resolve",
                    side_effect=lambda name, **_kwargs: (
                        target if name == MODEL else draft
                    ),
                ),
            ):
                result = desktop_models.check_model(MODEL, models_root=root / "models")
            self.assertEqual(result, MODEL)
            self.assertEqual(target.downloads, [])
            self.assertEqual(draft.downloads, [])

    def test_unsupported_quantization_family_and_geometry_stop_before_weights(self):
        unsupported = []
        bad_quantization = target_config()
        bad_quantization["quantization"]["bits"] = 2
        unsupported.append(("quantization", bad_quantization))

        bad_override = target_config()
        bad_override["quantization"]["language_model.model.layers.0.mlp.gate_proj"][
            "bits"
        ] = 6
        unsupported.append(("per-tensor quantization", bad_override))

        bad_family = target_config()
        bad_family["text_config"]["model_type"] = "qwen3_5_moe_text"
        unsupported.append(("family", bad_family))

        bad_geometry = target_config()
        bad_geometry["text_config"]["linear_num_value_heads"] = 32
        unsupported.append(("geometry", bad_geometry))

        for label, config in unsupported:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                target, draft = repositories(root, target=config)
                with (
                    mock.patch.object(
                        hub.Repository,
                        "resolve",
                        side_effect=lambda name, **_kwargs: (
                            target if name == MODEL else draft
                        ),
                    ),
                    self.assertRaises(models.ModelError),
                ):
                    desktop_models.check_model(MODEL, models_root=root / "models")
                self.assertEqual(target.downloads, [])
                self.assertEqual(draft.downloads, [])

    def test_serve_checks_selected_model_before_exec_without_changing_selection(self):
        events = []
        with (
            mock.patch.object(
                desktop_models,
                "check_model",
                side_effect=lambda model: events.append(("check", model)) or model,
            ),
            mock.patch.object(desktop_models.paths, "PYTHON", Path("/runtime/python")),
            mock.patch.object(desktop_models.paths, "ROOT", Path("/runtime")),
            mock.patch.object(
                desktop_models.os,
                "execv",
                side_effect=lambda _python, command: events.append(("exec", command)),
            ),
        ):
            desktop_models._serve(MODEL)
        self.assertEqual(events[0], ("check", MODEL))
        command = events[1][1]
        self.assertEqual(command[command.index("--model") + 1], MODEL)
        self.assertIn("--language-only", command)
        self.assertNotIn("--revision", command)

    def test_installed_listing_is_local_and_ignores_missing_assembly_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models_root = root / "models"
            target_path = root / "target-config.json"
            draft_path = root / "draft-config.json"
            target_weights = root / "target.safetensors"
            draft_weights = root / "draft.safetensors"
            target_path.write_text(json.dumps(target_config()))
            draft_path.write_text(json.dumps(draft_config()))
            target_weights.write_bytes(b"target")
            draft_weights.write_bytes(b"draft")
            record = {
                "version": 1,
                "model": MODEL,
                "family": "Qwen3.8-27B",
                "target_format": "mlx-affine",
                "vision_format": "none",
                "sources": {
                    "target": {"repo": MODEL, "revision": REVISION},
                    "draft": {
                        "repo": families.FAMILIES[0].draft.repo,
                        "revision": "b" * 40,
                    },
                },
            }
            files = {
                "config.json": target_path,
                "target/model.safetensors": target_weights,
                "draft/config.json": draft_path,
                "draft/model.safetensors": draft_weights,
            }
            record["files"] = {
                name: assembly.file_record(path) for name, path in files.items()
            }
            entry = assembly.build(
                models_root,
                record,
                files,
            )
            selection = models.Selection.of(models_root, MODEL, language_only=True)
            models.link_selection(selection.link, entry)

            with mock.patch.object(
                hub.Repository,
                "resolve",
                return_value=mock.Mock(files=frozenset()),
            ) as resolve:
                self.assertEqual(
                    desktop_models.check_model(MODEL, models_root=models_root), MODEL
                )
                self.assertEqual(resolve.call_count, 1)
                self.assertEqual(
                    desktop_models.installed_models(models_root), [{"model": MODEL}]
                )
                draft_path.unlink()
                self.assertEqual(desktop_models.installed_models(models_root), [])
                with self.assertRaises(models.ModelError):
                    desktop_models.check_model(MODEL, models_root=models_root)


def prism_config(**changes):
    config = target_config()
    config["quantization"] = {"mode": "affine", "bits": 2, "group_size": 128}
    config |= {"model_type": "prism_hadamard_qwen35", "schema_version": 2} | changes
    return config


class DiskSpaceTests(unittest.TestCase):
    GB = 10**9

    def repo(self, name, sizes):
        return mock.Mock(
            directory=None,
            files=set(sizes),
            name=name,
            sizes={n: (s, "blob-" + n) for n, s in sizes.items()},
        )

    def require(self, free, *, fmt="mlx-affine", cached=(), target_sizes=None):
        target = self.repo(MODEL, target_sizes or {"a.safetensors": 10 * self.GB})
        draft = self.repo("team/draft", {"d.safetensors": 2 * self.GB})
        draft.files |= {"model.safetensors.index.json"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for blob in cached:
                (root / "blobs").mkdir(exist_ok=True)
                (root / "blobs" / blob).touch()
            with (
                mock.patch.object(hub, "folder", return_value=root),
                mock.patch.dict(os.environ, {"SPLASH_WEIGHT_CACHE": str(root / "w")}),
                mock.patch.object(
                    upstream, "_weight_files", return_value={"d.safetensors"}
                ),
                mock.patch.object(
                    desktop_models.shutil,
                    "disk_usage",
                    return_value=mock.Mock(free=free),
                ),
            ):
                upstream.require_disk_space(
                    MODEL, fmt, target, set(target.files), draft
                )

    def test_insufficient_space_names_needed_and_available(self):
        # affine: 10 + 2 download + 10 prepared + 2.1 reserve = 24.1 GB
        with self.assertRaisesRegex(models.ModelError, r"24\.1 GB, 20\.0 GB available"):
            self.require(20 * self.GB)

    def test_sufficient_space_passes(self):
        self.require(25 * self.GB)

    def test_cached_files_are_not_counted_as_downloads(self):
        # target cached: 2 download + 10 prepared + 2.1 reserve = 14.1 GB
        self.require(15 * self.GB, cached=["blob-a.safetensors"])
        with self.assertRaises(models.ModelError):
            self.require(14 * self.GB, cached=["blob-a.safetensors"])

    def test_prism_estimates_prepared_images(self):
        # 10 + 2 download + 8.6 prepared + 2.1 reserve = 22.7 GB
        self.require(23 * self.GB, fmt="mlx-prism")
        with self.assertRaises(models.ModelError):
            self.require(22 * self.GB, fmt="mlx-prism")

    def test_unknown_sizes_skip_the_check(self):
        target = self.repo(MODEL, {})
        target.sizes = {}
        target.files = {"a.safetensors"}
        with (
            mock.patch.object(
                upstream, "_weight_files", return_value={"a.safetensors"}
            ),
            mock.patch.object(
                desktop_models.shutil, "disk_usage", side_effect=AssertionError
            ),
        ):
            upstream.require_disk_space(
                MODEL, "mlx-affine", target, target.files, target
            )


class PrismModelTests(unittest.TestCase):
    def check(self, config, *, hadamard=True, language_only=True):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, draft = repositories(root, target=config)
            if hadamard:
                target.files.add("hadamard.json")
            with mock.patch.object(
                hub.Repository,
                "resolve",
                side_effect=lambda name, **_kwargs: target if name == MODEL else draft,
            ):
                result = desktop_models.check_model(MODEL, models_root=root / "models")
            inspected = upstream.inspect_target(target, None, language_only)
            return result, inspected

    def test_prism_checkpoint_is_accepted_as_text_only(self):
        result, target = self.check(prism_config())
        self.assertEqual(result, MODEL)
        self.assertEqual((target.format, target.vision_format), ("mlx-prism", "none"))
        self.assertEqual(target.files["target/hadamard.json"], "hadamard.json")

    def test_prism_check_inspects_bonsai_draft(self):
        names = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, draft = repositories(root, target=prism_config())
            target.files.add("hadamard.json")

            def resolve(name, **_kwargs):
                names.append(name)
                return target if name == MODEL else draft

            with mock.patch.object(hub.Repository, "resolve", side_effect=resolve):
                desktop_models.check_model(MODEL, models_root=root / "models")
        self.assertEqual(names[1:], [upstream.PRISM_DRAFT_REPO])

    def test_default_draft_by_target_format_and_override(self):
        family = families.FAMILIES[0]
        for fmt, expected in (
            (upstream.PRISM_FORMAT, upstream.PRISM_DRAFT_REPO),
            ("mlx-affine", family.draft.repo),
        ):
            for override in (None, "mine/draft"):
                names = []
                selection = mock.Mock(draft_model=override)
                target = mock.Mock(files={"a"}, unreachable_reason=None)
                with mock.patch.object(
                    hub.Repository,
                    "resolve",
                    side_effect=lambda name, **_k: names.append(name),
                ):
                    upstream._resolve_draft(family, selection, None, target, fmt)
                self.assertEqual(names, [override or expected])
        self.assertNotEqual(upstream.PRISM_DRAFT_REPO, family.draft.repo)

    def test_two_bit_without_prism_discriminator_is_rejected(self):
        config = prism_config()
        del config["model_type"]
        with self.assertRaises(models.ModelError):
            self.check(config)

    def test_prism_with_wrong_schema_bits_or_missing_hadamard_is_rejected(self):
        wrong_bits = prism_config()
        wrong_bits["quantization"]["bits"] = 4
        wrong_group = prism_config()
        wrong_group["quantization"]["group_size"] = 64
        for label, config, hadamard in (
            ("schema", prism_config(schema_version=1), True),
            ("bits", wrong_bits, True),
            ("group", wrong_group, True),
            ("hadamard", prism_config(), False),
        ):
            with self.subTest(label), self.assertRaises(models.ModelError):
                self.check(config, hadamard=hadamard)

    def test_prism_requires_language_only(self):
        with self.assertRaises(models.ModelError):
            self.check(prism_config(), language_only=False)

    def test_ordinary_qwen_stays_affine_and_rejects_prism_shape_alone(self):
        result, target = self.check(target_config())
        self.assertEqual(target.format, "mlx-affine")
        self.assertNotIn("target/hadamard.json", target.files)

    def test_sixteen_gigabytes_admit_only_prism(self):
        with mock.patch.object(upstream, "_memory_bytes", return_value=16 * 1024**3):
            self.assertEqual(self.check(prism_config())[0], MODEL)
            with self.assertRaisesRegex(models.ModelError, "32 GB"):
                self.check(target_config())

    def test_install_gates_memory_and_disk_before_download_with_prism_draft(self):
        names = []
        for label, config, memory, free, message in (
            ("memory", target_config(), 16 * 1024**3, 100 * 10**9, "32 GB.*Bonsai"),
            ("disk", prism_config(), 16 * 1024**3, 10**9, "not enough free disk"),
        ):
            with self.subTest(label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                target, draft = repositories(root, target=config)
                target.files.add("hadamard.json")
                target.unreachable_reason = None
                selection = models.Selection.of(
                    root / "models", MODEL, language_only=True
                )

                def resolve(name, **_kwargs):
                    names.append(name)
                    return draft

                with (
                    mock.patch.object(hub.Repository, "resolve", side_effect=resolve),
                    mock.patch.object(upstream, "_memory_bytes", return_value=memory),
                    mock.patch.object(
                        upstream, "_weight_bytes", return_value=(10**10, 10**10)
                    ),
                    mock.patch.object(hub, "folder", return_value=root),
                    mock.patch.dict(
                        os.environ, {"SPLASH_WEIGHT_CACHE": str(root / "w")}
                    ),
                    mock.patch.object(
                        upstream.shutil, "disk_usage", return_value=mock.Mock(free=free)
                    ),
                    self.assertRaisesRegex(models.ModelError, message),
                ):
                    upstream._install(selection, target, None)
                self.assertEqual(target.downloads, [])
        self.assertEqual(names, [upstream.PRISM_DRAFT_REPO])

    def test_serve_uses_automatic_context_below_32_gigabytes(self):
        for memory, expected in ((16 * 1024**3, False), (32 * 1024**3, True)):
            commands = []
            with (
                mock.patch.object(
                    desktop_models, "check_model", side_effect=lambda model: model
                ),
                mock.patch.object(upstream, "_memory_bytes", return_value=memory),
                mock.patch.object(
                    desktop_models.os,
                    "execv",
                    side_effect=lambda _python, command: commands.append(command),
                ),
            ):
                desktop_models._serve(MODEL)
            with self.subTest(memory=memory):
                self.assertEqual("--max-context" in commands[0], expected)
                self.assertEqual(commands[0][-2:], ["--port", "8000"])


if __name__ == "__main__":
    unittest.main()
