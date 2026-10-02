"""Metadata-only model gate for the packaged Splash M1 desktop app."""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

if __package__:
    from . import assembly, families, hub, launcher, models, paths, upstream
else:
    import assembly
    import families
    import hub
    import launcher
    import models
    import paths
    import upstream

COMPATIBILITY_MESSAGE = (
    "Metadata is compatible; tensor validation follows during startup."
)
DESKTOP_FAMILY = "Qwen3.8-27B"
# Below this only the 2-bit Prism checkpoint fits, with an automatic context.
FULL_MEMORY_BYTES = 32 * 1024**3
HF_HOSTS = {"hf.co", "huggingface.co"}


def _memory_bytes():
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def _require_memory(target_format):
    if target_format != upstream.PRISM_FORMAT and _memory_bytes() < FULL_MEMORY_BYTES:
        raise models.ModelError(
            "this model requires 32 GB of memory; Macs with less can run the 2-bit Prism (Bonsai) checkpoint"
        )


def normalize_model_id(value):
    """Accept a Hub repository ID or its default-branch model URL."""
    if not isinstance(value, str) or not value.strip():
        raise models.ModelError("enter a Hugging Face model ID or model URL")
    value = value.strip()
    if "://" in value:
        try:
            parsed = urlsplit(value)
            host = parsed.hostname
            port = parsed.port
        except ValueError:
            raise models.ModelError("invalid Hugging Face model URL") from None
        if (
            parsed.scheme != "https"
            or host not in HF_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or parsed.query
            or parsed.fragment
        ):
            raise models.ModelError(
                "use an HTTPS Hugging Face model URL on huggingface.co or hf.co"
            )
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 4 and parts[2:] == ["tree", "main"]:
            parts = parts[:2]
        elif len(parts) != 2:
            raise models.ModelError(
                "use a model URL for the default branch, not a file or other revision"
            )
        value = "/".join(parts)

    try:
        models.parse_model_id(value)
        repo_id, variant = models.split_model_id(value)
    except (argparse.ArgumentTypeError, models.ModelError) as error:
        raise models.ModelError(str(error)) from None
    if variant is not None:
        raise models.ModelError("model variants are not supported by the desktop app")
    return repo_id


def _config_value(config, key):
    value = upstream._config_value(config, key)
    if value is None and key == "rope_parameters.rope_type":
        value = upstream._config_value(config, "rope_parameters.type")
    return value


def _matches(config, key, expected):
    value = _config_value(config, key)
    choices = expected if isinstance(expected, tuple) else (expected,)
    return any(upstream._same(value, choice) for choice in choices)


def _require_target_metadata(config, target_format="mlx-affine"):
    family = families.family_for(config)
    if family.name != DESKTOP_FAMILY:
        raise models.ModelError(
            f"the desktop app supports {DESKTOP_FAMILY} text checkpoints only"
        )

    expected = (
        ("intermediate_size", 17408),
        ("linear_num_key_heads", 16),
        ("linear_num_value_heads", 48),
        ("linear_key_head_dim", 128),
        ("linear_value_head_dim", 128),
        ("linear_conv_kernel_dim", 4),
        ("full_attention_interval", 4),
        ("rms_norm_eps", 1e-6),
        ("attention_bias", (False, 0)),
        ("attn_output_gate", (True, 1)),
        ("tie_word_embeddings", (False, 0)),
        ("hidden_act", "silu"),
        ("rope_parameters.rope_theta", 10000000),
        ("rope_parameters.partial_rotary_factor", 0.25),
        ("rope_parameters.rope_type", "default"),
    )
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise models.ModelError("upstream configuration has no text_config")
    for key, value in expected:
        if not _matches(text, key, value):
            raise models.ModelError(
                f"unsupported Qwen3.8 configuration: text_config.{key}"
            )

    layer_types = _config_value(text, "layer_types")
    expected_layers = tuple(
        "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
        for layer in range(64)
    )
    if layer_types != expected_layers:
        raise models.ModelError("unsupported Qwen3.8 attention layer schedule")

    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        raise models.ModelError("desktop models require MLX affine quantization")
    if target_format == upstream.PRISM_FORMAT:
        upstream.require_prism_config(config)
        if not set(quantization) <= {"mode", "bits", "group_size"}:
            raise models.ModelError("unsupported Prism quantization override")
        return family
    if (
        quantization.get("mode", "affine") != "affine"
        or quantization.get("bits") != 4
        or quantization.get("group_size") != 64
    ):
        raise models.ModelError(
            "desktop models require MLX affine Q4/group-64 quantization"
        )
    for name, entry in quantization.items():
        if name in {"mode", "bits", "group_size"}:
            continue
        if not isinstance(entry, dict):
            raise models.ModelError(f"invalid affine quantization override: {name}")
        mode = entry.get("mode", quantization.get("mode", "affine"))
        bits = entry.get("bits", quantization["bits"])
        group_size = entry.get("group_size", quantization["group_size"])
        if mode != "affine" or bits not in (4, 5) or group_size != 64:
            raise models.ModelError(f"unsupported affine quantization override: {name}")
    return family


def _require_draft_config(config, family):
    differences = [
        f"{key} {_config_value(config, key)!r}, not {expected!r}"
        for key, expected in family.draft.signature
        if not upstream._same(_config_value(config, key), expected)
    ]
    if differences:
        raise models.ModelError(
            f"draft configuration is incompatible with {family.name}: "
            + "; ".join(differences)
        )


def _require_draft_repository(repo, family):
    if "config.json" not in repo.files:
        raise models.ModelError(
            f"{repo.name} does not contain a DFlash2 checkpoint for {family.name} "
            "(no config.json)"
        )
    try:
        upstream._weight_files(repo)
    except models.ModelError as error:
        raise models.ModelError(
            f"{repo.name} does not contain a DFlash2 checkpoint for {family.name} "
            f"({error})"
        ) from error
    _require_draft_config(models.read_json(repo.file("config.json")), family)


def _verified_assembly(selection, model):
    if models.installation_kind(selection.link) != models.ASSEMBLY:
        return None
    try:
        record = assembly.verify(selection.link, full=False)
    except (models.ModelError, OSError):
        return None
    if record["model"] != model:
        return None
    return record


def _require_installed_metadata(link, record, model):
    if (
        record["model"] != model
        or record["target_format"] not in ("mlx-affine", upstream.PRISM_FORMAT)
        or record["vision_format"] != "none"
    ):
        raise models.ModelError("installed assembly is not a desktop text checkpoint")
    files = record["files"]
    if not any(
        name.startswith("target/") and name.endswith(".safetensors") for name in files
    ) or not any(
        name.startswith("draft/") and name.endswith(".safetensors") for name in files
    ):
        raise models.ModelError("installed assembly is missing checkpoint weights")
    target_config = models.read_json(link / "config.json")
    family = _require_target_metadata(target_config, record["target_format"])
    _require_memory(record["target_format"])
    if record["target_format"] == upstream.PRISM_FORMAT and (
        "target/hadamard.json" not in files
    ):
        raise models.ModelError("installed assembly is missing hadamard.json")
    if record["family"] != family.name:
        raise models.ModelError("installed model family does not match its metadata")
    draft_config = models.read_json(link / "draft/config.json")
    _require_draft_config(draft_config, family)
    return family


def check_model(value, *, models_root=None):
    """Check target and draft metadata without downloading model weights.

    Returns the canonical model ID used by both the desktop UI and launcher.
    """
    model = normalize_model_id(value)
    models_root = Path(models_root or paths.MODELS)
    selection = models.Selection.of(models_root, model, language_only=True)
    installed = _verified_assembly(selection, model)
    installed_revision = (
        installed["sources"]["target"]["revision"] if installed else None
    )
    target_repo = hub.Repository.resolve(
        model, installation=selection.link, installed=installed_revision
    )

    if not target_repo.files:
        if installed is None:
            raise models.ModelError(
                f"no verified installed assembly is available for {model}"
            )
        _require_installed_metadata(selection.link, installed, model)
        return model

    if not any(name.endswith(".safetensors") for name in target_repo.files):
        raise models.ModelError(
            "the desktop app requires an MLX safetensors checkpoint; GGUF is not supported"
        )
    with hub.as_model_errors(f"cannot inspect {model}"):
        target = upstream.inspect_target(target_repo, selection.variant, True)
    if (
        target.format not in ("mlx-affine", upstream.PRISM_FORMAT)
        or target.vision_format != "none"
    ):
        raise models.ModelError(
            "the desktop app requires a text-only MLX affine checkpoint"
        )
    family = _require_target_metadata(target.config, target.format)
    _require_memory(target.format)
    draft_repo = hub.Repository.resolve(
        upstream.default_draft(family, target.format), installation=selection.link
    )
    with hub.as_model_errors(f"cannot inspect the {family.name} draft"):
        _require_draft_repository(draft_repo, family)
    return model


def installed_models(models_root=None):
    """List compatible assemblies using local metadata only."""
    models_root = Path(models_root or paths.MODELS)
    installed = {}
    for link in models.selection_links(models_root):
        try:
            if models.installation_kind(link) != models.ASSEMBLY:
                continue
            record = assembly.verify(link, full=False)
            model = normalize_model_id(record["model"])
            _require_installed_metadata(link, record, model)
        except (models.ModelError, OSError, TypeError):
            continue
        installed.setdefault(model, {"model": model})
    return [installed[model] for model in sorted(installed)]


def _require_stopped():
    try:
        lock = (paths.RUNTIME / f"serve-{launcher.PORT}.lock").open("r")
    except FileNotFoundError:
        return
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise models.ModelError(
                f"Splash is serving{launcher._serve_lock_owner(lock)}; stop it first."
            ) from None
        fcntl.flock(lock, fcntl.LOCK_UN)


def _weight_cache():
    path = os.environ.get("SPLASH_WEIGHT_CACHE")
    return Path(path) if path else Path.home() / "Library/Caches/Splash/weights"


def _tree_bytes(path):
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _prepared_entries(assemblies):
    """Prepared weight entries made from the target or vision files of these
    assemblies; the draft's are shared and never listed."""
    prefixes = tuple(
        f"{path}/{part}" for path in assemblies for part in ("target", "vision")
    )
    entries = []
    cache = _weight_cache()
    for entry in sorted(cache.iterdir()) if cache.is_dir() else ():
        if not models.is_hex_digest(entry.name, 64):
            continue
        try:
            lines = (entry / "source").read_text().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        if any(line.removeprefix("source ").startswith(prefixes) for line in lines):
            entries.append(entry)
    return entries


def _only_splash_pins(snapshot, owners):
    """Whether no installation but owners pins the snapshot."""
    pins = snapshot.parent.parent / "refs" / "splash"
    return not any(
        pin.parent.name not in owners for pin in pins.glob(f"*/{snapshot.name}")
    )


def _revision_strategy(snapshot):
    from huggingface_hub import scan_cache_dir

    return scan_cache_dir(snapshot.parents[2]).delete_revisions(snapshot.name)


def _delete_plan(model, models_root):
    model = normalize_model_id(model)
    drafts = {family.draft.repo for family in families.FAMILIES} | {
        upstream.PRISM_DRAFT_REPO
    }
    links, assemblies, pins = [], set(), {}
    for link in models.selection_links(models_root):
        try:
            record = models.read_json(link / "model.json")
        except models.ModelError:
            continue
        if record.get("model") != model:
            continue
        links.append(link)
        resolved = link.resolve()
        assemblies |= {resolved, models_root / ".resolved" / resolved.name}
        pins[link] = assembly.recorded_pins(link)
    if not links:
        raise models.ModelError(f"{model} is not installed")
    _require_stopped()
    if any(assembly.is_held(link.resolve()) for link in links):
        raise models.ModelError(f"{model} is in use; stop Splash first.")
    owners = {hub.pin_owner(link) for link in links}
    snapshots = sorted(
        {
            snapshot
            for linked in pins.values()
            for snapshot, repo in linked
            if repo not in drafts and _only_splash_pins(snapshot, owners)
            if snapshot.is_dir()
        }
    )
    prepared = _prepared_entries(sorted(map(str, assemblies)))
    freed = sum(_revision_strategy(path).expected_freed_size for path in snapshots)
    return {
        "links": links,
        "pins": pins,
        "snapshots": snapshots,
        "prepared": prepared,
        "bytes": freed + sum(map(_tree_bytes, prepared)),
    }


def delete_plan(model, models_root=None):
    """What deleting model would remove, without removing anything."""
    return _delete_plan(model, Path(models_root or paths.MODELS))


def delete(model, models_root=None):
    """Delete the installation of model. Its target snapshot goes unless
    another installation pins it; the draft's snapshot and prepared weights
    are shared and stay."""
    models_root = Path(models_root or paths.MODELS)
    with models.installation_lock(models_root):
        plan = _delete_plan(model, models_root)
        for link, linked in plan["pins"].items():
            for snapshot, _repo in linked:
                pin = hub.pinned(snapshot, link)
                pin.unlink(missing_ok=True)
                try:
                    pin.parent.rmdir()
                except OSError:
                    pass
            link.unlink()
        assembly.collect_garbage(models_root)
        for snapshot in plan["snapshots"]:
            _revision_strategy(snapshot).execute()
    cache = _weight_cache()
    if plan["prepared"]:
        with (cache / "prepare.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for entry in plan["prepared"]:
                staging = entry.with_name(entry.name + ".partial")
                shutil.rmtree(staging, ignore_errors=True)
                try:
                    entry.rename(staging)
                except OSError:
                    continue
                shutil.rmtree(staging, ignore_errors=True)


def _serve(model):
    model = check_model(model)
    command = [
        str(paths.PYTHON),
        "-u",
        str(paths.ROOT / "install/launcher.py"),
        "serve",
        "--model",
        model,
        "--language-only",
        "--port",
        "8000",
    ]
    if _memory_bytes() >= FULL_MEMORY_BYTES:
        command[-2:-2] = ["--max-context", "32K"]
    os.execv(str(paths.PYTHON), command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "serve"))
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "serve":
            _serve(args.model)
        else:
            check_model(args.model)
            print(COMPATIBILITY_MESSAGE, flush=True)
            return 0
    except (models.ModelError, OSError) as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
