import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tools import check_architecture


class ArchitectureTests(unittest.TestCase):
    def test_serving_modules_depend_on_lower_layers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = root / "server"
            server.mkdir()
            (server / "server.py").write_text("from .frontend import Frontend")
            (server / "frontend.py").write_text("from .backend import Job")
            (server / "backend.py").write_text("from . import runtime")
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                for statement in (
                    "from .frontend import Frontend",
                    "from frontend import Frontend",
                    "from server.frontend import Frontend",
                    "from server import frontend",
                    "import server.frontend",
                    "import importlib\nimportlib.import_module('server.frontend')",
                    "from importlib import import_module\nimport_module('.frontend', 'server')",
                    "__import__('server.frontend')",
                ):
                    with self.subTest(statement=statement):
                        (server / "backend.py").write_text(statement)
                        self.assertEqual(
                            check_architecture.check(),
                            ["server/backend.py: imports upper serving layer frontend"],
                        )

    def test_serving_entrypoint_is_not_a_shared_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "server/output.py"
            source.parent.mkdir()
            source.write_text("def parse():\n    from .server import FrontendHandler\n")
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(
                    check_architecture.check(),
                    ["server/output.py: imports upper serving layer server"],
                )

    def test_workspace_policy_belongs_to_operators(self):
        symbols = (
            "PrefillAttentionWave",
            "prefillAttentionTiles",
            "kQ8VerifySplits",
            "kQ8PrefillAttentionTileRows",
            "moeMaximumTiles",
            "kMoePrefillTileRows",
            "kMoeDecodeTileRows",
            "Q4DecodeKind",
            "Q4DecodeShape",
            "Q4PrefillShape",
            "kQ4PrefillTileRows",
            "narrowAffineKind",
            "narrowResidualKind",
            "headKind",
            "gdnInputGroups",
            "attentionGroups",
            "addPrefill128",
            "LinearTile",
            "LinearConfig",
            "LinearSimdgroups",
            "PrefillSplitMultiplier",
            "PrefillAttentionConfig",
            "VerifySplitCount",
            "VerifyAttentionConfig",
            "AttentionScalePlacement",
            "MoeExpertTile",
            "MoeExpertSimdgroups",
            "MoeConfig",
            "DraftAttentionConfiguration",
            "selectorShards",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            operator = root / "runtime/ops/Attention.cpp"
            operator.parent.mkdir(parents=True)
            operator.write_text("\n".join(symbols))
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                for relative in (
                    "runtime/model/Runtime.cpp",
                    "runtime/engine/Scheduler.cpp",
                    "runtime/engine/Bootstrap.hpp",
                    "runtime/engine/Bootstrap.mm",
                    "runtime/engine/RuntimeResources.hpp",
                    "runtime/engine/RuntimeResources.mm",
                ):
                    source = root / relative
                    source.parent.mkdir(parents=True, exist_ok=True)
                    for symbol in symbols:
                        with self.subTest(source=relative, symbol=symbol):
                            source.write_text(f"auto size = ops::{symbol};")
                            errors = check_architecture.check()
                            layer = relative.split("/")[1]
                            self.assertEqual(
                                errors,
                                [
                                    f"{relative}: {layer} owns an operator workspace policy"
                                ],
                            )
                    source.unlink()

    def test_only_engine_assembly_depends_on_concrete_models(self):
        headers = (
            "model/DFlashDraft.hpp",
            "model/ModelFactory.hpp",
            "model/Qwen3_6Moe.hpp",
            "model/Qwen3_8.hpp",
            "model/QwenHybridLayout.hpp",
            "model/QwenState.hpp",
            "model/QwenTarget.hpp",
            "model/QwenTargetFiles.hpp",
            "model/QwenTargetLoader.hpp",
            "model/Runtime.hpp",
            "model/WeightStore.hpp",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime/engine").mkdir(parents=True)
            for assembly in (
                "Bootstrap.hpp",
                "Bootstrap.mm",
                "RuntimeResources.hpp",
                "RuntimeResources.mm",
            ):
                (root / "runtime/engine" / assembly).write_text(
                    "".join(f'#include "{header}"\n' for header in headers)
                )
            policy = root / "runtime/engine/Scheduler.cpp"
            startup = root / "runtime/main.mm"
            generic = (
                '#include "model/Model.hpp"\n#include "model/ModelDescriptor.hpp"\n'
            )
            policy.write_text(generic)
            startup.write_text(generic + "model::ModelDescriptor model;\n")
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                for header in headers:
                    with self.subTest(source="engine policy", header=header):
                        policy.write_text(f'#include "{header}"\n')
                        self.assertEqual(
                            check_architecture.check(),
                            [
                                "runtime/engine/Scheduler.cpp: engine policy "
                                f"depends on concrete model {header}"
                            ],
                        )
                policy.write_text(generic)
                for header in headers:
                    with self.subTest(source="startup", header=header):
                        startup.write_text(f'#include "{header}"\n')
                        if header == "model/ModelFactory.hpp":
                            self.assertEqual(check_architecture.check(), [])
                            continue
                        self.assertEqual(
                            check_architecture.check(),
                            [
                                "runtime/main.mm: startup depends on concrete "
                                f"model {header}"
                            ],
                        )
                for name in ("QwenTarget", "DFlashDraft", "Runtime"):
                    with self.subTest(source="startup", symbol=name):
                        startup.write_text(f"auto instance = model::{name}{{}};\n")
                        self.assertEqual(
                            check_architecture.check(),
                            ["runtime/main.mm: startup names a concrete model type"],
                        )

    def test_metal_depends_on_no_upper_production_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime/model").mkdir(parents=True)
            (root / "runtime/model/Model.hpp").write_text("")
            kernel = root / "runtime/metal/kernels/shared/rope.metal"
            kernel.parent.mkdir(parents=True)
            with mock.patch.object(check_architecture, "ROOT", root):
                for include, header in (
                    ('#include "engine/Engine.hpp"', "engine/Engine.hpp"),
                    ('#include "model/Model.hpp"', "model/Model.hpp"),
                    ('#include "ops/Linear.hpp"', "ops/Linear.hpp"),
                    ('#include "../../../model/Model.hpp"', "model/Model.hpp"),
                ):
                    with self.subTest(include=include):
                        kernel.write_text(include + "\n")
                        self.assertEqual(
                            check_architecture.check(),
                            [
                                "runtime/metal/kernels/shared/rope.metal: Metal "
                                f"depends on production layer {header}"
                            ],
                        )

    def test_includes_are_read_where_the_compiler_finds_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime/model").mkdir(parents=True)
            (root / "runtime/model/QwenTarget.hpp").write_text("")
            policy = root / "runtime/engine/Scheduler.cpp"
            policy.parent.mkdir()
            with mock.patch.object(check_architecture, "ROOT", root):
                for include in (
                    '# include "model/QwenTarget.hpp"',
                    '#include"model/QwenTarget.hpp"',
                    "#  import <model/QwenTarget.hpp>",
                    '#include "../model/QwenTarget.hpp"',
                    '#include "model/../model/QwenTarget.hpp"',
                ):
                    with self.subTest(include=include):
                        policy.write_text(include + "\n")
                        self.assertEqual(
                            check_architecture.check(),
                            [
                                "runtime/engine/Scheduler.cpp: engine policy "
                                "depends on concrete model model/QwenTarget.hpp"
                            ],
                        )

    def test_production_cannot_include_offline_tuning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runtime/ops/Linear.cpp"
            source.parent.mkdir(parents=True)
            source.write_text('#include "tuning/Measurement.hpp"\n')
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(
                    check_architecture.check(),
                    [
                        "runtime/ops/Linear.cpp: production depends on "
                        "offline tuning tuning/Measurement.hpp"
                    ],
                )

    def test_plan_orchestration_and_storage_geometry_remain_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "runtime/model/Runtime.cpp"
            assembly = root / "runtime/engine/RuntimeResources.mm"
            model.parent.mkdir(parents=True)
            assembly.parent.mkdir(parents=True)
            model.write_text(
                "auto bytes = PagedAttention::prefillWorkspace();\n"
                "constexpr auto tileRows = kv::kPageTokens;\n"
                "auto rows = ExecutionLimits::draftQueryRows;\n"
                "auto rank = layout.selectorRank;\n"
            )
            assembly.write_text(
                '#include "model/Runtime.hpp"\n'
                "ops::ExecutionPlans plans(device);\n"
                "ops::OperatorChoices choices;\n"
                "plans.install(choices);\n"
                "ops::tuning::MeasurementOptions options;\n"
            )
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
