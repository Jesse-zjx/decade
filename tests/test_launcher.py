"""Offline checks for configuration, launch admission, and metadata binding."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_study


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "config.json"
        self.example = json.loads((run_study.ROOT / "configs/f1.example.json").read_text(encoding="utf-8"))
        self.config_path.write_text(json.dumps(self.example), encoding="utf-8")

    def config(self):
        return run_study.load_config(self.config_path, self.root)

    def test_relative_credential_path_is_bound_to_repository(self):
        self.assertEqual(self.config()["credential_file"], str(self.root / ".secrets/service.env"))

    def test_unsafe_or_incompatible_config_is_rejected(self):
        for key, value in (("run_id", "../escape"), ("group", "../escape"),
                           ("search_seed", True), ("cpu", -1), ("task", "other"),
                           ("api_base_url", "http://api.example.invalid/v1")):
            with self.subTest(key=key):
                config = copy.deepcopy(self.example)
                config[key] = value
                self.config_path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.config()

    def test_failed_checks_never_launch_even_with_run_flag(self):
        with patch.object(run_study, "preflight", return_value=(["Data unavailable"], {})), \
                patch.object(run_study, "launch") as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_study.main(["--config", str(self.config_path), "--run"]), 1)
            launch.assert_not_called()

    def test_default_mode_remains_read_only_when_checks_pass(self):
        with patch.object(run_study, "preflight", return_value=([], {})), \
                patch.object(run_study, "launch") as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_study.main(["--config", str(self.config_path)]), 0)
            launch.assert_not_called()
        self.assertFalse((self.root / "engineering").exists())

    def prepared_config(self):
        config = self.config()
        config.update(runtime=str(self.root / "runtime"), api_base_url="https://service.example.test/v1")
        runtime = Path(config["runtime"]) / "bin/python"
        runtime.parent.mkdir(parents=True)
        runtime.touch()
        credential = Path(config["credential_file"])
        credential.parent.mkdir(parents=True)
        credential.write_text("BASE_URL=https://service.example.test/v1\nAPI_KEY=test-only\nMODEL_NAME=DeepSeek-V4-Flash\n", encoding="utf-8")
        credential.chmod(0o600)
        worker = self.root / "fuse_patch/rpc_worker.py"
        worker.parent.mkdir()
        worker.write_text("# Fixture; never executed.\n", encoding="utf-8")
        panel = self.root / "panels/tsp/group01/feedback-0.json"
        panel.parent.mkdir(parents=True)
        rows = [dict(instance_id=f"demo-{i}", execution_seed=i,
                     distance_matrix=[[0.0 if a == b else 1.0 for b in range(50)] for a in range(50)]) for i in range(64)]
        panel.write_text(json.dumps({"instances": rows}), encoding="utf-8")
        return config, panel

    def inspect(self, config):
        with patch.object(run_study.platform, "system", return_value="Linux"), \
                patch.object(run_study.shutil, "which", return_value="docker"), \
                patch.object(run_study.os, "sched_getaffinity", return_value={0}, create=True), \
                patch.object(run_study.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="sha256:" + "0" * 64)):
            return run_study.preflight(config, self.root)

    def test_panel_identity_and_existing_state_gate(self):
        config, panel = self.prepared_config()
        errors, identity = self.inspect(config)
        self.assertEqual(errors, [])
        self.assertEqual(len(identity["panels"]), 1)
        rows = json.loads(panel.read_text(encoding="utf-8"))
        rows["instances"][1]["instance_id"] = rows["instances"][0]["instance_id"]
        panel.write_text(json.dumps(rows), encoding="utf-8")
        self.assertTrue(any("64 valid" in error for error in self.inspect(config)[0]))
        (self.root / "runs").mkdir()
        self.assertTrue(any("Existing study state" in error for error in self.inspect(config)[0]))

    def test_launch_freezes_sources_and_passes_configuration_to_child(self):
        config, _ = self.prepared_config()
        errors, identity = self.inspect(config)
        self.assertEqual(errors, [])
        with patch.object(run_study.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as process:
            self.assertEqual(run_study.launch(config, identity, self.root), 0)
            args, kwargs = process.call_args
            self.assertEqual(json.loads(args[0][3])["method"], "F1")
            self.assertEqual(kwargs["env"]["DECADE_EXECUTION_ROOT"], str(self.root))
            self.assertEqual(kwargs["env"]["DECADE_CREDENTIAL_FILE"], config["credential_file"])
        runtime = json.loads((self.root / "engineering/RUNTIME.json").read_text(encoding="utf-8"))
        self.assertEqual(runtime["image"], identity["image"])
        frozen = json.loads((self.root / "engineering/SOURCE_FREEZE.json").read_text(encoding="utf-8"))
        self.assertIn("fuse_patch/rpc_worker.py", frozen["files"])
        with self.assertRaises(FileExistsError):
            run_study.launch(config, identity, self.root)


if __name__ == "__main__":
    unittest.main()
