"""Step 6: audit-script modes, pre-commit hook installer, Makefile and CI workflow."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from uni_thesis.cli import main as cli_main

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_anti_hallucination.py"


def _script(*args, cwd=REPO):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
                          timeout=600, cwd=cwd)


def test_audit_results_dir_mode(tmp_path):
    res = tmp_path / "res"
    assert cli_main(["run", "--profile", str(REPO / "profiles" / "demo_rsm_opt.yaml"), "--results-dir", str(res),
                     "--offline", "--no-mirror"]) == 0
    (res / "stray_dir").mkdir()  # ignored: no metrics/thesis
    proc = _script("--results-dir", str(res), "--json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["passed"] is True and len(doc["reports"]) == 1
    chap = res / "demo_rsm_opt" / "thesis" / "chapters" / "chap5_conclusion.tex"
    chap.write_text(chap.read_text(encoding="utf-8") + "\nThe yield reached 97.32 percent.\n", encoding="utf-8")
    proc = _script("--results-dir", str(res))
    assert proc.returncode == 1 and "FAILED" in proc.stdout


def test_audit_results_dir_empty(tmp_path):
    proc = _script("--results-dir", str(tmp_path / "none"))
    assert proc.returncode == 1 and "no generated thesis" in proc.stderr


def test_audit_hermetic_mode_and_bad_template(tmp_path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    shutil.copy(REPO / "profiles" / "demo_rsm_opt.yaml", profiles)
    proc = _script("--profiles-dir", str(profiles), "--template-dir", str(REPO / "templates" / "thesis_tex"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1/1 thesis audit(s) passed" in proc.stdout
    # a template with a hand-typed number must be rejected
    bad = tmp_path / "bad_tex"
    shutil.copytree(REPO / "templates" / "thesis_tex", bad, ignore=shutil.ignore_patterns("generated"))
    chap = bad / "chapters" / "chap4_experiments.tex"
    chap.write_text(chap.read_text(encoding="utf-8") + "\nOur method improves accuracy by 12.5\\%.\n",
                    encoding="utf-8")
    proc = _script("--profiles-dir", str(profiles), "--template-dir", str(bad), "--json")
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["passed"] is False
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = _script("--profiles-dir", str(empty))
    assert proc.returncode == 1 and "no profile" in proc.stderr


def test_audit_explicit_args_must_be_complete():
    proc = _script("--tex-dir", "x")
    assert proc.returncode == 2 and "must be given together" in proc.stderr


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_install_hooks_blocks_failing_commit(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "install_hooks.sh", repo / "scripts")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@e"}
    run = lambda *a, **kw: subprocess.run(list(a), cwd=repo, capture_output=True, text=True, env=kw.get("env", env))
    assert run("git", "init", "-q").returncode == 0
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    (hooks / "pre-commit").write_text("#!/bin/sh\nexit 0\n")
    out = run("bash", "scripts/install_hooks.sh")
    assert out.returncode == 0, out.stderr
    hook = hooks / "pre-commit"
    text = hook.read_text()
    assert os.access(hook, os.X_OK) and (hooks / "pre-commit.bak").exists()
    assert "pytest" in text and "audit_anti_hallucination.py" in text and "--cov-fail-under=90" in text
    # re-running is idempotent (no second backup of our own hook)
    (hooks / "pre-commit.bak").unlink()
    assert run("bash", "scripts/install_hooks.sh").returncode == 0
    assert not (hooks / "pre-commit.bak").exists()
    # a fake python whose pytest fails must block the commit with a non-zero status
    fake = tmp_path / "fakepy"
    fake.write_text("#!/bin/sh\nexit 3\n")
    fake.chmod(0o755)
    (repo / "f.txt").write_text("x")
    run("git", "add", "f.txt")
    blocked = run("git", "commit", "-q", "-m", "dirty", env={**env, "PYTHON": str(fake)})
    assert blocked.returncode != 0 and "BLOCKED" in blocked.stderr
    ok = tmp_path / "okpy"
    ok.write_text("#!/bin/sh\nexit 0\n")
    ok.chmod(0o755)
    passed = run("git", "commit", "-q", "-m", "clean", env={**env, "PYTHON": str(ok)})
    assert passed.returncode == 0, passed.stderr


def test_makefile_targets():
    text = (REPO / "Makefile").read_text()
    for target in ("init:", "test:", "run:", "audit:", "all:"):
        assert f"\n{target}" in text
    assert "--cov-fail-under=$(COV_MIN)" in text and "COV_MIN     ?= 90" in text
    if shutil.which("make"):
        dry = subprocess.run(["make", "-n", "run", "PROFILE=profiles/demo_rsm_opt.yaml"], cwd=REPO,
                             capture_output=True, text=True)
        assert dry.returncode == 0 and "run --profile profiles/demo_rsm_opt.yaml" in dry.stdout


def test_ci_workflow_structure():
    wf = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text())
    triggers = wf[True] if True in wf else wf["on"]  # PyYAML parses bare `on` as boolean True
    assert triggers["push"]["branches"] == ["main"] and triggers["push"]["tags"] == ["v*"]
    assert "pull_request" in triggers
    jobs = wf["jobs"]
    assert set(jobs) == {"test-and-audit", "build-thesis-pdf"}
    assert jobs["build-thesis-pdf"]["needs"] == "test-and-audit"
    steps1 = json.dumps(jobs["test-and-audit"]["steps"])
    assert "3.12" in steps1 and "--cov-fail-under=90" in steps1 and "audit_anti_hallucination.py" in steps1
    steps2 = jobs["build-thesis-pdf"]["steps"]
    blob = json.dumps(steps2)
    for pkg in ("texlive-latex-base", "texlive-latex-recommended", "texlive-latex-extra",
                "texlive-fonts-recommended", "texlive-science"):
        assert pkg in blob
    assert "thesis_biosignal.pdf" in blob and "thesis_rsm.pdf" in blob
    uses = [s.get("uses", "") for s in steps2]
    assert "actions/upload-artifact@v4" in uses and "softprops/action-gh-release@v2" in uses
    release = next(s for s in steps2 if s.get("uses") == "softprops/action-gh-release@v2")
    assert release["if"] == "startsWith(github.ref, 'refs/tags/v')"
    assert jobs["build-thesis-pdf"]["permissions"]["contents"] == "write"


def test_readme_and_tutorial_profile():
    from uni_thesis.schema import TopicProfile

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for needle in ("五层架构", "快速开始", "防幻觉机制", "tutorial_concrete_mix.yaml", "make all"):
        assert needle in readme
    tut = REPO / "profiles" / "examples" / "tutorial_concrete_mix.yaml"
    assert len(tut.read_text(encoding="utf-8").splitlines()) <= 50
    profile = TopicProfile.from_yaml(tut)
    assert profile.domain_type == "parameter_optimization" and profile.design is not None
