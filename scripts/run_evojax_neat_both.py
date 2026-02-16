#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

def detect_repo_dir(repo_arg: str | None) -> Path:
    if repo_arg:
        repo = Path(repo_arg).expanduser().resolve()
        if (repo / "pyproject.toml").exists() and (repo / "src" / "jax_neat_evojax").exists():
            return repo
        raise RuntimeError(f"Invalid --repo-dir: {repo}")

    candidates = [Path.cwd(), *Path.cwd().parents]
    for p in candidates:
        if (p / "pyproject.toml").exists() and (p / "src" / "jax_neat_evojax").exists():
            return p.resolve()
    raise RuntimeError("Could not find project root with pyproject.toml and src/jax_neat_evojax.")


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def run_mode(
    python_bin: str,
    repo_dir: Path,
    env: dict[str, str],
    mode: str,
    generations: int,
    pop_size: int,
    episodes_per_genome: int,
    max_steps: int,
    out_root: Path,
    evojax_path: Path,
) -> None:
    cmd = [
        python_bin,
        "-m",
        "jax_neat_evojax.cli",
        "--mode",
        mode,
        "--generations",
        str(generations),
        "--pop-size",
        str(pop_size),
        "--episodes-per-genome",
        str(episodes_per_genome),
        "--max-steps",
        str(max_steps),
        "--out-root",
        str(out_root),
        "--evojax-path",
        str(evojax_path),
    ]
    print(f"\nRunning mode: {mode}")
    t0 = time.time()
    run_cmd(cmd, cwd=repo_dir, env=env)
    print(f"Completed {mode} in {time.time() - t0:.1f}s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run both EvoJAX NEAT modes with fixed local paths.")
    p.add_argument("--repo-dir", type=str, default=None)
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--pop-size", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--episodes-direct", type=int, default=2)
    p.add_argument("--episodes-selfplay", type=int, default=1)
    p.add_argument("--install-deps", action="store_true")
    p.add_argument("--evojax-url", type=str, default="https://github.com/google/evojax.git")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = detect_repo_dir(args.repo_dir)
    out_root = repo_dir / "artifacts"
    evojax_path = repo_dir / "evojax"
    out_root.mkdir(parents=True, exist_ok=True)
    if not evojax_path.exists():
        run_cmd(["git", "clone", "--depth", "1", args.evojax_url, str(evojax_path)], cwd=repo_dir, env=os.environ.copy())

    src_dir = repo_dir / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{src_dir}:{env.get('PYTHONPATH', '')}"
    env["MPLCONFIGDIR"] = "/tmp/mplconfig"
    Path("/tmp/mplconfig").mkdir(parents=True, exist_ok=True)

    print("python_bin:", args.python_bin)
    print("repo_dir  :", repo_dir)
    print("out_root  :", out_root)
    print("evojax    :", evojax_path)

    if args.install_deps:
        run_cmd([args.python_bin, "-m", "pip", "install", "-U", "pip"], cwd=repo_dir, env=env)
        run_cmd([args.python_bin, "-m", "pip", "install", "-r", str(repo_dir / "requirements.txt")], cwd=repo_dir, env=env)
        run_cmd(
            [
                args.python_bin,
                "-m",
                "pip",
                "install",
                "evojax==0.2.17",
                "flax",
                "optax",
                "chex",
                "orbax-checkpoint",
                "tensorstore",
                "rich",
                "absl-py",
                "cma",
                "opencv-python-headless",
            ],
            cwd=repo_dir,
            env=env,
        )
        run_cmd([args.python_bin, "-m", "pip", "install", "-e", str(repo_dir)], cwd=repo_dir, env=env)

    run_mode(
        python_bin=args.python_bin,
        repo_dir=repo_dir,
        env=env,
        mode="direct_vs_builtin",
        generations=args.generations,
        pop_size=args.pop_size,
        episodes_per_genome=args.episodes_direct,
        max_steps=args.max_steps,
        out_root=out_root,
        evojax_path=evojax_path,
    )
    run_mode(
        python_bin=args.python_bin,
        repo_dir=repo_dir,
        env=env,
        mode="selfplay_then_builtin",
        generations=args.generations,
        pop_size=args.pop_size,
        episodes_per_genome=args.episodes_selfplay,
        max_steps=args.max_steps,
        out_root=out_root,
        evojax_path=evojax_path,
    )


if __name__ == "__main__":
    main()
