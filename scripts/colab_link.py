#!/usr/bin/env python3
"""Generate a Colab URL for a notebook tracked in the current git repo."""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote


def run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def normalize_remote(remote_url: str) -> str:
    if remote_url.startswith("git@github.com:"):
        remote_url = remote_url.replace("git@github.com:", "https://github.com/")
    if remote_url.endswith(".git"):
        remote_url = remote_url[:-4]
    if not remote_url.startswith("https://github.com/"):
        raise ValueError(
            "Only GitHub remotes are supported for Colab URL generation. Found: {0}".format(
                remote_url
            )
        )
    return remote_url


def build_urls(repo_url: str, branch: str, repo_relative_notebook: str) -> tuple[str, str]:
    repo_path = repo_url.replace("https://github.com/", "")
    github_blob_url = "https://github.com/{0}/blob/{1}/{2}".format(
        repo_path,
        quote(branch, safe=""),
        quote(repo_relative_notebook),
    )
    colab_url = "https://colab.research.google.com/github/{0}/blob/{1}/{2}".format(
        repo_path,
        quote(branch, safe=""),
        quote(repo_relative_notebook),
    )
    return github_blob_url, colab_url


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the GitHub and Google Colab URLs for a notebook in this repo."
    )
    parser.add_argument(
        "notebook",
        help="Path to the .ipynb file, absolute or relative to the current working directory.",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Git remote name to use. Default: origin",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Branch name to embed in the URL. Default: current branch",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated Colab URL in the default browser.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    notebook_path = Path(args.notebook).expanduser().resolve()
    if not notebook_path.exists():
        print("Notebook not found: {0}".format(notebook_path), file=sys.stderr)
        return 1

    git_root = Path(run_git(["rev-parse", "--show-toplevel"], cwd))
    try:
        relative_path = notebook_path.relative_to(git_root).as_posix()
    except ValueError:
        print(
            "Notebook must live inside the current git repository.\n"
            "Git root: {0}\nNotebook: {1}".format(git_root, notebook_path),
            file=sys.stderr,
        )
        return 1

    remote_url = normalize_remote(run_git(["remote", "get-url", args.remote], git_root))
    branch = args.branch or run_git(["branch", "--show-current"], git_root) or "main"
    github_blob_url, colab_url = build_urls(remote_url, branch, relative_path)

    print("Notebook path :", notebook_path)
    print("Repo root     :", git_root)
    print("Remote        :", remote_url)
    print("Branch        :", branch)
    print("GitHub URL    :", github_blob_url)
    print("Colab URL     :", colab_url)

    if args.open:
        webbrowser.open(colab_url)
        print("Opened Colab in your browser.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
