"""Restore missing submodule metadata at pinned commits without checking out files."""
import concurrent.futures
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
METADATA = (ROOT / ".git").resolve()


def git(*args, cwd=ROOT):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def entries(worktree):
    output = git("ls-files", "--stage", "-z", cwd=worktree)
    links = []
    for entry in output.split("\0"):
        if not entry.startswith("160000 "):
            continue
        info, path = entry.split("\t", 1)
        sha = info.split()[1]
        config = git("config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$", cwd=worktree)
        for line in config.splitlines():
            key, value = line.split(" ", 1)
            if value == path:
                url = git("config", "-f", ".gitmodules", "--get", key[:-4] + "url", cwd=worktree)
                links.append((worktree / path, sha, url, worktree, key[:-4]))
                break
        else:
            raise RuntimeError(f"No submodule URL for {worktree / path}")
    return links


def restore(entry):
    worktree, sha, url, parent, config_key = entry
    pointer = worktree / ".git"
    if not pointer.is_file():
        raise RuntimeError(f"Expected existing submodule gitfile: {pointer}")
    value = pointer.read_text().strip()
    if not value.startswith("gitdir: "):
        raise RuntimeError(f"Invalid gitfile: {pointer}")
    gitdir = (worktree / value[8:]).resolve()
    if not gitdir.is_relative_to(METADATA):
        raise RuntimeError(f"Metadata escapes root .git: {gitdir}")
    if not (gitdir / "HEAD").exists():
        gitdir.mkdir(parents=True, exist_ok=True)
        git("init", "--bare", str(gitdir))
        git("--git-dir", str(gitdir), "remote", "add", "origin", url)
        git("--git-dir", str(gitdir), "fetch", "--depth=1", "origin", sha)
        git("--git-dir", str(gitdir), "update-ref", "--no-deref", "HEAD", sha)
        git("--git-dir", str(gitdir), "config", "core.bare", "false")
        git("--git-dir", str(gitdir), "config", "core.worktree", str(worktree))
        git("read-tree", sha, cwd=worktree)
        print(f"Restored {worktree.relative_to(ROOT)} at {sha[:12]}", flush=True)
    current = git("rev-parse", "HEAD", cwd=worktree)
    if current != sha:
        raise RuntimeError(f"Existing submodule revision differs from pin: {worktree}: {current}")
    git("config", config_key + "url", url, cwd=parent)
    git("config", config_key + "active", "true", cwd=parent)
    return entries(worktree)


if __name__ == "__main__":
    pending = entries(ROOT)
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while pending:
            total += len(pending)
            results = list(pool.map(restore, pending))
            pending = [entry for children in results for entry in children]
    print(f"Verified {total} submodule repositories; working files preserved.", flush=True)
