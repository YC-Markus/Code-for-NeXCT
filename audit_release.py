"""Check a source-only release tree before an anonymous mirror/upload."""
import ast
import json
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parent
    allowed = {".py", ".md", ".json", ".toml", ".txt", ".sha256"}
    ignored = {".git", "__pycache__", ".pytest_cache"}
    private = re.compile(r"/Users/[^/\s]+|/home/[^/\s]+|/data/[^/\s]+/|(?:ghp_|github_pat_)[A-Za-z0-9_]{15,}")
    # Patterns above are intentionally present in this auditing tool itself.
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in ignored for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Release contains a symlink: {rel}")
        if not path.is_file():
            continue
        if path.suffix not in allowed and path.name != ".gitignore":
            raise ValueError(f"Unexpected release artifact: {rel}")
        text = path.read_text()
        if path.name != Path(__file__).name and private.search(text):
            raise ValueError(f"Private path or credential-like string: {rel}")
        if path.suffix == ".py":
            ast.parse(text, filename=str(rel))
        if path.suffix == ".json":
            json.loads(text)
        files.append(path)
    print(json.dumps(dict(status="passed", files=len(files), bytes=sum(p.stat().st_size for p in files),
                          checks=["source-only files", "no symlinks", "Python syntax", "JSON syntax",
                                  "private-path/credential-pattern scan"]), indent=2))
    print("This does not inspect Git history, GitHub account metadata or the anonymous mirror.")


if __name__ == "__main__":
    main()
