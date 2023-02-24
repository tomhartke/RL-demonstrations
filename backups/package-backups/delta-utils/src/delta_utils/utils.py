from pathlib import Path

DONT_SEARCH = [
    "venv",
    ".git",
    "__pycache__",
    "dist",
    "build",
    "tests",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
]


def get_current_dir():
    return Path("").resolve()


def find(name: str) -> Path:
    """Find file or directory with name `name` within all child directories of the directory this is
    run from."""
    # Gets directory we're running from
    path = get_current_dir()

    # Checks all child directories
    for item in path.iterdir():
        if item.is_file():
            # Checks files in current directory match
            if item.stem == name:
                return item.resolve()
        # Skip files and directories we don't want to search
        elif item.name in DONT_SEARCH:
            continue

        # Search all ancestors
        paths = list(filter(lambda x: x.stem == name, item.rglob("*")))
        if paths:
            return paths[0]

    raise FileNotFoundError(f"Could not find {name}")
