import argparse
import hashlib
from pathlib import Path

from delta_utils.utils import find
from dirhash import dirhash


def sha256_file(filename: Path) -> str:
    """stackoverflow.com/a/44873382."""
    hasher = hashlib.sha256()
    # Create a memory buffer
    buffer = bytearray(128 * 1024)
    mv = memoryview(buffer)
    with open(filename, "rb", buffering=0) as f:
        # Read the file into the buffer, chunk by chunk
        while chunk := f.readinto(mv):  # type: ignore
            hasher.update(mv[:chunk])
    # Hash the complete file
    return hasher.hexdigest()


def hash_game_mechanics(path: Path) -> str:
    """Call me to generate game_mechanics_hash."""
    if (path / "game_mechanics.py").exists():
        return sha256_file(path / "game_mechanics.py")
    else:
        return dirhash(directory=path, algorithm="sha256", match=["*.py"])


def load_game_mechanics_hash(path: Path) -> str:
    """Call me to load game_mechanics_hash."""
    with open(path / "game_mechanics_hash.txt", "r") as f:
        return f.read()


def main() -> None:
    """Call me to generate game_mechanics_hash.txt file.

    Designed to also work being run as a pre-commit hook.
    """
    # Committed files get parsed as arguments to the pre-commit hook
    tracked_files = ["game_mechanics.py", "game_mechanics_hash.txt", "game_mechanics"]
    parser = argparse.ArgumentParser(
        prog="Hash Game Mechanics Script",
        description="Hashes game_mechanics.py and stores it in game_mechanics_hash.txt",
    )
    parser.add_argument("filenames", default=tracked_files, nargs="*")
    args = parser.parse_args()

    # Find the file/directory
    path = find("game_mechanics")
    is_dir = path.is_dir()
    path = path if is_dir else path.parent

    # Run checks for whether this is valid
    filenames = [
        Path(name).parent.name if is_dir and Path(name).suffix != ".txt" else Path(name).name
        for name in args.filenames
    ]
    # We want to check that the user has changed game_mechanics_hash.txt - this is skipped if not
    file_changes_legal = (
        all(filename in tracked_files for filename in filenames)
        and "game_mechanics_hash.txt" in filenames
    )
    file_exists = (path / "game_mechanics_hash.txt").exists()
    hashes_match = (
        load_game_mechanics_hash(path) == hash_game_mechanics(path) if file_exists else False
    )

    # If it's all fine and dandy then return
    if file_changes_legal and file_exists and hashes_match:
        print("game_mechanics.py has not been changed")
        return

    # Houston, there's a problem. Regenerate the hash
    with open(path / "game_mechanics_hash.txt", "w") as f:
        game_mechanics_hash = hash_game_mechanics(path)
        f.write(game_mechanics_hash)

    # Spit out debugging messages
    if not file_changes_legal:
        print(
            f"Only changes to {args.filenames} were committed.\n"
            f"You must commit both game_mechanics files and game_mechanics_hash.txt when either is changed"
        )
    if not file_exists:
        print("game_mechanics_hash.txt does not exist")
    elif not hashes_match:
        print("game_mechanics_hash.txt does not match the hash of the current game_mechanics.py")
    print(f"Saved hash in {path / 'game_mechanics_hash.txt'}")

    # This tells pre-commit that the hook was unsuccessful :'(
    exit(1)
